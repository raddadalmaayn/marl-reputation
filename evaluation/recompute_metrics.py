"""
Tier-0 metric recomputation for the MARL reputation study.

Recomputes, from saved checkpoints via DETERMINISTIC eval only (never retrains):
  T0.1  three honest-% metrics  (participation / conditional / unconditional)
  T0.2  basin-aware statistics  (basin counts + Jeffreys CI on honest-dominant prob)
  T0.3  reputation accuracy vs a random baseline (MAE, Spearman rho, skill)

Per-action counts (A0..A11) and per-agent score/quality arrays are NOT present in
the saved training logs (train.py only stored honest_pct / dishonest_actions /
total_actions and aggregate env metrics). They are therefore collected here by
loading results/checkpoints/**/final.pt and running deterministic eval
(dist.mode = argmax) for a fixed set of episode seeds. Eval is cheap; training is
forbidden for this task.

Determinism / faithfulness notes
  * Action counts record the POLICY-SELECTED action (pre any in-env forcing of
    actions >=7 to noop for non-adversarial agents). This matches how train.py
    computes honest_pct (it counts the selected action). The forcing only ever
    touches non-rating buckets (>=7), so participation_rate / honest_given_rating
    are invariant to whether forced actions are bucketed as A0 or A_k.
  * Eval uses deterministic=True, matching train.py's eval loop, so the
    conditional aggregate is expected to reproduce paper Table 2 and the
    unconditional aggregate the codebase Section 9.
  * Episode seeds are fixed (--eval-base + episode index) so the whole script is
    idempotent. Raw eval output is cached under results/derived/eval_cache/.

Outputs (under results/derived/, never overwriting raw result JSONs):
  eval_cache/<config>_seed<s>.json   raw per-seed eval (action counts, scores...)
  honest_metrics.json                T0.1 per-config + per-seed
  basin_stats.json                   T0.2
  accuracy_vs_random.json            T0.3
  honest_metrics.md / basin_stats.md / accuracy_vs_random.md  human tables

Usage:
  python3 evaluation/recompute_metrics.py            # all configs, cached
  python3 evaluation/recompute_metrics.py --force    # recompute eval cache
  python3 evaluation/recompute_metrics.py --configs config1_baseline config2_mixed
  python3 evaluation/recompute_metrics.py --eval-episodes 20 --eval-base 10000
"""

import argparse
import json
import math
import os
import sys
from collections import Counter
from pathlib import Path

# Single-threaded determinism — set before importing torch.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
torch.set_num_threads(1)

from env.reputation_marl_env import ReputationMARLEnv
from env.reputation_engine import SystemConfig
from agents.mappo import MAPPOTrainer

DERIVED = REPO / "results" / "derived"
CACHE = DERIVED / "eval_cache"

# Basin thresholds (T0.2) — explicit and printed in output.
THR_PARTICIPATION_MIN = 0.01   # below this => abstention basin
THR_ADV_HGR_MAX       = 0.05   # honest_given_rating <= this => adversarial-dominant
THR_ADV_STAKE_FRAC    = 0.01   # final mean stake <= this*reset => "bankrupt"
THR_HON_HGR_MIN       = 0.95   # honest_given_rating >= this => honest-dominant


# --------------------------------------------------------------------------- #
# Config registry
# --------------------------------------------------------------------------- #
# Each entry: yaml path (for env kwargs + SystemConfig overrides), checkpoint
# root, number of seeds. Checkpoints live at <ckpt_root>/<name>/seed<s>/final.pt.
def _reg(name, yaml_rel, ckpt_root, n_seeds):
    return {"name": name, "yaml": REPO / yaml_rel,
            "ckpt_root": REPO / ckpt_root, "n_seeds": n_seeds}


MAIN_CONFIGS = [
    _reg(f"config{i}_{s}", f"configs/config{i}.yaml", "results/checkpoints", 5)
    for i, s in [
        (1, "baseline"), (2, "mixed"), (3, "sybil"), (4, "collusion"),
        (5, "adaptive"), (6, "self_rating"), (7, "admin_escalation"),
        (8, "evidence_tamper"), (9, "gate_bypass"), (10, "provenance_replay"),
        (11, "comprehensive"),
    ]
]

TERMINAL_CONFIGS = [
    _reg("config2_terminal", "configs/config2_terminal.yaml",
         "results/terminal/checkpoints", 3),
    _reg("config11_terminal", "configs/config11_terminal.yaml",
         "results/terminal/checkpoints", 3),
]

ABLATION_CONFIGS = [
    _reg(f"ablation_{v}", f"configs/ablation_{v}.yaml",
         "results/ablation/checkpoints", 3)
    for v in ["no_stake", "no_dispute", "no_decay", "no_wilson"]
]

REGISTRY = {c["name"]: c for c in MAIN_CONFIGS + TERMINAL_CONFIGS + ABLATION_CONFIGS}


def env_kwargs_from_yaml(cfg: dict) -> dict:
    return dict(
        n_agents=cfg["n_agents"],
        max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None),
        terminal_reward_coef=cfg.get("terminal_reward_coef", 0.0),
    )


def system_config_from_yaml(cfg: dict) -> SystemConfig:
    """Build SystemConfig honouring ablation overrides (train_ablation.py parity)."""
    return SystemConfig(
        decay_rate=cfg.get("decay_rate", 0.98),
        min_stake_required=cfg.get("min_stake_required", 10_000.0),
        dispute_cost=cfg.get("dispute_cost", 100.0),
        min_rater_weight=cfg.get("min_rater_weight", 0.5),
        max_rater_weight=cfg.get("max_rater_weight", 2.0),
    )


# --------------------------------------------------------------------------- #
# Deterministic eval
# --------------------------------------------------------------------------- #
def eval_seed(entry: dict, seed: int, eval_episodes: int, eval_base: int) -> dict:
    """Load final.pt and run deterministic eval. Returns raw collected data."""
    with open(entry["yaml"]) as f:
        cfg = yaml.safe_load(f)
    kwargs = env_kwargs_from_yaml(cfg)
    sys_cfg = system_config_from_yaml(cfg)
    reset_stake = sys_cfg.min_stake_required * 2.0

    ckpt = entry["ckpt_root"] / entry["name"] / f"seed{seed}" / "final.pt"
    if not ckpt.exists():
        return {"error": f"missing checkpoint: {ckpt}"}

    env = ReputationMARLEnv(config=sys_cfg, seed=seed, **kwargs)
    trainer = MAPPOTrainer(obs_dim=14, act_dim=12)
    trainer.init_agents(env.possible_agents)
    trainer.load(str(ckpt))

    counts = Counter()
    scores, quals, final_stakes = [], [], []

    for e in range(eval_episodes):
        env.reset(seed=eval_base + e)
        while env.agents:
            ag = env.agent_selection
            if env.terminations.get(ag, False) or env.truncations.get(ag, False):
                env.step(None)
                continue
            obs = env.observe(ag)
            a, _, _ = trainer.select_action(ag, obs, deterministic=True)
            env.step(a)
            counts[int(a)] += 1
        m = env.get_metrics()
        scores.extend(m["agent_scores"])
        quals.extend(m["agent_true_qualities"])
        final_stakes.append(float(np.mean(list(m["stakes"].values()))))

    action_counts = {str(a): int(counts.get(a, 0)) for a in range(12)}
    return {
        "config": entry["name"],
        "seed": seed,
        "checkpoint": str(ckpt),
        "eval_episodes": eval_episodes,
        "eval_base": eval_base,
        "reset_stake": reset_stake,
        "action_counts": action_counts,
        "final_mean_stake": float(np.mean(final_stakes)),
        "agent_scores": scores,
        "agent_true_qualities": quals,
    }


def get_eval(entry: dict, seed: int, eval_episodes: int, eval_base: int,
             force: bool) -> dict:
    CACHE.mkdir(parents=True, exist_ok=True)
    cpath = CACHE / f"{entry['name']}_seed{seed}.json"
    if cpath.exists() and not force:
        d = json.loads(cpath.read_text())
        if d.get("eval_episodes") == eval_episodes and d.get("eval_base") == eval_base:
            return d
    d = eval_seed(entry, seed, eval_episodes, eval_base)
    cpath.write_text(json.dumps(d))
    return d


# --------------------------------------------------------------------------- #
# T0.1 honest metrics
# --------------------------------------------------------------------------- #
def honest_metrics_seed(ev: dict) -> dict:
    ac = ev["action_counts"]
    A = {i: ac.get(str(i), 0) for i in range(12)}
    rating = A[1] + A[2] + A[3] + A[4]
    honest = A[1] + A[2]
    total = sum(A.values())
    participation = rating / total if total else float("nan")
    hgr = (honest / rating) if rating > 0 else float("nan")
    huncond = hgr if rating > 0 else 0.0
    return {
        "seed": ev["seed"],
        "rating_actions": rating,
        "honest_actions": honest,
        "total_actions": total,
        "participation_rate": participation,
        "honest_given_rating": hgr,
        "honest_unconditional": huncond,
        "final_mean_stake": ev["final_mean_stake"],
        "reset_stake": ev["reset_stake"],
    }


def aggregate_honest(per_seed: list) -> dict:
    hgr_active = [s["honest_given_rating"] for s in per_seed
                  if not math.isnan(s["honest_given_rating"])]
    huncond = [s["honest_unconditional"] for s in per_seed]
    cond_mean = float(np.mean(hgr_active)) if hgr_active else float("nan")
    cond_median = float(np.median(hgr_active)) if hgr_active else float("nan")
    uncond_mean = float(np.mean(huncond)) if huncond else float("nan")
    gap = (cond_mean - uncond_mean) if hgr_active else float("nan")
    return {
        "n_seeds": len(per_seed),
        "n_rating_active_seeds": len(hgr_active),
        "conditional_mean_honest": cond_mean,
        "conditional_median_honest": cond_median,
        "unconditional_mean_honest": uncond_mean,
        "abstention_gap": gap,
    }


# --------------------------------------------------------------------------- #
# T0.2 basins + Jeffreys CI
# --------------------------------------------------------------------------- #
def classify_basin(s: dict) -> str:
    part = s["participation_rate"]
    hgr = s["honest_given_rating"]
    stake_frac = (s["final_mean_stake"] / s["reset_stake"]) if s["reset_stake"] else 1.0
    if (not math.isnan(part)) and part < THR_PARTICIPATION_MIN:
        return "abstention"
    bankrupt = stake_frac <= THR_ADV_STAKE_FRAC
    if (not math.isnan(hgr) and hgr <= THR_ADV_HGR_MAX) or bankrupt:
        return "adversarial-dominant"
    if (not math.isnan(hgr) and hgr >= THR_HON_HGR_MIN) and \
       (not math.isnan(part) and part >= THR_PARTICIPATION_MIN):
        return "honest-dominant"
    return "mixed"


def jeffreys_ci(k: int, n: int, conf: float = 0.95):
    """Jeffreys 95% interval for a binomial proportion (Beta(k+.5, n-k+.5))."""
    from scipy.stats import beta as beta_dist
    if n == 0:
        return (float("nan"), float("nan"))
    a = conf  # noqa
    lo = 0.0 if k == 0 else float(beta_dist.ppf((1 - conf) / 2, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - (1 - conf) / 2, k + 0.5, n - k + 0.5))
    return (lo, hi)


def basin_stats(per_seed: list) -> dict:
    basins = [classify_basin(s) for s in per_seed]
    n = len(basins)
    counts = {b: basins.count(b) for b in
              ["abstention", "adversarial-dominant", "honest-dominant", "mixed"]}
    k_hon = counts["honest-dominant"]
    lo, hi = jeffreys_ci(k_hon, n)
    distinct = sum(1 for v in counts.values() if v > 0)
    # Bimodal if seeds split across >=2 basins (with at least 2 non-trivial groups).
    nonzero = [v for v in counts.values() if v > 0]
    bimodal = len(nonzero) >= 2
    return {
        "n_seeds": n,
        "basin_counts": counts,
        "per_seed_basin": {str(s["seed"]): b for s, b in zip(per_seed, basins)},
        "honest_dominant_prob": (k_hon / n) if n else float("nan"),
        "honest_dominant_jeffreys95": [lo, hi],
        "bimodal": bimodal,
        "distinct_basins": distinct,
    }


# --------------------------------------------------------------------------- #
# T0.3 accuracy vs random
# --------------------------------------------------------------------------- #
def accuracy_vs_random(scores: list, quals: list, n_random_draws: int,
                       rng: np.random.Generator) -> dict:
    from scipy.stats import spearmanr
    s = np.asarray(scores, dtype=float)
    q = np.asarray(quals, dtype=float)
    if s.size == 0:
        return {"error": "no scores"}
    system_mae = float(np.mean(np.abs(s - q)))
    if np.std(s) < 1e-12 or np.std(q) < 1e-12:
        rho = float("nan")
    else:
        rho = float(spearmanr(s, q).statistic)

    # Empirical random MAE: estimator ~ U[0,1] matched to the ACTUAL true qualities
    # (env draws q ~ Beta(8,2), NOT U[0,1]; see report). Average over n_random_draws.
    maes = []
    for _ in range(max(1, n_random_draws)):
        r = rng.uniform(0.0, 1.0, size=q.size)
        maes.append(np.mean(np.abs(r - q)))
    random_mae = float(np.mean(maes))
    analytic_uniform_mae = 1.0 / 3.0  # only valid if q ~ U[0,1]
    skill = (random_mae - system_mae) / random_mae if random_mae > 0 else float("nan")
    return {
        "n_points": int(s.size),
        "system_mae": system_mae,
        "system_spearman_rho": rho,
        "empirical_random_mae": random_mae,
        "analytic_uniform_random_mae": analytic_uniform_mae,
        "skill": skill,
        "at_or_below_random": bool(skill <= 0),
        "mean_score": float(np.mean(s)),
        "mean_true_quality": float(np.mean(q)),
    }


# --------------------------------------------------------------------------- #
# Driver
# --------------------------------------------------------------------------- #
def process_config(entry: dict, args) -> dict:
    per_seed_ev = []
    for seed in range(entry["n_seeds"]):
        ev = get_eval(entry, seed, args.eval_episodes, args.eval_base, args.force)
        if "error" in ev:
            print(f"  [WARN] {entry['name']} seed{seed}: {ev['error']}")
            continue
        per_seed_ev.append(ev)
    if not per_seed_ev:
        return {"name": entry["name"], "error": "no eval data"}

    hm_seeds = [honest_metrics_seed(ev) for ev in per_seed_ev]
    hm_agg = aggregate_honest(hm_seeds)
    bs = basin_stats(hm_seeds)

    pooled_scores, pooled_quals = [], []
    for ev in per_seed_ev:
        pooled_scores.extend(ev["agent_scores"])
        pooled_quals.extend(ev["agent_true_qualities"])
    rng = np.random.default_rng(12345)
    acc = accuracy_vs_random(pooled_scores, pooled_quals, args.random_draws, rng)

    return {
        "name": entry["name"],
        "honest": {"per_seed": hm_seeds, "aggregate": hm_agg},
        "basins": bs,
        "accuracy": acc,
    }


def fmt(x, nd=3):
    if isinstance(x, float) and math.isnan(x):
        return "NaN"
    if isinstance(x, float):
        return f"{x:.{nd}f}"
    return str(x)


def write_honest_md(results: list, path: Path):
    lines = ["# T0.1 Honest-% reconciliation", "",
             "Per-config aggregate (deterministic eval).", "",
             "| Config | seeds | rating-active | participation (mean) | "
             "conditional honest (mean / median) | unconditional honest (mean) | gap |",
             "|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            lines.append(f"| {r['name']} | ERROR | | | | | |")
            continue
        a = r["honest"]["aggregate"]
        part = np.nanmean([s["participation_rate"] for s in r["honest"]["per_seed"]])
        lines.append(
            f"| {r['name']} | {a['n_seeds']} | {a['n_rating_active_seeds']} | "
            f"{fmt(float(part))} | {fmt(a['conditional_mean_honest'])} / "
            f"{fmt(a['conditional_median_honest'])} | "
            f"{fmt(a['unconditional_mean_honest'])} | {fmt(a['abstention_gap'])} |")
    lines += ["", "## Per-seed detail", ""]
    for r in results:
        if "error" in r:
            continue
        lines.append(f"### {r['name']}")
        lines.append("| seed | participation | honest|rating | honest uncond | "
                     "final mean stake | basin |")
        lines.append("|---|---|---|---|---|---|")
        bmap = r["basins"]["per_seed_basin"]
        for s in r["honest"]["per_seed"]:
            lines.append(
                f"| {s['seed']} | {fmt(s['participation_rate'])} | "
                f"{fmt(s['honest_given_rating'])} | {fmt(s['honest_unconditional'])} | "
                f"{fmt(s['final_mean_stake'],1)} | {bmap.get(str(s['seed']),'')} |")
        lines.append("")
    path.write_text("\n".join(lines) + "\n")


def write_basin_md(results: list, path: Path):
    lines = ["# T0.2 Basin-aware statistics", "",
             "Thresholds: "
             f"abstention participation<{THR_PARTICIPATION_MIN}; "
             f"adversarial-dominant honest|rating<={THR_ADV_HGR_MAX} OR "
             f"final stake<={THR_ADV_STAKE_FRAC}*reset; "
             f"honest-dominant honest|rating>={THR_HON_HGR_MIN} AND "
             f"participation>={THR_PARTICIPATION_MIN}; else mixed.", "",
             "| Config | seeds | abst | adv | hon | mixed | "
             "P(honest-dominant) | Jeffreys 95% CI | bimodal? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r:
            continue
        b = r["basins"]
        c = b["basin_counts"]
        lo, hi = b["honest_dominant_jeffreys95"]
        lines.append(
            f"| {r['name']} | {b['n_seeds']} | {c['abstention']} | "
            f"{c['adversarial-dominant']} | {c['honest-dominant']} | {c['mixed']} | "
            f"{fmt(b['honest_dominant_prob'])} | [{fmt(lo)}, {fmt(hi)}] | "
            f"{'yes' if b['bimodal'] else 'no'} |")
    path.write_text("\n".join(lines) + "\n")


def write_accuracy_md(results: list, path: Path):
    lines = ["# T0.3 Reputation accuracy vs random baseline", "",
             "Random estimator ~ U[0,1]. NOTE: env draws true quality q ~ Beta(8,2) "
             "(mean 0.8), so the analytic U[0,1] random MAE of 1/3 does NOT apply; "
             "the empirical random MAE below is matched to the actual qualities.", "",
             "| Config | n | system MAE | Spearman rho | empirical random MAE | "
             "skill | mean score | mean true q | <= random? |",
             "|---|---|---|---|---|---|---|---|---|"]
    for r in results:
        if "error" in r or "error" in r.get("accuracy", {}):
            continue
        a = r["accuracy"]
        flag = "FLAG" if a["at_or_below_random"] else ""
        lines.append(
            f"| {r['name']} | {a['n_points']} | {fmt(a['system_mae'])} | "
            f"{fmt(a['system_spearman_rho'])} | {fmt(a['empirical_random_mae'])} | "
            f"{fmt(a['skill'])} | {fmt(a['mean_score'])} | {fmt(a['mean_true_quality'])} | "
            f"{flag} |")
    path.write_text("\n".join(lines) + "\n")


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--configs", nargs="*", default=None,
                    help="subset of config names (default: all in registry)")
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-base", type=int, default=10000)
    ap.add_argument("--random-draws", type=int, default=1000)
    ap.add_argument("--force", action="store_true", help="ignore eval cache")
    args = ap.parse_args()

    names = args.configs or list(REGISTRY.keys())
    DERIVED.mkdir(parents=True, exist_ok=True)

    print(f"Basin thresholds: participation<{THR_PARTICIPATION_MIN}, "
          f"adv honest|rating<={THR_ADV_HGR_MAX} or stake<={THR_ADV_STAKE_FRAC}*reset, "
          f"honest honest|rating>={THR_HON_HGR_MIN}")
    print(f"Eval: {args.eval_episodes} episodes, base seed {args.eval_base}, "
          f"deterministic=True\n")

    results = []
    for name in names:
        if name not in REGISTRY:
            print(f"[SKIP] unknown config: {name}")
            continue
        print(f"[{name}] evaluating...")
        results.append(process_config(REGISTRY[name], args))

    (DERIVED / "honest_metrics.json").write_text(
        json.dumps([{"name": r["name"], **{k: v for k, v in r.items()
                     if k in ("honest", "basins", "accuracy")}}
                    for r in results], indent=2))
    write_honest_md(results, DERIVED / "honest_metrics.md")
    write_basin_md(results, DERIVED / "basin_stats.md")
    write_accuracy_md(results, DERIVED / "accuracy_vs_random.md")
    (DERIVED / "basin_stats.json").write_text(json.dumps(
        [{"name": r["name"], "basins": r.get("basins")} for r in results
         if "error" not in r], indent=2))
    (DERIVED / "accuracy_vs_random.json").write_text(json.dumps(
        [{"name": r["name"], "accuracy": r.get("accuracy")} for r in results
         if "error" not in r], indent=2))

    print(f"\nWrote outputs under {DERIVED}/")
    for f in ["honest_metrics.md", "basin_stats.md", "accuracy_vs_random.md"]:
        print(f"  - {f}")


if __name__ == "__main__":
    main()
