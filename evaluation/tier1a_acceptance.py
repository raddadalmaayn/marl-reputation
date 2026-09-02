"""
Tier 1A acceptance checks 1A/1B/1C — compares the smoke-run per-seed summaries.

1A REGRESSION-BREAK : under stake_obs_mode=absolute_log, smoke no_stake (min_stake=1)
                      vs no_dispute (min_stake=10000) must produce DIFFERENT
                      action_counts (and/or final agent scores) — the Tier 0
                      byte-identical property is broken.
1B LEGACY-PRESERVE  : under stake_obs_mode=legacy_normalized, the same pair must
                      reproduce byte-identical action_counts (differing only by the
                      10000x stake ratio in final_mean_stake).
1C OBS SHAPE/BOUNDS : obs is length 14, finite, within clip bounds in both modes
                      (checked directly by instantiating the env).

Reads results/tier1a/training_logs/<name>_seed<s>.json. Run AFTER the smoke matrix.

Usage: python3 evaluation/tier1a_acceptance.py
"""

import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
TLOG = REPO / "results" / "tier1a" / "training_logs"


def load(name, seed):
    p = TLOG / f"{name}_seed{seed}.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def load_log(name, seed):
    p = TLOG / f"{name}_seed{seed}_log.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


# Behavioral fields that capture the action trajectory (the Tier 0 byte-identical
# property was over exactly these); stake-derived fields are EXPECTED to differ.
_BEHAVIORAL = ["honest_pct", "mean_score", "mean_reputation_accuracy",
               "total_ratings", "dishonest_actions", "total_actions",
               "attacks_attempted", "attacks_blocked"]
_STAKE_DERIVED = ["mean_reward", "total_reward", "stakes"]


def compare_log_sequences(mode_short, seeds=(0, 1)):
    """Primary 1A/1B evidence: per-episode log-sequence comparison of no_stake vs
    no_dispute (mirrors Tier 0 diagnose_ablation). Robust to final-policy collapse."""
    ns = f"smoke_no_stake_{mode_short}"
    nd = f"smoke_no_dispute_{mode_short}"
    rows = []
    for s in seeds:
        a, b = load_log(ns, s), load_log(nd, s)
        if a is None or b is None:
            rows.append({"seed": s, "error": f"missing log {ns}/{nd} seed{s}"})
            continue
        n = min(len(a), len(b))
        identical, differing = [], []
        for k in _BEHAVIORAL + _STAKE_DERIVED:
            same = all(a[i].get(k) == b[i].get(k) for i in range(n))
            (identical if same else differing).append(k)
        rows.append({
            "seed": s, "n_episodes": n,
            "behavioral_identical": [k for k in _BEHAVIORAL if k in identical],
            "behavioral_differing": [k for k in _BEHAVIORAL if k in differing],
            "stake_fields_differing": [k for k in _STAKE_DERIVED if k in differing],
        })
    return rows


def agent_scores(d):
    return d.get("final_env_metrics", {}).get("agent_scores")


def compare_pair(mode_short, seeds=(0, 1)):
    ns = f"smoke_no_stake_{mode_short}"
    nd = f"smoke_no_dispute_{mode_short}"
    rows = []
    for s in seeds:
        a, b = load(ns, s), load(nd, s)
        if a is None or b is None:
            rows.append({"seed": s, "error": f"missing {ns}/{nd} seed{s}"})
            continue
        ac_a, ac_b = a["action_counts"], b["action_counts"]
        same_counts = ac_a == ac_b
        sa, sb = agent_scores(a), agent_scores(b)
        same_scores = (sa is not None and sb is not None and sa == sb)
        rows.append({
            "seed": s,
            "action_counts_identical": same_counts,
            "agent_scores_identical": same_scores,
            "stake_ratio": (b["final_mean_stake"] / a["final_mean_stake"]
                            if a.get("final_mean_stake") else None),
            "no_stake_basin": a.get("basin"),
            "no_dispute_basin": b.get("basin"),
            "no_stake_action_counts": ac_a,
            "no_dispute_action_counts": ac_b,
        })
    return rows


def check_1c():
    from env.reputation_marl_env import ReputationMARLEnv
    from env.reputation_engine import SystemConfig
    out = {}
    for mode in ("legacy_normalized", "absolute_log"):
        res = []
        for ms in (1.0, 10000.0):
            e = ReputationMARLEnv(n_agents=20, max_steps=100,
                                  config=SystemConfig(min_stake_required=ms),
                                  seed=0, stake_obs_mode=mode)
            e.reset(seed=0)
            o = e.observe("agent_0")
            res.append({
                "min_stake": ms,
                "len14": o.shape == (14,),
                "finite": bool(np.all(np.isfinite(o))),
                "in_box": bool(o.min() >= -1.0 and o.max() <= 2.0),
                "stake_slot": float(o[2]),
            })
        out[mode] = res
    return out


def main():
    report = {
        # Primary evidence: per-episode log-sequence identity (the Tier 0 property).
        "1A_absolute_log_break_logseq": compare_log_sequences("absolute"),
        "1B_legacy_preserve_logseq": compare_log_sequences("legacy"),
        # Secondary: deterministic-eval action_counts / final agent scores.
        "1A_absolute_log_break_eval": compare_pair("absolute"),
        "1B_legacy_preserve_eval": compare_pair("legacy"),
        "1C_obs_shape_bounds": check_1c(),
    }

    real = lambda rows: [r for r in rows if "error" not in r]

    # 1A break: under absolute_log the behavioral sequences MUST differ for every seed.
    a_rows = real(report["1A_absolute_log_break_logseq"])
    v1a = bool(a_rows) and all(len(r["behavioral_differing"]) > 0 for r in a_rows)
    # 1B preserve: under legacy the behavioral sequences are byte-identical AND the
    # stake-derived fields differ (the 10000x regime difference is still present).
    b_rows = real(report["1B_legacy_preserve_logseq"])
    v1b = bool(b_rows) and all(
        len(r["behavioral_differing"]) == 0 and len(r["stake_fields_differing"]) > 0
        for r in b_rows)
    v1c = all(x["len14"] and x["finite"] and x["in_box"]
              for m in report["1C_obs_shape_bounds"].values() for x in m)
    report["verdicts"] = {"1A_break": bool(v1a), "1B_preserve": bool(v1b),
                          "1C_bounds": bool(v1c)}

    out = REPO / "results" / "tier1a" / "acceptance.json"
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    print("\nVERDICTS:", report["verdicts"])


if __name__ == "__main__":
    main()
