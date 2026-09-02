"""
Phase 1.2 — one tidy row per shard, all 540.

Stored fields come from results/tier1b/sweep/training_logs/*.json (read-only).
score_std / n_distinct_scores / attacks are NOT stored anywhere, so they are
recovered by deterministically re-scoring each shard's final.pt at the same
eval schedule the shard summary recorded (basin_eval_base=10000, 20 episodes).
This is a re-score, never a retrain: no optimizer step is taken.

Nothing is imputed. Where a value cannot be recovered it is written as an empty
CSV field, never a substitute.

Writes ONLY analysis/rho_2026_09/shards.csv.

Usage: python3 analysis/rho_2026_09/build_table.py [--workers 16]
"""
import argparse
import csv
import json
import math
import os
import sys
from multiprocessing import Pool
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

SWEEP = REPO / "results" / "tier1b" / "sweep"
OUT = REPO / "analysis" / "rho_2026_09"

CONFIGS30 = ["c1", "c2", "c3", "c6", "c7", "c8", "c9", "c11"]
CONFIGS10 = ["c4", "c5", "c10"]


def targets():
    t = []
    for c in CONFIGS30:
        for a in ("armA", "armB"):
            t += [(c, a, s) for s in range(30)]
    for c in CONFIGS10:
        for a in ("armA", "armB"):
            t += [(c, a, s) for s in range(10)]
    return t


def one(args):
    cbase, arm, seed = args
    import torch
    torch.set_num_threads(1)
    from env.reputation_marl_env import ReputationMARLEnv
    from agents.mappo import MAPPOTrainer
    from training.train import build_system_config

    tag = f"sweep_{cbase}_{arm}"
    summ_p = SWEEP / "training_logs" / f"{tag}_seed{seed}.json"
    d = json.load(open(summ_p))
    m0 = d["final_env_metrics"]

    row = {
        "config": cbase,
        "arm": "A" if arm == "armA" else "B",
        "seed": seed,
        "participation_rate": d.get("participation_rate"),
        "basin": d.get("basin"),
        "honest_given_rating": d.get("honest_given_rating"),
        "attacks_attempted_stored": m0.get("attacks_attempted"),
        "total_ratings_final_ep": m0.get("total_ratings"),
        "mean_score_stored": m0.get("mean_score"),
        "rho_stored": d.get("system_spearman_rho"),
        "mae_stored": d.get("system_mae"),
        "n_score_points": d.get("n_score_points"),
    }

    cfg = yaml.safe_load(open(REPO / "configs" / "tier1b" / f"{tag}.yaml"))
    ckpt = SWEEP / "checkpoints" / tag / f"seed{seed}" / "final.pt"
    if not ckpt.exists():
        row.update({"score_std": "", "n_distinct_scores": "", "qual_std": "",
                    "attacks_attempted": "", "attacks_blocked": "", "hold_rate": "",
                    "rho_recomputed": "", "rescore_ok": 0})
        return row

    sys_cfg = build_system_config(cfg)
    env = ReputationMARLEnv(
        n_agents=cfg["n_agents"], max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None),
        config=sys_cfg, seed=seed,
        stake_obs_mode=cfg.get("stake_obs_mode", "absolute_log"),
        participation_coef=cfg.get("participation_coef", 0.0),
        reputation_engine="beta",
        record_rating_events=True,
    )
    tr = MAPPOTrainer(obs_dim=14, act_dim=12)
    tr.init_agents(env.possible_agents)
    tr.load(str(ckpt))

    scores, quals, att, blk = [], [], 0, 0
    for e in range(20):
        env.reset(seed=10000 + e)
        while env.agents:
            ag = env.agent_selection
            if env.terminations.get(ag, False) or env.truncations.get(ag, False):
                env.step(None)
                continue
            a, _, _ = tr.select_action(ag, env.observe(ag), deterministic=True)
            env.step(a)
        mm = env.get_metrics()
        scores.extend(mm["agent_scores"])
        quals.extend(mm["agent_true_qualities"])
        att += mm.get("attacks_attempted", 0)
        blk += mm.get("attacks_blocked", 0)

    s = np.asarray(scores, float)
    q = np.asarray(quals, float)
    if s.size == 0 or np.std(s) < 1e-12 or np.std(q) < 1e-12:
        rho = float("nan")
    else:
        from scipy.stats import spearmanr
        rho = float(spearmanr(s, q).statistic)

    rs = d.get("system_spearman_rho")
    row.update({
        "score_std": float(np.std(s)),
        "n_distinct_scores": int(np.unique(s).size),
        "qual_std": float(np.std(q)),
        "attacks_attempted": att,
        "attacks_blocked": blk,
        "hold_rate": (blk / att) if att > 0 else "",
        "rho_recomputed": rho if not math.isnan(rho) else "",
        "rescore_ok": 1,
    })
    row["rho_defined"] = int(not math.isnan(rho))
    row["rho_matches_stored"] = int(
        (math.isnan(rho) and not (isinstance(rs, float) and not math.isnan(rs)))
        or (isinstance(rs, float) and not math.isnan(rs) and not math.isnan(rho)
            and abs(rho - rs) < 1e-9))
    return row


COLS = ["config", "arm", "seed", "participation_rate", "basin",
        "honest_given_rating", "attacks_attempted_stored", "total_ratings_final_ep",
        "mean_score_stored", "attacks_attempted", "attacks_blocked", "hold_rate",
        "rho_defined", "rho_stored", "rho_recomputed", "rho_matches_stored",
        "mae_stored", "score_std", "n_distinct_scores", "qual_std",
        "n_score_points", "rescore_ok"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t = targets()
    print(f"{len(t)} shards, {args.workers} workers")
    with Pool(args.workers) as p:
        rows = []
        for i, r in enumerate(p.imap_unordered(one, t), 1):
            rows.append(r)
            if i % 60 == 0:
                print(f"  {i}/{len(t)}", flush=True)
    rows.sort(key=lambda r: (r["config"], r["arm"], r["seed"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "shards.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLS, extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            for k, v in rr.items():
                if isinstance(v, float) and math.isnan(v):
                    rr[k] = ""
            w.writerow(rr)
    ok = sum(r.get("rho_matches_stored", 0) for r in rows)
    print(f"\nwrote {OUT/'shards.csv'}  ({len(rows)} rows)")
    print(f"rho reproduction vs stored: {ok}/{len(rows)} match")


if __name__ == "__main__":
    main()
