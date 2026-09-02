"""
Phase 2.3 — a discrimination metric that does not go undefined on ties.

WHY Kendall's tau-a.

Spearman's rho is undefined when the score vector is constant, but not because
the answer is unknowable. It is a 0/0 artifact: rho normalises by the standard
deviation of each vector, and a constant predictor makes that denominator zero.
The underlying question -- "how much does this reputation score tell you about
true quality?" -- has an unambiguous answer for a constant predictor: nothing.

Kendall's tau-a states that directly:

    tau_a = (C - D) / [n(n-1)/2]

C and D count concordant and discordant pairs; tied pairs contribute zero to the
numerator but are still counted in the denominator. The denominator depends only
on n, never on the tie structure, so tau-a is defined whenever n >= 2. A constant
score vector produces C = D = 0 and therefore tau_a = 0 exactly -- the correct
statement that the score ranks nothing.

This is NOT imputation. The score vector in a collapsed shard is fully observed
(every agent sits at the Beta prior, 0.5). Nothing is missing or reconstructed.
tau-a simply evaluates discrimination at a degenerate but completely known point,
where rho's formula happens to divide by zero. It is reported ALONGSIDE rho, never
as a substitute, so that the headline does not rest on a single estimator.

Note tau-b is NOT suitable here: it normalises by the tie-corrected pair counts,
so it reproduces exactly the same 0/0 collapse as rho.

Also caches the (score, quality) vectors to vectors.npz so no later phase needs
to re-score.

Writes ONLY under analysis/rho_2026_09/.
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


def tau_a(s, q):
    """Kendall tau-a: (C-D) / [n(n-1)/2]. Defined for all n>=2, 0 when constant."""
    n = s.size
    if n < 2:
        return float("nan")
    ss = np.sign(s[:, None] - s[None, :])
    sq = np.sign(q[:, None] - q[None, :])
    cd = float(np.sum(ss * sq)) / 2.0          # upper triangle counted once
    return cd / (n * (n - 1) / 2.0)


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
    cfg = yaml.safe_load(open(REPO / "configs" / "tier1b" / f"{tag}.yaml"))
    ckpt = SWEEP / "checkpoints" / tag / f"seed{seed}" / "final.pt"
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
        reputation_engine="beta", record_rating_events=True,
    )
    tr = MAPPOTrainer(obs_dim=14, act_dim=12)
    tr.init_agents(env.possible_agents)
    tr.load(str(ckpt))

    scores, quals = [], []
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

    s = np.asarray(scores, float)
    q = np.asarray(quals, float)
    collapsed = bool(np.std(s) < 1e-12)
    if collapsed or np.std(q) < 1e-12:
        rho = float("nan")
    else:
        from scipy.stats import spearmanr
        rho = float(spearmanr(s, q).statistic)
    return {
        "key": f"{cbase}|{'A' if arm=='armA' else 'B'}|{seed}",
        "config": cbase, "arm": "A" if arm == "armA" else "B", "seed": seed,
        "rho": rho, "tau_a": tau_a(s, q),
        "score_std": float(np.std(s)), "collapsed": int(collapsed),
    }, s.astype(np.float32), q.astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    t = targets()
    print(f"{len(t)} shards, {args.workers} workers")
    rows, vecs = [], {}
    with Pool(args.workers) as p:
        for i, (r, s, q) in enumerate(p.imap_unordered(one, t), 1):
            rows.append(r)
            vecs[r["key"] + "|s"] = s
            vecs[r["key"] + "|q"] = q
            if i % 60 == 0:
                print(f"  {i}/{len(t)}", flush=True)
    rows.sort(key=lambda r: (r["config"], r["arm"], r["seed"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "tau.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["config", "arm", "seed", "rho", "tau_a",
                                          "score_std", "collapsed"],
                           extrasaction="ignore")
        w.writeheader()
        for r in rows:
            rr = dict(r)
            if isinstance(rr["rho"], float) and math.isnan(rr["rho"]):
                rr["rho"] = ""
            w.writerow(rr)
    np.savez_compressed(OUT / "vectors.npz", **vecs)
    print(f"\nwrote {OUT/'tau.csv'} and {OUT/'vectors.npz'}")
    nc = sum(r["collapsed"] for r in rows)
    print(f"collapsed shards: {nc}/{len(rows)}")


if __name__ == "__main__":
    main()
