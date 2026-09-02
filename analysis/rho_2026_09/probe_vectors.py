"""
Phase 1.1 — recover the ACTUAL (score, true-quality) vectors passed to the
Spearman call, for a handful of shards, and name the mechanism that makes
rho undefined.

Read-only with respect to results/: loads configs/tier1b/*.yaml and the
existing final.pt checkpoints, re-scores them deterministically (no training,
no optimizer step), and prints the vectors. Writes NOTHING except an optional
JSON under analysis/rho_2026_09/.

Env construction mirrors evaluation/baseline_eval.py exactly, and the eval
seed schedule (eval_base=10000, 20 episodes) matches the basin_eval_base /
basin_eval_episodes recorded in every shard summary, so the reconstruction is
comparable to the stored system_spearman_rho.

Usage: python3 analysis/rho_2026_09/probe_vectors.py c9_armA:0 c11_armA:0 c3_armA:0
"""
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))

import torch
torch.set_num_threads(1)

from env.reputation_marl_env import ReputationMARLEnv
from agents.mappo import MAPPOTrainer
from training.train import build_system_config

SWEEP = REPO / "results" / "tier1b" / "sweep"
OUT = REPO / "analysis" / "rho_2026_09"


def probe(tag, seed, eval_episodes=20, eval_base=10000):
    cfg_path = REPO / "configs" / "tier1b" / f"sweep_{tag}.yaml"
    ckpt = SWEEP / "checkpoints" / f"sweep_{tag}" / f"seed{seed}" / "final.pt"
    cfg = yaml.safe_load(open(cfg_path))
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

    scores, quals, ratings = [], [], 0
    for e in range(eval_episodes):
        env.reset(seed=eval_base + e)
        while env.agents:
            ag = env.agent_selection
            if env.terminations.get(ag, False) or env.truncations.get(ag, False):
                env.step(None)
                continue
            a, _, _ = tr.select_action(ag, env.observe(ag), deterministic=True)
            env.step(a)
        m = env.get_metrics()
        scores.extend(m["agent_scores"])
        quals.extend(m["agent_true_qualities"])
        ratings += m.get("total_ratings", 0)

    s = np.asarray(scores, float)
    q = np.asarray(quals, float)
    stored = json.load(open(SWEEP / "training_logs" / f"sweep_{tag}_seed{seed}.json"))

    # exact guard from evaluation/basins.py:_score_quality_metrics
    if s.size == 0 or np.std(s) < 1e-12 or np.std(q) < 1e-12:
        rho = float("nan")
    else:
        from scipy.stats import spearmanr
        rho = float(spearmanr(s, q).statistic)

    return {
        "shard": f"sweep_{tag}_seed{seed}",
        "n_points": int(s.size),
        "score_std": float(np.std(s)),
        "qual_std": float(np.std(q)),
        "score_distinct": int(np.unique(s).size),
        "qual_distinct": int(np.unique(q).size),
        "score_min": float(s.min()), "score_max": float(s.max()),
        "qual_min": float(q.min()), "qual_max": float(q.max()),
        "total_ratings": int(ratings),
        "rho_recomputed": rho,
        "rho_stored": stored["system_spearman_rho"],
        "mae_recomputed": float(np.mean(np.abs(s - q))),
        "mae_stored": stored["system_mae"],
        "participation_stored": stored["participation_rate"],
        "basin_stored": stored["basin"],
        "score_head": [round(float(x), 6) for x in s[:12]],
        "qual_head": [round(float(x), 6) for x in q[:12]],
        # which branch of the guard fired
        "guard": ("score_collapse" if np.std(s) < 1e-12 else
                  "quality_collapse" if np.std(q) < 1e-12 else "none"),
    }


if __name__ == "__main__":
    targets = [a.split(":") for a in sys.argv[1:]] or [
        ["c9_armA", "0"], ["c11_armA", "0"], ["c3_armA", "0"]]
    res = []
    for tag, seed in targets:
        r = probe(tag, int(seed))
        res.append(r)
        print(f"\n=== {r['shard']}  [{r['basin_stored']}] ===")
        print(f"  n_points        {r['n_points']}")
        print(f"  total_ratings   {r['total_ratings']}   participation={r['participation_stored']}")
        print(f"  score  std={r['score_std']:.3e}  distinct={r['score_distinct']:<4} "
              f"range=[{r['score_min']:.6f}, {r['score_max']:.6f}]")
        print(f"  qual   std={r['qual_std']:.3e}  distinct={r['qual_distinct']:<4} "
              f"range=[{r['qual_min']:.6f}, {r['qual_max']:.6f}]")
        print(f"  GUARD FIRED: {r['guard']}")
        print(f"  scores[:12] {r['score_head']}")
        print(f"  quals [:12] {r['qual_head']}")
        print(f"  rho recomputed={r['rho_recomputed']}  stored={r['rho_stored']}")
        print(f"  mae recomputed={r['mae_recomputed']:.15f}  stored={r['mae_stored']:.15f}")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "probe_vectors.json").write_text(json.dumps(res, indent=2))
    print(f"\nwrote {OUT/'probe_vectors.json'}")
