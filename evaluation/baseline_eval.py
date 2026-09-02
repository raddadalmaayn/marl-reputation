"""
Tier 1B Stage 2 — apples-to-apples estimator comparison (Correction B).

Loads the MAIN-SWEEP Beta-trained checkpoints for a config/arm/seed and evaluates
the SAME learned adversary policy (Beta drives obs/actions during eval), recording
the rating-event stream. The identical stream is then scored under three estimators:

  beta        — the env's native in-loop Beta score (with temporal decay).
  flat        — FlatAverageEngine, maintained ONLINE / in-loop by replaying events
                in order (unweighted Beta count, no decay).
  eigentrust  — EigenTrustScorer, computed POST-HOC/global from the accumulated graph.

Asymmetry preserved & documented: flat is an online per-event update; eigentrust is a
batch global fixed point. Both consume the identical Beta-driven action stream, so the
adversary behavior and the attack/hold statistics are the SAME across estimators —
only the reputation estimate (hence rho vs true latent quality) differs. Hold rate is
therefore reported once per (config,arm,seed); rho is reported per estimator.

Usage:
  python3 evaluation/baseline_eval.py --config configs/tier1b/sweep_c2_armA.yaml \
      --ckpt results/tier1b/sweep/checkpoints/sweep_c2_armA/seed0/final.pt --seed 0 \
      --out results/tier1b/sweep/baselines --eval-episodes 20
"""

import argparse
import json
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import numpy as np
import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
torch.set_num_threads(1)

from env.reputation_marl_env import ReputationMARLEnv
from env.reputation_engine import (ReputationEngine, ReputationState, SystemConfig)
from agents.mappo import MAPPOTrainer
from evaluation.baseline_engines import FlatAverageEngine, EigenTrustScorer
from training.train import build_system_config


def _rho(scores, quals):
    from scipy.stats import spearmanr
    s = np.asarray(scores, float); q = np.asarray(quals, float)
    if s.size == 0 or np.std(s) < 1e-12 or np.std(q) < 1e-12:
        return float("nan")
    return float(spearmanr(s, q).statistic)


def _mae(scores, quals):
    s = np.asarray(scores, float); q = np.asarray(quals, float)
    return float(np.mean(np.abs(s - q))) if s.size else float("nan")


def flat_scores(events, n_agents, cfg):
    """Online replay of the event stream under the flat engine (in-loop)."""
    eng = FlatAverageEngine(cfg)
    st = [ReputationState(f"agent_{i}", "quality", cfg.initial_alpha, cfg.initial_beta,
                          last_ts=0.0) for i in range(n_agents)]
    for (r, t, o, w) in events:
        st[t] = eng.apply_rating(st[t], o, now=0.0)
    return [s.score for s in st]


def eigentrust_scores(events, n_agents):
    eig = EigenTrustScorer(n_agents)
    for (r, t, o, w) in events:
        eig.add_event(r, t, o, w)
    return list(eig.scores())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--eval-episodes", type=int, default=20)
    ap.add_argument("--eval-base", type=int, default=10000)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    name = cfg["name"]
    outdir = Path(args.out); outdir.mkdir(parents=True, exist_ok=True)
    outp = outdir / f"{name}_seed{args.seed}_baseline.json"
    if outp.exists() and not args.force:
        print(f"[skip] {outp}")
        return
    if not Path(args.ckpt).exists():
        print(f"[missing-ckpt] {args.ckpt}")
        return

    sys_cfg = build_system_config(cfg)
    env = ReputationMARLEnv(
        n_agents=cfg["n_agents"], max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None),
        config=sys_cfg, seed=args.seed,
        stake_obs_mode=cfg.get("stake_obs_mode", "absolute_log"),
        participation_coef=cfg.get("participation_coef", 0.0),
        reputation_engine="beta",          # Beta drives the shared adversaries
        record_rating_events=True,
    )
    tr = MAPPOTrainer(obs_dim=14, act_dim=12)
    tr.init_agents(env.possible_agents)
    tr.load(args.ckpt)

    beta_s, beta_q = [], []
    flat_s, flat_q = [], []
    eig_s, eig_q = [], []
    att = blk = 0
    n = cfg["n_agents"]
    for e in range(args.eval_episodes):
        env.reset(seed=args.eval_base + e)
        while env.agents:
            ag = env.agent_selection
            if env.terminations.get(ag, False) or env.truncations.get(ag, False):
                env.step(None); continue
            obs = env.observe(ag)
            a, _, _ = tr.select_action(ag, obs, deterministic=True)
            env.step(a)
        m = env.get_metrics()
        q = m["agent_true_qualities"]
        beta_s += m["agent_scores"]; beta_q += q
        ev = env._rating_events
        flat_s += flat_scores(ev, n, sys_cfg); flat_q += q
        eig_s += eigentrust_scores(ev, n); eig_q += q
        att += m["attacks_attempted"]; blk += m["attacks_blocked"]

    hold = (blk / att) if att > 0 else float("nan")
    res = {
        "config": name, "seed": args.seed, "arm":
            ("B" if cfg.get("participation_coef", 0.0) > 0 else "A"),
        "eval_episodes": args.eval_episodes,
        "hold_rate": hold, "attacks_attempted": att, "attacks_blocked": blk,
        "successful_attacks": att - blk,
        "engines": {
            "beta": {"rho": _rho(beta_s, beta_q), "mae": _mae(beta_s, beta_q)},
            "flat": {"rho": _rho(flat_s, flat_q), "mae": _mae(flat_s, flat_q)},
            "eigentrust": {"rho": _rho(eig_s, eig_q), "mae": _mae(eig_s, eig_q)},
        },
    }
    outp.write_text(json.dumps(res, indent=2))
    print(f"[done] {outp}  hold={hold:.3f}  "
          f"rho beta={res['engines']['beta']['rho']:.3f} "
          f"flat={res['engines']['flat']['rho']:.3f} "
          f"eig={res['engines']['eigentrust']['rho']:.3f}")


if __name__ == "__main__":
    main()
