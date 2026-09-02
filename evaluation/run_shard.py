"""
Tier 1B — run ONE (config, seed) training shard and write its per-seed summary.

This is the atomic unit for both the pilot and the distributed full sweep: every
(config x arm x seed) is one independent shard, so shards can be fanned out across
hosts/cores with no coordination. Reuses training.train.train_config unchanged.

Single-thread / fixed-seed reproducibility is enforced (OMP/MKL=1, np+torch seeded
inside train_config). Idempotent: skips if the per-seed summary already exists
unless --force.

Usage:
  python3 evaluation/run_shard.py --config configs/tier1b/pilot_c2_armB.yaml \
      --seed 0 --output results/tier1b/pilot
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

import yaml

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))

import torch
torch.set_num_threads(1)

from training.train import train_config


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config))
    out = Path(args.output)
    summ = out / "training_logs" / f"{cfg['name']}_seed{args.seed}.json"
    if summ.exists() and not args.force:
        print(f"[skip] {summ} exists")
        return
    out.mkdir(parents=True, exist_ok=True)
    print(f"[run] {cfg['name']} seed={args.seed} -> {out}")
    train_config(cfg, args.seed, out)
    print(f"[done] {summ}")


if __name__ == "__main__":
    main()
