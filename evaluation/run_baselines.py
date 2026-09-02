"""
Tier 1B Stage 2 driver — run baseline_eval.py on every COMPLETED {C2,C3,C11} x arm x
seed sweep checkpoint (idempotent; skips done; skips not-yet-trained). Apples-to-apples:
same Beta-trained policy, estimator swapped at eval.

Usage: python3 evaluation/run_baselines.py [--eval-episodes 20]
"""
import argparse
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
SWEEP = REPO / "results" / "tier1b" / "sweep"
OUT = SWEEP / "baselines"
CONFIGS = ["c2", "c3", "c11"]          # the baseline-comparison configs
SEEDS = 30                              # bimodal/key configs: 30 seeds/arm


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-episodes", type=int, default=20)
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    ran = skipped = pending = 0
    for cbase in CONFIGS:
        for arm in ("armA", "armB"):
            cfg = REPO / "configs" / "tier1b" / f"sweep_{cbase}_{arm}.yaml"
            for s in range(SEEDS):
                ck = SWEEP / "checkpoints" / f"sweep_{cbase}_{arm}" / f"seed{s}" / "final.pt"
                outp = OUT / f"sweep_{cbase}_{arm}_seed{s}_baseline.json"
                if outp.exists():
                    skipped += 1; continue
                if not ck.exists():
                    pending += 1; continue       # sweep shard not finished yet
                subprocess.run([sys.executable, str(REPO / "evaluation" / "baseline_eval.py"),
                                "--config", str(cfg), "--ckpt", str(ck), "--seed", str(s),
                                "--out", str(OUT), "--eval-episodes", str(args.eval_episodes)],
                               check=False)
                ran += 1
    print(f"\nbaseline_eval: ran={ran} skipped(done)={skipped} pending(no ckpt yet)={pending}")


if __name__ == "__main__":
    main()
