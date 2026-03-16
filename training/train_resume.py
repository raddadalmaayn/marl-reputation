"""
Resume training for a specific config/seed from final.pt checkpoint.
Usage:
  python3 training/train_resume.py --config configs/configN.yaml --seed N \
      --output results/ --extra-episodes 20000 --wall-hours 10
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import yaml
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.set_num_threads(1)

from env.reputation_marl_env import ReputationMARLEnv
from agents.mappo import MAPPOTrainer
from training.train import run_episode, check_convergence


def train_resume(cfg: dict, seed: int, output_dir: Path,
                 extra_episodes: int, wall_hours: float) -> dict:
    np.random.seed(seed)
    torch.manual_seed(seed + 10000)  # different seed offset for extended run

    env = ReputationMARLEnv(
        n_agents=cfg["n_agents"],
        max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None),
        seed=seed,
    )

    trainer = MAPPOTrainer(obs_dim=14, act_dim=12)
    trainer.init_agents(env.possible_agents)

    ckpt_dir = output_dir / "checkpoints" / cfg["name"] / f"seed{seed}"
    log_dir  = output_dir / "training_logs"

    # Load checkpoint
    ckpt_path = ckpt_dir / "final.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}")
    trainer.load(str(ckpt_path))
    print(f"  Loaded checkpoint from {ckpt_path}")

    # Load existing per-seed summary to find start episode
    summary_path = log_dir / f"{cfg['name']}_seed{seed}.json"
    start_ep = 0
    prev_result = {}
    if summary_path.exists():
        with open(summary_path) as f:
            prev_result = json.load(f)
        start_ep = prev_result.get("episodes_trained", 0)
    print(f"  Resuming from episode {start_ep}")

    # Load existing log for reward history (used for convergence detection)
    log_path = log_dir / f"{cfg['name']}_seed{seed}_log.json"
    existing_log = []
    reward_history = []
    if log_path.exists():
        with open(log_path) as f:
            existing_log = json.load(f)
        reward_history = [e["mean_reward"] for e in existing_log]

    ckpt_every = cfg.get("checkpoint_every", 250)
    conv_window = cfg.get("convergence_window", 100)
    conv_thresh = cfg.get("convergence_reward_variance_threshold", 0.05)
    hard_limit  = start_ep + extra_episodes

    wall_budget = wall_hours * 3600
    start_time  = time.time()

    new_log = []
    converged = False
    converged_at = None

    ep = start_ep
    print(f"  [{cfg['name']} seed={seed}] Extended run: ep {start_ep} → up to {hard_limit} "
          f"(wall budget {wall_hours:.1f}h)")

    while ep < hard_limit:
        ep += 1
        lr_mult = max(0.1, 1.0 - ep / hard_limit)
        ep_metrics = run_episode(env, trainer, collect=True)
        update_info = trainer.update(lr_multiplier=lr_mult)
        ep_metrics.update(update_info or {})
        ep_metrics["episode"] = ep
        ep_metrics["seed"] = seed
        new_log.append(ep_metrics)
        reward_history.append(ep_metrics["mean_reward"])

        if ep % 100 == 0:
            elapsed = time.time() - start_time
            print(f"  [{cfg['name']} seed={seed}] ep={ep} "
                  f"reward={ep_metrics['mean_reward']:.3f} "
                  f"honest={ep_metrics['honest_pct']:.2f} "
                  f"acc={ep_metrics['mean_reputation_accuracy']:.3f} "
                  f"elapsed={elapsed:.0f}s")

        if ep % ckpt_every == 0:
            trainer.save(str(ckpt_dir / f"ep{ep}.pt"))

        # Convergence check (over full reward history including prior run)
        if check_convergence(reward_history, conv_window, conv_thresh):
            converged = True
            converged_at = ep
            print(f"  [{cfg['name']} seed={seed}] CONVERGED at ep={ep}")
            break

        if time.time() - start_time > wall_budget:
            print(f"  [{cfg['name']} seed={seed}] Wall-clock budget exceeded at ep={ep}")
            break

    # Final checkpoint
    trainer.save(str(ckpt_dir / "final.pt"))

    # Evaluation: deterministic
    eval_ep = cfg.get("eval_episodes", 20)
    eval_rewards, eval_accs, eval_honest = [], [], []
    for _ in range(eval_ep):
        m = run_episode(env, trainer, deterministic=True, collect=False)
        eval_rewards.append(m["mean_reward"])
        eval_accs.append(m["mean_reputation_accuracy"])
        eval_honest.append(m["honest_pct"])

    total_time = (prev_result.get("training_time_s", 0)
                  + (time.time() - start_time))

    result = {
        "config": cfg["name"],
        "seed": seed,
        "converged": converged,
        "converged_at": converged_at,
        "episodes_trained": ep,
        "training_time_s": total_time,
        "eval_mean_reward":    float(np.mean(eval_rewards)),
        "eval_std_reward":     float(np.std(eval_rewards)),
        "eval_mean_accuracy":  float(np.mean(eval_accs)),
        "eval_std_accuracy":   float(np.std(eval_accs)),
        "eval_mean_honest_pct": float(np.mean(eval_honest)),
        "eval_std_honest_pct":  float(np.std(eval_honest)),
        "final_env_metrics": env.get_metrics(),
        "resumed_from_ep": start_ep,
    }

    # Overwrite seed summary
    with open(summary_path, "w") as f:
        json.dump(result, f, indent=2)

    # Append new log entries to existing log
    combined_log = existing_log + new_log
    with open(log_path, "w") as f:
        json.dump(combined_log, f)

    print(f"  [{cfg['name']} seed={seed}] Done. ep={ep}, "
          f"converged={converged}, reward={result['eval_mean_reward']:.3f}, "
          f"acc={result['eval_mean_accuracy']:.3f}")
    return result


def rebuild_summary(cfg: dict, output_dir: Path, n_seeds: int):
    """Rebuild the config-level summary JSON from per-seed results."""
    log_dir = output_dir / "training_logs"
    all_results = []
    for seed in range(n_seeds):
        path = log_dir / f"{cfg['name']}_seed{seed}.json"
        if path.exists():
            with open(path) as f:
                all_results.append(json.load(f))

    if not all_results:
        return

    summary = {
        "config": cfg["name"],
        "description": cfg.get("description", ""),
        "n_seeds": len(all_results),
        "converged_count": sum(r["converged"] for r in all_results),
        "mean_eval_reward":    float(np.mean([r["eval_mean_reward"]    for r in all_results])),
        "std_eval_reward":     float(np.std( [r["eval_mean_reward"]    for r in all_results])),
        "mean_eval_accuracy":  float(np.mean([r["eval_mean_accuracy"]  for r in all_results])),
        "std_eval_accuracy":   float(np.std( [r["eval_mean_accuracy"]  for r in all_results])),
        "mean_honest_pct":     float(np.mean([r["eval_mean_honest_pct"] for r in all_results])),
        "std_honest_pct":      float(np.std( [r["eval_mean_honest_pct"] for r in all_results])),
        "total_training_time_s": sum(r["training_time_s"] for r in all_results),
        "seed_results": all_results,
    }
    summary_path = log_dir / f"{cfg['name']}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"  Summary updated: {summary_path}")
    print(f"  Converged: {summary['converged_count']}/{len(all_results)} seeds")
    print(f"  Eval reward: {summary['mean_eval_reward']:.3f} ± {summary['std_eval_reward']:.3f}")
    print(f"  Accuracy:    {summary['mean_eval_accuracy']:.3f} ± {summary['std_eval_accuracy']:.3f}")
    print(f"  Honest %:    {summary['mean_honest_pct']:.3f} ± {summary['std_honest_pct']:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--extra-episodes", type=int, default=20000,
                        help="Additional episodes beyond start_ep")
    parser.add_argument("--wall-hours", type=float, default=10.0,
                        help="Wall-clock budget in hours")
    parser.add_argument("--rebuild-summary", action="store_true",
                        help="Rebuild config summary after run")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)

    print(f"\n{'='*60}")
    print(f"RESUME Training: {cfg['name']}  seed={args.seed}")
    print(f"Extra episodes: {args.extra_episodes}  Wall budget: {args.wall_hours}h")
    print(f"{'='*60}")

    train_resume(cfg, args.seed, output_dir, args.extra_episodes, args.wall_hours)

    if args.rebuild_summary:
        rebuild_summary(cfg, output_dir, n_seeds=cfg.get("seeds", 5))


if __name__ == "__main__":
    main()
