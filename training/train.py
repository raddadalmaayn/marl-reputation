"""
Training script for MARL reputation adversarial evaluation.
Usage: python3 training/train.py --config configs/configN.yaml --output results/ --seeds N
"""

import argparse
import json
import os
import sys
import time
import numpy as np
import yaml
from pathlib import Path

# Limit PyTorch to 1 thread per process — critical for parallel runs on CPU
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.set_num_threads(1)

from env.reputation_marl_env import ReputationMARLEnv
from agents.mappo import MAPPOTrainer


def run_episode(env: ReputationMARLEnv, trainer: MAPPOTrainer,
                deterministic: bool = False, collect: bool = True) -> dict:
    env.reset()
    episode_rewards = {ag: 0.0 for ag in env.possible_agents}
    honest_actions, dishonest_actions, total_actions = 0, 0, 0
    prev_obs = {}

    # Collect initial observations
    for ag in env.possible_agents:
        prev_obs[ag] = env.observe(ag)

    steps = 0
    while env.agents:
        agent = env.agent_selection
        # Handle dead agents (truncated/terminated): pass None per PettingZoo spec
        if env.terminations.get(agent, False) or env.truncations.get(agent, False):
            env.step(None)
            continue
        obs = env.observe(agent)
        action, logprob, value = trainer.select_action(agent, obs, deterministic)
        env.step(action)
        reward = env.rewards.get(agent, 0.0)
        done = env.terminations.get(agent, False) or env.truncations.get(agent, False)

        if collect:
            trainer.store_transition(agent, obs, action, logprob, reward, value, done)

        episode_rewards[agent] = episode_rewards.get(agent, 0.0) + reward
        total_actions += 1
        if action in (1, 2):
            honest_actions += 1
        elif action in (3, 4):
            dishonest_actions += 1
        steps += 1

    metrics = env.get_metrics()
    total_reward = sum(episode_rewards.values())
    honest_pct = honest_actions / max(1, honest_actions + dishonest_actions)

    return {
        "total_reward": total_reward,
        "mean_reward": total_reward / len(env.possible_agents),
        "honest_pct": honest_pct,
        "dishonest_actions": dishonest_actions,
        "total_actions": total_actions,
        **metrics,
    }


def check_convergence(reward_history: list, window: int, threshold: float) -> bool:
    if len(reward_history) < window:
        return False
    recent = reward_history[-window:]
    return float(np.var(recent)) < threshold


def train_config(cfg: dict, seed: int, output_dir: Path) -> dict:
    np.random.seed(seed)
    import torch; torch.manual_seed(seed)

    env = ReputationMARLEnv(
        n_agents=cfg["n_agents"],
        max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None),
        seed=seed,
        terminal_reward_coef=cfg.get("terminal_reward_coef", 0.0),
    )

    trainer = MAPPOTrainer(obs_dim=14, act_dim=12)
    trainer.init_agents(env.possible_agents)

    max_ep = cfg.get("episodes", 5000)
    max_ep_ext = cfg.get("max_episodes_extended", 10000)
    ckpt_every = cfg.get("checkpoint_every", 500)
    conv_window = cfg.get("convergence_window", 100)
    conv_thresh = cfg.get("convergence_reward_variance_threshold", 0.05)

    ckpt_dir = output_dir / "checkpoints" / cfg["name"] / f"seed{seed}"
    log_dir  = output_dir / "training_logs"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    log = []
    reward_history = []
    converged = False
    converged_at = None
    start_time = time.time()

    # Wall-clock budget: 8 hours per config per seed (unattended overnight runs)
    wall_budget = 10 * 3600

    ep = 0
    hard_limit = max_ep_ext
    while ep < hard_limit:
        ep += 1
        lr_mult = max(0.1, 1.0 - ep / hard_limit)
        ep_metrics = run_episode(env, trainer, collect=True)
        update_info = trainer.update(lr_multiplier=lr_mult)
        ep_metrics.update(update_info or {})
        ep_metrics["episode"] = ep
        ep_metrics["seed"] = seed
        log.append(ep_metrics)
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

        # Check convergence only after initial max_ep
        if ep >= max_ep and not converged:
            if check_convergence(reward_history, conv_window, conv_thresh):
                converged = True
                converged_at = ep
                print(f"  [{cfg['name']} seed={seed}] CONVERGED at ep={ep}")
                break

        if ep >= max_ep and not converged:
            pass  # continue to max_ep_ext

        if ep >= hard_limit:
            break

        # Wall-clock budget
        if time.time() - start_time > wall_budget:
            print(f"  [{cfg['name']} seed={seed}] Wall-clock budget exceeded, stopping at ep={ep}")
            break

    # Final checkpoint
    trainer.save(str(ckpt_dir / "final.pt"))

    # Evaluation: 100 episodes deterministic
    eval_rewards, eval_accs, eval_honest = [], [], []
    eval_ep = cfg.get("eval_episodes", 100)
    for _ in range(eval_ep):
        m = run_episode(env, trainer, deterministic=True, collect=False)
        eval_rewards.append(m["mean_reward"])
        eval_accs.append(m["mean_reputation_accuracy"])
        eval_honest.append(m["honest_pct"])

    total_time = time.time() - start_time

    result = {
        "config": cfg["name"],
        "seed": seed,
        "converged": converged,
        "converged_at": converged_at,
        "episodes_trained": ep,
        "training_time_s": total_time,
        "eval_mean_reward": float(np.mean(eval_rewards)),
        "eval_std_reward":  float(np.std(eval_rewards)),
        "eval_mean_accuracy": float(np.mean(eval_accs)),
        "eval_std_accuracy":  float(np.std(eval_accs)),
        "eval_mean_honest_pct": float(np.mean(eval_honest)),
        "eval_std_honest_pct":  float(np.std(eval_honest)),
        "final_env_metrics": env.get_metrics(),
        "log": log,
    }

    # Save per-seed result
    result_path = log_dir / f"{cfg['name']}_seed{seed}.json"
    with open(result_path, "w") as f:
        # Log is large: save separately
        light = {k: v for k, v in result.items() if k != "log"}
        json.dump(light, f, indent=2)

    log_path = log_dir / f"{cfg['name']}_seed{seed}_log.json"
    with open(log_path, "w") as f:
        json.dump(log, f)

    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--output", default="results/")
    parser.add_argument("--seeds", type=int, default=5)
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Training: {cfg['name']}")
    print(f"Description: {cfg.get('description','')}")
    print(f"Seeds: {args.seeds}")
    print(f"{'='*60}")

    all_results = []
    for seed in range(args.seeds):
        print(f"\n--- Seed {seed} ---")
        r = train_config(cfg, seed, output_dir)
        all_results.append(r)

    # Aggregate across seeds
    summary = {
        "config": cfg["name"],
        "description": cfg.get("description", ""),
        "n_seeds": args.seeds,
        "converged_count": sum(r["converged"] for r in all_results),
        "mean_eval_reward": float(np.mean([r["eval_mean_reward"] for r in all_results])),
        "std_eval_reward":  float(np.std( [r["eval_mean_reward"] for r in all_results])),
        "mean_eval_accuracy": float(np.mean([r["eval_mean_accuracy"] for r in all_results])),
        "std_eval_accuracy":  float(np.std( [r["eval_mean_accuracy"] for r in all_results])),
        "mean_honest_pct": float(np.mean([r["eval_mean_honest_pct"] for r in all_results])),
        "std_honest_pct":  float(np.std( [r["eval_mean_honest_pct"] for r in all_results])),
        "total_training_time_s": sum(r["training_time_s"] for r in all_results),
        "seed_results": [{k: v for k, v in r.items() if k != "log"} for r in all_results],
    }

    summary_path = output_dir / "training_logs" / f"{cfg['name']}_summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    print(f"\n{'='*60}")
    print(f"DONE: {cfg['name']}")
    print(f"  Converged: {summary['converged_count']}/{args.seeds} seeds")
    print(f"  Eval reward: {summary['mean_eval_reward']:.3f} ± {summary['std_eval_reward']:.3f}")
    print(f"  Honest %:   {summary['mean_honest_pct']:.3f} ± {summary['std_honest_pct']:.3f}")
    print(f"  Accuracy:   {summary['mean_eval_accuracy']:.3f} ± {summary['std_eval_accuracy']:.3f}")
    print(f"  Time: {summary['total_training_time_s']:.0f}s")
    print(f"  Summary: {summary_path}")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    main()
