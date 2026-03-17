"""
Compute Spearman rho (reputation discrimination) for all ablation configs.
Loads final.pt checkpoints, runs 100 deterministic episodes per seed,
computes spearmanr(final_scores, true_qualities) per episode, averages.
Outputs results to stdout and writes results/ablation/spearman_results.json.

Usage: python3 scripts/eval_ablation_spearman.py
"""

import json, sys
from pathlib import Path
import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
torch.set_num_threads(1)

from scipy.stats import spearmanr
from env.reputation_marl_env import ReputationMARLEnv
from agents.mappo import MAPPOTrainer

CKPT_BASE = Path("/home/raddad/marl-reputation/results/ablation/checkpoints")
CONF_BASE = Path("/home/raddad/marl-reputation/configs")
MAIN_LOG  = Path("/home/raddad/marl-reputation/results/training_logs")
OUT_FILE  = Path("/home/raddad/marl-reputation/results/ablation/spearman_results.json")
N_EVAL    = 20
N_SEEDS   = 3


def run_eval_episode(env, trainer):
    env.reset()
    while env.agents:
        agent = env.agent_selection
        if env.terminations.get(agent, False) or env.truncations.get(agent, False):
            env.step(None)
            continue
        obs = env.observe(agent)
        action, _, _ = trainer.select_action(agent, obs, deterministic=True)
        env.step(action)
    m = env.get_metrics()
    scores = m["agent_scores"]
    qualities = m["agent_true_qualities"]
    if len(set(scores)) < 2 or len(set(qualities)) < 2:
        return None  # degenerate case
    rho, _ = spearmanr(scores, qualities)
    return float(rho)


def eval_config(cfg_name, cfg_path, seeds=N_SEEDS):
    with open(cfg_path) as f:
        cfg = yaml.safe_load(f)

    env = ReputationMARLEnv(
        n_agents=cfg["n_agents"],
        max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None),
        seed=42,
    )

    rhos_all = []
    for seed in range(seeds):
        ckpt_dir = CKPT_BASE / cfg_name / f"seed{seed}"
        ckpt_path = ckpt_dir / "final.pt"
        if not ckpt_path.exists():
            print(f"  SKIP: {ckpt_path} not found")
            continue

        trainer = MAPPOTrainer(obs_dim=14, act_dim=12)
        trainer.init_agents(env.possible_agents)
        trainer.load(str(ckpt_path))

        seed_rhos = []
        for ep in range(N_EVAL):
            rho = run_eval_episode(env, trainer)
            if rho is not None:
                seed_rhos.append(rho)

        if seed_rhos:
            mean_rho = float(np.mean(seed_rhos))
            rhos_all.extend(seed_rhos)
            print(f"  {cfg_name} seed{seed}: rho={mean_rho:.4f} (n={len(seed_rhos)})")

    if not rhos_all:
        return None
    return {
        "mean_rho": float(np.mean(rhos_all)),
        "std_rho":  float(np.std(rhos_all)),
        "n_episodes": len(rhos_all),
    }


def main():
    ablation_configs = [
        ("ablation_no_stake",   CONF_BASE / "ablation_no_stake.yaml"),
        ("ablation_no_wilson",  CONF_BASE / "ablation_no_wilson.yaml"),
        ("ablation_no_decay",   CONF_BASE / "ablation_no_decay.yaml"),
        ("ablation_no_dispute", CONF_BASE / "ablation_no_dispute.yaml"),
    ]

    # Baseline: full system (config2_mixed), using main training checkpoints
    baseline_ckpt = Path("/home/raddad/marl-reputation/results/checkpoints/config2_mixed")
    full_cfg_path = CONF_BASE / "config2.yaml"

    results = {}

    # Full system baseline
    print("Evaluating: Full System (config2_mixed)")
    with open(full_cfg_path) as f:
        cfg = yaml.safe_load(f)
    env = ReputationMARLEnv(
        n_agents=cfg["n_agents"], max_steps=cfg["max_steps"],
        adversarial_ids=cfg.get("adversarial_ids", []),
        adversarial_reward_bonus=cfg.get("adversarial_reward_bonus", 0.0),
        collusion_groups=cfg.get("collusion_groups", []),
        max_sybils_per_agent=cfg.get("max_sybils_per_agent", 0),
        enabled_attacks=cfg.get("enabled_attacks", None), seed=42,
    )
    full_rhos = []
    for seed in range(5):
        ckpt_path = baseline_ckpt / f"seed{seed}" / "final.pt"
        if not ckpt_path.exists():
            continue
        trainer = MAPPOTrainer(obs_dim=14, act_dim=12)
        trainer.init_agents(env.possible_agents)
        trainer.load(str(ckpt_path))
        for ep in range(N_EVAL):
            rho = run_eval_episode(env, trainer)
            if rho is not None:
                full_rhos.append(rho)
    if full_rhos:
        results["full_system"] = {
            "mean_rho": float(np.mean(full_rhos)),
            "std_rho":  float(np.std(full_rhos)),
        }
        print(f"  Full System: rho={results['full_system']['mean_rho']:.4f} ± {results['full_system']['std_rho']:.4f}")

    # Ablations
    for cfg_name, cfg_path in ablation_configs:
        print(f"Evaluating: {cfg_name}")
        r = eval_config(cfg_name, cfg_path)
        if r:
            results[cfg_name] = r
            print(f"  {cfg_name}: rho={r['mean_rho']:.4f} ± {r['std_rho']:.4f}")

    OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_FILE, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to {OUT_FILE}")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
