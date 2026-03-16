# MARL Adversarial Evaluation of a Blockchain Reputation System

A Multi-Agent Reinforcement Learning (MARL) framework for adversarial evaluation of a
Bayesian Beta-reputation system designed for additive manufacturing (AM) supply chains.
Learned adversaries are trained simultaneously with honest agents using MAPPO, continuously
searching for exploitable weaknesses across a 12-action, 14-dimensional observation space.

This is the companion evaluation framework to the blockchain system described in:
> *Unified Blockchain-Based Provenance and Bayesian Reputation Framework for Additive
> Manufacturing Supply Chains* (under review, IEEE TDSC 2026)

---

## Overview

Instead of static scripted attack injection, this framework trains adversarial agents via
reinforcement learning to discover optimal manipulation strategies against a reputation
system with:

- **Bayesian Beta scoring** with Wilson 95% confidence intervals
- **Exponential temporal decay** (λ = 0.98/day)
- **Stake-based economic incentives** (slashing on dishonest behavior)
- **Layered access controls** (admin gates, reputation gates)

The environment models all 9 attack vectors from the companion blockchain security evaluation,
enabling a direct comparison between scripted and learned adversarial outcomes.

---

## Results Summary

| Config | Attack Type | Honest % | Defense Block Rate |
|--------|-------------|----------|--------------------|
| C1 | Baseline (all honest) | 99.84% | — |
| C2 | Mixed adversaries | 99.70% | 51.9% |
| C3 | Sybil flooding | 87.55% | 52.0% |
| C4 | Collusion ring | 99.46% | 58.3% |
| C5 | Adaptive (2× bonus) | 99.75% | 50.1% |
| C6 | Self-rating | 93.17% | **100.0%** |
| C7 | Admin escalation | 99.87% | **100.0%** |
| C8 | Evidence tampering | 99.69% | 81.5% |
| C9 | Gate bypass | 99.74% | **100.0%** |
| C10 | Provenance replay | 99.86% | 27.1% |
| C11 | All attacks combined | 69.86% | 54.1% |

*Metrics: mean over final 20% of training episodes, 5 seeds each.*

---

## Repository Structure

```
marl-reputation/
├── env/
│   ├── reputation_marl_env.py   # PettingZoo AEC environment (12 actions, 14-dim obs)
│   └── reputation_engine.py     # Bayesian Beta scoring, Wilson CI, stake logic
├── agents/
│   ├── mappo.py                 # MAPPO with shared actor-critic (256→128 MLP)
│   └── independent_ppo.py       # Independent PPO baseline
├── training/
│   └── train.py                 # Training loop, convergence detection, checkpointing
├── evaluation/
│   ├── evaluate.py              # Post-training evaluation across seeds
│   ├── generate_figures.py      # Matplotlib figure generation
│   └── generate_tables.py       # LaTeX table generation
├── configs/
│   ├── config1.yaml             # Baseline: all honest agents
│   ├── config2.yaml             # Mixed: 5 adversarial + 15 honest
│   ├── config3.yaml             # Sybil flooding
│   ├── config4.yaml             # Collusion ring
│   ├── config5.yaml             # Adaptive adversary (2× reward bonus)
│   ├── config6.yaml             # Self-rating attack
│   ├── config7.yaml             # Admin escalation attack
│   ├── config8.yaml             # Evidence tampering (p=0.80 detection)
│   ├── config9.yaml             # Reputation gate bypass
│   ├── config10.yaml            # Provenance replay
│   └── config11.yaml            # Comprehensive: all 9 attacks combined
├── scripts/
│   └── run_post_training.sh     # Pipeline: eval → figures → tables → paper copy
├── results/
│   ├── training_logs/           # Per-config summary JSONs (seed logs excluded)
│   └── figures/                 # Generated evaluation figures
└── README.md
```

---

## Environment Design

### Action Space (12 discrete actions)

| Action | Description | Defense |
|--------|-------------|---------|
| 0 | No-operation (abstain) | — |
| 1 | Submit positive honest rating | — |
| 2 | Submit negative honest rating | — |
| 3 | Submit positive dishonest rating | 85% detection → stake slash |
| 4 | Submit negative dishonest rating | 85% detection → stake slash |
| 5 | Create Sybil identity | CI widening, stake cost |
| 6 | File dispute | — |
| 7 | Attempt self-rating | Deterministic block, −5.0 reward |
| 8 | Attempt admin escalation | Deterministic block, −5.0 reward |
| 9 | Tamper with evidence | 80% detection, −2.0 reward |
| 10 | Attempt gate bypass | Deterministic block, −3.0 reward |
| 11 | Attempt provenance replay | Deterministic block, −5.0 reward |

### Observation Space (14 dimensions)

```
[own_score, ci_width, stake, alpha, beta,
 target_score, target_ci_width,
 ratings_submitted, disputes_received, disputes_lost,
 normalized_time, sybil_count, blocks_normalized, gate_eligible]
```

---

## Installation

```bash
git clone https://github.com/raddadalmaayn/marl-reputation.git
cd marl-reputation
pip install torch numpy pyyaml pettingzoo matplotlib
```

**Requirements:** Python 3.9+, PyTorch 2.x, PettingZoo 1.24+

---

## Running Training

```bash
# Single config, 5 seeds
python3 training/train.py --config configs/config1.yaml --output results/ --seeds 5

# All 11 configs in parallel (background)
for i in $(seq 1 11); do
    python3 training/train.py --config configs/config${i}.yaml \
        --output results/ --seeds 5 &
done
wait
```

**Note:** Set `OMP_NUM_THREADS=1` and `MKL_NUM_THREADS=1` before running parallel jobs
to prevent CPU thread contention (already set inside `train.py`).

### Post-training pipeline

After all 11 configs complete, the post-training script auto-runs evaluation,
figure generation, and table generation:

```bash
bash scripts/run_post_training.sh
```

This waits for all 11 `*_summary.json` files to appear, then runs the full pipeline.

---

## MAPPO Architecture

- **Network:** MLP 256 → 128 → ReLU, shared actor + critic heads
- **Algorithm:** PPO with clipped surrogate (ε = 0.2)
- **Advantage:** GAE (γ = 0.99, λ = 0.95)
- **Entropy coefficient:** 0.05 (higher than default to encourage exploration of 12 actions)
- **Learning rate:** 3×10⁻⁴, linearly decayed to 3×10⁻⁵
- **Episodes:** up to 10,000 per seed; early stop if reward variance < 0.05 over 100 episodes
- **Agents:** 20 per environment (N_adv varies by config)

---

## License

MIT License. See [LICENSE](LICENSE).

---

## Contact

**Raddad Almaayn** — ralmaayn@unm.edu
Dept. of Electrical and Computer Engineering, University of New Mexico, USA

> This repository is a draft. Author list will be updated upon publication.

---

## Citation

> Citation will be added upon publication.
