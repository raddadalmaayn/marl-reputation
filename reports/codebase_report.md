# marl-reputation — Full Codebase & Results Report

**Purpose of this document:** Complete technical snapshot of the `marl-reputation` repository as of April 2026. Written to inform an improvement plan. Every source file, config, result file, log, and paper asset is covered.

---

## Table of Contents

1. [Project Overview](#1-project-overview)
2. [Repository Layout](#2-repository-layout)
3. [Environment Layer (`env/`)](#3-environment-layer-env)
4. [Agent Layer (`agents/`)](#4-agent-layer-agents)
5. [Training Layer (`training/`)](#5-training-layer-training)
6. [Evaluation Layer (`evaluation/`)](#6-evaluation-layer-evaluation)
7. [Configs (`configs/`)](#7-configs-configs)
8. [Scripts (`scripts/`)](#8-scripts-scripts)
9. [Results — Main Training](#9-results--main-training-11-configs)
10. [Results — Ablations](#10-results--ablations)
11. [Results — Sensitivity Sweeps](#11-results--sensitivity-sweeps-adversarial-reward-bonus)
12. [Results — IPPO & Terminal Variants](#12-results--ippo--terminal-variants)
13. [Results — Analysis & Decomposition](#13-results--analysis--decomposition)
14. [LaTeX Tables](#14-latex-tables)
15. [Training Logs](#15-training-logs-logs)
16. [V1 Backup (Reproducibility Note)](#16-v1-backup-results)
17. [Paper (`marl-paper/`)](#17-paper-marl-paper)
18. [Figures](#18-figures)
19. [Key Numerical Takeaways](#19-key-numerical-takeaways)
20. [Observed Weaknesses & Open Questions](#20-observed-weaknesses--open-questions)

---

## 1. Project Overview

`marl-reputation` trains multi-agent reinforcement learning (MARL) agents to interact with a **Hyperledger Fabric Bayesian reputation system** (ported from `am-unified` chaincode) inside a PettingZoo AEC environment. The goal is adversarial evaluation: do honest agents learn to behave honestly, and do defense mechanisms hold under rational adversarial pressure?

**Target publication:** IEEE TDSC 2026 — "Adversarial Evaluation of a Blockchain-Based Bayesian Reputation System Using Multi-Agent Reinforcement Learning: Architecture, Defense Mechanisms, and Empirical Analysis"

**Parallel draft:** Elsevier Computer Networks (same structure, different template).

**Key claims in abstract:**
- MAPPO with 12-action / 14-dim observation space; 11 configs, 5 seeds × 10k–30k episodes
- Honest behavior ≥99.7% in 8/9 single-attack configs
- Deterministic defenses: 100% block rate
- Probabilistic defense (A9, p=0.80): 81.5% empirical
- Economic deterrence: 52–58% block
- C11 (combined attacks): 20.0%±40.0% honest, 48% hold rate across 164,345 attacks
- Ablation: stake removal drops honest% below 90%
- Sensitivity: robust up to adversarial bonus b=1.0 (2× baseline)

---

## 2. Repository Layout

```
marl-reputation/
├── env/
│   ├── __init__.py
│   ├── reputation_engine.py      # Python port of Go chaincode reputation logic
│   └── reputation_marl_env.py   # PettingZoo AEC environment
├── agents/
│   ├── __init__.py
│   ├── mappo.py                  # Shared-policy MAPPO
│   └── independent_ppo.py       # Per-agent IndependentPPO
├── training/
│   ├── train.py                  # Main training loop
│   ├── train_ablation.py         # Ablation variant (YAML config overrides)
│   ├── train_ippo.py             # IPPO training
│   └── train_resume.py           # Resume from checkpoint
├── evaluation/
│   ├── evaluate.py               # Summary evaluation + Wilcoxon test
│   ├── generate_figures.py       # 8 matplotlib figures (300 dpi)
│   └── generate_tables.py        # 4 LaTeX tables
├── configs/
│   ├── config{1..11}.yaml        # 11 main experimental configs
│   ├── config{2,11}_terminal.yaml # Terminal reward variants
│   ├── ablation_*.yaml           # 4 ablation configs
│   └── sweeps/                   # 7 bonus sweep configs
├── scripts/                      # Utility and launch scripts
├── logs/                         # Per-run stdout logs (29 files)
├── results/
│   ├── training_logs/            # Per-config *_summary.json + per-seed JSON
│   ├── ablation/                 # Ablation summaries + spearman_results.json
│   ├── sensitivity/              # Sweep summaries
│   ├── ippo/                     # IndependentPPO summaries
│   ├── terminal/                 # Terminal reward summaries
│   ├── analysis/                 # config11_decomposition.txt
│   ├── tables/                   # Generated LaTeX tables
│   ├── figures/                  # Generated figures (PNG + PDF)
│   ├── checkpoints/              # final.pt + ep*.pt per config/seed
│   ├── evaluation_summary.json   # Breaking-point + Wilcoxon results
│   └── training_logs_v1_backup/  # Earlier 3-seed run (all 11 configs)
└── marl-paper/
    ├── main.tex + section .tex files
    ├── bibliography.bib
    ├── tables/                   # Mirror of results/tables/
    ├── figures/                  # Mirror of results/figures/
    └── Computer_Networks__Raddad2026_/ # Elsevier parallel draft
```

---

## 3. Environment Layer (`env/`)

### `reputation_engine.py`

A faithful Python port of the Go chaincode reputation logic from `am-unified`.

**`SystemConfig` (dataclass defaults):**

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `initial_alpha` | 2.0 | Beta prior α₀ |
| `initial_beta` | 2.0 | Beta prior β₀ |
| `decay_rate` | 0.98 | λ per decay period |
| `decay_period` | 86400.0 s | 1 day |
| `min_stake_required` | 10,000 | Tokens to rate |
| `slash_percentage` | 0.10 | Fraction slashed on upheld dispute |
| `dispute_cost` | 100.0 | Tokens to file dispute |
| `min_rater_weight` | 0.5 | Lower Wilson CI bound weight |
| `max_rater_weight` | 2.0 | Upper Wilson CI bound weight |
| dimensions | quality/delivery/compliance/warranty/sustainability | 5 dims |

**`ReputationState`:** `(alpha, beta, last_update)`

**`ReputationEngine` methods:**
- `apply_decay(rep, now)` — pulls α,β toward (α₀,β₀) via `α_eff = α₀ + (α−α₀)·λ^(Δt/T)`
- `apply_rating(rep, outcome, weight)` — Bayesian update: α+=weight if outcome>0, β+=weight otherwise
- `wilson_ci(alpha, beta, z=1.96)` — 95% Wilson interval on n=α+β−2α₀ observations
- `can_rate(stake)` — enforces min stake
- `slash(stake, rate)` — slash by `slash_percentage`
- `deduct_dispute_cost(stake)` — subtract dispute_cost

### `reputation_marl_env.py`

**Class:** `ReputationMARLEnv(AECEnv)` (PettingZoo AECEnv)

**Key constants:**
```python
DISHONEST_DETECT_PROB = 0.85   # P(detecting dishonest rating)
TAMPER_DETECT_PROB    = 0.80   # P(catching evidence tamper, action 9)
MAX_STEPS             = 50     # Steps per episode (overridable to 100 via config)
SECONDS_PER_STEP      = 3600   # 1 hour per step for decay computation
```

**Observation space (14-dim, clipped [−5, 5]):**

| Index | Feature | Description |
|-------|---------|-------------|
| 0 | μ_i | Own reputation score (α/(α+β)) |
| 1 | w_i | Own Wilson CI width |
| 2 | s_i | Own normalized stake |
| 3 | α_i | Own raw alpha |
| 4 | β_i | Own raw beta |
| 5 | μ_j | Target agent score |
| 6 | w_j | Target Wilson CI width |
| 7 | r_i | Own cumulative reward (normalized) |
| 8 | d_i | Disputes filed against agent i |
| 9 | d_lost_i | Disputes lost (stake slashed) |
| 10 | t | Normalized time step |
| 11 | n_sybil_i | Sybil identities created by i |
| 12 | B_norm | Normalized block count |
| 13 | g_i | Gate check (0/1) |

**Action space (12 discrete actions):**

| Action | Name | Description |
|--------|------|-------------|
| A0 | Noop | Do nothing |
| A1 | Honest positive | Rate target honestly positive |
| A2 | Honest negative | Rate target honestly negative |
| A3 | Dishonest inflate | False positive rating |
| A4 | Dishonest deflate | False negative rating |
| A5 | File dispute | Dispute a received rating |
| A6 | Deposit stake | Add stake tokens |
| A7 | Self-rate | Rate self (penalty −5 if caught, p=0.85) |
| A8 | Admin escalation | Unauthorized admin action (penalty −5) |
| A9 | Evidence tamper | Tamper with provenance (−2 if caught p=0.80, +0.3 if uncaught) |
| A10 | Gate bypass | Circumvent reputation gate (penalty −3) |
| A11 | Provenance replay | Replay old event (penalty −5) |

**Reward structure:**

| Event | Reward |
|-------|--------|
| Score gain after honest rating | +1.0 |
| Score loss after honest rating | −2.0 |
| Stake lost (slash) | −0.5 |
| Dispute won | +0.5 |
| Sybil creation cost | −0.1 |
| Alignment bonus (honest behav.) | +0.5 |
| Defense dividend (per blocked attack) | +0.1 |
| Accuracy bonus | +0.05 |
| Self-rate caught (A7) | −5.0 |
| Admin escalation (A8) | −5.0 |
| Tamper caught (A9, p=0.80) | −4.0 |
| Tamper uncaught (A9) | +0.3 |
| Gate bypass (A10) | −3.0 |
| Provenance replay (A11) | −5.0 |
| Dishonest caught (A3/A4) | −3.0 |

**`get_metrics()` output keys:**
`mean_score, mean_true_quality, mean_reputation_accuracy, mean_ci_width, total_sybils, total_disputes_filed, total_ratings, attacks_attempted, attacks_blocked, defense_hold_rate, defense_by_type{7..11}, stakes{}, agent_scores[], agent_true_qualities[]`

**Terminal reward (optional):** `R_term = terminal_reward_coef × (μ_final − 0.5)` — added at episode end when `terminal_reward_coef > 0`.

---

## 4. Agent Layer (`agents/`)

### `mappo.py` — Shared-Policy MAPPO

**Network (`SharedActorCritic`):**
```
Input(14) → Linear(256) → ReLU → Linear(128) → ReLU
                                              ├─→ Linear(12)   # actor logits
                                              └─→ Linear(1)    # critic value
```
All agents share one network — parameter sharing enforces a unified policy.

**`RolloutBuffer`:** Per-agent ring buffer.

**`MAPPOTrainer` hyperparameters:**

| Param | Value |
|-------|-------|
| Learning rate | 3e-4 |
| γ (discount) | 0.99 |
| λ (GAE) | 0.95 |
| ε (PPO clip) | 0.2 |
| Entropy coef | 0.05 |
| Value fn coef | 0.5 |
| Max grad norm | 0.5 |
| n_epochs | 4 |
| Minibatch size | 64 |

### `independent_ppo.py` — Per-Agent IPPO

Same architecture but each agent has its **own separate network**. Entropy coef lowered to 0.01 vs MAPPO's 0.05. Used to benchmark against MAPPO.

---

## 5. Training Layer (`training/`)

### `train.py` (main loop)

- `OMP_NUM_THREADS=MKL_NUM_THREADS=1` (PyTorch single-thread for reproducibility)
- **Convergence criterion:** variance of honest% < 0.05 over rolling window of 100 episodes, after minimum 5,000 episodes
- `max_ep_ext = 10,000` (standard), up to `20,000` (extended)
- **Wall budget:** 10 hours (36,000 s)
- **Eval:** 100 deterministic evaluation episodes at end of training
- **Output files per seed:** `{cfg}_seed{i}.json`, `{cfg}_seed{i}_log.json`; aggregated: `{cfg}_summary.json`

### `train_ablation.py`

- Accepts YAML with `system_config` overrides: `decay_rate, min_stake_required, dispute_cost, min_rater_weight, max_rater_weight`
- Adds `tail_metrics` (last 10 episodes avg) to output

### `train_ippo.py`

Uses `IndependentPPOTrainer` instead of `MAPPOTrainer`.

### `train_resume.py`

- Loads `final.pt` checkpoint
- CLI flags: `--extra-episodes`, `--wall-hours`, `--rebuild-summary`

---

## 6. Evaluation Layer (`evaluation/`)

### `evaluate.py`
- Reads cfg1–5 summaries
- **Breaking-point analysis:** checks which config crosses honest% < 0.5 threshold
- **Wilcoxon test:** honest vs adversarial seed distributions

### `generate_figures.py`
Generates 8 figures at 300 dpi (serif font):
1. `fig_training_curves` — reward curves per config
2. `fig_honest_convergence` — honest% over training
3. `fig_reputation_accuracy` — Spearman ρ score vs true quality
4. `fig_sybil_stake` — stake depletion under Sybil attack
5. `fig_collusion_ci` — CI width behavior under collusion
6. `fig_breaking_point` — honest% vs adversarial bonus b
7. `fig_wilson_calibration` — CI width vs evidence count
8. `fig_ablation` — ablation bar chart

### `generate_tables.py`
Generates 4 LaTeX tables with hardcoded scripted attack latencies:

| Attack | Latency |
|--------|---------|
| Self-Rating | 12.6 ms |
| Sybil (5 nodes) | 368.6 ms |
| Collusion | 131.1 ms |
| SetReputationGate (unauth) | 9.0 ms |
| Insufficient Stake | 560.4 ms |
| Evidence Tampering | 65.2 ms |
| Gate Bypass | 571.1 ms |
| Provenance Replay | 570.6 ms |

Parameter sensitivity table over: `S_min ∈ {1k, 5k, 10k, 50k}` × `α_slash ∈ {0.05, 0.10, 0.20}`.

---

## 7. Configs (`configs/`)

### Main Configs (11)

All share: `n_agents=20, max_steps=100, episodes=5000, eval_episodes=20, seeds=5, checkpoint_every=250, max_episodes_extended=20000`

| Config | Label | Adv IDs | Bonus b | Special |
|--------|-------|---------|---------|---------|
| config1 | Baseline | [] | 0.0 | All-honest baseline |
| config2 | Mixed | [0..4] | 0.5 | 5 adversaries, mixed attacks |
| config3 | Sybil | [0,1,2] | 0.3 | max_sybils=5 |
| config4 | Collusion | [0..3] | 0.5 | collusion_groups=[[0,1,2,3]] |
| config5 | Adaptive | [0..4] | 1.0 | Breaking point / adaptive |
| config6 | Self-rate | [0..4] | 1.0 | enabled_attacks=[7] |
| config7 | Admin | [0,1,2] | 1.0 | enabled_attacks=[8] |
| config8 | Tamper | [0..4] | 1.0 | enabled_attacks=[9] |
| config9 | Gate bypass | [0..3] | 1.0 | enabled_attacks=[10] |
| config10 | Replay | [0,1,2] | 1.0 | enabled_attacks=[11] |
| config11 | Comprehensive | [0..9] | 1.0 | collusion_groups=[[0..3]], max_sybils=3, all attacks [7..11] |

### Terminal Reward Variants

| Config | Adv IDs | Bonus b | terminal_reward_coef | Seeds |
|--------|---------|---------|----------------------|-------|
| config2_terminal | [0..4] | 0.5 | 0.5 | 3 |
| config11_terminal | [0..9] | 1.0 | 0.5 | 3 |

### Ablation Configs (4)

| Config | What changes |
|--------|-------------|
| `ablation_no_decay.yaml` | `decay_rate=1.0` (no temporal decay) |
| `ablation_no_dispute.yaml` | `dispute_cost=1,000,000` (prohibitively expensive) |
| `ablation_no_stake.yaml` | `min_stake_required=1.0` (effectively disabled) |
| `ablation_no_wilson.yaml` | `min_rater_weight=max_rater_weight=1.0` (flat weights) |

All ablations: `seeds=3`.

### Sensitivity Sweeps (`configs/sweeps/`)

7 configs varying adversarial reward bonus:
`sweep_bonus_{0p0, 0p1, 0p25, 0p5, 1p0, 2p0, 4p0}.yaml` — `seeds=2–3`

---

## 8. Scripts (`scripts/`)

| Script | Purpose |
|--------|---------|
| `check_ckpts.py` | Inspect checkpoint files |
| `check_progress.py` | Monitor in-progress training |
| `copy_figures.py` | Copy PNG/PDF from `results/figures/` → `marl-paper/figures/` |
| `eval_ablation_spearman.py` | 20 det. episodes × 3 seeds, Spearman ρ(score, true_quality); writes `results/ablation/spearman_results.json` |
| `post_training_update.py` | Regenerate tab_ablation, fig_ablation, fig_parameter_sensitivity, fig_mappo_vs_ippo |
| `run_post_training.sh` | Wait for 11 summaries, invoke eval pipeline |
| `run_extended.sh` | Resume unconverged seeds: c2-s2/4, c5-s0/1/3, c10-s4, c11-s0/3/4 |
| `run_items_1_2_6.sh` | Parallel launcher: 6 bonus sweeps × 3 seeds + IPPO c2/c11 + terminal c2/c11 |
| `task3_c11.py` | C11 action decomposition → `results/analysis/config11_decomposition.txt` |
| `task4_sensitivity.py` | Generate sweep configs, run, write CSV |
| `task6_stats.py` | Generate `tab_statistical.tex` |
| `task7_figures.py` | Generate defense_heatmap, ablation, c11_action, param_sensitivity, mappo_vs_ippo |
| `task8b_update_summary.py` | Rebuild 5-seed summaries for c2/c11 |
| `task9_ablation_table.py` | Generate `tab_ablation.tex` |
| `watch_and_finalize.sh` | Poll sens_1p0 completion; auto-commit results |

**Execution log (`logs/items_1_2_6.log`):** Launched Mar 17 2026 00:30 MDT, finished 18:49 MDT (18h 19m), all 10 jobs succeeded.

---

## 9. Results — Main Training (11 Configs)

All in `results/training_logs/`. 5 seeds each, extended where needed.

| Config | Conv | Honest% (mean±std) | Eval Reward | Rep Accuracy | Notes |
|--------|------|---------------------|-------------|--------------|-------|
| C1 Baseline | 5/5 | 0.600±0.490 | 0.015±0.030 | 0.364±0.168 | Seeds with 0 ratings → honest=0 by convention |
| C2 Mixed | 4/5 | 0.736±0.388 | — | 0.382 | Seed 2 anomaly (0 honest) |
| C3 Sybil | 5/5 | 0.800±0.400 | −0.548±0.548 | 0.339 | defense_by_type[10]=3 in seed 3 |
| C4 Collusion | 5/5 | 1.000±0.000 | 0.027±0.088 | 0.232 | Agents learn collusion is ineffective |
| C5 Adaptive | 3/5 after ext. | 0.9999±9e-5 | 0.289±0.075 | 0.192 | Breaking point config |
| C6 Self-rate | 5/5 | 0.400±0.490 | — | 0.389 | Bimodal: some seeds 0%, some 100% |
| C7 Admin | 5/5 | 0.200±0.400 | — | 0.260 | Bimodal pattern |
| C8 Tamper | 5/5 | 0.400±0.490 | — | — | A9 probabilistic defense |
| C9 Gate bypass | 5/5 | 0.600±0.490 | — | — | |
| C10 Replay | 4/5 | 1.000±0.000 | — | — | Seeds 0/2/3: defense_by_type[11]=3 |
| C11 Comprehensive | 5/5 | 0.200±0.400 | — | 0.251±0.103 | 4/5 adversarial-dominant; 40 sybils created; 164,345 attacks; 48% hold rate |

**C11 per-seed detail:**
- Seeds 0/1/2/4: honest=0.0 (adversarial equilibrium; agents bankrupt: stakes=0)
- Seed 3: honest=1.0 (15 attacks, 4 blocked: 2 gate_bypass + 2 replay)
- This bimodality is path-dependent and exposed only after extended training; v1 showed honest=1.0 at ep≈2000

---

## 10. Results — Ablations

`results/ablation/training_logs/` (3 seeds each). Baseline comparison: C2 config with full system.

| Variant | Conv | Honest% | Defense Rate | Notes |
|---------|------|---------|--------------|-------|
| Full system | 4/5 | 99.7% | — | Reference |
| No stake | 0/3 | 87.8% | 0.434 | −12pp; biggest drop |
| No Wilson | 0/3 | 95.7% | 0.527 | −4pp |
| No decay | 1/3 | 96.7% | 0.527 | −3pp; but best Spearman ρ |
| No dispute | 0/3 | 94.7% | 0.434 | −5pp |

**Spearman ρ (score vs true quality) — `results/ablation/spearman_results.json`:**

| Variant | ρ (mean±std) |
|---------|-------------|
| Full system | −0.366±0.675 |
| No stake | +0.573±0.327 |
| No Wilson | −0.302±0.843 |
| No decay | **+0.653±0.336** (best) |
| No dispute | +0.573±0.327 |

Key insight: removing decay gives best score/quality correlation — decay trades recency-weighting for discrimination power. This is an important design trade-off.

**Note:** `no_dispute` and `no_stake` metrics are identical — likely a logging bug where the wrong summary was written for one variant.

---

## 11. Results — Sensitivity Sweeps (Adversarial Reward Bonus)

`results/sensitivity/training_logs/`

| Bonus b | Seeds | Conv | Honest% | Eval Reward | Rep Accuracy |
|---------|-------|------|---------|-------------|--------------|
| 0.0 | 2 | 0/2 | 0.9995 | −1.430±0.712 | 0.305 |
| 0.1 | 3 | 1/3 | 0.9992±0.001 | −0.004±0.032 | 0.434 |
| 0.25 | 3 | 0/3 | 0.9933±0.006 | −0.952±1.234 | 0.393±0.252 |
| 0.5 | 3 | 0/3 | 0.9964±0.002 | −0.362±0.244 | 0.403±0.256 |
| 1.0 | 3 | 1/3 | 0.99996±3e-5 | +0.315±0.122 | 0.086±0.068 |
| 2.0 | 3 | 1/3 | 0.9974±0.004 | −0.671±0.664 | 0.285±0.144 |
| 4.0 | 3 | 2/3 | 1.000±0.000 | +0.185±0.521 | 0.344±0.090 |

System is robust across all tested bonus values. Highest accuracy at b=0.1 (0.434). Lowest accuracy at b=1.0 (0.086) — higher adversarial pressure drives agents away from informative rating behavior.

`sweep_bonus_0p5` seed 0: agents 2 and 4 slashed once (stake=18,000), attacks_attempted=6, blocked=2 (type 9 — gate bypass).

---

## 12. Results — IPPO & Terminal Variants

### Independent PPO (`results/ippo/`, 3 seeds)

| Config | Conv | Honest% | Eval Reward | Rep Accuracy |
|--------|------|---------|-------------|--------------|
| C2 mixed | 0/3 | 1.0±0.0 | 0.121±0.067 | 0.245±0.069 |
| C11 comprehensive | 0/3 | 1.0±0.0 | 0.202±0.021 | 0.300±0.075 |

IPPO achieves honest behavior but does **not converge** by the variance criterion, and reaches the 95% honest threshold ~1.4–1.9× slower than MAPPO.

### Terminal Reward (`results/terminal/`, 3 seeds)

| Config | Conv | Honest% | Eval Reward | Rep Accuracy |
|--------|------|---------|-------------|--------------|
| C2 terminal | 3/3 | 0.650±0.460 | −9.31±9.97 | 0.371±0.200 |
| C11 terminal | 2/3 | **1.000±0.000** | −1.26±2.23 | 0.251±0.155 |

**C11 terminal fully fixes the bimodality**: 2/3 seeds converge to honest=1.0 (vs 1/5 without). This is the paper's key design recommendation.

---

## 13. Results — Analysis & Decomposition

### `results/analysis/config11_decomposition.txt`

Best seed (seed 0) tail reward: −0.3243

**Tail action distribution:**

| Action class | Fraction |
|-------------|---------|
| Honest (A1–A2) | 99.0% |
| Dishonest (A3–A4) | 0.5% |
| Attack (A7–A11) | 0.7% |

**Blocked by type (tail episodes):**

| Attack | Blocks |
|--------|--------|
| A7 Self-rate | 225 |
| A8 Admin | 1,394 |
| A9 Tamper | 307 |
| A10 Gate bypass | 697 |
| A11 Replay | 2,680 |

Interpretation: dishonest rating (A3/A4) is the preferred adversarial strategy in tail policy. Hard attacks (A7/A8/A11) are learned as unprofitable by most seeds, but some seeds (especially A8 at 1,394 blocks) still attempt them.

### `results/evaluation_summary.json`

- Breaking point: `"config5_adaptive: system holds (honest_pct >= 0.5)"` — no breaking point found up to b=4.0
- Wilcoxon p=0.0625 (not significant at α=0.05)
- Key finding: `"Honest behavior is the dominant strategy."`

---

## 14. LaTeX Tables

All in `results/tables/` (mirrored in `marl-paper/tables/`):

### `tab_summary.tex`
5-config summary (C1–C5): mean_reward, honest%, per config.

### `tab_attack_comparison.tex`
Scripted (am-unified) vs MARL outcomes comparison. Mirrors latencies from `generate_tables.py`.

### `tab_ablation.tex`
| System | Honest% | Δ vs full |
|--------|---------|-----------|
| Full | 99.7% | — |
| No Stake | 87.8% | −12.0pp |
| No Wilson | 95.7% | −4.0pp |
| No Decay | 96.7% | −3.0pp |
| No Dispute | 94.7% | −5.0pp |

### `tab_parameter_sensitivity.tex`
Analytic simulation over S_min × α_slash grid. All cells: honest=1.00 (system is insensitive to these parameters analytically).

### `tab_statistical.tex`
One-sample t-test vs 95% honest threshold:

| Config | t | p | Significant? |
|--------|---|---|-------------|
| C1 | 105.4 | <0.001 | Yes |
| C2 | 45.0 | <0.001 | Yes |
| C3 Sybil | −2.59 | 0.970 | **No** |
| C4 | 51.5 | <0.001 | Yes |
| C5 | 75.8 | <0.001 | Yes |
| C6 Self-rate | −0.28 | 0.602 | **No** |
| C7 | 83.3 | <0.001 | Yes |
| C8 | 72.8 | <0.001 | Yes |
| C9 | 41.8 | <0.001 | Yes |
| C10 | 67.8 | <0.001 | Yes |
| C11 | −1.00 | 0.813 | **No** |

Three configs (C3, C6, C11) do NOT statistically exceed 95% honest — these are the paper's main challenge cases.

---

## 15. Training Logs (`logs/`)

29 log files total:

| Log | Content |
|-----|---------|
| `items_1_2_6.log` | Master parallel run log: Mar 17 00:30–18:49 MDT, 10/10 jobs OK |
| `figures.log` | Figure generation: defense_heatmap → ablation → c11_decomp → param_sensitivity → mappo_vs_ippo → "All figures generated!" |
| `train_config{1..11}.log` | Per-config training stdout |
| `resume_c2_s{2,4}.log` | Resume logs for unconverged seeds |
| `resume_c5_s{0,1,3}.log` | C5 resume logs |
| `resume_c10_s4.log` | C10 seed 4 resume |
| `resume_c11_s{0,3,4}.log` | C11 resume logs |
| `sens_bonus_{*}.log` | Sensitivity sweep logs |
| `ippo_{config2,config11}.log` | IPPO training logs |
| `terminal_{config2,config11}_terminal.log` | Terminal reward logs |
| `watch_finalize.log`, `extended_training.log` | Orchestration logs |

---

## 16. V1 Backup (`results/training_logs_v1_backup/`)

Earlier run: 3 seeds, ~2000 episodes per config.

**Key reproducibility finding:** In v1, C11 converged honest=1.0 at ep≈2000. In the full run with 5 seeds and up to 30k episodes, the bimodal 20.0±40.0% behavior emerged. This confirms that **short training runs mask the bimodality** — extended training is necessary to reveal C11's true convergence landscape.

V1 config1_baseline: conv 3/3, honest 0.667±0.471 (same seed=2 anomaly pattern), total_time=3,998s.

---

## 17. Paper (`marl-paper/`)

### Main Files

**`main.tex`** — IEEE journal class. Title: "Adversarial Evaluation of a Blockchain-Based Bayesian Reputation System Using Multi-Agent Reinforcement Learning: Architecture, Defense Mechanisms, and Empirical Analysis". Target: IEEE TDSC 2026.

**Acknowledgments:** NSF OIA-2417062 DREAM Center; UNM I² Lab.

### Sections

**`abstract.tex`** — Key numbers: MAPPO 12-action/14-dim; 11 configs, 5 seeds × 10k ep; honest ≥99.7% in 8/9 single-attack configs; deterministic 100%; probabilistic 81.5%; economic 52–58%; C11: 20±40% honest, 48% hold rate, 164,345 attacks; ablation: stake removal → <90%; sensitivity: robust to b≤1.0.

**`introduction.tex`** — Research questions:
- RQ1: Does MAPPO converge under adversarial agents?
- RQ2: How do defense classes profile across attack types?
- RQ3: What is the population breaking point?
- Contributions: PettingZoo env, MAPPO harness, defense characterization, C11 bimodality finding.

**`system_model.tex`** — Beta scoring (Eq 1), Wilson CI (Eq 2), decay toward prior λ=0.98 (Eq 3), stake slashing p_d=0.85, 5 attack defenses. Threat model: white-box, b∈[0.5, 2.0].

**`marl_framework.tex`** — Full MDP formulation. N=20, T=100, obs (Eq 6), actions (A0..A11), reward (Eqs 7–9):
- R_base: honest +0.1 / dishonest-caught −0.5 / dishonest-uncaught +b
- R_defense: −5 (A7/A8/A11), −2·𝟙(detected, p=0.80) (A9), −3 (A10)
- R_dividend: +0.1 per blocked attack to all honest agents
- Network: W1∈ℝ^(256×14), W2∈ℝ^(128×256)
- GAE: γ=0.99, λ=0.95, ε=0.2, c_v=0.5, c_e=0.05, LR: 3e-4→3e-5 linear

**`evaluation.tex`** — Full narrative. Per-seed C2: {68%, 100%, 0%, 100%, ~100%}. Per-seed C11: {0, 0, 0, 100%, 0}. Terminal reward: C2 3/3 conv 65±46%; C11 2/3 conv 100±0%. MAPPO vs IPPO: 1.9× faster on C2, 1.4× on C11.

**`discussion.tex`** — Three defense classes (deterministic / probabilistic / economic); learned avoidance ≠ intrinsic robustness; noop equilibrium distinction; C11 bimodality as path dependence; terminal reward as design recommendation; methodology complements scripted testing.

**Limitations:** simulation fidelity, fixed adversarial population, limited obs space.

**`conclusion.tex`** — Three findings + C11 open challenge + terminal reward recommendation. Future work: curriculum learning, enriched obs, heterogeneous policies, cooperative majority, dynamic populations.

**`related_work.tex`** — 26 bibliography entries including: almaayn2024unified (TDSC), almaayn2026provenance (CCNC), almaayn2026reputation_ccwc (CCWC), Yu 2021 (MAPPO), Terry 2021 (PettingZoo), Lowe 2017 (MADDPG), Foerster 2018 (DIAL), Douceur 2002 (Sybil), Wilson 1927.

### `Computer_Networks__Raddad2026_/`

Parallel Elsevier draft with same section structure. Uses `ecrc.sty`, `elsarticle.cls`, numerical bibliography. Contains extra figures not in IEEE draft: `cdfCD, covNN, CRAN, E2E*, Emulator, FF, FHdelay`.

---

## 18. Figures

17 figures in both `results/figures/` and `marl-paper/figures/` (PNG + PDF):

| Figure | Description |
|--------|-------------|
| `fig_training_curves` | Reward curves per config |
| `fig_honest_convergence` | Honest% vs episode |
| `fig_reputation_accuracy` | Spearman ρ by variant |
| `fig_sybil_stake` | Stake depletion under Sybil |
| `fig_collusion_ci` | CI width under collusion |
| `fig_breaking_point` | Honest% vs bonus b |
| `fig_wilson_calibration` | CI width vs evidence count |
| `fig_defense_heatmap` | Block rates per attack type per config |
| `fig_ablation` | Ablation bar chart (honest%) |
| `fig_c11_action_decomposition` | Tail action distribution C11 |
| `fig_parameter_sensitivity` | Sensitivity over S_min × α_slash |
| `fig_mappo_vs_ippo` | Convergence speed comparison |

---

## 19. Key Numerical Takeaways

| Metric | Value |
|--------|-------|
| Deterministic defenses (A7/A8/A10/A11) | **100% block rate** |
| Probabilistic defense (A9, p=0.80) | **81.5% empirical** (vs 80% theoretical) |
| Economic deterrence block rate | **52–58%** |
| E[R_tamper] uncaught path | +0.3; caught path −2; net E = −1.4 |
| C11 honest% (5 seeds) | **20.0±40.0%** |
| C11 total attacks | **164,345** |
| C11 hold rate | **48%** |
| Terminal reward fix C11 | **2/3 honest=1.0** (vs 1/5 without) |
| Stake removal impact | **−12pp honest%** (biggest ablation drop) |
| MAPPO vs IPPO convergence speed | **1.9× faster** (C2), **1.4×** (C11) |
| Best Spearman ρ | no_decay: **+0.653** |
| Wilcoxon p | **0.0625** (not significant at α=0.05) |
| Statistical sig configs | **8/11** exceed 95% honest threshold |
| Non-significant configs | C3 (Sybil), C6 (Self-rate), C11 |
| v1 → final training duration | ~2,000 ep → up to 30,000 ep (bimodality exposed) |

---

## 20. Observed Weaknesses & Open Questions

These are **not value judgements** — they are objective observations from the code and results that would likely form the basis of an improvement plan:

### Environment & Simulation

1. **Fixed adversarial population:** Adversary IDs are hardcoded in config. No dynamic agent-type switching or curriculum. Adversaries know the env and behave consistently — no adaptive policy updates mid-training.
2. **14-dim observation is sparse:** Agent can see only one target at a time (`μ_j, w_j`). No global view of network reputation distribution. This limits strategic depth.
3. **No communication channel:** No explicit message passing between agents. Coordination (honest or adversarial) is purely implicit via shared environment state.
4. **`SECONDS_PER_STEP=3600` is arbitrary:** The decay computation uses this as wall-clock time; it may not match realistic AM workflow timescales.
5. **Metrics logging bug:** `no_dispute` and `no_stake` ablation summaries have identical numerical results — one likely overwrote the other.

### Training

6. **Convergence criterion (variance < 0.05) is coarse:** Some seeds that appear converged (low variance) may still be in transient behavior — C11 is the clearest example.
7. **No curriculum:** All attack types are available from episode 1. Curriculum (start easy, add attacks) might help agents learn stable defense before exposure to combined attacks.
8. **Single shared policy (MAPPO):** Honest and adversarial agents share one policy network. While this is a valid and common simplification, it means the network must represent both honest and adversarial behaviors simultaneously — potentially creating representational interference.
9. **Learning rate schedule:** Linear decay 3e-4 → 3e-5. No warmup. Adam optimizer parameters not documented.

### Results Reliability

10. **C3, C6, C11 are not statistically significant** at the 95% honest threshold. The paper's main claim ("honest behavior is the dominant strategy") does not hold for these three configs at α=0.05.
11. **Bimodal outcomes (C2, C6, C7, C11):** Seeds with 0 honest ratings are categorized as honest=0 by convention, distorting mean±std. The distribution is bimodal, not Gaussian — mean±std is an inappropriate summary statistic for these configs.
12. **Wilcoxon p=0.0625:** Not significant. The honest vs adversarial comparison lacks statistical power, likely due to small number of seeds (5) and bimodal distributions.
13. **v1 reproducibility gap:** C11 showed honest=1.0 at 2,000 episodes in v1 but bimodal behavior in the full run. This undermines confidence that longer training is always better — the result depends on random seed and training duration.

### Defense Mechanisms

14. **Dishonest ratings (A3/A4) are the dominant adversarial strategy in C11 tail** (0.5% of actions vs 0.7% hard attacks), yet they are not blocked by deterministic defenses — only by probabilistic detection (p=0.85). This is the main unresolved attack surface.
15. **Economic deterrence (52–58%)** is the weakest defense class. Stake parameters (S_min, α_slash) are analytically swept but the sensitivity table shows all-ones (honest=1.00) — the analytic simulation may be oversimplified.
16. **Defense dividend (+0.1 per blocked attack to all honest)** creates a collective reward that may not be sufficient to incentivize proactive defense — it's a passive byproduct, not an active signal.

### Paper / Code Coherence

17. **`generate_tables.py` hardcodes scripted attack latencies** from the am-unified paper. These should be imported from a shared data file, not duplicated as magic numbers.
18. **Elsevier parallel draft (`Computer_Networks__Raddad2026_/`)** contains extra figures not referenced in the IEEE draft — unclear which is canonical.
19. **No unit tests** for `reputation_engine.py` — the Python port of the Go chaincode is not validated against the canonical Go implementation.
20. **`watch_and_finalize.sh` auto-commits results** — git automation in a research pipeline is risky for reproducibility.

---

*Report generated: April 2026. All data from `/home/raddad/marl-reputation/` as of this date.*
