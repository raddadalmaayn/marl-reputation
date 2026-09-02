# Tier 1B Full Sweep — PRE-FLIGHT GATE

All three gate checks **PASS** → the full sweep is cleared to launch.

## Corrections in place
- **A. ρ logging ON for every shard.** `training/train.py` calls
  `collect_basin_report(..., with_rho=True)`; `system_spearman_rho` (system score vs
  true latent quality) is written to every per-seed summary, both arms, every config.
- **B. Baselines folded in (apples-to-apples).** `evaluation/baseline_eval.py` loads
  the main-sweep **Beta-trained** checkpoints and evaluates the SAME learned policy,
  re-scoring the identical eval rating-stream under {beta, flat, eigentrust}. Flat is
  online/in-loop (per-event replay); EigenTrust is post-hoc/global. Adversary behavior
  and attack/hold stats are identical across estimators; only ρ differs. (Opt-in env
  flag `record_rating_events`, default off, exposes the stream — behavior unchanged.)

## Check 1 — resolved Arm-B shard config
`configs/tier1b/sweep_c2_armB.yaml` resolved through `train.build_system_config`:
- `participation_coef == 1.277` ✓ (FIXED from Stage 0; not re-tuned)
- `stake_obs_mode == "absolute_log"` ✓
- rater clamp `== [0.1, 5.0]` ✓
- `train.py` passes `with_rho=True` ✓

**CHECK 1: PASS**

## Check 2 — ρ logged, non-NaN, both arms (C2, seed 0, 400 ep)
| arm | ρ present | ρ value | participation | basin |
|---|---|---|---|---|
| A | yes | **0.9335** | 0.873 | honest-dominant |
| B | yes | **0.9397** | 1.000 | honest-dominant |

ρ is present and non-NaN for both arms. **CHECK 2: PASS**
(Note: at 400 ep C2 ρ≈0.93; the pilot saw ρ degrade by 1000 ep — ρ is non-monotone in
training length. The sweep's longer schedule will settle this; flagged for analysis.)

## Check 3 — baseline eval, 3 engines readable ρ (C2 armA checkpoint)
| engine | ρ | readable |
|---|---|---|
| beta | 0.9356 | ✓ |
| flat | 0.9398 | ✓ |
| eigentrust | 0.6712 | ✓ |

All three estimators produce a ρ the aggregator can read. **CHECK 3: PASS**
(hold_rate was NaN here because the converged C2 policy attempted no attacks in
deterministic eval; hold rate is estimator-independent and is reported from the
training-tail for configs that do attack — C3/C11.)

## Verdict: **ALL CHECKS PASS — cleared to launch the full sweep.**
