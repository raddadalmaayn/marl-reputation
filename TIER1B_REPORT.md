# Tier 1B (plan b) Report — Scaled Two-Arm Adversarial Evaluation on the Corrected Env

**Scope:** controlled before/after on the Tier 1A corrected environment
(`stake_obs_mode="absolute_log"`, rater clamp `[0.1, 5.0]` — both confirmed live).
Arm A = no participation incentive (legacy-exact, `participation_coef=0`); Arm B =
the slash-grounded participation incentive `c` derived in Stage 0. Prior artifacts
under `results/**`, `results/derived/**`, `results/tier1a/**` are untouched; all new
outputs are under `results/tier1b/`. No git commit.

## Live-environment confirmation (READ FIRST)
- `stake_obs_mode` default = **`absolute_log`**, `STAKE_OBS_SCALE=50000` — confirmed.
- rater-weight clamp default = **`[0.1, 5.0]`** (Go-matching) — confirmed.
- `min_stake=10000`, reset `=20000`, `slash=0.10`, `dispute_cost=100`,
  `DISHONEST_DETECT_PROB=0.85`, `TAMPER_DETECT_PROB=0.80`, `terminal_reward_coef`
  hook present — confirmed from live constants.

## Extracted reward function (the basis for Stage 0; NOT the paper's Eq. 6/7)
Per `_execute_action` / `_execute_rating`, for an agent eligible to rate, relative
to noop (EV=0):

| Action class | reward terms (as implemented) | immediate EV |
|---|---|---|
| noop (A0) | — | 0 |
| honest rating (A1,A2) | rating updates the **target**'s Beta state; rater's own score unchanged; never slashed | **0** |
| dishonest rating (A3,A4) | submitted; detected w.p. `0.85` → slash + `R_DISHONEST_CAUGHT=-3.0`; else 0; plus ~`-0.005` stake-loss side-cost on slash | **≈ -2.554** |
| attacks (A7–A11) | always/probabilistically blocked → `-3…-5` + slash | strongly negative |

**Key code-vs-paper divergence (confirmed):** the adversarial bonus is
`b·max(0, own_score_delta)` — keyed on the rater's *own* reputation gain, which a
rating action does not produce in-step. So **`b` cancels** in the rate/noop and
honest/dishonest comparisons and **does not enter `E[dishonest]`** — unlike the
paper's Eq.(6)/(7). The participation incentive is therefore designed against the
*actual* economics.

---

## STAGE 0 — Participation-incentive derivation & GATE

Computed by `evaluation/participation_economics.py` →
`results/tier1b/participation_economics.{json,md}`. Analytic EVs are **empirically
validated** by forced-action rollouts (4000 trials/action) — they match exactly:

| Action | analytic EV | empirical mean (Arm A) |
|---|---|---|
| noop | 0 | +0.0000 |
| honest (A1/A2) | 0 | +0.0000 |
| dishonest (A3/A4) | −2.554 | −2.557 ± 1.07 |

The incentive `c` is paid on the **observable** act of submitting a rating (actions
1–4 past the `can_rate`+dedup gates), independent of hidden honesty. Feasibility
(incentive compatibility):

- (P1) honest beats abstention: `E[honest] + c > 0` ⟹ **c > 0**
- (P2) cheating stays net-negative: `E[dishonest] + c < 0` ⟹ **c < 2.554**
- closed form: **`E[dishonest] < −c < E[honest]`**, i.e. **c ∈ (0, 2.554)**, and this
  interval is **independent of b** for all b ∈ [0.5, 2.0] (b cancels, shown above).

**Recommended c = 1.277** (midpoint of the feasible interval; simultaneously the
worst-case-b choice since the interval is b-invariant). Empirical check at c=1.277:
honest rating EV = **+1.277 (>0 ✓)**, dishonest rating EV = **−1.280 (<0 ✓)**, noop = 0.

### GATE DECISION: **PROCEED — Arm B enabled at c = 1.277.**
A feasible, b-robust, incentive-compatible `c` exists; plan (b) does NOT collapse to
plan (a). The derivation is itself a paper asset (an incentive-compatibility
statement) and is fully reproducible from the script.

---

## STAGE 1a — PILOT (5 seeds, 1000 episodes, {C1,C2,C7,C11} × both arms)

40 shards (`evaluation/run_shard.py`, 16-way parallel, single-thread each) →
`results/tier1b/pilot/`; aggregated by `evaluation/aggregate_arms.py`.

| Config | arm | particip. | cond honest | ρ (mean) | basins a/adv/hon/mix | P(hon) [Jeffreys95] | hold rate |
|---|---|---|---|---|---|---|---|
| C1 | A | 0.081 | 1.000 | −0.519 | 4/0/1/0 | 0.200 [0.023,0.629] | — (no adv) |
| C1 | B | **1.000** | 1.000 | 0.337 | 0/0/5/0 | **1.000** [0.621,1.000] | — |
| C2 | A | 0.843 | 1.000 | −0.471 | 0/0/5/0 | 1.000 | 0.182 |
| C2 | B | 1.000 | 1.000 | −0.367 | 0/0/5/0 | 1.000 | 0.899 |
| C7 | A | **0.000** | NaN | NaN | 5/0/0/0 | 0.000 [0.000,0.379] | 1.000 |
| C7 | B | **0.990** | 1.000 | 0.015 | 0/0/5/0 | **1.000** [0.621,1.000] | 1.000 |
| C11 | A | 0.599 | 1.000 | 0.077 | 1/0/4/0 | 0.800 [0.371,0.977] | 0.123 |
| C11 | B | 1.000 | 1.000 | 0.114 | 0/0/5/0 | 1.000 [0.621,1.000] | 0.267 |

### Pilot acceptance — ALL MET
1. **Arm A reproduces the Tier 0 abstention pattern on the corrected obs:** C7 Arm A
   is **pure abstention** (participation 0.000, 5/5 abstention basin); C1 Arm A is
   mostly abstention (participation 0.081, 4/5 abstention); C11 Arm A is bimodal
   (1 abstention / 4 honest-dominant). ✓
2. **Arm B measurably shifts participation upward on C1 and C7 (minimal proof,
   exceeded):** C1 Δparticipation **+0.919 [0.694, 1.144]**, C7 **+0.990 [0.963,
   1.018]** — both CIs exclude 0. C7 goes from *pure abstention* to *full
   participation, 5/5 honest-dominant*. ✓
3. **Basin labels self-report; ρ computed and logged per seed.** ✓

**PILOT PASSES** ⇒ Stage 1b is unblocked.

### Headline — Arm delta (B − A), with CIs

| Config | Δparticipation [95% CI] | ΔP(honest-dom) [Newcombe95] | Δρ [95% CI] |
|---|---|---|---|
| C1 | **+0.919 [0.694, 1.144]** | **+0.800 [0.193, 0.964]** | +0.856 (CI NaN: A had 4/5 ρ-undefined) |
| C2 | +0.157 [−0.267, 0.580] | 0.000 [−0.434, 0.434] | +0.104 [−0.903, 1.111] |
| C7 | **+0.990 [0.963, 1.018]** | **+1.000 [0.386, 1.000]** | NaN (A had 0 ratings) |
| C11 | +0.401 [−0.120, 0.922] | +0.200 [−0.264, 0.624] | +0.037 [−1.175, 1.249] |

**Two clean findings:**
- **Participation / basin: solved.** The derived incentive moves agents out of the
  abstention basin into **honest-dominant** at full participation. The effect is
  large and CI-significant where Arm A abstained (C1, C7); where Arm A already
  participated (C2 0.84, C11 0.60) it saturates participation to 1.0 and clears the
  remaining abstention/adversarial seeds.
- **Reputation discrimination (ρ): NOT solved.** Δρ CIs all straddle 0; Arm B mean ρ
  stays near-random or negative (C1 0.337, C2 **−0.367**, C7 0.015, C11 0.114) with
  enormous per-seed spread (e.g. C2 Arm B ρ = [−0.17, −0.83, +0.64, −0.87, −0.60]).
  Full honest participation does **not** make the Beta scores rank true quality.
  This separates the *participation* problem (fixed here) from the *accuracy* problem
  (Tier 0; structural — α0=β0 prior pull + decay + noisy binary outcome sampling),
  which persists. **The incentive is not a fix for discrimination.**

### Security cost — there is none (a benefit, if anything)
Absolute attack attempts (tail-pooled) and hold rate:

| Config | Arm A attempts → hold | Arm B attempts → hold |
|---|---|---|
| C2 | 8956 → 0.182 | **317** → 0.899 |
| C11 | 10628 → 0.123 | 8550 → 0.267 |
| C7 | 64 → 1.000 | 128 → 1.000 |

Arm B does **not** raise successful attacks or lower hold rate — adversaries find
honest rating (EV +1.277) more profitable than attacking (EV < 0), so attack
*attempts fall* (C2: 28× fewer) and hold rate rises. Participation did not re-open a
surface that abstention had masked.

---

## STAGE 1b — FULL TWO-ARM SWEEP (infrastructure delivered; distributed run pending)

Generated by `evaluation/make_sweep_manifest.py`:
- 22 sweep configs `configs/tier1b/sweep_<cN>_<arm>.yaml` (real 5000-ep + extension
  schedule; tiered seeds: 30 for {C1,C2,C3,C6,C7,C8,C9,C11}, 10 for {C4,C5,C10}).
- **540 shards** (270/arm), `results/tier1b/RUN_MANIFEST.tsv` (host,config,arm,seed,cmd).
- Per-host launch scripts `results/tier1b/launch/run_<host>.sh` and `RUN_PLAN.md`.
- Balanced map: strong host 216 shards, 4 desktops 81 each; **critical-path ≈ 20 h**
  at ~1.5 h/shard (assumption — re-estimate from the first wave; Arm B converges
  faster than Arm A).

**Status / data gap:** the full 540-shard sweep is a multi-host ~20 h job and is
**not run inline.** The pilot above (a 40-run, 1000-ep subset across the four key
configs and both arms) is the substantive empirical result delivered; it already
establishes the headline. The full sweep would tighten the CIs (30 seeds) and extend
to the remaining configs — it is ready to dispatch via the launch scripts and is
idempotent (`run_shard.py` skips finished seeds). Endpoints are computed by re-running
`aggregate_arms.py --dir results/tier1b/sweep`.

---

## STAGE 2 — EXTERNAL BASELINES (implemented + smoke-validated; full runs deferred)

`evaluation/baseline_engines.py`:
- **FlatAverageEngine** — true in-loop drop-in (unweighted Beta count; no rater-weight,
  no decay), selectable via the env flag `reputation_engine="flat"`.
- **EigenTrustScorer** — row-normalized global trust-flow; computed from the recorded
  rating graph (it is a *global* fixed point, so not a per-target Beta update — scored
  post-hoc, documented limitation).

**Smoke validation** (`smoke_validate`, honest rating stream, ρ vs true quality):
Beta ρ≈0.95, Flat ρ≈0.95–0.98, EigenTrust ρ≈0.72–0.92 — all finite and positive;
interface validated against the Beta engine. In-loop flag confirmed (beta default
preserved; flat selectable, produces distinct scores).

**Deferred (lowest priority; sweep pending):** the {C2,C3,C11} × both-arms in-loop
comparison. 6 configs + 60 dispatch commands are ready at
`results/tier1b/baselines/RUN_COMMANDS.sh`. EigenTrust-in-the-loop (driving per-step
obs) is left as future work; the post-hoc scorer + smoke validation stand in here.

---

## Files changed / added; flags (all flag-gated, legacy defaults intact)

| File | change | flag / default |
|---|---|---|
| `env/reputation_marl_env.py` | `participation_coef` (paid per submitted rating in `_execute_rating`); `reputation_engine` selector | `participation_coef=0.0` (Arm A, legacy-exact); `reputation_engine="beta"` |
| `env/reputation_engine.py` | (unchanged; FlatAverageEngine lives in evaluation) | — |
| `evaluation/basins.py` | `collect_basin_report(with_rho=False)` — additive ρ/MAE pooling | default off ⇒ Tier 1A behavior unchanged |
| `training/train.py` | wire `participation_coef` + `reputation_engine`; log them + ρ | defaults reproduce prior behavior |
| `evaluation/participation_economics.py` | **new** — Stage 0 derivation + gate | — |
| `evaluation/run_shard.py` | **new** — one (config,seed) shard (pilot + sweep unit) | — |
| `evaluation/aggregate_arms.py` | **new** — per-arm endpoints + arm-delta (Newcombe/Welch/Jeffreys) | — |
| `evaluation/make_sweep_manifest.py` | **new** — 540-shard manifest + RUN_PLAN + launch scripts | — |
| `evaluation/baseline_engines.py` | **new** — Flat + EigenTrust + smoke validation | — |
| `configs/tier1b/*.yaml` | **new** — 8 pilot + 22 sweep + 6 baseline configs | — |
| `RUN_PLAN.md`, `results/tier1b/RUN_MANIFEST.tsv`, `results/tier1b/launch/` | **new** — Stage 1b dispatch | — |

## Data gaps / caveats
1. **Full 540-shard sweep not run inline** (≈20 h distributed) — pilot (40 runs) is the
   delivered empirical result; sweep is dispatch-ready.
2. **ρ is undefined (NaN) for full-abstention seeds** (constant score 0.5) — correctly
   propagated; it shrinks the effective n for Δρ CIs (e.g. C1/C7 Arm A), so several Δρ
   CIs are NaN/wide. Real, not a bug.
3. **ρ per-seed variance is enormous** at 5 seeds; the near-random conclusion is robust
   (means hug 0, CIs straddle 0) but exact values need the 30-seed sweep.
4. **EigenTrust** is evaluated post-hoc, not in-loop; full in-loop baseline runs deferred.
5. Hold rate is pooled over the **training-log tail** (last 20%), not the last eval
   episode (which often has zero attacks under converged policies).

---

## Console output

```

########## pytest ##########
....................                                                     [100%]
20 passed in 0.11s

########## Stage 0 gate (participation_economics.py) ##########
  action 0: mean reward = +0.0000 (+/-0.000, n=4000)
  action 1: mean reward = +0.0000 (+/-0.000, n=4000)
  action 3: mean reward = -2.5573 (+/-1.070, n=4000)
  action 0: mean reward = +0.0000
  action 1: mean reward = +1.2771
  action 3: mean reward = -1.2801
GATE: PROCEED_ARM_B
Recommended c = 1.2771249999999998

########## Stage 2 baseline-engine smoke validation ##########
    "beta_rho": 0.9624060150375938,
    "flat_rho": 0.9593984962406013,
    "eigentrust_rho": 0.9218045112781955,
    "beta_rho": 0.9473684210526315,
    "flat_rho": 0.9518796992481202,
    "eigentrust_rho": 0.8842105263157894,
    "beta_rho": 0.9684210526315788,
    "flat_rho": 0.9774436090225563,
    "eigentrust_rho": 0.724812030075188,
All three engines produce finite scores and positive rho vs true quality on the honest stream — interface validated.

########## Pilot arm-delta headline ##########
## Arm delta (B − A) with CIs
| Config | Δparticipation [95% CI] | Δrho [95% CI] | ΔP(hon) [Newcombe95] | hold rate A→B |
|---|---|---|---|---|
| pilot_c1 | 0.919 [0.694,1.144] | 0.856 [NaN,NaN] | 0.800 [0.193,0.964] | NaN→NaN (NaN) |
| pilot_c11 | 0.401 [-0.120,0.922] | 0.037 [-1.175,1.249] | 0.200 [-0.264,0.624] | 0.123→0.267 (0.145) |
| pilot_c2 | 0.157 [-0.267,0.580] | 0.104 [-0.903,1.111] | 0.000 [-0.434,0.434] | 0.182→0.899 (0.717) |
| pilot_c7 | 0.990 [0.963,1.018] | NaN [NaN,NaN] | 1.000 [0.386,1.000] | 1.000→1.000 (0.000) |
```
