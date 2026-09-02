# Tier-0 Report — Correctness & Reporting Fixes

**Repo:** `marl-reputation` · **Scope:** recomputation/validation only — **no models were retrained.**
All behavioral numbers below come from deterministic evaluation (`dist.mode`, single-thread,
`OMP_NUM_THREADS=MKL_NUM_THREADS=1`) of the existing `results/checkpoints/**/final.pt`,
with fixed episode seeds (`--eval-base 10000`, 20 episodes/seed) so every number is
reproducible. Raw result JSONs were **not** modified; all derived outputs live under
`results/derived/`.

## How the metrics are computed (and what the logs actually contain)

`train.py`/`train_ablation.py` saved, per seed: `eval_mean_honest_pct`, `dishonest_actions`,
`total_actions`, and aggregate `final_env_metrics` — but **not** the per-action class counts
A0..A11, and **not** `agent_scores[]` / `agent_true_qualities[]` (the env's `get_metrics()`
returns those today, but the saved logs predate that and omit them). Per the task, the
per-action counts and per-agent score/quality arrays were therefore **collected by
deterministic eval from `final.pt`** (`evaluation/recompute_metrics.py`).

Definitions (identical to the task spec):
`rating = A1+A2+A3+A4`, `honest = A1+A2`, `total = ΣA0..A11`,
`participation = rating/total`, `honest_given_rating = honest/rating` (NaN if no rating),
`honest_unconditional = honest_given_rating if rating>0 else 0`.

Faithfulness note: action counts record the **policy-selected** action (matching how `train.py`
computes `honest_pct`). The env forces actions ≥7 to noop for non-adversarial agents; that
only ever touches non-rating buckets, so `participation` and `honest_given_rating` are
invariant to it.

---

## T0.1 — Reconciled three-metric table (all 11 configs + terminal + ablations)

Source: `results/derived/honest_metrics.{json,md}`.

| Config | seeds | rating-active | participation (mean) | **conditional** honest (mean / median) | **unconditional** honest (mean) | **gap** |
|---|---|---|---|---|---|---|
| config1_baseline | 5 | 3 | 0.248 | 1.000 / 1.000 | 0.600 | **0.400** |
| config2_mixed | 5 | 5 | 0.631 | 0.729 / 1.000 | 0.729 | 0.000 |
| config3_sybil | 5 | 4 | 0.286 | 1.000 / 1.000 | 0.800 | 0.200 |
| config4_collusion | 5 | 5 | 0.354 | 1.000 / 1.000 | 1.000 | 0.000 |
| config5_adaptive | 5 | 5 | 0.542 | 1.000 / 1.000 | 1.000 | 0.000 |
| config6_self_rating | 5 | 2 | 0.097 | 1.000 / 1.000 | 0.400 | **0.600** |
| config7_admin_escalation | 5 | 1 | 0.021 | 1.000 / 1.000 | 0.200 | **0.800** |
| config8_evidence_tamper | 5 | 2 | 0.372 | 1.000 / 1.000 | 0.400 | **0.600** |
| config9_gate_bypass | 5 | 3 | 0.074 | 1.000 / 1.000 | 0.600 | **0.400** |
| config10_provenance_replay | 5 | 5 | 0.233 | 1.000 / 1.000 | 1.000 | 0.000 |
| config11_comprehensive | 5 | 3 | 0.482 | 0.333 / 0.000 | 0.200 | 0.133 |
| config2_terminal | 3 | 3 | 0.792 | 0.649 / 0.947 | 0.649 | 0.000 |
| config11_terminal | 3 | 3 | 0.469 | 1.000 / 1.000 | 1.000 | 0.000 |
| ablation_no_stake | 3 | 3 | 0.740 | 1.000 / 1.000 | 1.000 | 0.000 |
| ablation_no_dispute | 3 | 3 | 0.740 | 1.000 / 1.000 | 1.000 | 0.000 |
| ablation_no_decay | 3 | 3 | 0.748 | 1.000 / 1.000 | 1.000 | 0.000 |
| ablation_no_wilson | 3 | 3 | 0.812 | 1.000 / 1.000 | 1.000 | 0.000 |

**The reconciliation is exact:**

* The **unconditional** column reproduces **codebase report §9** seed-for-seed:
  C1 0.600, C3 0.800, C6 0.400, C7 0.200, C8 0.400, C9 0.600, C11 0.200 — identical.
  (§9 explicitly notes "Seeds with 0 ratings → honest=0 by convention".)
* The **conditional** column reproduces the **paper** (`tab_statistical` / abstract: "≥99.7%
  honest in 8/9 single-attack configs"): for every single-attack config the honest-given-rating
  is 1.000 (≈ the paper's 0.997–1.000). C2 conditional mean 0.729 / median 1.000 matches the
  paper's "73.6% (mean) / 99.7% (median)".

**The gap IS the abstention finding.** For the deterministic-defense single-attack configs the
*dominant outcome is abstention, not honest rating*: C7 has only **1 of 5** seeds that ever
rate (4 converge to pure noop), C6 and C8 have 2/5, C9 has 3/5. The paper's "≥99.7% honest"
is true **only conditional on rating** and describes a *minority* of seeds; the system "passes"
these configs largely by agents declining to participate at all. This conditional-vs-unconditional
split is also internally inconsistent **within the paper**: `evaluation.tex` reports C6 as
"93.2% ± 13.3%" (conditional) yet its own per-seed breakdown is {0,0,100,100,0} (unconditional,
mean 40%).

---

## T0.2 — Basin-aware statistics

Source: `results/derived/basin_stats.{json,md}`. **Explicit thresholds (printed by the script):**
`abstention: participation < 0.01`; `adversarial-dominant: honest_given_rating ≤ 0.05 OR final
mean stake ≤ 0.01·reset`; `honest-dominant: honest_given_rating ≥ 0.95 AND participation ≥ 0.01`;
else `mixed`.

| Config | seeds | abst | adv | hon | mixed | P(honest-dom) | Jeffreys 95% CI | bimodal? |
|---|---|---|---|---|---|---|---|---|
| config1_baseline | 5 | 2 | 0 | 3 | 0 | 0.600 | [0.209, 0.906] | yes |
| config2_mixed | 5 | 0 | 1 | 3 | 1 | 0.600 | [0.209, 0.906] | yes |
| config3_sybil | 5 | 1 | 0 | 4 | 0 | 0.800 | [0.371, 0.977] | yes |
| config4_collusion | 5 | 0 | 0 | 5 | 0 | 1.000 | [0.621, 1.000] | **no** |
| config5_adaptive | 5 | 0 | 0 | 5 | 0 | 1.000 | [0.621, 1.000] | **no** |
| config6_self_rating | 5 | 3 | 0 | 2 | 0 | 0.400 | [0.094, 0.791] | yes |
| config7_admin_escalation | 5 | 4 | 0 | 1 | 0 | 0.200 | [0.023, 0.629] | yes |
| config8_evidence_tamper | 5 | 3 | 0 | 2 | 0 | 0.400 | [0.094, 0.791] | yes |
| config9_gate_bypass | 5 | 2 | 0 | 3 | 0 | 0.600 | [0.209, 0.906] | yes |
| config10_provenance_replay | 5 | 0 | 0 | 5 | 0 | 1.000 | [0.621, 1.000] | **no** |
| config11_comprehensive | 5 | 2 | 2 | 1 | 0 | 0.200 | [0.023, 0.629] | yes |
| config2_terminal | 3 | 0 | 1 | 1 | 1 | 0.333 | [0.039, 0.823] | yes |
| config11_terminal | 3 | 0 | 1 | 2 | 0 | 0.667 | [0.177, 0.961] | yes |

The unimodal **near-1** configs (C4, C5, C10) — all 5 seeds honest-dominant — are the only ones
where a one-sample t-test against a >95% threshold is defensible; for those the t-test in the
paper's `tab_statistical` is appropriate. For **every bimodal config (C1, C2, C3, C6, C7, C8,
C9, C11)** the mean±std and one-sample t-test are invalid (the per-seed distribution is a mix of
0/1 basins, not Gaussian); the Jeffreys interval on the honest-dominant probability is the
correct summary.

### Terminal-reward claim — explicit CI statement

The paper/codebase claim the terminal-progress reward "fixes" C11/C2 bimodality
("Terminal reward fix C11: 2/3 honest=1.0 vs 1/5 without"). With **n = 3 seeds** this claim
cannot be supported:

* **config2_terminal**: 1/3 honest-dominant (1 adversarial-dominant, 1 mixed at hgr=0.947).
  Jeffreys 95% CI on P(honest) = **[0.039, 0.823]**.
* **config11_terminal**: under deterministic re-eval, **2/3** honest-dominant (seed 1 rates
  honestly but goes **bankrupt** → adversarial-dominant by the stake criterion). Jeffreys 95%
  CI = **[0.177, 0.961]**. Even taking the saved-log reading of **3/3** at face value, the
  Jeffreys CI for 3/3 is **[0.464, 1.000]** — a lower bound of 0.46 cannot exclude a ~50%
  adversarial-collapse rate.

**Conclusion:** at n=3 (≤5000 ep, b=0.5/1.0) neither terminal config supports a claim that
bimodality is "eliminated". The CIs are enormous and overlap the failure region. A larger seed
count (≥20) would be required to make any such claim.

*Edge case flagged:* config11_terminal seed 1 has `honest_given_rating = 1.0` but `final mean
stake = 0`; the task-specified "stake ≈ 0 ⇒ adversarial-dominant" rule classifies it as
adversarial-dominant (an honest-but-bankrupt collapse). This is faithful to the spec; the raw
honest-given-rating is preserved in `honest_metrics.json` for transparency.

---

## T0.3 — Reputation accuracy vs random baseline

Source: `results/derived/accuracy_vs_random.{json,md}`. **Important baseline correction:** the
env draws true quality `q ~ Beta(8,2)` (mean 0.8), **not** `q ~ U[0,1]`. The analytic random
MAE of `1/3 ≈ 0.333` (random estimator vs uniform truth) therefore does **not** apply. The
empirical random MAE below is a `U[0,1]` estimator matched to the **actual** Beta(8,2)
qualities (averaged over 1000 draws) and equals **≈ 0.355**.

| Config | n | system MAE | Spearman ρ | empirical random MAE | **skill** | mean score | ≤ random? |
|---|---|---|---|---|---|---|---|
| config1_baseline | 2000 | 0.361 | −0.129 | 0.355 | −0.016 | 0.449 | **FLAG** |
| config2_mixed | 2000 | 0.382 | −0.239 | 0.355 | −0.076 | 0.441 | **FLAG** |
| config3_sybil | 2000 | 0.338 | −0.056 | 0.355 | 0.048 | 0.472 | |
| config4_collusion | 2000 | 0.228 | 0.276 | 0.355 | 0.359 | 0.586 | |
| config5_adaptive | 2000 | 0.192 | 0.421 | 0.355 | 0.458 | 0.618 | |
| config6_self_rating | 2000 | 0.390 | −0.178 | 0.355 | −0.098 | 0.413 | **FLAG** |
| config7_admin_escalation | 2000 | 0.261 | 0.111 | 0.355 | 0.265 | 0.547 | |
| config8_evidence_tamper | 2000 | 0.305 | 0.002 | 0.355 | 0.140 | 0.502 | |
| config9_gate_bypass | 2000 | 0.359 | −0.085 | 0.355 | −0.012 | 0.446 | **FLAG** |
| config10_provenance_replay | 2000 | 0.294 | 0.070 | 0.355 | 0.171 | 0.522 | |
| config11_comprehensive | 2000 | 0.250 | 0.098 | 0.355 | 0.295 | 0.557 | |
| config2_terminal | 1200 | 0.365 | −0.207 | 0.355 | −0.029 | 0.492 | **FLAG** |
| config11_terminal | 1200 | 0.246 | 0.020 | 0.355 | 0.307 | 0.565 | |
| ablation_no_stake | 1200 | 0.139 | 0.494 | 0.355 | 0.610 | 0.672 | |
| ablation_no_dispute | 1200 | 0.139 | 0.494 | 0.355 | 0.610 | 0.672 | |
| ablation_no_decay | 1200 | **0.119** | **0.531** | 0.355 | **0.664** | 0.691 | |
| ablation_no_wilson | 1200 | 0.408 | −0.312 | 0.355 | −0.150 | 0.398 | **FLAG** |

**At or below random (skill ≤ 0):** C1, C2, C6, C9, config2_terminal, ablation_no_wilson.
These are the configs where the system is actively *miscalibrated* — Spearman ρ is **negative**,
i.e. scores are anti-correlated with true quality.

### Diagnostic — WHY (prior floor + decay)

* **α0 = β0 = 2 prior pulls every score toward 0.5.** An agent with no/low evidence has score
  exactly `2/4 = 0.5`. Because true quality is `Beta(8,2)` (mean 0.8, concentrated high), a
  *constant 0.5 predictor* already scores MAE **0.302** vs the random MAE **0.354**, i.e. a
  spurious **skill ≈ +0.147 with ρ = 0**. So a positive MAE-skill in the abstention-heavy
  configs (C7, C8) is an **artifact of the prior sitting near the quality mean**, not evidence of
  discrimination — confirmed by their near-zero Spearman ρ (C8 ρ = 0.002, C7 ρ = 0.111). The
  mean score is ≤ 0.69 everywhere (vs true mean 0.80): the 0.5 prior imposes a persistent
  downward bias.
* **Temporal decay destroys ranking.** Decay continually pulls accumulated (α,β) back toward the
  (2,2) prior, erasing evidence. Removing it gives the best discrimination: **no_decay ρ =
  +0.531 (pooled)**, cross-referencing the existing `spearman_results.json` per-episode mean of
  **+0.653**; full-system ρ is **−0.366** there. (Methodology differs — pooled single ρ vs mean
  of per-episode ρ — but the sign and ordering agree: no_decay ≫ full-system.)

We **measured and exposed** the accuracy problem only; no accuracy fix was attempted (per task).

---

## T0.4 — Python engine validation + the Wilson-n discrepancy

Tests: `tests/test_reputation_engine.py` — **19 tests, all passing.**

### The Wilson-n discrepancy: **resolved — the alleged bug does not exist**

The report claimed the code computes Wilson `n = α+β−2α0` while paper Eq.(2) uses `n = α+β`.
**Both implementations actually use `n = α+β`:**

* Python `env/reputation_engine.py:156` → `n = alpha + beta`.
* Go canonical `~/am-unified/chaincode/unified/helpers.go:275` → `n := alpha + beta`.
* An exhaustive grep found **no** `α+β−2α0` (or `n−2·alpha0`) form anywhere in either codebase.

So the code is consistent with paper Eq.(2); there is no discrepancy to fix. Two tests pin this:
`test_wilson_matches_n_equals_alpha_plus_beta` (asserts the engine == `n=α+β` form) and
`test_wilson_is_not_n_minus_2alpha0` (guard that fails if it ever switches to the `n−2α0` form).

### Go reference: **found** — golden cross-check added

The canonical Go chaincode was located at `~/am-unified/chaincode/unified/` (and an identical
copy under `~/AM/am-unified/...`). The Python engine matches the Go math exactly for:

| Function | Go location | Match |
|---|---|---|
| `applyDynamicDecay` (decay toward prior, floor at prior) | helpers.go:239 | ✅ identical formula |
| `calculateWilsonCI` (`n=α+β`) | helpers.go:274 | ✅ identical |
| Bayesian update `α += w·v`, `β += w·(1−v)` | reputation_contract.go:1073 | ✅ identical |
| slash `balance·(1−0.10)` | reputation_contract.go:1249 | ✅ identical |
| dispute cost `balance −= 100` | reputation_contract.go:686 | ✅ identical |
| stake gate `balance ≥ min_stake` | reputation_contract.go:574 | ✅ identical |

**One documented divergence (asserted, not papered over):** the Go *default* config clamps
rater weight to **[0.1, 5.0]** (reputation_contract.go:52-53) whereas the Python `SystemConfig`
uses **[0.5, 2.0]**. The core math is identical; only these clamp bounds differ. Recorded by
`test_rater_weight_clamp_divergence` so it cannot regress silently. All other SystemConfig
defaults (priors 2/2, decay 0.98/86400, min_stake 10000, dispute 100, slash 0.10) match the Go
defaults (`test_config_defaults_match_go`).

---

## T0.5 — Bug fixes

### (a) Ablation "logging collision" (no_stake vs no_dispute)

Source: `evaluation/diagnose_ablation.py` → `results/derived/ablation_diagnosis.{json,md}`.

**Diagnosis: NOT a file overwrite / data loss.** Both variants' raw per-seed logs exist and were
**independently trained** — distinct wall-clock times (no_stake seed0 4338.6s vs no_dispute
4971.9s, etc.) and distinct stake magnitudes (final mean stake ~2.0 vs ~20000.0). A field-by-field
comparison over all 3 seeds × 5000 episodes shows:

* **Identical** across every episode: `honest_pct, dishonest_actions, total_actions, total_ratings,
  attacks_attempted, attacks_blocked, defense_by_type, defense_hold_rate, mean_reputation_accuracy,
  mean_score, mean_ci_width, total_sybils, total_disputes_filed` (+ loss terms).
* **Differing**: `stakes, total_reward, mean_reward` only — and `no_dispute` stakes are exactly
  `10000×` the `no_stake` stakes (the `min_stake` ratio), e.g. agent_0 1.8 vs 18000.0 (one 10% slash).

**Root cause (a real design flaw, not a logging bug):** the env normalizes stake by `min_stake`
in the observation — `obs[2] = balance / (4·min_stake)` and reset balance `= 2·min_stake`, so the
normalized stake observation is **identical (0.5·0.9ᵏ)** regardless of `min_stake`; and
`dispute_cost` never enters the observation. The shared policy therefore receives identical inputs
under both configs and, with identical seeds, produces identical action trajectories. The identical
summaries are thus **correct given the data**, and regenerating each summary from its own raw logs
reproduces them (done — `ablation_diagnosis.json`).

**Re-run required for data recovery: NO** (data is present and independently generated). Two
honest caveats for the paper:
1. The `no_stake` ablation does **not** actually remove economic deterrence from the agent's
   decision input (stake is normalized away), so it does **not** test what it claims. A
   *methodologically* meaningful re-run would require changing the observation's stake
   normalization to an absolute scale — out of Tier-0 scope, flagged for future work.
2. The codebase report §10 lists `no_stake` 87.8% / `no_dispute` 94.7% honest and "stake removal
   −12pp", but the actual summary JSONs both show **99.79%** conditional honest. The §10/§19
   "−12pp" / "<90%" / "biggest ablation drop" claims are **not supported** by the saved data.
   Contingency command if a fresh independent `no_dispute` run is nonetheless desired:
   `python3 training/train_ablation.py --config configs/ablation_no_dispute.yaml --output results/ablation/ --seeds 3`

### (b) Corresponding-author email

* **IEEE `marl-paper/main.tex`**: byline is anonymized (`\author{[Draft — Authors Omitted for
  Review]}`) — **no email present, nothing to fix** (the running header "Almaayn et al." carries
  no email).
* **Elsevier `marl-paper/Computer_Networks__Raddad2026_/mainSD.tex`**: `\corref{cor1}` was on
  Raddad Almaayn but `\ead{ralmaayn@unm.edu}` followed `\author{Michael Devetsikiotis}`, so
  elsarticle bound the email to Devetsikiotis. **Fixed:** moved `\ead{ralmaayn@unm.edu}` to
  directly follow the Almaayn `\author` line (and removed the misplaced one after the
  `\cortext`). The corresponding-author name now matches the email owner (Raddad Almaayn).

---

## Files changed / added

**Added (all idempotent, CLI-driven):**
* `evaluation/recompute_metrics.py` — T0.1/T0.2/T0.3 deterministic-eval recompute + tables.
* `evaluation/diagnose_ablation.py` — T0.5(a) diagnosis + summary regeneration.
* `tests/test_reputation_engine.py` — T0.4 engine validation (19 tests).
* `results/derived/` — `honest_metrics.{json,md}`, `basin_stats.{json,md}`,
  `accuracy_vs_random.{json,md}`, `ablation_diagnosis.{json,md}`, `eval_cache/*.json`.

**Edited:**
* `marl-paper/Computer_Networks__Raddad2026_/mainSD.tex` — corresponding-author email (T0.5b).

**Not modified:** any raw result JSON, any checkpoint, any training code, the IEEE `main.tex`
(nothing to fix). No git commit/push performed.

## Data gaps found

1. Per-action counts A0..A11 and `agent_scores[]`/`agent_true_qualities[]` are absent from the
   saved training logs → collected here by deterministic eval from `final.pt` (authorized).
2. Terminal & ablation summary JSONs have empty `seed_results: []`; per-seed metadata was read
   from the individual `*_seedN.json` files instead.
3. Deterministic re-eval (fixed seeds) does not reproduce the *exact* episodes of the original
   non-seeded eval, so a few borderline seeds shift basin (e.g. config2_terminal seed1 at hgr
   0.947 vs saved 0.950; config11_terminal seed1 honest-but-bankrupt). Raw per-seed values are
   preserved in `honest_metrics.json`. This is expected and does not affect the conditional /
   unconditional reconciliation, which is exact against §9 and the paper.
4. Codebase report §10/§19 ablation honest-% (87.8/94.7) disagree with the summary JSONs (99.79);
   flagged above, not reconciled (would need the original §10 computation script, which was not
   found).

---

## End-to-end run output

(Generated by `scripts`/pytest; see the very end of this file for the pasted console output.)

```
########## pytest ##########
...................                                                      [100%]
19 passed in 0.17s

########## recompute_metrics.py (cached) ##########
[ablation_no_dispute] evaluating...
[ablation_no_decay] evaluating...
[ablation_no_wilson] evaluating...

Wrote outputs under /home/raddad/marl-reputation/results/derived/
  - honest_metrics.md
  - basin_stats.md
  - accuracy_vs_random.md

########## diagnose_ablation.py ##########
Wrote results/derived/ablation_diagnosis.{json,md}
runs_independent_by_walltime=True
identical fields: ['attacks_attempted', 'attacks_blocked', 'defense_by_type', 'defense_hold_rate', 'dishonest_actions', 'entropy', 'episode', 'honest_pct', 'mean_ci_width', 'mean_reputation_accuracy', 'mean_score', 'mean_true_quality', 'pg_loss', 'seed', 'total_actions', 'total_disputes_filed', 'total_loss', 'total_ratings', 'total_sybils', 'vf_loss']
differing fields: ['mean_reward', 'stakes', 'total_reward']
```
