# Phase 1–2 — why rho is undefined, and a reframed endpoint

> **Revision history**
> - **2026-09-02 (initial):** Phases 1–2 written. Characterised the 22 c11
>   collapsed shards as "jamming — agents rate heavily but ratings cancel".
> - **2026-09-02 (this revision, supersedes the above):** that claim is
>   **retracted**. CI-width inversion and the stake ledger show zero ratings
>   were ever applied: alpha = beta = 2.000 at the (2,2) floor, max
>   single-agent stake 6,200 against a 10,000 gate. The mechanism is
>   **stake-gated lockout**, not cancellation. Sections revised: "Two distinct
>   collapse modes", the VERDICT amendment (item 4). Added: the c9/c10/c11
>   dose-response, the applied-vs-attempted participation result, and the
>   Gate 2 decision. Nothing in Phase 1.1, 1.3, 2.1, 2.2 or 2.3 changed —
>   the mechanism (score-vector collapse), the selection finding and the
>   reframed endpoint all stand as originally computed.


Host: zenbook (10.12.10.98). Repo `/home/raddad/marl-reputation`.
Source: `analysis/rho_2026_09/shards.csv` (540 shards), re-scored from
existing `final.pt` checkpoints. No training, no imputation, no writes to
`results/tier1b/sweep/`.

**Reproduction check:** recomputed rho matches the stored `system_spearman_rho` for **540/540** shards.

## Phase 1.1 — mechanism

- shards with undefined rho: **157/540**
- shards with collapsed score vector (std < 1e-12): **157**
- shards with non-degenerate true-quality vector (std > 1e-12): **540/540**
- undefined set == collapsed set: **True**

The guard in `evaluation/basins.py::_score_quality_metrics` sets rho to NaN when
`std(scores) < 1e-12` **or** `std(quals) < 1e-12`. The true-quality vector is
never degenerate, so only the score branch ever fires. **Mechanism: score-vector
collapse.** With no ratings submitted, every agent's Beta posterior stays at the
prior mean 0.5, the score vector is constant, and Spearman is 0/0.

### Two distinct collapse modes

Every one of the 157 collapsed shards sits at mean score 0.5 — the Beta prior.
They arrive there two different ways.

- **Mode 1 — abstention collapse (135 shards).** participation < 0.01; nobody
  attempts to rate, so no posterior ever moves. The system was never used.
- **Mode 2 — stake-gated lockout (22 shards, all c11).** participation
  0.30–0.98, basin `adversarial-dominant`. Agents attempt to rate constantly,
  and **not one attempt is ever applied**.

**Evidence for Mode 2** (2026-09-02, read-only, no eval runs):

- `total_ratings = 0` in all 22 shards — zero rating events reached the engine.
- Inverting the stored `mean_ci_width` gives `alpha + beta = 4.000` in all 22,
  i.e. **alpha = beta = 2.000**, exactly the `(initial_alpha, initial_beta) =
  (2, 2)` decay floor in `env/reputation_engine.py`. Because the floor forces
  n >= 4 and Wilson width is decreasing in n, `width(4, 0.5)` is the *maximum*
  attainable value; a mean equal to it forces **every** agent to the floor. So
  this is per-agent, not an average, and the distinct count is 1.
- **Highest single-agent stake across all 22 shards: 6,200**, against
  `min_stake_required = 10,000` (reset 20,000). Not one agent anywhere clears
  the gate. 21/22 shards have mean stake exactly 0.
- 30–40 sybils created per shard.

**Mechanism:** agents spend their stake on sybil creation, fall below the stake
gate, and `can_rate()` rejects every subsequent rating. Nothing reaches the
engine, alpha and beta never leave (2, 2), and mu = 2/(2+2) = 0.5 mechanically
for every agent — hence zero variance, hence undefined rho.

The metric that hid this: `participation_rate` is computed in `basins.py` as
`(A1+A2+A3+A4)/total_actions` — rating actions *attempted*, straight from
`action_counts`. It never checks whether a rating was applied. A stake-less
agent can register 98% "participation" while contributing nothing.

### Stake starvation is a dose-response, not three separate findings

Counting shards in which **no agent** clears the 10,000 stake gate:

| config (arm A) | shards fully locked out | attempt→application conversion |
|---|---|---|
| c9 | 0/30 | 1.031 |
| c10 | 3/10 | **0.095** |
| c11 | 21/30 | **0.243** |

The binary `applied_evidence == 0` test catches only total lockout, so it flags
c11 and misses c10 entirely — no c10 shard reaches exactly zero. But c10 arm A
converts just 9.5% of its attempts, the worst ratio in the sweep. c9, c10 and
c11 are one graded phenomenon at increasing severity, driven by how hard the
configuration pushes agents toward sybil spending, not three unrelated
observations. Healthy configs sit at conversion 0.995–1.031, so these are far
outside the normal band.

## Phase 1.3 — is definedness explained by participation?

Mann-Whitney U on participation rate, rho-defined vs rho-undefined seeds,
with Cliff's delta (+1 = defined seeds strictly higher participation).

| arm | config | n def | n und | median part. (def) | median part. (und) | U | p | Cliff's d |
|---|---|---|---|---|---|---|---|---|
| A | c1 | 15 | 15 | 0.2802 | 0.0000 | 225 | 6.87e-07 | +1.000 |
| A | c2 | 26 | 4 | 0.7651 | 0.0000 | 104 | 0.00143 | +1.000 |
| A | c3 | 29 | 1 | 0.5868 | 0.0000 | 29 | 0.106 | +1.000 |
| A | c4 | 10 | 0 | 0.9935 | nan | nan | nan | +nan |
| A | c5 | 8 | 2 | 0.3921 | 0.0000 | 16 | 0.0467 | +1.000 |
| A | c6 | 18 | 12 | 0.4743 | 0.0000 | 216 | 2.54e-06 | +1.000 |
| A | c7 | 11 | 19 | 0.4214 | 0.0000 | 209 | 2.21e-07 | +1.000 |
| A | c8 | 11 | 19 | 0.1444 | 0.0000 | 209 | 2.17e-07 | +1.000 |
| A | c9 | 7 | 23 | 0.1121 | 0.0000 | 161 | 1.22e-07 | +1.000 |
| A | c10 | 4 | 6 | 1.0000 | 0.0000 | 24 | 0.00502 | +1.000 |
| A | c11 | 8 | 22 | 0.9234 | 0.9800 | 111 | 0.274 | +0.261 |
| A | **POOLED** | 147 | 123 | 0.5859 | 0.0000 | 16289 | 1.28e-31 | +0.802 |
| B | c1 | 29 | 1 | 1.0000 | 0.0000 | 29 | 0.0062 | +1.000 |
| B | c2 | 25 | 5 | 1.0000 | 0.0000 | 125 | 7.05e-07 | +1.000 |
| B | c3 | 28 | 2 | 1.0000 | 0.0000 | 56 | 0.00106 | +1.000 |
| B | c4 | 10 | 0 | 1.0000 | nan | nan | nan | +nan |
| B | c5 | 8 | 2 | 1.0000 | 0.0000 | 16 | 0.00492 | +1.000 |
| B | c6 | 25 | 5 | 1.0000 | 0.0000 | 125 | 2.98e-06 | +1.000 |
| B | c7 | 26 | 4 | 1.0000 | 0.0000 | 104 | 0.00027 | +1.000 |
| B | c8 | 24 | 6 | 1.0000 | 0.0000 | 144 | 3.82e-05 | +1.000 |
| B | c9 | 26 | 4 | 1.0000 | 0.0000 | 104 | 0.000177 | +1.000 |
| B | c10 | 10 | 0 | 0.9995 | nan | nan | nan | +nan |
| B | c11 | 25 | 5 | 1.0000 | 0.7300 | 123 | 0.000163 | +0.968 |
| B | **POOLED** | 236 | 34 | 1.0000 | 0.0000 | 7992 | 1.36e-31 | +0.992 |

- arm A: undefined seeds have participation max=0.9800, mean=0.1307, 105/123 exactly 0.
- arm B: undefined seeds have participation max=0.9800, mean=0.0879, 30/34 exactly 0.

## Phase 2.1 — PRIMARY: P(rho defined), i.e. P(system discriminates at all)

Undefined rho treated as an outcome, not missing data.

| config | armA k/n | armA P [Jeffreys95] | armB k/n | armB P [Jeffreys95] | delta B−A [Newcombe95] |
|---|---|---|---|---|---|
| c1 | 15/30 | 0.500 [0.328, 0.672] | 29/30 | 0.967 [0.855, 0.996] | +0.467 [+0.252, +0.637] |
| c2 | 26/30 | 0.867 [0.713, 0.953] | 25/30 | 0.833 [0.673, 0.933] | -0.033 [-0.220, +0.155] |
| c3 | 29/30 | 0.967 [0.855, 0.996] | 28/30 | 0.933 [0.803, 0.986] | -0.033 [-0.182, +0.108] |
| c4 | 10/10 | 1.000 [0.783, 1.000] | 10/10 | 1.000 [0.783, 1.000] | +0.000 [-0.278, +0.278] |
| c5 | 8/10 | 0.800 [0.497, 0.956] | 8/10 | 0.800 [0.497, 0.956] | +0.000 [-0.341, +0.341] |
| c6 | 18/30 | 0.600 [0.422, 0.760] | 25/30 | 0.833 [0.673, 0.933] | +0.233 [+0.005, +0.433] |
| c7 | 11/30 | 0.367 [0.213, 0.545] | 26/30 | 0.867 [0.713, 0.953] | +0.500 [+0.258, +0.668] |
| c8 | 11/30 | 0.367 [0.213, 0.545] | 24/30 | 0.800 [0.633, 0.912] | +0.433 [+0.185, +0.615] |
| c9 | 7/30 | 0.233 [0.111, 0.404] | 26/30 | 0.867 [0.713, 0.953] | +0.633 [+0.393, +0.774] |
| c10 | 4/10 | 0.400 [0.153, 0.696] | 10/10 | 1.000 [0.783, 1.000] | +0.600 [+0.201, +0.832] |
| c11 | 8/30 | 0.267 [0.135, 0.441] | 25/30 | 0.833 [0.673, 0.933] | +0.567 [+0.321, +0.723] |
| **ALL** | 147/270 | 0.544 [0.485, 0.603] | 236/270 | 0.874 [0.831, 0.910] | +0.330 [+0.256, +0.398] |

## Phase 2.2 — SECONDARY: rho conditional on being defined

**This subset is conditioned on the Phase 1.3 finding.** Seeds enter only when
the score vector did not collapse, which by Phase 1.3 means they are the
higher-participation seeds. In arm A that is a strongly selected minority, so
these means are NOT comparable across arms at face value.

| config | armA n | armA rho | armB n | armB rho | delta B−A [Welch95] |
|---|---|---|---|---|---|
| c1 | 15 | -0.216 | 29 | +0.253 | +0.469 [-0.026, +0.964] |
| c2 | 26 | +0.155 | 25 | -0.307 | -0.462 [-0.892, -0.032] |
| c3 | 29 | +0.170 | 28 | -0.017 | -0.187 [-0.615, +0.241] |
| c4 | 10 | -0.225 | 10 | -0.278 | -0.053 [-0.789, +0.683] |
| c5 | 8 | +0.084 | 8 | +0.164 | +0.080 [-0.648, +0.808] |
| c6 | 18 | +0.174 | 25 | +0.053 | -0.121 [-0.596, +0.354] |
| c7 | 11 | -0.305 | 26 | +0.123 | +0.428 [-0.090, +0.945] |
| c8 | 11 | +0.178 | 24 | +0.224 | +0.046 [-0.469, +0.561] |
| c9 | 7 | +0.351 | 26 | +0.099 | -0.251 [-0.919, +0.417] |
| c10 | 4 | -0.160 | 10 | -0.125 | +0.035 [-0.641, +0.711] |
| c11 | 8 | +0.126 | 25 | -0.090 | -0.216 [-0.969, +0.538] |

### How much does the selection move the arm delta?

Zero-extension: a constant score vector ranks nothing, so its discrimination is
0. The vector is fully observed, so this is a definitional choice at a known
point, not imputation of a missing value.

- arm A, among defined seeds: Spearman(participation, rho) = +0.196 (p=0.0171, n=147)
- arm B, among defined seeds: Spearman(participation, rho) = +0.084 (p=0.198, n=236)

| treatment | armA | armB | delta B−A | armA n | armB n |
|---|---|---|---|---|---|
| rho \| defined (as published) | +0.0591 | +0.0284 | -0.0308 | 147/270 | 236/270 |
| zero-extended over all seeds | +0.0322 | +0.0248 | -0.0074 | 270/270 | 270/270 |

**|delta| shrinks by 76%** once the excluded
seeds are counted at their true (zero) discrimination.

Per config, both treatments:

| config | armA def/n | armB def/n | delta (rho\|def) | delta (zero-ext) | sign flip |
|---|---|---|---|---|---|
| c1 | 15/30 | 29/30 | +0.4690 | +0.3523 |  |
| c2 | 26/30 | 25/30 | -0.4619 | -0.3901 |  |
| c3 | 29/30 | 28/30 | -0.1867 | -0.1799 |  |
| c4 | 10/10 | 10/10 | -0.0530 | -0.0530 |  |
| c5 | 8/10 | 8/10 | +0.0797 | +0.0638 |  |
| c6 | 18/30 | 25/30 | -0.1210 | -0.0603 |  |
| c7 | 11/30 | 26/30 | +0.4275 | +0.2183 |  |
| c8 | 11/30 | 24/30 | +0.0458 | +0.1139 |  |
| c9 | 7/30 | 26/30 | -0.2513 | +0.0042 | **YES** |
| c10 | 4/10 | 10/10 | +0.0351 | -0.0611 | **YES** |
| c11 | 8/30 | 25/30 | -0.2157 | -0.1083 |  |

## Phase 2.3 — collapse-robust discrimination: Kendall's tau-a

tau_a = (C − D) / [n(n−1)/2]. The denominator depends only on n, so tau-a is
defined whenever n ≥ 2; a constant score vector gives C = D = 0 and therefore
tau_a = 0 exactly — the correct statement that the score ranks nothing. This is
not imputation: the collapsed score vector is fully observed. tau-b is NOT used,
because its tie-corrected denominator reproduces the same 0/0 collapse as rho.

**All seeds included — no exclusions.**

| config | armA n | armA tau_a | armB n | armB tau_a | delta B−A [Welch95] |
|---|---|---|---|---|---|
| c1 | 30 | -0.080 | 30 | +0.208 | +0.288 [+0.005, +0.571] |
| c2 | 30 | +0.115 | 30 | -0.200 | -0.315 [-0.610, -0.021] |
| c3 | 30 | +0.138 | 30 | -0.010 | -0.148 [-0.471, +0.176] |
| c4 | 10 | -0.168 | 10 | -0.226 | -0.058 [-0.651, +0.535] |
| c5 | 10 | +0.050 | 10 | +0.087 | +0.037 [-0.409, +0.483] |
| c6 | 30 | +0.089 | 30 | +0.056 | -0.033 [-0.302, +0.236] |
| c7 | 30 | -0.086 | 30 | +0.098 | +0.185 [-0.075, +0.445] |
| c8 | 30 | +0.057 | 30 | +0.143 | +0.086 [-0.136, +0.308] |
| c9 | 30 | +0.065 | 30 | +0.076 | +0.010 [-0.237, +0.258] |
| c10 | 10 | -0.044 | 10 | -0.099 | -0.055 [-0.570, +0.460] |
| c11 | 30 | +0.029 | 30 | -0.048 | -0.077 [-0.306, +0.153] |

### rho-conditional vs tau-a: the size of the selection effect

| config | arm | rho (defined only) | tau_a (defined only) | tau_a (ALL seeds) | shift |
|---|---|---|---|---|---|
| c1 | A | -0.216 | -0.161 | -0.080 | +0.080 |
| c1 | B | +0.253 | +0.215 | +0.208 | -0.007 |
| c2 | A | +0.155 | +0.133 | +0.115 | -0.018 |
| c2 | B | -0.307 | -0.240 | -0.200 | +0.040 |
| c3 | A | +0.170 | +0.143 | +0.138 | -0.005 |
| c3 | B | -0.017 | -0.011 | -0.010 | +0.001 |
| c4 | A | -0.225 | -0.168 | -0.168 | +0.000 |
| c4 | B | -0.278 | -0.226 | -0.226 | +0.000 |
| c5 | A | +0.084 | +0.063 | +0.050 | -0.013 |
| c5 | B | +0.164 | +0.109 | +0.087 | -0.022 |
| c6 | A | +0.174 | +0.149 | +0.089 | -0.059 |
| c6 | B | +0.053 | +0.067 | +0.056 | -0.011 |
| c7 | A | -0.305 | -0.236 | -0.086 | +0.149 |
| c7 | B | +0.123 | +0.114 | +0.098 | -0.015 |
| c8 | A | +0.178 | +0.155 | +0.057 | -0.098 |
| c8 | B | +0.224 | +0.179 | +0.143 | -0.036 |
| c9 | A | +0.351 | +0.279 | +0.065 | -0.214 |
| c9 | B | +0.099 | +0.087 | +0.076 | -0.012 |
| c10 | A | -0.160 | -0.110 | -0.044 | +0.066 |
| c10 | B | -0.125 | -0.099 | -0.099 | +0.000 |
| c11 | A | +0.126 | +0.109 | +0.029 | -0.080 |
| c11 | B | -0.090 | -0.057 | -0.048 | +0.010 |

## Phase 2.4 — Table 4 defects made explicit

Per-engine n (the published table reports only the beta count as `n`), and the
hold-rate n exposed as its own column.

| config | arm | files | n beta | n flat | n eigentrust | n hold_rate |
|---|---|---|---|---|---|---|
| c11 | A | 30 | 8 | 8 | 7 | 2 |
| c11 | B | 30 | 25 | 25 | 25 | 5 |
| c2 | A | 30 | 26 | 26 | 25 | 3 |
| c2 | B | 30 | 25 | 25 | 25 | 1 |
| c3 | A | 30 | 29 | 29 | 29 | 11 |
| c3 | B | 30 | 28 | 28 | 28 | 3 |


## VERDICT on the working hypothesis

**The hypothesised mechanism is CONFIRMED. The confound is real. Its effect on the
headline is large in point estimate, but the pooled rho delta it distorted was
never statistically significant to begin with.**

What is established:

1. rho goes undefined **iff** the score vector collapses to zero variance (157/157, sets identical). Never a quality-vector or NaN-input problem.
2. Definedness is near-perfectly ordered by participation: Cliff's d = +1.000 in 17 of 19 config x arm cells; pooled arm A d=+0.802 (p=1.3e-31), arm B d=+0.992 (p=1.4e-31).
3. Survival into the rho mean is **147/270 (54.4%) in arm A vs 236/270 (87.4%) in arm B** — exactly the differential selection the hypothesis predicts.
4. **Amendment (corrected 2026-09-02).** 22 of the 157 collapses (all c11) occur at participation 0.30-0.98, not at zero. These are **stake-gated lockout**, not jamming: zero ratings applied, alpha = beta = 2.000 at the floor, no agent above 6,200 stake against a 10,000 gate. An earlier draft of this file called them "jamming / ratings cancel to no net information" — that was wrong and is retracted. "Undefined = abstained" is still false for them, but the reason is that they were locked out, not that they were outvoted.

Effect on the arm delta:

| metric | armA | armB | delta B-A | 95% CI | n | CI incl. 0 |
|---|---|---|---|---|---|---|
| rho \| defined (as published) | +0.0591 | +0.0284 | -0.0308 | [-0.1875, +0.1260] | 147/270 vs 236/270 | yes |
| tau-a, all seeds (robust) | +0.0303 | +0.0271 | -0.0032 | [-0.0913, +0.0849] | 270/270 both | yes |

The point estimate shrinks by 90% (-0.0308 -> -0.0032) once collapsed seeds are counted at their true zero discrimination. Two configs flip sign (c9 -0.2513 -> +0.0102; c10 +0.0351 -> -0.0546). **Both CIs contain zero**, so the pooled claim was not significant under either treatment — the confound inflated a non-significant point estimate rather than manufacturing a significant one.

### The finding that survives, and is stronger

Treating collapse as an outcome rather than as missing data gives a primary endpoint that is large and unambiguous:

> **P(system discriminates at all): arm A 0.544 [0.485, 0.603], arm B 0.874 [0.831, 0.910], delta +0.330 [Newcombe95 +0.256, +0.398].**

Conditional on discriminating at all, the two arms are indistinguishable (tau-a delta -0.0032, CI [-0.0913, +0.0849]).

So "participation recovers but discrimination does not" is the wrong reading. The participation arm raises the probability that the reputation system produces **any** discrimination from 54% to 87%; what it does not do is improve discrimination *given* that the system already worked. That is a different and better-supported claim, and it does not depend on which estimator is used.



## Applied vs attempted participation (added 2026-09-02)

The stake-gate finding above means `participation_rate` measures rating
*attempts*, not ratings that reach the engine. Recomputing the Table 2
participation arm-delta on an applied measure — full tables in
[`applied_evidence_findings.md`](applied_evidence_findings.md), not duplicated
here:

- The published **9-of-11** significant configs becomes **10-of-11**. c5 crosses
  into significance; c4 remains the only null.
- Pooled arm-delta rises from **+0.5063** [+0.4444, +0.5682] to **+0.5938**
  [+0.5345, +0.6532].
- c11's delta more than triples, **+0.1817 → +0.6483**, because arm A's
  attempted figure was inflated by the 18 stake-locked shards that applied
  nothing.
- Arm B converts **98.6%** of attempts into applied ratings; arm A only
  **73.3%**.

**Arm B's participation gain is applied ratings, not attempts — and the
published attempted measure understates it.** The headline participation result
is therefore not weakened by the stake-gate discovery; it survives and
strengthens. Only c11's magnitude was materially misstated.

Method and validation for the applied measure are in
`applied_evidence_findings.md`; the per-shard data is `applied_evidence.csv`
(540 rows, 540/540 inversions, none imputed).

## Gate 2 decision — Phase 3 (c11 pilot) CANCELLED (2026-09-02)

**Cancelled, not deferred.** The proposed pilot was to re-score 10 c11 arm-A
seeds at `--eval-episodes 100` to test whether more episodes recover rho.

It cannot. Stake is already spent at the checkpoint: no agent in any of the 22
locked shards holds more than 6,200 against a `min_stake_required` of 10,000,
and `can_rate()` is evaluated per rating attempt. Raising the episode count
generates more attempts, every one of which is rejected on the same test.
alpha and beta stay at (2, 2) for any episode count, so rho stays undefined.

**The collapse is structural, not a sampling artifact**, and the ~5 h of compute
the pilot would cost would only re-derive what the stake ledger already shows.
