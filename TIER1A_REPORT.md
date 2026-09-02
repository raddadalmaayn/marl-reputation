# Tier 1A Report — Environment-Correctness Fixes

**Scope:** modify env/engine/training code behind config flags with legacy defaults
intact; validate with **smoke runs only** (single-thread, fixed seeds, ≤150 episodes —
well under the 3000 cap). No full matrix, no seed sweep, no extended episodes, no git
commit. All v1 artifacts preserved; every new output is under `results/tier1a/` and
every new checkpoint under `results/tier1a/checkpoints/`.

---

## READ FIRST — what was confirmed against live code (before editing)

| Claim (Tier 0) | Live code | Confirmed |
|---|---|---|
| `obs[2] = balance/(4·min_stake)` | `reputation_marl_env.py:527` with `max_stake = min_stake·4` at `:519` | ✅ exact |
| reset balance = `2·min_stake` | `reputation_marl_env.py:207` | ✅ |
| `can_rate` gate = `balance ≥ min_stake` | `reputation_engine.py:177` (`can_rate`) | ✅ |
| rater-weight clamp `max(min_rw, min(max_rw, w))` | `reputation_engine.py:132` (`apply_rating`) | ✅ |
| clamp defaults `[0.5, 2.0]` | `SystemConfig` `:34-35` | ✅ (now changed — WS2) |
| env `config=None → SystemConfig()`; `train.py` used default config (no YAML SystemConfig), `train_ablation.py` built one from YAML with **hardcoded `[0.5,2.0]` fallbacks** (`:91-92`) | confirmed | ✅ |

**Invariance proof reproduced directly:** instantiating the env with `min_stake=1` vs
`min_stake=10000` in `legacy_normalized` mode yields `obs[2] = 0.50000` in **both** cases
(min_stake cancels — the bug). This is the formula Tier 0 described, verbatim.

---

## WS1 — Stake observation fix

**New flag `stake_obs_mode`** (env `__init__`, default **`absolute_log`**), values:

* `legacy_normalized` — v1 behavior, `obs[2] = clip(balance/(4·min_stake), 0, 1)`.
  Kept solely for exact v1 reproduction.
* `absolute_log` — **new default**, `obs[2] = clip(log1p(balance)/log1p(STAKE_OBS_SCALE), 0, 1)`
  with module constant `STAKE_OBS_SCALE = 50000.0`, **independent of `min_stake`**.

**Why this transform.** It is (i) a function of *absolute* balance, so it does not cancel
`min_stake`; (ii) monotone increasing in balance, preserving ordering; (iii) bounded in
`[0,1]` via the existing clip, so the obs stays 14-dim and within the env's `[-1,2]` Box with
no architecture change; (iv) `log1p` compresses the wide stake range (≈2 … 20000+) into a
usable dynamic range — e.g. balance 2 → 0.102, balance 20000 → 0.915, so the two ablation
regimes are now clearly distinguished by the policy. The economics are untouched: reset
balance is still `2·min_stake` and the `can_rate` gate is still `balance ≥ min_stake`; only
the *presentation* of stake to the policy changed.

Direct evidence (no training needed):

```
[legacy_normalized] obs[2]: min_stake=1 -> 0.50000, min_stake=10000 -> 0.50000 | equal=True
[absolute_log]      obs[2]: min_stake=1 -> 0.10154, min_stake=10000 -> 0.91532 | equal=False
```

---

## WS2 — Rater-weight clamp matched to Go chaincode

`SystemConfig` default clamp changed **`[0.5, 2.0]` → `[0.1, 5.0]`** to match the canonical
Go default (`reputation_contract.go:52-53`). Legacy `[0.5, 2.0]` remains selectable via a YAML
`system_config` override (keys `min_rater_weight` / `max_rater_weight`), wired through both
`train.py` and `train_ablation.py` via a new `build_system_config()` that starts from
`SystemConfig()` defaults and overrides only keys present in the YAML (this also removed the
old hardcoded `[0.5,2.0]` fallbacks in `train_ablation.py:91-92`).

Engine tests updated (WS2):
* `test_rater_weight_clamp_divergence` → **inverted** to
  `test_rater_weight_clamp_matches_go_default` (asserts default == `[0.1, 5.0]`).
* Added `test_rater_weight_legacy_override_still_selectable` (override yields `[0.5, 2.0]`;
  weight 9.0 clamps to 2.0 → α=4.0, not the new 7.0).
* `test_update_weight_clamped_high/low` updated to the new bounds (→ 7.0 and 2.1).

**Engine test count: 19 (Tier 0) → 20 (Tier 1A), all passing.**

---

## WS3 — Basin classifier promoted to a training-time signal

New module **`evaluation/basins.py`** with the EXACT Tier 0 thresholds (printed in its
docstring): `abstention: participation<0.01`; `adversarial-dominant: honest_given_rating≤0.05
OR final_mean_stake≤0.01·reset_stake`; `honest-dominant: honest_given_rating≥0.95 AND
participation≥0.01`; else `mixed`. `train.py` and `train_ablation.py` now call
`collect_basin_report(...)` at end-of-training deterministic eval and write into each per-seed
summary: `action_counts` (A0..A11), `participation_rate`, `honest_given_rating`,
`final_mean_stake`, `reset_stake`, and the resulting `basin` label (plus `stake_obs_mode`).

---

## Acceptance tests — results (`evaluation/tier1a_acceptance.py` → `results/tier1a/acceptance.json`)

Smoke matrix: `ablation_no_stake` (min_stake=1) and `ablation_no_dispute` (min_stake=10000,
dispute_cost=1e6) × 2 seeds × {legacy_normalized, absolute_log}; `config2_mixed` × 2 seeds ×
absolute_log (via the main `train.py` path). 150 episodes each, single-thread, fixed seeds.

The **primary** evidence mirrors the exact Tier 0 property: per-episode behavioral log-sequence
identity between the no_stake and no_dispute runs (robust to a final-policy collapse, which a
150-episode smoke can show). Eval-level `action_counts`/`agent_scores` are corroborating.

### 1A — REGRESSION-BREAK (absolute_log) — **PASS** ✅
Under `absolute_log`, no_stake vs no_dispute behavioral sequences now **differ** on every
behavioral field, both seeds:

```
seed0 behavioral_differing: [honest_pct, mean_score, mean_reputation_accuracy, total_ratings,
                             dishonest_actions, attacks_attempted, attacks_blocked]
seed1 behavioral_differing: [same 7 fields]
```
Eval-level corroboration: `action_counts_identical = False` (both seeds); seed1 lands in a
**different basin** (no_stake → `honest-dominant`, no_dispute → `abstention`) with different
agent scores. The Tier 0 byte-identical property is **broken** — the policy now perceives the
stake regime.

### 1B — LEGACY-PRESERVE (legacy_normalized) — **PASS** ✅
Under `legacy_normalized`, no_stake vs no_dispute behavioral sequences are **byte-identical**
(empty differing set) and only the stake magnitude differs, both seeds:

```
seed0 behavioral_differing: []   stake_fields_differing: [stakes]
seed1 behavioral_differing: []   stake_fields_differing: [stakes]
```
Eval-level: `action_counts_identical = True`, `agent_scores_identical = True`, and
`final_mean_stake` ratio = **10000.0** (exactly the min_stake ratio). The v1 property is
reproduced exactly — the only delta comes from the intended flag; nothing else regressed.

### 1C — OBS SHAPE / BOUNDS — **PASS** ✅
Both modes, both min_stake regimes: obs length **14**, all finite, all within the env's
`[-1, 2]` Box, stake slot within `[0, 1]`.

| mode | min_stake=1 stake slot | min_stake=10000 stake slot |
|---|---|---|
| legacy_normalized | 0.50000 | 0.50000 (invariant — the bug) |
| absolute_log | 0.10154 | 0.91532 (regime-distinguishing — the fix) |

**All three verdicts: `{1A_break: true, 1B_preserve: true, 1C_bounds: true}`.**

---

## Worked basin-label verification (WS3 acceptance)

Manual recompute of the basin from the logged `action_counts` + `final_mean_stake` matches the
logged `basin` for every checked seed:

| run | mode | participation | honest\|rating | final stake / reset | logged basin | manual | match |
|---|---|---|---|---|---|---|---|
| smoke_no_stake_absolute seed1 | absolute_log | 0.3897 | 1.0 | 2.0 / 2.0 | honest-dominant | honest-dominant | ✅ |
| smoke_no_dispute_absolute seed1 | absolute_log | 0.0000 | NaN | 20000 / 20000 | abstention | abstention | ✅ |
| smoke_config2_absolute seed0 (train.py path) | absolute_log | 0.0000 | NaN | 20000 / 20000 | abstention | abstention | ✅ |

The `config2_mixed` row confirms the **main `train.py` path** runs end-to-end and writes the
populated `basin`, `action_counts`, `participation_rate`, `honest_given_rating`,
`final_mean_stake`, `reset_stake`, and `stake_obs_mode` fields into each per-seed summary.

---

## Files changed and new flags

| File | Change | Default | Legacy/v1 path |
|---|---|---|---|
| `env/reputation_engine.py` | `SystemConfig.min_rater_weight/max_rater_weight` → Go-matching | **[0.1, 5.0]** | set `min_rater_weight: 0.5`, `max_rater_weight: 2.0` in YAML |
| `env/reputation_marl_env.py` | new `stake_obs_mode` param + `_stake_obs()` + `STAKE_OBS_MODES`/`STAKE_OBS_SCALE=50000` | **`absolute_log`** | `stake_obs_mode: legacy_normalized` |
| `evaluation/basins.py` | **new** — basin classifier + `collect_basin_report()` (Tier 0 thresholds) | — | — |
| `training/train.py` | `build_system_config()` from YAML; wire `stake_obs_mode`; basin logging | obs `absolute_log`, clamp `[0.1,5.0]` | YAML overrides |
| `training/train_ablation.py` | same wiring; removed hardcoded `[0.5,2.0]` fallbacks | same | YAML overrides |
| `tests/test_reputation_engine.py` | inverted clamp test; legacy-override test; updated clamp bounds | — | — |
| `configs/tier1a/*.yaml` | **new** — 5 smoke configs (no_stake/no_dispute × 2 modes, config2 absolute) | — | — |
| `evaluation/tier1a_acceptance.py` | **new** — acceptance 1A/1B/1C harness | — | — |

**New config flags (all flag-gated, legacy selectable):** `stake_obs_mode`
(`absolute_log` | `legacy_normalized`), `stake_obs_scale` (env param, default 50000), and
SystemConfig overrides now honored by `train.py` too (`min_rater_weight`, `max_rater_weight`,
`decay_rate`, `decay_period`, `min_stake_required`, `slash_percentage`, `dispute_cost`,
`initial_alpha`, `initial_beta`).

**Engine test count: 19 → 20, all passing.** Preserved: no `results/**`, `results/checkpoints/**`,
or `results/derived/**` v1 artifact was modified; all Tier 1A outputs are under `results/tier1a/`.
No git commit/push.

### Note on stake-derived field divergence under legacy (1B)
In this 150-episode smoke, only `stakes` differs between the legacy pair (not `mean_reward`).
Tier 0 saw `mean_reward` also differ in 65/5000 episodes — those differences come from rare
dispute-floor events that did not occur in 150 episodes. The behavioral identity + non-trivial
stake-magnitude difference is exactly the Tier 0 property, so 1B holds.

---

## End-to-end console output

```

########## pytest (engine tests) ##########
....................                                                     [100%]
20 passed in 0.17s

########## acceptance verdicts (evaluation/tier1a_acceptance.py) ##########
  "verdicts": {
    "1A_break": true,
    "1B_preserve": true,
    "1C_bounds": true
  }
}


########## 1A primary evidence (absolute_log: behavioral sequences DIFFER) ##########
 seed 0 differing: ['honest_pct', 'mean_score', 'mean_reputation_accuracy', 'total_ratings', 'dishonest_actions', 'attacks_attempted', 'attacks_blocked']
 seed 1 differing: ['honest_pct', 'mean_score', 'mean_reputation_accuracy', 'total_ratings', 'dishonest_actions', 'attacks_attempted', 'attacks_blocked']
########## 1B primary evidence (legacy: behavioral IDENTICAL, stake differs) ##########
 seed 0 behavioral_differing: [] stake_differing: ['stakes']
 seed 1 behavioral_differing: [] stake_differing: ['stakes']
########## 1B eval (legacy: action_counts & scores identical, stake ratio 10000x) ##########
 seed 0 counts_id= True scores_id= True ratio= 10000.0
 seed 1 counts_id= True scores_id= True ratio= 10000.0

########## smoke matrix summary (per-config honest%) ##########
  DONE: smoke_no_stake_legacy
    Honest %: 1.000 +- 0.000
  DONE: smoke_no_stake_absolute
    Honest %: 1.000 +- 0.000
  DONE: smoke_no_dispute_legacy
    Honest %: 1.000 +- 0.000
  DONE: smoke_no_dispute_absolute
    Honest %: 1.000 +- 0.000
  DONE: smoke_config2_absolute
    Honest %:   0.000 ± 0.000
  ######## SMOKE MATRIX DONE Tue Jun  2 02:36:23 PM MDT 2026 ########
```
