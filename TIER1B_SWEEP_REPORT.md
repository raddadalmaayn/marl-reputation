# Tier 1B Full Sweep — Report

Final, multi-seed, two-arm evaluation on the **corrected** environment
(`stake_obs_mode="absolute_log"`, rater clamp `[0.1, 5.0]` — confirmed live before
launch). Arm A `participation_coef=0.0`; Arm B `participation_coef=1.277` (the
Stage-0-derived `c`, FIXED — not re-tuned). All outputs under `results/tier1b/sweep/`;
prior artifacts (results/**, derived/**, tier1a/**, tier1b/pilot/**) untouched. No git commit.

## Corrections applied (the reason for re-touching the built sweep)
- **A — ρ logging ON for every shard.** `train.py` → `collect_basin_report(with_rho=True)`;
  `system_spearman_rho` (system reputation score vs true latent quality) is recorded
  per seed, both arms, every config. This is the discrimination-recovery half of the thesis.
- **B — baselines folded in, apples-to-apples.** `evaluation/baseline_eval.py` re-scores
  the SAME Beta-trained learned-adversary eval stream under {beta, flat, eigentrust}
  (Flat online/in-loop; EigenTrust post-hoc/global — asymmetry preserved). Same seeds,
  same trained policies, estimator swapped only — no double-training. Enabled by the
  opt-in env flag `record_rating_events` (default off; behavior verified unchanged).

## Pre-flight gate — see `PREFLIGHT.md` (ALL 3 PASS)
1. Resolved Arm-B config: `participation_coef==1.277`, `absolute_log`, clamp `[0.1,5.0]`,
   `with_rho=True` ✓.
2. C2 both arms, 1 seed, 400 ep: ρ present & non-NaN (A 0.933, B 0.940) ✓.
3. C2 baseline eval: readable ρ for all three engines (beta 0.936, flat 0.940,
   eigentrust 0.671) ✓.

## Run design
- 540 training shards (270/arm; 30 seeds for {C1,C2,C3,C6,C7,C8,C9,C11}, 10 for
  {C4,C5,C10}); `results/tier1b/RUN_MANIFEST.tsv`. ρ + baselines add NO training shards.
- **Local (this session):** strong host runs a **coverage-first** order (seed-major)
  at 16-way concurrency → `results/tier1b/sweep/`. Per-shard ≈ 1.8–2.5 h (5000-ep floor
  + extension). 540 shards on one host is multi-day → the run is distributed.
- **Dispatch (4 desktops):** `results/tier1b/launch/run_desktop{1..4}.sh`, or
  `python3 evaluation/sweep_ledger.py --host desktopN --emit-missing`. Idempotent/resumable.
- **Completion ledger:** `python3 evaluation/sweep_ledger.py` →
  `results/tier1b/sweep/ledger.json`.

---

## What ran vs what is missing (ledger)

The full sweep is a **multi-day distributed job**: 540 shards × ~2.8 h/shard
(measured: e.g. `sweep_c5_armB_seed0` converged at ep 5548 in 10067 s ≈ 2.8 h;
~1.8 s/ep under 16-way contention). On this single 20-core host it cannot finish
in one session. The run was launched coverage-first and **continues in the
background**; `results/tier1b/sweep/ledger.json` is the live truth.

**Completed at time of writing: 5 / 540 shards** (all seed 0):
`c1_armA, c4_armA, c6_armA, c10_armA, c5_armB`. 18 shards training; the rest queued.
**Matched A/B pairs so far: 0** (Arm B shards run ~50–90 min longer than Arm A
because full participation = every agent rates every step → more `apply_rating`
calls + a heavier ρ-eval), so the per-config arm-delta cannot yet be computed from
the sweep. **No numbers are imputed for missing shards.**

Resume / finish: idempotent — `run_shard.py` skips completed seeds. Strong host
continues automatically; desktops dispatch via
`results/tier1b/launch/run_desktop{1..4}.sh` or
`python3 evaluation/sweep_ledger.py --host desktopN --emit-missing`. Tables refresh
by re-running `evaluation/aggregate_sweep.py` (+ `run_baselines.py`).

## Table 1 — Per-config, per-arm summary (COMPLETED shards only, n = seeds done)

| Config | arm | n | particip. | cond honest | ρ (mean±sd) | P(hon-dom) [Jeffreys95] | hold rate |
|---|---|---|---|---|---|---|---|
| c1  | A | 1 | 0.159 | 1.000 | −0.752 | 1.000 [0.147,1.000] | — (no adv) |
| c4  | A | 1 | 1.000 | 1.000 | −0.923 | 1.000 [0.147,1.000] | 0.357 |
| c5  | B | 1 | 1.000 | 1.000 | +0.496 | 1.000 [0.147,1.000] | 0.793 |
| c6  | A | 1 | 0.995 | 1.000 | +0.935 | 1.000 [0.147,1.000] | 1.000 |
| c10 | A | 1 | 0.000 | NaN | NaN (abstention) | 0.000 [0.000,0.853] | 0.153 |

These are **single-seed, not settled** — they are shown only to demonstrate the
sweep produces valid final-table rows. They already echo the pilot's qualitative
pattern (ρ wildly seed-dependent and frequently negative even at the full 5000-ep
schedule: c4_A −0.923, c1_A −0.752, c6_A +0.935), reinforcing the
discrimination-not-recovered finding — but settling it needs the 30-seed budget.

## Tables 2–4 — PENDING (require matched pairs / {C2,C3,C11} checkpoints)
- **Table 2 (arm-delta)**: 0 matched A/B pairs complete → not computable from the
  sweep yet. Auto-fills once both arms of any config×seed finish.
- **Table 3 (security cost)**: same dependency (needs both arms).
- **Table 4 (baseline-under-attack)**: 0/180 baseline-eval jobs ready —
  {C2,C3,C11} sweep checkpoints not finished yet (they are among the slowest:
  adversarial + attacks + collusion). `run_baselines.py` will execute them
  automatically as the checkpoints appear.

## Interim reference — the validated PILOT deltas (NOT the sweep; 5 seeds, 1000 ep)
Until the 30-seed sweep settles, the **completed pilot** (`results/tier1b/pilot/`,
reported in `TIER1B_REPORT.md`) is the best available estimate of the headline. It is
included here as context only — it is NOT the sweep and is not mixed into the sweep
tables:
- Δparticipation: C1 **+0.919 [0.694,1.144]**, C7 **+0.990 [0.963,1.018]** (CIs exclude 0).
- ΔP(honest-dominant): C1 **+0.800 [0.193,0.964]**, C7 **+1.000 [0.386,1.000]** (Newcombe).
- Δρ: all CIs straddle 0 (C1 +0.86, C2 +0.10, C11 +0.04); Arm-B mean ρ stays
  near-random/negative (C2 −0.37, C7 0.02, C11 0.11) — **ρ does NOT leave the
  near-random range** even at full honest participation.
- Security: **no cost** — attacks *fall* under Arm B (C2 8956→317 attempts; hold
  0.18→0.90), adversaries prefer honest rating.
The early sweep single-seed ρ values (above) are consistent with this.

## Data gaps
1. **Sweep incomplete: 5/540** (one session, one host; ~2.8 h/shard). Most shards
   pending; all tracked in `ledger.json`, none imputed.
2. **No matched A/B pairs yet** ⇒ sweep arm-delta / security tables empty; will fill
   automatically as Arm B catches up.
3. **Baselines pending** ({C2,C3,C11} checkpoints not done); harness validated in
   pre-flight (β/flat/eigentrust all readable).
4. The sweep is designed to be finished on the 5-host testbed (~20 h) via the
   dispatch scripts; this report will be regenerated by re-running the two
   aggregators once shards land.

