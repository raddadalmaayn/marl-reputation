# Tier 1B — Stage 1b Full Two-Arm Sweep: RUN PLAN

Recommended participation coef (Arm B) **c = 1.2771** (Stage 0).

## Shard budget
| Config set | configs | seeds/arm | arms | shards |
|---|---|---|---|---|
| bimodal/key | 8 (C1,C2,C3,C6,C7,C8,C9,C11) | 30 | 2 | 480 |
| stable unimodal | 3 (C4,C5,C10) | 10 | 2 | 60 |
| **total** | 11 | — | 2 | **540** |

## Per-host shard map (weighted round-robin by concurrency)
| Host | concurrency | shards | est. wall-clock |
|---|---|---|---|
| strong | 16 | 216 | ~20.2 h |
| desktop1 | 6 | 81 | ~20.2 h |
| desktop2 | 6 | 81 | ~20.2 h |
| desktop3 | 6 | 81 | ~20.2 h |
| desktop4 | 6 | 81 | ~20.2 h |

**Critical-path wall-clock ≈ 20.2 h** (assumes ~1.5 h/shard avg for the 5000-ep + extension schedule; Arm B converges faster, Arm A may run longer — treat as a rough upper bound; measure on the first wave and re-estimate).

## Launch
- Strong host (run locally): `bash results/tier1b/launch/run_strong.sh`
- Each desktop: copy the repo (or shared FS), then `bash results/tier1b/launch/run_<host>.sh`.
- Shards are idempotent (`run_shard.py` skips existing summaries), so a host can be re-run after interruption without redoing finished seeds.
- Collect: rsync each host's `results/tier1b/sweep/training_logs/` back, then `python3 evaluation/aggregate_arms.py --dir results/tier1b/sweep --out results/tier1b/sweep/arm_summary`.

## Manifest
Full `(host, config, arm, seed, command)` table: `results/tier1b/RUN_MANIFEST.tsv` (one shard per row).

---

## Tier 1B FULL SWEEP — revised plan (rho + baselines folded in)

**Shard count unchanged at 540 training shards** (270/arm; bimodal/key configs 30
seeds, stable configs 10 seeds). The two corrections do NOT add training shards:
- **rho** is logged inside every existing shard (`with_rho=True`) — no new shards.
- **baselines** are an EVALUATION pass over the {C2,C3,C11} Beta-trained checkpoints
  (`evaluation/baseline_eval.py`), re-scoring the same learned-adversary eval stream
  under {beta, flat, eigentrust}. 6 config-arms × 30 seeds = up to 180 baseline-eval
  jobs, each a few minutes (no training). Run after the corresponding sweep shards.

**Local execution (this session):** the strong host runs a **coverage-first** order
(seed-major: seed 0 of all 22 config-arms, then seed 1, …) so that partial completion
yields full config coverage rather than a few configs at all seeds. Command list:
`/tmp/sweep_coverage_order.txt`; launched at 16-way concurrency to
`results/tier1b/sweep/`.

**Per-shard wall-clock** ≈ 1.8–2.5 h (5000-ep floor + extension; Arm B converges
faster). 540 shards on one 20-core host ≈ multi-day; this is why the run is
distributed. **Dispatch for the 4 desktops:** `results/tier1b/launch/run_desktop{1..4}.sh`
(or `python3 evaluation/sweep_ledger.py --host desktopN --emit-missing`). Idempotent:
`run_shard.py` skips any seed whose summary already exists, so hosts can resume.

**Completion tracking:** `python3 evaluation/sweep_ledger.py` (done/missing per host
and per config-arm); `--json results/tier1b/sweep/ledger.json` for the report.
