#!/bin/bash
# dispatch_desktop_4.sh  —  desktop4-regulator (10.12.10.126), 12 cores
# Generated 2026-06-09 15:47 on zenbook laptop. SELF-CONTAINED: no laptop/network needed once started.
# Assigned shards: 97   Parallelism: -P 11   (~284 MB/shard, ~3124 MB peak)
# Idempotent: run_shard.py prints [skip] if output already exists.
set -u
cd /home/iot-lab/marl-reputation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/tier1b/sweep/training_logs
echo "[dispatch_desktop_4] START $(date) — 97 shards @ -P 11 on desktop4-regulator"
cat <<'SHARDS' | xargs -P 11 -L1 bash -c 'python3 evaluation/run_shard.py --config "configs/tier1b/$1.yaml" --seed "$2" --output results/tier1b/sweep' _
sweep_c1_armA 12
sweep_c1_armA 16
sweep_c1_armA 19
sweep_c1_armA 22
sweep_c1_armA 25
sweep_c1_armA 29
sweep_c1_armB 12
sweep_c1_armB 15
sweep_c1_armB 18
sweep_c1_armB 22
sweep_c1_armB 25
sweep_c1_armB 28
sweep_c2_armA 12
sweep_c2_armA 15
sweep_c2_armA 18
sweep_c2_armA 21
sweep_c2_armA 25
sweep_c2_armA 28
sweep_c2_armB 11
sweep_c2_armB 14
sweep_c2_armB 18
sweep_c2_armB 21
sweep_c2_armB 24
sweep_c2_armB 28
sweep_c3_armA 11
sweep_c3_armA 14
sweep_c3_armA 17
sweep_c3_armA 21
sweep_c3_armA 24
sweep_c3_armA 27
sweep_c3_armB 10
sweep_c3_armB 14
sweep_c3_armB 17
sweep_c3_armB 20
sweep_c3_armB 24
sweep_c3_armB 27
sweep_c6_armA 10
sweep_c6_armA 13
sweep_c6_armA 17
sweep_c6_armA 20
sweep_c6_armA 23
sweep_c6_armA 26
sweep_c6_armB 10
sweep_c6_armB 13
sweep_c6_armB 16
sweep_c6_armB 20
sweep_c6_armB 23
sweep_c6_armB 26
sweep_c6_armB 29
sweep_c7_armA 13
sweep_c7_armA 16
sweep_c7_armA 19
sweep_c7_armA 22
sweep_c7_armA 26
sweep_c7_armA 29
sweep_c7_armB 12
sweep_c7_armB 16
sweep_c7_armB 19
sweep_c7_armB 22
sweep_c7_armB 25
sweep_c7_armB 29
sweep_c8_armA 12
sweep_c8_armA 15
sweep_c8_armA 18
sweep_c8_armA 22
sweep_c8_armA 25
sweep_c8_armA 28
sweep_c8_armB 12
sweep_c8_armB 15
sweep_c8_armB 18
sweep_c8_armB 21
sweep_c8_armB 25
sweep_c8_armB 28
sweep_c9_armA 11
sweep_c9_armA 14
sweep_c9_armA 18
sweep_c9_armA 21
sweep_c9_armA 24
sweep_c9_armA 28
sweep_c9_armB 11
sweep_c9_armB 14
sweep_c9_armB 17
sweep_c9_armB 21
sweep_c9_armB 24
sweep_c9_armB 27
sweep_c11_armA 10
sweep_c11_armA 14
sweep_c11_armA 17
sweep_c11_armA 20
sweep_c11_armA 24
sweep_c11_armA 27
sweep_c11_armB 10
sweep_c11_armB 13
sweep_c11_armB 17
sweep_c11_armB 20
sweep_c11_armB 23
sweep_c11_armB 26
SHARDS
echo "[dispatch_desktop_4] ALL ASSIGNED SHARDS PROCESSED $(date)"
