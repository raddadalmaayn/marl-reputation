#!/bin/bash
# dispatch_desktop_2.sh  —  desktop2-supplier (10.12.10.136), 4 cores
# Generated 2026-06-09 15:47 on zenbook laptop. SELF-CONTAINED: no laptop/network needed once started.
# Assigned shards: 27   Parallelism: -P 3   (~284 MB/shard, ~852 MB peak)
# Idempotent: run_shard.py prints [skip] if output already exists.
set -u
cd /home/iot-lab/marl-reputation
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
mkdir -p results/tier1b/sweep/training_logs
echo "[dispatch_desktop_2] START $(date) — 27 shards @ -P 3 on desktop2-supplier"
cat <<'SHARDS' | xargs -P 3 -L1 bash -c 'python3 evaluation/run_shard.py --config "configs/tier1b/$1.yaml" --seed "$2" --output results/tier1b/sweep' _
sweep_c1_armA 13
sweep_c1_armA 26
sweep_c1_armB 19
sweep_c1_armB 29
sweep_c2_armA 22
sweep_c2_armB 15
sweep_c2_armB 25
sweep_c3_armA 18
sweep_c3_armB 11
sweep_c3_armB 21
sweep_c6_armA 14
sweep_c6_armA 27
sweep_c6_armB 17
sweep_c7_armA 10
sweep_c7_armA 23
sweep_c7_armB 13
sweep_c7_armB 26
sweep_c8_armA 19
sweep_c8_armA 29
sweep_c8_armB 22
sweep_c9_armA 15
sweep_c9_armA 25
sweep_c9_armB 18
sweep_c11_armA 11
sweep_c11_armA 21
sweep_c11_armB 14
sweep_c11_armB 27
SHARDS
echo "[dispatch_desktop_2] ALL ASSIGNED SHARDS PROCESSED $(date)"
