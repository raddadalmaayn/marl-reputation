#!/usr/bin/env bash
# Extended training run — resumes all unconverged seeds in parallel.
# Wall budget: 10 hours per seed. Extra episodes: 20000 beyond ep10000.
# Run from the marl-reputation directory:
#   nohup bash scripts/run_extended.sh > logs/extended_training.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

WALL_HOURS=10
EXTRA_EP=20000
OUT=results/

LOG_DIR=logs
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Extended MARL training — $(date)"
echo "Wall budget: ${WALL_HOURS}h  Extra episodes: ${EXTRA_EP}"
echo "========================================"

# List of unconverged seeds: "config_yaml seed"
JOBS=(
  "configs/config2.yaml 2"
  "configs/config2.yaml 4"
  "configs/config5.yaml 0"
  "configs/config5.yaml 1"
  "configs/config5.yaml 3"
  "configs/config10.yaml 4"
  "configs/config11.yaml 0"
  "configs/config11.yaml 3"
  "configs/config11.yaml 4"
)

PIDS=()
for JOB in "${JOBS[@]}"; do
  read -r CFG SEED <<< "$JOB"
  CFG_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CFG'))['name'])")
  LOG="$LOG_DIR/resume_${CFG_NAME}_seed${SEED}.log"
  echo "Launching: $CFG  seed=$SEED  -> $LOG"
  python3 training/train_resume.py \
    --config "$CFG" \
    --seed "$SEED" \
    --output "$OUT" \
    --extra-episodes "$EXTRA_EP" \
    --wall-hours "$WALL_HOURS" \
    --rebuild-summary \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} jobs launched (PIDs: ${PIDS[*]})"
echo "Monitor with:  tail -f logs/resume_*.log"
echo "or:            tail -f logs/extended_training.log"
echo ""

# Wait for all seeds to finish
FAILED=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  JOB=${JOBS[$i]}
  if wait "$PID"; then
    echo "OK  [$JOB]  PID=$PID"
  else
    echo "FAIL [$JOB]  PID=$PID  exit=$?"
    FAILED=$((FAILED+1))
  fi
done

echo ""
echo "========================================"
echo "Extended training complete — $(date)"
echo "Failed jobs: $FAILED / ${#PIDS[@]}"
echo "========================================"

# Print final convergence summary
echo ""
echo "=== Final convergence status ==="
python3 - <<'PYEOF'
import json, pathlib
configs = [
    ("config2_mixed",  "configs/config2.yaml"),
    ("config5_adaptive", "configs/config5.yaml"),
    ("config10_provenance_replay", "configs/config10.yaml"),
    ("config11_comprehensive", "configs/config11.yaml"),
]
for name, _ in configs:
    p = pathlib.Path(f"results/training_logs/{name}_summary.json")
    if p.exists():
        d = json.loads(p.read_text())
        print(f"{name}: converged={d['converged_count']}/{d['n_seeds']}  "
              f"reward={d['mean_eval_reward']:.3f}  acc={d['mean_eval_accuracy']:.3f}")
PYEOF
