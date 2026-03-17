#!/usr/bin/env bash
# Launch all training for paper fixes:
#   Item 1 — sensitivity sweep b={0.1,0.25,0.5,1.0,2.0,4.0} 3 seeds each
#   Item 2 — IPPO training config2+config11 seeds 1+2
#   Item 6 — terminal reward config2_terminal + config11_terminal 3 seeds each
#
# Run from marl-reputation directory:
#   nohup bash scripts/run_items_1_2_6.sh > logs/items_1_2_6.log 2>&1 &

set -euo pipefail
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

LOG_DIR=logs
mkdir -p "$LOG_DIR"

echo "========================================"
echo "Items 1/2/6 training launch — $(date)"
echo "========================================"

PIDS=()

# ---------- ITEM 1: Sensitivity sweep (3 seeds each, all 6 bonus values) ----------
SENS_BONUSES=(0p1 0p25 0p5 1p0 2p0 4p0)
for B in "${SENS_BONUSES[@]}"; do
  LOG="$LOG_DIR/sens_bonus_${B}.log"
  echo "Sensitivity sweep bonus=${B} → $LOG"
  python3 training/train.py \
    --config "configs/sweeps/sweep_bonus_${B}.yaml" \
    --output results/sensitivity/ \
    --seeds 3 \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

# ---------- ITEM 2: IPPO seeds 1 and 2 for config2 and config11 ----------
# Need to run seeds 1 and 2 (seed 0 already exists)
# train_ippo.py runs seeds 0..N-1; running with --seeds 3 retrains all 3
# For clean comparison we run all 3 seeds fresh
for CFG in config2 config11; do
  CFG_FILE="configs/${CFG}.yaml"
  LOG="$LOG_DIR/ippo_${CFG}.log"
  echo "IPPO ${CFG} seeds 0-2 → $LOG"
  python3 training/train_ippo.py \
    --config "$CFG_FILE" \
    --output results/ippo/ \
    --seeds 3 \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

# ---------- ITEM 6: Terminal reward configs 3 seeds each ----------
for CFG in config2_terminal config11_terminal; do
  LOG="$LOG_DIR/terminal_${CFG}.log"
  echo "Terminal reward ${CFG} → $LOG"
  python3 training/train.py \
    --config "configs/${CFG}.yaml" \
    --output results/terminal/ \
    --seeds 3 \
    > "$LOG" 2>&1 &
  PIDS+=($!)
done

echo ""
echo "All ${#PIDS[@]} jobs launched (PIDs: ${PIDS[*]})"
echo "Monitor: tail -f logs/sens_bonus_*.log logs/ippo_*.log logs/terminal_*.log"
echo ""

FAILED=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  if wait "$PID"; then
    echo "OK  PID=$PID"
  else
    echo "FAIL PID=$PID exit=$?"
    FAILED=$((FAILED+1))
  fi
done

echo "========================================"
echo "All jobs done — $(date)  Failed: $FAILED/${#PIDS[@]}"
echo "========================================"

# Generate all updated figures once training is done
echo "Generating figures..."
python3 scripts/task7_figures.py > "$LOG_DIR/figures.log" 2>&1
echo "Figures done. See $LOG_DIR/figures.log"

# Copy figures to paper dir
python3 - <<'PYEOF'
import shutil
from pathlib import Path
src = Path("results/figures")
dst = Path("marl-paper/figures")
dst.mkdir(exist_ok=True)
for f in src.iterdir():
    if f.suffix in ('.png', '.pdf'):
        shutil.copy2(f, dst / f.name)
print(f"Copied {len(list(src.iterdir()))} figures to {dst}")
PYEOF
