#!/bin/bash
# collect_results.sh — run from the laptop when training is done (or anytime).
# Pulls every machine's training_logs into the laptop's training_logs (merge, no delete).
echo "Collecting results from all lab machines..."
mkdir -p ~/marl-reputation/results/tier1b/sweep/training_logs/
for ip in 10.12.11.48 10.12.10.136 10.12.10.92 10.12.10.126; do
  echo "--- $ip ---"
  rsync -az --info=stats0 \
    iot-lab@$ip:/home/iot-lab/marl-reputation/results/tier1b/sweep/training_logs/ \
    ~/marl-reputation/results/tier1b/sweep/training_logs/
done
RESULTS=$(ls ~/marl-reputation/results/tier1b/sweep/training_logs/*_seed*.json 2>/dev/null | grep -v "_log\.json" | wc -l)
echo
echo "Total shards collected: $RESULTS / 540"
echo "Now run: cd ~/marl-reputation && python3 evaluation/aggregate_sweep.py"
