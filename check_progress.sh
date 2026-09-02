#!/bin/bash
# check_progress.sh — run from the laptop to see sweep status on the 4 lab machines.
# Each machine starts with the 220 synced baseline shards and only WRITES its own
# newly-assigned ones, so we report "new" = (files now) - 220 to avoid double-counting.
BASELINE=220
echo "=== SWEEP PROGRESS $(date) ==="
declare -A IPS=( [1]=10.12.11.48 [2]=10.12.10.136 [3]=10.12.10.92 [4]=10.12.10.126 )
declare -A NAME=( [1]=manufacturer [2]=supplier [3]=logistics [4]=regulator )
declare -A ASSIGNED=( [1]=98 [2]=27 [3]=98 [4]=97 )
grand_new=0; reachable=0
for i in 1 2 3 4; do
  ip=${IPS[$i]}
  out=$(ssh -o ConnectTimeout=5 iot-lab@$ip '
    cd ~/marl-reputation 2>/dev/null || exit 9
    procs=$(pgrep -cf "[r]un_shard.py --config")
    done=$(ls results/tier1b/sweep/training_logs/*_seed*.json 2>/dev/null | grep -v "_log\.json" | wc -l)
    latest=$(ls -t results/tier1b/sweep/training_logs/*_seed*.json 2>/dev/null | grep -v "_log\.json" | head -1 | xargs -r basename)
    disp=$(pgrep -cf "[d]ispatch_desktop")
    echo "$procs|$done|$latest|$disp"
  ' 2>/dev/null)
  echo "--- desktop$i (${NAME[$i]}, $ip) ---"
  if [ -z "$out" ]; then echo "  UNREACHABLE"; continue; fi
  reachable=$((reachable+1))
  IFS="|" read procs done latest disp <<< "$out"
  new=$(( done - BASELINE )); [ "$new" -lt 0 ] && new=0
  grand_new=$(( grand_new + new ))
  echo "  workers: ${procs:-0} running   dispatch alive: ${disp:-0}"
  echo "  finished here: $new / ${ASSIGNED[$i]} assigned   latest: ${latest:-none}"
done
echo
echo "=== TOTAL: $(( BASELINE + grand_new )) / 540   (220 baseline + $grand_new new from $reachable/4 machines) ==="
echo "When done, run ./collect_results.sh to pull everything to the laptop."
