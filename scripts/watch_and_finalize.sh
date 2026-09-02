#!/usr/bin/env bash
# Watch for training completion and regenerate figures + update paper
cd /home/raddad/marl-reputation

LOG=logs/watch_finalize.log
echo "Watch started $(date)" > $LOG

check_and_update() {
    local updated=0

    # Check if sens_1p0 seed0 is done (need partial summary)
    if [ -f results/sensitivity/training_logs/sweep_bonus_1p0_seed0.json ] && \
       [ ! -f results/sensitivity/training_logs/sweep_bonus_1p0_summary.json ]; then
        python3 -c "
import json, numpy as np
from pathlib import Path
files = list(Path('results/sensitivity/training_logs').glob('sweep_bonus_1p0_seed*.json'))
files = [f for f in files if '_log' not in f.name]
if files:
    data = [json.loads(f.read_text()) for f in files]
    s = {'config':'sweep_bonus_1p0','n_seeds':len(data),'converged_count':sum(d.get('converged',False) for d in data),
         'mean_honest_pct':float(np.mean([d['eval_mean_honest_pct'] for d in data])),
         'std_honest_pct':float(np.std([d['eval_mean_honest_pct'] for d in data])),
         'mean_eval_reward':float(np.mean([d['eval_mean_reward'] for d in data])),
         'std_eval_reward':float(np.std([d['eval_mean_reward'] for d in data]))}
    Path('results/sensitivity/training_logs/sweep_bonus_1p0_summary.json').write_text(json.dumps(s,indent=2))
    print('Created sweep_bonus_1p0 partial summary')
" >> $LOG 2>&1
        updated=1
    fi

    # Regenerate figures if we have new data
    if [ $updated -eq 1 ]; then
        python3 scripts/task7_figures.py >> $LOG 2>&1
        python3 - << 'PYEOF' >> $LOG 2>&1
import shutil
from pathlib import Path
for f in Path('results/figures').iterdir():
    if f.suffix in ('.png', '.pdf'):
        shutil.copy2(f, Path('marl-paper/figures') / f.name)
print("Figures copied")
PYEOF
        git add marl-paper/figures/ results/sensitivity/training_logs/sweep_bonus_1p0_summary.json
        git commit -m "Update sensitivity figure with b=1.0 data

Co-Authored-By: Claude Sonnet 4.6 <noreply@anthropic.com>" >> $LOG 2>&1
        git push origin main >> $LOG 2>&1
        echo "Pushed b=1.0 update $(date)" >> $LOG
    fi
}

# Poll every 5 minutes for up to 8 hours
for i in $(seq 1 96); do
    sleep 300
    check_and_update
done

echo "Watch ended $(date)" >> $LOG
