import yaml, subprocess, json, os
from pathlib import Path
import numpy as np

base_dir = Path('/home/raddad/marl-reputation')
Path(base_dir / 'configs/sweeps').mkdir(exist_ok=True)
Path(base_dir / 'results/sensitivity').mkdir(exist_ok=True)

with open(base_dir / 'configs/config2.yaml') as f:
    base = yaml.safe_load(f)

base['max_episodes_extended'] = 3000
base['seeds'] = 2

sweep_configs = []

for bonus in [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]:
    cfg = dict(base)
    cfg['name'] = f'sweep_bonus_{str(bonus).replace(".","p")}'
    cfg['adversarial_reward_bonus'] = bonus
    cfg['description'] = f'Sensitivity sweep: adversarial_reward_bonus={bonus}'
    fname = base_dir / f'configs/sweeps/{cfg["name"]}.yaml'
    with open(fname, 'w') as f:
        yaml.dump(cfg, f)
    sweep_configs.append(str(fname))
    print(f'Created: {fname}')

print(f'\nRunning {len(sweep_configs)} sweep configs sequentially (2 seeds, 3000 episodes each)...')
for cfg_file in sweep_configs:
    cfg_name = Path(cfg_file).stem
    print(f'\nTraining {cfg_name}...')
    result = subprocess.run(
        ['python3', 'training/train_ablation.py', '--config', cfg_file,
         '--output', 'results/sensitivity/', '--seeds', '2'],
        cwd=base_dir,
        capture_output=True, text=True, timeout=7200
    )
    if result.returncode != 0:
        print(f'ERROR: {result.stderr[-500:]}')
    else:
        # Print last few lines of stdout
        lines = result.stdout.strip().split('\n')
        for line in lines[-5:]:
            print(line)

print('\nSensitivity sweeps complete')

# Compile results
Path(base_dir / 'results/tables').mkdir(exist_ok=True)
log_dir = base_dir / 'results/sensitivity/training_logs'
rows = []
for sp in sorted(log_dir.glob('sweep_bonus*_summary.json')):
    s = json.loads(sp.read_text())
    name = s.get('config', sp.stem.replace('_summary', ''))
    # Extract bonus value from name
    try:
        bonus_str = name.split('sweep_bonus_')[1] if 'sweep_bonus_' in name else '0p0'
        bonus = float(bonus_str.replace('p', '.'))
    except:
        bonus = 0.0
    rows.append({
        'bonus': bonus,
        'mean_honest_pct': s.get('mean_honest_pct', 0),
        'std_honest_pct': s.get('std_honest_pct', 0),
        'mean_eval_reward': s.get('mean_eval_reward', 0),
        'mean_eval_accuracy': s.get('mean_eval_accuracy', 0),
    })
    print(f"bonus={bonus}: honest={s.get('mean_honest_pct',0):.4f} reward={s.get('mean_eval_reward',0):.3f}")

rows.sort(key=lambda r: r['bonus'])

# Write CSV
import csv
csv_path = base_dir / 'results/tables/parameter_sensitivity.csv'
with open(csv_path, 'w', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=['bonus', 'mean_honest_pct', 'std_honest_pct', 'mean_eval_reward', 'mean_eval_accuracy'])
    writer.writeheader()
    writer.writerows(rows)
print(f'\nSaved {csv_path}')
