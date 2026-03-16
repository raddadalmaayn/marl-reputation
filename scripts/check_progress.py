"""Check training progress across all background runs."""
import os, time
from pathlib import Path

checkpoints = Path('/home/raddad/marl-reputation/results')

print("=== Ablation Progress ===")
for abl in ['ablation_no_stake', 'ablation_no_wilson', 'ablation_no_decay', 'ablation_no_dispute']:
    base = checkpoints / 'ablation' / 'checkpoints' / abl
    if not base.exists():
        print(f"  {abl}: not started")
        continue
    for seed_dir in sorted(base.glob('seed*')):
        pts = sorted(seed_dir.glob('ep*.pt'))
        if pts:
            latest = pts[-1].stem  # ep3500 etc
            mod_time = os.path.getmtime(pts[-1])
            age = time.time() - mod_time
            print(f"  {abl}/{seed_dir.name}: latest={latest}, {age:.0f}s ago")

print("\n=== Sensitivity Progress ===")
for cfg_dir in sorted((checkpoints / 'sensitivity' / 'checkpoints').glob('sweep_bonus_*')):
    for seed_dir in sorted(cfg_dir.glob('seed*')):
        pts = sorted(seed_dir.glob('ep*.pt'))
        if pts:
            latest = pts[-1].stem
            mod_time = os.path.getmtime(pts[-1])
            age = time.time() - mod_time
            print(f"  {cfg_dir.name}/{seed_dir.name}: latest={latest}, {age:.0f}s ago")

print("\n=== IPPO Progress ===")
for cfg_dir in sorted((checkpoints / 'ippo' / 'checkpoints').glob('*')):
    for seed_dir in sorted(cfg_dir.glob('seed*')):
        pts = sorted(seed_dir.glob('ep*.pt'))
        if pts:
            latest = pts[-1].stem
            mod_time = os.path.getmtime(pts[-1])
            age = time.time() - mod_time
            print(f"  {cfg_dir.name}/{seed_dir.name}: latest={latest}, {age:.0f}s ago")

print("\n=== Log files ===")
for log_dir in [checkpoints / 'ablation' / 'training_logs', checkpoints / 'sensitivity' / 'training_logs', checkpoints / 'ippo' / 'training_logs']:
    n = len(list(log_dir.glob('*.json'))) if log_dir.exists() else 0
    print(f"  {log_dir.parent.name}/training_logs: {n} json files")
