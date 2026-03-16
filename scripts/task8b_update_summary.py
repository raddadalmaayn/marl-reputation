"""Read computed 5-seed summaries and print updated values for tab_summary.tex."""
import json
from pathlib import Path

LOG_DIR = Path('/home/raddad/marl-reputation/results/training_logs')

for name in ['config2_mixed', 'config11_comprehensive']:
    sp = LOG_DIR / f'{name}_summary.json'
    s = json.loads(sp.read_text())
    print(f"\n{name}:")
    print(f"  mean_honest_pct = {s['mean_honest_pct']*100:.2f} +- {s['std_honest_pct']*100:.2f}")
    print(f"  mean_eval_reward = {s.get('mean_eval_reward',0):.3f} +- {s.get('std_eval_reward',0):.3f}")
    print(f"  mean_eval_accuracy = {s.get('mean_eval_accuracy',0):.3f}")
    print(f"  mean_defense_rate = {s['mean_defense_rate']*100:.1f}%")
    print(f"  total_attacks_blocked = {s['total_attacks_blocked']}")
    print(f"  total_attacks_attempted = {s['total_attacks_attempted']}")
    print(f"  converged_count = {s['converged_count']}/5")
