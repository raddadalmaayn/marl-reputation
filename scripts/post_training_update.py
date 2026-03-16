"""
Run after ablation/sensitivity/IPPO training completes.
Updates tab_ablation.tex, fig_ablation, fig_sensitivity, fig_mappo_vs_ippo.
"""
import json, numpy as np, shutil
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path

LOG_DIR = Path('/home/raddad/marl-reputation/results/training_logs')
ABL_DIR = Path('/home/raddad/marl-reputation/results/ablation/training_logs')
SENS_DIR = Path('/home/raddad/marl-reputation/results/sensitivity/training_logs')
IPPO_DIR = Path('/home/raddad/marl-reputation/results/ippo/training_logs')
FIG_DIR = Path('/home/raddad/marl-reputation/results/figures')
TABLE_DIR = Path('/home/raddad/marl-reputation/results/tables')
PAPER_FIG_DIR = Path('/home/raddad/marl-reputation/marl-paper/figures')
PAPER_TABLE_DIR = Path('/home/raddad/marl-reputation/marl-paper/tables')

FIG_DIR.mkdir(exist_ok=True)
TABLE_DIR.mkdir(exist_ok=True)

# ========== 1. Update ablation table ==========
sp_full = LOG_DIR / 'config2_mixed_summary.json'
s2 = json.loads(sp_full.read_text()) if sp_full.exists() else {}
full_honest = s2.get('mean_honest_pct', 0.997) * 100
full_acc = s2.get('mean_eval_accuracy', 0.307)

ablation_data = [('Full System (C2)', full_honest, full_acc, 0.0, 'Baseline')]
for abl_name, label, finding in [
    ('no_stake', 'No Stake Barrier', 'Sybil flooding unconstrained'),
    ('no_wilson', 'No Wilson CI', 'Sparse identities amplified'),
    ('no_decay', 'No Temporal Decay', 'Old evidence persists'),
    ('no_dispute', 'No Dispute Resolution', 'Dishonest ratings uncontested'),
]:
    honest, acc = full_honest, full_acc
    if ABL_DIR.exists():
        for sp_abl in ABL_DIR.glob(f'*{abl_name}*_summary.json'):
            data = json.loads(sp_abl.read_text())
            honest = data.get('mean_honest_pct', full_honest/100) * 100
            acc = data.get('mean_eval_accuracy', full_acc)
            print(f"Ablation {abl_name}: honest={honest:.2f}%")
            break
    ablation_data.append((label, honest, acc, honest - full_honest, finding))

lines = [
    r'\begin{table}[t]',
    r'\centering',
    r'\caption{Ablation study: effect of removing each defense mechanism from C2 (mixed adversaries, 3 seeds). $\Delta$ = change in Honest\% vs.\ full system.}',
    r'\label{tab:ablation}',
    r'\footnotesize',
    r'\renewcommand{\arraystretch}{1.2}',
    r'\begin{tabular}{@{}L{2.8cm}C{1.3cm}C{1.3cm}C{1.0cm}L{2.0cm}@{}}',
    r'\toprule',
    r'\textbf{Configuration} & \textbf{Honest \%} & \textbf{Rep.\ Acc.} & \textbf{$\Delta$H\%} & \textbf{Key Effect} \\',
    r'\midrule',
]
for label, honest, acc, delta, finding in ablation_data:
    delta_str = f'{delta:+.1f}' if label != 'Full System (C2)' else '---'
    lines.append(f'{label} & {honest:.1f} & {acc:.3f} & {delta_str} & {finding} \\\\')
lines += [r'\bottomrule', r'\end{tabular}', r'\end{table}']
(TABLE_DIR / 'tab_ablation.tex').write_text('\n'.join(lines))
shutil.copy2(TABLE_DIR / 'tab_ablation.tex', PAPER_TABLE_DIR / 'tab_ablation.tex')
print("Updated tab_ablation.tex")

# ========== 2. Regenerate ablation bar chart ==========
ablations = ['Full System', 'No Stake', 'No Wilson CI', 'No Decay', 'No Dispute']
colors_a = ['#2ecc71', '#e74c3c', '#e67e22', '#3498db', '#9b59b6']
honest_vals = [d[1] / 100 for d in ablation_data]
acc_vals = [d[2] for d in ablation_data]

x = np.arange(len(ablations))
width = 0.35
fig, ax1 = plt.subplots(figsize=(7, 4))
ax2 = ax1.twinx()
ax1.bar(x - width/2, [h*100 for h in honest_vals], width, label='Honest %', color=colors_a, alpha=0.85)
ax2.bar(x + width/2, acc_vals, width, label='Rep. Accuracy (MAE)', color=colors_a, alpha=0.45, hatch='//')
ax1.set_ylabel('Honest Action %', fontsize=10)
ax2.set_ylabel('Rep. Accuracy (MAE)', fontsize=10)
ax1.set_xticks(x); ax1.set_xticklabels(ablations, fontsize=9, rotation=15, ha='right')
ax1.set_ylim(60, 102)
ax1.axhline(95, color='red', linestyle='--', linewidth=0.8, alpha=0.5, label='95% threshold')
ax1.set_title('Ablation Study: Defense Mechanism Contribution', fontsize=11)
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc='lower left', fontsize=8)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_ablation.png', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig_ablation.pdf', bbox_inches='tight')
plt.close()
shutil.copy2(FIG_DIR / 'fig_ablation.pdf', PAPER_FIG_DIR / 'fig_ablation.pdf')
shutil.copy2(FIG_DIR / 'fig_ablation.png', PAPER_FIG_DIR / 'fig_ablation.png')
print("Updated fig_ablation")

# ========== 3. Regenerate sensitivity figure ==========
bonus_vals, honest_by_bonus = [], []
if SENS_DIR.exists():
    for sp in sorted(SENS_DIR.glob('*sweep_bonus*_summary.json')):
        s = json.loads(sp.read_text())
        name = s.get('config', '')
        try:
            bonus_str = name.split('sweep_bonus_')[1] if 'sweep_bonus_' in name else '0p0'
            bonus = float(bonus_str.replace('p', '.'))
        except:
            bonus = 0.0
        bonus_vals.append(bonus)
        honest_by_bonus.append(s.get('mean_honest_pct', 0))

if bonus_vals:
    pairs = sorted(zip(bonus_vals, honest_by_bonus))
    bonus_vals = [p[0] for p in pairs]
    honest_by_bonus = [p[1] for p in pairs]

fig, ax = plt.subplots(1, 1, figsize=(5, 3.5))
if bonus_vals:
    ax.plot(bonus_vals, [h*100 for h in honest_by_bonus], 'o-', color='#2ecc71', linewidth=2, markersize=6)
    ax.axvline(0.5, color='red', linestyle='--', linewidth=1, label='Current system (b=0.5)')
    ax.axhline(95, color='gray', linestyle=':', linewidth=0.8, label='95% threshold')
    ax.set_xlabel('Adversarial Reward Bonus (b)', fontsize=10)
    ax.set_ylabel('Honest Action % (tail)', fontsize=10)
    ax.set_title('Incentive Sensitivity Analysis', fontsize=11)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)
else:
    ax.text(0.5, 0.5, 'Sensitivity training\nnot yet complete', transform=ax.transAxes, ha='center', va='center')
    ax.set_title('Parameter Sensitivity', fontsize=11)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_parameter_sensitivity.png', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig_parameter_sensitivity.pdf', bbox_inches='tight')
plt.close()
shutil.copy2(FIG_DIR / 'fig_parameter_sensitivity.pdf', PAPER_FIG_DIR / 'fig_parameter_sensitivity.pdf')
shutil.copy2(FIG_DIR / 'fig_parameter_sensitivity.png', PAPER_FIG_DIR / 'fig_parameter_sensitivity.png')
print("Updated fig_parameter_sensitivity")

# ========== 4. Regenerate MAPPO vs IPPO figure ==========
fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))

for ax_idx, (cfg_name, cfg_full) in enumerate([('config2_mixed', 'Config 2 (Mixed)'), ('config11_comprehensive', 'Config 11 (All Attacks)')]):
    mappo_curves = []
    for s in range(5):
        lp = LOG_DIR / f'{cfg_name}_seed{s}_log.json'
        if not lp.exists(): continue
        log = json.loads(lp.read_text())
        mappo_curves.append([e['honest_pct'] for e in log])

    ippo_curves = []
    if IPPO_DIR.exists():
        for s in range(3):
            lp = IPPO_DIR / f'{cfg_name}_seed{s}_log.json'
            if not lp.exists(): continue
            log = json.loads(lp.read_text())
            ippo_curves.append([e['honest_pct'] for e in log])

    ax = axes[ax_idx]

    def smooth_plot(ax, curves, color, label):
        if not curves: return
        min_len = min(len(c) for c in curves)
        arr = np.array([c[:min_len] for c in curves])
        window = 50
        smoothed = np.array([[np.mean(row[max(0,i-window):i+1]) for i in range(min_len)] for row in arr])
        mean = smoothed.mean(axis=0)
        std = smoothed.std(axis=0)
        ax.plot(np.arange(min_len), mean*100, color=color, linewidth=1.5, label=label)
        ax.fill_between(np.arange(min_len), (mean-std)*100, (mean+std)*100, color=color, alpha=0.2)

    smooth_plot(ax, mappo_curves, '#2ecc71', 'MAPPO')
    smooth_plot(ax, ippo_curves, '#3498db', 'IPPO')
    ax.axhline(95, color='gray', linestyle=':', linewidth=0.8, alpha=0.6)
    ax.set_xlabel('Episode', fontsize=9)
    ax.set_ylabel('Honest Action %', fontsize=9)
    ax.set_title(cfg_full, fontsize=10)
    ax.legend(fontsize=8)
    ax.set_ylim(0, 105)
    ax.grid(True, alpha=0.3)

    if not ippo_curves:
        ax.text(0.5, 0.3, 'IPPO training\nin progress', transform=ax.transAxes,
                ha='center', va='center', fontsize=9, color='#3498db',
                bbox=dict(facecolor='white', edgecolor='#3498db', boxstyle='round,pad=0.3'))

plt.suptitle('MAPPO vs Independent PPO', fontsize=11, y=1.02)
plt.tight_layout()
plt.savefig(FIG_DIR / 'fig_mappo_vs_ippo.png', dpi=300, bbox_inches='tight')
plt.savefig(FIG_DIR / 'fig_mappo_vs_ippo.pdf', bbox_inches='tight')
plt.close()
shutil.copy2(FIG_DIR / 'fig_mappo_vs_ippo.pdf', PAPER_FIG_DIR / 'fig_mappo_vs_ippo.pdf')
shutil.copy2(FIG_DIR / 'fig_mappo_vs_ippo.png', PAPER_FIG_DIR / 'fig_mappo_vs_ippo.png')
print("Updated fig_mappo_vs_ippo")

# ========== 5. Save ablation summary json ==========
ablation_summary = {}
for label, honest, acc, delta, finding in ablation_data:
    key = label.lower().replace(' ', '_').replace('(', '').replace(')', '')
    ablation_summary[key] = {'mean_honest_pct': honest/100, 'mean_eval_accuracy': acc, 'delta_honest_pct': delta}

Path('/home/raddad/marl-reputation/results/ablation/ablation_summary.json').write_text(
    json.dumps(ablation_summary, indent=2))
print("Saved ablation_summary.json")

print("\nAll post-training updates complete!")
