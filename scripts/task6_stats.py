import json, numpy as np
from pathlib import Path
from scipy import stats

LOG_DIR = Path('/home/raddad/marl-reputation/results/training_logs')
config_names = {
    'config1': 'config1_baseline',
    'config2': 'config2_mixed',
    'config3': 'config3_sybil',
    'config4': 'config4_collusion',
    'config5': 'config5_adaptive',
    'config6': 'config6_self_rating',
    'config7': 'config7_admin_escalation',
    'config8': 'config8_evidence_tamper',
    'config9': 'config9_gate_bypass',
    'config10': 'config10_provenance_replay',
    'config11': 'config11_comprehensive',
}

rows = []
for ckey, cname in config_names.items():
    honest_seeds = []
    for s in range(5):
        lp = LOG_DIR / f'{cname}_seed{s}_log.json'
        if not lp.exists(): continue
        log = json.loads(lp.read_text())
        tail = log[int(len(log)*0.8):]
        honest_seeds.append(np.mean([e['honest_pct'] for e in tail]))

    if len(honest_seeds) < 2: continue

    t_stat, p_val = stats.ttest_1samp(honest_seeds, 0.95)
    # One-tailed: is mean > 0.95?
    p_one_tail = p_val / 2 if t_stat > 0 else 1.0 - p_val / 2
    significant = 'Yes' if p_one_tail < 0.05 and np.mean(honest_seeds) > 0.95 else 'No'
    rows.append({
        'config': cname,
        'n_seeds': len(honest_seeds),
        'honest_mean': float(np.mean(honest_seeds)),
        'honest_std': float(np.std(honest_seeds)),
        't_stat': float(t_stat),
        'p_value': float(p_one_tail),
        'significant_gt_95pct': significant
    })
    print(f"{cname}: honest={np.mean(honest_seeds):.4f}+-{np.std(honest_seeds):.4f} t={t_stat:.3f} p={p_one_tail:.4f} sig={significant}")

Path('/home/raddad/marl-reputation/results/tables').mkdir(exist_ok=True)
lines = [
    r'\begin{table}[t]',
    r'\centering',
    r'\caption{Statistical significance of honest behavior ($>95\%$ threshold, one-sample $t$-test).}',
    r'\label{tab:statistical}',
    r'\footnotesize',
    r'\renewcommand{\arraystretch}{1.2}',
    r'\begin{tabular}{@{}L{2.8cm}C{0.6cm}C{2.0cm}C{1.2cm}C{1.2cm}C{0.8cm}@{}}',
    r'\toprule',
    r'\textbf{Configuration} & \textbf{N} & \textbf{Honest \%} & \textbf{$t$} & \textbf{$p$} & \textbf{Sig.} \\',
    r'\midrule',
]
for r in rows:
    sig = r'Yes$^*$' if r['significant_gt_95pct'] == 'Yes' else 'No'
    config_short = r['config'].replace('_',' ')
    lines.append(f"{config_short} & {r['n_seeds']} & ${r['honest_mean']:.3f}\\pm{r['honest_std']:.3f}$ & ${r['t_stat']:.2f}$ & ${r['p_value']:.3f}$ & {sig} \\\\")
lines += [
    r'\bottomrule',
    r'\multicolumn{6}{l}{\scriptsize $^*p<0.05$, one-tailed test vs.\ honest\%$=95\%$ threshold.}',
    r'\end{tabular}',
    r'\end{table}',
]
Path('/home/raddad/marl-reputation/results/tables/tab_statistical.tex').write_text('\n'.join(lines))
print("\nSaved results/tables/tab_statistical.tex")
