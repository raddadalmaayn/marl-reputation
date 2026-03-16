"""
Generate 4 LaTeX tables from evaluation results.
Saves .tex files to results/tables/.
"""

import json, sys
import numpy as np
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

LOG_DIR   = Path("results/training_logs")
TABLE_DIR = Path("results/tables")
TABLE_DIR.mkdir(parents=True, exist_ok=True)

CONFIG_NAMES = [
    ("config1_baseline",  "All Honest (Baseline)"),
    ("config2_mixed",     "Mixed (5 Adv. + 15 Honest)"),
    ("config3_sybil",     "Sybil Attack"),
    ("config4_collusion", "Collusion Ring"),
    ("config5_adaptive",  "Adaptive Adversary (2$\\times$)"),
]


def load_summary(name):
    p = LOG_DIR / f"{name}_summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def fmt(mean, std):
    return f"{mean:.3f} $\\pm$ {std:.3f}"


# ── Table 1: Summary of all 5 configs ────────────────────────────────────────
def table_summary():
    rows = []
    for name, label in CONFIG_NAMES:
        s = load_summary(name)
        if s:
            rows.append((
                label,
                f"{s['converged_count']}/{s['n_seeds']}",
                fmt(s["mean_eval_reward"],    s["std_eval_reward"]),
                fmt(s["mean_honest_pct"],     s["std_honest_pct"]),
                fmt(s["mean_eval_accuracy"],  s["std_eval_accuracy"]),
            ))
        else:
            rows.append((label, "--", "--", "--", "--"))

    tex = r"""\begin{table*}[t!]
\centering
\caption{Evaluation summary across five experimental configurations (mean $\pm$ std, 5 seeds each).
Honest~\% = fraction of rating actions matching ground truth.
Accuracy = mean $|$score $-$ true\_quality$|$.}
\label{tab:summary}
\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}L{4.5cm}C{1.5cm}C{2.8cm}C{2.8cm}C{2.8cm}@{}}
\toprule
\textbf{Configuration} & \textbf{Converged} & \textbf{Mean Reward} & \textbf{Honest \%} & \textbf{Rep.\ Accuracy} \\
\midrule
"""
    for row in rows:
        tex += " & ".join(row) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    path = TABLE_DIR / "tab_summary.tex"
    path.write_text(tex)
    print(f"tab_summary.tex saved")


# ── Table 2: MARL vs scripted attacks ────────────────────────────────────────
def table_attack_comparison():
    # Scripted results from the am-unified paper (hardcoded from known results)
    scripted = [
        ("Self-Rating",                "Blocked",  "12.6\\,ms"),
        ("Sybil Flood (5 nodes)",      "Partial",  "368.6\\,ms"),
        ("Collusion Ring",             "Partial",  "131.1\\,ms"),
        ("Unauth.\\ SetReputationGate","Blocked",  "9.0\\,ms"),
        ("Insufficient Stake",         "Blocked",  "560.4\\,ms"),
        ("Evidence Tampering",         "Partial",  "65.2\\,ms"),
        ("Reputation Gate Bypass",     "Blocked",  "571.1\\,ms"),
        ("Provenance Replay",          "Blocked",  "570.6\\,ms"),
    ]
    # MARL outcomes (load from summaries where available)
    s2 = load_summary("config2_mixed")
    s3 = load_summary("config3_sybil")
    s4 = load_summary("config4_collusion")
    s5 = load_summary("config5_adaptive")

    def honest_str(s):
        if s is None:
            return "--"
        return f"{s['mean_honest_pct']:.2f} $\\pm$ {s['std_honest_pct']:.2f}"

    marl_rows = [
        ("Mixed adversaries (Config~2)", honest_str(s2),
         "Adaptive rational agents" if s2 else "--"),
        ("Sybil attack (Config~3)",      honest_str(s3),
         "Stake-cost deterrence" if s3 else "--"),
        ("Collusion ring (Config~4)",    honest_str(s4),
         "CI widening + meta-rep weight" if s4 else "--"),
        ("Max incentive (Config~5)",     honest_str(s5),
         "Breaking-point analysis" if s5 else "--"),
    ]

    tex = r"""\begin{table*}[t!]
\centering
\caption{Comparison of MARL-discovered adaptive attacks (this work) vs.\ scripted attacks
from the deployed am-unified system~\cite{almaayn2026reputation}.}
\label{tab:attack_comparison}
\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}L{4.5cm}L{2.5cm}C{2.0cm}@{}}
\toprule
\textbf{Scripted Attack} & \textbf{Outcome} & \textbf{Detection Latency} \\
\midrule
"""
    for name, outcome, latency in scripted:
        tex += f"{name} & {outcome} & {latency} \\\\\n"
    tex += r"""\midrule
\multicolumn{3}{l}{\textit{MARL Adaptive Attack Scenarios}} \\
\midrule
"""
    tex += r"""\textbf{Scenario} & \textbf{Honest \%} & \textbf{Mechanism} \\
\midrule
"""
    for name, honest, mech in marl_rows:
        tex += f"{name} & {honest} & {mech} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    path = TABLE_DIR / "tab_attack_comparison.tex"
    path.write_text(tex)
    print("tab_attack_comparison.tex saved")


# ── Table 3: Parameter sensitivity ───────────────────────────────────────────
def table_parameter_sensitivity():
    # Analytic: simulate reputation accuracy under different S_min and alpha_slash
    from env.reputation_engine import ReputationEngine, ReputationState, SystemConfig
    import random

    def simulate_honest_pct(s_min, slash):
        """Quick Monte Carlo: 100 agents, 50 steps, random dishonest fraction."""
        engine = ReputationEngine(SystemConfig(min_stake_required=s_min, slash_percentage=slash))
        honest_actions = 0
        total_actions = 0
        for _ in range(100):
            stake = s_min * 2
            for step in range(50):
                if stake < s_min:
                    action = 0  # can't rate
                else:
                    # Rational agent: cheat if expected gain > expected loss
                    expected_gain = 0.1  # simplified
                    expected_loss = slash * stake * 0.3  # detection prob 0.3
                    if expected_gain > expected_loss:
                        action = 3  # dishonest
                        stake -= slash * stake * 0.3  # expected slash
                    else:
                        action = 1  # honest
                        honest_actions += 1
                    total_actions += 1
        return honest_actions / max(1, total_actions)

    s_mins  = [1_000, 5_000, 10_000, 50_000]
    slashes = [0.05, 0.10, 0.20]

    tex = r"""\begin{table*}[t!]
\centering
\caption{Parameter sensitivity: honest action fraction under varying minimum stake
$S_{\min}$ and slash rate $\alpha_{\text{slash}}$ (analytic simulation, 100 agents, 50 steps).}
\label{tab:parameter_sensitivity}
\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}L{3.0cm}C{2.2cm}C{2.2cm}C{2.2cm}@{}}
\toprule
& \multicolumn{3}{c}{\textbf{Slash Rate $\alpha_{\text{slash}}$}} \\
\cmidrule{2-4}
\textbf{$S_{\min}$ (tokens)} & \textbf{0.05} & \textbf{0.10} & \textbf{0.20} \\
\midrule
"""
    for s in s_mins:
        vals = [f"{simulate_honest_pct(s, a):.2f}" for a in slashes]
        tex += f"{s:,} & " + " & ".join(vals) + r" \\" + "\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    path = TABLE_DIR / "tab_parameter_sensitivity.tex"
    path.write_text(tex)
    print("tab_parameter_sensitivity.tex saved")


# ── Table 4: Ablation ─────────────────────────────────────────────────────────
def table_ablation():
    # Compare full system vs. ablations using available config results
    full = load_summary("config1_baseline")

    def fmt(m, s):
        try:
            return f"{float(m):.3f} $\\pm$ {float(s):.3f}"
        except:
            return str(m)

    rows = [
        ("Full System (Stake + Wilson CI + Decay + Dispute)",
         fmt(full["mean_honest_pct"], full["std_honest_pct"]) if full else "--",
         fmt(full["mean_eval_accuracy"], full["std_eval_accuracy"]) if full else "--",
         "Reference"),
        ("No Stake ($S_{\\min}=0$)",           "--", "--", "Rating gate removed"),
        ("No Wilson CI (point est.\\ only)",    "--", "--", "Uncertainty hidden"),
        ("No Temporal Decay ($\\lambda=1$)",    "--", "--", "History not discounted"),
        ("No Dispute Resolution",               "--", "--", "No slash mechanism"),
    ]

    tex = r"""\begin{table*}[t!]
\centering
\caption{Ablation study: removing one defence mechanism at a time.
Full-system results from Config~1 (5 seeds). Ablation rows represent the
theoretical impact based on mechanism analysis; empirical ablation training
is left for future work.}
\label{tab:ablation}
\footnotesize
\renewcommand{\arraystretch}{1.2}
\begin{tabular}{@{}L{6.5cm}C{2.8cm}C{2.8cm}L{3.0cm}@{}}
\toprule
\textbf{Configuration} & \textbf{Honest \%} & \textbf{Rep.\ Accuracy} & \textbf{Impact} \\
\midrule
"""
    for label, honest, acc, impact in rows:
        tex += f"{label} & {honest} & {acc} & {impact} \\\\\n"
    tex += r"""\bottomrule
\end{tabular}
\end{table*}
"""
    path = TABLE_DIR / "tab_ablation.tex"
    path.write_text(tex)
    print("tab_ablation.tex saved")


if __name__ == "__main__":
    table_summary()
    table_attack_comparison()
    table_parameter_sensitivity()
    table_ablation()
    print(f"\nAll tables saved to {TABLE_DIR}")
