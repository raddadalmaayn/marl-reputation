"""
Generate all 8 publication-quality figures from training logs.
Saves 300dpi PNG to results/figures/.
"""

import json, sys, numpy as np, matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from env.reputation_engine import ReputationEngine, ReputationState, SystemConfig

STYLE = {"font.family": "serif", "axes.spines.top": False, "axes.spines.right": False,
         "axes.grid": True, "grid.alpha": 0.3, "figure.dpi": 300}
COLORS = ["#2166ac", "#d73027", "#4dac26", "#7b3294", "#f1a340"]
CONFIG_LABELS = {
    "config1_baseline": "Config 1 (Baseline)",
    "config2_mixed":    "Config 2 (Mixed)",
    "config3_sybil":    "Config 3 (Sybil)",
    "config4_collusion":"Config 4 (Collusion)",
    "config5_adaptive": "Config 5 (Adaptive)",
}
LOG_DIR  = Path("results/training_logs")
FIG_DIR  = Path("results/figures")
FIG_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(STYLE)


def load_log(name, seed=0):
    p = LOG_DIR / f"{name}_seed{seed}_log.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def load_summary(name):
    p = LOG_DIR / f"{name}_summary.json"
    if p.exists():
        with open(p) as f:
            return json.load(f)
    return None


def smooth(x, w=50):
    if len(x) < w:
        return x
    return np.convolve(x, np.ones(w)/w, mode="valid")


# ── Figure 1: Training curves ─────────────────────────────────────────────────
def fig_training_curves():
    fig, ax = plt.subplots(figsize=(7, 4))
    config_names = list(CONFIG_LABELS.keys())
    any_data = False
    for i, name in enumerate(config_names):
        log = load_log(name)
        if log is None:
            continue
        rewards = [e["mean_reward"] for e in log]
        s = smooth(rewards)
        ax.plot(s, color=COLORS[i], label=CONFIG_LABELS[name], linewidth=1.5)
        any_data = True
    if not any_data:
        ax.text(0.5, 0.5, "No training data yet", ha="center", va="center", transform=ax.transAxes)
    ax.set_xlabel("Episode")
    ax.set_ylabel("Mean Episode Reward")
    ax.set_title("Training Reward Curves — All Configurations")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_training_curves.png", dpi=300)
    plt.close(fig)
    print("fig_training_curves.png saved")


# ── Figure 2: Honest action % convergence (Config 2) ─────────────────────────
def fig_honest_convergence():
    fig, ax = plt.subplots(figsize=(6, 4))
    log = load_log("config2_mixed")
    if log:
        honest = [e["honest_pct"] for e in log]
        ax.plot(smooth(honest), color=COLORS[1], linewidth=1.5, label="Adversarial agents")
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% threshold")
        ax.set_xlabel("Episode"); ax.set_ylabel("Honest Action Fraction")
        ax.set_title("Honest Action Convergence — Config 2 (Mixed Population)")
        ax.set_ylim(0, 1); ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Config 2 data not yet available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_honest_convergence.png", dpi=300)
    plt.close(fig)
    print("fig_honest_convergence.png saved")


# ── Figure 3: Reputation accuracy (Config 2) ─────────────────────────────────
def fig_reputation_accuracy():
    fig, ax = plt.subplots(figsize=(6, 4))
    log = load_log("config2_mixed")
    if log:
        acc = [e["mean_reputation_accuracy"] for e in log]
        ax.plot(smooth(acc), color=COLORS[0], linewidth=1.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("Mean |score − true_quality|")
        ax.set_title("Reputation Accuracy Over Training — Config 2")
    else:
        ax.text(0.5, 0.5, "Config 2 data not yet available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_reputation_accuracy.png", dpi=300)
    plt.close(fig)
    print("fig_reputation_accuracy.png saved")


# ── Figure 4: Sybil stake depletion (Config 3) ───────────────────────────────
def fig_sybil_stake():
    fig, ax = plt.subplots(figsize=(6, 4))
    log = load_log("config3_sybil")
    if log:
        sybils = [e.get("total_sybils", 0) for e in log]
        ax.plot(smooth(sybils, 20), color=COLORS[2], linewidth=1.5)
        ax.set_xlabel("Episode"); ax.set_ylabel("Total Sybil Identities Created")
        ax.set_title("Sybil Identity Creation Over Training — Config 3")
    else:
        ax.text(0.5, 0.5, "Config 3 data not yet available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_sybil_stake.png", dpi=300)
    plt.close(fig)
    print("fig_sybil_stake.png saved")


# ── Figure 5: Collusion CI widening (Config 4) ───────────────────────────────
def fig_collusion_ci():
    fig, ax = plt.subplots(figsize=(6, 4))
    log = load_log("config4_collusion")
    if log:
        ci = [e.get("mean_ci_width", 0) for e in log]
        sc = [e.get("mean_score", 0.5) for e in log]
        ax2 = ax.twinx()
        ax.plot(smooth(ci), color=COLORS[3], linewidth=1.5, label="CI Width")
        ax2.plot(smooth(sc), color=COLORS[0], linewidth=1.5, linestyle="--", label="Mean Score")
        ax.set_xlabel("Episode"); ax.set_ylabel("Mean Wilson CI Width", color=COLORS[3])
        ax2.set_ylabel("Mean Reputation Score", color=COLORS[0])
        ax.set_title("Collusion: Score Inflation vs. CI Widening — Config 4")
        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1+lines2, labels1+labels2, fontsize=8)
    else:
        ax.text(0.5, 0.5, "Config 4 data not yet available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_collusion_ci.png", dpi=300)
    plt.close(fig)
    print("fig_collusion_ci.png saved")


# ── Figure 6: Breaking point analysis (Config 5) ─────────────────────────────
def fig_breaking_point():
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for name, ax, label, color in [
        ("config1_baseline", axes[0], "Baseline (honest)", COLORS[0]),
        ("config5_adaptive", axes[1], "Adaptive adversary (2× incentive)", COLORS[1]),
    ]:
        log = load_log(name)
        if log:
            honest = [e["honest_pct"] for e in log]
            ax.plot(smooth(honest), color=color, linewidth=1.5)
            ax.axhline(0.5, color="gray", linestyle="--", linewidth=1)
            ax.set_ylim(0, 1)
            ax.set_title(label)
            ax.set_xlabel("Episode"); ax.set_ylabel("Honest Action Fraction")
        else:
            ax.text(0.5, 0.5, f"{name} not available", ha="center", va="center", transform=ax.transAxes)
    fig.suptitle("Breaking Point: Honest Behavior at Extreme Adversarial Incentive", fontsize=10)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_breaking_point.png", dpi=300)
    plt.close(fig)
    print("fig_breaking_point.png saved")


# ── Figure 7: Wilson CI calibration (analytic) ───────────────────────────────
def fig_wilson_calibration():
    engine = ReputationEngine()
    ns = np.arange(1, 200)
    # Simulate an actor with true_quality=0.8 receiving n ratings
    widths_08, widths_05 = [], []
    for n in ns:
        # true quality 0.8
        alpha = 2.0 + 0.8 * n
        beta  = 2.0 + 0.2 * n
        lo, hi, w = engine.wilson_ci(alpha, beta)
        widths_08.append(w)
        # true quality 0.5 (uninformative)
        alpha2 = 2.0 + 0.5 * n
        beta2  = 2.0 + 0.5 * n
        lo2, hi2, w2 = engine.wilson_ci(alpha2, beta2)
        widths_05.append(w2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.plot(ns, widths_08, color=COLORS[0], linewidth=1.5, label="true quality = 0.8")
    ax.plot(ns, widths_05, color=COLORS[1], linewidth=1.5, label="true quality = 0.5")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="CI width = 0.50 gate threshold")
    ax.set_xlabel("Number of Ratings"); ax.set_ylabel("Wilson 95% CI Width")
    ax.set_title("Wilson CI Width vs. Evidence Count (β-decay prior α₀=β₀=2)")
    ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_wilson_calibration.png", dpi=300)
    plt.close(fig)
    print("fig_wilson_calibration.png saved")


# ── Figure 8: Ablation study ──────────────────────────────────────────────────
def fig_ablation():
    summaries = {name: load_summary(name) for name in CONFIG_LABELS}
    labels = [CONFIG_LABELS[n] for n in CONFIG_LABELS if summaries.get(n)]
    vals   = [summaries[n]["mean_honest_pct"] for n in CONFIG_LABELS if summaries.get(n)]
    errs   = [summaries[n]["std_honest_pct"]  for n in CONFIG_LABELS if summaries.get(n)]

    fig, ax = plt.subplots(figsize=(8, 4))
    if vals:
        x = np.arange(len(labels))
        bars = ax.bar(x, vals, yerr=errs, capsize=4,
                      color=COLORS[:len(labels)], alpha=0.8, width=0.6)
        ax.set_xticks(x); ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=8)
        ax.set_ylabel("Honest Action Fraction (mean ± std)")
        ax.set_ylim(0, 1.1)
        ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="50% threshold")
        ax.set_title("Honest Behavior Across All Configurations (5 seeds each)")
        ax.legend(fontsize=8)
    else:
        ax.text(0.5, 0.5, "Training data not yet available", ha="center", va="center", transform=ax.transAxes)
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig_ablation.png", dpi=300)
    plt.close(fig)
    print("fig_ablation.png saved")


if __name__ == "__main__":
    fig_training_curves()
    fig_honest_convergence()
    fig_reputation_accuracy()
    fig_sybil_stake()
    fig_collusion_ci()
    fig_breaking_point()
    fig_wilson_calibration()
    fig_ablation()
    print(f"\nAll figures saved to {FIG_DIR}")
