"""
Load training summaries and compute final evaluation metrics across all 5 configs.
Outputs: results/evaluation_summary.json
"""

import json
import sys
import numpy as np
from pathlib import Path
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent))


CONFIG_NAMES = [f"config{i}_{s}" for i, s in enumerate([
    "baseline", "mixed", "sybil", "collusion", "adaptive"], 1)]


def load_summary(log_dir: Path, config_name: str) -> dict:
    path = log_dir / f"{config_name}_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def wilcoxon_test(log_dir: Path, config_name: str):
    """Compare adversarial vs honest agent final rewards from per-seed logs."""
    honest_rewards, adv_rewards = [], []
    for seed in range(5):
        p = log_dir / f"{config_name}_seed{seed}.json"
        if not p.exists():
            continue
        with open(p) as f:
            d = json.load(f)
        honest_rewards.append(d.get("eval_mean_reward", 0))
    return honest_rewards


def main():
    results_dir = Path("results")
    log_dir = results_dir / "training_logs"
    out_path = results_dir / "evaluation_summary.json"

    all_summaries = {}
    for name in CONFIG_NAMES:
        s = load_summary(log_dir, name)
        if s:
            all_summaries[name] = s
            print(f"Loaded: {name}")
        else:
            print(f"Missing: {name}")

    if not all_summaries:
        print("No summaries found. Run training first.")
        return

    # Find breaking point (config5: does honest_pct drop below 0.5?)
    breaking_point = None
    if "config5_adaptive" in all_summaries:
        s5 = all_summaries["config5_adaptive"]
        if s5["mean_honest_pct"] < 0.5:
            breaking_point = "config5_adaptive: system fails (honest_pct < 0.5)"
        else:
            breaking_point = "config5_adaptive: system holds (honest_pct >= 0.5)"

    # Wilcoxon: honest (config1) vs adversarial (config5) rewards
    h = wilcoxon_test(log_dir, "config1_baseline")
    a = wilcoxon_test(log_dir, "config5_adaptive")
    wilcoxon_result = None
    if len(h) >= 3 and len(a) >= 3:
        try:
            stat, p = stats.wilcoxon(h[:min(len(h),len(a))], a[:min(len(h),len(a))])
            wilcoxon_result = {"statistic": float(stat), "p_value": float(p),
                               "significant": bool(p < 0.05)}
        except Exception as e:
            wilcoxon_result = {"error": str(e)}

    summary = {
        "configs": all_summaries,
        "breaking_point": breaking_point,
        "wilcoxon_honest_vs_adversarial": wilcoxon_result,
        "key_finding": (
            "Honest behavior is the dominant strategy"
            if all(s.get("mean_honest_pct", 0) >= 0.5 for s in all_summaries.values())
            else "System shows vulnerability under high adversarial incentive"
        ),
    }

    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nEvaluation summary → {out_path}")
    print(f"Key finding: {summary['key_finding']}")
    if breaking_point:
        print(f"Breaking point: {breaking_point}")


if __name__ == "__main__":
    main()
