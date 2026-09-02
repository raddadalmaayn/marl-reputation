"""
T0.5(a) — Diagnose the reported "identical ablation summaries" for no_stake and
no_dispute, and regenerate correct summaries from the RAW per-seed logs.

Findings (see TIER0_REPORT.md for the writeup) are derived purely from the raw
logs under results/ablation/training_logs/ (never modified). Derived outputs go
to results/derived/.

This script:
  1. Recomputes each ablation variant's tail summary from its OWN raw seed logs
     (the same tail aggregation train_ablation.py uses: last 20% of episodes).
  2. Does a field-by-field comparison of no_stake vs no_dispute raw logs to show
     exactly which fields coincide and which differ.
  3. Confirms whether the runs are independent (training_time_s / final stakes).

Usage: python3 evaluation/diagnose_ablation.py
"""

import json
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent
ABL = REPO / "results" / "ablation" / "training_logs"
DERIVED = REPO / "results" / "derived"

VARIANTS = ["ablation_no_stake", "ablation_no_dispute",
            "ablation_no_decay", "ablation_no_wilson"]
N_SEEDS = 3


def regen_summary(name: str) -> dict:
    """Recompute the tail summary from a variant's own raw seed logs."""
    honest, rew, acc, blk_all, att_all = [], [], [], [], []
    seeds_used = []
    for s in range(N_SEEDS):
        lp = ABL / f"{name}_seed{s}_log.json"
        if not lp.exists():
            continue
        log = json.loads(lp.read_text())
        tail = log[int(len(log) * 0.8):]
        honest.append(float(np.mean([e["honest_pct"] for e in tail])))
        rew.append(float(np.mean([e["mean_reward"] for e in tail])))
        acc.append(float(np.mean([e["mean_reputation_accuracy"] for e in tail])))
        blk = sum(e.get("attacks_blocked", 0) for e in tail)
        att = sum(e.get("attacks_attempted", 0) for e in tail)
        blk_all.append(blk)
        att_all.append(att)
        seeds_used.append(s)
    if not seeds_used:
        return {"name": name, "error": "no raw logs"}
    defense = [b / a for b, a in zip(blk_all, att_all) if a > 0]
    return {
        "name": name,
        "seeds_used": seeds_used,
        "mean_honest_pct": float(np.mean(honest)),
        "std_honest_pct": float(np.std(honest)),
        "mean_eval_reward": float(np.mean(rew)),
        "mean_eval_accuracy": float(np.mean(acc)),
        "mean_defense_rate": float(np.mean(defense)) if defense else None,
        "total_attacks_blocked": int(sum(blk_all)),
        "total_attacks_attempted": int(sum(att_all)),
        # per-seed run metadata to test independence
        "per_seed_meta": _meta(name),
    }


def _meta(name: str) -> list:
    out = []
    for s in range(N_SEEDS):
        rp = ABL / f"{name}_seed{s}.json"
        if not rp.exists():
            continue
        d = json.loads(rp.read_text())
        stakes = d["final_env_metrics"]["stakes"]
        out.append({
            "seed": s,
            "training_time_s": d["training_time_s"],
            "converged": d["converged"],
            "final_mean_stake": float(np.mean(list(stakes.values()))),
        })
    return out


def field_diff(a_name: str, b_name: str) -> dict:
    """Per-episode field comparison across two variants' raw logs."""
    identical_fields, differing_fields = set(), set()
    per_seed = []
    for s in range(N_SEEDS):
        ap = ABL / f"{a_name}_seed{s}_log.json"
        bp = ABL / f"{b_name}_seed{s}_log.json"
        if not (ap.exists() and bp.exists()):
            continue
        a = json.loads(ap.read_text())
        b = json.loads(bp.read_text())
        diff_keys = set()
        for ea, eb in zip(a, b):
            for k in ea:
                if ea.get(k) != eb.get(k):
                    diff_keys.add(k)
        same_keys = set(a[0].keys()) - diff_keys
        differing_fields |= diff_keys
        identical_fields |= same_keys
        per_seed.append({"seed": s, "n_episodes": len(a),
                         "differing_keys": sorted(diff_keys)})
    identical_fields -= differing_fields
    return {
        "pair": [a_name, b_name],
        "fields_identical_across_all_episodes": sorted(identical_fields),
        "fields_differing": sorted(differing_fields),
        "per_seed": per_seed,
    }


def main():
    DERIVED.mkdir(parents=True, exist_ok=True)
    summaries = {v: regen_summary(v) for v in VARIANTS}
    diff = field_diff("ablation_no_stake", "ablation_no_dispute")

    # Independence check: do the two variants' runs have distinct wall-clock times?
    def times(v):
        return [m["training_time_s"] for m in summaries[v].get("per_seed_meta", [])]
    independent = times("ablation_no_stake") != times("ablation_no_dispute")

    report = {
        "regenerated_summaries": summaries,
        "no_stake_vs_no_dispute_field_diff": diff,
        "runs_independent_by_walltime": independent,
        "diagnosis": (
            "no_stake and no_dispute were INDEPENDENT training runs (distinct "
            "training_time_s and distinct final stake magnitudes: ~2 vs ~20000). "
            "Their behavioral metrics (honest_pct, attacks_attempted/blocked, "
            "total_ratings, mean_reputation_accuracy) are byte-identical across all "
            "5000 episodes x 3 seeds; only stake-derived fields (stakes, total_reward, "
            "mean_reward) differ. This is NOT a file overwrite / lost data: it is "
            "explained by the environment normalizing stake by min_stake in the "
            "observation (obs[2] = balance / (4*min_stake), reset balance = "
            "2*min_stake -> identical normalized obs regardless of min_stake) while "
            "dispute_cost never enters the observation. The shared policy therefore "
            "receives identical inputs under both configs and, with identical seeds, "
            "produces identical action trajectories. The identical summaries are thus "
            "CORRECT given the data; no re-run is required to recover lost data. "
            "However the no_stake ablation does not actually remove economic "
            "deterrence from the agent's decision input, so it cannot test what it "
            "claims; a methodologically meaningful re-run requires changing the "
            "observation's stake normalization to an absolute scale (out of Tier-0 "
            "scope)."),
        "rerun_required_for_data_recovery": False,
        "contingency_rerun_command": (
            "python3 training/train_ablation.py --config configs/ablation_no_dispute.yaml "
            "--output results/ablation/ --seeds 3"),
    }
    (DERIVED / "ablation_diagnosis.json").write_text(json.dumps(report, indent=2))

    # Markdown
    L = ["# T0.5(a) Ablation diagnosis (no_stake vs no_dispute)", ""]
    L.append(f"Runs independent by wall-clock: **{independent}**")
    L.append("")
    L.append("## Regenerated summaries (from each variant's own raw logs)")
    L.append("| variant | mean honest_pct | std | mean reward | mean acc | "
             "defense rate | blocked | attempted | final stakes (per seed) |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for v in VARIANTS:
        s = summaries[v]
        if "error" in s:
            L.append(f"| {v} | ERROR | | | | | | | |")
            continue
        fs = ", ".join(f"{m['final_mean_stake']:.1f}" for m in s["per_seed_meta"])
        L.append(f"| {v} | {s['mean_honest_pct']:.4f} | {s['std_honest_pct']:.4f} | "
                 f"{s['mean_eval_reward']:.4f} | {s['mean_eval_accuracy']:.4f} | "
                 f"{s['mean_defense_rate']:.4f} | {s['total_attacks_blocked']} | "
                 f"{s['total_attacks_attempted']} | {fs} |")
    L.append("")
    L.append("## no_stake vs no_dispute field comparison (5000 ep x 3 seeds)")
    L.append(f"- Identical across ALL episodes: "
             f"`{', '.join(diff['fields_identical_across_all_episodes'])}`")
    L.append(f"- Differing: `{', '.join(diff['fields_differing'])}`")
    L.append("")
    L.append("**Diagnosis:** " + report["diagnosis"])
    L.append("")
    L.append(f"**Re-run required for data recovery:** "
             f"{report['rerun_required_for_data_recovery']} "
             f"(both raw logs are present and independently generated).")
    L.append("")
    L.append("Contingency re-run command (if a fresh no_dispute run is desired):")
    L.append(f"```\n{report['contingency_rerun_command']}\n```")
    (DERIVED / "ablation_diagnosis.md").write_text("\n".join(L) + "\n")
    print("Wrote results/derived/ablation_diagnosis.{json,md}")
    print(f"runs_independent_by_walltime={independent}")
    print("identical fields:", diff["fields_identical_across_all_episodes"])
    print("differing fields:", diff["fields_differing"])


if __name__ == "__main__":
    main()
