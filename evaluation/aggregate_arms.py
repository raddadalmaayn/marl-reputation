"""
Tier 1B — aggregate per-seed shard summaries into per-config, per-arm endpoints
and the ARM-DELTA (B - A) headline table. Works for the pilot and the full sweep.

Per (config, arm):
  participation rate (mean over seeds), conditional honest-given-rating (mean/median
  over rating-active seeds), Spearman rho (mean over seeds), basin counts +
  P(honest-dominant) with a Jeffreys 95% CI, defense hold rate (attacks_blocked /
  attacks_attempted pooled from final_env_metrics).

Arm delta (B - A):
  participation and rho: mean difference + Welch 95% CI (independent seeds; n small,
    reported explicitly).
  P(honest-dominant): Newcombe (Wilson-based) 95% CI for the difference of two
    independent proportions.

Arms are identified by participation_coef in each summary: 0.0 -> A, >0 -> B.
Config base name = summary "config" with a trailing _armA/_armB stripped.

Usage:
  python3 evaluation/aggregate_arms.py --dir results/tier1b/pilot \
      --out results/tier1b/pilot/arm_summary
"""

import argparse
import json
import math
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import beta as beta_dist, t as t_dist

import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.basins import classify_basin  # reuse Tier 0/1A thresholds


def jeffreys_ci(k, n, conf=0.95):
    if n == 0:
        return (float("nan"), float("nan"))
    lo = 0.0 if k == 0 else float(beta_dist.ppf((1 - conf) / 2, k + 0.5, n - k + 0.5))
    hi = 1.0 if k == n else float(beta_dist.ppf(1 - (1 - conf) / 2, k + 0.5, n - k + 0.5))
    return (lo, hi)


def wilson(k, n, z=1.96):
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    m = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / d
    return (c - m, c + m)


def newcombe_diff_ci(k1, n1, k2, n2):
    """Newcombe method 10 — 95% CI for p2 - p1 (arm B - arm A)."""
    if n1 == 0 or n2 == 0:
        return (float("nan"), float("nan"))
    l1, u1 = wilson(k1, n1)
    l2, u2 = wilson(k2, n2)
    p1, p2 = k1 / n1, k2 / n2
    d = p2 - p1
    lo = d - math.sqrt((p2 - l2) ** 2 + (u1 - p1) ** 2)
    hi = d + math.sqrt((u2 - p2) ** 2 + (p1 - l1) ** 2)
    return (lo, hi)


def welch_ci(a, b, conf=0.95):
    """95% CI for mean(b) - mean(a), Welch (unequal var). Returns (diff, lo, hi)."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    a, b = a[~np.isnan(a)], b[~np.isnan(b)]
    if len(a) < 2 or len(b) < 2:
        d = (np.mean(b) if len(b) else float("nan")) - (np.mean(a) if len(a) else float("nan"))
        return (float(d), float("nan"), float("nan"))
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / len(a) + vb / len(b))
    diff = b.mean() - a.mean()
    if se == 0:
        return (float(diff), float(diff), float(diff))
    df = (va / len(a) + vb / len(b)) ** 2 / (
        (va / len(a)) ** 2 / (len(a) - 1) + (vb / len(b)) ** 2 / (len(b) - 1))
    tc = t_dist.ppf(1 - (1 - conf) / 2, df)
    return (float(diff), float(diff - tc * se), float(diff + tc * se))


def load_shards(dir_):
    tlog = Path(dir_) / "training_logs"
    rows = defaultdict(lambda: defaultdict(list))  # config_base -> arm -> [summary]
    for p in sorted(tlog.glob("*_seed*.json")):
        if p.name.endswith("_log.json") or "_summary" in p.name:
            continue
        d = json.loads(p.read_text())
        if "participation_coef" not in d or "basin" not in d:
            continue  # not a Tier 1B shard
        arm = "B" if d.get("participation_coef", 0.0) > 0 else "A"
        base = re.sub(r"_arm[AB]$", "", d["config"])
        rows[base][arm].append(d)
    return rows


def hold_rate(summaries, dir_=None, tail_frac=0.2):
    """Defense hold rate pooled over the TAIL (last `tail_frac`) of each seed's
    training log — robust to the last eval episode having no attacks. Falls back to
    final_env_metrics if a log is missing."""
    blk = att = 0
    tlog = Path(dir_) / "training_logs" if dir_ else None
    for s in summaries:
        lp = (tlog / f"{s['config']}_seed{s['seed']}_log.json") if tlog else None
        if lp and lp.exists():
            log = json.loads(lp.read_text())
            tail = log[int(len(log) * (1 - tail_frac)):]
            blk += sum(e.get("attacks_blocked", 0) for e in tail)
            att += sum(e.get("attacks_attempted", 0) for e in tail)
        else:
            blk += s["final_env_metrics"].get("attacks_blocked", 0)
            att += s["final_env_metrics"].get("attacks_attempted", 0)
    return (blk / att) if att > 0 else float("nan"), blk, att


def arm_endpoints(summaries, dir_=None):
    parts = [s["participation_rate"] for s in summaries]
    rhos = [s.get("system_spearman_rho", float("nan")) for s in summaries]
    hgr = [s["honest_given_rating"] for s in summaries
           if not (isinstance(s["honest_given_rating"], float)
                   and math.isnan(s["honest_given_rating"]))]
    basins = [classify_basin(s["participation_rate"], s["honest_given_rating"],
                             s["final_mean_stake"], s["reset_stake"]) for s in summaries]
    n = len(summaries)
    k_hon = basins.count("honest-dominant")
    hr, blk, att = hold_rate(summaries, dir_)
    return {
        "n_seeds": n,
        "participation_mean": float(np.nanmean(parts)) if parts else float("nan"),
        "participation_per_seed": [round(float(x), 4) for x in parts],
        "rho_mean": float(np.nanmean(rhos)) if rhos else float("nan"),
        "rho_per_seed": [None if math.isnan(x) else round(float(x), 4) for x in rhos],
        "cond_honest_mean": float(np.mean(hgr)) if hgr else float("nan"),
        "cond_honest_median": float(np.median(hgr)) if hgr else float("nan"),
        "basin_counts": {b: basins.count(b) for b in
                         ["abstention", "adversarial-dominant", "honest-dominant", "mixed"]},
        "p_honest_dominant": k_hon / n if n else float("nan"),
        "p_honest_jeffreys95": list(jeffreys_ci(k_hon, n)),
        "k_honest_dominant": k_hon,
        "hold_rate": hr, "attacks_blocked": blk, "attacks_attempted": att,
        "_parts": parts, "_rhos": rhos,  # internal, for deltas
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    rows = load_shards(args.dir)
    report = {}
    for base in sorted(rows):
        arms = rows[base]
        ep = {arm: arm_endpoints(arms[arm], args.dir) for arm in arms if arms[arm]}
        entry = {"arms": {a: {k: v for k, v in ep[a].items() if not k.startswith("_")}
                          for a in ep}}
        if "A" in ep and "B" in ep:
            dpart = welch_ci(ep["A"]["_parts"], ep["B"]["_parts"])
            drho = welch_ci([x for x in ep["A"]["_rhos"]],
                            [x for x in ep["B"]["_rhos"]])
            nc = newcombe_diff_ci(ep["A"]["k_honest_dominant"], ep["A"]["n_seeds"],
                                  ep["B"]["k_honest_dominant"], ep["B"]["n_seeds"])
            entry["delta_B_minus_A"] = {
                "participation": {"diff": dpart[0], "ci95": [dpart[1], dpart[2]]},
                "rho": {"diff": drho[0], "ci95": [drho[1], drho[2]]},
                "p_honest_dominant_newcombe95": {
                    "diff": ep["B"]["p_honest_dominant"] - ep["A"]["p_honest_dominant"],
                    "ci95": list(nc)},
                "hold_rate": {"armA": ep["A"]["hold_rate"], "armB": ep["B"]["hold_rate"],
                              "diff": ep["B"]["hold_rate"] - ep["A"]["hold_rate"]},
            }
        report[base] = entry

    outp = Path(args.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.with_suffix(".json").write_text(json.dumps(report, indent=2))

    # Markdown
    def f(x, n=3):
        return "NaN" if (isinstance(x, float) and math.isnan(x)) else (
            f"{x:.{n}f}" if isinstance(x, float) else str(x))
    L = ["# Tier 1B — Arm A vs Arm B endpoints & deltas", ""]
    L.append("| Config | arm | seeds | particip. | cond honest | rho | "
             "basin a/adv/hon/mix | P(hon) [Jeffreys95] | hold rate |")
    L.append("|---|---|---|---|---|---|---|---|---|")
    for base in sorted(report):
        for arm in ("A", "B"):
            a = report[base]["arms"].get(arm)
            if not a:
                continue
            bc = a["basin_counts"]
            jl, jh = a["p_honest_jeffreys95"]
            L.append(f"| {base} | {arm} | {a['n_seeds']} | {f(a['participation_mean'])} | "
                     f"{f(a['cond_honest_mean'])} | {f(a['rho_mean'])} | "
                     f"{bc['abstention']}/{bc['adversarial-dominant']}/"
                     f"{bc['honest-dominant']}/{bc['mixed']} | "
                     f"{f(a['p_honest_dominant'])} [{f(jl)},{f(jh)}] | {f(a['hold_rate'])} |")
    L.append("")
    L.append("## Arm delta (B − A) with CIs")
    L.append("| Config | Δparticipation [95% CI] | Δrho [95% CI] | "
             "ΔP(hon) [Newcombe95] | hold rate A→B |")
    L.append("|---|---|---|---|---|")
    for base in sorted(report):
        d = report[base].get("delta_B_minus_A")
        if not d:
            continue
        pp = d["participation"]; rr = d["rho"]; ph = d["p_honest_dominant_newcombe95"]
        hr = d["hold_rate"]
        L.append(f"| {base} | {f(pp['diff'])} [{f(pp['ci95'][0])},{f(pp['ci95'][1])}] | "
                 f"{f(rr['diff'])} [{f(rr['ci95'][0])},{f(rr['ci95'][1])}] | "
                 f"{f(ph['diff'])} [{f(ph['ci95'][0])},{f(ph['ci95'][1])}] | "
                 f"{f(hr['armA'])}→{f(hr['armB'])} ({f(hr['diff'])}) |")
    outp.with_suffix(".md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {outp.with_suffix('.json')} and {outp.with_suffix('.md')}")


if __name__ == "__main__":
    main()
