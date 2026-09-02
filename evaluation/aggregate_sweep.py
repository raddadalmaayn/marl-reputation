"""
Tier 1B full sweep — FINAL aggregation into the 4 paper tables. Reads ONLY
results/tier1b/sweep/ (never the pilot). Produces .json + .md:

  1. per-config per-arm summary  (participation, P(honest-dom)+Jeffreys95, cond honest,
     rho mean+/-, hold rate)
  2. ARM-DELTA (B-A)             (participation, rho [Welch95], P(honest-dom) [Newcombe95])
  3. SECURITY-COST              (Arm A vs B successful-attack count + hold rate)
  4. BASELINE-UNDER-ATTACK      ({C2,C3,C11} x {beta,flat,eigentrust} x arm: rho + hold)

Usage: python3 evaluation/aggregate_sweep.py
"""
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np
import sys
REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from evaluation.aggregate_arms import (load_shards, arm_endpoints, welch_ci,
                                       newcombe_diff_ci)

SWEEP = REPO / "results" / "tier1b" / "sweep"


def f(x, n=3):
    return "NaN" if (isinstance(x, float) and math.isnan(x)) else (
        f"{x:.{n}f}" if isinstance(x, (int, float)) else str(x))


def main():
    rows = load_shards(SWEEP)            # config_base -> arm -> [summaries]
    report = {"per_config_arm": {}, "arm_delta": {}, "security_cost": {}}

    for base in sorted(rows):
        ep = {a: arm_endpoints(rows[base][a], str(SWEEP)) for a in rows[base] if rows[base][a]}
        report["per_config_arm"][base] = {
            a: {k: v for k, v in ep[a].items() if not k.startswith("_")} for a in ep}
        if "A" in ep and "B" in ep:
            dp = welch_ci(ep["A"]["_parts"], ep["B"]["_parts"])
            dr = welch_ci(ep["A"]["_rhos"], ep["B"]["_rhos"])
            nc = newcombe_diff_ci(ep["A"]["k_honest_dominant"], ep["A"]["n_seeds"],
                                  ep["B"]["k_honest_dominant"], ep["B"]["n_seeds"])
            report["arm_delta"][base] = {
                "participation": {"diff": dp[0], "ci95": [dp[1], dp[2]]},
                "rho": {"diff": dr[0], "ci95": [dr[1], dr[2]]},
                "p_honest_dominant": {
                    "diff": ep["B"]["p_honest_dominant"] - ep["A"]["p_honest_dominant"],
                    "newcombe95": list(nc)},
            }
            report["security_cost"][base] = {
                "armA": {"successful_attacks": ep["A"]["attacks_attempted"] - ep["A"]["attacks_blocked"],
                         "attempts": ep["A"]["attacks_attempted"], "hold_rate": ep["A"]["hold_rate"]},
                "armB": {"successful_attacks": ep["B"]["attacks_attempted"] - ep["B"]["attacks_blocked"],
                         "attempts": ep["B"]["attacks_attempted"], "hold_rate": ep["B"]["hold_rate"]},
            }

    # Table 4 — baselines
    base_dir = SWEEP / "baselines"
    btab = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))  # cfg->arm->engine->[rho]
    bhold = defaultdict(lambda: defaultdict(list))
    n_base = 0
    if base_dir.exists():
        for p in sorted(base_dir.glob("*_baseline.json")):
            d = json.loads(p.read_text()); n_base += 1
            cfg = d["config"].replace("sweep_", "").rsplit("_arm", 1)[0]
            arm = d["arm"]
            for eng in ("beta", "flat", "eigentrust"):
                r = d["engines"][eng]["rho"]
                if not (isinstance(r, float) and math.isnan(r)):
                    btab[cfg][arm][eng].append(r)
            if not (isinstance(d["hold_rate"], float) and math.isnan(d["hold_rate"])):
                bhold[cfg][arm].append(d["hold_rate"])
    report["baselines"] = {
        c: {a: {"n": len(btab[c][a]["beta"]),
                "rho": {e: (float(np.mean(btab[c][a][e])) if btab[c][a][e] else float("nan"))
                        for e in ("beta", "flat", "eigentrust")},
                "hold_rate": (float(np.mean(bhold[c][a])) if bhold[c][a] else float("nan"))}
            for a in btab[c]} for c in btab}
    report["baselines_n_files"] = n_base

    (SWEEP / "final_tables.json").write_text(json.dumps(report, indent=2))

    # ---- Markdown ----
    L = ["# Tier 1B Full Sweep — Final Tables", "",
         "Aggregated ONLY from `results/tier1b/sweep/`. See `ledger.json` for "
         "shard completeness; per-config seed counts (n) below reflect COMPLETED "
         "shards only — missing shards are not imputed.", ""]
    L += ["## Table 1 — Per-config, per-arm summary", "",
          "| Config | arm | n | particip. | cond honest | ρ (mean±sd) | "
          "P(hon-dom) [Jeffreys95] | hold rate |", "|---|---|---|---|---|---|---|---|"]
    for base in sorted(report["per_config_arm"]):
        for arm in ("A", "B"):
            a = report["per_config_arm"][base].get(arm)
            if not a:
                continue
            rsd = np.nanstd([x for x in a["rho_per_seed"] if x is not None]) if any(
                x is not None for x in a["rho_per_seed"]) else float("nan")
            jl, jh = a["p_honest_jeffreys95"]
            L.append(f"| {base} | {arm} | {a['n_seeds']} | {f(a['participation_mean'])} | "
                     f"{f(a['cond_honest_mean'])} | {f(a['rho_mean'])}±{f(rsd,2)} | "
                     f"{f(a['p_honest_dominant'])} [{f(jl)},{f(jh)}] | {f(a['hold_rate'])} |")
    L += ["", "## Table 2 — Arm delta (B − A), with CIs", "",
          "| Config | Δparticipation [95% CI] | Δρ [Welch95] | ΔP(hon-dom) [Newcombe95] |",
          "|---|---|---|---|"]
    for base in sorted(report["arm_delta"]):
        d = report["arm_delta"][base]
        L.append(f"| {base} | {f(d['participation']['diff'])} "
                 f"[{f(d['participation']['ci95'][0])},{f(d['participation']['ci95'][1])}] | "
                 f"{f(d['rho']['diff'])} [{f(d['rho']['ci95'][0])},{f(d['rho']['ci95'][1])}] | "
                 f"{f(d['p_honest_dominant']['diff'])} "
                 f"[{f(d['p_honest_dominant']['newcombe95'][0])},"
                 f"{f(d['p_honest_dominant']['newcombe95'][1])}] |")
    L += ["", "## Table 3 — Security cost (successful attacks & hold rate)", "",
          "| Config | A success/attempts | A hold | B success/attempts | B hold |",
          "|---|---|---|---|---|"]
    for base in sorted(report["security_cost"]):
        s = report["security_cost"][base]
        L.append(f"| {base} | {s['armA']['successful_attacks']}/{s['armA']['attempts']} | "
                 f"{f(s['armA']['hold_rate'])} | "
                 f"{s['armB']['successful_attacks']}/{s['armB']['attempts']} | "
                 f"{f(s['armB']['hold_rate'])} |")
    L += ["", "## Table 4 — Baseline under attack ({C2,C3,C11}, same learned adversaries)",
          "", "ρ per estimator (re-scored from the identical Beta-driven eval stream); "
          "hold rate is estimator-independent (same actions).", "",
          "| Config | arm | n | ρ beta | ρ flat | ρ eigentrust | hold rate |",
          "|---|---|---|---|---|---|---|"]
    for c in sorted(report["baselines"]):
        for arm in ("A", "B"):
            b = report["baselines"][c].get(arm)
            if not b:
                continue
            L.append(f"| {c} | {arm} | {b['n']} | {f(b['rho']['beta'])} | "
                     f"{f(b['rho']['flat'])} | {f(b['rho']['eigentrust'])} | "
                     f"{f(b['hold_rate'])} |")
    if not report["baselines"]:
        L.append("| — | — | 0 | — | — | — | — |  *(no baseline_eval outputs yet)*")
    (SWEEP / "final_tables.md").write_text("\n".join(L) + "\n")
    print("\n".join(L))
    print(f"\nWrote {SWEEP}/final_tables.{{json,md}}  (baseline files: {n_base})")


if __name__ == "__main__":
    main()
