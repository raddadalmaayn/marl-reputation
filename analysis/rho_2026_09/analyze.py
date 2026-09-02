"""
Phase 1.3 + Phase 2 — test the selection hypothesis and build the reframed endpoint.

Reads analysis/rho_2026_09/shards.csv and tau.csv (both produced locally) plus
results/tier1b/sweep/baselines/*.json (read-only). Writes ONLY
analysis/rho_2026_09/findings.md. Never touches final_tables.*.
"""
import csv
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(REPO))
from evaluation.aggregate_arms import jeffreys_ci, newcombe_diff_ci, welch_ci

OUT = REPO / "analysis" / "rho_2026_09"
SWEEP = REPO / "results" / "tier1b" / "sweep"
ORDER = ["c1", "c2", "c3", "c4", "c5", "c6", "c7", "c8", "c9", "c10", "c11"]


def f(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load():
    rows = list(csv.DictReader(open(OUT / "shards.csv")))
    tau = {}
    if (OUT / "tau.csv").exists():
        for r in csv.DictReader(open(OUT / "tau.csv")):
            tau[(r["config"], r["arm"], int(r["seed"]))] = r
    for r in rows:
        r["seed"] = int(r["seed"])
        r["participation_rate"] = f(r["participation_rate"])
        r["score_std"] = f(r["score_std"])
        r["defined"] = int(r["rho_defined"] or 0)
        r["rho"] = f(r["rho_recomputed"])
        t = tau.get((r["config"], r["arm"], r["seed"]))
        r["tau_a"] = f(t["tau_a"]) if t else float("nan")
    return rows


def cliffs_delta(a, b):
    """P(a>b) - P(a<b). +1 = a strictly above b."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    if a.size == 0 or b.size == 0:
        return float("nan")
    gt = sum(int((x > b).sum()) for x in a)
    lt = sum(int((x < b).sum()) for x in a)
    return (gt - lt) / (a.size * b.size)


def mannwhitney(a, b):
    try:
        from scipy.stats import mannwhitneyu
        if len(a) == 0 or len(b) == 0:
            return float("nan"), float("nan")
        u, p = mannwhitneyu(a, b, alternative="two-sided")
        return float(u), float(p)
    except Exception:
        return float("nan"), float("nan")


def main():
    rows = load()
    L = []
    A = L.append

    A("# Phase 1–2 — why rho is undefined, and a reframed endpoint")
    A("")
    A("Host: zenbook (10.12.10.98). Repo `/home/raddad/marl-reputation`.")
    A(f"Source: `analysis/rho_2026_09/shards.csv` ({len(rows)} shards), re-scored from")
    A("existing `final.pt` checkpoints. No training, no imputation, no writes to")
    A("`results/tier1b/sweep/`.")
    A("")

    # ---------- reproduction check ----------
    match = sum(int(r["rho_matches_stored"] or 0) for r in rows)
    A(f"**Reproduction check:** recomputed rho matches the stored "
      f"`system_spearman_rho` for **{match}/{len(rows)}** shards.")
    A("")

    # ---------- Phase 1.1 mechanism ----------
    coll = [r for r in rows if r["score_std"] < 1e-12]
    und = [r for r in rows if not r["defined"]]
    qok = [r for r in rows if f(r["qual_std"]) > 1e-12]
    A("## Phase 1.1 — mechanism")
    A("")
    A(f"- shards with undefined rho: **{len(und)}/{len(rows)}**")
    A(f"- shards with collapsed score vector (std < 1e-12): **{len(coll)}**")
    A(f"- shards with non-degenerate true-quality vector (std > 1e-12): "
      f"**{len(qok)}/{len(rows)}**")
    same = len({r["config"] + str(r["seed"]) for r in coll} ^
               {r["config"] + str(r["seed"]) for r in und}) == 0
    A(f"- undefined set == collapsed set: **{len(coll) == len(und) and same}**")
    A("")
    A("The guard in `evaluation/basins.py::_score_quality_metrics` sets rho to NaN when")
    A("`std(scores) < 1e-12` **or** `std(quals) < 1e-12`. The true-quality vector is")
    A("never degenerate, so only the score branch ever fires. **Mechanism: score-vector")
    A("collapse.** With no ratings submitted, every agent's Beta posterior stays at the")
    A("prior mean 0.5, the score vector is constant, and Spearman is 0/0.")
    A("")

    # ---------- Phase 1.1b collapse modes ----------
    A("### Two distinct collapse modes")
    A("")
    m1 = [r for r in coll if r["participation_rate"] < 0.01]
    m2 = [r for r in coll if r["participation_rate"] >= 0.01]
    ms = sorted({round(f(r["mean_score_stored"]), 4) for r in coll})
    A(f"Every one of the {len(coll)} collapsed shards sits at mean score {ms} — the")
    A("Beta prior. But they arrive there two different ways:")
    A("")
    A(f"- **Mode 1 — abstention collapse ({len(m1)} shards).** participation < 0.01;")
    A("  nobody rates, so no posterior ever moves. The system was never used.")
    if m2:
        cfgs = sorted({(r["config"], r["arm"]) for r in m2})
        pr = [r["participation_rate"] for r in m2]
        bs = sorted({r["basin"] for r in m2})
        A(f"- **Mode 2 — jamming collapse ({len(m2)} shards).** participation "
          f"{min(pr):.2f}–{max(pr):.2f}; agents rate heavily yet every score still ends")
        A(f"  at exactly 0.5. Confined to {cfgs}, basin {bs}. The system was used and")
        A("  defeated: ratings arrive but carry no net information, so every posterior")
        A("  balances out to the prior.")
    A("")
    A("These are opposite failures — idle vs actively jammed — and the current")
    A("pipeline records both as the same NaN. Any claim that undefined rho means")
    A("\"agents abstained\" is therefore false for the Mode 2 shards.")
    A("")

    # ---------- Phase 1.3 hypothesis test ----------
    A("## Phase 1.3 — is definedness explained by participation?")
    A("")
    A("Mann-Whitney U on participation rate, rho-defined vs rho-undefined seeds,")
    A("with Cliff's delta (+1 = defined seeds strictly higher participation).")
    A("")
    A("| arm | config | n def | n und | median part. (def) | median part. (und) | U | p | Cliff's d |")
    A("|---|---|---|---|---|---|---|---|---|")
    for arm in ("A", "B"):
        for c in ORDER + ["POOLED"]:
            sub = [r for r in rows if r["arm"] == arm and
                   (c == "POOLED" or r["config"] == c)]
            d = [r["participation_rate"] for r in sub if r["defined"]]
            u_ = [r["participation_rate"] for r in sub if not r["defined"]]
            if not sub:
                continue
            U, p = mannwhitney(d, u_)
            cd = cliffs_delta(d, u_)
            md = np.median(d) if d else float("nan")
            mu = np.median(u_) if u_ else float("nan")
            lbl = "**POOLED**" if c == "POOLED" else c
            A(f"| {arm} | {lbl} | {len(d)} | {len(u_)} | {md:.4f} | {mu:.4f} | "
              f"{U:.0f} | {p:.3g} | {cd:+.3f} |")
    A("")

    # participation of undefined seeds — are they all ~0?
    for arm in ("A", "B"):
        u_ = [r["participation_rate"] for r in rows
              if r["arm"] == arm and not r["defined"]]
        if u_:
            A(f"- arm {arm}: undefined seeds have participation "
              f"max={max(u_):.4f}, mean={np.mean(u_):.4f}, "
              f"{sum(1 for x in u_ if x == 0.0)}/{len(u_)} exactly 0.")
    A("")

    # ---------- Phase 2.1 primary endpoint ----------
    A("## Phase 2.1 — PRIMARY: P(rho defined), i.e. P(system discriminates at all)")
    A("")
    A("Undefined rho treated as an outcome, not missing data.")
    A("")
    A("| config | armA k/n | armA P [Jeffreys95] | armB k/n | armB P [Jeffreys95] | delta B−A [Newcombe95] |")
    A("|---|---|---|---|---|---|")
    prim = {}
    for c in ORDER:
        a = [r for r in rows if r["config"] == c and r["arm"] == "A"]
        b = [r for r in rows if r["config"] == c and r["arm"] == "B"]
        ka, na = sum(r["defined"] for r in a), len(a)
        kb, nb = sum(r["defined"] for r in b), len(b)
        la, ha = jeffreys_ci(ka, na)
        lb, hb = jeffreys_ci(kb, nb)
        dl, dh = newcombe_diff_ci(ka, na, kb, nb)   # returns CI for p2-p1 = B-A
        prim[c] = (ka, na, kb, nb)
        A(f"| {c} | {ka}/{na} | {ka/na:.3f} [{la:.3f}, {ha:.3f}] | {kb}/{nb} | "
          f"{kb/nb:.3f} [{lb:.3f}, {hb:.3f}] | {kb/nb-ka/na:+.3f} [{dl:+.3f}, {dh:+.3f}] |")
    ka = sum(v[0] for v in prim.values()); na = sum(v[1] for v in prim.values())
    kb = sum(v[2] for v in prim.values()); nb = sum(v[3] for v in prim.values())
    la, ha = jeffreys_ci(ka, na); lb, hb = jeffreys_ci(kb, nb)
    dl, dh = newcombe_diff_ci(ka, na, kb, nb)   # returns CI for p2-p1 = B-A
    A(f"| **ALL** | {ka}/{na} | {ka/na:.3f} [{la:.3f}, {ha:.3f}] | {kb}/{nb} | "
      f"{kb/nb:.3f} [{lb:.3f}, {hb:.3f}] | {kb/nb-ka/na:+.3f} [{dl:+.3f}, {dh:+.3f}] |")
    A("")

    # ---------- Phase 2.2 secondary ----------
    A("## Phase 2.2 — SECONDARY: rho conditional on being defined")
    A("")
    A("**This subset is conditioned on the Phase 1.3 finding.** Seeds enter only when")
    A("the score vector did not collapse, which by Phase 1.3 means they are the")
    A("higher-participation seeds. In arm A that is a strongly selected minority, so")
    A("these means are NOT comparable across arms at face value.")
    A("")
    A("| config | armA n | armA rho | armB n | armB rho | delta B−A [Welch95] |")
    A("|---|---|---|---|---|---|")
    for c in ORDER:
        ra = [r["rho"] for r in rows if r["config"] == c and r["arm"] == "A" and r["defined"]]
        rb = [r["rho"] for r in rows if r["config"] == c and r["arm"] == "B" and r["defined"]]
        if len(ra) < 2 or len(rb) < 2:
            A(f"| {c} | {len(ra)} | {np.mean(ra) if ra else float('nan'):.3f} | "
              f"{len(rb)} | {np.mean(rb) if rb else float('nan'):.3f} | n too small |")
            continue
        dfm, lo, hi = welch_ci(ra, rb)   # (diff, lo, hi) for B-A
        A(f"| {c} | {len(ra)} | {np.mean(ra):+.3f} | {len(rb)} | {np.mean(rb):+.3f} | "
          f"{np.mean(rb)-np.mean(ra):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
    A("")

    # ---------- selection effect magnitude ----------
    A("### How much does the selection move the arm delta?")
    A("")
    A("Zero-extension: a constant score vector ranks nothing, so its discrimination is")
    A("0. The vector is fully observed, so this is a definitional choice at a known")
    A("point, not imputation of a missing value.")
    A("")
    from scipy.stats import spearmanr as _sp
    for arm in ("A", "B"):
        s_ = [r for r in rows if r["arm"] == arm and r["defined"]]
        rr, pv = _sp([x["participation_rate"] for x in s_], [x["rho"] for x in s_])
        A(f"- arm {arm}, among defined seeds: Spearman(participation, rho) = "
          f"{rr:+.3f} (p={pv:.3g}, n={len(s_)})")
    A("")
    A("| treatment | armA | armB | delta B−A | armA n | armB n |")
    A("|---|---|---|---|---|---|")
    ra = [r["rho"] for r in rows if r["arm"] == "A" and r["defined"]]
    rb = [r["rho"] for r in rows if r["arm"] == "B" and r["defined"]]
    nA = len([r for r in rows if r["arm"] == "A"])
    nB = len([r for r in rows if r["arm"] == "B"])
    c1_, c2_ = float(np.mean(ra)), float(np.mean(rb))
    z1, z2 = sum(ra) / nA, sum(rb) / nB
    A(f"| rho \\| defined (as published) | {c1_:+.4f} | {c2_:+.4f} | {c2_-c1_:+.4f} | "
      f"{len(ra)}/{nA} | {len(rb)}/{nB} |")
    A(f"| zero-extended over all seeds | {z1:+.4f} | {z2:+.4f} | {z2-z1:+.4f} | "
      f"{nA}/{nA} | {nB}/{nB} |")
    A("")
    if abs(c2_ - c1_) > 0:
        A(f"**|delta| shrinks by {100*(1-abs(z2-z1)/abs(c2_-c1_)):.0f}%** once the excluded")
        A("seeds are counted at their true (zero) discrimination.")
    A("")
    A("Per config, both treatments:")
    A("")
    A("| config | armA def/n | armB def/n | delta (rho\\|def) | delta (zero-ext) | sign flip |")
    A("|---|---|---|---|---|---|")
    for c in ORDER:
        Aa = [r for r in rows if r["config"] == c and r["arm"] == "A"]
        Bb = [r for r in rows if r["config"] == c and r["arm"] == "B"]
        raa = [r["rho"] for r in Aa if r["defined"]]
        rbb = [r["rho"] for r in Bb if r["defined"]]
        if not raa or not rbb:
            continue
        d1 = float(np.mean(rbb) - np.mean(raa))
        d2 = sum(rbb) / len(Bb) - sum(raa) / len(Aa)
        flip = "**YES**" if d1 * d2 < 0 else ""
        A(f"| {c} | {len(raa)}/{len(Aa)} | {len(rbb)}/{len(Bb)} | {d1:+.4f} | {d2:+.4f} | {flip} |")
    A("")

    # ---------- Phase 2.3 collapse-robust ----------
    have_tau = any(not math.isnan(r["tau_a"]) for r in rows)
    A("## Phase 2.3 — collapse-robust discrimination: Kendall's tau-a")
    A("")
    if not have_tau:
        A("_tau.csv not present; run `collapse_robust.py` first._")
    else:
        A("tau_a = (C − D) / [n(n−1)/2]. The denominator depends only on n, so tau-a is")
        A("defined whenever n ≥ 2; a constant score vector gives C = D = 0 and therefore")
        A("tau_a = 0 exactly — the correct statement that the score ranks nothing. This is")
        A("not imputation: the collapsed score vector is fully observed. tau-b is NOT used,")
        A("because its tie-corrected denominator reproduces the same 0/0 collapse as rho.")
        A("")
        A("**All seeds included — no exclusions.**")
        A("")
        A("| config | armA n | armA tau_a | armB n | armB tau_a | delta B−A [Welch95] |")
        A("|---|---|---|---|---|---|")
        for c in ORDER:
            ta = [r["tau_a"] for r in rows if r["config"] == c and r["arm"] == "A"
                  and not math.isnan(r["tau_a"])]
            tb = [r["tau_a"] for r in rows if r["config"] == c and r["arm"] == "B"
                  and not math.isnan(r["tau_a"])]
            if len(ta) < 2 or len(tb) < 2:
                continue
            dfm, lo, hi = welch_ci(ta, tb)   # (diff, lo, hi) for B-A
            A(f"| {c} | {len(ta)} | {np.mean(ta):+.3f} | {len(tb)} | {np.mean(tb):+.3f} | "
              f"{np.mean(tb)-np.mean(ta):+.3f} [{lo:+.3f}, {hi:+.3f}] |")
        A("")
        A("### rho-conditional vs tau-a: the size of the selection effect")
        A("")
        A("| config | arm | rho (defined only) | tau_a (defined only) | tau_a (ALL seeds) | shift |")
        A("|---|---|---|---|---|---|")
        for c in ORDER:
            for arm in ("A", "B"):
                sub = [r for r in rows if r["config"] == c and r["arm"] == arm]
                rd = [r["rho"] for r in sub if r["defined"]]
                td = [r["tau_a"] for r in sub if r["defined"] and not math.isnan(r["tau_a"])]
                ta = [r["tau_a"] for r in sub if not math.isnan(r["tau_a"])]
                if not rd or not ta:
                    continue
                A(f"| {c} | {arm} | {np.mean(rd):+.3f} | {np.mean(td):+.3f} | "
                  f"{np.mean(ta):+.3f} | {np.mean(ta)-np.mean(td):+.3f} |")
        A("")

    # ---------- Phase 2.4 Table 4 defects ----------
    A("## Phase 2.4 — Table 4 defects made explicit")
    A("")
    cells = defaultdict(lambda: {"n": 0, "rho_ok": defaultdict(int), "hold_n": 0})
    for p in sorted((SWEEP / "baselines").glob("*_baseline.json")):
        d = json.loads(p.read_text())
        cfg = d["config"].replace("sweep_", "").rsplit("_arm", 1)[0]
        k = (cfg, d["arm"])
        cells[k]["n"] += 1
        for e in ("beta", "flat", "eigentrust"):
            r = d["engines"][e]["rho"]
            if isinstance(r, float) and not math.isnan(r):
                cells[k]["rho_ok"][e] += 1
        h = d.get("hold_rate")
        if isinstance(h, float) and not math.isnan(h):
            cells[k]["hold_n"] += 1
    A("Per-engine n (the published table reports only the beta count as `n`), and the")
    A("hold-rate n exposed as its own column.")
    A("")
    A("| config | arm | files | n beta | n flat | n eigentrust | n hold_rate |")
    A("|---|---|---|---|---|---|---|")
    for k in sorted(cells):
        c = cells[k]
        A(f"| {k[0]} | {k[1]} | {c['n']} | {c['rho_ok']['beta']} | {c['rho_ok']['flat']} | "
          f"{c['rho_ok']['eigentrust']} | {c['hold_n']} |")
    A("")

    (OUT / "findings.md").write_text("\n".join(L) + "\n")
    print(f"wrote {OUT/'findings.md'}  ({len(L)} lines)")


if __name__ == "__main__":
    main()
