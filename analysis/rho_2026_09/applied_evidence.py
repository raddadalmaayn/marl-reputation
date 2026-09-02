"""
Invert mean_ci_width -> n = alpha+beta for all 540 shards; define
applied_evidence = n - 4 (evidence above the (2,2) prior floor).

METHOD AND ITS LIMITS
  Wilson width for one agent is width(n_i, p_i), strictly decreasing in n_i for
  fixed p_i. Stored per shard are the per-agent p_i (`agent_scores`, which is
  exactly alpha/(alpha+beta)) and the MEAN of the per-agent widths -- the
  individual widths are not stored. So we solve for the single scalar n with

      mean_i width(n, p_i) = observed mean_ci_width

  which is decreasing in n and therefore has a unique root. This n is an
  EQUIVALENT UNIFORM evidence count. It is exact when all agents share the same
  n -- which is guaranteed for every collapsed shard, because the floor makes
  n>=4 and width(4,0.5) the maximum attainable value, so a mean equal to it
  forces every agent to it. Where evidence is heterogeneous across agents it is
  an aggregate summary, not a per-agent truth, and is labelled as such.

  applied_evidence = 0 is EXACT regardless: it holds iff every agent is at the
  floor, which is directly observable as all scores == 0.5.

SCOPE
  env.reset() re-initialises ReputationState to the prior, so reputation
  evidence does NOT accumulate across episodes. `final_env_metrics` is captured
  after collect_basin_report, i.e. at the end of the LAST of the 20 eval
  episodes. So n, applied_evidence and total_ratings are all last-episode
  quantities, whereas participation_rate is averaged over all 20. That
  asymmetry is stated wherever the two are compared.

Read-only. Writes only analysis/rho_2026_09/applied_evidence.csv.
"""
import csv
import json
import math
from pathlib import Path

import numpy as np
from scipy.optimize import brentq

REPO = Path(__file__).resolve().parent.parent.parent
SWEEP = REPO / "results" / "tier1b" / "sweep"
OUT = REPO / "analysis" / "rho_2026_09"
Z = 1.96


def width(n, p):
    if n <= 0:
        return float("nan")
    d = 1.0 + (Z * Z) / n
    c = (p + (Z * Z) / (2.0 * n)) / d
    m = (Z * math.sqrt(max(p * (1 - p), 0.0) / n + (Z * Z) / (4 * n * n))) / d
    return min(1.0, c + m) - max(0.0, c - m)


def mean_width(n, ps):
    return float(np.mean([width(n, p) for p in ps]))


def invert(W, ps):
    """Solve mean_i width(n, p_i) = W for scalar n >= 4. None if no root."""
    lo, hi = 4.0, 1e9
    f_lo = mean_width(lo, ps) - W
    if f_lo < -1e-12:           # observed width wider than the floor allows
        return None
    if abs(f_lo) <= 1e-12:
        return 4.0
    if mean_width(hi, ps) - W > 0:
        return None
    try:
        return float(brentq(lambda n: mean_width(n, ps) - W, lo, hi, xtol=1e-10))
    except Exception:
        return None


def main():
    rows = []
    for p in sorted((SWEEP / "training_logs").glob("sweep_*_seed*.json")):
        if p.name.endswith("_log.json"):
            continue
        d = json.loads(p.read_text())
        m = d["final_env_metrics"]
        name = d["config"]
        cbase = name.replace("sweep_", "").rsplit("_arm", 1)[0]
        arm = "B" if name.endswith("armB") else "A"
        ps = [float(x) for x in m["agent_scores"]]
        W = float(m["mean_ci_width"])
        n = invert(W, ps)
        ac = d.get("action_counts", {})
        total_actions = sum(int(v) for v in ac.values())
        rating_actions = sum(int(ac.get(str(i), ac.get(i, 0))) for i in (1, 2, 3, 4))
        n_agents = len(ps)
        # per-agent-step rates so attempted and applied are unit-comparable
        eval_eps = int(d.get("basin_eval_episodes", 20))
        steps_per_ep = total_actions / (n_agents * eval_eps) if total_actions else float("nan")
        applied_per_step = (m["total_ratings"] / (n_agents * steps_per_ep)
                            if steps_per_ep and not math.isnan(steps_per_ep) else float("nan"))
        rows.append({
            "config": cbase, "arm": arm, "seed": int(d["seed"]),
            "participation_rate": d["participation_rate"],
            "basin": d["basin"],
            "mean_ci_width": W,
            "n_alpha_plus_beta": n if n is not None else "",
            "applied_evidence": (n - 4.0) if n is not None else "",
            "all_scores_half": int(all(abs(x - 0.5) < 1e-12 for x in ps)),
            "total_ratings_last_ep": m["total_ratings"],
            "rating_actions": rating_actions,
            "total_actions": total_actions,
            "applied_per_agent_step": applied_per_step,
            "n_agents": n_agents,
            "rho_defined": int(not (isinstance(d["system_spearman_rho"], float)
                                    and math.isnan(d["system_spearman_rho"]))),
        })
    rows.sort(key=lambda r: (int(r["config"][1:]), r["arm"], r["seed"]))
    OUT.mkdir(parents=True, exist_ok=True)
    with open(OUT / "applied_evidence.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    fails = [r for r in rows if r["n_alpha_plus_beta"] == ""]
    print(f"wrote {OUT/'applied_evidence.csv'} ({len(rows)} rows)")
    print(f"inversion failed (reported blank, never imputed): {len(fails)}")
    for r in fails[:10]:
        print("   ", r["config"], r["arm"], r["seed"], r["mean_ci_width"])


if __name__ == "__main__":
    main()
