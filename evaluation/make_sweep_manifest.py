"""
Tier 1B — Stage 1b: generate the full two-arm sweep configs + a per-host launch
manifest partitioning every (config x arm x seed) shard across the 5 testbed hosts.

Seed budget (justified by Tier 0 bimodality):
  bimodal/key {C1,C2,C3,C6,C7,C8,C9,C11}: 30 seeds/arm
  stable unimodal {C4,C5,C10}            : 10 seeds/arm
Two arms: A (participation_coef=0), B (participation_coef=C_RECOMMENDED).

Configs are derived from the ORIGINAL configs/configN.yaml (exact adversarial/
collusion/attack params) with only: name, participation_coef, stake_obs_mode, and
the (preserved) episode schedule set. Outputs:
  configs/tier1b/sweep_<cN>_<arm>.yaml      (22 files)
  results/tier1b/RUN_MANIFEST.tsv           (host, config, arm, seed, command)
  RUN_PLAN.md                               (shard map + wall-clock estimate)

Usage: python3 evaluation/make_sweep_manifest.py
"""

import copy
import json
from pathlib import Path

import yaml

REPO = Path(__file__).resolve().parent.parent
PART = json.loads((REPO / "results" / "tier1b" / "participation_economics.json").read_text())
C_REC = PART["feasibility"]["recommended_c"]

# base config name -> original yaml, seed budget
CONFIGS = {
    "c1": ("config1", 30), "c2": ("config2", 30), "c3": ("config3", 30),
    "c4": ("config4", 10), "c5": ("config5", 10), "c6": ("config6", 30),
    "c7": ("config7", 30), "c8": ("config8", 30), "c9": ("config9", 30),
    "c10": ("config10", 10), "c11": ("config11", 30),
}
ARMS = {"armA": 0.0, "armB": C_REC}

# (host, concurrency weight) — strong host + 4 desktops
HOSTS = [("strong", 16), ("desktop1", 6), ("desktop2", 6),
         ("desktop3", 6), ("desktop4", 6)]
EST_SHARD_HOURS = 1.5   # avg wall-clock per shard (5000ep + extension mix), assumption


def main():
    cfg_dir = REPO / "configs" / "tier1b"
    cfg_dir.mkdir(parents=True, exist_ok=True)
    out_dir = REPO / "results" / "tier1b"

    # 1) sweep configs
    written = []
    for cbase, (orig, _) in CONFIGS.items():
        src = yaml.safe_load((REPO / "configs" / f"{orig}.yaml").read_text())
        for arm, coef in ARMS.items():
            d = copy.deepcopy(src)
            d["name"] = f"sweep_{cbase}_{arm}"
            d["description"] = f"Tier1B full sweep {cbase} {arm} (participation_coef={coef})"
            d["participation_coef"] = coef
            d["stake_obs_mode"] = "absolute_log"
            # preserve original episode schedule (episodes / max_episodes_extended /
            # checkpoint_every / convergence_*); they already exist in src.
            (cfg_dir / f"sweep_{cbase}_{arm}.yaml").write_text(yaml.safe_dump(d, sort_keys=False))
            written.append(f"sweep_{cbase}_{arm}.yaml")

    # 2) flat shard list (config, arm, seed)
    shards = []
    for cbase, (_, nseed) in CONFIGS.items():
        for arm, coef in ARMS.items():
            for s in range(nseed):
                shards.append((cbase, arm, s))

    # 3) balance by wall-clock: shards per host proportional to concurrency, so
    #    n_h / w_h is ~equal across hosts (interleaved so each host gets a mix).
    total_w = sum(w for _, w in HOSTS)
    n = len(shards)
    quota = {h: round(n * w / total_w) for h, w in HOSTS}
    # fix rounding drift so quotas sum to n
    drift = n - sum(quota.values())
    quota[HOSTS[0][0]] += drift
    # interleave assignment by repeatedly serving the host with the largest
    # remaining (quota - served)/quota fraction
    served = {h: 0 for h, _ in HOSTS}
    assign = {}
    for shard in shards:
        host = max((h for h, _ in HOSTS),
                   key=lambda h: (quota[h] - served[h]) / max(1, quota[h]))
        assign.setdefault(host, []).append(shard)
        served[host] += 1

    # 4) manifest TSV
    lines = ["host\tconfig\tarm\tseed\tcommand"]
    for host, _ in HOSTS:
        for cbase, arm, s in assign.get(host, []):
            cmd = (f"OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 python3 evaluation/run_shard.py "
                   f"--config configs/tier1b/sweep_{cbase}_{arm}.yaml --seed {s} "
                   f"--output results/tier1b/sweep")
            lines.append(f"{host}\t{cbase}\t{arm}\t{s}\t{cmd}")
    (out_dir / "RUN_MANIFEST.tsv").write_text("\n".join(lines) + "\n")

    # 5) per-host launch scripts (xargs -P) for the desktops + strong host
    launch_dir = out_dir / "launch"
    launch_dir.mkdir(exist_ok=True)
    conc = {h: w for h, w in HOSTS}
    for host, _ in HOSTS:
        cmds = [l.split("\t")[-1] for l in lines[1:] if l.startswith(host + "\t")]
        body = ["#!/bin/bash", "set -e", "cd \"$(dirname \"$0\")/../../..\"",
                f"# {host}: {len(cmds)} shards, concurrency {conc[host]}",
                "cat <<'CMDS' | xargs -P %d -I {} bash -c '{}'" % conc[host]]
        body += cmds + ["CMDS"]
        (launch_dir / f"run_{host}.sh").write_text("\n".join(body) + "\n")

    # 6) RUN_PLAN.md
    per_host = {h: len(assign.get(h, [])) for h, _ in HOSTS}
    L = ["# Tier 1B — Stage 1b Full Two-Arm Sweep: RUN PLAN", ""]
    L.append(f"Recommended participation coef (Arm B) **c = {C_REC:.4f}** (Stage 0).")
    L.append("")
    L.append("## Shard budget")
    L.append("| Config set | configs | seeds/arm | arms | shards |")
    L.append("|---|---|---|---|---|")
    nb = sum(1 for c, (_, n) in CONFIGS.items() if n == 30)
    nu = sum(1 for c, (_, n) in CONFIGS.items() if n == 10)
    L.append(f"| bimodal/key | {nb} (C1,C2,C3,C6,C7,C8,C9,C11) | 30 | 2 | {nb*30*2} |")
    L.append(f"| stable unimodal | {nu} (C4,C5,C10) | 10 | 2 | {nu*10*2} |")
    L.append(f"| **total** | 11 | — | 2 | **{len(shards)}** |")
    L.append("")
    L.append("## Per-host shard map (weighted round-robin by concurrency)")
    L.append("| Host | concurrency | shards | est. wall-clock |")
    L.append("|---|---|---|---|")
    for host, w in HOSTS:
        n = per_host[host]
        wall = n / w * EST_SHARD_HOURS
        L.append(f"| {host} | {w} | {n} | ~{wall:.1f} h |")
    crit = max(per_host[h] / w for h, w in HOSTS) * EST_SHARD_HOURS
    L.append("")
    L.append(f"**Critical-path wall-clock ≈ {crit:.1f} h** "
             f"(assumes ~{EST_SHARD_HOURS} h/shard avg for the 5000-ep + extension "
             f"schedule; Arm B converges faster, Arm A may run longer — treat as a "
             f"rough upper bound; measure on the first wave and re-estimate).")
    L.append("")
    L.append("## Launch")
    L.append("- Strong host (run locally): `bash results/tier1b/launch/run_strong.sh`")
    L.append("- Each desktop: copy the repo (or shared FS), then "
             "`bash results/tier1b/launch/run_<host>.sh`.")
    L.append("- Shards are idempotent (`run_shard.py` skips existing summaries), so a "
             "host can be re-run after interruption without redoing finished seeds.")
    L.append("- Collect: rsync each host's `results/tier1b/sweep/training_logs/` back, "
             "then `python3 evaluation/aggregate_arms.py --dir results/tier1b/sweep "
             "--out results/tier1b/sweep/arm_summary`.")
    L.append("")
    L.append("## Manifest")
    L.append("Full `(host, config, arm, seed, command)` table: "
             "`results/tier1b/RUN_MANIFEST.tsv` (one shard per row).")
    (REPO / "RUN_PLAN.md").write_text("\n".join(L) + "\n")

    print(f"wrote {len(written)} sweep configs, manifest ({len(shards)} shards), "
          f"{len(HOSTS)} launch scripts, RUN_PLAN.md")
    print("per-host:", per_host, "| critical path ~%.1fh" % crit)


if __name__ == "__main__":
    main()
