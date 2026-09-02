"""
Tier 1B full sweep — shard-completion ledger. Scans RUN_MANIFEST.tsv against the
actual per-seed summaries under results/tier1b/sweep/training_logs/ and reports
done / missing per host and per (config, arm). Never fabricates: a shard with no
summary is reported MISSING. Used for monitoring and for the final report.

Usage:
  python3 evaluation/sweep_ledger.py                 # human summary
  python3 evaluation/sweep_ledger.py --emit-missing  # print resume commands
  python3 evaluation/sweep_ledger.py --json results/tier1b/sweep/ledger.json
"""

import argparse
import json
import re
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
MANIFEST = REPO / "results" / "tier1b" / "RUN_MANIFEST.tsv"
TLOG = REPO / "results" / "tier1b" / "sweep" / "training_logs"


def parse_manifest():
    rows = []
    for line in MANIFEST.read_text().splitlines()[1:]:
        host, cbase, arm, seed, cmd = line.split("\t")
        rows.append({"host": host, "config": cbase, "arm": arm,
                     "seed": int(seed), "cmd": cmd})
    return rows


def shard_done(cbase, arm, seed):
    p = TLOG / f"sweep_{cbase}_{arm}_seed{seed}.json"
    return p.exists()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--emit-missing", action="store_true")
    ap.add_argument("--host", default=None, help="filter to one host")
    ap.add_argument("--json", default=None)
    args = ap.parse_args()

    rows = parse_manifest()
    for r in rows:
        r["done"] = shard_done(r["config"], r["arm"], r["seed"])

    by_host = defaultdict(lambda: [0, 0])
    by_cfg = defaultdict(lambda: [0, 0])
    for r in rows:
        by_host[r["host"]][0] += int(r["done"]); by_host[r["host"]][1] += 1
        by_cfg[(r["config"], r["arm"])][0] += int(r["done"])
        by_cfg[(r["config"], r["arm"])][1] += 1

    total_done = sum(r["done"] for r in rows)
    print(f"=== SWEEP LEDGER: {total_done}/{len(rows)} shards complete ===")
    print("\nBy host:")
    for h in sorted(by_host):
        d, n = by_host[h]
        print(f"  {h:10s} {d:3d}/{n:3d}")
    print("\nBy config x arm (done/total):")
    for (c, a) in sorted(by_cfg):
        d, n = by_cfg[(c, a)]
        print(f"  {c:4s} {a}  {d:2d}/{n:2d}")

    missing = [r for r in rows if not r["done"]]
    if args.host:
        missing = [r for r in missing if r["host"] == args.host]
    if args.emit_missing:
        print(f"\n# {len(missing)} missing shard commands"
              + (f" (host={args.host})" if args.host else ""))
        for r in missing:
            print(r["cmd"])

    if args.json:
        Path(args.json).write_text(json.dumps({
            "total": len(rows), "done": total_done,
            "by_host": {h: by_host[h] for h in by_host},
            "by_config_arm": {f"{c}_{a}": by_cfg[(c, a)] for (c, a) in by_cfg},
            "missing": [{"host": r["host"], "config": r["config"], "arm": r["arm"],
                         "seed": r["seed"]} for r in missing],
        }, indent=2))
        print(f"\nwrote {args.json}")


if __name__ == "__main__":
    main()
