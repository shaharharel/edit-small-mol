#!/usr/bin/env python3
"""Regenerate the RL covgeom verdict report by reading existing summary_V*.json files.

Does NOT re-sample or re-score.  Only aggregates the summaries produced by
run_rl_covgeom_ablation.py.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))
from experiments.run_rl_covgeom_ablation import write_verdict_report, OUT_DATA


def main():
    summaries = {}
    for tag in ["V1", "V2", "V3", "V4", "V5"]:
        p = OUT_DATA / f"summary_{tag}.json"
        if p.exists():
            summaries[tag] = json.loads(p.read_text())
            print(f"Loaded {tag}: {p}", file=sys.stderr)
        else:
            print(f"Missing {tag}: {p}", file=sys.stderr)
    if not summaries:
        print("No summaries found — nothing to regenerate.", file=sys.stderr)
        return
    write_verdict_report(summaries)
    print(f"Report written to {OUT_DATA / 'rl_covgeom_report.md'}", file=sys.stderr)


if __name__ == "__main__":
    main()
