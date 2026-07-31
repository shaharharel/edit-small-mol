"""Concatenate posefree shard CSVs into the canonical pose-free strain CSV.

Reads `data/paper_evaluation/posefree_shards/shard_*.csv` (each a full CSV with
header) and writes a single deduped CSV at
`data/paper_evaluation/rdkit_strain_posefree.csv`.

Safe to call at any time while the shard producers are still running -- it
only takes a consistent snapshot of whatever has been flushed.

Usage:
    python experiments/merge_posefree_shards.py
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SHARD_DIR = PROJECT_ROOT / "data" / "paper_evaluation" / "posefree_shards"
OUT_PATH = PROJECT_ROOT / "data" / "paper_evaluation" / "rdkit_strain_posefree.csv"

FIELDS = ["row_id", "strain_kcal_mol", "e_bound_kcal_mol",
          "e_free_kcal_mol", "success_flag", "error"]


def main() -> int:
    if not SHARD_DIR.exists():
        print(f"No shard dir at {SHARD_DIR}", file=sys.stderr)
        return 1
    rows_by_id: dict[int, dict] = {}
    n_seen = 0
    for f in sorted(SHARD_DIR.glob("shard_*.csv")):
        if f.stat().st_size == 0:
            continue
        with open(f) as fh:
            r = csv.DictReader(fh)
            for row in r:
                n_seen += 1
                try:
                    rid = int(float(row["row_id"]))
                except (KeyError, ValueError, TypeError):
                    continue
                # Last write wins.
                rows_by_id[rid] = row
    OUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_PATH, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=FIELDS)
        w.writeheader()
        for rid in sorted(rows_by_id):
            row = rows_by_id[rid]
            w.writerow({k: row.get(k, "") for k in FIELDS})
    print(f"Merged {n_seen} raw rows -> {len(rows_by_id)} unique row_ids -> {OUT_PATH}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
