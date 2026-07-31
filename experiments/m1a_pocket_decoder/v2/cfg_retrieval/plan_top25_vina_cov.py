"""Plan top-25/cell for the --local_only follow-up pass.

Reads covalent_vina_cov_panel*.csv (from --score_only pass), ranks per cell
by ascending vina_cov_score (lower = better clashy score), takes top-25.
Writes cofold_plan_top25_vina_cov.csv (columns: cell, sample_idx, smi,
vina_cov_score).

Used by `vina_covalent_batch.py --mode local_only --plan_csv ...` for the
follow-up pass that gives usable absolute affinities.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_csv", required=True,
                    help="covalent_vina_cov_panel*.csv from --score_only pass")
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--per_cell", type=int, default=25)
    args = ap.parse_args()

    d = pd.read_csv(args.in_csv)
    d = d[d["vina_cov_ok"] == True].copy()
    d["vina_cov_score"] = d["vina_cov_score"].astype(float)
    d = d.dropna(subset=["vina_cov_score"])
    plan = d.sort_values(["cell", "vina_cov_score"]).groupby("cell").head(args.per_cell)
    plan = plan[["cell", "sample_idx", "smi", "vina_cov_score"]]
    plan.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}  (n={len(plan)})", flush=True)
    print(plan.groupby("cell").agg(n=("sample_idx","size"),
                                        min_score=("vina_cov_score","min"),
                                        median_score=("vina_cov_score","median")))


if __name__ == "__main__":
    main()
