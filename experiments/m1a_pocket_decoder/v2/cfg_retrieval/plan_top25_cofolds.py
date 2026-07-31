"""Plan the TOP-25/cell cofold list by Vina score.

Reads covalent_vina_panel.csv, ranks per cell by ascending vina_free_score
(lower = better), takes top-25 (or fewer if cell < 25 valid), writes:

    data/paper_pair_training/cfg_retrieval/cofold_plan_top25.csv

with columns cell, sample_idx, smi, vina_free_score.

This is consumed by a modified batch_boltz_cofold.py.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vina_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "covalent_vina_panel.csv"))
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval/"
                     "cofold_plan_top25.csv"))
    ap.add_argument("--per_cell", type=int, default=25)
    args = ap.parse_args()

    d = pd.read_csv(args.vina_csv)
    d = d[d["vina_free_ok"] == True].copy()
    d["vina_free_score"] = d["vina_free_score"].astype(float)
    d = d.dropna(subset=["vina_free_score"])
    plan = d.sort_values(["cell", "vina_free_score"]).groupby("cell").head(args.per_cell)
    plan = plan[["cell", "sample_idx", "smi", "vina_free_score",
                    "vina_free_d_sg", "vina_free_bd_ang", "vina_free_phi_planar",
                    "bd_ready_vina"]]
    plan.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv}", flush=True)
    print(plan.groupby("cell").agg(n=("sample_idx", "size"),
                                        min_score=("vina_free_score", "min"),
                                        median_score=("vina_free_score", "median")))


if __name__ == "__main__":
    main()
