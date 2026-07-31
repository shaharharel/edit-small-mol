#!/usr/bin/env python3
"""Concat all /tmp/vv_*.csv vanilla-Vina shards, dedupe by row_id, split per cohort prefix.

Writes /tmp/vv_cohort_<tag>.csv per cohort and a manifest.
"""
import sys
from pathlib import Path
import pandas as pd
import re

SRC_FILES = [
    "/tmp/vv_chem_a.csv",
    "/tmp/vv_chem2_b.csv",
    "/tmp/vv_chem2_c11.csv",
    "/tmp/vv_chem3.csv",
]

# Map row_id prefix -> cohort tag used in cohort CSV naming
COHORT_TAGS = [
    "mol1RL_v5_seed_mol1_only",
    "thiq_rl_exp2_mol1only",
    "thiq_rl_exp2_zap70",
    "thiq_rl_exp2_kinase",
    "thiq_rl_mol1only",
    "thiq_rl_zap70",
    "thiq_rl_kinase",
    "murcko_rl_exp2_zap70",
    "murcko_rl_exp2_kinase",
    "murcko_rl_zap70",
    "murcko_rl_kinase",
]


def main():
    dfs = []
    for p in SRC_FILES:
        pth = Path(p)
        if not pth.exists():
            print(f"[skip] missing {p}")
            continue
        d = pd.read_csv(pth)
        d["__src"] = pth.name
        dfs.append(d)
        print(f"[load] {pth.name}: {len(d)} rows")
    if not dfs:
        print("no inputs")
        sys.exit(1)
    big = pd.concat(dfs, ignore_index=True)
    print(f"[concat] total {len(big)}")
    # Prefer ok_from_pdbqt status; drop NaN vina; dedupe keeping best (most negative) per row_id
    big = big.dropna(subset=["row_id", "vina_kcalmol"])
    # If a row_id has multiple, keep most negative
    big = big.sort_values("vina_kcalmol").drop_duplicates(subset="row_id", keep="first")
    print(f"[dedupe] kept {len(big)} unique row_ids")

    # Split by cohort prefix (sorted by length descending to ensure longest match)
    tags_sorted = sorted(COHORT_TAGS, key=len, reverse=True)
    rid_to_tag = {}
    for rid in big["row_id"].astype(str):
        for tag in tags_sorted:
            if rid.startswith(tag + "_"):
                rid_to_tag[rid] = tag
                break
    big["__cohort"] = big["row_id"].astype(str).map(rid_to_tag)
    unmapped = big[big["__cohort"].isna()]
    if len(unmapped):
        print(f"[warn] {len(unmapped)} unmapped row_ids; sample:")
        print(unmapped[["row_id"]].head(5).to_string())
    big = big.dropna(subset=["__cohort"])

    for tag, g in big.groupby("__cohort"):
        out = Path(f"/tmp/vv_cohort_{tag}.csv")
        g[["row_id", "smiles", "vina_kcalmol", "status"]].to_csv(out, index=False)
        print(f"[write] {out.name}: {len(g)} rows")


if __name__ == "__main__":
    main()
