"""Merge MM-GBSA 838 single-frame results into data/tier4_scored/F4_boltz_full.csv.

Join key: smiles (canonicalized via RDKit if needed; raw match should also work
since both were derived from the same backend SMILES strings).

Backs up the original F4_boltz_full.csv to a timestamped .bak before writing.
"""
from __future__ import annotations

import argparse
import shutil
from datetime import datetime
from pathlib import Path

import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mmgbsa", required=True, type=Path,
                    help="Path to mmgbsa_838_singleframe.csv")
    ap.add_argument("--f4", default=Path("data/tier4_scored/F4_boltz_full.csv"), type=Path)
    args = ap.parse_args()

    mmgbsa = pd.read_csv(args.mmgbsa)
    print(f"MMGBSA results: {len(mmgbsa)} rows ({(mmgbsa['status'] == 'OK').sum()} OK)")
    print(f"  dG_GB_kcalmol distribution: min={mmgbsa['dG_GB_kcalmol'].min():.2f}, "
          f"med={mmgbsa['dG_GB_kcalmol'].median():.2f}, "
          f"max={mmgbsa['dG_GB_kcalmol'].max():.2f}")

    f4 = pd.read_csv(args.f4, low_memory=False)
    print(f"F4_boltz_full: {len(f4)} rows")

    # Keep only OK rows from MMGBSA
    ok = mmgbsa[mmgbsa["status"] == "OK"].copy()
    new_cols = ["dG_GB_kcalmol", "ggas_kcalmol", "gsolv_kcalmol",
                "E_vdw_kcalmol", "E_eel_kcalmol"]
    keep = ok[["smiles"] + new_cols].dropna(subset=["smiles"])
    keep = keep.drop_duplicates("smiles", keep="first")
    print(f"Joining {len(keep)} OK MMGBSA rows by SMILES")

    # Backup
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    bak = args.f4.with_suffix(f".csv.bak.mmgbsa_{ts}")
    shutil.copy(args.f4, bak)
    print(f"Backup: {bak}")

    # Merge: overwrite ggas_kcalmol and gsolv_kcalmol with new values (the old
    # ones were NoneType placeholders from the MD protocol that never ran).
    # New columns: dG_GB_kcalmol, E_vdw_kcalmol, E_eel_kcalmol.
    for col in new_cols:
        f4[col] = pd.NA  # ensure column exists
    # Map smiles -> col value
    smap = keep.set_index("smiles")
    matched = 0
    for idx, row in f4.iterrows():
        smi = row["smiles"]
        if smi in smap.index:
            for col in new_cols:
                f4.at[idx, col] = smap.at[smi, col]
            matched += 1
    print(f"Matched {matched}/{len(f4)} F4 rows by smiles")

    f4.to_csv(args.f4, index=False)
    print(f"Wrote {args.f4} with {len(new_cols)} new MM-GBSA columns")
    print(f"Distribution in merged: "
          f"dG_GB_kcalmol coverage = {f4['dG_GB_kcalmol'].notna().sum()} / {len(f4)}")


if __name__ == "__main__":
    main()
