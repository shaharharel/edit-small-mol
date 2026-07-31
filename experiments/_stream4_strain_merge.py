"""Pull partial/final ai-chem strain results and merge into the main backfill CSV.

Run on demand:
    python experiments/_stream4_strain_merge.py
"""
import subprocess, sys, time
from pathlib import Path
import numpy as np
import pandas as pd

ROOT = Path('/Users/shaharharel/Documents/github/edit-small-mol')
OUT = ROOT/'data/paper_evaluation/l3_cheap_backfill.csv'
COL = "rdkit_strain_posefree_kcal_mol"

REMOTE_FILES = ["strain_aichem_out.csv", "strain_mac_out.csv"]
LOCAL_TMP = Path('/tmp/strain_remote_pull')
LOCAL_TMP.mkdir(parents=True, exist_ok=True)

def pull():
    """SCP both partial CSVs back from ai-chem."""
    for fn in REMOTE_FILES:
        local = LOCAL_TMP / fn
        cmd = ["gcloud", "compute", "scp",
               f"ai-chem:/home/shaharh_quris_ai/edit-small-mol/{fn}",
               str(local),
               "--zone=us-east1-b"]
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        if r.returncode == 0:
            print(f"  [pull] {fn} -> {local} ({local.stat().st_size} bytes)")
        else:
            print(f"  [pull] {fn} FAILED: {r.stderr[:200]}")

def merge():
    pull()
    out_df = pd.read_csv(OUT)
    n_pre = out_df[COL].notna().sum()
    print(f"[merge] CSV pre-merge: {n_pre:,} strain values filled")

    all_new = []
    for fn in REMOTE_FILES:
        p = LOCAL_TMP / fn
        if not p.exists() or p.stat().st_size == 0:
            print(f"  [skip] {fn} (missing or empty)")
            continue
        try:
            d = pd.read_csv(p)
        except Exception as e:
            print(f"  [skip] {fn} parse error: {e}")
            continue
        d = d.dropna(subset=[COL])
        print(f"  {fn}: {len(d):,} rows with strain")
        all_new.append(d)

    if not all_new:
        print("[merge] nothing to add")
        return

    new = pd.concat(all_new, ignore_index=True).drop_duplicates("row_id", keep="last")
    lookup = dict(zip(new["row_id"], new[COL]))
    mapped = out_df["row_id"].map(lookup)
    mask = mapped.notna() & out_df[COL].isna()
    out_df.loc[mask, COL] = mapped[mask].values
    n_post = out_df[COL].notna().sum()
    print(f"[merge] +{n_post - n_pre:,} new strain values added -> {n_post:,} total "
          f"({100*n_post/len(out_df):.1f}%)")
    out_df.to_csv(OUT, index=False)
    print(f"[merge] wrote {OUT}")

if __name__ == "__main__":
    merge()
