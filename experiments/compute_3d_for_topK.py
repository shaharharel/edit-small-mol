#!/usr/bin/env python3
"""
Compute 3D similarity metrics for the top-100 candidates per method, merge into
all_methods_bulk_scored_v2.csv → v3.csv.

Adds columns:
  - shape_Tc_seed     : RDKit O3A-aligned Gaussian-volume Tanimoto vs Mol 1 (4-conformer ensemble, best alignment)
  - esp_sim_seed      : Espsim Gasteiger-charge electrostatic Tanimoto vs Mol 1
  - warhead_dev_deg   : MIN angle (degrees) between Mol 1's C=C-C(=O)-N vector and candidate's,
                        over the conformer ensemble after O3A alignment

Outside top-100, columns are NaN.

Runtime: ~20 min on local 8-core for 1200 mols.
"""

import sys
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

from src.utils.mol1_scoring import (
    shape_tanimoto_seed, esp_sim_seed, warhead_vector_deviation,
)

RES = PROJECT_ROOT / "results" / "paper_evaluation"
INPUT_CSV = RES / "all_methods_bulk_scored_v2.csv"
OUTPUT_CSV = RES / "all_methods_bulk_scored_v3.csv"

TOP_K_PER_METHOD = 100
N_PROCS = max(1, mp.cpu_count() - 2)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _score_one(smi):
    """Worker: compute 3D similarities for one SMILES."""
    if not isinstance(smi, str):
        return None
    try:
        return {
            "smiles": smi,
            "shape_Tc_seed": shape_tanimoto_seed(smi),
            "esp_sim_seed": esp_sim_seed(smi),
            "warhead_dev_deg": warhead_vector_deviation(smi),
        }
    except Exception as e:
        return {"smiles": smi, "shape_Tc_seed": float('nan'),
                "esp_sim_seed": float('nan'), "warhead_dev_deg": float('nan')}


def main():
    log("=" * 70)
    log("3D SIMILARITY METRICS FOR TOP-K PER METHOD")
    log("=" * 70)

    if not INPUT_CSV.exists():
        log(f"ERROR: {INPUT_CSV} not found"); return
    df = pd.read_csv(INPUT_CSV)
    log(f"Loaded {len(df):,}")

    # Pick top-K per method by best available pIC50 (prefer ensemble mean if present)
    pic_col = "pIC50_mean" if "pIC50_mean" in df.columns else (
              "pIC50_method" if "pIC50_method" in df.columns else "pIC50_film")
    log(f"Sorting by {pic_col} desc, picking top-{TOP_K_PER_METHOD} per method...")
    df = df.sort_values(pic_col, ascending=False, na_position="last").reset_index(drop=True)
    df["row_id"] = df.index

    # Also drop disconnected SMILES (consistent with backend filter)
    df_clean = df[~df["smiles"].astype(str).str.contains(".", regex=False, na=False)]

    top_idx = []
    for m, sub in df_clean.groupby("method"):
        top_idx.extend(sub.head(TOP_K_PER_METHOD).index.tolist())
    log(f"  {len(top_idx):,} candidates to score (3D)")

    # Init NaN columns for everyone
    for c in ["shape_Tc_seed", "esp_sim_seed", "warhead_dev_deg"]:
        df[c] = np.nan

    smiles_list = df.loc[top_idx, "smiles"].tolist()

    log(f"Workers: {N_PROCS}")
    log(f"This will take ~{len(smiles_list)/30/60:.0f}-{len(smiles_list)/15/60:.0f} min...")
    n_done = 0
    last_log = 0
    with mp.Pool(N_PROCS) as pool:
        results = []
        for r in pool.imap(_score_one, smiles_list, chunksize=4):
            results.append(r)
            n_done += 1
            if n_done - last_log >= 100:
                last_log = n_done
                log(f"  {n_done:,}/{len(smiles_list):,} done")

    log("Writing results...")
    for j, top_j in enumerate(top_idx):
        if results[j] is not None:
            df.at[top_j, "shape_Tc_seed"] = results[j]["shape_Tc_seed"]
            df.at[top_j, "esp_sim_seed"] = results[j]["esp_sim_seed"]
            df.at[top_j, "warhead_dev_deg"] = results[j]["warhead_dev_deg"]

    df.to_csv(OUTPUT_CSV, index=False)
    log(f"Saved → {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size/1024/1024:.0f} MB)")
    n_3d = df["shape_Tc_seed"].notna().sum()
    log(f"  3D scored: {n_3d:,} / {len(df):,}")
    log(f"  shape_Tc range: {df['shape_Tc_seed'].min():.3f}–{df['shape_Tc_seed'].max():.3f}")
    log(f"  esp_sim range:  {df['esp_sim_seed'].min():.3f}–{df['esp_sim_seed'].max():.3f}")
    log(f"  warhead_dev:    {df['warhead_dev_deg'].min():.1f}–{df['warhead_dev_deg'].max():.1f}°")


if __name__ == "__main__":
    main()
