#!/usr/bin/env python3
"""
Compute 3D similarity metrics for ALL 533K candidates on ai-chem (16-core).

Loads all_methods_bulk_scored_v3.csv (top-100 already has 3D), and fills in the
remaining ~520K. Saves as v4.

Skip optimizations:
  - Rows that already have non-NaN shape_Tc_seed are skipped (top-100 per method)
  - Multiprocessing with chunksize=8 to amortize worker init
  - Periodic checkpointing to v4_partial.csv every 50K mols (resumable)
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


RES = PROJECT_ROOT / "results" / "paper_evaluation"
INPUT_CSV = RES / "all_methods_bulk_scored_v3.csv"
OUTPUT_CSV = RES / "all_methods_bulk_scored_v4.csv"
PARTIAL_CSV = RES / "all_methods_bulk_scored_v4_partial.csv"

N_PROCS = max(1, mp.cpu_count() - 1)
CHECKPOINT_EVERY = 25000


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def _score_one(smi):
    """Module-level worker. Imports inside to avoid pickling issues."""
    if not isinstance(smi, str):
        return None
    # Skip disconnected SMILES
    if "." in smi:
        return None
    try:
        from src.utils.mol1_scoring import (
            shape_tanimoto_seed, esp_sim_seed, warhead_vector_deviation,
        )
        return {
            "smiles": smi,
            "shape_Tc_seed": shape_tanimoto_seed(smi),
            "esp_sim_seed": esp_sim_seed(smi),
            "warhead_dev_deg": warhead_vector_deviation(smi),
        }
    except Exception:
        return {"smiles": smi, "shape_Tc_seed": float('nan'),
                "esp_sim_seed": float('nan'), "warhead_dev_deg": float('nan')}


def main():
    log("=" * 70)
    log("3D METRICS FOR ALL 533K CANDIDATES (ai-chem 16-core)")
    log("=" * 70)
    log(f"Workers: {N_PROCS}")

    if not INPUT_CSV.exists():
        log(f"ERROR: {INPUT_CSV} not found"); return
    df = pd.read_csv(INPUT_CSV)
    log(f"Loaded {len(df):,} from v3")

    # Ensure 3D columns exist
    for c in ["shape_Tc_seed", "esp_sim_seed", "warhead_dev_deg"]:
        if c not in df.columns:
            df[c] = np.nan

    # Identify rows that need 3D
    needs = df["shape_Tc_seed"].isna() & df["smiles"].astype(str).str.contains(".", regex=False, na=False).eq(False) & df["smiles"].notna()
    todo_idx = df.index[needs].tolist()
    log(f"Rows already with 3D: {(~needs).sum():,}")
    log(f"Rows to compute:      {len(todo_idx):,}")

    # Resume support: if v4_partial exists, prefer its values
    if PARTIAL_CSV.exists():
        log(f"Found {PARTIAL_CSV.name} — restoring partial results...")
        prev = pd.read_csv(PARTIAL_CSV)
        # Match by row_id
        if "row_id" in prev.columns and "row_id" in df.columns:
            prev_idx = prev.set_index("row_id")
            for col in ["shape_Tc_seed", "esp_sim_seed", "warhead_dev_deg"]:
                if col in prev_idx.columns:
                    df.loc[df["row_id"].isin(prev_idx.index), col] = df.loc[
                        df["row_id"].isin(prev_idx.index), "row_id"
                    ].map(prev_idx[col])
            new_needs = df["shape_Tc_seed"].isna() & df["smiles"].astype(str).str.contains(".", regex=False, na=False).eq(False) & df["smiles"].notna()
            todo_idx = df.index[new_needs].tolist()
            log(f"After restore, rows to compute: {len(todo_idx):,}")

    smiles_list = df.loc[todo_idx, "smiles"].tolist()
    log(f"\nProcessing {len(smiles_list):,} mols on {N_PROCS} workers...")

    n_done = 0
    last_checkpoint = 0
    last_log = 0

    with mp.Pool(N_PROCS) as pool:
        results_buf = []
        for r in pool.imap(_score_one, smiles_list, chunksize=4):
            results_buf.append(r)
            n_done += 1
            if n_done - last_log >= 5000:
                last_log = n_done
                pct = 100*n_done/len(smiles_list)
                log(f"  {n_done:,}/{len(smiles_list):,} ({pct:.1f}%)")

            # Checkpoint every CHECKPOINT_EVERY mols
            if n_done - last_checkpoint >= CHECKPOINT_EVERY:
                # Apply buffered results so far
                for j, res in enumerate(results_buf):
                    if res is not None:
                        global_idx = todo_idx[last_checkpoint + j]
                        df.at[global_idx, "shape_Tc_seed"] = res["shape_Tc_seed"]
                        df.at[global_idx, "esp_sim_seed"] = res["esp_sim_seed"]
                        df.at[global_idx, "warhead_dev_deg"] = res["warhead_dev_deg"]
                df.to_csv(PARTIAL_CSV, index=False)
                last_checkpoint = n_done
                results_buf = []
                log(f"  Checkpoint saved at {n_done:,}")

    # Apply any remaining buffered results
    for j, res in enumerate(results_buf):
        if res is not None:
            global_idx = todo_idx[last_checkpoint + j]
            df.at[global_idx, "shape_Tc_seed"] = res["shape_Tc_seed"]
            df.at[global_idx, "esp_sim_seed"] = res["esp_sim_seed"]
            df.at[global_idx, "warhead_dev_deg"] = res["warhead_dev_deg"]

    df.to_csv(OUTPUT_CSV, index=False)
    log(f"\nSaved → {OUTPUT_CSV} ({OUTPUT_CSV.stat().st_size/1024/1024:.0f} MB)")
    n_3d = df["shape_Tc_seed"].notna().sum()
    log(f"  3D scored: {n_3d:,} / {len(df):,}")
    log(f"  shape_Tc range: {df['shape_Tc_seed'].min():.3f}–{df['shape_Tc_seed'].max():.3f}")
    log(f"  esp_sim range:  {df['esp_sim_seed'].min():.3f}–{df['esp_sim_seed'].max():.3f}")
    log(f"  warhead_dev:    {df['warhead_dev_deg'].min():.1f}–{df['warhead_dev_deg'].max():.1f}°")
    log(f"\nDone: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
