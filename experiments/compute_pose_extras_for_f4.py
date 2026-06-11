"""Compute warhead_dev_deg + n_contacts_total + n_salt_bridges +
atp_pocket_fraction for all 2,221 F4 Boltz mols and merge into
data/tier4_scored/F4_boltz_full.csv (+ the two _with_boltz tier CSVs).

Sources:
  - warhead_dev_deg → src.utils.mol1_scoring.warhead_vector_deviation (SMILES-only)
  - contacts (covalent + h_bonds + salt + pi) → anchordiff/compute_contacts_occupancy.py
  - atp_pocket_fraction → anchordiff/compute_pose_quality_v2.py

Usage: python experiments/compute_pose_extras_for_f4.py [--workers 8]
"""
from __future__ import annotations
import sys
import json
import argparse
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))
sys.path.insert(0, str(PROJECT / "experiments"))
sys.path.insert(0, str(PROJECT / "anchordiff"))

import pandas as pd
import numpy as np

# Index of cofold dirs from existing pipeline
from compute_full_boltz_metrics import build_row_id_index


# ── per-row compute ──────────────────────────────────────────────────────────

def compute_pose_cols(row_id_smiles_cif_tuple):
    """Worker: takes (row_id, smiles, cif_path_str) → dict of new col values."""
    row_id, smiles, cif_path_str = row_id_smiles_cif_tuple
    out = {
        "row_id": row_id,
        "warhead_dev_deg": None,
        "n_contacts_total": None,
        "n_salt_bridges": None,
        "atp_pocket_fraction": None,
        "error": None,
    }

    # 1) warhead_dev_deg from SMILES (no CIF needed)
    if smiles:
        try:
            from src.utils.mol1_scoring import warhead_vector_deviation
            val = warhead_vector_deviation(smiles)
            if val is not None and np.isfinite(val):
                out["warhead_dev_deg"] = float(val)
        except Exception as e:
            out["error"] = f"warhead_dev:{e}"

    # 2) Pose-based: contacts + atp_pocket_fraction
    if cif_path_str and Path(cif_path_str).exists() and Path(cif_path_str).stat().st_size > 100:
        try:
            import gemmi
            from compute_contacts_occupancy import (
                gather_atoms, find_covalent, find_h_bonds,
                find_salt_bridges, find_pi_pi,
            )
            from compute_pose_quality_v2 import atp_pocket_fraction as atp_pf

            st = gemmi.read_structure(str(cif_path_str))
            prot, lig = gather_atoms(st)
            if prot and lig:
                n_cov, _ = find_covalent(prot, lig)
                n_hb, _  = find_h_bonds(prot, lig)
                n_sb, _  = find_salt_bridges(prot, lig)
                n_pi, _  = find_pi_pi(prot, lig, smiles or "")
                out["n_contacts_total"] = int(n_cov + n_hb + n_sb + n_pi)
                out["n_salt_bridges"]   = int(n_sb)
                apf = atp_pf(prot, lig, cutoff_A=4.0)
                if apf is not None and np.isfinite(apf):
                    out["atp_pocket_fraction"] = float(apf)
        except Exception as e:
            err = f"pose:{e}"
            out["error"] = (out["error"] + "|" + err) if out["error"] else err

    return out


# ── driver ───────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit", type=int, default=None,
                    help="Process only this many rows (for smoke testing)")
    args = ap.parse_args()

    # Build index
    print("Building row_id → cofold dir index...", flush=True)
    t0 = time.time()
    index = build_row_id_index()
    print(f"  indexed {len(index):,} cofold dirs in {time.time()-t0:.1f}s", flush=True)

    # Load both pool + extra
    pool_path = PROJECT / "data/tier4_scored/F4_boltz_pool_v2_with_boltz.csv"
    extra_path = PROJECT / "data/tier4_scored/F4_boltz_extra_v2_with_boltz.csv"

    work = []
    for csv_path, prefix in [(pool_path, "pool"), (extra_path, "extra")]:
        df = pd.read_csv(csv_path, low_memory=False)
        # row_id resolution
        if "row_id" in df.columns:
            id_col = "row_id"
        else:
            # extra uses EXTRA_NNNNN positional cofold names
            id_col = "_cofold_name"
            pref = "row" if prefix == "pool" else "EXTRA_"
            df[id_col] = [f"{pref}{i:05d}" for i in range(len(df))]

        for i, row in df.iterrows():
            rid = str(row[id_col])
            pred_dir = index.get(rid)
            cif_path = None
            if pred_dir is not None and pred_dir.exists():
                cofold_name = pred_dir.name
                cif = pred_dir / f"{cofold_name}_model_0.cif"
                if cif.exists():
                    cif_path = str(cif)
            work.append((rid, str(row.get("smiles", "")), cif_path, prefix, i))

    if args.limit:
        work = work[:args.limit]
    print(f"Processing {len(work):,} mols with {args.workers} workers...", flush=True)

    # Process in parallel
    results = {}  # (prefix, df_idx) → dict
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(compute_pose_cols, (rid, smi, cif)): (prefix, idx)
                for rid, smi, cif, prefix, idx in work}
        done = 0
        for fut in as_completed(futs):
            prefix, idx = futs[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"error": f"worker_died:{e}"}
            results[(prefix, idx)] = res
            done += 1
            if done % 100 == 0 or done == len(work):
                elapsed = time.time() - t0
                rate = done / elapsed if elapsed > 0 else 0
                eta = (len(work) - done) / rate if rate > 0 else float('inf')
                print(f"  [{done:5,}/{len(work):,}]  "
                      f"elapsed {elapsed:5.0f}s  rate {rate:.1f}/s  ETA {eta:5.0f}s",
                      flush=True)

    print(f"\nDone computing in {time.time()-t0:.0f}s\n", flush=True)

    # Write back into each CSV
    NEW_COLS = ["warhead_dev_deg", "n_contacts_total",
                "n_salt_bridges", "atp_pocket_fraction"]
    for csv_path, prefix in [(pool_path, "pool"), (extra_path, "extra")]:
        df = pd.read_csv(csv_path, low_memory=False)
        for c in NEW_COLS:
            df[c] = pd.NA
        n_filled = {c: 0 for c in NEW_COLS}
        for i in range(len(df)):
            r = results.get((prefix, i))
            if not r:
                continue
            for c in NEW_COLS:
                v = r.get(c)
                if v is not None:
                    df.at[i, c] = v
                    n_filled[c] += 1
        df.to_csv(csv_path, index=False)
        print(f"{prefix:6s}: wrote {csv_path.name}  cols filled: {n_filled}",
              flush=True)

    # Rebuild F4_boltz_full.csv
    m = pd.read_csv(pool_path)
    e = pd.read_csv(extra_path)
    m["_tier"] = "main"
    e["_tier"] = "rescue"
    common = [c for c in m.columns if c in e.columns]
    full = pd.concat([m[common], e[common]], ignore_index=True)
    full_path = PROJECT / "data/tier4_scored/F4_boltz_full.csv"
    full.to_csv(full_path, index=False)
    print(f"\nRebuilt {full_path.name}: {len(full):,} rows", flush=True)
    for c in NEW_COLS:
        if c in full.columns:
            n_pop = full[c].notna().sum()
            print(f"  {c:30s}: {n_pop:,}/{len(full):,} ({n_pop/len(full)*100:.1f}%)")


if __name__ == "__main__":
    main()
