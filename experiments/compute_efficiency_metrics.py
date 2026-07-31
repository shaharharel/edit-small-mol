#!/usr/bin/env python3
"""
Compute classic medchem ligand-efficiency metrics for the 520K-candidate dashboard.

Metrics (all FREE - simple ratios of existing columns + RDKit fsp3):
  1. LLE  = pIC50 - LogP                                       (Hopkins 2014)
  2. LE   = (1.4 * pIC50) / HeavyAtoms                         (Hopkins, Groom, Alex 2004)
  3. BEI  = (pIC50 * 1000) / MW                                (Abad-Zapatero & Metz 2005)
  4. SEI  = (pIC50 * 100)  / TPSA                              (Abad-Zapatero & Metz 2005)
  5. SILE = pIC50 - 0.6*LogP - 0.0035*MW + 4.0                 (Reynolds 2008)
  6. fsp3 = Lipinski.FractionCSP3(mol)                         (Lovering 2009)
  7. Lipinski_violations = sum([MW>500, LogP>5, HBA>10, HBD>5])

Inputs:
  results/paper_evaluation/all_methods_bulk_scored_v4.csv

Outputs:
  data/paper_evaluation/efficiency_metrics.csv
"""

from __future__ import annotations

import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import Lipinski

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
INPUT_CSV = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
OUTPUT_DIR = PROJECT_ROOT / "data" / "paper_evaluation"
OUTPUT_CSV = OUTPUT_DIR / "efficiency_metrics.csv"

N_WORKERS = 8


def _fsp3_one(smi: str) -> float:
    """Return Lipinski.FractionCSP3 for a SMILES, NaN on parse failure."""
    if not isinstance(smi, str) or not smi:
        return np.nan
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return np.nan
        return float(Lipinski.FractionCSP3(mol))
    except Exception:
        return np.nan


def compute_fsp3_parallel(smiles: list[str], n_workers: int = N_WORKERS) -> np.ndarray:
    """Compute fsp3 for a list of SMILES using a worker pool."""
    with Pool(processes=n_workers) as pool:
        out = pool.map(_fsp3_one, smiles, chunksize=1024)
    return np.asarray(out, dtype=float)


def main() -> None:
    print(f"Loading {INPUT_CSV} ...")
    df = pd.read_csv(INPUT_CSV)
    print(f"  Loaded {len(df):,} rows  cols={len(df.columns)}")

    # The backend applies dashboard filters (drop Tier 4 / Method A,B; require
    # warhead_intact; drop disconnected SMILES) AFTER load. We mirror the same
    # logic here so row_id matches the dashboard's row_id (post-filter index).
    if "method" in df.columns:
        df["method"] = df["method"].replace({
            "Tier 1 — Med-Chem Playbook (rule-based)": "Medchem Rules",
            "Tier 1.5 — Warhead Controls + Med-Chem Tricks": "Medchem Rules",
            "Tier 2 — Fragment Replacement (curated 204)": "Amine Replacements",
            "Tier 2 SCALED — Fragment Replacement (498K)": "Amine Replacements",
            "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)": "Amine Replacements",
        })
        dropped = [
            "Tier 4 — De Novo unconstrained",
            "Tier 4 — Mol2Mol unconstrained",
            "Method A — De Novo FiLMDelta-driven",
            "Method B — Mol2Mol FiLMDelta-driven",
        ]
        before = len(df)
        df = df[~df["method"].isin(dropped)].reset_index(drop=True)
        print(f"  After dropping Tier 4 / Methods A,B: {before:,} -> {len(df):,}")

    if "warhead_intact" in df.columns:
        before = len(df)
        df = df[df["warhead_intact"] == True].reset_index(drop=True)
        print(f"  After warhead_intact==True: {before:,} -> {len(df):,}")

    if "smiles" in df.columns:
        before = len(df)
        df = df[~df["smiles"].astype(str).str.contains(".", regex=False, na=False)].reset_index(drop=True)
        print(f"  After dropping disconnected SMILES: {before:,} -> {len(df):,}")

    df["row_id"] = df.index
    print(f"  Final row count: {len(df):,}")

    # Pull existing columns (cast to numeric, coerce errors -> NaN)
    pIC50 = pd.to_numeric(df["pIC50_method"], errors="coerce")
    MW = pd.to_numeric(df["MW"], errors="coerce")
    LogP = pd.to_numeric(df["LogP"], errors="coerce")
    HeavyAtoms = pd.to_numeric(df["HeavyAtoms"], errors="coerce")
    TPSA = pd.to_numeric(df["TPSA"], errors="coerce")
    HBA = pd.to_numeric(df["HBA"], errors="coerce") if "HBA" in df.columns else pd.Series([np.nan] * len(df))
    HBD = pd.to_numeric(df["HBD"], errors="coerce") if "HBD" in df.columns else pd.Series([np.nan] * len(df))

    # --- Vectorised metrics ---
    print("Computing vectorised metrics (LLE, LE, BEI, SEI, SILE) ...")
    LLE = pIC50 - LogP
    LE = np.where(HeavyAtoms > 0, (1.4 * pIC50) / HeavyAtoms, np.nan)
    BEI = np.where(MW > 0, (pIC50 * 1000.0) / MW, np.nan)
    SEI = np.where(TPSA > 0, (pIC50 * 100.0) / TPSA, np.nan)
    SILE = pIC50 - 0.6 * LogP - 0.0035 * MW + 4.0

    # Propagate NaN where any input was NaN
    LE = pd.Series(LE, index=df.index).where(pIC50.notna() & HeavyAtoms.notna(), np.nan)
    BEI = pd.Series(BEI, index=df.index).where(pIC50.notna() & MW.notna(), np.nan)
    SEI = pd.Series(SEI, index=df.index).where(pIC50.notna() & TPSA.notna(), np.nan)

    # Lipinski violations - count of: MW>500, LogP>5, HBA>10, HBD>5
    print("Computing Lipinski violations ...")
    viol = (
        (MW > 500).astype(int)
        + (LogP > 5).astype(int)
        + (HBA > 10).astype(int)
        + (HBD > 5).astype(int)
    )
    # If a required column is NaN, the corresponding flag is NaN.
    mask_any_nan = MW.isna() | LogP.isna() | HBA.isna() | HBD.isna()
    Lipinski_violations = viol.astype(float)
    Lipinski_violations[mask_any_nan] = np.nan

    # --- fsp3 via multiprocessing ---
    print(f"Computing fsp3 with {N_WORKERS} workers ...")
    smiles_list = df["smiles"].astype(str).tolist()
    fsp3 = compute_fsp3_parallel(smiles_list, N_WORKERS)

    out = pd.DataFrame({
        "row_id": df["row_id"].astype(int).values,
        "LLE": LLE.values,
        "LE": LE.values,
        "BEI": BEI.values,
        "SEI": SEI.values,
        "SILE": SILE.values,
        "fsp3": fsp3,
        "Lipinski_violations": Lipinski_violations.values,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out.to_csv(OUTPUT_CSV, index=False)
    print(f"\nWrote {OUTPUT_CSV} ({len(out):,} rows)")

    # ── Sanity check ─────────────────────────────────────────────────────────
    print("\n=== Distribution summary ===")
    desc = out[["LLE", "LE", "BEI", "SEI", "SILE", "fsp3", "Lipinski_violations"]].describe(
        percentiles=[0.05, 0.5, 0.95]
    )
    print(desc.to_string())

    print("\nLipinski_violations value counts:")
    print(out["Lipinski_violations"].value_counts(dropna=False).sort_index().to_string())

    # Merge SMILES + pIC50 for top-K display
    merged = out.merge(
        df[["row_id", "smiles", "method", "pIC50_method", "MW", "LogP", "HeavyAtoms", "TPSA"]],
        on="row_id", how="left",
    )

    for metric in ["LLE", "LE", "BEI", "SEI", "SILE"]:
        top = merged.nlargest(5, metric)[
            ["row_id", "smiles", "method", "pIC50_method", "MW", "LogP", metric]
        ]
        print(f"\nTop-5 by {metric}:")
        print(top.to_string(index=False))

    # ── Correlation matrix ───────────────────────────────────────────────────
    corr_df = merged[["pIC50_method", "LLE", "LE", "BEI", "SEI", "SILE", "fsp3",
                       "Lipinski_violations"]]
    corr = corr_df.corr(method="pearson")
    print("\n=== Pearson correlation matrix ===")
    print(corr.round(3).to_string())


if __name__ == "__main__":
    main()
