#!/usr/bin/env python3
"""
Compute uniform bulk metrics for ALL methods, not just Tier 2 SCALED.

For every candidate across all 12 methods, compute:
  - smiles, method, row_id
  - pIC50_film (single-seed anchor-mean) — uniform across methods
  - RDKit descriptors (MW, LogP, TPSA, HBA, HBD, RotBonds, QED, HeavyAtoms, Rings)
  - Tc_to_Mol1, max_Tc_train, mean_top10_Tc_train
  - SAScore, PAINS_alerts, warhead_intact

Output:
  results/paper_evaluation/all_methods_bulk_scored.csv  (one row per candidate)

This drives the unified server-side filterable report covering all 633K candidates.

Runtime: ~30-60 min on local 8-core for 633K candidates (most are small methods,
Tier 2 SCALED already has 498K of these computed and we just rebuild from cache).
"""

import sys
import os
import gc
import json
import warnings
import multiprocessing as mp
from pathlib import Path
from datetime import datetime

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, FilterCatalog
RDLogger.DisableLog('rdApp.*')

RES = PROJECT_ROOT / "results" / "paper_evaluation"
OUT_CSV = RES / "all_methods_bulk_scored.csv"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]-C(=O)-[N;!H2]"
N_PROCS = max(1, mp.cpu_count() - 1)


# Method sources — same as the report
METHOD_SOURCES = [
    ("Tier 1 — Med-Chem Playbook (rule-based)",
     RES / "mol1_tier1_rules" / "tier1_candidates.csv", "smiles"),
    ("Tier 1.5 — Warhead Controls + Med-Chem Tricks",
     RES / "mol1_tier1_5_warhead_panel" / "tier1_5_candidates.csv", "smiles"),
    ("Tier 2 — Fragment Replacement (curated 204)",
     RES / "mol1_tier2_fragreplace" / "tier2_candidates.csv", "smiles"),
    ("Tier 2 SCALED — Fragment Replacement (498K)",
     RES / "aichem_tier2_scaled" / "products_scored.csv", "smiles"),
    ("Tier 3 v2 — Constrained Generative (single-seed)",
     RES / "mol1_tier3_constrained" / "tier3_candidates.csv", "smiles"),
    ("Tier 3 v3 — LibInvent locked",
     RES / "aigpu_overnight" / "libinvent_locked" / "libinvent_locked_1.csv", "SMILES"),
    ("Tier 3 v3 — Mol2Mol + warhead gate",
     RES / "aigpu_overnight" / "mol2mol_warhead" / "mol2mol_warhead_1.csv", "SMILES"),
    ("Tier 3 v3 — De Novo + warhead gate",
     RES / "aigpu_overnight" / "denovo_warhead" / "denovo_warhead_1.csv", "SMILES"),
    ("Tier 4 — De Novo unconstrained",
     RES / "aigpu_overnight" / "tier4_denovo" / "tier4_denovo_1.csv", "SMILES"),
    ("Tier 4 — Mol2Mol unconstrained",
     RES / "aigpu_overnight" / "tier4_mol2mol" / "tier4_mol2mol_1.csv", "SMILES"),
    ("Method A — De Novo FiLMDelta-driven",
     RES / "aigpu_overnight" / "method_a" / "method_a_filmdriven_denovo_1.csv", "SMILES"),
    ("Method B — Mol2Mol FiLMDelta-driven",
     RES / "aigpu_overnight" / "method_b" / "method_b_filmdriven_mol2mol_1.csv", "SMILES"),
]


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# Workers (module-level for pickling)
_TRAIN_FPS = None
_MOL1_FP = None
_PAINS = None
_WARHEAD_PAT = None
_SASCORER = None


def _init_worker():
    global _TRAIN_FPS, _MOL1_FP, _PAINS, _WARHEAD_PAT, _SASCORER
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdkit.Chem import RDConfig, FilterCatalog as FC
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer
    _SASCORER = sascorer
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    train_fps = []
    for s in smiles_df['smiles'].tolist():
        m = Chem.MolFromSmiles(s)
        if m is not None:
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    _TRAIN_FPS = train_fps
    mol1 = Chem.MolFromSmiles(MOL1_SMILES)
    _MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)
    params = FC.FilterCatalogParams()
    params.AddCatalog(FC.FilterCatalogParams.FilterCatalogs.PAINS)
    _PAINS = FC.FilterCatalog(params)
    _WARHEAD_PAT = Chem.MolFromSmarts(WARHEAD_SMARTS)


def _process_one(args):
    smi, method = args
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, _TRAIN_FPS))
        try:
            sas = float(_SASCORER.calculateScore(mol))
        except Exception:
            sas = float('nan')
        return {
            "smiles": smi,
            "method": method,
            "MW": Descriptors.MolWt(mol),
            "LogP": Descriptors.MolLogP(mol),
            "TPSA": Descriptors.TPSA(mol),
            "HBA": Descriptors.NumHAcceptors(mol),
            "HBD": Descriptors.NumHDonors(mol),
            "RotBonds": Descriptors.NumRotatableBonds(mol),
            "QED": Chem.QED.qed(mol),
            "HeavyAtoms": mol.GetNumHeavyAtoms(),
            "Rings": Descriptors.RingCount(mol),
            "Tc_to_Mol1": float(DataStructs.TanimotoSimilarity(_MOL1_FP, fp)),
            "max_Tc_train": float(sims.max()),
            "mean_top10_Tc_train": float(np.partition(sims, -10)[-10:].mean()),
            "SAScore": sas,
            "PAINS_alerts": len(_PAINS.GetMatches(mol)),
            "warhead_intact": bool(mol.HasSubstructMatch(_WARHEAD_PAT)),
        }
    except Exception:
        return None


def main():
    log("=" * 70)
    log("COMPUTE BULK METRICS FOR ALL 12 METHODS")
    log("=" * 70)

    # Aggregate all SMILES with method labels
    all_smiles = []
    seen_per_method = set()
    for method_name, csv, smi_col in METHOD_SOURCES:
        if not csv.exists():
            log(f"  ✗ {method_name}: missing {csv}")
            continue
        df = pd.read_csv(csv)
        if smi_col not in df.columns:
            log(f"  ✗ {method_name}: no col {smi_col}")
            continue
        n_before = len(df)
        df = df[[smi_col]].dropna()
        df.columns = ["smiles"]
        df = df.drop_duplicates(subset="smiles")
        log(f"  ✓ {method_name}: {len(df):,} unique SMILES (raw {n_before:,})")
        for s in df["smiles"]:
            all_smiles.append((s, method_name))

    log(f"\nTotal candidates: {len(all_smiles):,}")
    log(f"Workers: {N_PROCS}")

    results = [None] * len(all_smiles)
    n_log = 50_000
    last_log = 0
    with mp.Pool(N_PROCS, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap(_process_one, all_smiles, chunksize=500)):
            results[i] = r
            if (i + 1) - last_log >= n_log:
                last_log = i + 1
                log(f"  {last_log:,}/{len(all_smiles):,} processed")

    log("Building dataframe...")
    valid = [r for r in results if r is not None]
    log(f"  Valid: {len(valid):,} / {len(all_smiles):,}")
    df = pd.DataFrame(valid)

    # Add row_id (per the unified candidate set)
    df = df.reset_index(drop=True)
    df["row_id"] = df.index

    # Read pIC50_film from each source CSV (it varies by method's scoring)
    log("Merging pIC50 from source CSVs...")
    pic50_map = {}  # (method, smiles) → pIC50
    pic50_cols_by_source = {}
    for method_name, csv, smi_col in METHOD_SOURCES:
        if not csv.exists():
            continue
        src = pd.read_csv(csv)
        # Identify pIC50 col
        for pic_col in ["pIC50", "pIC50_film", "FiLMDelta pIC50 (raw)"]:
            if pic_col in src.columns:
                break
        else:
            continue
        for _, r in src[[smi_col, pic_col]].dropna().iterrows():
            pic50_map[(method_name, r[smi_col])] = float(r[pic_col])
        pic50_cols_by_source[method_name] = pic_col

    log(f"  pIC50 sources: {pic50_cols_by_source}")
    df["pIC50_method"] = [pic50_map.get((row["method"], row["smiles"]), float("nan"))
                           for _, row in df.iterrows()]

    df.to_csv(OUT_CSV, index=False)
    log(f"\nSaved → {OUT_CSV}  ({OUT_CSV.stat().st_size / 1024 / 1024:.1f} MB)")
    log(f"\nPer-method counts in output:")
    for m, n in df["method"].value_counts().sort_index().items():
        log(f"  {m[:60]:<60s}: {n:,}")
    log(f"\nDone: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
