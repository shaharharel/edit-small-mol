#!/usr/bin/env python3
"""
Compute cheap-but-useful metrics for all 498K Tier 2 SCALED candidates.

Adds to products_scored.csv:
  - Tc_to_Mol1     (Morgan FP r=2 / 2048 Tanimoto)
  - max_Tc_train   (Tc to nearest of 280 ZAP70 training mols)
  - mean_top10_Tc_train (mean Tc to 10 nearest train mols)
  - SAScore        (Ertl-Schuffenhauer)
  - PAINS_alerts   (count)
  - warhead_intact (boolean)

Output: writes products_scored_full.csv with all original cols + new cols.

Runs in ~10-15 min on local 8-core via multiprocessing.
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
INPUT = RES / "aichem_tier2_scaled" / "products_scored.csv"
OUTPUT = RES / "aichem_tier2_scaled" / "products_scored_full.csv"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]-C(=O)-[N;!H2]"  # !H2 = NOT primary amine; allows secondary AND tertiary amide N (Mol 1 has tertiary)
N_PROCS = max(1, mp.cpu_count() - 1)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# Load training SMILES + precompute their FPs (small: 280 mols)
def load_train_fps():
    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    smiles = smiles_df['smiles'].tolist()
    fps = []
    for s in smiles:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    return smiles, fps


# Workers — must be module-level for pickling
_TRAIN_FPS = None
_MOL1_FP = None
_PAINS = None
_WARHEAD_PAT = None
_SASCORER = None


def _init_worker():
    """Pool worker initializer — preload shared state."""
    global _TRAIN_FPS, _MOL1_FP, _PAINS, _WARHEAD_PAT, _SASCORER
    import sys
    sys.path.insert(0, str(PROJECT_ROOT))
    from rdkit.Chem import RDConfig, FilterCatalog
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer
    _SASCORER = sascorer

    from experiments.run_zap70_v3 import load_zap70_molecules
    smiles_df, _ = load_zap70_molecules()
    train_fps = []
    for s in smiles_df['smiles'].tolist():
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    _TRAIN_FPS = train_fps

    mol1 = Chem.MolFromSmiles(MOL1_SMILES)
    _MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, nBits=2048)

    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    _PAINS = FilterCatalog.FilterCatalog(params)

    _WARHEAD_PAT = Chem.MolFromSmarts(WARHEAD_SMARTS)


def _process_one(smi):
    """Compute all metrics for one SMILES. Returns dict."""
    if not isinstance(smi, str):
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    try:
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        # Tc to Mol1
        tc_mol1 = float(DataStructs.TanimotoSimilarity(_MOL1_FP, fp))
        # Tc to training (vector)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(fp, _TRAIN_FPS))
        max_tc_train = float(sims.max())
        top10 = np.partition(sims, -10)[-10:]
        mean_top10 = float(top10.mean())
        # SAScore
        try:
            sas = float(_SASCORER.calculateScore(mol))
        except Exception:
            sas = float('nan')
        # PAINS
        pains = len(_PAINS.GetMatches(mol))
        # Warhead intact
        wh = bool(mol.HasSubstructMatch(_WARHEAD_PAT))
        return {
            "smiles": smi,
            "Tc_to_Mol1": tc_mol1,
            "max_Tc_train": max_tc_train,
            "mean_top10_Tc_train": mean_top10,
            "SAScore": sas,
            "PAINS_alerts": pains,
            "warhead_intact": wh,
        }
    except Exception:
        return None


def main():
    log("=" * 70)
    log("COMPUTE BULK METRICS ON 498K CANDIDATES")
    log("=" * 70)

    if not INPUT.exists():
        log(f"ERROR: {INPUT} not found"); return
    df = pd.read_csv(INPUT)
    log(f"Loaded {len(df):,} candidates from {INPUT.name}")

    smiles = df["smiles"].tolist()
    log(f"Processing on {N_PROCS} workers...")

    results = [None] * len(smiles)
    with mp.Pool(N_PROCS, initializer=_init_worker) as pool:
        for i, r in enumerate(pool.imap(_process_one, smiles, chunksize=500)):
            results[i] = r
            if (i + 1) % 50_000 == 0:
                log(f"  {i+1:,}/{len(smiles):,} processed")

    log(f"Building output dataframe...")
    extras = []
    for r in results:
        if r is None:
            extras.append({
                "Tc_to_Mol1": float('nan'), "max_Tc_train": float('nan'),
                "mean_top10_Tc_train": float('nan'), "SAScore": float('nan'),
                "PAINS_alerts": -1, "warhead_intact": False,
            })
        else:
            extras.append({k: r[k] for k in ["Tc_to_Mol1", "max_Tc_train", "mean_top10_Tc_train",
                                              "SAScore", "PAINS_alerts", "warhead_intact"]})
    extras_df = pd.DataFrame(extras)
    out = pd.concat([df.reset_index(drop=True), extras_df], axis=1)
    out.to_csv(OUTPUT, index=False)
    log(f"Saved → {OUTPUT}  ({OUTPUT.stat().st_size / 1024 / 1024:.1f} MB)")
    log(f"\nSummary stats:")
    log(f"  warhead_intact: {out['warhead_intact'].sum():,} / {len(out):,}")
    log(f"  PAINS == 0:    {(out['PAINS_alerts']==0).sum():,} / {len(out):,}")
    log(f"  SAScore < 4:   {(out['SAScore']<4).sum():,} / {len(out):,}")
    log(f"  pIC50 ≥ 7:     {(out['pIC50_film']>=7).sum():,} / {len(out):,}")
    log(f"  max_Tc≥0.3:    {(out['max_Tc_train']>=0.3).sum():,} / {len(out):,}")
    log(f"\nDone: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
