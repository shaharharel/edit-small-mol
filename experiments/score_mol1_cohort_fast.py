"""Fast scorer for Mol1-anchored cohort sampling.csv files.

Computes the fast metrics matching the existing Tier 4 scored CSV schema:
  - RDKit panel: MW, LogP, TPSA, HBA, HBD, RotBonds, HeavyAtoms, Rings, QED, Lipinski
  - Substructure: warhead_intact (acrylamide), THIQ-acryl, Mol1 Murcko
  - Tc to Mol1, Tc to training set (ChEMBL ZAP70 280 mols)
  - FiLM pIC50
  - Novelty: not in ChEMBL ZAP70 set
  - Brenk + PAINS alerts

Heavy metrics (Vina, Boltz) deferred. Writes to data/tier4_scored/{cohort_tag}_scored.csv.

Usage: python experiments/score_mol1_cohort_fast.py <input.csv> <output.csv> --tag <cohort_tag> [--seed <seed_smi>]
"""
from __future__ import annotations

import argparse
import sys
import warnings
from pathlib import Path
from typing import Optional

warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, FilterCatalog, Lipinski, QED, RDConfig, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT))

# Mol1 reference
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
_m1 = Chem.MolFromSmiles(MOL1)
MOL1_CANON = Chem.MolToSmiles(_m1)
MOL1_MURCKO = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(_m1))
MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(_m1, 2, 2048)
THIQ_ACRYL = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")
ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)[N]")

# ChEMBL ZAP70 set for novelty + max_Tc_train
def _load_zap70():
    df = pd.read_csv(PROJECT / "data/docking_chembl_zap70/docking_results.csv")
    canon = set()
    fps = []
    for s in df["smiles"]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        canon.add(Chem.MolToSmiles(m))
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
    return canon, fps


ZAP70_CANON, ZAP70_FPS = _load_zap70()
print(f"[scorer] Loaded ChEMBL ZAP70: {len(ZAP70_CANON)} unique, {len(ZAP70_FPS)} fps", file=sys.stderr)

# FilterCatalog for Brenk + PAINS
def _build_filter_catalog():
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    return FilterCatalog.FilterCatalog(params)


FILTER_CATALOG = _build_filter_catalog()

# SAscore
try:
    sys.path.append(str(Path(RDConfig.RDContribDir) / "SA_Score"))
    import sascorer  # type: ignore
except Exception:
    sascorer = None


def score_mol(smi: str) -> Optional[dict]:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    canon = Chem.MolToSmiles(m)

    # RDKit panel
    mw = Descriptors.MolWt(m)
    logp = Crippen.MolLogP(m)
    tpsa = Descriptors.TPSA(m)
    hba = Lipinski.NumHAcceptors(m)
    hbd = Lipinski.NumHDonors(m)
    rotb = Lipinski.NumRotatableBonds(m)
    hatoms = m.GetNumHeavyAtoms()
    rings = rdMolDescriptors.CalcNumRings(m)
    qed = QED.qed(m)
    fsp3 = Lipinski.FractionCSP3(m)
    lipinski_viol = int((mw > 500) + (logp > 5) + (hba > 10) + (hbd > 5))

    # Substructure
    warhead_intact = bool(m.HasSubstructMatch(ACRYL))
    thiq_core = bool(m.HasSubstructMatch(THIQ_ACRYL))
    murcko_match = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) == MOL1_MURCKO

    # Tc to Mol1
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    tc_mol1 = DataStructs.TanimotoSimilarity(MOL1_FP, fp)

    # Tc to ChEMBL ZAP70 training set
    sims = DataStructs.BulkTanimotoSimilarity(fp, ZAP70_FPS)
    sims_sorted = sorted(sims, reverse=True)
    max_tc_train = sims_sorted[0] if sims_sorted else 0.0
    mean_top10 = float(np.mean(sims_sorted[:10])) if sims_sorted else 0.0

    # Novelty (not in ChEMBL ZAP70 set)
    novel = canon not in ZAP70_CANON

    # Brenk + PAINS
    brenk_alerts = 0
    pains_alerts = 0
    for m_match in FILTER_CATALOG.GetMatches(m):
        cat = m_match.GetProp("FilterSet")
        if cat == "BRENK":
            brenk_alerts += 1
        elif cat == "PAINS":
            pains_alerts += 1

    # SAScore
    sa = float(sascorer.calculateScore(m)) if sascorer is not None else None

    return {
        "smiles": canon,
        "MW": round(mw, 3),
        "LogP": round(logp, 3),
        "TPSA": round(tpsa, 2),
        "HBA": hba,
        "HBD": hbd,
        "RotBonds": rotb,
        "HeavyAtoms": hatoms,
        "Rings": rings,
        "QED": round(qed, 4),
        "fsp3": round(fsp3, 3),
        "Lipinski_violations": lipinski_viol,
        "warhead_intact": warhead_intact,
        "thiq_core": thiq_core,
        "mol1_murcko_match": murcko_match,
        "Tc_to_Mol1": round(tc_mol1, 4),
        "max_Tc_train": round(max_tc_train, 4),
        "mean_top10_Tc_train": round(mean_top10, 4),
        "Brenk_alerts": brenk_alerts,
        "PAINS_alerts": pains_alerts,
        "SAScore": round(sa, 3) if sa is not None else None,
        "novel_vs_chembl_zap70": novel,
    }


def score_film_batch(smiles_list, model, scaler, anchor_embs, anchor_pIC50):
    """Bulk FiLM scoring — single load, all mols."""
    from experiments.reinvent4_film_scorer import score_smiles
    return score_smiles(smiles_list, model, scaler, anchor_embs, anchor_pIC50)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("input_csv")
    ap.add_argument("output_csv")
    ap.add_argument("--tag", required=True, help="cohort tag, becomes 'method' col")
    ap.add_argument("--seed", default=None, help="seed SMILES for this cohort")
    ap.add_argument("--smi-col", default="SMILES", help="SMILES column name in input")
    ap.add_argument("--no-film", action="store_true", help="skip FiLM pIC50 (faster)")
    args = ap.parse_args()

    print(f"[scorer] Reading {args.input_csv}")
    df = pd.read_csv(args.input_csv)
    smi_col = args.smi_col if args.smi_col in df.columns else (
        "SMILES" if "SMILES" in df.columns else "smiles"
    )
    if smi_col not in df.columns:
        print(f"[scorer] ERROR: no SMILES column. Columns: {list(df.columns)}")
        sys.exit(1)
    print(f"[scorer] Scoring {len(df):,} mols (SMILES col: '{smi_col}')")

    # Score each mol
    rows = []
    for i, smi in enumerate(df[smi_col]):
        rec = score_mol(str(smi))
        if rec is None:
            continue
        rec["row_id"] = f"{args.tag}_{i}"
        rec["method"] = args.tag
        if args.seed:
            rec["seed_smi"] = args.seed
        rows.append(rec)
        if (i + 1) % 5000 == 0:
            print(f"  scored {i+1:,}/{len(df):,}")

    out = pd.DataFrame(rows)
    print(f"[scorer] Scored {len(out):,} valid mols (of {len(df):,} input)")

    # FiLM pIC50 in batch
    if not args.no_film:
        print(f"[scorer] FiLM pIC50 scoring...")
        try:
            from experiments.reinvent4_film_scorer import load_film_model
            model, scaler, anchor_embs, anchor_pIC50 = load_film_model()
            scores = score_film_batch(out["smiles"].tolist(), model, scaler, anchor_embs, anchor_pIC50)
            out["pIC50_film"] = [round(s, 4) if s is not None and not np.isnan(s) else None for s in scores]
            out["pIC50_mean"] = out["pIC50_film"]
            out["delta_vs_mol1"] = [round(s - 6.59, 4) if s is not None and not np.isnan(s) else None for s in scores]
            print(f"[scorer]   FiLM done. Mean pIC50: {np.nanmean(out['pIC50_film']):.3f}")
        except Exception as e:
            print(f"[scorer] FiLM scoring failed: {e}")

    out.to_csv(args.output_csv, index=False)
    print(f"[scorer] Wrote {args.output_csv}")

    # Summary
    print()
    print(f"=== {args.tag} summary ===")
    print(f"  N: {len(out):,}")
    if "warhead_intact" in out.columns:
        print(f"  Acrylamide intact: {out['warhead_intact'].mean()*100:.1f}%")
    if "thiq_core" in out.columns:
        print(f"  THIQ-acryl core:   {out['thiq_core'].mean()*100:.1f}%")
    if "mol1_murcko_match" in out.columns:
        print(f"  Mol1 Murcko match: {out['mol1_murcko_match'].mean()*100:.2f}%")
    print(f"  Median Tc-to-Mol1: {out['Tc_to_Mol1'].median():.3f}")
    print(f"  Tc≥0.5: {(out['Tc_to_Mol1']>=0.5).mean()*100:.1f}%")
    print(f"  Median QED: {out['QED'].median():.3f}")
    if "pIC50_film" in out.columns:
        print(f"  Median FiLM pIC50: {out['pIC50_film'].median():.2f}")
    print(f"  Novel vs ChEMBL ZAP70: {out['novel_vs_chembl_zap70'].mean()*100:.1f}%")
    print(f"  Median Brenk alerts: {out['Brenk_alerts'].median():.1f}")


if __name__ == "__main__":
    main()
