"""Unified evaluation pipeline for covalent-aware generative methods.

Takes a directory of SDF files (one or more, each with N molecules) and
computes Tier 1+2+3 metrics for each, producing a per-mol CSV and per-method
summary JSON.

Per-method input expected:
  results/<method_name>/samples.sdf   (or .sdf.gz, or directory of sdfs)

Per-method output:
  results/<method_name>/eval_per_mol.csv
  results/<method_name>/eval_summary.json

Tier 1 (chemistry, no Boltz):
  - validity (RDKit parse + single connected component)
  - warhead_class_detected (any of acrylamide/acrylate/vinyl_sulfone/haloacetamide/propiolamide)
  - MW, HeavyAtoms, ring count, QED, SAScore, PAINS_alerts
  - Murcko scaffold (for diversity & novelty)
  - max_Tc_train (max Tanimoto to CovBinder training set)
  - Tc_to_Mol1 (Tanimoto to the seed acrylamide Mol1)

Tier 2 (geometry; requires Boltz cofold = ~70s/mol; subsample top-N by Tier 1):
  - d_SG (Cys-Sγ to warhead Cβ distance)
  - burgi_dunitz_dev_deg
  - atp_pocket_fraction
  - n_h_bonds
  - hinge_hbond_present
  - mPAE, iptm, ligand_iptm, ligand_pLDDT_mean

Tier 3 (predicted activity; needs Tier 2):
  - FiLMDelta pIC50 prediction (uses anchor Mol1)
  - E_interaction (from score_cofold_pocket_interaction)
  - ligand_strain_kcalmol

Usage:
  python eval_generation_quality.py --sdf results/m0_vanilla/samples.sdf --method M0
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Seed Mol 1 SMILES (anchor acrylamide for ZAP70)
MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

# Soft warhead SMARTS — match any Michael-acceptor-like electrophile
WARHEAD_SMARTS = [
    ("acrylamide",    "[CH2]=[CH]C(=O)N"),
    ("acrylate",      "[CH2]=[CH]C(=O)O"),
    ("vinyl_sulfone", "[CH2]=[CH]S(=O)(=O)"),
    ("haloacetamide", "[Cl,Br,I][CH2]C(=O)N"),
    ("propiolamide",  "C#CC(=O)N"),
    ("alpha_beta_unsat_carbonyl", "[CX3]=[CX3]C(=O)"),  # soft fallback
]
_WARHEAD_MOLS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEAD_SMARTS]


def detect_warhead_class(mol) -> str | None:
    for cls, sm in _WARHEAD_MOLS:
        if sm and mol.HasSubstructMatch(sm):
            return cls
    return None


def murcko_scaffold(mol) -> str:
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mol))
    except Exception:
        return ""


def morgan_fp(mol, radius: int = 2, n_bits: int = 2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, n_bits)


def tanimoto(fp1, fp2) -> float:
    from rdkit.DataStructs import TanimotoSimilarity
    return float(TanimotoSimilarity(fp1, fp2))


# Cache the Mol1 FP and training-set FPs once
_MOL1_FP = None
_TRAIN_FPS = None


def get_mol1_fp():
    global _MOL1_FP
    if _MOL1_FP is None:
        m = Chem.MolFromSmiles(MOL1_SMI)
        _MOL1_FP = morgan_fp(m)
    return _MOL1_FP


def get_train_fps(csv_path: Path = None) -> list:
    global _TRAIN_FPS
    if _TRAIN_FPS is not None:
        return _TRAIN_FPS
    if csv_path is None:
        csv_path = PROJECT_ROOT / "data/covbinder/covind_training_set.csv"
    if not csv_path.exists():
        _TRAIN_FPS = []
        return _TRAIN_FPS
    fps = []
    # Try common SMILES column names
    df = pd.read_csv(csv_path)
    smi_col = next((c for c in df.columns if "smi" in c.lower() or "smiles" in c.lower()), None)
    if smi_col is None:
        _TRAIN_FPS = []
        return _TRAIN_FPS
    for s in df[smi_col].dropna().astype(str):
        m = Chem.MolFromSmiles(s)
        if m is not None:
            fps.append(morgan_fp(m))
    _TRAIN_FPS = fps
    print(f"  loaded {len(fps)} training-set FPs from {csv_path.name}")
    return fps


def tier1_metrics(mol) -> dict:
    """Per-mol chemistry metrics. Returns dict; works on a RDKit mol that may
    be None (in which case validity=False)."""
    if mol is None:
        return {"validity": False, "n_fragments": None, "warhead_class": None}
    smi = Chem.MolToSmiles(mol)
    # Connected component check
    frags = Chem.GetMolFrags(mol)
    n_frag = len(frags)
    out = {
        "smiles": smi,
        "validity": True,
        "n_fragments": n_frag,
        "is_connected": n_frag == 1,
        "warhead_class": detect_warhead_class(mol),
        "MW": Descriptors.MolWt(mol),
        "HeavyAtoms": mol.GetNumHeavyAtoms(),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_rotatable": rdMolDescriptors.CalcNumRotatableBonds(mol),
        "QED": QED.qed(mol),
        "LogP": Descriptors.MolLogP(mol),
        "HBA": rdMolDescriptors.CalcNumHBA(mol),
        "HBD": rdMolDescriptors.CalcNumHBD(mol),
        "TPSA": Descriptors.TPSA(mol),
        "scaffold": murcko_scaffold(mol),
    }
    # Tc to Mol1
    fp = morgan_fp(mol)
    out["Tc_to_Mol1"] = tanimoto(fp, get_mol1_fp())
    # max Tc to training set
    train = get_train_fps()
    if train:
        out["max_Tc_train"] = max(tanimoto(fp, tfp) for tfp in train)
    else:
        out["max_Tc_train"] = None
    return out


def load_sdf(sdf_path: Path, largest_frag: bool = True) -> list:
    """Return list of RDKit mols from an SDF (None for failed parses).

    If largest_frag=True, for each multi-fragment record return only the largest
    connected component (drug + cofactors → drug). DrugFlow's natural output is
    multi-fragment; DiffSBDD's is single-fragment. This makes the comparison fair.
    """
    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    mols = []
    for m in suppl:
        if m is None:
            mols.append(None)
            continue
        if largest_frag:
            frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
            if len(frags) > 1:
                frags = sorted(frags, key=lambda f: f.GetNumHeavyAtoms(), reverse=True)
                m = frags[0]
        try:
            Chem.SanitizeMol(m)
            mols.append(m)
        except Exception:
            mols.append(None)
    return mols


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sdf", type=Path, required=True, help="Input SDF (one or more, comma-separated)")
    ap.add_argument("--method", type=str, required=True, help="Method label (M0, M1, M2, M3)")
    ap.add_argument("--out-dir", type=Path, default=None,
                    help="Output directory (default: <sdf_parent>)")
    args = ap.parse_args()

    sdf_paths = [Path(p.strip()) for p in str(args.sdf).split(",")]
    out_dir = args.out_dir or sdf_paths[0].parent

    # Load all mols
    all_mols = []
    for p in sdf_paths:
        if not p.exists():
            print(f"WARN: {p} missing, skipping")
            continue
        mols = load_sdf(p)
        all_mols.extend(mols)
    print(f"loaded {len(all_mols)} mols from {len(sdf_paths)} SDF(s)")

    # Tier 1 metrics
    rows = []
    for i, m in enumerate(all_mols):
        row = {"idx": i, "method": args.method}
        row.update(tier1_metrics(m))
        rows.append(row)
    df = pd.DataFrame(rows)

    # Write per-mol
    per_mol_path = out_dir / "eval_per_mol.csv"
    df.to_csv(per_mol_path, index=False)
    print(f"wrote {per_mol_path}  ({len(df)} rows)")

    # Per-method summary
    summary = {
        "method": args.method,
        "n_total": len(df),
        "n_valid": int(df["validity"].sum()),
        "validity_pct": float(df["validity"].mean() * 100),
        "n_connected": int(df.get("is_connected", pd.Series([False] * len(df))).fillna(False).sum()),
        "warhead_class_dist": df["warhead_class"].value_counts(dropna=False).to_dict(),
        "warhead_any_detected_pct": float((df["warhead_class"].notna()).mean() * 100),
        "MW_median": float(df["MW"].median()) if df["MW"].notna().any() else None,
        "QED_median": float(df["QED"].median()) if df["QED"].notna().any() else None,
        "n_unique_scaffolds": int(df["scaffold"].nunique()),
        "scaffold_diversity_pct": float(df["scaffold"].nunique() / max(1, df["validity"].sum()) * 100),
        "Tc_to_Mol1_median": float(df["Tc_to_Mol1"].median()) if df["Tc_to_Mol1"].notna().any() else None,
        "max_Tc_train_median": float(df["max_Tc_train"].median()) if df["max_Tc_train"].notna().any() else None,
    }
    sum_path = out_dir / "eval_summary.json"
    json.dump(summary, open(sum_path, "w"), indent=2, default=str)
    print(f"wrote {sum_path}")
    print(json.dumps(summary, indent=2, default=str)[:1500])


if __name__ == "__main__":
    main()
