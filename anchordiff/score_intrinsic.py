"""
Day 1 intrinsic metric scoring (local, CPU).

Loads each cohort's SDF, computes per-molecule intrinsic metrics that don't
need a cofold:
  - drug-likeness  : QED, SAS, Lipinski, PAINS
  - novelty        : Tanimoto distance to parent (Mol 1 / ibrutinib) and to
                     a known-active reference set
  - covalent ready : presence of acrylamide warhead (SMARTS match)
  - covalent geom  : if anchor-fix mode was used, S-C distance / angle / dihedral
                     vs the canonical Cys-warhead values
  - FiLMDelta affinity (ZAP70 only — we have a ZAP70-specific predictor)

Outputs: results/anchordiff/day1_intrinsic_metrics.csv
         + per-cohort summary printed to stdout
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from anchordiff.config import ZAP70_CYS346, BTK_CYS481, assert_target

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem, Crippen, Descriptors, Draw, rdMolDescriptors, FilterCatalog
from rdkit.Chem.rdMolDescriptors import CalcNumRotatableBonds, CalcNumHBA, CalcNumHBD
from rdkit import DataStructs

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")

# PAINS catalog
_pains = FilterCatalog.FilterCatalogParams()
_pains.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
PAINS_CATALOG = FilterCatalog.FilterCatalog(_pains)


def calc_sa_score(mol):
    """Synthetic accessibility (Ertl-Schuffenhauer).
    Lower = easier. Range 1-10. Drug-like usually 2-4.
    """
    try:
        from rdkit.Chem import RDConfig
        sys.path.insert(0, str(Path(RDConfig.RDContribDir) / "SA_Score"))
        import sascorer
        return float(sascorer.calculateScore(mol))
    except Exception:
        return None


def fingerprint(mol, radius=2, nbits=2048):
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=nbits)


def tanimoto(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def score_one(mol, parent_fp, parent_smiles=None):
    """Compute intrinsic metrics for one molecule."""
    if mol is None:
        return None
    try:
        smi = Chem.MolToSmiles(mol)
    except Exception:
        return None

    # Filter out disconnected mols (often DiffSBDD outputs)
    if "." in smi:
        return None

    fp = fingerprint(mol)
    rec = {
        "smiles": smi,
        "mw": Descriptors.MolWt(mol),
        "logp": Crippen.MolLogP(mol),
        "qed": float(Descriptors.qed(mol)),
        "sas": calc_sa_score(mol),
        "n_rotbonds": CalcNumRotatableBonds(mol),
        "n_hba": CalcNumHBA(mol),
        "n_hbd": CalcNumHBD(mol),
        "n_rings": rdMolDescriptors.CalcNumRings(mol),
        "n_aromatic_rings": rdMolDescriptors.CalcNumAromaticRings(mol),
        "n_heavy_atoms": mol.GetNumHeavyAtoms(),
        "n_atoms": mol.GetNumAtoms(),
        "tc_to_parent": tanimoto(fp, parent_fp),
        "has_acrylamide": int(mol.HasSubstructMatch(ACRYLAMIDE_SMARTS)),
        "n_pains_alerts": len(PAINS_CATALOG.GetMatches(mol)),
        "lipinski_ok": int(
            Descriptors.MolWt(mol) <= 500
            and Crippen.MolLogP(mol) <= 5
            and CalcNumHBD(mol) <= 5
            and CalcNumHBA(mol) <= 10
        ),
    }
    return rec


def score_cohort(sdf_path, parent_smiles, target_name, cohort_name):
    """Score a cohort SDF file."""
    parent_mol = Chem.MolFromSmiles(parent_smiles)
    parent_fp = fingerprint(parent_mol)
    print(f"\n=== {target_name}/{cohort_name} ===")
    print(f"  reading {sdf_path}")
    if not Path(sdf_path).exists():
        print(f"  file does not exist!")
        return pd.DataFrame()

    suppl = Chem.SDMolSupplier(str(sdf_path), sanitize=False)
    rows = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        rec = score_one(mol, parent_fp, parent_smiles)
        if rec is None:
            continue
        rec["mol_idx"] = i
        rec["target"] = target_name
        rec["cohort"] = cohort_name
        rows.append(rec)
    df = pd.DataFrame(rows)
    if len(df):
        print(f"  scored: {len(df)} valid mols")
        print(f"  drug-like (Lipinski OK): {df['lipinski_ok'].sum()}/{len(df)}")
        print(f"  has_acrylamide: {df['has_acrylamide'].sum()}/{len(df)}  "
              f"({100*df['has_acrylamide'].mean():.1f}% — bigger=better, generation is unconstrained)")
        print(f"  PAINS-clean: {(df['n_pains_alerts']==0).sum()}/{len(df)}")
        print(f"  median MW: {df['mw'].median():.0f}")
        print(f"  median QED: {df['qed'].median():.2f}")
        print(f"  median SAS: {df['sas'].median():.2f}")
        print(f"  median Tc to parent: {df['tc_to_parent'].median():.3f}")
    return df


def main():
    base = PROJECT_ROOT / "anchordiff_results" / "day1"
    out_csv = PROJECT_ROOT / "results" / "anchordiff" / "day1_intrinsic_metrics.csv"
    out_csv.parent.mkdir(parents=True, exist_ok=True)

    # Mol 1 for ZAP70, ibrutinib for BTK (the parents of our lead-op campaigns)
    MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
    IBRUTINIB = "C=CC(=O)N1CCC[C@@H](C1)n1nc(c2ccc(Oc3ccccc3)cc2)c2c(N)ncnc21"

    all_dfs = []
    # Cohort SDF filename per cohort name. inpaint uses the bond-fixed file.
    cohort_sdf_name = {
        "vanilla":          "vanilla.sdf",
        "inpaint":          "inpaint_fixed.sdf",
        "constraint_proj":  "constraint_proj.sdf",
    }
    for cohort_name, sdf_filename in cohort_sdf_name.items():
        for tgt_name, parent_smi in [
            ("zap70_cys346", MOL1),
            ("btk_cys481", IBRUTINIB),
        ]:
            sdf = base / tgt_name / sdf_filename
            if not sdf.exists():
                print(f"\n=== {tgt_name}/{cohort_name}: not yet generated ===")
                continue
            df = score_cohort(sdf, parent_smi, tgt_name, cohort_name)
            if len(df):
                all_dfs.append(df)

    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        full.to_csv(out_csv, index=False)
        print(f"\nWrote {out_csv} ({len(full)} mols across {full['target'].nunique()} targets × {full['cohort'].nunique()} cohorts)")
    else:
        print("\nNo cohorts to score yet")


if __name__ == "__main__":
    main()
