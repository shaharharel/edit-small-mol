#!/usr/bin/env python3
"""For each per_mol.csv under per_cohort/, look up the cohort's input.sdf and
add input-pose geometry columns (d_cb_sg_input, bd_angle_input, and their
diff-from-target variants).

Idempotent: existing values are overwritten with fresh computation.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
PER_COHORT = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison" / "per_cohort"
COHORTS_INPUT = PROJECT_ROOT / "data" / "cohort_comparison" / "cohorts"

CYS346_SG = np.array([18.888, -3.650, -29.979])
ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
TARGET_BD_DEG = 107.0
TARGET_DSG = 1.85


def find_warhead_indices(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None
    m = matches[0]
    return {"ch2_terminal": m[0], "ch_beta": m[1]}


def compute_geo(mol, smiles):
    wh = find_warhead_indices(smiles)
    if not wh or mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    cb = conf.GetAtomPosition(wh["ch_beta"])
    ct = conf.GetAtomPosition(wh["ch2_terminal"])
    cb_np = np.array([cb.x, cb.y, cb.z])
    ct_np = np.array([ct.x, ct.y, ct.z])
    d_sg = float(np.linalg.norm(cb_np - CYS346_SG))
    v1 = CYS346_SG - cb_np; v2 = ct_np - cb_np
    nv1 = np.linalg.norm(v1); nv2 = np.linalg.norm(v2)
    bd = None
    if nv1 > 1e-6 and nv2 > 1e-6:
        cosang = max(-1.0, min(1.0, float(np.dot(v1, v2) / (nv1 * nv2))))
        bd = float(np.degrees(np.arccos(cosang)))
    return {"d_cb_sg_input": d_sg, "bd_angle_input": bd}


def enrich_cohort(cohort_dir: Path):
    name = cohort_dir.name
    csv_path = cohort_dir / "per_mol.csv"
    if not csv_path.exists():
        return False
    input_sdf = COHORTS_INPUT / name / "input.sdf"
    if not input_sdf.exists():
        print(f"  [skip] {name}: no input.sdf")
        return False
    geo_by_name = {}
    geo_by_idx = {}
    supp = Chem.SDMolSupplier(str(input_sdf), removeHs=False, sanitize=True)
    for i, m in enumerate(supp):
        if m is None:
            continue
        try:
            smi = m.GetProp("smi") if m.HasProp("smi") else Chem.MolToSmiles(m)
            nm = m.GetProp("_Name") if m.HasProp("_Name") else f"{name}_{i}"
            geo = compute_geo(m, smi)
            if geo:
                geo_by_name[nm] = geo
                geo_by_idx[i] = geo
        except Exception:
            continue

    df = pd.read_csv(csv_path)
    # Choose lookup key: prefer name, fallback to mol_idx
    def lookup(row):
        g = geo_by_name.get(row["name"])
        if g is None and "mol_idx" in row:
            g = geo_by_idx.get(int(row["mol_idx"]))
        return g

    d_sgs, bds, d_diffs, bd_diffs = [], [], [], []
    for _, row in df.iterrows():
        g = lookup(row)
        if g is None:
            d_sgs.append(np.nan); bds.append(np.nan)
            d_diffs.append(np.nan); bd_diffs.append(np.nan)
            continue
        d_sg = g["d_cb_sg_input"]
        bd = g["bd_angle_input"]
        d_sgs.append(d_sg if d_sg is not None else np.nan)
        bds.append(bd if bd is not None else np.nan)
        d_diffs.append(abs(d_sg - TARGET_DSG) if d_sg is not None else np.nan)
        bd_diffs.append(abs(bd - TARGET_BD_DEG) if bd is not None else np.nan)

    df["d_cb_sg_input"] = d_sgs
    df["bd_angle_input"] = bds
    df["d_cb_sg_input_diff_from_185"] = d_diffs
    df["bd_angle_input_diff_from_107"] = bd_diffs
    df.to_csv(csv_path, index=False)
    n_geo = sum(1 for x in d_sgs if not np.isnan(x))
    print(f"  [ok]  {name}: enriched {n_geo}/{len(df)} mols with input geometry")
    return True


def main():
    if not PER_COHORT.exists():
        print(f"No per_cohort dir: {PER_COHORT}")
        sys.exit(1)
    cohorts = sorted([d for d in PER_COHORT.iterdir() if d.is_dir()])
    print(f"Enriching {len(cohorts)} cohorts with input-pose geometry...")
    for cd in cohorts:
        enrich_cohort(cd)


if __name__ == "__main__":
    main()
