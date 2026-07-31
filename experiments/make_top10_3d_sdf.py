#!/usr/bin/env python3
"""For each cohort, write a top10.sdf containing the 3D docked top-pose for the
ten best-scoring molecules. Atom coords come from the pose PDBQT (top MODEL),
bonds inferred from the SMILES.

Idempotent — overwrites existing top10.sdf in each cohort directory.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from cohort_comparison_dock import (
    parse_pdbqt_all_models, parse_meeko_smiles_idx, BOX_CENTER, CYS346_SG,
)

PROJECT_ROOT = Path(__file__).parent.parent
PER_COHORT = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison" / "per_cohort"


def pose_to_3d_mol(smi: str, pose_pdbqt: Path, lig_pdbqt: Path):
    """Build an RDKit mol with 3D coords from the top docked pose.

    Returns mol with conformer set to the pose coords.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    poses = parse_pdbqt_all_models(pose_pdbqt)
    if not poses:
        return None
    top = poses[0]
    coords = top["coords"]
    # Get heavy-atom mapping (meeko REMARK preferred)
    meeko_map = parse_meeko_smiles_idx(lig_pdbqt)
    if not meeko_map:
        return None
    n_heavy = mol.GetNumHeavyAtoms()
    if len(meeko_map) < n_heavy:
        return None
    # Build conformer
    conf = Chem.Conformer(n_heavy)
    for smi_idx_0b, pdbqt_num_1b in meeko_map.items():
        pose_row = pdbqt_num_1b - 1
        if pose_row >= len(coords):
            continue
        if smi_idx_0b >= n_heavy:
            continue
        x, y, z = coords[pose_row]
        conf.SetAtomPosition(smi_idx_0b, (float(x), float(y), float(z)))
    mol.AddConformer(conf, assignId=True)
    return mol


def make_for_cohort(cohort_dir: Path, top_n: int = 10):
    csv = cohort_dir / "per_mol.csv"
    if not csv.exists():
        return
    df = pd.read_csv(csv)
    df_ok = df[df["success"] & df["vina_score"].notna()].sort_values("vina_score").head(top_n)
    pose_dir = cohort_dir / "poses"
    lig_dir = cohort_dir / "ligands_pdbqt"
    out_sdf = cohort_dir / "top10.sdf"
    writer = Chem.SDWriter(str(out_sdf))
    n_written = 0
    for _, row in df_ok.iterrows():
        name = row["name"]
        smi = row["smi"]
        pose = pose_dir / f"{name}_pose.pdbqt"
        lig = lig_dir / f"{name}.pdbqt"
        if not pose.exists() or not lig.exists():
            continue
        m = pose_to_3d_mol(smi, pose, lig)
        if m is None:
            # Fallback: write 2D mol with metadata
            m = Chem.MolFromSmiles(smi)
            if m is None:
                continue
        m.SetProp("_Name", name)
        m.SetProp("vina_score", f"{row['vina_score']:.3f}")
        m.SetProp("cohort", cohort_dir.name)
        if pd.notna(row.get("d_cb_sg_top1")):
            m.SetProp("d_cb_sg_top1", f"{row['d_cb_sg_top1']:.2f}")
        writer.write(m)
        n_written += 1
    writer.close()
    print(f"  [ok] {cohort_dir.name}: wrote {n_written}-mol top10.sdf with 3D poses")


def main():
    if not PER_COHORT.exists():
        print(f"No per_cohort dir: {PER_COHORT}")
        sys.exit(1)
    cohorts = sorted([d for d in PER_COHORT.iterdir() if d.is_dir()])
    print(f"Writing top10 3D SDFs for {len(cohorts)} cohorts...")
    for cd in cohorts:
        make_for_cohort(cd)


if __name__ == "__main__":
    main()
