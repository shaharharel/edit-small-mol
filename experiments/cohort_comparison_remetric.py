#!/usr/bin/env python3
"""Re-compute pose-derived metrics for already-docked cohorts using updated
thresholds (e.g. relaxed hinge_hbond_top1 cutoff). Reads cached pose PDBQTs +
ligand PDBQTs and rewrites per_mol.csv columns in-place.

Idempotent — safe to run multiple times.
"""
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

sys.path.insert(0, str(Path(__file__).parent))
from cohort_comparison_dock import (
    CYS346_SG, MET414_N, MET414_C, HINGE_HBOND_DIST_MAX, EXPECTED_CB_SG_DIST,
    BD_TARGET_DEG, parse_pdbqt_all_models, parse_meeko_smiles_idx,
    find_warhead_indices, map_mol_to_pose_indices, compute_pose_metrics,
    load_pocket_atoms, BOX_CENTER, RECEPTOR_PDB, acrylamide_on_largest_frag,
)

PROJECT_ROOT = Path(__file__).parent.parent
PER_COHORT = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison" / "per_cohort"


def remetric_cohort(cohort_dir: Path):
    csv_path = cohort_dir / "per_mol.csv"
    if not csv_path.exists():
        return False
    df = pd.read_csv(csv_path)
    pose_dir = cohort_dir / "poses"
    lig_dir = cohort_dir / "ligands_pdbqt"
    pocket_coords = load_pocket_atoms(RECEPTOR_PDB, BOX_CENTER, radius=12.0)
    n_updated = 0
    for i, row in df.iterrows():
        if not row.get("success"):
            continue
        name = row["name"]
        smi = row["smi"]
        pose = pose_dir / f"{name}_pose.pdbqt"
        lig = lig_dir / f"{name}.pdbqt"
        if not pose.exists():
            continue
        geo = compute_pose_metrics(pose, lig, smi)
        df.at[i, "d_cb_sg_top1"] = geo["d_cb_sg_top1"]
        df.at[i, "d_cb_sg_diff_from_185"] = geo["d_cb_sg_diff_from_185"]
        df.at[i, "bd_angle_top1"] = geo["bd_angle_top1"]
        df.at[i, "bd_angle_diff_from_107"] = (
            abs(geo["bd_angle_top1"] - BD_TARGET_DEG) if geo["bd_angle_top1"] is not None else None
        )
        df.at[i, "hinge_hbond_top1"] = geo["hinge_hbond_top1"]
        df.at[i, "d_hinge_top1"] = geo.get("d_hinge_top1")
        df.at[i, "any_pose_feasible"] = geo["any_pose_feasible"]
        df.at[i, "n_poses"] = geo["n_poses"]
        df.at[i, "warhead_found_in_pose"] = int(geo["warhead_found"])

        # Clash count from top model
        if pocket_coords is not None:
            poses_all = parse_pdbqt_all_models(pose)
            if poses_all:
                top = poses_all[0]
                heavy_mask = np.array([e != "H" for e in top["elements"]])
                if heavy_mask.sum() > 0:
                    lig_coords = top["coords"][heavy_mask]
                    dmat = np.linalg.norm(lig_coords[:, None, :] - pocket_coords[None, :, :], axis=2)
                    df.at[i, "clash_count_top1"] = int((dmat < 2.0).sum())
        # Ligand efficiency
        n_heavy = row.get("n_heavy_atoms", 0)
        vina_score = row.get("vina_score")
        if pd.notna(n_heavy) and n_heavy > 0 and pd.notna(vina_score):
            df.at[i, "ligand_efficiency"] = vina_score / n_heavy
        n_updated += 1
    df.to_csv(csv_path, index=False)
    print(f"  [ok] {cohort_dir.name}: re-metric'd {n_updated}/{len(df)} rows")
    return True


def main():
    if not PER_COHORT.exists():
        print(f"No per_cohort dir: {PER_COHORT}")
        sys.exit(1)
    cohorts = sorted([d for d in PER_COHORT.iterdir() if d.is_dir()])
    print(f"Re-metric'ing {len(cohorts)} cohorts (relaxed hinge threshold = {HINGE_HBOND_DIST_MAX} A)...")
    for cd in cohorts:
        remetric_cohort(cd)


if __name__ == "__main__":
    main()
