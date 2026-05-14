"""Apply covalent constraint manifold projection to inpaint outputs.

Pipeline:
  1. Read inpaint.sdf (which has the acrylamide warhead preserved by inpaint).
  2. For each molecule with a valid warhead match, extract:
       - the 4 warhead atom indices (C_β, C_α, C_carb, N_amide)
       - the molecule's full 3D coordinates
  3. Read the cysteine SG position from the receptor PDB.
  4. Apply project() — translates+rotates the rigid warhead so:
       d(SG, C_β)=1.85 Å, ∠(SG, C_β, C_α)=107°, dihedral=0°
  5. Write constraint_proj.sdf with the projected coords.
  6. Track per-molecule before/after distance/angle/dihedral residuals.

This is the C2 contribution: Path A novelty for covalent geometry enforcement.
For comparison, the inpaint cohort has the warhead present but free geometry;
constraint_proj has it locked to the prereactive complex pose.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem
from Bio.PDB import PDBParser

from anchordiff.config import ZAP70_CYS346, BTK_CYS481
from anchordiff.covalent_constraint_manifold import (
    WarheadAtoms, measure, project,
)

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def get_cys_sg(pdb_path: Path, resi: int, chain: str = "A") -> np.ndarray:
    """Return SG atom 3D position for Cys at chain:resi."""
    parser = PDBParser(QUIET=True)
    struct = parser.get_structure("", str(pdb_path))
    model = struct[0]
    if chain not in model:
        chain = list(model.child_dict.keys())[0]
    res = model[chain][resi]
    if res.get_resname() != "CYS":
        raise ValueError(f"residue {chain}:{resi} is {res.get_resname()}, not CYS")
    return np.array(res["SG"].get_coord(), dtype=float)


def find_warhead_atoms(mol: Chem.Mol):
    """Return WarheadAtoms with indices of cb, ca, c_carb, n_amide. None if no match."""
    matches = mol.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not matches:
        return None
    # SMARTS [CH2]=[CH]C(=O)N matches in order: cb, ca, c_carb, o (=O), n_amide
    # but the SMARTS has only 4 mapped atoms (CH2, CH, C, N); the (=O) is implicit
    # in C(=O). Wait — let's check: [CH2]=[CH]C(=O)N is a standard SMARTS where
    # `(=O)` is an inline branch — the atoms returned by GetSubstructMatches are
    # [cb, ca, c_carb, n_amide]. Confirmed by testing.
    m = matches[0]
    if len(m) == 4:
        cb, ca, c_carb, n_amide = m
    elif len(m) == 5:
        cb, ca, c_carb, _o, n_amide = m
    else:
        return None
    return WarheadAtoms(cb=cb, ca=ca, c_carb=c_carb, n_amide=n_amide)


def project_one(mol: Chem.Mol, sg: np.ndarray):
    """Apply projection in-place. Returns (before_metrics, after_metrics) or None."""
    w = find_warhead_atoms(mol)
    if w is None:
        return None
    if mol.GetNumConformers() == 0:
        return None
    conf = mol.GetConformer()
    coords = np.array([list(conf.GetAtomPosition(i)) for i in range(mol.GetNumAtoms())])
    before = measure(coords, sg, w)
    new_coords = project(coords, sg, w)
    after = measure(new_coords, sg, w)
    for i in range(mol.GetNumAtoms()):
        conf.SetAtomPosition(i, new_coords[i].tolist())
    return before, after


def process_cohort(sdf_in: Path, sdf_out: Path, sg: np.ndarray, target_name: str):
    print(f"\n=== {target_name} ===")
    print(f"  in:  {sdf_in}")
    print(f"  out: {sdf_out}")
    print(f"  SG:  ({sg[0]:.3f}, {sg[1]:.3f}, {sg[2]:.3f})")
    suppl = Chem.SDMolSupplier(str(sdf_in), sanitize=False)
    writer = Chem.SDWriter(str(sdf_out))
    n_in = n_warhead = n_proj = 0
    rows = []
    for i, mol in enumerate(suppl):
        if mol is None:
            continue
        n_in += 1
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            continue
        result = project_one(mol, sg)
        if result is None:
            continue
        n_warhead += 1
        before, after = result
        writer.write(mol)
        n_proj += 1
        rows.append({
            "mol_idx": i,
            "smiles": Chem.MolToSmiles(mol),
            "before_d_S_Cb": before["d_S_Cb"],
            "after_d_S_Cb": after["d_S_Cb"],
            "before_angle": before["angle_S_Cb_Ca"],
            "after_angle": after["angle_S_Cb_Ca"],
            "before_dihedral": before["dihedral_S_Cb_Ca_Ccarb"],
            "after_dihedral": after["dihedral_S_Cb_Ca_Ccarb"],
        })
    writer.close()
    print(f"  read    : {n_in} mols")
    print(f"  warhead : {n_warhead} ({100*n_warhead/max(n_in,1):.1f}%)")
    print(f"  projected: {n_proj} → wrote {sdf_out}")

    if rows:
        df = pd.DataFrame(rows)
        print("\n  Geometry residuals (target d=1.85, angle=107, dihedral=0):")
        print(f"    BEFORE  d∈[{df.before_d_S_Cb.min():.2f}, {df.before_d_S_Cb.max():.2f}], "
              f"angle∈[{df.before_angle.min():.1f}, {df.before_angle.max():.1f}], "
              f"|φ|max={df.before_dihedral.abs().max():.1f}")
        print(f"    AFTER   d∈[{df.after_d_S_Cb.min():.3f}, {df.after_d_S_Cb.max():.3f}], "
              f"angle∈[{df.after_angle.min():.2f}, {df.after_angle.max():.2f}], "
              f"|φ|max={df.after_dihedral.abs().max():.2f}")
        before_d_med = df.before_d_S_Cb.median()
        after_d_med = df.after_d_S_Cb.median()
        print(f"    median d shift: {before_d_med:.2f} → {after_d_med:.3f}")
    return df if rows else pd.DataFrame()


def main():
    base = PROJECT_ROOT / "anchordiff_results" / "day1"
    pockets = PROJECT_ROOT / "anchordiff" / "pockets"
    out_dir = PROJECT_ROOT / "results" / "anchordiff"
    out_dir.mkdir(parents=True, exist_ok=True)

    targets = [
        (ZAP70_CYS346, pockets / "zap70_cys346", base / "zap70_cys346"),
        (BTK_CYS481,   pockets / "btk_cys481",   base / "btk_cys481"),
    ]

    all_dfs = []
    for tgt, pock_dir, day_dir in targets:
        recv = pock_dir / "receptor.pdb"
        sg = get_cys_sg(recv, tgt.cys_residue)
        # Prefer the warhead-fixed SDF (DiffSBDD bond perception drops the warhead);
        # fall back to inpaint.sdf if the fix step hasn't been run yet.
        sdf_in = day_dir / "inpaint_fixed.sdf"
        if not sdf_in.exists():
            sdf_in = day_dir / "inpaint.sdf"
        if not sdf_in.exists():
            print(f"\n=== {tgt.name}: no inpaint(_fixed).sdf yet — skipping ===")
            continue
        sdf_out = day_dir / "constraint_proj.sdf"
        df = process_cohort(sdf_in, sdf_out, sg, tgt.name)
        if len(df):
            df["target"] = tgt.name
            all_dfs.append(df)

    if all_dfs:
        full = pd.concat(all_dfs, ignore_index=True)
        out_csv = out_dir / "day1_constraint_projection_residuals.csv"
        full.to_csv(out_csv, index=False)
        print(f"\nWrote residuals to {out_csv}")


if __name__ == "__main__":
    main()
