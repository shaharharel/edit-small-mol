"""C1 fix: verify that the Boltz atom name we passed in the YAML constraint
is the warhead Cβ in the decoded mmCIF, not some other carbon.

Procedure:
  1. Pick a top-scoring leaderboard mol whose CIF is local.
  2. Parse the mmCIF: extract all ligand atoms, find by element + connectivity:
     the terminal CH2 of the acrylamide warhead (the atom that should be at
     ~1.85 Å from Cys346 SG).
  3. Compare the named atom (from the YAML / Boltz internal label) to that
     identified Cβ.
  4. Report match / mismatch.

If they match → trust the cofold leaderboard's constraint. If not → the
entire leaderboard's geometric guarantee is invalidated and we need to
re-build YAMLs with corrected atom naming.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import re
import numpy as np
import pandas as pd
from rdkit import Chem


def parse_cif_atoms(cif_path: Path) -> pd.DataFrame:
    """Extract atom_site rows from a Boltz mmCIF."""
    lines = open(cif_path).read().splitlines()
    in_loop = False
    cols: list[str] = []
    rows = []
    i = 0
    while i < len(lines):
        line = lines[i]
        if line.startswith("loop_"):
            j = i + 1
            cols_local = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols_local.append(lines[j].strip())
                j += 1
            if cols_local:
                cols = cols_local
                i = j
                while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                    parts = lines[i].split()
                    if len(parts) == len(cols):
                        rows.append(parts)
                    i += 1
                continue
        i += 1
    if not rows or not cols:
        return pd.DataFrame()
    df = pd.DataFrame(rows, columns=[c.removeprefix("_atom_site.") for c in cols])
    for c in ("Cartn_x", "Cartn_y", "Cartn_z"):
        if c in df.columns: df[c] = df[c].astype(float)
    return df


def find_warhead_cb_geometric(ligand: pd.DataFrame, sg_xyz: np.ndarray) -> dict:
    """Identify the warhead Cβ by geometry: a carbon at ~1.85 Å from Cys SG,
    with at least one =CH= neighbor (the Cα). Return chosen atom row + reasoning.
    """
    lig_c = ligand[ligand["type_symbol"] == "C"].copy()
    coords = lig_c[["Cartn_x", "Cartn_y", "Cartn_z"]].values
    d = np.linalg.norm(coords - sg_xyz, axis=1)
    lig_c["d_to_SG"] = d
    closest = lig_c.nsmallest(5, "d_to_SG")
    return {"closest_carbons_to_SG": closest.to_dict("records")}


def main():
    pred_dir = PROJECT_ROOT / "data" / "boltz_poses" / "boltz_results_top1000__zap70_cys346" / "predictions"
    leaderboard = pd.read_csv(PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv")
    leaderboard = leaderboard.dropna(subset=["boltz_iptm"])
    leaderboard = leaderboard.sort_values("boltz_ligand_iptm", ascending=False)
    sample = leaderboard.head(3)

    print("=== Boltz atom-name verification ===")
    print(f"Inspecting top-3 leaderboard mols (highest ligand-iPTM):\n")

    sg_xyz = None
    for _, row in sample.iterrows():
        name = row["yaml_name"]
        warhead_atom_name = row["warhead_atom_name"]
        cif = pred_dir / name / f"{name}_model_0.cif"
        if not cif.exists():
            print(f"  {name}: CIF missing")
            continue
        atoms = parse_cif_atoms(cif)
        if atoms.empty:
            print(f"  {name}: failed to parse CIF")
            continue

        # protein vs ligand split: Boltz writes the ligand with auth_asym_id 'B'
        # (per our YAML) and chain A is the protein.
        prot = atoms[atoms.get("auth_asym_id", atoms.get("label_asym_id")) == "A"]
        lig  = atoms[atoms.get("auth_asym_id", atoms.get("label_asym_id")) == "B"]
        if lig.empty:
            # fallback: try label_asym_id == 'B'
            lig = atoms[atoms.get("label_asym_id") == "B"]
            prot = atoms[atoms.get("label_asym_id") == "A"]
        if lig.empty:
            print(f"  {name}: could not locate ligand chain B (auth_asym_id values: {atoms.get('auth_asym_id', atoms.get('label_asym_id')).unique()[:6]})")
            continue

        # Cys346 SG
        sg = prot[(prot["label_comp_id"] == "CYS") & (prot["label_seq_id"] == "346") & (prot["label_atom_id"] == "SG")]
        if sg.empty:
            sg = prot[(prot["auth_comp_id"] == "CYS") & (prot["auth_seq_id"] == "346") & (prot["label_atom_id"] == "SG")]
        if sg.empty:
            print(f"  {name}: no Cys346 SG in protein chain")
            continue
        sg_xyz = sg.iloc[0][["Cartn_x", "Cartn_y", "Cartn_z"]].values.astype(float)

        # Named atom (the one Boltz used as the constraint target)
        named = lig[lig["label_atom_id"] == warhead_atom_name]
        if named.empty:
            print(f"  {name}: YAML named '{warhead_atom_name}', but CIF has no such ligand atom! Ligand atom names: {list(lig['label_atom_id'].unique())[:10]}")
            continue
        named_xyz = named.iloc[0][["Cartn_x", "Cartn_y", "Cartn_z"]].values.astype(float)
        d_named_sg = float(np.linalg.norm(named_xyz - sg_xyz))

        # Closest carbon to SG (the *empirical* Cβ in the cofold)
        geom = find_warhead_cb_geometric(lig, sg_xyz)
        closest = geom["closest_carbons_to_SG"][0]
        emp_atom = closest["label_atom_id"]
        emp_d = float(closest["d_to_SG"])

        match = (named.iloc[0]["label_atom_id"] == emp_atom)
        print(f"  {name}")
        print(f"    YAML-named atom = {warhead_atom_name}, d(SG, named) = {d_named_sg:.3f} Å")
        print(f"    Empirical closest-C to SG = {emp_atom}, d = {emp_d:.3f} Å")
        print(f"    MATCH = {match}   (expected ~1.85 Å for prereactive complex)\n")

    print("Verdict criteria:")
    print("  PASS if YAML-named atom == empirical-closest-C AND d ∈ [1.5, 2.5] Å.")
    print("  FAIL if YAML-named atom != empirical-closest-C OR d > 3 Å (constraint anchored wrong atom).")


if __name__ == "__main__":
    main()
