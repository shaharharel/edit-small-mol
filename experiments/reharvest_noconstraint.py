#!/usr/bin/env python3
"""Re-harvest Boltz no-constraint cofolds with correct β-C atom identification.

The original extract_metrics_one() picks the "closest ligand C to Sγ", which is
wrong when the pose is far from Cys (any random alkyl C could be closest). This
script instead:

  1. Parses each YAML in the yaml_dir to recover the SMILES + warhead atom name
     (using the same boltz_atom_name() logic as build_yamls).
  2. Locates that exact atom in the cofold CIF ligand block.
  3. Computes d_β-Sγ, θ_BD (β-C -> Sγ, β-C -> α-C), and phi.

Falls back to the original heuristic only if the exact atom name is not present
in the CIF (e.g., if canonical numbering changed).

Usage:
    python experiments/reharvest_noconstraint.py \\
        --cofold_root data/paper_pair_training/v2_curriculum_clean/noconstraint_cofolds \\
        --yaml_root data/paper_pair_training/v2_curriculum_clean/noconstraint_cofolds/yamls
"""
from __future__ import annotations
import argparse
import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# For rdkit + boltz_atom_name + parse_cif_atoms
sys.path.insert(0, "/Users/shaharharel/Documents/github/edit-small-mol/experiments")


def boltz_atom_name(smi: str):
    from rdkit import Chem, RDLogger
    from rdkit.Chem import AllChem
    RDLogger.DisableLog("rdApp.*")
    ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol_h = Chem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACRYL)
    if not matches:
        return None, None
    term_ch2_idx = matches[0][0]
    return f"C{can[term_ch2_idx] + 1}", term_ch2_idx


def parse_yaml_smi(yaml_path: Path):
    """Extract SMILES from a Boltz YAML (simple regex parse)."""
    for line in yaml_path.read_text().splitlines():
        line = line.strip()
        if line.startswith("smiles:"):
            v = line.removeprefix("smiles:").strip()
            return v.strip("'\"")
    return None


def parse_cif_atoms(cif_path: Path):
    """Extract chain-A protein atoms + chain-B ligand atoms via gemmi."""
    import gemmi
    st = gemmi.read_structure(str(cif_path))
    prot, lig = [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    rec = {"chain": chain.name, "resname": res.name,
                           "resi": res.seqid.num, "name": atom.name,
                           "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z],
                                            dtype=float)}
                    if chain.name == "A":
                        prot.append(rec)
                    else:
                        lig.append(rec)
        break
    return prot, lig


def compute_metrics(cif_path: Path, warhead_atom_name: str, target_cys: int = 346):
    prot, lig = parse_cif_atoms(cif_path)
    # SG of TARGET_CYS
    sg = None
    for a in prot:
        if a["resi"] == target_cys and a["name"] == "SG":
            sg = a["pos"]
            break
    if sg is None:
        return {"cofold_ok": False, "err": "no_sg"}
    # Find warhead β-C by name in ligand
    beta = None
    for a in lig:
        if a["name"] == warhead_atom_name:
            beta = a["pos"]
            break
    if beta is None:
        # Fallback: closest C to sg
        Cs = [a for a in lig if a["name"].startswith("C")]
        if not Cs:
            return {"cofold_ok": False, "err": "no_c"}
        dists = [np.linalg.norm(a["pos"] - sg) for a in Cs]
        j = int(np.argmin(dists))
        beta = Cs[j]["pos"]
        used_atom = f"fallback:{Cs[j]['name']}"
    else:
        used_atom = warhead_atom_name
    d = float(np.linalg.norm(beta - sg))
    # Find neighbors of β-C: bonded atoms within 1.8Å
    nb = []
    for a in lig:
        if np.allclose(a["pos"], beta):
            continue
        dd = np.linalg.norm(a["pos"] - beta)
        if dd < 1.8:
            nb.append((dd, a))
    nb.sort(key=lambda x: x[0])
    m = {"cofold_ok": True, "d_b_nuc_angstrom": d,
         "warhead_atom_used": used_atom, "n_beta_neighbors": len(nb)}
    if nb:
        alpha = nb[0][1]["pos"]
        v1 = sg - beta
        v2 = alpha - beta
        cosv = float(np.dot(v1, v2) / (np.linalg.norm(v1) *
                                        np.linalg.norm(v2) + 1e-9))
        cosv = float(np.clip(cosv, -1, 1))
        m["bd_angle_deg"] = float(np.degrees(np.arccos(cosv)))
    else:
        m["bd_angle_deg"] = float("nan")
    # phi_planar_deg: dihedral Sγ - β-C - α-C - next
    if len(nb) >= 2:
        alpha = nb[0][1]["pos"]
        c3 = nb[1][1]["pos"]
        b1 = beta - sg
        b2 = alpha - beta
        b3 = c3 - alpha
        n1 = np.cross(b1, b2)
        n2 = np.cross(b2, b3)
        b2n = b2 / (np.linalg.norm(b2) + 1e-9)
        m1 = np.cross(n1, b2n)
        x = float(np.dot(n1, n2))
        y = float(np.dot(m1, n2))
        phi = float(np.degrees(np.arctan2(y, x)))
        phi_wrap = ((phi + 180.0) % 360.0) - 180.0
        m["phi_planar_deg"] = float(min(abs(phi_wrap),
                                          abs(180.0 - abs(phi_wrap))))
    else:
        m["phi_planar_deg"] = float("nan")
    return m


def _load_confidence(pred_dir: Path, name: str):
    cj = pred_dir / f"confidence_{name}_model_0.json"
    if not cj.exists():
        return {}
    try:
        d = json.loads(cj.read_text())
        return {"complex_iptm": d.get("iptm") or d.get("complex_iptm"),
                "ligand_iptm": d.get("ligand_iptm"),
                "complex_plddt": d.get("complex_plddt")}
    except Exception:
        return {}


def reharvest_cohort(cofold_root: Path, yaml_root: Path, cohort_name: str,
                     out_csv: Path):
    yaml_dir = yaml_root / cohort_name
    cofold_dir = cofold_root / "cofolds" / cohort_name
    if not cofold_dir.exists():
        print(f"[warn] no cofold dir for {cohort_name}")
        return
    rows = []
    for sub in sorted(cofold_dir.glob("boltz_results_*")):
        name = sub.name.removeprefix("boltz_results_")
        pred_dir = sub / "predictions" / name
        cif = pred_dir / f"{name}_model_0.cif"
        if not cif.exists():
            continue
        yaml_path = yaml_dir / f"{name}.yaml"
        smi = parse_yaml_smi(yaml_path) if yaml_path.exists() else None
        wh_name, _ = boltz_atom_name(smi) if smi else (None, None)
        m = compute_metrics(cif, wh_name)
        m["name"] = name
        m["smiles"] = smi
        m["cohort"] = cohort_name
        m.update(_load_confidence(pred_dir, name))
        rows.append(m)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    print(f"[reharvest] {cohort_name}: wrote {len(df)} rows → {out_csv}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofold_root", required=True)
    ap.add_argument("--yaml_root", required=True)
    args = ap.parse_args()

    cofold_root = Path(args.cofold_root)
    yaml_root = Path(args.yaml_root)
    for c in ["theta_90", "theta_105", "theta_130", "null_pose"]:
        cohort_name = f"v2curr_clean_NOCONSTR_{c}"
        out_csv = cofold_root / f"track_A_{cohort_name}.csv"
        reharvest_cohort(cofold_root, yaml_root, cohort_name, out_csv)


if __name__ == "__main__":
    main()
