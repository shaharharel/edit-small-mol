#!/usr/bin/env python3
"""Extract per-mol Boltz metrics from cofold results.

Reads ~/boltz_run/results/boltz_results_<uid>/predictions/<uid>/*
Writes: ~/boltz_run/lingo3dmol_boltz_metrics.csv

Per mol:
  uid, smi, status,
  boltz_d_SG (Angstrom: ligand warhead C to Cys346 SG),
  boltz_BD_angle (Burgi-Dunitz: CA-CB-SG-warheadC dihedral / SG-C-C angle),
  boltz_ligand_iptm, boltz_complex_iptm, boltz_complex_plddt, boltz_confidence_score,
  pose_RMSD_to_native (if native pose available),
  Met414_hinge_dist (closest ligand heavy atom to Met414 backbone N).
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
import numpy as np

from rdkit import Chem
from rdkit.Chem import AllChem

HOME = Path.home()
RUN = HOME / "boltz_run"
RESULTS = RUN / "results"
UNIQUE_CSV = RUN / "lingo3dmol_all_unique.csv"
NATIVE_SDF = RUN / "lingo3dmol_native_poses.sdf"
OUT_CSV = RUN / "lingo3dmol_boltz_metrics.csv"

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def parse_cif_atoms(cif: Path):
    """Return list of dicts: chain, resn, resi, atom_name, elem, x, y, z."""
    lines = cif.read_text().splitlines()
    atoms = []
    i = 0
    while i < len(lines):
        if lines[i].startswith("loop_"):
            j = i + 1
            cols = []
            while j < len(lines) and lines[j].startswith("_atom_site."):
                cols.append(lines[j].strip().removeprefix("_atom_site."))
                j += 1
            if not cols:
                i = j; continue
            # Index relevant columns
            try:
                ci = {n: cols.index(n) for n in ["auth_asym_id","label_atom_id","auth_comp_id","auth_seq_id","Cartn_x","Cartn_y","Cartn_z","type_symbol"]}
            except ValueError:
                # try fallbacks
                try:
                    ci = {
                        "auth_asym_id": cols.index("label_asym_id"),
                        "label_atom_id": cols.index("label_atom_id"),
                        "auth_comp_id": cols.index("label_comp_id"),
                        "auth_seq_id": cols.index("label_seq_id"),
                        "Cartn_x": cols.index("Cartn_x"),
                        "Cartn_y": cols.index("Cartn_y"),
                        "Cartn_z": cols.index("Cartn_z"),
                        "type_symbol": cols.index("type_symbol"),
                    }
                except ValueError:
                    i = j; continue
            i = j
            while i < len(lines) and not lines[i].startswith("#") and not lines[i].startswith("loop_") and lines[i].strip():
                parts = lines[i].split()
                if len(parts) >= len(cols):
                    try:
                        atoms.append({
                            "chain": parts[ci["auth_asym_id"]].strip('"'),
                            "atom": parts[ci["label_atom_id"]].strip('"'),
                            "resn": parts[ci["auth_comp_id"]].strip('"'),
                            "resi": int(parts[ci["auth_seq_id"]].strip('"')),
                            "elem": parts[ci["type_symbol"]].strip('"'),
                            "x": float(parts[ci["Cartn_x"]]),
                            "y": float(parts[ci["Cartn_y"]]),
                            "z": float(parts[ci["Cartn_z"]]),
                        })
                    except (ValueError, IndexError):
                        pass
                i += 1
            continue
        i += 1
    return atoms


def vec(a, b):
    return np.array([b["x"]-a["x"], b["y"]-a["y"], b["z"]-a["z"]])


def norm(v):
    return v / (np.linalg.norm(v) + 1e-9)


def angle_deg(p1, p2, p3):
    """Angle (deg) at p2 formed by p1-p2-p3."""
    v1 = norm(vec(p2, p1))
    v2 = norm(vec(p2, p3))
    c = np.clip(np.dot(v1, v2), -1, 1)
    return float(np.degrees(np.arccos(c)))


def closest_ligand_to_residue_n(lig_atoms, prot_atoms, resi):
    """Min distance from any ligand heavy atom to backbone N of resi."""
    bN = [a for a in prot_atoms if a["resi"] == resi and a["atom"] == "N"]
    if not bN: return None
    Np = np.array([bN[0]["x"], bN[0]["y"], bN[0]["z"]])
    dists = []
    for a in lig_atoms:
        if a["elem"] == "H": continue
        p = np.array([a["x"], a["y"], a["z"]])
        dists.append(float(np.linalg.norm(p - Np)))
    return min(dists) if dists else None


def compute_rmsd(lig_sdf: Path, native_mol):
    """Pose RMSD between Boltz-predicted ligand (SDF) and native pose."""
    try:
        pred = Chem.SDMolSupplier(str(lig_sdf), removeHs=True, sanitize=True)[0]
        if pred is None: return None
        nat = Chem.RemoveHs(native_mol)
        if pred.GetNumAtoms() != nat.GetNumAtoms(): return None
        # Try GetBestRMS (canonicalizes atom mapping by graph isomorphism)
        try:
            return float(AllChem.GetBestRMS(pred, nat))
        except Exception:
            return float(AllChem.AlignMol(pred, nat))
    except Exception:
        return None


def main():
    if not UNIQUE_CSV.exists():
        print(f"ERROR: missing {UNIQUE_CSV}", file=sys.stderr); sys.exit(1)

    # Load unique mols
    mols = list(csv.DictReader(UNIQUE_CSV.open()))
    by_uid = {r["uid"]: r for r in mols}
    print(f"Loaded {len(mols)} unique mols")

    # Load native poses indexed by uid (name) + canonical_smi
    native_by_uid = {}
    if NATIVE_SDF.exists():
        for m in Chem.SDMolSupplier(str(NATIVE_SDF), removeHs=False, sanitize=True):
            if m is None: continue
            nm = m.GetProp("_Name") if m.HasProp("_Name") else None
            if nm and nm in by_uid:
                native_by_uid[nm] = m
        print(f"Loaded {len(native_by_uid)} native poses for RMSD")

    rows = []
    n_ok = n_fail = n_missing = 0
    for r in mols:
        uid = r["uid"]
        smi = r["canonical_smi"]
        pred_dir = RESULTS / f"boltz_results_{uid}" / "predictions" / uid
        cif = pred_dir / f"{uid}_model_0.cif"
        conf_json = pred_dir / f"confidence_{uid}_model_0.json"
        lig_sdf = pred_dir / f"{uid}_model_0.lig.sdf"
        if not (cif.exists() and conf_json.exists()):
            n_missing += 1
            rows.append({"uid": uid, "smi": smi, "status": "missing", **{k: "" for k in [
                "boltz_d_SG","boltz_BD_angle","boltz_ligand_iptm","boltz_complex_iptm",
                "boltz_complex_plddt","boltz_confidence_score","pose_RMSD_to_native","Met414_hinge_dist"]}})
            continue
        try:
            conf = json.loads(conf_json.read_text())
            atoms = parse_cif_atoms(cif)
            prot = [a for a in atoms if a["chain"] == "A"]
            lig = [a for a in atoms if a["chain"] == "B"]
            # Find Cys346 SG, CA, CB
            sg = next((a for a in prot if a["resi"] == 346 and a["atom"] == "SG"), None)
            ca = next((a for a in prot if a["resi"] == 346 and a["atom"] == "CA"), None)
            cb = next((a for a in prot if a["resi"] == 346 and a["atom"] == "CB"), None)
            # Warhead carbon: heavy atom in ligand closest to SG (assume covalent attachment)
            warhead = None
            if sg and lig:
                Sg_pos = np.array([sg["x"], sg["y"], sg["z"]])
                heavy = [a for a in lig if a["elem"] != "H"]
                heavy_sorted = sorted(heavy, key=lambda a: np.linalg.norm(np.array([a["x"],a["y"],a["z"]]) - Sg_pos))
                if heavy_sorted: warhead = heavy_sorted[0]
            d_SG = None; BD = None
            if sg and warhead:
                d_SG = float(np.linalg.norm(np.array([sg["x"],sg["y"],sg["z"]]) - np.array([warhead["x"],warhead["y"],warhead["z"]])))
            if cb and sg and warhead:
                BD = angle_deg(cb, sg, warhead)
            met414 = closest_ligand_to_residue_n(lig, prot, 414)
            # RMSD
            rmsd = None
            if uid in native_by_uid and lig_sdf.exists():
                rmsd = compute_rmsd(lig_sdf, native_by_uid[uid])
            rows.append({
                "uid": uid, "smi": smi, "status": "ok",
                "boltz_d_SG": d_SG, "boltz_BD_angle": BD,
                "boltz_ligand_iptm": conf.get("ligand_iptm"),
                "boltz_complex_iptm": conf.get("iptm") or conf.get("complex_iptm"),
                "boltz_complex_plddt": conf.get("complex_plddt"),
                "boltz_confidence_score": conf.get("confidence_score"),
                "pose_RMSD_to_native": rmsd,
                "Met414_hinge_dist": met414,
            })
            n_ok += 1
        except Exception as e:
            n_fail += 1
            rows.append({"uid": uid, "smi": smi, "status": f"err:{e}", **{k:"" for k in [
                "boltz_d_SG","boltz_BD_angle","boltz_ligand_iptm","boltz_complex_iptm",
                "boltz_complex_plddt","boltz_confidence_score","pose_RMSD_to_native","Met414_hinge_dist"]}})

    with OUT_CSV.open("w") as fh:
        fields = ["uid","smi","status","boltz_d_SG","boltz_BD_angle","boltz_ligand_iptm",
                  "boltz_complex_iptm","boltz_complex_plddt","boltz_confidence_score",
                  "pose_RMSD_to_native","Met414_hinge_dist"]
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader(); w.writerows(rows)
    print(f"Wrote {OUT_CSV}  ok={n_ok} fail={n_fail} missing={n_missing}")


if __name__ == "__main__":
    main()
