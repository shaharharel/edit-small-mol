"""Compute Boltz-derived columns for F4 cofolds pulled from A100 VMs.

Reads Boltz cofold dirs from:
  data/boltz_f4_results/<vm>/boltz_results_<row>/predictions/<row>/...
  data/boltz_f4_extra_results/<vm>/boltz_results_<row>/predictions/<row>/...

Computes the following columns per row_id and joins back to:
  data/tier4_scored/F4_boltz_pool_v2.csv    →  F4_boltz_pool_v2_with_boltz.csv
  data/tier4_scored/F4_boltz_extra_v2.csv   →  F4_boltz_extra_v2_with_boltz.csv

Columns produced:
  - boltz_ligand_iptm, boltz_iptm, complex_pde   (confidence JSON, direct)
  - mPAE_paper                                   (London 2026: min over protein×ligand block)
  - d_SG, burgi_dunitz_dev_deg, geom_ok          (warhead Cβ vs Cys346 SG)
  - n_h_bonds, n_stabilizing_contacts            (CIF-based contact analysis)
  - pocket_occupancy_pct                         (ligand atoms within 4Å of ATP-pocket residues)
  - vina_rescore_affinity_kcalmol, vina_rescore_intra_kcalmol   (Vina --score_only)
  - rdkit_strain_kcal_mol                        (MMFF94 strain on un-relaxed pose)
  - pKa_Cys346                                   (PROPKA3 on cofolded complex)
  - shape_Tc_seed, esp_sim_seed                  (3D shape Tanimoto + ESP-Sim vs Mol1)

For row_id → boltz dir mapping:
  - Pool cohort: each VM has a manifest.csv (row00001 → semantic row_id)
  - Extra cohort: EXTRA_NNNNN dirs map directly to row_id

The script writes the output CSV every BATCH_SIZE rows so partial results
are available even if the script crashes.
"""
from __future__ import annotations
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

# Suppress noisy warnings BEFORE importing rdkit
import warnings
warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

import gemmi


PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
POOL_DIR = PROJECT_ROOT / "data/boltz_f4_results"
EXTRA_DIR = PROJECT_ROOT / "data/boltz_f4_extra_results"
TIER4_DIR = PROJECT_ROOT / "data/tier4_scored"

POOL_IN = TIER4_DIR / "F4_boltz_pool_v2.csv"
POOL_OUT = TIER4_DIR / "F4_boltz_pool_v2_with_boltz.csv"
EXTRA_IN = TIER4_DIR / "F4_boltz_extra_v2.csv"
EXTRA_OUT = TIER4_DIR / "F4_boltz_extra_v2_with_boltz.csv"

# Mol1 SMILES — the seed/reference for shape_Tc and esp_sim
MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cnc(NC(=O)c4cn(C)nc4C(C)C)cn3)c2C1"

# Use exporting from PATH (env quris)
QURIS_BIN = "/opt/miniconda3/envs/quris/bin"
VINA_CMD = f"{QURIS_BIN}/vina"
OBABEL_CMD = f"{QURIS_BIN}/obabel"
PROPKA_CMD = f"{QURIS_BIN}/propka3"
MEEKO_CMD = f"{QURIS_BIN}/mk_prepare_ligand.py"

# Constants from anchordiff/compute_pose_quality_v2.py
CYS_RESI = 346
ATP_POCKET_RESIS = {344, 345, 346, 347, 348, 349, 350, 352, 367, 369, 399,
                    414, 415, 416, 417, 418, 420, 421, 424,
                    461, 465, 466, 468, 479, 482}
BURGI_DUNITZ_DEG = 107.0
HB_MIN_A, HB_MAX_A = 2.5, 3.5
VDW_MIN_A, VDW_MAX_A = 3.5, 4.5  # Carbon-carbon contact range for stabilizing vdW
GEOM_OK_DSG = 5.0  # d_SG threshold for geom_ok


# ----------- row_id → cofold dir resolution -----------------------------------

def build_row_id_index(pool_dir: Path = POOL_DIR, extra_dir: Path = EXTRA_DIR) -> dict[str, Path]:
    """Map row_id → predictions dir (where _model_0.cif lives).

    For the main pool: read each <vm>/manifest.csv to map row00001 → semantic row_id.
    For the extra cohort: dir names already match row_id.
    """
    index: dict[str, Path] = {}

    # POOL: each VM has manifest.csv mapping name (rowNNNNN) → row_id
    for vm_dir in sorted(pool_dir.iterdir()) if pool_dir.exists() else []:
        if not vm_dir.is_dir():
            continue
        manifest = vm_dir / "manifest.csv"
        if manifest.exists():
            man = pd.read_csv(manifest)
            for _, mr in man.iterrows():
                cofold_name = mr["name"]
                row_id = mr["row_id"]
                pred_dir = vm_dir / f"boltz_results_{cofold_name}" / "predictions" / cofold_name
                if pred_dir.is_dir():
                    index[row_id] = pred_dir

    # EXTRA: dir name = row_id directly (EXTRA_NNNNN)
    for vm_dir in sorted(extra_dir.iterdir()) if extra_dir.exists() else []:
        if not vm_dir.is_dir():
            continue
        for sub in vm_dir.iterdir():
            if not sub.is_dir() or not sub.name.startswith("boltz_results_"):
                continue
            cofold_name = sub.name[len("boltz_results_"):]
            pred_dir = sub / "predictions" / cofold_name
            if pred_dir.is_dir():
                index[cofold_name] = pred_dir

    return index


# ----------- CIF parsing ------------------------------------------------------

def parse_cofold_cif(cif_path: Path) -> tuple[list[dict], list[dict]]:
    """Return (prot_atoms, lig_atoms): list of dicts {chain, resname, resi, name, element, pos}.

    Chain A = protein, chain B = ligand by Boltz convention.
    """
    st = gemmi.read_structure(str(cif_path))
    try:
        st.setup_entities()
    except Exception:
        pass
    prot, lig = [], []
    for model in st:
        for chain in model:
            for res in chain:
                for atom in res:
                    rec = {
                        "chain": chain.name, "resname": res.name, "resi": res.seqid.num,
                        "name": atom.name, "element": atom.element.name,
                        "pos": np.array([atom.pos.x, atom.pos.y, atom.pos.z], dtype=float),
                    }
                    if chain.name == "A":
                        prot.append(rec)
                    elif chain.name == "B":
                        lig.append(rec)
        break  # only first model
    return prot, lig


# ----------- Confidence + mPAE ------------------------------------------------

def load_confidence(json_path: Path) -> dict:
    if not json_path.exists():
        return {"boltz_iptm": None, "boltz_ligand_iptm": None, "complex_pde": None}
    try:
        d = json.loads(json_path.read_text())
    except Exception:
        return {"boltz_iptm": None, "boltz_ligand_iptm": None, "complex_pde": None}
    return {
        "boltz_iptm": d.get("iptm"),
        "boltz_ligand_iptm": d.get("ligand_iptm"),
        "complex_pde": d.get("complex_pde"),
    }


def mpae_paper(pae_npz: Path, n_lig_atoms: int) -> float | None:
    """London 2026 mPAE: min over protein×ligand block of PAE matrix.

    PAE is square (n_total, n_total) of pairwise expected aligned error.
    Ligand atoms are the last n_lig_atoms entries (Boltz convention).
    """
    if not pae_npz.exists():
        return None
    try:
        d = np.load(pae_npz)
        pae = d["pae"] if "pae" in d else d[list(d.keys())[0]]
    except Exception:
        return None
    if pae.ndim != 2:
        return None
    N = pae.shape[0]
    lig_lo = N - n_lig_atoms
    if lig_lo <= 0 or lig_lo >= N:
        return None
    cross = pae[:lig_lo, lig_lo:]
    if cross.size == 0:
        return None
    return float(np.min(cross))


# ----------- Geometry: d_SG, Bürgi-Dunitz -------------------------------------

# Soft warhead SMARTS (matches anchordiff/compute_pose_quality_v2.py)
WARHEAD_SMARTS_RAW = [
    ("acrylamide",    "[CH2]=[CH]C(=O)N"),
    ("acrylate",      "[CH2]=[CH]C(=O)O"),
    ("vinyl_sulfone", "[CH2]=[CH]S(=O)(=O)"),
    ("haloacetamide", "[Cl,Br,I][CH2]C(=O)N"),
    ("propiolamide",  "C#CC(=O)N"),
]
_WARHEAD_SMARTS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEAD_SMARTS_RAW]


def derive_warhead_atom_name(smi: str) -> str | None:
    """Return the boltz-style ligand atom name (e.g. 'C36') for the warhead Cβ.

    Matches the canonical-rank logic in experiments/gen_covalid_boltz_yamls.py:
        canonical_rank[term_ch2_idx] + 1 → atom name 'C<n>'
    """
    if not smi:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    for wh_class, sm in _WARHEAD_SMARTS:
        if sm is None:
            continue
        matches = mol_h.GetSubstructMatches(sm)
        if matches:
            term_idx = matches[0][0]  # first atom of SMARTS = Cβ (CH2= or [Cl/Br/I])
            return f"C{can[term_idx] + 1}"
    return None


def d_SG_from_cofold(prot: list[dict], lig: list[dict], warhead_atom_name: str | None) -> float | None:
    """Distance from Cys346.SG to the named ligand atom (warhead Cβ).
    No fallback if warhead_atom_name is missing."""
    if not warhead_atom_name:
        return None
    sg = next((p["pos"] for p in prot
               if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "SG"), None)
    if sg is None:
        return None
    cb = next((l["pos"] for l in lig if l["name"] == warhead_atom_name), None)
    if cb is None:
        return None
    return float(np.linalg.norm(sg - cb))


def find_alpha_carbon(lig: list[dict], cb_name: str) -> dict | None:
    """Closest other carbon to Cβ within 1.0-1.9 Å (typical C-C bond)."""
    cb = next((l for l in lig if l["name"] == cb_name), None)
    if cb is None:
        return None
    candidates = []
    for l in lig:
        if l["element"] != "C" or l["name"] == cb_name:
            continue
        d = float(np.linalg.norm(l["pos"] - cb["pos"]))
        if 1.0 <= d <= 1.9:
            candidates.append((d, l))
    if not candidates:
        return None
    candidates.sort(key=lambda x: x[0])
    return candidates[0][1]


def burgi_dunitz_dev(prot: list[dict], lig: list[dict], warhead_atom_name: str | None) -> float | None:
    """|angle(SG, Cβ, Cα) − 107°|. None if any atom missing."""
    if not warhead_atom_name:
        return None
    sg = next((p["pos"] for p in prot
               if p["resi"] == CYS_RESI and p["resname"] == "CYS" and p["name"] == "SG"), None)
    cb = next((l["pos"] for l in lig if l["name"] == warhead_atom_name), None)
    if sg is None or cb is None:
        return None
    ca = find_alpha_carbon(lig, warhead_atom_name)
    if ca is None:
        return None
    v1 = sg - cb
    v2 = ca["pos"] - cb
    cos_a = float(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2) + 1e-12))
    cos_a = max(-1.0, min(1.0, cos_a))
    angle_deg = float(np.degrees(np.arccos(cos_a)))
    return abs(angle_deg - BURGI_DUNITZ_DEG)


# ----------- Contacts / occupancy ---------------------------------------------

def count_h_bonds(prot: list[dict], lig: list[dict]) -> int:
    """Count ligand N/O ↔ protein N/O contacts at HB_MIN..HB_MAX Å (excl. Cys346.SG)."""
    n = 0
    for l in lig:
        if l["element"] not in ("N", "O"):
            continue
        for p in prot:
            if p["element"] not in ("N", "O"):
                continue
            if p["resi"] == CYS_RESI and p["name"] == "SG":
                continue
            d = float(np.linalg.norm(l["pos"] - p["pos"]))
            if HB_MIN_A <= d <= HB_MAX_A:
                n += 1
    return n


def count_stabilizing_contacts(prot: list[dict], lig: list[dict]) -> int:
    """Composite: H-bonds + non-polar (C-C) contacts at vdW range.

    Per scientific-analyst notes this is a HEURISTIC count, no double-counting with d_SG.
    """
    n_hb = count_h_bonds(prot, lig)
    n_vdw = 0
    for l in lig:
        if l["element"] != "C":
            continue
        for p in prot:
            if p["element"] != "C":
                continue
            d = float(np.linalg.norm(l["pos"] - p["pos"]))
            if VDW_MIN_A <= d <= VDW_MAX_A:
                n_vdw += 1
    return n_hb + n_vdw


def pocket_occupancy_pct(prot: list[dict], lig: list[dict], cutoff_A: float = 4.0) -> float | None:
    """Fraction of ligand heavy atoms within `cutoff` of any ATP-pocket residue heavy atom."""
    pocket_atoms = [p["pos"] for p in prot
                    if p["resi"] in ATP_POCKET_RESIS and p["element"] != "H"]
    lig_heavies = [l["pos"] for l in lig if l["element"] != "H"]
    if not lig_heavies or not pocket_atoms:
        return None
    pocket_arr = np.array(pocket_atoms)
    n_in = 0
    for lh in lig_heavies:
        d_min = float(np.min(np.linalg.norm(pocket_arr - lh, axis=1)))
        if d_min <= cutoff_A:
            n_in += 1
    return 100.0 * n_in / len(lig_heavies)


# ----------- Vina rescore -----------------------------------------------------

def cif_to_pdb_split(cif_path: Path, prot_pdb: Path, lig_sdf: Path) -> bool:
    """Convert cofold CIF into separate protein PDB + ligand SDF using gemmi+RDKit.

    Returns True on success.
    """
    try:
        st = gemmi.read_structure(str(cif_path))
        # Write protein chain (A) only to a PDB
        out_st = gemmi.Structure()
        out_st.cell = st.cell
        out_st.spacegroup_hm = st.spacegroup_hm
        new_model = gemmi.Model("1")
        for model in st:
            for chain in model:
                if chain.name == "A":
                    new_model.add_chain(chain.clone())
            break
        out_st.add_model(new_model)
        out_st.write_pdb(str(prot_pdb))

        # Extract ligand atoms (chain B) and write a minimal PDB block, then sanitize via RDKit
        lig_lines = []
        atom_count = 0
        for model in st:
            for chain in model:
                if chain.name != "B":
                    continue
                for res in chain:
                    for atom in res:
                        atom_count += 1
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                        name = atom.name.ljust(4)
                        elem = atom.element.name.rjust(2)
                        # PDB HETATM record
                        line = (
                            f"HETATM{atom_count:5d} {name} LIG B   1    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem}\n"
                        )
                        lig_lines.append(line)
            break
        if not lig_lines:
            return False
        pdb_block = "".join(lig_lines) + "END\n"
        mol = Chem.MolFromPDBBlock(pdb_block, removeHs=False, sanitize=False)
        if mol is None:
            return False
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                )
            except Exception:
                pass
        try:
            mol = Chem.AddHs(mol, addCoords=True)
        except Exception:
            pass
        w = Chem.SDWriter(str(lig_sdf))
        w.write(mol)
        w.close()
        return True
    except Exception:
        return False


def vina_rescore(cif_path: Path) -> dict:
    """Run Vina --score_only on the cofold pose. Returns dict with vina_rescore_affinity/intra."""
    out = {
        "vina_rescore_affinity_kcalmol": None,
        "vina_rescore_intra_kcalmol": None,
    }
    with tempfile.TemporaryDirectory(prefix="vina_") as tmpd:
        tmp = Path(tmpd)
        prot_pdb = tmp / "prot.pdb"
        lig_sdf = tmp / "lig.sdf"
        prot_pdbqt = tmp / "prot.pdbqt"
        lig_pdbqt = tmp / "lig.pdbqt"

        if not cif_to_pdb_split(cif_path, prot_pdb, lig_sdf):
            return out

        # Find Cys346.SG position from protein PDB
        sg = None
        try:
            with open(prot_pdb) as f:
                for line in f:
                    if (line.startswith("ATOM") and line[12:16].strip() == "SG"
                            and line[17:20].strip() == "CYS"
                            and line[22:26].strip() == "346"):
                        sg = (float(line[30:38]), float(line[38:46]), float(line[46:54]))
                        break
        except Exception:
            pass
        if sg is None:
            return out

        # Protein → PDBQT
        try:
            r = subprocess.run([OBABEL_CMD, str(prot_pdb), "-O", str(prot_pdbqt), "-xr"],
                               capture_output=True, text=True, timeout=120)
            if r.returncode != 0 or not prot_pdbqt.exists() or prot_pdbqt.stat().st_size == 0:
                return out
        except Exception:
            return out

        # Ligand → PDBQT (try meeko, fall back to obabel)
        meeko_ok = False
        try:
            r = subprocess.run([MEEKO_CMD, "-i", str(lig_sdf), "-o", str(lig_pdbqt)],
                               capture_output=True, text=True, timeout=120)
            if r.returncode == 0 and lig_pdbqt.exists() and lig_pdbqt.stat().st_size > 0:
                meeko_ok = True
        except Exception:
            pass
        if not meeko_ok:
            try:
                r = subprocess.run([OBABEL_CMD, str(lig_sdf), "-O", str(lig_pdbqt), "-h"],
                                   capture_output=True, text=True, timeout=120)
                if r.returncode != 0 or not lig_pdbqt.exists() or lig_pdbqt.stat().st_size == 0:
                    return out
            except Exception:
                return out

        # Compute box covering both Cys346.SG and ligand extent
        try:
            mins = None; maxs = None
            with open(lig_sdf) as f:
                lines = f.readlines()
            n_atoms = int(lines[3][:3])
            coords = []
            for i in range(4, 4 + n_atoms):
                parts = lines[i].split()
                coords.append([float(parts[0]), float(parts[1]), float(parts[2])])
            arr = np.array(coords)
            mins = arr.min(axis=0); maxs = arr.max(axis=0)
            sg_arr = np.array(sg)
            box_lo = np.minimum(mins, sg_arr) - 8.0
            box_hi = np.maximum(maxs, sg_arr) + 8.0
            center = (box_lo + box_hi) / 2.0
            extents = box_hi - box_lo
            box_size = np.maximum(extents, 22.0)
        except Exception:
            return out

        # Run vina
        try:
            cmd = [
                VINA_CMD,
                "--receptor", str(prot_pdbqt),
                "--ligand", str(lig_pdbqt),
                "--score_only",
                "--center_x", f"{center[0]:.3f}",
                "--center_y", f"{center[1]:.3f}",
                "--center_z", f"{center[2]:.3f}",
                "--size_x", f"{box_size[0]:.3f}",
                "--size_y", f"{box_size[1]:.3f}",
                "--size_z", f"{box_size[2]:.3f}",
            ]
            r = subprocess.run(cmd, capture_output=True, text=True, timeout=120)
            if r.returncode != 0:
                return out
            affinity = None; intra = None
            for line in r.stdout.splitlines():
                s = line.strip()
                if s.startswith("Estimated Free Energy of Binding"):
                    m = re.search(r":\s*([-+]?[0-9]*\.?[0-9]+)", s)
                    if m:
                        affinity = float(m.group(1))
                elif s.startswith("Affinity:"):
                    m = re.search(r":\s*([-+]?[0-9]*\.?[0-9]+)", s)
                    if m:
                        affinity = float(m.group(1))
                elif "Final Total Internal Energy" in s:
                    m = re.search(r":\s*([-+]?[0-9]*\.?[0-9]+)", s)
                    if m:
                        intra = float(m.group(1))
            out["vina_rescore_affinity_kcalmol"] = affinity
            out["vina_rescore_intra_kcalmol"] = intra
        except Exception:
            pass
    return out


# ----------- RDKit strain + shape Tc + ESP-Sim --------------------------------

def extract_ligand_mol_from_cif(cif_path: Path) -> Chem.Mol | None:
    """Build an RDKit ligand mol from chain B of the CIF.
    Returns mol with 3D coords and inferred bonds; None on failure."""
    try:
        st = gemmi.read_structure(str(cif_path))
        lig_lines = []
        atom_count = 0
        for model in st:
            for chain in model:
                if chain.name != "B":
                    continue
                for res in chain:
                    for atom in res:
                        atom_count += 1
                        x, y, z = atom.pos.x, atom.pos.y, atom.pos.z
                        name = atom.name.ljust(4)
                        elem = atom.element.name.rjust(2)
                        line = (
                            f"HETATM{atom_count:5d} {name} LIG B   1    "
                            f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {elem}\n"
                        )
                        lig_lines.append(line)
            break
        if not lig_lines:
            return None
        block = "".join(lig_lines) + "END\n"
        mol = Chem.MolFromPDBBlock(block, removeHs=False, sanitize=False)
        if mol is None:
            return None
        try:
            Chem.SanitizeMol(mol)
        except Exception:
            try:
                Chem.SanitizeMol(
                    mol,
                    sanitizeOps=Chem.SanitizeFlags.SANITIZE_ALL
                    ^ Chem.SanitizeFlags.SANITIZE_KEKULIZE
                    ^ Chem.SanitizeFlags.SANITIZE_SETAROMATICITY,
                )
            except Exception:
                return None
        return mol
    except Exception:
        return None


def rdkit_strain_kcal_mol(cif_path: Path) -> float | None:
    """MMFF94 strain on un-relaxed pose: E(pose) - E(MMFF-minimized).

    Per existing rdkit_strain_posefree_kcal_mol semantics in the schema.
    """
    mol = extract_ligand_mol_from_cif(cif_path)
    if mol is None:
        return None
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        props = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94")
        if props is None:
            return None
        ff_pose = AllChem.MMFFGetMoleculeForceField(mol_h, props)
        if ff_pose is None:
            return None
        e_pose = ff_pose.CalcEnergy()
        # Relax (limited iter) and compute relaxed energy
        mol_min = Chem.Mol(mol_h)
        props_min = AllChem.MMFFGetMoleculeProperties(mol_min, mmffVariant="MMFF94")
        ff_min = AllChem.MMFFGetMoleculeForceField(mol_min, props_min)
        if ff_min is None:
            return None
        ff_min.Minimize(maxIts=500)
        e_min = ff_min.CalcEnergy()
        return float(e_pose - e_min)
    except Exception:
        return None


def shape_tc_seed(cif_path: Path, ref_mol: Chem.Mol) -> float | None:
    """3D shape Tanimoto between cofolded ligand and the reference Mol1 conformer."""
    mol = extract_ligand_mol_from_cif(cif_path)
    if mol is None:
        return None
    try:
        # Use RDKit ShapeTanimotoDist (lower = more similar; Tc = 1 - dist)
        from rdkit.Chem import rdShapeHelpers
        # Both mols need 3D coords
        if ref_mol.GetNumConformers() == 0 or mol.GetNumConformers() == 0:
            return None
        # Align with O3A as a best-effort common substructure alignment
        try:
            from rdkit.Chem import rdMolAlign
            o3a = rdMolAlign.GetO3A(mol, ref_mol)
            if o3a is not None:
                o3a.Align()
        except Exception:
            pass
        dist = rdShapeHelpers.ShapeTanimotoDist(mol, ref_mol)
        return float(1.0 - dist)
    except Exception:
        return None


def esp_sim_seed(cif_path: Path, ref_mol: Chem.Mol) -> float | None:
    """ESP-Sim between cofold ligand and reference Mol1."""
    mol = extract_ligand_mol_from_cif(cif_path)
    if mol is None:
        return None
    try:
        from espsim import GetEspSim
        # Need 3D coords + Hs
        mol_h = Chem.AddHs(mol, addCoords=True)
        ref_h = Chem.AddHs(ref_mol, addCoords=True)
        return float(GetEspSim(mol_h, ref_h))
    except Exception:
        return None


# ----------- PROPKA pKa -------------------------------------------------------

def pKa_cys346(cif_path: Path) -> float | None:
    """Run PROPKA3 on the cofolded protein (chain A) and parse the Cys346 pKa."""
    with tempfile.TemporaryDirectory(prefix="propka_") as tmpd:
        tmp = Path(tmpd)
        prot_pdb = tmp / "prot.pdb"
        try:
            st = gemmi.read_structure(str(cif_path))
            out_st = gemmi.Structure()
            out_st.cell = st.cell
            out_st.spacegroup_hm = st.spacegroup_hm
            new_model = gemmi.Model("1")
            for model in st:
                for chain in model:
                    if chain.name == "A":
                        new_model.add_chain(chain.clone())
                break
            out_st.add_model(new_model)
            out_st.write_pdb(str(prot_pdb))
        except Exception:
            return None

        try:
            # propka3 writes <stem>.pka in CWD by default
            r = subprocess.run(
                [PROPKA_CMD, str(prot_pdb.name)],
                capture_output=True, text=True, timeout=180, cwd=str(tmp),
            )
            pka_file = tmp / (prot_pdb.stem + ".pka")
            if not pka_file.exists():
                return None
            for line in pka_file.read_text().splitlines():
                # PROPKA output format:  "   CYS 346 A    9.49"
                m = re.match(r"\s*CYS\s+346\s+\S+\s+([-+]?[0-9]*\.?[0-9]+)", line)
                if m:
                    return float(m.group(1))
        except Exception:
            return None
    return None


# ----------- Per-row dispatch -------------------------------------------------

def _ref_mol_3d() -> Chem.Mol | None:
    """Build Mol1 3D conformer once for shape Tc + ESP."""
    m = Chem.MolFromSmiles(MOL1_SMILES)
    if m is None:
        return None
    m = Chem.AddHs(m)
    try:
        AllChem.EmbedMolecule(m, randomSeed=42)
        AllChem.MMFFOptimizeMolecule(m, maxIters=200)
    except Exception:
        return None
    return m


_REF_MOL = None


def get_ref_mol() -> Chem.Mol | None:
    global _REF_MOL
    if _REF_MOL is None:
        _REF_MOL = _ref_mol_3d()
    return _REF_MOL


def compute_row(row_id: str, pred_dir: Path, smiles: str | None,
                enable_vina: bool = True, enable_propka: bool = True,
                enable_shape: bool = True) -> dict:
    out = {
        "row_id": row_id,
        "boltz_iptm": None,
        "boltz_ligand_iptm": None,
        "complex_pde": None,
        "mPAE_paper": None,     # PROXY: scalar from confidence.json.complex_pde (legacy, comparable to prior cohorts)
        "mPAE_london": None,    # TRUE: min over protein×ligand block of raw PAE matrix (London JACS 2026)
        "d_SG": None,
        "burgi_dunitz_dev_deg": None,
        "geom_ok": None,
        "n_h_bonds": None,
        "n_stabilizing_contacts": None,
        "pocket_occupancy_pct": None,
        "vina_rescore_affinity_kcalmol": None,
        "vina_rescore_intra_kcalmol": None,
        "rdkit_strain_kcal_mol": None,
        "pKa_Cys346": None,
        "shape_Tc_seed": None,
        "esp_sim_seed": None,
        "boltz_compute_error": None,
    }
    # Find cofold name from pred_dir
    cofold_name = pred_dir.name
    cif = pred_dir / f"{cofold_name}_model_0.cif"
    pae = pred_dir / f"pae_{cofold_name}_model_0.npz"
    conf = pred_dir / f"confidence_{cofold_name}_model_0.json"
    skip_marker = pred_dir / "_SKIP_DUP_MARKER"

    if skip_marker.exists():
        out["boltz_compute_error"] = "skip_dup"
        return out
    if not cif.exists():
        out["boltz_compute_error"] = "no_cif"
        return out

    try:
        # 1. Confidence JSON
        out.update(load_confidence(conf))

        # 2. Parse CIF
        prot, lig = parse_cofold_cif(cif)
        n_lig_atoms = len(lig)
        if n_lig_atoms == 0:
            out["boltz_compute_error"] = "no_ligand_atoms"
            return out

        # 3. mPAE — TWO columns (user request 2026-06-10):
        # - mPAE_paper: PROXY from confidence.json.complex_pde (legacy, COMPARABLE with prior cohorts)
        # - mPAE_london: TRUE min over protein×ligand block of raw PAE matrix (London JACS 2026)
        out["mPAE_paper"] = out.get("complex_pde")          # proxy = the scalar from confidence.json
        out["mPAE_london"] = mpae_paper(pae, n_lig_atoms)   # London-formula min-PAE on the raw matrix

        # 4. Geometry (d_SG, Bürgi-Dunitz, geom_ok)
        warhead_atom = derive_warhead_atom_name(smiles) if smiles else None
        d_sg = d_SG_from_cofold(prot, lig, warhead_atom)
        bd = burgi_dunitz_dev(prot, lig, warhead_atom)
        out["d_SG"] = d_sg
        out["burgi_dunitz_dev_deg"] = bd
        out["geom_ok"] = bool(d_sg is not None and d_sg < GEOM_OK_DSG and warhead_atom is not None)

        # 5. Contacts / pocket
        out["n_h_bonds"] = count_h_bonds(prot, lig)
        out["n_stabilizing_contacts"] = count_stabilizing_contacts(prot, lig)
        out["pocket_occupancy_pct"] = pocket_occupancy_pct(prot, lig)

        # 6. RDKit strain
        out["rdkit_strain_kcal_mol"] = rdkit_strain_kcal_mol(cif)

        # 7. Shape Tc + ESP-Sim
        if enable_shape:
            ref = get_ref_mol()
            if ref is not None:
                out["shape_Tc_seed"] = shape_tc_seed(cif, ref)
                out["esp_sim_seed"] = esp_sim_seed(cif, ref)

        # 8. PROPKA Cys346 pKa
        if enable_propka:
            out["pKa_Cys346"] = pKa_cys346(cif)

        # 9. Vina rescore (slowest)
        if enable_vina:
            out.update(vina_rescore(cif))

    except Exception as e:
        out["boltz_compute_error"] = f"{type(e).__name__}: {str(e)[:200]}"
    return out


# ----------- Main driver ------------------------------------------------------

def process_batch(args_tuple):
    row_id, pred_dir_str, smiles, enable_vina, enable_propka, enable_shape = args_tuple
    return compute_row(row_id, Path(pred_dir_str), smiles, enable_vina, enable_propka, enable_shape)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", choices=["pool", "extra", "both"], default="both")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=200,
                    help="Write intermediate CSV every N rows")
    ap.add_argument("--no-vina", action="store_true")
    ap.add_argument("--no-propka", action="store_true")
    ap.add_argument("--no-shape", action="store_true")
    ap.add_argument("--resume", action="store_true",
                    help="Skip rows already present in output CSV with non-null boltz_iptm")
    args = ap.parse_args()

    print(f"=== Building row_id → cofold dir index ===", flush=True)
    index = build_row_id_index()
    print(f"  {len(index)} cofold dirs indexed", flush=True)

    targets = []
    if args.cohort in ("pool", "both"):
        targets.append(("pool", POOL_IN, POOL_OUT))
    if args.cohort in ("extra", "both"):
        targets.append(("extra", EXTRA_IN, EXTRA_OUT))

    enable_vina = not args.no_vina
    enable_propka = not args.no_propka
    enable_shape = not args.no_shape

    for cohort, in_csv, out_csv in targets:
        print(f"\n=== Processing {cohort} ===", flush=True)
        df_in = pd.read_csv(in_csv)
        print(f"  Loaded {len(df_in)} rows from {in_csv.name}", flush=True)

        # Resume: load existing output if any
        # Idempotent rule: a row is "done" iff boltz_iptm OR boltz_compute_error is set
        # AND mPAE_london is populated (mPAE_london is the NEW column added 2026-06-10;
        # rows scored before the script update have boltz_iptm but no mPAE_london and
        # need to be recomputed).
        existing_results: dict[str, dict] = {}
        if args.resume and out_csv.exists():
            try:
                df_prev = pd.read_csv(out_csv)
                if "boltz_iptm" in df_prev.columns:
                    has_iptm_or_err = (df_prev["boltz_iptm"].notna()
                                       | df_prev["boltz_compute_error"].notna())
                    if "mPAE_london" in df_prev.columns:
                        has_london = df_prev["mPAE_london"].notna()
                    else:
                        has_london = pd.Series(False, index=df_prev.index)
                    done_mask = has_iptm_or_err & has_london
                    done = df_prev[done_mask]
                    needs_recompute = df_prev[has_iptm_or_err & ~has_london]
                    for _, r in done.iterrows():
                        existing_results[r["row_id"]] = r.to_dict()
                    print(f"  Resume: {len(existing_results)} rows already fully computed "
                          f"(incl. mPAE_london); {len(needs_recompute)} rows need recompute "
                          f"for new mPAE_london column", flush=True)
            except Exception as e:
                print(f"  resume read failed: {e}", flush=True)

        # Build the list of rows to compute
        to_do = []
        for _, r in df_in.iterrows():
            row_id = r["row_id"]
            if row_id in existing_results:
                continue
            if row_id not in index:
                # No cofold pulled yet — skip but mark in output later
                continue
            pred_dir = index[row_id]
            cif = pred_dir / f"{pred_dir.name}_model_0.cif"
            if not cif.exists():
                continue
            smi = r.get("smiles") or r.get("seed_smi")
            to_do.append((row_id, str(pred_dir), smi, enable_vina, enable_propka, enable_shape))

        if args.limit:
            to_do = to_do[:args.limit]
        print(f"  {len(to_do)} rows to compute (cofolds available)", flush=True)

        # Worker pool
        results: list[dict] = list(existing_results.values())
        n_done = 0
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(process_batch, t): t[0] for t in to_do}
            for fut in as_completed(futures):
                row_id = futures[fut]
                try:
                    res = fut.result()
                except Exception as e:
                    res = {"row_id": row_id, "boltz_compute_error": f"future_error: {e}"}
                results.append(res)
                n_done += 1
                if n_done % 25 == 0:
                    print(f"  [{n_done}/{len(to_do)}] {row_id}  "
                          f"iptm={res.get('boltz_iptm')}  "
                          f"d_SG={res.get('d_SG')}  "
                          f"mPAE={res.get('mPAE_paper')}", flush=True)
                # Save partial every BATCH_SIZE rows
                if n_done % args.batch_size == 0:
                    save_merged(df_in, results, out_csv)
                    print(f"  → saved partial ({n_done} done) to {out_csv.name}", flush=True)

        save_merged(df_in, results, out_csv)
        print(f"  ✓ wrote {len(results)} results merged into {out_csv}", flush=True)


BOLTZ_COMPUTED_COLS = [
    "boltz_iptm", "boltz_ligand_iptm", "complex_pde",
    "mPAE_paper", "mPAE_london", "d_SG", "burgi_dunitz_dev_deg", "geom_ok",
    "n_h_bonds", "n_stabilizing_contacts", "pocket_occupancy_pct",
    "vina_rescore_affinity_kcalmol", "vina_rescore_intra_kcalmol",
    "rdkit_strain_kcal_mol", "pKa_Cys346",
    "shape_Tc_seed", "esp_sim_seed", "boltz_compute_error",
]


def save_merged(df_in: pd.DataFrame, results: list[dict], out_csv: Path):
    """Merge results onto df_in (left join on row_id) and write.

    If df_in already has columns clashing with BOLTZ_COMPUTED_COLS (e.g. empty
    placeholder cols), drop them so the computed values land in the output.
    """
    if not results:
        out_csv.parent.mkdir(parents=True, exist_ok=True)
        df_in.to_csv(out_csv, index=False)
        return
    df_res = pd.DataFrame(results)
    # Drop any duplicate row_ids in results, keep last
    df_res = df_res.drop_duplicates(subset=["row_id"], keep="last")
    # Drop any clashing pre-existing boltz cols from df_in (they're typically all-NaN placeholders)
    drop_cols = [c for c in BOLTZ_COMPUTED_COLS if c in df_in.columns]
    df_in_clean = df_in.drop(columns=drop_cols) if drop_cols else df_in
    # Only keep row_id + new boltz columns from df_res
    keep_cols = ["row_id"] + [c for c in df_res.columns
                              if c in BOLTZ_COMPUTED_COLS and c != "row_id"]
    df_res = df_res[[c for c in keep_cols if c in df_res.columns]]
    df_out = df_in_clean.merge(df_res, on="row_id", how="left")
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df_out.to_csv(out_csv, index=False)


if __name__ == "__main__":
    main()
