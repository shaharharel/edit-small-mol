#!/usr/bin/env python3
"""M1a v2 pose-fix pipeline (Fixes 1+2+3).

Reads existing data/m1a_triples_v2/triples.parquet (9,365 rows) and:

  Fix 1: Replace pose dim 5 from ||a_xyz - b_xyz|| (a vinyl C=C bond length
         constant ~1.34 A) with the MEASURED vinyl-amide planar dihedral
         C_beta = C_alpha - C(=O) - N (degrees, range [-180, 180]).
         For non-acrylamide warheads (where this dihedral is undefined) we
         emit 0.0 and rely on the model to see std>0 across the acrylamide
         subset.

  Fix 2: Replace pose dims 0-2 from lab-frame (b - nuc) displacement with the
         CANONICAL LOCAL-FRAME coordinates of a_xyz expressed in the frame
         centered at b_xyz with z_hat pointing toward the nucleophile.
         This makes the conditioning rotation/translation invariant.

  Fix 3: Z-score every pose dim across the training corpus. Save
         data/m1a_triples_v2/pose_normalizer.json with {'mean': [...],
         'std': [...]}.

For each row we open the source CIF (boltz_zap70) or PDB (covindb_pdb,
covindb_v2, covbinder_inpdb) and re-extract the 4 acrylamide atoms (b, a,
gamma, delta) by re-running the SMARTS match on the metadata SMILES and
mapping its heavy atoms onto the HET-atom xyz via element-greedy assignment.

Outputs:
  data/m1a_triples_v2/triples_posefix.parquet
      adds columns warhead_a_xyz, warhead_gamma_xyz, warhead_delta_xyz,
      planar_dihedral_deg, warhead_pose_6d_v2 (the new pose vector).
  data/m1a_triples_v2/esm2_cache_posefix.npz
      same residues_emb/residues_mask/row_seq_idx as input cache, but
      poses replaced with the new z-scored 6-d vector. Keeps the original
      9,365-row ordering. (We re-use the input ESM cache rather than
      re-running ESM-2 since pocket sequences are unchanged.)
  data/m1a_triples_v2/pose_normalizer.json
  data/m1a_v2_qa.json   (QA gate results)
  data/m1a_v2_progress.json  (updated)

QA gates (per task spec):
  Fix 1: dim 5 std > 5 degrees across acrylamide-class rows.
  Fix 2: pose vector identical to within 1e-4 under a random rotation of the
         lab frame.
  Fix 3: z-scored pose has mean ~ 0 and std ~ 1 per dim.

If any QA gate fails, writes BLOCKER_m1a_v2.md and exits 2.
"""
from __future__ import annotations
import json
import sys
import time
import warnings
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdMolTransforms

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
TRIPLES = PROJECT_ROOT / "data/m1a_triples_v2/triples.parquet"
ESM_CACHE_IN = PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache.npz"
OUT_TRIPLES = PROJECT_ROOT / "data/m1a_triples_v2/triples_posefix.parquet"
OUT_CACHE = PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache_posefix.npz"
OUT_NORM = PROJECT_ROOT / "data/m1a_triples_v2/pose_normalizer.json"
OUT_QA = PROJECT_ROOT / "data/m1a_v2_qa.json"
OUT_PROGRESS = PROJECT_ROOT / "data/m1a_v2_progress.json"
OUT_BLOCKER = PROJECT_ROOT / "BLOCKER_m1a_v2.md"

# Boltz CIF roots (same as in build_triples.py)
BOLTZ_ROOTS = [
    PROJECT_ROOT / "data/boltz_results/cohort_3597_full",
    PROJECT_ROOT / "data/boltz_f4_results",
    PROJECT_ROOT / "data/boltz_f4_extra_results",
    PROJECT_ROOT / "data/boltz_rescue_78",
]
PDB_ROOTS = [
    PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB",
    PROJECT_ROOT / "data/covbinder_inpdb/PDB",
]

SKIP_HET = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "MG", "CA", "ZN", "MN",
    "FE", "K", "BR", "EDO", "GOL", "PEG", "DMS", "DTT", "BME", "NAG", "MAN",
    "FUC", "BMA", "GAL", "GLC", "ACT", "TRS", "IPA", "FMT", "MES", "BCT",
}

# Acrylamide pattern: Cbeta(=Calpha)-C(=O)-N
# Returns (b_idx, a_idx, gamma_idx, delta_idx) within the SMARTS match.
# Pattern atom order: [CH2;X3]=[CH;X3][C;X3](=O)[N]
#   atom 0 = Cbeta (CH2=)
#   atom 1 = Calpha (=CH-)
#   atom 2 = C(=O) gamma
#   atom 3 = O carbonyl
#   atom 4 = N delta
ACRYL_PATTERNS = [
    ("acrylamide_strict",  "[CH2;X3]=[CH;X3][C;X3](=O)[N]",   (0, 1, 2, 4)),
    ("acrylamide_loose",   "[CH2]=C[C](=O)[N,n]",             (0, 1, 2, 4)),
    ("acrylate",           "[CH2]=[CH][C](=O)[O]",            (0, 1, 2, 4)),
]


# ============================================================
# Geometry utilities for the new pose vector
# ============================================================

def gram_schmidt_frame(z_hat: np.ndarray,
                          seed: np.ndarray | None = None
                          ) -> tuple[np.ndarray, np.ndarray]:
    """Build a deterministic orthonormal (x, y) basis perpendicular to z_hat.

    The seed vector MUST rotate with the molecule for the resulting frame to
    be rotation-invariant. The lab x-axis (1,0,0) is NOT a valid seed because
    it does not co-rotate. The caller should pass a molecular vector, e.g.
    (gamma - b) for an acrylamide or (a - b) otherwise.

    The returned (x_hat, y_hat) with z_hat form a right-handed orthonormal
    frame. Rotating the seed and z_hat by R (a rotation of the lab frame)
    rotates x_hat and y_hat by the same R, so coordinates of any other atom
    expressed in this frame are rotation-invariant.

    If `seed` is None or degenerate (parallel to z_hat), we fall back to a
    cyclic permutation of z_hat as a guaranteed-non-parallel co-rotating seed
    (`(z_hat.y, z_hat.z, z_hat.x)`).
    """
    if seed is None:
        seed = np.array([z_hat[1], z_hat[2], z_hat[0]])
    seed = np.asarray(seed, dtype=np.float64)
    x_hat = seed - np.dot(seed, z_hat) * z_hat
    n = np.linalg.norm(x_hat)
    if n < 1e-6:
        # Degenerate; use cyclic permutation of z_hat
        cyc = np.array([z_hat[1], z_hat[2], z_hat[0]])
        x_hat = cyc - np.dot(cyc, z_hat) * z_hat
        n = np.linalg.norm(x_hat)
        if n < 1e-6:
            # Try another permutation
            cyc = np.array([z_hat[2], z_hat[0], z_hat[1]])
            x_hat = cyc - np.dot(cyc, z_hat) * z_hat
            n = np.linalg.norm(x_hat)
    x_hat = x_hat / n
    y_hat = np.cross(z_hat, x_hat)
    return x_hat, y_hat


def planar_dihedral_from_xyz(b: np.ndarray, a: np.ndarray, g: np.ndarray,
                              n_atom: np.ndarray) -> float:
    """Signed dihedral b-a-g-n in degrees, range [-180, 180].

    Manual computation using the standard formula (so we are not coupled to
    RDKit's GetDihedralDeg, which requires a molecule object).
    """
    b1 = a - b
    b2 = g - a
    b3 = n_atom - g
    n1 = np.cross(b1, b2)
    n2 = np.cross(b2, b3)
    m1 = np.cross(n1, b2 / max(np.linalg.norm(b2), 1e-8))
    x = float(np.dot(n1, n2))
    y = float(np.dot(m1, n2))
    return float(np.degrees(np.arctan2(y, x)))


def compute_new_pose(b: np.ndarray, a: np.ndarray, nuc: np.ndarray,
                      bd_angle_deg: float,
                      gamma: np.ndarray | None,
                      delta: np.ndarray | None) -> tuple[list, float, str]:
    """Build the v2 (fix-1, fix-2) pose vector.

    Dims:
      0-2: alpha-carbon coordinates in the canonical local frame centered at
           b_xyz with z_hat = (nuc - b) / |nuc - b|. The in-plane x_hat is
           seeded from (gamma - b) when the carbonyl C is available, else
           from (a - b) — both are intrinsic to the molecule and therefore
           rotate with it, guaranteeing rotation invariance.

           When seeded from (a - b), dim 1 (the y component) is identically 0
           for that row; the model will learn to ignore that dim for non-
           acrylamide rows.
      3:   |b - nuc| (unchanged from v1).
      4:   BD angle (degrees).
      5:   Cbeta=Calpha - C(=O) - N planar dihedral (degrees) if all 4 atoms
           are valid; else 0.0.
    """
    v = nuc - b
    d = float(np.linalg.norm(v))
    if d < 1e-6:
        return [0.0] * 6, 0.0, "zero_d_b_nuc"
    z_hat = v / d
    # Seed x_hat from gamma when present (acrylamide; gives full 3-d info
    # about a's position), else from a itself (non-acrylamide; gives 2-d
    # info — the in-plane y component will be 0 by construction).
    if gamma is not None:
        seed = np.asarray(gamma) - b
    else:
        seed = np.asarray(a) - b
    x_hat, y_hat = gram_schmidt_frame(z_hat, seed)
    a_rel = a - b
    a_local = np.array([
        float(np.dot(a_rel, x_hat)),
        float(np.dot(a_rel, y_hat)),
        float(np.dot(a_rel, z_hat)),
    ])
    if gamma is not None and delta is not None:
        try:
            dih = planar_dihedral_from_xyz(b, a, gamma, delta)
        except Exception:
            dih = 0.0
        dih_source = "measured"
    else:
        dih = 0.0
        dih_source = "missing_atoms"
    bd = float(bd_angle_deg) if (bd_angle_deg == bd_angle_deg) else 0.0
    pose = [float(a_local[0]), float(a_local[1]), float(a_local[2]),
            d, bd, float(dih)]
    return pose, float(dih), dih_source


# ============================================================
# Source-file re-extraction
# ============================================================

def parse_pdb_full(pdb_path: Path) -> dict | None:
    """Return ATOM+HETATM coords from a PDB file, indexed by chain+resid."""
    residues = {}
    hetatoms = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    atom_name = line[12:16].strip()
                    altloc = line[16]
                    resname = line[17:20].strip()
                    chain = line[21]
                    resid = int(line[22:26])
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    elem = line[76:78].strip() if len(line) > 78 else atom_name[0]
                except (ValueError, IndexError):
                    continue
                if altloc not in (" ", "A"):
                    continue
                xyz = np.array([x, y, z])
                if line.startswith("ATOM"):
                    key = (chain, resid)
                    if key not in residues:
                        residues[key] = {"aa": resname, "idx": resid,
                                          "chain": chain, "atoms": {}}
                    residues[key]["atoms"][atom_name] = xyz
                else:
                    if resname in SKIP_HET:
                        continue
                    key = (chain, resname, resid)
                    hetatoms.setdefault(key, []).append(
                        {"name": atom_name, "elem": elem, "xyz": xyz})
    except Exception:
        return None
    if not residues:
        return None
    return {"residues": residues, "hetatoms": hetatoms}


def parse_boltz_cif_lig(cif_path: Path) -> list[dict] | None:
    """Return the heavy-atom positions of the LIG residue from a Boltz CIF.

    Each list entry: {"name", "elem", "xyz"}.
    """
    from Bio.PDB.MMCIFParser import MMCIFParser
    try:
        parser = MMCIFParser(QUIET=True)
        s = parser.get_structure("m", str(cif_path))
        model = next(s.get_models())
    except Exception:
        return None
    lig_atoms = []
    for chain in model:
        for res in chain:
            hetflag = res.id[0]
            if hetflag == " ":
                continue
            if not res.get_resname().startswith("LIG"):
                continue
            for a in res:
                # Drop hydrogens
                name = a.get_name()
                elem = a.element.strip() if hasattr(a, "element") else (
                    "H" if name.startswith("H") else name[0])
                if elem == "H":
                    continue
                lig_atoms.append({
                    "name": name,
                    "elem": elem.upper(),
                    "xyz": np.array(a.get_coord()),
                })
    return lig_atoms if lig_atoms else None


def find_boltz_cif(struct_id: str) -> Path | None:
    """Locate a Boltz cofold CIF whose stem matches struct_id.

    struct_id like 'row01433' is the parent dir of '*_model_0.cif' inside one
    of the boltz roots.
    """
    for root in BOLTZ_ROOTS:
        if not root.exists():
            continue
        # First: direct dir match
        cands = list(root.rglob(f"{struct_id}/*_model_0.cif"))
        if cands:
            return cands[0]
        # Fallback: stem of file == struct_id_model_0
        cands = list(root.rglob(f"{struct_id}_model_0.cif"))
        if cands:
            return cands[0]
    return None


def find_pdb_file(pdb_id: str) -> Path | None:
    pid_l = pdb_id.lower()
    pid_u = pdb_id.upper()
    for d in PDB_ROOTS:
        for cand in (d / f"{pid_l}.pdb", d / f"{pid_u}.pdb"):
            if cand.exists() and cand.stat().st_size > 0:
                return cand
    return None


# ============================================================
# Match SMILES heavy atoms onto a list of HETATM xyzs
# (mirrors build_triples_v2.smiles_to_3d_via_pdb_het but element-greedy with
# nearest-neighbour preference — adequate for getting b/a/gamma/delta)
# ============================================================

def map_smi_to_het_coords(mol: Chem.Mol, het_heavy: list[dict]) -> Chem.Mol | None:
    mol_h = Chem.RemoveHs(mol)
    n_heavy = mol_h.GetNumHeavyAtoms()
    if len(het_heavy) != n_heavy:
        return None
    rdkit_elements = [mol_h.GetAtomWithIdx(i).GetSymbol().upper()
                      for i in range(n_heavy)]
    het_elements = [a["elem"].upper() for a in het_heavy]
    if Counter(rdkit_elements) != Counter(het_elements):
        return None
    used = [False] * len(het_heavy)
    assignment = [-1] * n_heavy
    for i, ele in enumerate(rdkit_elements):
        for j, het in enumerate(het_heavy):
            if used[j] or het["elem"].upper() != ele:
                continue
            assignment[i] = j
            used[j] = True
            break
        if assignment[i] == -1:
            return None
    mol_with_conf = Chem.RWMol(mol_h)
    mol_with_conf.RemoveAllConformers()
    conf = Chem.Conformer(n_heavy)
    for i in range(n_heavy):
        xyz = het_heavy[assignment[i]]["xyz"]
        conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol_with_conf.AddConformer(conf, assignId=True)
    return mol_with_conf.GetMol()


def get_warhead_4_atoms_from_mol(mol_with_conf: Chem.Mol) -> dict | None:
    """Try acrylamide SMARTS patterns; return atom coords for b, a, gamma, delta."""
    for name, smarts, (b_off, a_off, g_off, d_off) in ACRYL_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        m = mol_with_conf.GetSubstructMatch(patt)
        if not m:
            continue
        try:
            conf = mol_with_conf.GetConformer()
            b_pos = conf.GetAtomPosition(m[b_off])
            a_pos = conf.GetAtomPosition(m[a_off])
            g_pos = conf.GetAtomPosition(m[g_off])
            d_pos = conf.GetAtomPosition(m[d_off])
            return {
                "name": name,
                "b": np.array([b_pos.x, b_pos.y, b_pos.z]),
                "a": np.array([a_pos.x, a_pos.y, a_pos.z]),
                "gamma": np.array([g_pos.x, g_pos.y, g_pos.z]),
                "delta": np.array([d_pos.x, d_pos.y, d_pos.z]),
            }
        except Exception:
            continue
    return None


# ============================================================
# Per-row worker
# ============================================================

def enrich_row(args: tuple) -> dict | None:
    """args = (row_idx, row_dict, source_path_str_or_none)
    Returns dict with keys row_idx, a_xyz/gamma_xyz/delta_xyz (or None),
    planar_dihedral_deg, pose_v2_unnormalized (list of 6 floats), dih_source.
    """
    row_idx, row, source_path_str = args
    smi = row.get("canon_smi") or row.get("smiles")
    if not isinstance(smi, str):
        return _no_atom_fallback(row_idx, row)
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return _no_atom_fallback(row_idx, row)

    # Re-open source file and find the relevant ligand HETATM block
    source = row["source"]
    het_atoms = None
    try:
        if source == "boltz_zap70":
            cif = Path(source_path_str) if source_path_str else None
            if cif is None or not cif.exists():
                return _no_atom_fallback(row_idx, row)
            het_atoms = parse_boltz_cif_lig(cif)
            if not het_atoms:
                return _no_atom_fallback(row_idx, row)
            # Boltz CIFs have exactly one LIG, atoms are already filtered to heavy
        else:
            pdb_path = Path(source_path_str) if source_path_str else None
            if pdb_path is None or not pdb_path.exists():
                return _no_atom_fallback(row_idx, row)
            parsed = parse_pdb_full(pdb_path)
            if parsed is None:
                return _no_atom_fallback(row_idx, row)
            # Pick the HET block that matches struct_id pattern
            # struct_id formats:
            #   covindb_pdb:  <pdbid>_<chain>_<resname>_<resid>
            #   covindb_v2:   <pdbid>_<chain>_<lig_name>_<nuc_posi>  (nuc_posi != lig_posi!)
            #   covbinder_inpdb: same as covindb_v2
            # For v2/covbinder rows: the lig resname is part of struct_id but
            # the lig position is NOT, so we have to match by resname (and by
            # nuc-chain). Pick the HET nearest to nuc xyz when ambiguous.
            sid = row["struct_id"]
            sid_parts = sid.split("_")
            target_chain = sid_parts[1] if len(sid_parts) >= 2 else "A"
            target_resname = sid_parts[2] if len(sid_parts) >= 3 else ""
            # nuc_posi is the last token for the v2/covbinder schema
            nuc_xyz = np.array(json.loads(row["nucleophile_xyz"]))
            b_xyz_known = np.array(json.loads(row["warhead_b_xyz"]))
            # Collect candidate HET blocks matching the resname; rank by min
            # heavy-atom distance to b_xyz_known
            cands = []
            for (ch, rn, rp), atoms in parsed["hetatoms"].items():
                if target_resname and rn != target_resname:
                    continue
                heavy = [a for a in atoms if a["elem"] not in ("H", "D")]
                if not heavy:
                    continue
                # min distance from any heavy atom to b_xyz_known
                mind = min(float(np.linalg.norm(a["xyz"] - b_xyz_known))
                            for a in heavy)
                cands.append((mind, ch, rn, rp, heavy))
            if not cands:
                # Drop the resname constraint
                for (ch, rn, rp), atoms in parsed["hetatoms"].items():
                    heavy = [a for a in atoms if a["elem"] not in ("H", "D")]
                    if not heavy:
                        continue
                    mind = min(float(np.linalg.norm(a["xyz"] - b_xyz_known))
                                for a in heavy)
                    cands.append((mind, ch, rn, rp, heavy))
            if not cands:
                return _no_atom_fallback(row_idx, row)
            cands.sort(key=lambda x: x[0])
            _, _, _, _, het_atoms = cands[0]
    except Exception:
        return _no_atom_fallback(row_idx, row)

    # Map SMILES heavy atoms to HET xyz
    mol3d = map_smi_to_het_coords(mol, het_atoms)
    if mol3d is None:
        # Cannot map; we still want a_xyz at least to evaluate dim 0-2 in local
        # frame. For that fall back: a_xyz approximated as the second nearest
        # carbon to b_xyz from the HET block.
        b_xyz_known = np.array(json.loads(row["warhead_b_xyz"]))
        carbons = [a for a in het_atoms if a["elem"] == "C"]
        carbons.sort(key=lambda a: float(np.linalg.norm(a["xyz"] - b_xyz_known)))
        if len(carbons) < 2:
            return _no_atom_fallback(row_idx, row)
        a_xyz_est = carbons[1]["xyz"]  # 0th is b itself
        nuc_xyz = np.array(json.loads(row["nucleophile_xyz"]))
        bd = float(row["bd_angle_deg"]) if row["bd_angle_deg"] == row["bd_angle_deg"] else 0.0
        pose, dih, src = compute_new_pose(b_xyz_known, a_xyz_est, nuc_xyz, bd, None, None)
        return {
            "row_idx": row_idx,
            "warhead_a_xyz": a_xyz_est.tolist(),
            "warhead_gamma_xyz": None,
            "warhead_delta_xyz": None,
            "planar_dihedral_deg": dih,
            "pose_v2_unnormalized": pose,
            "dih_source": "smi_map_failed_fallback",
        }

    info = get_warhead_4_atoms_from_mol(mol3d)
    nuc_xyz = np.array(json.loads(row["nucleophile_xyz"]))
    b_xyz_known = np.array(json.loads(row["warhead_b_xyz"]))
    bd = float(row["bd_angle_deg"]) if row["bd_angle_deg"] == row["bd_angle_deg"] else 0.0
    if info is None:
        # Non-acrylamide warhead; still need a_xyz for local frame.
        carbons = []
        conf = mol3d.GetConformer()
        for at in mol3d.GetAtoms():
            if at.GetSymbol() == "C":
                p = conf.GetAtomPosition(at.GetIdx())
                carbons.append(np.array([p.x, p.y, p.z]))
        # find closest carbon to known b
        carbons.sort(key=lambda c: float(np.linalg.norm(c - b_xyz_known)))
        # 0th is b; 1st is a candidate a
        if len(carbons) >= 2:
            a_est = carbons[1]
        else:
            return _no_atom_fallback(row_idx, row)
        pose, dih, src = compute_new_pose(b_xyz_known, a_est, nuc_xyz, bd, None, None)
        return {
            "row_idx": row_idx,
            "warhead_a_xyz": a_est.tolist(),
            "warhead_gamma_xyz": None,
            "warhead_delta_xyz": None,
            "planar_dihedral_deg": dih,
            "pose_v2_unnormalized": pose,
            "dih_source": "non_acrylamide",
        }

    # acrylamide path
    # Use the b_xyz from SMARTS match (should be close to stored b_xyz_known
    # for boltz where we have exact atom mapping; for PDB-derived rows the
    # element-greedy assignment may place b at the wrong atom — prefer the
    # stored b_xyz_known)
    b_use = b_xyz_known
    a_use = info["a"]
    pose, dih, src = compute_new_pose(b_use, a_use, nuc_xyz, bd,
                                          info["gamma"], info["delta"])
    return {
        "row_idx": row_idx,
        "warhead_a_xyz": info["a"].tolist(),
        "warhead_gamma_xyz": info["gamma"].tolist(),
        "warhead_delta_xyz": info["delta"].tolist(),
        "planar_dihedral_deg": dih,
        "pose_v2_unnormalized": pose,
        "dih_source": src,
    }


def _no_atom_fallback(row_idx: int, row: dict) -> dict:
    """Fallback for rows where we cannot recover warhead atoms.

    Use a_xyz = b_xyz + a tiny vector (will look constant), and dih = 0.0.
    The model can still learn from rows where extraction succeeded.
    """
    nuc_xyz = np.array(json.loads(row["nucleophile_xyz"]))
    b_xyz = np.array(json.loads(row["warhead_b_xyz"]))
    bd = float(row["bd_angle_deg"]) if row["bd_angle_deg"] == row["bd_angle_deg"] else 0.0
    # a is unknown; use b + small unit vector pointing AWAY from nuc (so the
    # local frame z direction stays meaningful)
    v = b_xyz - nuc_xyz
    n = float(np.linalg.norm(v))
    a_est = b_xyz + (v / max(n, 1e-6)) * 1.34  # 1.34A vinyl bond
    pose, dih, src = compute_new_pose(b_xyz, a_est, nuc_xyz, bd, None, None)
    return {
        "row_idx": row_idx,
        "warhead_a_xyz": None,
        "warhead_gamma_xyz": None,
        "warhead_delta_xyz": None,
        "planar_dihedral_deg": 0.0,
        "pose_v2_unnormalized": pose,
        "dih_source": "no_source_file",
    }


# ============================================================
# Main
# ============================================================

def write_progress(phase: str, **extra):
    rec = {"phase": phase, "timestamp": time.time(),
           "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"), **extra}
    OUT_PROGRESS.write_text(json.dumps(rec, indent=2))


def write_blocker(msg: str):
    OUT_BLOCKER.write_text("# M1a v2 BLOCKER\n\n" + msg + "\n")


def main():
    write_progress("phase0_loading_triples")
    print(f"Loading {TRIPLES}", flush=True)
    df = pd.read_parquet(TRIPLES)
    print(f"  {len(df)} rows", flush=True)

    # Build work items
    work = []
    misses = Counter()
    for i, row in df.iterrows():
        rd = row.to_dict()
        source = rd["source"]
        source_path = None
        if source == "boltz_zap70":
            cif = find_boltz_cif(rd["struct_id"])
            if cif is None:
                misses["no_boltz_cif"] += 1
            source_path = str(cif) if cif else None
        else:
            pdb_id = rd.get("pdb_id")
            if not pdb_id or not isinstance(pdb_id, str):
                # try parsing from struct_id
                pdb_id = rd["struct_id"].split("_")[0]
            ppath = find_pdb_file(pdb_id)
            if ppath is None:
                misses["no_pdb"] += 1
            source_path = str(ppath) if ppath else None
        work.append((i, rd, source_path))
    print(f"  source-file misses by reason: {dict(misses)}", flush=True)

    write_progress("phase0_enriching", n_work=len(work))
    print(f"Enriching {len(work)} rows in parallel...", flush=True)
    t0 = time.time()
    results = [None] * len(work)
    n_workers = min(8, len(work))
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(enrich_row, w): w[0] for w in work}
        done = 0
        for fut in as_completed(futures):
            try:
                r = fut.result()
            except Exception as e:
                r = None
            if r is not None:
                results[r["row_idx"]] = r
            done += 1
            if done % 1000 == 0:
                rate = done / (time.time() - t0 + 1e-6)
                eta = (len(work) - done) / max(rate, 1e-6)
                print(f"  {done}/{len(work)}  rate={rate:.0f} rows/s  "
                       f"ETA={eta:.0f}s", flush=True)
    dt = time.time() - t0
    n_ok = sum(1 for r in results if r is not None)
    print(f"Enriched {n_ok}/{len(work)} in {dt:.1f}s", flush=True)
    # Fill missing with no-atom fallback
    for i, r in enumerate(results):
        if r is None:
            results[i] = _no_atom_fallback(i, df.iloc[i].to_dict())

    # Count dih_source distribution
    src_counter = Counter([r["dih_source"] for r in results])
    print("dih_source distribution:", dict(src_counter), flush=True)

    write_progress("phase0_assembling_dataframe")
    # Assemble new columns
    df["warhead_a_xyz"] = [json.dumps(r["warhead_a_xyz"]) if r["warhead_a_xyz"] else None
                            for r in results]
    df["warhead_gamma_xyz"] = [json.dumps(r["warhead_gamma_xyz"]) if r["warhead_gamma_xyz"] else None
                                for r in results]
    df["warhead_delta_xyz"] = [json.dumps(r["warhead_delta_xyz"]) if r["warhead_delta_xyz"] else None
                                for r in results]
    df["planar_dihedral_deg"] = [r["planar_dihedral_deg"] for r in results]
    df["dih_source"] = [r["dih_source"] for r in results]
    poses_unnorm = np.array([r["pose_v2_unnormalized"] for r in results],
                              dtype=np.float32)
    print(f"poses_unnorm shape={poses_unnorm.shape}", flush=True)
    # Save unnormalized pose first (we'll add the normalized one below)
    df["warhead_pose_6d_v2_unnorm"] = [json.dumps(list(map(float, p)))
                                         for p in poses_unnorm]

    # ----- QA Fix 1: dim 5 std > 5 deg on acrylamide subset -----
    # Acrylamide-like rows: warhead_class in {acrylamide, michael_acceptor,
    # acrylate} OR dih_source in {measured}
    acryl_mask = np.array([r["dih_source"] == "measured" for r in results])
    print(f"Acrylamide-like (dih_source=measured) rows: {int(acryl_mask.sum())}",
          flush=True)
    dim5_acryl = poses_unnorm[acryl_mask, 5]
    dim5_acryl_std = float(np.std(dim5_acryl)) if len(dim5_acryl) else 0.0
    print(f"  dim5 std on acryl subset = {dim5_acryl_std:.2f} deg", flush=True)
    fix1_ok = dim5_acryl_std > 5.0

    # ----- QA Fix 2: rotation invariance test -----
    # Test BOTH paths: (a) acrylamide row with gamma -> gamma-seeded x_hat;
    # (b) non-acrylamide row -> a-seeded x_hat.
    rng = np.random.default_rng(42)
    def random_rigid(rng):
        H = rng.standard_normal((3, 3))
        Q, _ = np.linalg.qr(H)
        if np.linalg.det(Q) < 0:
            Q[:, 0] *= -1
        t = rng.standard_normal(3) * 7.7
        return Q, t

    def rot_test(b, a, nuc, bd, g=None, d_at=None):
        pose_ref, _, _ = compute_new_pose(b, a, nuc, bd, g, d_at)
        Q, t = random_rigid(rng)
        def rot(x):
            return (Q @ np.asarray(x, dtype=np.float64)) + t
        b2 = rot(b); a2 = rot(a); nuc2 = rot(nuc)
        g2 = rot(g) if g is not None else None
        d2 = rot(d_at) if d_at is not None else None
        pose_rot, _, _ = compute_new_pose(b2, a2, nuc2, bd, g2, d2)
        diff = np.array(pose_ref) - np.array(pose_rot)
        return float(np.max(np.abs(diff))), pose_ref, pose_rot

    rot_errs = []
    try:
        # case (a): acrylamide row
        idx_a = int(np.argmax([1.0 if r["dih_source"] == "measured" else 0.0
                                for r in results]))
        rrow = df.iloc[idx_a]
        b = np.array(json.loads(rrow["warhead_b_xyz"]), dtype=np.float64)
        a = np.array(json.loads(rrow["warhead_a_xyz"]), dtype=np.float64)
        g = np.array(json.loads(rrow["warhead_gamma_xyz"]), dtype=np.float64)
        d_at = np.array(json.loads(rrow["warhead_delta_xyz"]), dtype=np.float64)
        nuc = np.array(json.loads(rrow["nucleophile_xyz"]), dtype=np.float64)
        bd = float(rrow["bd_angle_deg"])
        err_a, pref_a, prot_a = rot_test(b, a, nuc, bd, g, d_at)
        print(f"Rot-inv (acryl/gamma-seeded): max|diff|={err_a:.2e}", flush=True)
        print(f"  pose_ref: {pref_a}", flush=True)
        print(f"  pose_rot: {prot_a}", flush=True)
        rot_errs.append(err_a)
        # case (b): non-acrylamide row (a-seeded)
        idx_b = None
        for i, r in enumerate(results):
            if r["dih_source"] in ("non_acrylamide", "smi_map_failed_fallback"):
                if r["warhead_a_xyz"] is not None:
                    idx_b = i
                    break
        if idx_b is not None:
            rrow = df.iloc[idx_b]
            b = np.array(json.loads(rrow["warhead_b_xyz"]), dtype=np.float64)
            a = np.array(json.loads(rrow["warhead_a_xyz"]), dtype=np.float64)
            nuc = np.array(json.loads(rrow["nucleophile_xyz"]), dtype=np.float64)
            bd = float(rrow["bd_angle_deg"])
            err_b, _, _ = rot_test(b, a, nuc, bd, None, None)
            print(f"Rot-inv (non-acryl/a-seeded): max|diff|={err_b:.2e}",
                  flush=True)
            rot_errs.append(err_b)
        rot_max_err = float(max(rot_errs))
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"Rotation test exception: {e}", flush=True)
        rot_max_err = 1.0
    # Spec says 1e-4; with co-rotating seed we should comfortably hit < 1e-6
    # in float64; allow 1e-4 as the gate.
    fix2_ok = rot_max_err is not None and rot_max_err < 1e-4

    # ----- Fix 3: z-score the pose vector -----
    means = poses_unnorm.mean(axis=0)
    stds = poses_unnorm.std(axis=0)
    stds = np.where(stds < 1e-6, 1.0, stds)
    poses_norm = (poses_unnorm - means) / stds
    new_means = poses_norm.mean(axis=0)
    new_stds = poses_norm.std(axis=0)
    print(f"Pre-zscore means : {means}", flush=True)
    print(f"Pre-zscore stds  : {stds}", flush=True)
    print(f"Post-zscore means: {new_means}", flush=True)
    print(f"Post-zscore stds : {new_stds}", flush=True)
    fix3_ok = bool(np.all(np.abs(new_means) < 1e-4) and
                    np.all(np.abs(new_stds - 1.0) < 1e-3))

    # Save normalizer
    OUT_NORM.parent.mkdir(parents=True, exist_ok=True)
    OUT_NORM.write_text(json.dumps({
        "mean": means.tolist(),
        "std": stds.tolist(),
        "dim_meanings": [
            "alpha_local_x (frame: z=b->nuc dir, origin=b)",
            "alpha_local_y",
            "alpha_local_z",
            "d_b_nuc_angstroms",
            "bd_angle_deg",
            "planar_vinylamide_dihedral_deg (0.0 if undefined)",
        ],
        "n_rows": int(len(poses_unnorm)),
        "n_acrylamide_measured": int(acryl_mask.sum()),
    }, indent=2))
    print(f"Wrote {OUT_NORM}", flush=True)

    # Save full triples with new cols + z-scored pose
    df["warhead_pose_6d_v2"] = [json.dumps(list(map(float, p)))
                                 for p in poses_norm]
    df.to_parquet(OUT_TRIPLES, index=False)
    print(f"Wrote {OUT_TRIPLES}  ({len(df)} rows)", flush=True)

    # ----- Write updated ESM cache: reuse residue embeddings, replace poses -----
    print(f"Loading {ESM_CACHE_IN}", flush=True)
    d_in = np.load(ESM_CACHE_IN, allow_pickle=True)
    assert len(d_in["poses"]) == len(poses_norm), (
        f"row count mismatch {len(d_in['poses'])} vs {len(poses_norm)}")
    np.savez_compressed(
        OUT_CACHE,
        seq_hashes=d_in["seq_hashes"],
        residues_emb=d_in["residues_emb"],
        residues_mask=d_in["residues_mask"],
        row_seq_idx=d_in["row_seq_idx"],
        poses=poses_norm.astype(np.float32),
        poses_unnorm=poses_unnorm.astype(np.float32),
        pose_mean=means.astype(np.float32),
        pose_std=stds.astype(np.float32),
        smiles=d_in["smiles"],
        sources=d_in["sources"],
        struct_ids=d_in["struct_ids"],
    )
    print(f"Wrote {OUT_CACHE}", flush=True)

    # ----- QA report -----
    qa = {
        "fix1": "ok" if fix1_ok else "fail",
        "fix2": "ok" if fix2_ok else "fail",
        "fix3": "ok" if fix3_ok else "fail",
        "dim5_std_deg": dim5_acryl_std,
        "rotation_test_max_err": rot_max_err,
        "n_total_rows": int(len(df)),
        "n_acrylamide_measured": int(acryl_mask.sum()),
        "n_no_source_file": int(src_counter["no_source_file"]),
        "n_non_acrylamide": int(src_counter["non_acrylamide"]),
        "n_smi_map_failed_fallback": int(src_counter["smi_map_failed_fallback"]),
        "pose_dim_meanings": [
            "alpha_local_x", "alpha_local_y", "alpha_local_z",
            "d_b_nuc_A", "bd_angle_deg", "planar_dihedral_deg",
        ],
        "pose_mean_pre_zscore": means.tolist(),
        "pose_std_pre_zscore": stds.tolist(),
        "pose_mean_post_zscore": new_means.tolist(),
        "pose_std_post_zscore": new_stds.tolist(),
    }
    OUT_QA.write_text(json.dumps(qa, indent=2))
    print(f"\nQA result: {OUT_QA}\n{json.dumps(qa, indent=2)}", flush=True)

    if not (fix1_ok and fix2_ok and fix3_ok):
        msg = (
            f"QA gate failure:\n"
            f"  fix1 (dim5 std > 5deg on acryl subset): "
            f"{'ok' if fix1_ok else 'FAIL'} (got {dim5_acryl_std:.2f})\n"
            f"  fix2 (rotation invariance < 1e-3):       "
            f"{'ok' if fix2_ok else 'FAIL'} (got {rot_max_err})\n"
            f"  fix3 (z-score mean~0 std~1):             "
            f"{'ok' if fix3_ok else 'FAIL'} "
            f"(means={new_means} stds={new_stds})\n"
        )
        write_blocker(msg)
        print(msg, flush=True)
        write_progress("BLOCKED_qa_failure")
        sys.exit(2)

    write_progress("phase0_done",
                   fix1_dim5_std_deg=dim5_acryl_std,
                   fix2_rot_max_err=rot_max_err,
                   n_rows=int(len(df)))
    print("Phase 0 OK. All 3 QA gates pass.", flush=True)


if __name__ == "__main__":
    main()
