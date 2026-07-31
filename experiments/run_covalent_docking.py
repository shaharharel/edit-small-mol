#!/usr/bin/env python3
"""Covalent docking re-evaluation for the cohort-comparison cohorts.

Two methods, both on the same 2712 molecules from
results/paper_evaluation/cohort_comparison/all_cohorts_metrics.csv :

  1. AD-CovDock — meeko CovalentBuilder tethers the warhead Cβ to the
     Cys346 CB position (1.85 A from SG, BD ≈107°), then Vina docks
     against a receptor PDBQT with Cys346 CB+SG stripped (canonical
     AD-CovDock-Vina recipe). Pose's β-C should stay near SG.

  2. Restrained Vina — REUSE the non-covalent Vina poses already
     produced by the cohort-comparison eval pipeline (per_cohort/<C>/
     poses/<C>_<i>_pose.pdbqt). Re-measure d(Cβ-SG) and BD angle with
     the CORRECT atom-index rule (β-C = SMARTS match[0], terminal CH2
     of [CH2;X3]=[CH;X3][C;X3](=O)[N]) and filter on
     d ∈ [1.55, 2.15] Å AND BD ∈ [102°, 112°].

Output: results/paper_evaluation/cohort_comparison/covalent_docking_results.csv
        results/paper_evaluation/cohort_comparison/covalent_docking_per_cohort.csv
        results/paper_evaluation/cohort_comparison/covalent_docking_stats.csv
"""
from __future__ import annotations
import argparse
import json
import os
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog('rdApp.*')
warnings.filterwarnings('ignore')

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

# ---- Receptor + Cys346 anchor (4K2R) ----
RECEPTOR_PDB = PROJECT_ROOT / "data/docking_500/4K2R.pdb"
RECEPTOR_PDBQT = PROJECT_ROOT / "data/docking_500/receptor.pdbqt"
RECEPTOR_PDBQT_STRIPPED = PROJECT_ROOT / "data/docking_500/receptor_cys346_stripped.pdbqt"
ANCHOR_JSON = PROJECT_ROOT / "data/lingo3dmol_anchor_zap70_cys346.json"

CYS346_SG = np.array([18.888, -3.650, -29.979])
CYS346_CB = np.array([17.193, -4.221, -30.247])
CYS346_CA = np.array([16.292, -4.283, -28.999])

VINA_BIN = PROJECT_ROOT / "tools/vina"
EXHAUSTIVENESS = 4
NUM_MODES = 9
TIMEOUT_S = 600

# Small box centered on Cys346 SG (matches the eval pipeline)
BOX_CENTER = CYS346_SG.copy()
BOX_SIZE = np.array([20.0, 20.0, 20.0])
# Larger box for AD-CovDock --local_only — tethered ligands often extend
# outside the eval's 20Å box because the warhead anchor pulls the molecule
# toward the active-site Cys, and Lingo/H2 mols can have 25+Å arms.
BOX_SIZE_ADCOV = np.array([40.0, 40.0, 40.0])

# Restrained-Vina geometric thresholds (canonical Bürgi–Dunitz)
D_SG_LO, D_SG_HI = 1.55, 2.15      # Å
BD_LO, BD_HI = 102.0, 112.0        # deg

# Warhead SMARTS (acrylamide first; chloroacetamide / vinyl-sulfonamide fall-back)
WARHEAD_SMARTS_PATTERNS = [
    ("acrylamide",        "[CH2;X3]=[CH;X3][C;X3](=O)[N]",    (0, 1, 2)),
    ("acrylamide_loose",  "[CH2]=C[C](=O)[N,n]",              (0, 1, 2)),
    ("chloroacetamide",   "Cl[CH2][C](=O)[N]",                (1, 2, 0)),  # β=CH2, α=C(=O), γ=Cl
    ("vinyl_sulfonamide", "[CH2]=[CH][S](=O)(=O)[N]",         (0, 1, 2)),
]


# ============================================================
# Cohort / molecule loading
# ============================================================

def load_cohort_index(metrics_csv: Path) -> pd.DataFrame:
    """Load the 2712-row eval CSV and return (cohort, mol_idx, smi)."""
    df = pd.read_csv(metrics_csv)
    keep = ['cohort', 'mol_idx', 'name', 'smi']
    df = df[keep].copy()
    return df


def load_input_sdf(cohort: str) -> dict[int, Chem.Mol]:
    """Map mol_idx -> RDKit mol (with 3D coords) for a cohort's input.sdf."""
    p = PROJECT_ROOT / f"data/cohort_comparison/cohorts/{cohort}/input.sdf"
    if not p.exists():
        return {}
    out = {}
    supp = Chem.SDMolSupplier(str(p), removeHs=False, sanitize=True)
    for i, mol in enumerate(supp):
        if mol is None:
            continue
        # SDF order ≡ mol_idx 1..N (matches eval pipeline's per_mol.csv enumeration)
        out[i + 1] = mol
    return out


# ============================================================
# Warhead atom-index resolution (CRITICAL fix from coordinator)
# ============================================================

def find_warhead_atoms(smi: str) -> dict | None:
    """Find warhead electrophilic β-C (SMARTS match[0]) and adjacent atoms.

    Returns dict with keys: name, b_idx (β-C / electrophile), a_idx (α-C),
    g_idx (carbonyl-C or leaving-group anchor for BD angle).
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    for name, smarts, (b_pos, a_pos, g_pos) in WARHEAD_SMARTS_PATTERNS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if matches:
            m = matches[0]
            return {
                "warhead_name": name,
                "b_idx": int(m[b_pos]),
                "a_idx": int(m[a_pos]),
                "g_idx": int(m[g_pos]),
            }
    return None


# ============================================================
# AD-CovDock: tether ligand via meeko.CovalentBuilder + dock w/ stripped receptor
# ============================================================

_COVBUILDER_CACHE: dict = {}


def get_covbuilder():
    if "cb" not in _COVBUILDER_CACHE:
        from meeko import CovalentBuilder
        import prody
        rec = prody.parsePDB(str(RECEPTOR_PDB))
        _COVBUILDER_CACHE["cb"] = CovalentBuilder(rec, "A:CYS:346:CA,CB")
    return _COVBUILDER_CACHE["cb"]


def prepare_stripped_receptor():
    """Write receptor PDBQT minus Cys346 SG and CB (one-time, on disk)."""
    if RECEPTOR_PDBQT_STRIPPED.exists():
        return
    src = RECEPTOR_PDBQT.read_text()
    lines = []
    for ln in src.split("\n"):
        if ln.startswith("ATOM") and "CYS A 346" in ln:
            an = ln[12:16].strip()
            if an in ("CB", "SG"):
                continue
        lines.append(ln)
    RECEPTOR_PDBQT_STRIPPED.write_text("\n".join(lines))


def build_cov_tethered_pdbqt(mol: Chem.Mol, smi: str, out_path: Path) -> dict:
    """Use meeko CovalentBuilder to align warhead Cβ→CB, α-C→CA, then write PDBQT.

    Returns dict with keys: ok, msg, smarts, b_idx (in original SMILES).
    """
    from meeko import MoleculePreparation, PDBQTWriterLegacy
    info = find_warhead_atoms(smi)
    if info is None:
        return {"ok": False, "msg": "no_warhead", "smarts": None, "b_idx": None}

    name = info["warhead_name"]
    # Pick the SMARTS used in CovalentBuilder (CovBuilder re-greps so we need the
    # canonical SMARTS string + indices that put β-C → CB)
    if name in ("acrylamide", "acrylamide_loose"):
        smarts = "[CH2;X3]=[CH;X3][C;X3](=O)[N]"
        smarts_indices = [1, 0]  # α-C → CA, β-C → CB
    elif name == "chloroacetamide":
        smarts = "Cl[CH2][C](=O)[N]"
        smarts_indices = [2, 1]  # γ (carbonyl-C) → CA, β (CH2) → CB
    elif name == "vinyl_sulfonamide":
        smarts = "[CH2]=[CH][S](=O)(=O)[N]"
        smarts_indices = [1, 0]
    else:
        return {"ok": False, "msg": f"warhead_{name}_no_cov_smarts", "smarts": None, "b_idx": None}

    try:
        mol_h = Chem.AddHs(mol, addCoords=(mol.GetNumConformers() > 0))
        if mol_h.GetNumConformers() == 0:
            # Re-embed if no coords
            params = AllChem.ETKDGv3(); params.randomSeed = 42
            if AllChem.EmbedMolecule(mol_h, params) == -1:
                params.useRandomCoords = True
                if AllChem.EmbedMolecule(mol_h, params) == -1:
                    return {"ok": False, "msg": "embed_fail", "smarts": smarts, "b_idx": info["b_idx"]}
        cb = get_covbuilder()
        results = list(cb.process(mol_h, smarts=smarts, smarts_indices=smarts_indices, first_only=True))
        if not results:
            return {"ok": False, "msg": "covbuilder_no_match", "smarts": smarts, "b_idx": info["b_idx"]}
        aligned = results[0].mol  # rdkit Mol, conformer aligned
        prep = MoleculePreparation()
        setups = prep.prepare(aligned)
        if not setups:
            return {"ok": False, "msg": "meeko_setup_fail", "smarts": smarts, "b_idx": info["b_idx"]}
        pdbqt_str, is_ok, msg = PDBQTWriterLegacy.write_string(setups[0])
        if not is_ok:
            return {"ok": False, "msg": f"pdbqt_write_fail:{msg}", "smarts": smarts, "b_idx": info["b_idx"]}
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(pdbqt_str)
        return {"ok": True, "msg": "ok", "smarts": smarts, "b_idx": info["b_idx"]}
    except Exception as e:
        return {"ok": False, "msg": f"exc:{type(e).__name__}:{str(e)[:120]}", "smarts": smarts, "b_idx": info["b_idx"]}


def vina_dock(lig_pdbqt: Path, out_pose: Path, mode: str = "score_only", threads: int = 1) -> dict:
    """Run Vina against the Cys346-stripped receptor.

    Three modes:
      - "score_only": evaluate the tethered pose AS-IS (no relaxation). This
        is the canonical AD-CovDock-Vina recipe — the tether geometry from
        meeko.CovalentBuilder is preserved exactly. Score is high due to
        backbone clashes (warhead overlapping with Cys346 N/CA volume) but
        it IS the geometry the user designed for. Fast (~0.3s/mol).
      - "local_only": Newton minimization (relaxes some clashes but DOESN'T
        enforce the tether — Vina is non-covalent, so the warhead drifts away
        from SG when possible). Not the canonical recipe.
      - "global": full Vina dock with exhaustiveness=4. Diagnostic only.

    Returns dict: ok, msg, dt_s, score (float|None).
    """
    bs = BOX_SIZE_ADCOV if mode != "global" else BOX_SIZE
    cmd = [
        str(VINA_BIN),
        "--receptor", str(RECEPTOR_PDBQT_STRIPPED),
        "--ligand", str(lig_pdbqt),
        "--center_x", str(BOX_CENTER[0]),
        "--center_y", str(BOX_CENTER[1]),
        "--center_z", str(BOX_CENTER[2]),
        "--size_x", str(bs[0]),
        "--size_y", str(bs[1]),
        "--size_z", str(bs[2]),
        "--cpu", str(threads),
    ]
    if mode == "score_only":
        cmd.append("--score_only")
    elif mode == "local_only":
        cmd.extend(["--local_only", "--out", str(out_pose)])
    elif mode == "global":
        cmd.extend([
            "--exhaustiveness", str(EXHAUSTIVENESS),
            "--num_modes", str(NUM_MODES),
            "--out", str(out_pose),
        ])
    else:
        return {"ok": False, "msg": f"unknown_mode:{mode}", "dt_s": 0.0, "score": None}

    t0 = time.time()
    try:
        r = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        dt = time.time() - t0
        if r.returncode != 0:
            return {"ok": False, "msg": f"vina_rc={r.returncode}:{r.stderr[-120:]}", "dt_s": dt,
                    "score": None}
        score = None
        for ln in r.stdout.split("\n"):
            if "Estimated Free Energy of Binding" in ln:
                try:
                    score = float(ln.split(":")[1].split("(")[0].strip())
                except Exception:
                    pass
                break
        return {"ok": True, "msg": "ok", "dt_s": dt, "score": score}
    except subprocess.TimeoutExpired:
        return {"ok": False, "msg": "timeout", "dt_s": time.time() - t0, "score": None}
    except Exception as e:
        return {"ok": False, "msg": f"exc:{type(e).__name__}:{e}", "dt_s": time.time() - t0, "score": None}


# ============================================================
# Pose parsing (shared by both methods)
# ============================================================

def parse_pdbqt_models(pose_path: Path) -> list[dict]:
    """Parse ALL MODEL blocks of a Vina output PDBQT.

    Returns list of dicts: {coords (Nx3), elements (list), score (float)}.
    Atom order is the PDBQT order (heavy atoms only; H stripped if any).

    Also handles --local_only output which lacks MODEL/ENDMDL wrappers (single
    pose, score in REMARK VINA RESULT before ROOT).
    """
    if not pose_path.exists():
        return []
    out = []
    cur_coords, cur_elems, cur_score = [], [], None
    in_model = False
    has_model_tag = False
    for line in pose_path.read_text().split("\n"):
        if line.startswith("MODEL"):
            has_model_tag = True
            in_model = True
            cur_coords, cur_elems, cur_score = [], [], None
            continue
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:":
                    try:
                        cur_score = float(parts[i + 1])
                    except Exception:
                        pass
                    break
        if line.startswith("ENDMDL"):
            if cur_coords:
                out.append({
                    "coords": np.array(cur_coords, dtype=float),
                    "elements": list(cur_elems),
                    "score": cur_score,
                })
            in_model = False
            continue
        # Local-only mode: no MODEL tag, but we still want to collect atoms.
        collect = in_model or (not has_model_tag)
        if collect and (line.startswith("ATOM") or line.startswith("HETATM")):
            try:
                x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                ad_type = line[77:79].strip() if len(line) >= 79 else ""
                if ad_type in ("HD", "H"):
                    continue
                elem = ad_type[0] if ad_type else line[12:14].strip()[0]
                cur_coords.append([x, y, z])
                cur_elems.append(elem)
            except Exception:
                pass
    # Flush single-model local-only output
    if not has_model_tag and cur_coords:
        out.append({
            "coords": np.array(cur_coords, dtype=float),
            "elements": list(cur_elems),
            "score": cur_score,
        })
    return out


def parse_meeko_smiles_idx(pdbqt_path: Path) -> dict | None:
    """Map SMILES atom idx (0-based) → PDBQT heavy-atom parsed-list idx (0-based,
    same indexing as parse_pdbqt_models which filters out hydrogens).

    Meeko writes pairs (smi_idx_1based, pdbqt_serial_1based) across many lines,
    where pdbqt_serial_1based numbers ALL atoms (incl. explicit polar Hs / HD).
    We need to remap to the H-filtered parsed list. Otherwise, if any H atom
    has a serial < β-C serial, the mapping returns the wrong index by N (where
    N is the number of H atoms before β-C).

    Bug history: prior to 2026-06-01 backfill, this returned `serial - 1`
    unconditionally. For Lingo H1/H2/H3/L_locked the bug was latent because
    meeko's output placed Hs after the β-C in serial order. LibInvent_locked
    (re-prepped) and Amine_Replacements have Hs before the β-C, exposing the bug.
    """
    if not pdbqt_path.exists():
        return None
    # 1. Build serial(1b, all atoms) → parsed_idx(0b, heavy-only)
    serial_to_parsed = {}
    parsed_idx = 0
    serial = 0
    for ln in pdbqt_path.read_text().split("\n"):
        if not (ln.startswith("ATOM") or ln.startswith("HETATM")):
            continue
        serial += 1
        ad_type = ln[77:79].strip() if len(ln) >= 79 else ""
        if ad_type in ("HD", "H"):
            continue
        serial_to_parsed[serial] = parsed_idx
        parsed_idx += 1
    # 2. Parse REMARK SMILES IDX pairs and remap via serial_to_parsed
    all_ints = []
    found = False
    for ln in pdbqt_path.read_text().split("\n"):
        if ln.startswith("REMARK SMILES IDX"):
            found = True
            parts = ln.split()
            for p in parts[3:]:
                try:
                    all_ints.append(int(p))
                except ValueError:
                    pass
    if not found or len(all_ints) % 2 != 0:
        return None
    mapping = {}
    for i in range(0, len(all_ints), 2):
        smi_0b = all_ints[i] - 1
        serial_1b = all_ints[i + 1]
        # If serial is a hydrogen (not in serial_to_parsed), skip — shouldn't
        # happen because meeko SMILES IDX REMARK is keyed on heavy atoms only,
        # but we'll be safe.
        if serial_1b in serial_to_parsed:
            mapping[smi_0b] = serial_to_parsed[serial_1b]
    return mapping


def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 == 0 or n2 == 0:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def pose_geom(pose: dict, smi: str, lig_pdbqt: Path) -> dict:
    """Compute β-C—SG distance and BD angle (S–Cβ–Cα) for a single pose.

    Uses MEEKO REMARK SMILES IDX mapping to locate SMILES atoms in the PDBQT.
    """
    out = {"d_sg": None, "bd_angle": None, "warhead_found": False}
    info = find_warhead_atoms(smi)
    if info is None:
        return out
    smi_to_pose = parse_meeko_smiles_idx(lig_pdbqt)
    if smi_to_pose is None:
        return out
    b_pose = smi_to_pose.get(info["b_idx"])
    a_pose = smi_to_pose.get(info["a_idx"])
    if b_pose is None or a_pose is None:
        return out
    if b_pose >= len(pose["coords"]) or a_pose >= len(pose["coords"]):
        return out
    b_xyz = pose["coords"][b_pose]
    a_xyz = pose["coords"][a_pose]
    d_sg = float(np.linalg.norm(b_xyz - CYS346_SG))
    # Bürgi–Dunitz: angle between (SG→Cβ) and (Cβ→Cα)
    bd = angle_deg(CYS346_SG - b_xyz, a_xyz - b_xyz)
    out["d_sg"] = d_sg
    out["bd_angle"] = bd
    out["warhead_found"] = True
    return out


def read_tethered_geom(lig_pdbqt: Path, smi: str) -> dict:
    """Compute d(Cβ-SG) and BD-angle directly from the meeko-tethered PDBQT.

    For AD-CovDock with --score_only, we never write a relaxed pose — the
    geometry IS the input. Parse the lig_pdbqt's ATOM records, locate β-C
    and α-C via REMARK SMILES IDX, and compute the canonical metrics.
    """
    out = {"d_sg": None, "bd_angle": None}
    if not lig_pdbqt.exists():
        return out
    poses = parse_pdbqt_models(lig_pdbqt)
    if not poses:
        return out
    g = pose_geom(poses[0], smi, lig_pdbqt)
    return {"d_sg": g["d_sg"], "bd_angle": g["bd_angle"]}


def pose_valid_covalent(d_sg: float | None, bd: float | None) -> int:
    if d_sg is None or bd is None:
        return 0
    if not (D_SG_LO <= d_sg <= D_SG_HI):
        return 0
    if not (BD_LO <= bd <= BD_HI):
        return 0
    return 1


# ============================================================
# Worker: AD-CovDock one mol
# ============================================================

def _adcov_one(args):
    cohort, mol_idx, smi, sdf_path, work_dir = args
    work_dir = Path(work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    lig_pdbqt = work_dir / f"{cohort}_{mol_idx}_cov_lig.pdbqt"
    pose_pdbqt = work_dir / f"{cohort}_{mol_idx}_cov_pose.pdbqt"

    # 1. Load mol from SDF (preserve 3D coords)
    supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    mol = None
    if 1 <= mol_idx <= len(supp):
        mol = supp[mol_idx - 1]
    if mol is None:
        return {
            "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
            "AD_CovDock_score": None, "AD_CovDock_d_sg": None,
            "AD_CovDock_bd_angle": None, "AD_CovDock_pose_valid": 0,
            "AD_CovDock_msg": "sdf_load_fail", "AD_CovDock_dt_s": 0.0,
        }

    # 2. Build tethered PDBQT
    t0 = time.time()
    prep = build_cov_tethered_pdbqt(mol, smi, lig_pdbqt)
    if not prep["ok"]:
        return {
            "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
            "AD_CovDock_score": None, "AD_CovDock_d_sg": None,
            "AD_CovDock_bd_angle": None, "AD_CovDock_pose_valid": 0,
            "AD_CovDock_msg": f"prep_fail:{prep['msg']}",
            "AD_CovDock_dt_s": time.time() - t0,
        }

    # 3. Vina --score_only on the meeko-tethered pose (canonical AD-CovDock-Vina).
    #    The pose's β-C is at Cys346 CB by construction (d_SG ≈ 1.85 Å).
    dock = vina_dock(lig_pdbqt, pose_pdbqt, mode="score_only", threads=1)
    if not dock["ok"]:
        return {
            "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
            "AD_CovDock_score": None, "AD_CovDock_d_sg": None,
            "AD_CovDock_bd_angle": None, "AD_CovDock_pose_valid": 0,
            "AD_CovDock_any_pose_valid": 0, "AD_CovDock_n_poses": 0,
            "AD_CovDock_msg": f"score_fail:{dock['msg']}",
            "AD_CovDock_dt_s": time.time() - t0,
        }

    # 4. Compute geometry from the TETHERED PDBQT directly (score_only doesn't
    #    write a pose file; the tether IS the pose).
    g = read_tethered_geom(lig_pdbqt, smi)
    valid = pose_valid_covalent(g["d_sg"], g["bd_angle"])

    return {
        "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
        "AD_CovDock_score": dock.get("score"),
        "AD_CovDock_d_sg": g["d_sg"],
        "AD_CovDock_bd_angle": g["bd_angle"],
        "AD_CovDock_pose_valid": valid,
        "AD_CovDock_any_pose_valid": valid,
        "AD_CovDock_n_poses": 1,
        "AD_CovDock_msg": "ok",
        "AD_CovDock_dt_s": time.time() - t0,
    }


# ============================================================
# Worker: Restrained-Vina recompute from EXISTING poses
# ============================================================

def _restrained_one(args):
    """Recompute d(Cβ-SG) + BD from the existing eval-pipeline Vina pose
    using the CORRECT atom-index rule (terminal CH2 = β-C)."""
    cohort, mol_idx, smi, pose_path, lig_pdbqt_path = args
    pose_path = Path(pose_path); lig_pdbqt_path = Path(lig_pdbqt_path)
    if not pose_path.exists() or not lig_pdbqt_path.exists():
        return {
            "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
            "Restrained_Vina_score": None, "Restrained_Vina_d_sg": None,
            "Restrained_Vina_bd_angle": None, "Restrained_Vina_pose_valid": 0,
            "Restrained_Vina_any_pose_valid": 0, "Restrained_Vina_n_poses": 0,
            "Restrained_Vina_msg": "no_pose_file",
        }
    poses = parse_pdbqt_models(pose_path)
    if not poses:
        return {
            "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
            "Restrained_Vina_score": None, "Restrained_Vina_d_sg": None,
            "Restrained_Vina_bd_angle": None, "Restrained_Vina_pose_valid": 0,
            "Restrained_Vina_any_pose_valid": 0, "Restrained_Vina_n_poses": 0,
            "Restrained_Vina_msg": "no_poses_in_file",
        }
    top = poses[0]
    g = pose_geom(top, smi, lig_pdbqt_path)
    valid = pose_valid_covalent(g["d_sg"], g["bd_angle"])
    any_valid = 0
    for p in poses:
        gp = pose_geom(p, smi, lig_pdbqt_path)
        if pose_valid_covalent(gp["d_sg"], gp["bd_angle"]):
            any_valid = 1
            break

    return {
        "cohort": cohort, "mol_idx": mol_idx, "smi": smi,
        "Restrained_Vina_score": float(top["score"]) if top["score"] is not None else None,
        "Restrained_Vina_d_sg": g["d_sg"],
        "Restrained_Vina_bd_angle": g["bd_angle"],
        "Restrained_Vina_pose_valid": valid,
        "Restrained_Vina_any_pose_valid": any_valid,
        "Restrained_Vina_n_poses": len(poses),
        "Restrained_Vina_msg": "ok",
    }


# ============================================================
# Driver
# ============================================================

def run_adcovdock(df_index: pd.DataFrame, work_dir: Path, n_workers: int) -> pd.DataFrame:
    """Build tethered ligand + run Vina for every (cohort, mol_idx)."""
    prepare_stripped_receptor()
    print(f"Receptor stripped: {RECEPTOR_PDBQT_STRIPPED.name} (Cys346 CB+SG removed)")

    tasks = []
    for cohort in df_index['cohort'].unique():
        sdf = PROJECT_ROOT / f"data/cohort_comparison/cohorts/{cohort}/input.sdf"
        sub = df_index[df_index['cohort'] == cohort]
        for _, row in sub.iterrows():
            tasks.append((cohort, int(row['mol_idx']), row['smi'], str(sdf), str(work_dir / cohort)))
    print(f"AD-CovDock tasks: {len(tasks)}; n_workers={n_workers}")

    results = []
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_adcov_one, tasks, chunksize=4)):
            results.append(r)
            if (i + 1) % 100 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta = (len(tasks) - i - 1) / rate
                print(f"  AD-CovDock {i+1}/{len(tasks)}  "
                      f"({rate:.2f} mol/s, ETA {eta/60:.1f} min)")
    return pd.DataFrame(results)


def run_restrained_vina(df_index: pd.DataFrame, eval_dir: Path, n_workers: int) -> pd.DataFrame:
    """Re-score every existing eval-pipeline pose with the corrected atom-index."""
    tasks = []
    for cohort in df_index['cohort'].unique():
        per_cohort = eval_dir / cohort
        sub = df_index[df_index['cohort'] == cohort]
        for _, row in sub.iterrows():
            mol_idx = int(row['mol_idx'])
            pose_path = per_cohort / "poses" / f"{cohort}_{mol_idx}_pose.pdbqt"
            lig_pdbqt = per_cohort / "ligands_pdbqt" / f"{cohort}_{mol_idx}.pdbqt"
            tasks.append((cohort, mol_idx, row['smi'], str(pose_path), str(lig_pdbqt)))
    print(f"Restrained-Vina tasks: {len(tasks)}; n_workers={n_workers}")

    results = []
    t0 = time.time()
    with Pool(processes=n_workers) as pool:
        for i, r in enumerate(pool.imap_unordered(_restrained_one, tasks, chunksize=32)):
            results.append(r)
            if (i + 1) % 500 == 0:
                elapsed = time.time() - t0
                print(f"  Restrained {i+1}/{len(tasks)}  ({(i+1)/elapsed:.1f}/s)")
    return pd.DataFrame(results)


def merge_with_eval_csv(adcov_df: pd.DataFrame, restr_df: pd.DataFrame,
                       eval_csv: Path) -> pd.DataFrame:
    """Merge AD-CovDock + Restrained-Vina + non-cov Vina from eval CSV."""
    eval_df = pd.read_csv(eval_csv)[
        ['cohort', 'mol_idx', 'name', 'smi', 'vina_score',
         'd_cb_sg_top1', 'bd_angle_top1', 'any_pose_feasible']
    ].copy()
    eval_df = eval_df.rename(columns={
        'vina_score': 'NonCovVina_score',
        'd_cb_sg_top1': 'NonCovVina_d_sg_OLD',   # OLD = used wrong atom (atom 1)
        'bd_angle_top1': 'NonCovVina_bd_OLD',
        'any_pose_feasible': 'NonCovVina_any_pose_feasible_OLD',
    })
    m = eval_df.merge(adcov_df, on=['cohort', 'mol_idx', 'smi'], how='left')
    m = m.merge(restr_df, on=['cohort', 'mol_idx', 'smi'], how='left')
    return m


def per_cohort_summary(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cohort, sub in merged.groupby('cohort'):
        rows.append({
            "cohort": cohort,
            "N": len(sub),
            "AD_CovDock_score_mean": sub['AD_CovDock_score'].mean(),
            "AD_CovDock_score_std": sub['AD_CovDock_score'].std(),
            "AD_CovDock_score_median": sub['AD_CovDock_score'].median(),
            "AD_CovDock_score_pct_lt_minus6": float((sub['AD_CovDock_score'] < -6.0).mean()),
            "AD_CovDock_d_sg_mean": sub['AD_CovDock_d_sg'].mean(),
            "AD_CovDock_bd_mean": sub['AD_CovDock_bd_angle'].mean(),
            "AD_CovDock_pose_valid_pct": float(sub['AD_CovDock_pose_valid'].mean()),
            "AD_CovDock_any_pose_valid_pct": float(sub['AD_CovDock_any_pose_valid'].mean()),
            "Restrained_score_mean": sub['Restrained_Vina_score'].mean(),
            "Restrained_d_sg_mean": sub['Restrained_Vina_d_sg'].mean(),
            "Restrained_bd_mean": sub['Restrained_Vina_bd_angle'].mean(),
            "Restrained_pose_valid_pct": float(sub['Restrained_Vina_pose_valid'].mean()),
            "Restrained_any_pose_valid_pct": float(sub['Restrained_Vina_any_pose_valid'].mean()),
            "NonCovVina_score_mean": sub['NonCovVina_score'].mean(),
        })
    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--eval-csv", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison/all_cohorts_metrics.csv"))
    ap.add_argument("--eval-dir", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison/per_cohort"))
    ap.add_argument("--out-dir", default=str(PROJECT_ROOT / "results/paper_evaluation/cohort_comparison"))
    ap.add_argument("--work-dir", default=str(PROJECT_ROOT / "data/cov_docking_work"))
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0,
                    help="If >0, only process the first N rows (debug)")
    ap.add_argument("--sample-by-cohort", type=int, default=0,
                    help="If >0, sample N mols from EACH cohort (debug)")
    ap.add_argument("--skip-adcov", action="store_true", help="Skip AD-CovDock pass")
    ap.add_argument("--skip-restrained", action="store_true", help="Skip Restrained-Vina pass")
    args = ap.parse_args()

    eval_csv = Path(args.eval_csv); eval_dir = Path(args.eval_dir)
    out_dir = Path(args.out_dir); work_dir = Path(args.work_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    work_dir.mkdir(parents=True, exist_ok=True)

    print(f"Loading eval CSV: {eval_csv}")
    df = load_cohort_index(eval_csv)
    if args.limit > 0:
        df = df.head(args.limit).copy()
    if args.sample_by_cohort > 0:
        df = df.groupby('cohort').head(args.sample_by_cohort).reset_index(drop=True)
    print(f"N mols total: {len(df)}; cohorts: {df['cohort'].nunique()}")

    # Run Restrained-Vina first (fast, just re-parsing) - lets us validate pipeline
    restr_path = out_dir / "covalent_docking_restrained_vina.csv"
    if not args.skip_restrained:
        print("\n=== PASS 1: Restrained-Vina (re-measure existing poses) ===")
        restr_df = run_restrained_vina(df, eval_dir, n_workers=args.workers)
        restr_df.to_csv(restr_path, index=False)
        print(f"wrote {restr_path}  rows={len(restr_df)}")
    else:
        restr_df = pd.read_csv(restr_path)

    # Run AD-CovDock
    adcov_path = out_dir / "covalent_docking_adcov.csv"
    if not args.skip_adcov:
        print("\n=== PASS 2: AD-CovDock (tethered Vina) ===")
        adcov_df = run_adcovdock(df, work_dir, n_workers=args.workers)
        adcov_df.to_csv(adcov_path, index=False)
        print(f"wrote {adcov_path}  rows={len(adcov_df)}")
    else:
        adcov_df = pd.read_csv(adcov_path)

    # Merge + summary
    print("\n=== Merging + per-cohort summary ===")
    merged = merge_with_eval_csv(adcov_df, restr_df, eval_csv)
    merged_path = out_dir / "covalent_docking_results.csv"
    merged.to_csv(merged_path, index=False)
    print(f"wrote {merged_path}  rows={len(merged)}")

    summary = per_cohort_summary(merged)
    summary_path = out_dir / "covalent_docking_per_cohort.csv"
    summary.to_csv(summary_path, index=False)
    print(f"wrote {summary_path}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
