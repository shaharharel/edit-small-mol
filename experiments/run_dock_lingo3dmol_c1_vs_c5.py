#!/usr/bin/env python3
"""Dock the L2 scaffold-anchor C1-aliphatic cohort and the new C5-aromatic cohort
against ZAP70 (PDB 4K2R) using AutoDock Vina.

Purpose: validate empirically that fixing the C1->C5 regiochemistry actually
improves predicted binding affinity and positions the recognition arm toward
the Met414 hinge.

REUSES existing infrastructure:
- Receptor PDBQT: data/docking_500/receptor.pdbqt (PDB 4K2R, prepared previously)
- Vina binary: tools/vina (AutoDock Vina v1.2.7)
- Ligand prep recipe: same meeko/MoleculePreparation flow as run_dock_chembl_zap70.py

Box: centered on Cys346 SG (18.888, -3.650, -29.979), 20x20x20 A
(tighter than the docking_500 ATP box; covalent ligands should dock here).

Mac CPU only.

Usage:
    conda run --no-capture-output -n quris python -u \\
        experiments/run_dock_lingo3dmol_c1_vs_c5.py
"""
import json
import os
import subprocess
import sys
import time
import warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
VINA_BIN = PROJECT_ROOT / "tools" / "vina"
RECEPTOR_PDBQT = PROJECT_ROOT / "data" / "docking_500" / "receptor.pdbqt"

# --- Pocket: covalent box centered on Cys346 SG ---
# Cys346 SG coordinates from data/docking_500/receptor_clean.pdb
CYS346_SG = np.array([18.888, -3.650, -29.979])
# Met414 backbone N: hinge proxy
MET414_N = np.array([1.671, -5.312, -27.925])
# Met414 backbone C (alternative hinge proxy)
MET414_C = np.array([1.313, -3.739, -26.006])

BOX_CENTER = CYS346_SG.copy()
BOX_SIZE = np.array([20.0, 20.0, 20.0])

# --- Cohort inputs ---
COHORTS = {
    "C1": PROJECT_ROOT / "data" / "lingo3dmol_L2_scaffold_anchor" / "samples_T10.sdf",
    "C5": PROJECT_ROOT / "data" / "lingo3dmol_L2_scaffold_C5" / "samples_T10.sdf",
}

OUT_DIR = PROJECT_ROOT / "data" / "lingo3dmol_docking"
OUT_DIR.mkdir(parents=True, exist_ok=True)

N_CPUS = min(cpu_count(), 6)
EXHAUSTIVENESS = 8
NUM_MODES = 9
TIMEOUT_S = 600

# Acrylamide warhead SMARTS: terminal C=C-C(=O)-N
ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"


# --- Ligand preparation ---

def sdf_to_pdbqt(mol, out_path: Path):
    """Convert an RDKit mol (with 3D coords) to a PDBQT via meeko.

    Keeps the 3D coordinates from the SDF (DON'T re-embed; the inpaint pose
    is informative as a starting structure but Vina re-searches anyway).
    """
    try:
        mol_h = Chem.AddHs(mol, addCoords=True)
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol_h)
        pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if is_ok:
            out_path.write_text(pdbqt_str)
            return True
    except Exception as e:
        return False
    return False


def reembed_and_pdbqt(smiles: str, out_path: Path, seed: int = 42):
    """Fallback: SMILES -> 3D embed -> PDBQT."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return False
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    res = AllChem.EmbedMolecule(mol, params)
    if res == -1:
        params.useRandomCoords = True
        res = AllChem.EmbedMolecule(mol, params)
        if res == -1:
            return False
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    try:
        from meeko import MoleculePreparation, PDBQTWriterLegacy
        prep = MoleculePreparation()
        setups = prep.prepare(mol)
        pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(setups[0])
        if is_ok:
            out_path.write_text(pdbqt_str)
            return True
    except Exception:
        return False
    return False


# --- PDBQT score parsing ---

def parse_pdbqt_scores(pose_path: Path):
    result = {"vina_score": None, "vina_scores_all": []}
    if not pose_path.exists():
        return result
    scores = []
    for line in pose_path.read_text().split("\n"):
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:":
                    try:
                        scores.append(float(parts[i + 1]))
                    except (IndexError, ValueError):
                        pass
                    break
    if scores:
        result["vina_score"] = scores[0]
        result["vina_scores_all"] = scores
    return result


# --- Parse a PDBQT pose model (best mode only) into atom coordinates ---

def parse_pdbqt_top_model(pose_path: Path):
    """Extract the first MODEL's heavy-atom (incl. H) coordinates and elements.

    Returns dict with 'coords' (N,3) and 'elements' (list of str) and the
    raw ATOM lines (for downstream mapping via PDB index).
    """
    if not pose_path.exists():
        return None
    coords = []
    elements = []
    atom_lines = []
    in_model = False
    saw_model = False
    for line in pose_path.read_text().split("\n"):
        if line.startswith("MODEL"):
            if saw_model:
                break  # only first model
            in_model = True
            saw_model = True
            continue
        if line.startswith("ENDMDL"):
            break
        if in_model and (line.startswith("ATOM") or line.startswith("HETATM")):
            try:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                # PDBQT element is in cols 77-78 (autodock atom type), but
                # the first letter is the element for most
                ad_type = line[77:79].strip() if len(line) >= 79 else ""
                elem = ad_type[0] if ad_type else line[12:14].strip()[0]
                coords.append([x, y, z])
                elements.append(elem)
                atom_lines.append(line)
            except (ValueError, IndexError):
                pass
    if not coords:
        return None
    return {
        "coords": np.array(coords),
        "elements": elements,
        "atom_lines": atom_lines,
    }


# --- Geometric measurements on top docked pose ---

def find_acrylamide_warhead_indices(smiles: str):
    """Return atom indices in the ORIGINAL (no-H) RDKit mol for the acrylamide
    warhead: (CH2_terminal, CH_beta, C_carbonyl, O, N).

    Used to map onto the docked pose via atom-order correspondence with meeko.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        return None
    # take first match: (CH2, CH, C=O, N)  — the SMARTS pattern is [CH2]=[CH]C(=O)N
    m = matches[0]
    # m = (idx_CH2, idx_CH_beta, idx_C_carbonyl, idx_N)
    return {
        "ch2_terminal": m[0],  # C-alpha of warhead
        "ch_beta": m[1],       # C-beta (electrophile)
        "c_carbonyl": m[2],
        "n_amide": m[3],
        "all_warhead_idx": list(m),
    }


def measure_pose_geometry(pose_path: Path, smiles: str):
    """For the top docked pose, measure:
    - d(C_beta_warhead, Cys346_SG)
    - d(recognition_arm_centroid, Met414_N)
    - arm_centroid coords (for QC)

    The recognition arm = all heavy atoms NOT in the warhead.
    """
    pose = parse_pdbqt_top_model(pose_path)
    if pose is None:
        return None

    warhead = find_acrylamide_warhead_indices(smiles)
    if warhead is None:
        # No acrylamide — only Cys distance to nearest C, and arm-centroid = all atoms
        coords = pose["coords"]
        elements = pose["elements"]
        # nearest carbon to SG
        carbon_mask = np.array([e == "C" for e in elements])
        if carbon_mask.sum() == 0:
            return None
        dists = np.linalg.norm(coords[carbon_mask] - CYS346_SG, axis=1)
        d_sg = float(dists.min())
        arm_centroid = coords.mean(axis=0)
        d_met = float(np.linalg.norm(arm_centroid - MET414_N))
        return {
            "d_warhead_Cbeta_to_Cys346SG": d_sg,
            "d_recognition_arm_to_Met414N": d_met,
            "d_recognition_arm_to_Met414C": float(np.linalg.norm(arm_centroid - MET414_C)),
            "warhead_found": False,
            "arm_centroid": arm_centroid.tolist(),
        }

    # We need to map the SMILES-derived warhead atom indices onto the PDBQT
    # pose atom order. meeko PDBQT preserves the order of heavy atoms from
    # the prepared mol (which is AddHs of the original). The PDBQT ATOM list
    # also contains polar H atoms — but heavy atoms come in canonical order.
    # Strategy: index by ELEMENT-only, taking the i-th C, i-th N, etc.
    elements = pose["elements"]
    coords = pose["coords"]

    # Build map: original heavy atom index -> pose index by walking the RDKit
    # mol heavy atoms in order and matching to PDBQT heavy atoms in order.
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # The order of heavy atoms in PDBQT (from meeko) generally matches RDKit
    # canonical atom order of the input mol. We'll match by element sequence.

    heavy_atoms_rdkit = [a.GetSymbol() for a in mol.GetAtoms()]  # excludes H
    # Filter PDBQT atoms to heavy atoms only (PDBQT element 'H' or 'HD')
    pose_heavy_idx = [i for i, e in enumerate(elements) if e != "H"]
    pose_heavy_elements = [elements[i] for i in pose_heavy_idx]

    if len(pose_heavy_elements) != len(heavy_atoms_rdkit):
        # element count mismatch; fall back to nearest-carbon-to-SG approach
        carbon_mask = np.array([e == "C" for e in elements])
        dists = np.linalg.norm(coords[carbon_mask] - CYS346_SG, axis=1)
        d_sg = float(dists.min())
        arm_centroid = coords.mean(axis=0)
        d_met = float(np.linalg.norm(arm_centroid - MET414_N))
        return {
            "d_warhead_Cbeta_to_Cys346SG": d_sg,
            "d_recognition_arm_to_Met414N": d_met,
            "d_recognition_arm_to_Met414C": float(np.linalg.norm(arm_centroid - MET414_C)),
            "warhead_found": False,
            "fallback_reason": f"element_count_mismatch:{len(pose_heavy_elements)}vs{len(heavy_atoms_rdkit)}",
            "arm_centroid": arm_centroid.tolist(),
        }

    # Sequential element match check
    elements_match = all(a == b for a, b in zip(pose_heavy_elements, heavy_atoms_rdkit))
    if not elements_match:
        # try permutation by element: walk pose, match next mol atom of same element
        # this is risky, so fall back
        carbon_mask = np.array([e == "C" for e in elements])
        dists = np.linalg.norm(coords[carbon_mask] - CYS346_SG, axis=1)
        d_sg = float(dists.min())
        arm_centroid = coords.mean(axis=0)
        d_met = float(np.linalg.norm(arm_centroid - MET414_N))
        return {
            "d_warhead_Cbeta_to_Cys346SG": d_sg,
            "d_recognition_arm_to_Met414N": d_met,
            "d_recognition_arm_to_Met414C": float(np.linalg.norm(arm_centroid - MET414_C)),
            "warhead_found": False,
            "fallback_reason": "element_sequence_mismatch",
            "arm_centroid": arm_centroid.tolist(),
        }

    # Map mol heavy idx -> pose absolute idx
    mol2pose = {i: pose_heavy_idx[i] for i in range(len(heavy_atoms_rdkit))}

    cbeta_pose_idx = mol2pose[warhead["ch_beta"]]
    cbeta_xyz = coords[cbeta_pose_idx]
    d_sg = float(np.linalg.norm(cbeta_xyz - CYS346_SG))

    # Recognition arm: all heavy atoms NOT in the warhead set
    warhead_set = set(warhead["all_warhead_idx"])
    arm_heavy_pose_idx = [mol2pose[i] for i in range(len(heavy_atoms_rdkit))
                          if i not in warhead_set]
    if not arm_heavy_pose_idx:
        return None
    arm_coords = coords[arm_heavy_pose_idx]
    arm_centroid = arm_coords.mean(axis=0)
    d_met_N = float(np.linalg.norm(arm_centroid - MET414_N))
    d_met_C = float(np.linalg.norm(arm_centroid - MET414_C))

    # Also report the minimum distance from any arm heavy atom to Met414 N
    d_met_N_min = float(np.linalg.norm(arm_coords - MET414_N, axis=1).min())

    return {
        "d_warhead_Cbeta_to_Cys346SG": d_sg,
        "d_recognition_arm_to_Met414N": d_met_N,
        "d_recognition_arm_to_Met414N_min": d_met_N_min,
        "d_recognition_arm_to_Met414C": d_met_C,
        "warhead_found": True,
        "arm_centroid": arm_centroid.tolist(),
        "cbeta_xyz": cbeta_xyz.tolist(),
    }


# --- Docking driver ---

def dock_single(args):
    cohort, mol_idx, name, smiles, sdf_mol_block, ligand_pdbqt_dir, pose_dir = args

    pose_path = pose_dir / f"{name}_pose.pdbqt"
    ligand_pdbqt = ligand_pdbqt_dir / f"{name}.pdbqt"
    score_cache = pose_dir / f"{name}_scores.json"

    result = {
        "cohort": cohort,
        "mol_idx": mol_idx,
        "name": name,
        "smi": smiles,
        "vina_kcalmol": None,
        "vina_all_modes": [],
        "top_mode_sdf_path": "",
        "pose_pdbqt_path": str(pose_path),
        "dock_time_s": 0.0,
        "success": False,
        "error": None,
    }

    if score_cache.exists():
        cached = json.loads(score_cache.read_text())
        result.update(cached)
        result["success"] = cached.get("vina_kcalmol") is not None
        return result

    # 1. Prepare ligand PDBQT (reuse SDF coords first, fallback to re-embed)
    if not ligand_pdbqt.exists():
        prepped = False
        if sdf_mol_block is not None:
            try:
                mol = Chem.MolFromMolBlock(sdf_mol_block, removeHs=False)
                if mol is not None:
                    prepped = sdf_to_pdbqt(mol, ligand_pdbqt)
            except Exception:
                prepped = False
        if not prepped:
            prepped = reembed_and_pdbqt(smiles, ligand_pdbqt)
        if not prepped:
            result["error"] = "ligand_prep_failed"
            score_cache.write_text(json.dumps({k: result[k] for k in
                ["cohort", "mol_idx", "name", "vina_kcalmol", "vina_all_modes",
                 "dock_time_s", "success", "error"]}, indent=2))
            return result

    # 2. Run Vina
    cmd = [
        str(VINA_BIN),
        "--receptor", str(RECEPTOR_PDBQT),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{BOX_CENTER[0]:.3f}",
        "--center_y", f"{BOX_CENTER[1]:.3f}",
        "--center_z", f"{BOX_CENTER[2]:.3f}",
        "--size_x", f"{BOX_SIZE[0]:.1f}",
        "--size_y", f"{BOX_SIZE[1]:.1f}",
        "--size_z", f"{BOX_SIZE[2]:.1f}",
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--out", str(pose_path),
    ]
    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=TIMEOUT_S)
        result["dock_time_s"] = time.time() - t0
        parsed = parse_pdbqt_scores(pose_path)
        result["vina_kcalmol"] = parsed["vina_score"]
        result["vina_all_modes"] = parsed["vina_scores_all"]
        if result["vina_kcalmol"] is not None:
            result["success"] = True
            result["top_mode_sdf_path"] = str(pose_path)
        elif proc.returncode != 0:
            result["error"] = f"vina_exit_{proc.returncode}"
        else:
            result["error"] = "no_score_parsed"
    except subprocess.TimeoutExpired:
        result["dock_time_s"] = time.time() - t0
        result["error"] = f"timeout_{TIMEOUT_S}s"
    except Exception as e:
        result["dock_time_s"] = time.time() - t0
        result["error"] = str(e)[:200]

    score_cache.write_text(json.dumps({k: result[k] for k in
        ["cohort", "mol_idx", "name", "vina_kcalmol", "vina_all_modes",
         "dock_time_s", "success", "error", "top_mode_sdf_path"]}, indent=2))
    return result


# --- Pipeline ---

def load_cohort(cohort_name: str, sdf_path: Path):
    """Load mols from an SDF and yield (name, smiles, mol_block) tuples."""
    supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=True)
    out = []
    for i, mol in enumerate(supp):
        if mol is None:
            out.append({
                "cohort": cohort_name, "mol_idx": i, "name": f"{cohort_name}_{i}_PARSE_FAIL",
                "smi": "", "mol_block": None,
            })
            continue
        try:
            smi = Chem.MolToSmiles(mol)
        except Exception:
            smi = ""
        try:
            name = mol.GetProp("_Name") if mol.HasProp("_Name") else f"{cohort_name}_{i}"
        except Exception:
            name = f"{cohort_name}_{i}"
        # Re-prefix with cohort to avoid name clashes
        name = f"{cohort_name}_{i}_{name}"
        try:
            mol_block = Chem.MolToMolBlock(mol)
        except Exception:
            mol_block = None
        out.append({
            "cohort": cohort_name, "mol_idx": i, "name": name,
            "smi": smi, "mol_block": mol_block,
        })
    return out


def main():
    print("=" * 72)
    print("  L2 Scaffold-Anchor C1 vs C5 Cohort Docking against ZAP70 (4K2R)")
    print(f"  Box center (Cys346 SG): ({BOX_CENTER[0]:.3f}, {BOX_CENTER[1]:.3f}, {BOX_CENTER[2]:.3f})")
    print(f"  Box size:               ({BOX_SIZE[0]:.1f}, {BOX_SIZE[1]:.1f}, {BOX_SIZE[2]:.1f}) A")
    print(f"  Receptor:               {RECEPTOR_PDBQT}")
    print(f"  CPUs:                   {N_CPUS}")
    print(f"  Exhaustiveness:         {EXHAUSTIVENESS}, num_modes: {NUM_MODES}")
    print("=" * 72)

    if not VINA_BIN.exists():
        print(f"ERROR: Vina binary not found: {VINA_BIN}")
        sys.exit(1)
    if not RECEPTOR_PDBQT.exists():
        print(f"ERROR: Receptor PDBQT not found: {RECEPTOR_PDBQT}")
        sys.exit(1)

    # Load both cohorts
    all_records = []
    for cohort, sdf in COHORTS.items():
        records = load_cohort(cohort, sdf)
        print(f"  {cohort}: loaded {len(records)} mols from {sdf.name}")
        all_records.extend(records)

    # Per-cohort output dirs
    cohort_dirs = {}
    for cohort in COHORTS:
        d = OUT_DIR / cohort
        (d / "ligands_pdbqt").mkdir(parents=True, exist_ok=True)
        (d / "poses").mkdir(parents=True, exist_ok=True)
        cohort_dirs[cohort] = d

    # Build job list
    jobs = []
    for rec in all_records:
        if rec["smi"] == "":
            continue
        cohort = rec["cohort"]
        jobs.append((
            cohort,
            rec["mol_idx"],
            rec["name"],
            rec["smi"],
            rec["mol_block"],
            cohort_dirs[cohort] / "ligands_pdbqt",
            cohort_dirs[cohort] / "poses",
        ))

    print(f"\n--- Docking {len(jobs)} mols with {N_CPUS} parallel workers ---")
    t_start = time.time()
    results = []
    with Pool(processes=N_CPUS) as pool:
        for i, r in enumerate(pool.imap_unordered(dock_single, jobs), start=1):
            results.append(r)
            if i % 10 == 0 or i == len(jobs):
                n_ok = sum(1 for x in results if x["success"])
                best = min((x["vina_kcalmol"] for x in results if x["vina_kcalmol"] is not None),
                           default=0.0)
                elapsed = time.time() - t_start
                rate = i / elapsed if elapsed > 0 else 0
                eta = (len(jobs) - i) / rate if rate > 0 else 0
                print(f"  [{i}/{len(jobs)}] ok={n_ok} best={best:.2f} "
                      f"rate={rate:.2f}/s ETA={eta/60:.1f}min")
    total_t = time.time() - t_start
    print(f"\n--- Docking finished in {total_t/60:.1f} min ---")

    # Geometry measurement on successful poses
    print("\n--- Measuring pose geometry ---")
    geometry = {}
    for r in results:
        if not r["success"]:
            geometry[r["name"]] = None
            continue
        geo = measure_pose_geometry(Path(r["pose_pdbqt_path"]), r["smi"])
        geometry[r["name"]] = geo

    # Build per-cohort CSVs
    for cohort in COHORTS:
        rows = []
        for r in [x for x in results if x["cohort"] == cohort]:
            geo = geometry.get(r["name"]) or {}
            rows.append({
                "smi": r["smi"],
                "name": r["name"],
                "vina_kcalmol": r["vina_kcalmol"],
                "top_mode_sdf_path": r["top_mode_sdf_path"],
                "vina_all_modes": ";".join(f"{x:.2f}" for x in r["vina_all_modes"]),
                "dock_time_s": round(r["dock_time_s"], 2),
                "success": r["success"],
                "error": r["error"],
                "d_warhead_Cbeta_to_Cys346SG": geo.get("d_warhead_Cbeta_to_Cys346SG"),
                "d_recognition_arm_to_Met414N": geo.get("d_recognition_arm_to_Met414N"),
                "d_recognition_arm_to_Met414N_min": geo.get("d_recognition_arm_to_Met414N_min"),
                "d_recognition_arm_to_Met414C": geo.get("d_recognition_arm_to_Met414C"),
                "warhead_found": geo.get("warhead_found"),
            })
        df = pd.DataFrame(rows)
        out_csv = OUT_DIR / f"{cohort}_vina.csv"
        df.to_csv(out_csv, index=False)
        print(f"  {cohort}: wrote {out_csv} ({len(df)} rows, "
              f"{df['success'].sum()} successful)")

    # Combined JSON
    combined = {
        "box_center_cys346_sg": BOX_CENTER.tolist(),
        "box_size": BOX_SIZE.tolist(),
        "met414_N": MET414_N.tolist(),
        "met414_C": MET414_C.tolist(),
        "exhaustiveness": EXHAUSTIVENESS,
        "num_modes": NUM_MODES,
        "n_cpus": N_CPUS,
        "total_time_min": round(total_t / 60.0, 2),
        "results": [
            {**{k: r[k] for k in ["cohort", "mol_idx", "name", "smi",
                                  "vina_kcalmol", "vina_all_modes",
                                  "dock_time_s", "success", "error"]},
             "geometry": geometry.get(r["name"])}
            for r in results
        ],
    }
    (OUT_DIR / "combined_results.json").write_text(json.dumps(combined, indent=2, default=str))
    print(f"  Combined JSON: {OUT_DIR / 'combined_results.json'}")

    # Quick summary
    print("\n--- Summary ---")
    for cohort in COHORTS:
        cohort_results = [x for x in results if x["cohort"] == cohort and x["success"]]
        if not cohort_results:
            print(f"  {cohort}: 0 successful docks")
            continue
        scores = np.array([x["vina_kcalmol"] for x in cohort_results])
        print(f"  {cohort} (n={len(scores)}): median={np.median(scores):.2f} "
              f"mean={scores.mean():.2f} std={scores.std():.2f} "
              f"best={scores.min():.2f} worst={scores.max():.2f}")


if __name__ == "__main__":
    main()
