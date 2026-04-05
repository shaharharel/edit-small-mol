#!/usr/bin/env python3
"""Dock ALL ChEMBL ZAP70 (CHEMBL2803) molecules against PDB 4K2R using AutoDock Vina.

Reuses the receptor PDBQT and pocket definition from the existing docking_500 run.
Parses Vina scores from PDBQT REMARK lines (not stdout) for reliability.
Reports correlation between Vina docking scores and experimental pIC50.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_dock_chembl_zap70.py
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

# Input paths — reuse receptor from docking_500
RECEPTOR_PDBQT = PROJECT_ROOT / "data" / "docking_500" / "receptor.pdbqt"
POCKET_JSON = PROJECT_ROOT / "data" / "docking_500" / "pocket_definition.json"
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"

# Output paths
CACHE_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
LIGANDS_DIR = CACHE_DIR / "ligands_pdbqt"
POSES_DIR = CACHE_DIR / "poses"

for d in [CACHE_DIR, LIGANDS_DIR, POSES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_CPUS = min(cpu_count(), 8)
EXHAUSTIVENESS = 8
NUM_MODES = 5
ZAP70_ID = "CHEMBL2803"


# ── Data loading ─────────────────────────────────────────────────

def load_zap70_molecules():
    """Load all ChEMBL ZAP70 molecules with averaged pIC50 values."""
    raw = pd.read_csv(RAW_FILE)
    zap = raw[raw["target_chembl_id"] == ZAP70_ID].copy()
    mol_data = zap.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    print(f"  ZAP70 molecules: {len(mol_data)}")
    print(f"  pIC50 range: {mol_data['pIC50'].min():.2f} - {mol_data['pIC50'].max():.2f} "
          f"(mean={mol_data['pIC50'].mean():.2f}, std={mol_data['pIC50'].std():.2f})")
    return mol_data


# ── Pocket definition ────────────────────────────────────────────

def load_pocket():
    """Load pre-computed pocket center and box size from docking_500."""
    if not POCKET_JSON.exists():
        print(f"ERROR: Pocket definition not found: {POCKET_JSON}")
        print("Run experiments/run_docking_500.py first to prepare the receptor.")
        sys.exit(1)
    pocket = json.loads(POCKET_JSON.read_text())
    center = np.array(pocket["center"])
    box_size = np.array(pocket["box_size"])
    print(f"  Pocket center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  Box size: ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f})")
    return center, box_size


# ── Ligand preparation (same as run_docking_500.py) ──────────────

def prepare_ligand(smiles, mol_id):
    """Convert SMILES to 3D PDBQT via meeko, cache result. Returns path or None."""
    pdbqt_path = LIGANDS_DIR / f"{mol_id}.pdbqt"
    if pdbqt_path.exists():
        return pdbqt_path

    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = 42
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        params.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol, params)
        if result == -1:
            return None

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
        mol_setups = prep.prepare(mol)
        pdbqt_str, is_ok, _ = PDBQTWriterLegacy.write_string(mol_setups[0])
        if is_ok:
            pdbqt_path.write_text(pdbqt_str)
            return pdbqt_path
    except Exception:
        pass
    return None


# ── PDBQT score parsing ─────────────────────────────────────────

def parse_pdbqt_scores(pose_path):
    """Parse VINA RESULT, INTER, and INTRA energies from pose PDBQT REMARK lines.

    Returns dict with vina_score, vina_inter, vina_intra, and all_scores list.
    """
    result = {
        "vina_score": None,
        "vina_inter": None,
        "vina_intra": None,
        "vina_scores_all": [],
    }
    if not pose_path.exists():
        return result

    scores = []
    inter_energies = []
    intra_energies = []

    for line in pose_path.read_text().split("\n"):
        # Parse VINA RESULT lines: "REMARK VINA RESULT:   -8.3      0.000      0.000"
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:":
                    try:
                        scores.append(float(parts[i + 1]))
                    except (IndexError, ValueError):
                        pass
                    break

        # Parse INTER + INTRA energy: "REMARK  INTER:  -10.1234"
        if "REMARK" in line:
            stripped = line.strip()
            if "INTER:" in stripped:
                try:
                    inter_energies.append(float(stripped.split("INTER:")[-1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass
            elif "INTRA:" in stripped:
                try:
                    intra_energies.append(float(stripped.split("INTRA:")[-1].strip().split()[0]))
                except (ValueError, IndexError):
                    pass

    if scores:
        result["vina_score"] = scores[0]  # Best pose
        result["vina_scores_all"] = scores
    if inter_energies:
        result["vina_inter"] = inter_energies[0]  # Best pose
    if intra_energies:
        result["vina_intra"] = intra_energies[0]  # Best pose

    return result


# ── Docking ──────────────────────────────────────────────────────

def dock_single(args):
    """Dock a single ligand. Called by multiprocessing pool."""
    mol_id, smiles, chembl_id, pIC50, receptor_pdbqt, center, box_size = args

    pose_path = POSES_DIR / f"{mol_id}_pose.pdbqt"
    score_cache = POSES_DIR / f"{mol_id}_scores.json"

    result = {
        "mol_id": mol_id,
        "chembl_id": chembl_id,
        "smiles": smiles,
        "pIC50_exp": pIC50,
        "vina_score": None,
        "vina_inter": None,
        "vina_intra": None,
        "vina_scores_all": [],
        "dock_time_s": 0,
        "success": False,
        "error": None,
    }

    # Check cache
    if score_cache.exists():
        cached = json.loads(score_cache.read_text())
        result.update(cached)
        result["success"] = cached.get("vina_score") is not None
        return result

    # Prepare ligand
    ligand_pdbqt = prepare_ligand(smiles, mol_id)
    if ligand_pdbqt is None:
        result["error"] = "ligand_prep_failed"
        score_cache.write_text(json.dumps({
            k: result[k] for k in [
                "mol_id", "chembl_id", "vina_score", "vina_inter", "vina_intra",
                "vina_scores_all", "dock_time_s", "success", "error",
            ]
        }, indent=2))
        return result

    # Run Vina
    cmd = [
        str(VINA_BIN),
        "--receptor", str(receptor_pdbqt),
        "--ligand", str(ligand_pdbqt),
        "--center_x", f"{center[0]:.3f}",
        "--center_y", f"{center[1]:.3f}",
        "--center_z", f"{center[2]:.3f}",
        "--size_x", f"{box_size[0]:.1f}",
        "--size_y", f"{box_size[1]:.1f}",
        "--size_z", f"{box_size[2]:.1f}",
        "--exhaustiveness", str(EXHAUSTIVENESS),
        "--num_modes", str(NUM_MODES),
        "--out", str(pose_path),
    ]

    t0 = time.time()
    try:
        proc = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        result["dock_time_s"] = time.time() - t0

        # Parse scores from PDBQT REMARK lines (most reliable)
        parsed = parse_pdbqt_scores(pose_path)
        result["vina_score"] = parsed["vina_score"]
        result["vina_inter"] = parsed["vina_inter"]
        result["vina_intra"] = parsed["vina_intra"]
        result["vina_scores_all"] = parsed["vina_scores_all"]

        if result["vina_score"] is not None:
            result["success"] = True

        if proc.returncode != 0 and not result["success"]:
            result["error"] = f"vina_exit_{proc.returncode}"

    except subprocess.TimeoutExpired:
        result["dock_time_s"] = time.time() - t0
        result["error"] = "timeout_600s"
    except Exception as e:
        result["dock_time_s"] = time.time() - t0
        result["error"] = str(e)

    # Cache individual result
    score_cache.write_text(json.dumps({
        k: result[k] for k in [
            "mol_id", "chembl_id", "vina_score", "vina_inter", "vina_intra",
            "vina_scores_all", "dock_time_s", "success", "error",
        ]
    }, indent=2))

    return result


# ── Main ─────────────────────────────────────────────────────────

def main():
    print(f"{'='*70}")
    print(f"  AutoDock Vina Docking: ChEMBL ZAP70 ({ZAP70_ID}) Molecules")
    print(f"  PDB: 4K2R | CPUs: {N_CPUS} | Exhaustiveness: {EXHAUSTIVENESS}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"{'='*70}\n")

    # Verify prerequisites
    if not RECEPTOR_PDBQT.exists():
        print(f"ERROR: Receptor PDBQT not found: {RECEPTOR_PDBQT}")
        print("Run experiments/run_docking_500.py first to prepare the receptor.")
        sys.exit(1)
    if not VINA_BIN.exists():
        print(f"ERROR: Vina binary not found: {VINA_BIN}")
        sys.exit(1)

    # Load molecules
    print("--- Loading ZAP70 molecules ---")
    mol_data = load_zap70_molecules()

    # Load pocket
    print("\n--- Pocket Definition (from docking_500) ---")
    center, box_size = load_pocket()

    # Count already-docked
    n_cached = sum(1 for _, row in mol_data.iterrows()
                   if (POSES_DIR / f"{row['molecule_chembl_id']}_scores.json").exists())
    remaining = len(mol_data) - n_cached
    est_time = remaining * 6.0 / N_CPUS
    print(f"\nAlready cached: {n_cached}/{len(mol_data)}")
    print(f"Remaining: {remaining}, estimated time: {est_time/60:.1f} min ({N_CPUS} CPUs)")

    # Prepare args
    args_list = []
    for _, row in mol_data.iterrows():
        mol_id = row["molecule_chembl_id"]
        args_list.append((
            mol_id,
            row["smiles"],
            mol_id,
            float(row["pIC50"]),
            str(RECEPTOR_PDBQT),
            center.tolist(),
            box_size.tolist(),
        ))

    # Run docking in parallel
    print(f"\n--- Docking {len(args_list)} molecules ({N_CPUS} parallel workers) ---")
    t_start = time.time()

    results = []
    n_done = 0
    with Pool(processes=N_CPUS) as pool:
        for result in pool.imap_unordered(dock_single, args_list):
            results.append(result)
            n_done += 1
            if n_done % 50 == 0 or n_done == len(args_list):
                n_ok = sum(1 for r in results if r["success"])
                elapsed = time.time() - t_start
                rate = n_done / elapsed if elapsed > 0 else 0
                eta = (len(args_list) - n_done) / rate if rate > 0 else 0
                best = min((r["vina_score"] for r in results if r["vina_score"] is not None), default=0)
                print(f"  [{n_done}/{len(args_list)}] "
                      f"success={n_ok} "
                      f"best={best:.1f} kcal/mol "
                      f"rate={rate:.1f}/s "
                      f"ETA={eta/60:.1f}min")

    total_time = time.time() - t_start
    n_success = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_success

    print(f"\n--- Results ---")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Success: {n_success}/{len(results)} ({100*n_success/len(results):.0f}%)")
    print(f"Failed: {n_fail}")

    # ── Output 1: CSV ────────────────────────────────────────────
    successful = [r for r in results if r["success"]]
    failed = [r for r in results if not r["success"]]

    if n_success > 0:
        scores = [r["vina_score"] for r in successful]
        print(f"\nVina scores: {np.mean(scores):.2f} +/- {np.std(scores):.2f} kcal/mol")
        print(f"  Best: {min(scores):.2f}, Worst: {max(scores):.2f}")

    csv_rows = []
    for r in results:
        csv_rows.append({
            "smiles": r["smiles"],
            "chembl_id": r["chembl_id"],
            "pIC50_exp": r["pIC50_exp"],
            "vina_score": r["vina_score"],
            "vina_inter": r["vina_inter"],
            "vina_intra": r["vina_intra"],
            "dock_time_s": round(r["dock_time_s"], 2),
            "success": r["success"],
        })
    df_out = pd.DataFrame(csv_rows)
    csv_path = CACHE_DIR / "docking_results.csv"
    df_out.to_csv(csv_path, index=False)
    print(f"\nCSV: {csv_path}")

    # ── Output 2: Full JSON ──────────────────────────────────────
    full_json_path = CACHE_DIR / "docking_results_full.json"
    full_json_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"Full JSON: {full_json_path}")

    # ── Correlation analysis ─────────────────────────────────────
    if len(successful) > 10:
        from scipy import stats

        vina_arr = np.array([r["vina_score"] for r in successful])
        pic50_arr = np.array([r["pIC50_exp"] for r in successful])

        spearman_r, sp_p = stats.spearmanr(vina_arr, pic50_arr)
        pearson_r, pe_p = stats.pearsonr(vina_arr, pic50_arr)

        print(f"\n{'='*50}")
        print(f"Correlation: Vina Score vs Experimental pIC50")
        print(f"  N molecules (docked): {len(successful)}")
        print(f"  Spearman rho: {spearman_r:.4f} (p={sp_p:.2e})")
        print(f"  Pearson r:    {pearson_r:.4f} (p={pe_p:.2e})")
        print(f"  (Negative correlation = agreement: lower Vina = better binding = higher pIC50)")
        print(f"{'='*50}")

        # Also report INTER energy correlation (often better predictor)
        inter_arr = np.array([r["vina_inter"] for r in successful if r["vina_inter"] is not None])
        if len(inter_arr) > 10:
            inter_pic50 = np.array([r["pIC50_exp"] for r in successful if r["vina_inter"] is not None])
            inter_sp, inter_sp_p = stats.spearmanr(inter_arr, inter_pic50)
            inter_pe, inter_pe_p = stats.pearsonr(inter_arr, inter_pic50)
            print(f"\nINTER energy correlation:")
            print(f"  Spearman rho: {inter_sp:.4f} (p={inter_sp_p:.2e})")
            print(f"  Pearson r:    {inter_pe:.4f} (p={inter_pe_p:.2e})")

        # Error breakdown
        if failed:
            error_counts = {}
            for r in failed:
                err = r.get("error", "unknown")
                error_counts[err] = error_counts.get(err, 0) + 1
            print(f"\nFailure breakdown:")
            for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
                print(f"  {err}: {cnt}")

    print(f"\n{'='*70}")
    print(f"DOCKING COMPLETE")
    print(f"  Output dir: {CACHE_DIR}")
    print(f"  Ligand PDBQTs: {LIGANDS_DIR} ({len(list(LIGANDS_DIR.glob('*.pdbqt')))} files)")
    print(f"  Poses: {POSES_DIR} ({len(list(POSES_DIR.glob('*_pose.pdbqt')))} files)")
    print(f"  Scores cache: {POSES_DIR} ({len(list(POSES_DIR.glob('*_scores.json')))} files)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
