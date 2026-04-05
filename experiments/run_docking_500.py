#!/usr/bin/env python3
"""Dock 509 expert-panel candidates against ZAP70 (PDB 4K2R) using AutoDock Vina.

Uses multiprocessing for parallel docking. Caches all intermediate files
(ligand PDBQT, output poses, scores) for downstream reprocessing.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_docking_500.py
"""
import json
import os
import subprocess
import sys
import tempfile
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
CACHE_DIR = PROJECT_ROOT / "data" / "docking_500"
LIGANDS_DIR = CACHE_DIR / "ligands_pdbqt"
POSES_DIR = CACHE_DIR / "poses"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

for d in [CACHE_DIR, LIGANDS_DIR, POSES_DIR]:
    d.mkdir(parents=True, exist_ok=True)

N_CPUS = min(cpu_count(), 8)  # Use up to 8 cores
EXHAUSTIVENESS = 8
NUM_MODES = 5  # Save top 5 poses per molecule for later analysis
PDB_ID = "4K2R"  # ZAP70 kinase structure


# ── Protein preparation ──────────────────────────────────────────

def download_pdb(pdb_id):
    """Download PDB from RCSB, cache locally."""
    pdb_path = CACHE_DIR / f"{pdb_id}.pdb"
    if pdb_path.exists():
        print(f"PDB {pdb_id} already cached")
        return pdb_path
    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, pdb_path)
    print(f"Saved: {pdb_path}")
    return pdb_path


def prepare_receptor(pdb_path):
    """Prepare receptor PDBQT and define binding pocket from co-crystal ligand."""
    receptor_pdbqt = CACHE_DIR / "receptor.pdbqt"
    pocket_json = CACHE_DIR / "pocket_definition.json"

    if receptor_pdbqt.exists() and pocket_json.exists():
        pocket = json.loads(pocket_json.read_text())
        center = np.array(pocket["center"])
        box_size = np.array(pocket["box_size"])
        print(f"Receptor already prepared. Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
        return receptor_pdbqt, center, box_size

    # Parse PDB: find ligand atoms for pocket, keep protein ATOM lines
    ligand_atoms = []
    protein_lines = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("ATOM"):
                protein_lines.append(line)
            elif line.startswith("HETATM"):
                resname = line[17:20].strip()
                if resname not in ("HOH", "SO4", "GOL", "EDO", "PEG", "CL", "NA", "MG", "ZN", "CA"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ligand_atoms.append([x, y, z])

    if ligand_atoms:
        coords = np.array(ligand_atoms)
        center = coords.mean(axis=0)
        box_size = np.ptp(coords, axis=0) + 12.0  # 12A padding
        box_size = np.maximum(box_size, 24.0)
        print(f"  Pocket from {len(ligand_atoms)} ligand atoms")
    else:
        # Fallback: use ATP-binding site region from known coordinates
        center = np.array([10.0, 25.0, 15.0])  # approximate for 4K2R
        box_size = np.array([25.0, 25.0, 25.0])
        print("  WARNING: No co-crystal ligand found, using default ATP site")

    print(f"  Center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  Box: ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f})")

    # Write clean protein PDB
    clean_pdb = CACHE_DIR / "receptor_clean.pdb"
    with open(clean_pdb, "w") as f:
        for line in protein_lines:
            f.write(line)
        f.write("END\n")

    # Convert to PDBQT
    _pdb_to_pdbqt(clean_pdb, receptor_pdbqt)

    # Cache pocket definition
    pocket_json.write_text(json.dumps({
        "center": center.tolist(),
        "box_size": box_size.tolist(),
        "pdb_id": PDB_ID,
        "n_ligand_atoms": len(ligand_atoms),
    }, indent=2))

    return receptor_pdbqt, center, box_size


def _pdb_to_pdbqt(pdb_path, pdbqt_path):
    """Simple PDB to PDBQT: add Gasteiger charges placeholder + AD4 types."""
    ad_type_map = {
        "C": "C", "CA": "C", "CB": "C", "CG": "C", "CD": "C", "CE": "C", "CZ": "C",
        "CG1": "C", "CG2": "C", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C",
        "N": "NA", "NE": "NA", "NH1": "N", "NH2": "N", "NZ": "N",
        "ND1": "NA", "ND2": "N", "NE1": "NA", "NE2": "NA",
        "O": "OA", "OG": "OA", "OG1": "OA", "OD1": "OA", "OD2": "OA",
        "OE1": "OA", "OE2": "OA", "OH": "OA", "OXT": "OA",
        "S": "SA", "SG": "SA", "SD": "SA",
        "H": "HD",
    }
    lines = []
    with open(pdb_path) as f:
        for line in f:
            if not line.startswith("ATOM"):
                if line.startswith("END"):
                    lines.append(line.rstrip())
                continue
            atom_name = line[12:16].strip()
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            ad_type = ad_type_map.get(atom_name, ad_type_map.get(element, "C"))
            pdbqt_line = line[:54] + "  0.00  0.00"
            pdbqt_line = pdbqt_line.ljust(77) + f" {ad_type:<2s}"
            lines.append(pdbqt_line.rstrip())
    with open(pdbqt_path, "w") as f:
        f.write("\n".join(lines) + "\n")
    print(f"  Receptor PDBQT: {pdbqt_path} ({len(lines)} atoms)")


# ── Ligand preparation ───────────────────────────────────────────

def prepare_ligand(smiles, mol_id):
    """Convert SMILES to 3D PDBQT, cache result. Returns path or None."""
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

    # Convert to PDBQT via meeko
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


# ── Docking ──────────────────────────────────────────────────────

def dock_single(args):
    """Dock a single ligand. Called by multiprocessing pool."""
    mol_id, smiles, pIC50, source, receptor_pdbqt, center, box_size = args

    pose_path = POSES_DIR / f"{mol_id}_pose.pdbqt"
    result = {
        "mol_id": mol_id,
        "smiles": smiles,
        "pIC50": pIC50,
        "source": source,
        "vina_score": None,
        "vina_scores_all": [],
        "dock_time_s": 0,
        "success": False,
        "error": None,
    }

    # Skip if already docked
    score_cache = POSES_DIR / f"{mol_id}_scores.json"
    if score_cache.exists():
        cached = json.loads(score_cache.read_text())
        result.update(cached)
        result["success"] = cached.get("vina_score") is not None
        return result

    # Prepare ligand
    ligand_pdbqt = prepare_ligand(smiles, mol_id)
    if ligand_pdbqt is None:
        result["error"] = "ligand_prep_failed"
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

        # Parse scores from output PDBQT (most reliable)
        scores = []
        if pose_path.exists():
            for line in pose_path.read_text().split("\n"):
                if "VINA RESULT" in line:
                    parts = line.split()
                    for i, p in enumerate(parts):
                        if p == "RESULT:":
                            try:
                                scores.append(float(parts[i + 1]))
                            except (IndexError, ValueError):
                                pass

        if scores:
            result["vina_score"] = scores[0]  # Best score
            result["vina_scores_all"] = scores
            result["success"] = True

        if proc.returncode != 0 and not scores:
            result["error"] = f"vina_exit_{proc.returncode}"

    except subprocess.TimeoutExpired:
        result["dock_time_s"] = time.time() - t0
        result["error"] = "timeout_600s"
    except Exception as e:
        result["dock_time_s"] = time.time() - t0
        result["error"] = str(e)

    # Cache individual result
    score_cache.write_text(json.dumps({
        "mol_id": result["mol_id"],
        "vina_score": result["vina_score"],
        "vina_scores_all": result["vina_scores_all"],
        "dock_time_s": result["dock_time_s"],
        "success": result["success"],
        "error": result["error"],
    }, indent=2))

    return result


# ── Main ─────────────────────────────────────────────────────────

def main():
    print(f"{'='*70}")
    print(f"  AutoDock Vina Docking: 509 ZAP70 Candidates")
    print(f"  PDB: {PDB_ID} | CPUs: {N_CPUS} | Exhaustiveness: {EXHAUSTIVENESS}")
    print(f"  Cache: {CACHE_DIR}")
    print(f"{'='*70}\n")

    # Load candidates
    candidates_path = Path("/tmp/candidates_500_props.json")
    if not candidates_path.exists():
        # Try cached copy
        candidates_path = CACHE_DIR / "candidates_input.json"
    if not candidates_path.exists():
        print("ERROR: No candidates file found. Run the expert panel pipeline first.")
        sys.exit(1)

    candidates = json.loads(candidates_path.read_text())
    print(f"Loaded {len(candidates)} candidates")

    # Cache a copy of the input
    input_cache = CACHE_DIR / "candidates_input.json"
    if not input_cache.exists():
        input_cache.write_text(json.dumps(candidates, indent=2))

    # Prepare receptor
    print(f"\n--- Receptor Preparation (PDB {PDB_ID}) ---")
    pdb_path = download_pdb(PDB_ID)
    receptor_pdbqt, center, box_size = prepare_receptor(pdb_path)

    # Count already-docked
    n_cached = 0
    for i in range(len(candidates)):
        if (POSES_DIR / f"mol_{i}_scores.json").exists():
            n_cached += 1
    print(f"\nAlready cached: {n_cached}/{len(candidates)}")
    remaining = len(candidates) - n_cached
    est_time = remaining * 6.0 / N_CPUS  # ~6s per mol
    print(f"Remaining: {remaining}, estimated time: {est_time/60:.1f} min ({N_CPUS} CPUs)")

    # Prepare args for parallel execution
    args_list = []
    for i, c in enumerate(candidates):
        mol_id = f"mol_{i}"
        args_list.append((
            mol_id,
            c["smiles"],
            c.get("pIC50", 0),
            c.get("source", "unknown"),
            str(receptor_pdbqt),
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
            if n_done % 25 == 0 or n_done == len(args_list):
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

    if n_success > 0:
        scores = [r["vina_score"] for r in results if r["vina_score"] is not None]
        print(f"Vina scores: {np.mean(scores):.2f} +/- {np.std(scores):.2f} kcal/mol")
        print(f"  Best: {min(scores):.2f}, Worst: {max(scores):.2f}")

        # Per-source breakdown
        for src in ["Mol2Mol", "LibInvent", "DeNovo"]:
            src_scores = [r["vina_score"] for r in results if r["source"] == src and r["vina_score"] is not None]
            if src_scores:
                print(f"  {src}: {np.mean(src_scores):.2f} +/- {np.std(src_scores):.2f} (n={len(src_scores)})")

    # Save comprehensive results
    # 1. Full results JSON (all fields)
    full_results_path = CACHE_DIR / "docking_results_full.json"
    full_results_path.write_text(json.dumps(results, indent=2, default=str))
    print(f"\nFull results: {full_results_path}")

    # 2. Summary CSV for easy analysis
    df = pd.DataFrame(results)
    csv_path = CACHE_DIR / "docking_results.csv"
    df.to_csv(csv_path, index=False)
    print(f"CSV: {csv_path}")

    # 3. Summary JSON for report integration
    summary = {
        "pdb_id": PDB_ID,
        "n_candidates": len(candidates),
        "n_docked": n_success,
        "n_failed": n_fail,
        "total_time_min": total_time / 60,
        "time_per_mol_s": total_time / len(candidates),
        "n_cpus": N_CPUS,
        "exhaustiveness": EXHAUSTIVENESS,
        "num_modes": NUM_MODES,
        "vina_score_mean": float(np.mean(scores)) if scores else None,
        "vina_score_std": float(np.std(scores)) if scores else None,
        "vina_score_best": float(min(scores)) if scores else None,
        "vina_score_worst": float(max(scores)) if scores else None,
        "pocket_center": center.tolist(),
        "pocket_box_size": box_size.tolist(),
    }

    # Top 20 by Vina score
    successful = [r for r in results if r["success"]]
    successful.sort(key=lambda x: x["vina_score"])
    summary["top_20_by_vina"] = [
        {"smiles": r["smiles"], "vina_score": r["vina_score"],
         "pIC50": r["pIC50"], "source": r["source"]}
        for r in successful[:20]
    ]

    # Correlation with pIC50
    if len(successful) > 10:
        from scipy import stats
        vina_arr = np.array([r["vina_score"] for r in successful])
        pic50_arr = np.array([r["pIC50"] for r in successful])
        spearman, sp_p = stats.spearmanr(vina_arr, pic50_arr)
        pearson, pe_p = stats.pearsonr(vina_arr, pic50_arr)
        summary["correlation"] = {
            "spearman_r": float(spearman),
            "spearman_p": float(sp_p),
            "pearson_r": float(pearson),
            "pearson_p": float(pe_p),
            "note": "Negative Vina = better binding; negative correlation with pIC50 = agreement",
        }
        print(f"\nCorrelation (Vina vs pIC50):")
        print(f"  Spearman: {spearman:.3f} (p={sp_p:.2e})")
        print(f"  Pearson:  {pearson:.3f} (p={pe_p:.2e})")

    summary_path = RESULTS_DIR / "docking_500_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"Summary: {summary_path}")

    # 4. Per-source summary
    print(f"\n{'='*70}")
    print(f"DOCKING COMPLETE")
    print(f"  Cached at: {CACHE_DIR}")
    print(f"  Ligand PDBQTs: {LIGANDS_DIR} ({len(list(LIGANDS_DIR.glob('*.pdbqt')))} files)")
    print(f"  Poses: {POSES_DIR} ({len(list(POSES_DIR.glob('*_pose.pdbqt')))} files)")
    print(f"  Scores: {POSES_DIR} ({len(list(POSES_DIR.glob('*_scores.json')))} files)")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
