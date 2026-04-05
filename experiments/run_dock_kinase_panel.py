#!/usr/bin/env python3
"""Dock ALL unique molecules from the kinase pretraining panel against ZAP70 (PDB 4K2R).

Loads molecules from 9 kinase targets (ZAP70 + 8 panel kinases), deduplicates
by SMILES, and docks against the ZAP70 pocket using AutoDock Vina.

Reuses receptor PDBQT and pocket definition from the existing docking_500 run.
Parses Vina scores from PDBQT REMARK lines (not stdout) for reliability.
Computes interaction fingerprints and creates a unified embedding cache.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_dock_kinase_panel.py
"""
import gc
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
sys.path.insert(0, str(PROJECT_ROOT))

VINA_BIN = PROJECT_ROOT / "tools" / "vina"

# Input paths — reuse receptor from docking_500
RECEPTOR_PDBQT = PROJECT_ROOT / "data" / "docking_500" / "receptor.pdbqt"
POCKET_JSON = PROJECT_ROOT / "data" / "docking_500" / "pocket_definition.json"
RAW_FILE = PROJECT_ROOT / "data" / "overlapping_assays" / "molecule_pIC50_minimal.csv"

# Output paths
CACHE_DIR = PROJECT_ROOT / "data" / "docking_kinase_panel"
LIGANDS_DIR = CACHE_DIR / "ligands_pdbqt"
POSES_DIR = CACHE_DIR / "poses"

# Unified embedding cache
EMBEDDING_CACHE_DIR = PROJECT_ROOT / "data" / "embedding_cache"

# Existing docking results to merge
ZAP70_DOCK_DIR = PROJECT_ROOT / "data" / "docking_chembl_zap70"
DOCKING_500_DIR = PROJECT_ROOT / "data" / "docking_500"

N_CPUS = min(cpu_count(), 8)
EXHAUSTIVENESS = 8
NUM_MODES = 5

# 9 kinase targets: ZAP70 + 8 panel kinases
KINASE_TARGETS = {
    "ZAP70": "CHEMBL2803",
    "SYK": "CHEMBL2599",
    "LCK": "CHEMBL258",
    "JAK2": "CHEMBL2971",
    "ABL1": "CHEMBL1862",
    "SRC": "CHEMBL267",
    "BTK": "CHEMBL5251",
    "ITK": "CHEMBL3009",
    "FYN": "CHEMBL1841",
}


# ── Data loading ─────────────────────────────────────────────────


def load_kinase_molecules():
    """Load molecules from all kinase targets, deduplicate by SMILES.

    Returns:
        mol_data: DataFrame with columns [molecule_chembl_id, smiles, pIC50_mean, targets]
        per_target_counts: dict mapping target name to molecule count
    """
    raw = pd.read_csv(RAW_FILE)
    target_ids = set(KINASE_TARGETS.values())
    kinase = raw[raw["target_chembl_id"].isin(target_ids)].copy()

    # Reverse map: chembl_id -> target name
    id_to_name = {v: k for k, v in KINASE_TARGETS.items()}

    # Report per-target counts (before dedup)
    per_target_counts = {}
    for name, chembl_id in KINASE_TARGETS.items():
        mask = kinase["target_chembl_id"] == chembl_id
        n_mols = kinase.loc[mask, "molecule_chembl_id"].nunique()
        per_target_counts[name] = n_mols
        print(f"  {name} ({chembl_id}): {n_mols} unique molecules")

    # Build per-molecule aggregated data
    # For each molecule: average pIC50 across all targets, track which targets
    mol_targets = {}
    for _, row in kinase.iterrows():
        mid = row["molecule_chembl_id"]
        tname = id_to_name.get(row["target_chembl_id"], row["target_chembl_id"])
        if mid not in mol_targets:
            mol_targets[mid] = set()
        mol_targets[mid].add(tname)

    mol_agg = kinase.groupby("molecule_chembl_id").agg({
        "smiles": "first",
        "pIC50": "mean",
    }).reset_index()
    mol_agg = mol_agg.rename(columns={"pIC50": "pIC50_mean"})
    mol_agg["targets"] = mol_agg["molecule_chembl_id"].map(
        lambda mid: ",".join(sorted(mol_targets.get(mid, set())))
    )

    # Deduplicate by SMILES (keep the entry with most target annotations)
    mol_agg["n_targets"] = mol_agg["targets"].str.count(",") + 1
    mol_agg = mol_agg.sort_values("n_targets", ascending=False)
    mol_agg = mol_agg.drop_duplicates(subset="smiles", keep="first")
    mol_agg = mol_agg.drop(columns=["n_targets"]).reset_index(drop=True)

    n_multitarget = (mol_agg["targets"].str.contains(",")).sum()
    print(f"\n  Total unique molecules (by SMILES): {len(mol_agg)}")
    print(f"  Multi-target molecules: {n_multitarget}")
    print(f"  pIC50 range: {mol_agg['pIC50_mean'].min():.2f} - "
          f"{mol_agg['pIC50_mean'].max():.2f} "
          f"(mean={mol_agg['pIC50_mean'].mean():.2f})")

    return mol_agg, per_target_counts


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


# ── Ligand preparation ───────────────────────────────────────────


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
        if "VINA RESULT" in line:
            parts = line.split()
            for i, p in enumerate(parts):
                if p == "RESULT:":
                    try:
                        scores.append(float(parts[i + 1]))
                    except (IndexError, ValueError):
                        pass
                    break

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
        result["vina_score"] = scores[0]
        result["vina_scores_all"] = scores
    if inter_energies:
        result["vina_inter"] = inter_energies[0]
    if intra_energies:
        result["vina_intra"] = intra_energies[0]

    return result


# ── Check existing ZAP70 docking cache ───────────────────────────


def load_existing_zap70_cache():
    """Load already-docked results from run_dock_chembl_zap70.py if available.

    Returns dict mapping molecule_chembl_id -> score dict, and set of cached SMILES.
    """
    cached_by_id = {}
    cached_smiles = {}

    # Check ZAP70 docking CSV
    zap70_csv = ZAP70_DOCK_DIR / "docking_results.csv"
    if zap70_csv.exists():
        df = pd.read_csv(zap70_csv)
        for _, row in df.iterrows():
            if row.get("success", False):
                mid = row.get("chembl_id", "")
                smi = row.get("smiles", "")
                cached_by_id[mid] = {
                    "vina_score": row.get("vina_score"),
                    "vina_inter": row.get("vina_inter"),
                    "vina_intra": row.get("vina_intra"),
                }
                if smi:
                    cached_smiles[smi] = cached_by_id[mid]
        print(f"  Loaded {len(cached_by_id)} cached ZAP70 docking results")

    return cached_by_id, cached_smiles


# ── Docking ──────────────────────────────────────────────────────


def dock_single(args):
    """Dock a single ligand. Called by multiprocessing pool."""
    mol_id, smiles, targets, pIC50_mean, receptor_pdbqt, center, box_size = args

    pose_path = POSES_DIR / f"{mol_id}_pose.pdbqt"
    score_cache = POSES_DIR / f"{mol_id}_scores.json"

    result = {
        "mol_id": mol_id,
        "smiles": smiles,
        "targets": targets,
        "pIC50_mean": pIC50_mean,
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
        try:
            cached = json.loads(score_cache.read_text())
            result.update(cached)
            result["success"] = cached.get("vina_score") is not None
            # Restore fields that may not be in cache
            result["smiles"] = smiles
            result["targets"] = targets
            result["pIC50_mean"] = pIC50_mean
            return result
        except (json.JSONDecodeError, Exception):
            pass

    # Also check if pose exists from ZAP70-only run (reuse ligand PDBQTs too)
    zap70_pose = ZAP70_DOCK_DIR / "poses" / f"{mol_id}_pose.pdbqt"
    zap70_score = ZAP70_DOCK_DIR / "poses" / f"{mol_id}_scores.json"
    if zap70_score.exists():
        try:
            cached = json.loads(zap70_score.read_text())
            if cached.get("vina_score") is not None:
                result["vina_score"] = cached["vina_score"]
                result["vina_inter"] = cached.get("vina_inter")
                result["vina_intra"] = cached.get("vina_intra")
                result["vina_scores_all"] = cached.get("vina_scores_all", [])
                result["dock_time_s"] = cached.get("dock_time_s", 0)
                result["success"] = True
                # Copy pose file if it exists
                if zap70_pose.exists() and not pose_path.exists():
                    import shutil
                    shutil.copy2(str(zap70_pose), str(pose_path))
                # Save to our cache
                _save_score_cache(score_cache, result)
                return result
        except (json.JSONDecodeError, Exception):
            pass

    # Also check ZAP70-only ligand PDBQT dir for reuse
    zap70_ligand = ZAP70_DOCK_DIR / "ligands_pdbqt" / f"{mol_id}.pdbqt"
    our_ligand = LIGANDS_DIR / f"{mol_id}.pdbqt"
    if zap70_ligand.exists() and not our_ligand.exists():
        import shutil
        shutil.copy2(str(zap70_ligand), str(our_ligand))

    # Prepare ligand
    ligand_pdbqt = prepare_ligand(smiles, mol_id)
    if ligand_pdbqt is None:
        result["error"] = "ligand_prep_failed"
        _save_score_cache(score_cache, result)
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

    _save_score_cache(score_cache, result)
    return result


def _save_score_cache(score_cache, result):
    """Save individual molecule docking result to JSON cache."""
    cache_keys = [
        "mol_id", "vina_score", "vina_inter", "vina_intra",
        "vina_scores_all", "dock_time_s", "success", "error",
    ]
    score_cache.write_text(json.dumps(
        {k: result[k] for k in cache_keys if k in result},
        indent=2, default=str,
    ))


# ── Interaction features ─────────────────────────────────────────


def compute_interaction_features_for_panel(mol_ids):
    """Compute interaction fingerprints for all successfully docked molecules."""
    from src.data.utils.interaction_features import compute_all_interaction_features

    cache_path = str(CACHE_DIR / "interaction_features.npz")

    print(f"\n--- Computing interaction features for {len(mol_ids)} molecules ---")
    features = compute_all_interaction_features(
        poses_dir=str(POSES_DIR),
        receptor_path=str(RECEPTOR_PDBQT),
        mol_ids=mol_ids,
        pose_filename_template="{mol_id}_pose.pdbqt",
        cache_path=cache_path,
    )

    n_valid = sum(1 for v in features.values() if not np.all(np.isnan(v)))
    print(f"  Valid interaction features: {n_valid}/{len(mol_ids)}")
    return features


# ── Unified embedding cache ──────────────────────────────────────


def create_unified_cache(panel_results, panel_interaction_feats):
    """Merge panel results with existing ZAP70 and docking_500 results.

    Creates unified caches at:
      - data/embedding_cache/docking_vina_scores.npz
      - data/embedding_cache/docking_interaction_fps.npz
    """
    EMBEDDING_CACHE_DIR.mkdir(parents=True, exist_ok=True)

    # Collect all Vina scores: {smiles: [vina_score, vina_inter, vina_intra, vina_std]}
    vina_cache = {}
    ifp_cache = {}

    # 1. Panel results
    for r in panel_results:
        if r["success"] and r["vina_score"] is not None:
            smi = r["smiles"]
            scores_all = r.get("vina_scores_all", [])
            vina_std = float(np.std(scores_all)) if len(scores_all) > 1 else 0.0
            vina_cache[smi] = np.array([
                r["vina_score"],
                r["vina_inter"] if r["vina_inter"] is not None else np.nan,
                r["vina_intra"] if r["vina_intra"] is not None else np.nan,
                vina_std,
            ], dtype=np.float32)

    # Panel interaction features (keyed by mol_id, need to map to SMILES)
    panel_id_to_smiles = {r["mol_id"]: r["smiles"] for r in panel_results}
    for mol_id, feat_vec in panel_interaction_feats.items():
        smi = panel_id_to_smiles.get(mol_id)
        if smi and not np.all(np.isnan(feat_vec)):
            ifp_cache[smi] = feat_vec

    # 2. Existing ZAP70-only docking results
    zap70_csv = ZAP70_DOCK_DIR / "docking_results.csv"
    if zap70_csv.exists():
        df_zap = pd.read_csv(zap70_csv)
        for _, row in df_zap.iterrows():
            smi = row.get("smiles", "")
            if smi and row.get("success", False) and smi not in vina_cache:
                vina_cache[smi] = np.array([
                    row.get("vina_score", np.nan),
                    row.get("vina_inter", np.nan),
                    row.get("vina_intra", np.nan),
                    0.0,  # no std available from CSV
                ], dtype=np.float32)
        print(f"  Merged ZAP70-only docking: {len(df_zap)} entries")

    # 3. Docking_500 (generated molecules)
    d500_csv = DOCKING_500_DIR / "docking_results.csv"
    if d500_csv.exists():
        df_500 = pd.read_csv(d500_csv)
        for _, row in df_500.iterrows():
            smi = row.get("smiles", "")
            if smi and row.get("success", False) and smi not in vina_cache:
                scores_str = row.get("vina_scores_all", "")
                try:
                    scores_list = json.loads(scores_str.replace("'", '"')) if isinstance(scores_str, str) else []
                except (json.JSONDecodeError, AttributeError):
                    scores_list = []
                vina_std = float(np.std(scores_list)) if len(scores_list) > 1 else 0.0
                vina_cache[smi] = np.array([
                    row.get("vina_score", np.nan),
                    np.nan,  # inter not in docking_500 CSV
                    np.nan,  # intra not in docking_500 CSV
                    vina_std,
                ], dtype=np.float32)
        print(f"  Merged docking_500: {len(df_500)} entries")

    # Save unified caches
    vina_path = EMBEDDING_CACHE_DIR / "docking_vina_scores.npz"
    np.savez_compressed(str(vina_path), **{s: v for s, v in vina_cache.items()})
    print(f"  Saved unified Vina scores: {vina_path} ({len(vina_cache)} molecules)")

    ifp_path = EMBEDDING_CACHE_DIR / "docking_interaction_fps.npz"
    np.savez_compressed(str(ifp_path), **{s: v for s, v in ifp_cache.items()})
    print(f"  Saved unified interaction FPs: {ifp_path} ({len(ifp_cache)} molecules)")

    return vina_cache, ifp_cache


# ── Correlation analysis ─────────────────────────────────────────


def compute_correlations(results):
    """Compute per-target and overall correlation between Vina score and pIC50."""
    from scipy import stats

    successful = [r for r in results if r["success"] and r["vina_score"] is not None]
    if len(successful) < 10:
        print("  Too few successful dockings for correlation analysis")
        return {}

    # Overall
    vina_arr = np.array([r["vina_score"] for r in successful])
    pic50_arr = np.array([r["pIC50_mean"] for r in successful])
    sp_r, sp_p = stats.spearmanr(vina_arr, pic50_arr)
    pe_r, pe_p = stats.pearsonr(vina_arr, pic50_arr)

    print(f"\n{'='*60}")
    print(f"Overall Correlation: Vina Score vs Experimental pIC50")
    print(f"  N molecules: {len(successful)}")
    print(f"  Spearman rho: {sp_r:.4f} (p={sp_p:.2e})")
    print(f"  Pearson r:    {pe_r:.4f} (p={pe_p:.2e})")
    print(f"  (Negative = agreement: lower Vina = better binding = higher pIC50)")

    corr_results = {
        "overall": {
            "n": len(successful),
            "spearman_rho": float(sp_r),
            "spearman_p": float(sp_p),
            "pearson_r": float(pe_r),
            "pearson_p": float(pe_p),
        },
        "per_target": {},
    }

    # INTER energy correlation
    inter_ok = [r for r in successful if r.get("vina_inter") is not None]
    if len(inter_ok) > 10:
        inter_arr = np.array([r["vina_inter"] for r in inter_ok])
        inter_pic50 = np.array([r["pIC50_mean"] for r in inter_ok])
        inter_sp, _ = stats.spearmanr(inter_arr, inter_pic50)
        inter_pe, _ = stats.pearsonr(inter_arr, inter_pic50)
        print(f"\n  INTER energy correlation:")
        print(f"    Spearman rho: {inter_sp:.4f}")
        print(f"    Pearson r:    {inter_pe:.4f}")
        corr_results["overall"]["inter_spearman"] = float(inter_sp)
        corr_results["overall"]["inter_pearson"] = float(inter_pe)

    # Per-target correlations
    print(f"\n{'='*60}")
    print(f"Per-Target Correlations:")
    print(f"  {'Target':<10} {'N':>6} {'Spr rho':>10} {'Pear r':>10}")
    print(f"  {'-'*40}")

    for tname, chembl_id in KINASE_TARGETS.items():
        target_mols = [r for r in successful if tname in r.get("targets", "")]
        if len(target_mols) < 10:
            print(f"  {tname:<10} {len(target_mols):>6}   (too few)")
            continue

        t_vina = np.array([r["vina_score"] for r in target_mols])
        t_pic50 = np.array([r["pIC50_mean"] for r in target_mols])
        t_sp, t_sp_p = stats.spearmanr(t_vina, t_pic50)
        t_pe, t_pe_p = stats.pearsonr(t_vina, t_pic50)

        print(f"  {tname:<10} {len(target_mols):>6} {t_sp:>10.4f} {t_pe:>10.4f}")
        corr_results["per_target"][tname] = {
            "n": len(target_mols),
            "spearman_rho": float(t_sp),
            "spearman_p": float(t_sp_p),
            "pearson_r": float(t_pe),
            "pearson_p": float(t_pe_p),
        }

    return corr_results


# ── Main ─────────────────────────────────────────────────────────


def main():
    print(f"{'='*70}")
    print(f"  AutoDock Vina Docking: Kinase Panel → ZAP70 Pocket (PDB 4K2R)")
    print(f"  Targets: {len(KINASE_TARGETS)} kinases | CPUs: {N_CPUS} | "
          f"Exhaustiveness: {EXHAUSTIVENESS}")
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

    # Create output directories
    for d in [CACHE_DIR, LIGANDS_DIR, POSES_DIR, EMBEDDING_CACHE_DIR]:
        d.mkdir(parents=True, exist_ok=True)

    # Load molecules
    print("--- Loading kinase panel molecules ---")
    mol_data, per_target_counts = load_kinase_molecules()

    # Load pocket
    print("\n--- Pocket Definition (from docking_500) ---")
    center, box_size = load_pocket()

    # Check existing ZAP70 docking cache
    print("\n--- Checking existing docking caches ---")
    zap70_cached_by_id, zap70_cached_smiles = load_existing_zap70_cache()

    # Count already-docked in our cache
    n_cached = sum(
        1 for _, row in mol_data.iterrows()
        if (POSES_DIR / f"{row['molecule_chembl_id']}_scores.json").exists()
    )
    # Also count those available from ZAP70 cache
    n_from_zap70 = sum(
        1 for _, row in mol_data.iterrows()
        if row["molecule_chembl_id"] in zap70_cached_by_id
        and not (POSES_DIR / f"{row['molecule_chembl_id']}_scores.json").exists()
    )
    remaining = len(mol_data) - n_cached - n_from_zap70
    est_time = remaining * 6.0 / N_CPUS

    print(f"\n  Already cached (panel): {n_cached}")
    print(f"  Reusable from ZAP70 cache: {n_from_zap70}")
    print(f"  Need to dock: {remaining}")
    print(f"  Estimated time: {est_time/60:.1f} min ({N_CPUS} CPUs)")

    # Prepare args
    args_list = []
    for _, row in mol_data.iterrows():
        mol_id = row["molecule_chembl_id"]
        args_list.append((
            mol_id,
            row["smiles"],
            row["targets"],
            float(row["pIC50_mean"]),
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
                best = min(
                    (r["vina_score"] for r in results if r["vina_score"] is not None),
                    default=0,
                )
                print(f"  [{n_done}/{len(args_list)}] "
                      f"success={n_ok} "
                      f"best={best:.1f} kcal/mol "
                      f"rate={rate:.1f}/s "
                      f"ETA={eta/60:.1f}min")

    total_time = time.time() - t_start
    n_success = sum(1 for r in results if r["success"])
    n_fail = len(results) - n_success

    print(f"\n--- Docking Results ---")
    print(f"Total time: {total_time/60:.1f} min")
    print(f"Success: {n_success}/{len(results)} ({100*n_success/len(results):.0f}%)")
    print(f"Failed: {n_fail}")

    successful = [r for r in results if r["success"]]
    if n_success > 0:
        scores = [r["vina_score"] for r in successful]
        print(f"\nVina scores: {np.mean(scores):.2f} +/- {np.std(scores):.2f} kcal/mol")
        print(f"  Best: {min(scores):.2f}, Worst: {max(scores):.2f}")

    # ── Output 1: CSV ────────────────────────────────────────────
    csv_rows = []
    for r in results:
        scores_all = r.get("vina_scores_all", [])
        vina_std = float(np.std(scores_all)) if len(scores_all) > 1 else None
        csv_rows.append({
            "molecule_chembl_id": r["mol_id"],
            "smiles": r["smiles"],
            "targets": r["targets"],
            "pIC50_mean": r["pIC50_mean"],
            "vina_score": r["vina_score"],
            "vina_inter": r["vina_inter"],
            "vina_intra": r["vina_intra"],
            "vina_std": round(vina_std, 4) if vina_std is not None else None,
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

    # ── Per-target statistics ────────────────────────────────────
    print(f"\n--- Per-Target Docking Statistics ---")
    print(f"  {'Target':<10} {'Total':>6} {'Docked':>7} {'Rate':>6} "
          f"{'Mean':>8} {'Std':>7} {'Best':>7}")
    print(f"  {'-'*55}")

    target_stats = {}
    for tname in KINASE_TARGETS:
        target_mols = [r for r in results if tname in r.get("targets", "")]
        target_ok = [r for r in target_mols if r["success"]]
        if target_ok:
            t_scores = [r["vina_score"] for r in target_ok]
            mean_s = np.mean(t_scores)
            std_s = np.std(t_scores)
            best_s = min(t_scores)
            print(f"  {tname:<10} {len(target_mols):>6} {len(target_ok):>7} "
                  f"{100*len(target_ok)/len(target_mols):>5.0f}% "
                  f"{mean_s:>8.2f} {std_s:>7.2f} {best_s:>7.2f}")
            target_stats[tname] = {
                "n_total": len(target_mols),
                "n_docked": len(target_ok),
                "success_rate": len(target_ok) / len(target_mols),
                "vina_mean": float(mean_s),
                "vina_std": float(std_s),
                "vina_best": float(best_s),
            }
        else:
            print(f"  {tname:<10} {len(target_mols):>6} {0:>7}    0%")
            target_stats[tname] = {
                "n_total": len(target_mols),
                "n_docked": 0,
                "success_rate": 0,
            }

    # ── Correlation analysis ─────────────────────────────────────
    corr_results = compute_correlations(results)

    # ── Error breakdown ──────────────────────────────────────────
    failed = [r for r in results if not r["success"]]
    if failed:
        error_counts = {}
        for r in failed:
            err = r.get("error", "unknown")
            error_counts[err] = error_counts.get(err, 0) + 1
        print(f"\nFailure breakdown:")
        for err, cnt in sorted(error_counts.items(), key=lambda x: -x[1]):
            print(f"  {err}: {cnt}")

    # ── Summary JSON ─────────────────────────────────────────────
    summary = {
        "n_targets": len(KINASE_TARGETS),
        "targets": KINASE_TARGETS,
        "n_total_molecules": len(results),
        "n_unique_smiles": len(mol_data),
        "n_success": n_success,
        "n_fail": n_fail,
        "success_rate": n_success / len(results) if results else 0,
        "total_time_s": total_time,
        "per_target": target_stats,
        "correlations": corr_results,
        "per_target_mol_counts": per_target_counts,
    }
    summary_path = CACHE_DIR / "docking_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nSummary JSON: {summary_path}")

    # ── Interaction features ─────────────────────────────────────
    docked_mol_ids = [r["mol_id"] for r in results if r["success"]]
    if docked_mol_ids:
        interaction_feats = compute_interaction_features_for_panel(docked_mol_ids)
    else:
        interaction_feats = {}

    # ── Unified embedding cache ──────────────────────────────────
    print(f"\n--- Creating unified embedding cache ---")
    vina_cache, ifp_cache = create_unified_cache(results, interaction_feats)

    # ── Final summary ────────────────────────────────────────────
    print(f"\n{'='*70}")
    print(f"KINASE PANEL DOCKING COMPLETE")
    print(f"  Output dir: {CACHE_DIR}")
    print(f"  Ligand PDBQTs: {LIGANDS_DIR} "
          f"({len(list(LIGANDS_DIR.glob('*.pdbqt')))} files)")
    print(f"  Poses: {POSES_DIR} "
          f"({len(list(POSES_DIR.glob('*_pose.pdbqt')))} files)")
    print(f"  Score caches: {POSES_DIR} "
          f"({len(list(POSES_DIR.glob('*_scores.json')))} files)")
    print(f"  Interaction features: {CACHE_DIR / 'interaction_features.npz'}")
    print(f"  Unified Vina cache: {EMBEDDING_CACHE_DIR / 'docking_vina_scores.npz'} "
          f"({len(vina_cache)} molecules)")
    print(f"  Unified IFP cache: {EMBEDDING_CACHE_DIR / 'docking_interaction_fps.npz'} "
          f"({len(ifp_cache)} molecules)")
    print(f"{'='*70}")

    # Cleanup
    gc.collect()


if __name__ == "__main__":
    main()
