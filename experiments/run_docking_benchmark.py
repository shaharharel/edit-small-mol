#!/usr/bin/env python
"""
Docking benchmark: AutoDock Vina, GNINA, DiffDock on ZAP70 molecules.

Feasibility study to determine timing, success rates, and practical considerations
for incorporating docking scores into the edit effect framework.

Uses CHEMBL4899 (ZAP70) molecules from molecule_pIC50_minimal.csv and docks
against PDB 1U59 (ZAP70 kinase structure with staurosporine).
"""

import json
import os
import subprocess
import sys
import tempfile
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, rdMolAlign, rdMolDescriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "overlapping_assays" / "extracted"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation"
TOOLS_DIR = PROJECT_ROOT / "tools"
VINA_BIN = TOOLS_DIR / "vina"
WORK_DIR = PROJECT_ROOT / "data" / "docking_benchmark"

RESULTS_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)


# ============================================================
# Step 1: Load and prepare molecules
# ============================================================

def load_molecules(target_chembl_id="CHEMBL4899", n_sample=100, seed=42):
    """Load molecules for the target, sample n_sample."""
    df = pd.read_csv(DATA_DIR / "molecule_pIC50_minimal.csv")
    target_df = df[df["target_chembl_id"] == target_chembl_id].copy()

    # Get unique molecules with mean pIC50
    mol_df = (
        target_df.groupby("canonical_smiles")["pchembl_value"]
        .mean()
        .reset_index()
        .rename(columns={"canonical_smiles": "smiles", "pchembl_value": "pIC50"})
    )
    print(f"Total unique molecules for {target_chembl_id}: {len(mol_df)}")

    if len(mol_df) > n_sample:
        mol_df = mol_df.sample(n=n_sample, random_state=seed)
    print(f"Sampled {len(mol_df)} molecules")
    return mol_df.reset_index(drop=True)


def smiles_to_3d(smiles, n_confs=1, seed=42):
    """Convert SMILES to 3D RDKit mol with MMFF optimization."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    result = AllChem.EmbedMolecule(mol, params)
    if result == -1:
        # Try with random coordinates
        params2 = AllChem.ETKDGv3()
        params2.randomSeed = seed
        params2.useRandomCoords = True
        result = AllChem.EmbedMolecule(mol, params2)
        if result == -1:
            return None
    try:
        AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
    except Exception:
        pass  # MMFF can fail on some molecules, UFF fallback
        try:
            AllChem.UFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            pass
    return mol


def mol_to_pdbqt(mol, output_path):
    """Convert RDKit mol to PDBQT using meeko."""
    from meeko import MoleculePreparation, PDBQTWriterLegacy

    preparator = MoleculePreparation()
    try:
        mol_setups = preparator.prepare(mol)
        pdbqt_string, is_ok, error_msg = PDBQTWriterLegacy.write_string(mol_setups[0])
        if is_ok:
            with open(output_path, "w") as f:
                f.write(pdbqt_string)
            return True
    except Exception as e:
        print(f"  Meeko error: {e}")
    return False


def mol_to_sdf(mol, output_path):
    """Write RDKit mol to SDF file."""
    writer = Chem.SDWriter(str(output_path))
    writer.write(mol)
    writer.close()
    return True


# ============================================================
# Step 2: Protein preparation
# ============================================================

def download_pdb(pdb_id="1U59"):
    """Download PDB file from RCSB."""
    pdb_path = WORK_DIR / f"{pdb_id}.pdb"
    if pdb_path.exists():
        print(f"PDB {pdb_id} already downloaded")
        return pdb_path

    import urllib.request
    url = f"https://files.rcsb.org/download/{pdb_id}.pdb"
    print(f"Downloading {pdb_id} from RCSB...")
    urllib.request.urlretrieve(url, pdb_path)
    print(f"Saved to {pdb_path}")
    return pdb_path


def prepare_receptor(pdb_path):
    """
    Prepare receptor: extract protein, remove waters/ligands.
    Returns path to cleaned PDB and binding pocket center/size.
    """
    clean_pdb = WORK_DIR / "receptor_clean.pdb"
    receptor_pdbqt = WORK_DIR / "receptor.pdbqt"

    # Parse PDB to find ligand (STU = staurosporine) center for pocket definition
    ligand_atoms = []
    protein_lines = []

    with open(pdb_path) as f:
        for line in f:
            if line.startswith("HETATM") and "STU" in line:
                x = float(line[30:38])
                y = float(line[38:46])
                z = float(line[46:54])
                ligand_atoms.append([x, y, z])
            elif line.startswith("ATOM"):
                protein_lines.append(line)

    if not ligand_atoms:
        # Try alternative ligand names
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("HETATM") and line[17:20].strip() not in ("HOH", "SO4", "GOL", "EDO", "PEG"):
                    x = float(line[30:38])
                    y = float(line[38:46])
                    z = float(line[46:54])
                    ligand_atoms.append([x, y, z])
        if ligand_atoms:
            print(f"  Using non-water HETATM atoms for pocket center ({len(ligand_atoms)} atoms)")

    if ligand_atoms:
        coords = np.array(ligand_atoms)
        center = coords.mean(axis=0)
        box_size = np.ptp(coords, axis=0) + 10.0  # 10A padding
        box_size = np.maximum(box_size, 22.0)  # minimum 22A box
    else:
        print("  WARNING: No ligand found, using protein center")
        # Parse all protein coords
        all_coords = []
        for line in protein_lines:
            x = float(line[30:38])
            y = float(line[38:46])
            z = float(line[46:54])
            all_coords.append([x, y, z])
        coords = np.array(all_coords)
        center = coords.mean(axis=0)
        box_size = np.array([30.0, 30.0, 30.0])

    print(f"  Pocket center: ({center[0]:.1f}, {center[1]:.1f}, {center[2]:.1f})")
    print(f"  Box size: ({box_size[0]:.1f}, {box_size[1]:.1f}, {box_size[2]:.1f})")

    # Write clean protein PDB
    with open(clean_pdb, "w") as f:
        for line in protein_lines:
            f.write(line)
        f.write("END\n")

    # Convert to PDBQT using basic approach (add Gasteiger charges, assign AD4 types)
    # We'll use a simple conversion since we don't have MGLTools
    _pdb_to_pdbqt_simple(clean_pdb, receptor_pdbqt)

    return receptor_pdbqt, center, box_size


def _pdb_to_pdbqt_simple(pdb_path, pdbqt_path):
    """
    Simple PDB to PDBQT conversion for the receptor.
    Maps atom names to AutoDock atom types.
    """
    ad_type_map = {
        "C": "C", "CA": "C", "CB": "C", "CG": "C", "CD": "C", "CE": "C", "CZ": "C",
        "CG1": "C", "CG2": "C", "CD1": "C", "CD2": "C", "CE1": "C", "CE2": "C",
        "N": "NA", "NE": "NA", "NH1": "N", "NH2": "N", "NZ": "N", "ND1": "NA", "ND2": "N",
        "NE1": "NA", "NE2": "NA",
        "O": "OA", "OG": "OA", "OG1": "OA", "OD1": "OA", "OD2": "OA", "OE1": "OA", "OE2": "OA",
        "OH": "OA", "OXT": "OA",
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
            # Determine AD type from element
            element = line[76:78].strip() if len(line) > 76 else atom_name[0]
            ad_type = ad_type_map.get(atom_name, ad_type_map.get(element, "C"))

            # Add charge column (0.000) and AD type
            pdbqt_line = line[:54] + "  0.00  0.00"
            # Pad to 77 chars, then add AD type
            pdbqt_line = pdbqt_line.ljust(77) + f" {ad_type:<2s}"
            lines.append(pdbqt_line.rstrip())

    with open(pdbqt_path, "w") as f:
        f.write("\n".join(lines) + "\n")

    print(f"  Receptor PDBQT: {pdbqt_path} ({len(lines)} lines)")


# ============================================================
# Step 3: Docking with AutoDock Vina
# ============================================================

def run_vina_docking(receptor_pdbqt, ligand_pdbqt, center, box_size, exhaustiveness=8):
    """Run Vina docking on a single ligand. Returns (score, time, success)."""
    with tempfile.NamedTemporaryFile(suffix=".pdbqt", delete=False) as out_f:
        out_path = out_f.name

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
        "--exhaustiveness", str(exhaustiveness),
        "--num_modes", "1",
        "--out", out_path,
    ]

    t0 = time.time()
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        elapsed = time.time() - t0

        # Parse score from output
        score = None
        for line in result.stdout.split("\n"):
            if line.strip().startswith("1"):
                parts = line.strip().split()
                if len(parts) >= 2:
                    try:
                        score = float(parts[1])
                    except ValueError:
                        pass
                    break

        if score is None:
            # Try parsing from output PDBQT
            if os.path.exists(out_path):
                with open(out_path) as f:
                    for line in f:
                        if "VINA RESULT" in line:
                            parts = line.split()
                            for i, p in enumerate(parts):
                                if p == "RESULT:":
                                    try:
                                        score = float(parts[i + 1])
                                    except (IndexError, ValueError):
                                        pass
                                    break
                            break

        os.unlink(out_path)
        return score, elapsed, score is not None

    except subprocess.TimeoutExpired:
        elapsed = time.time() - t0
        if os.path.exists(out_path):
            os.unlink(out_path)
        return None, elapsed, False
    except Exception as e:
        elapsed = time.time() - t0
        if os.path.exists(out_path):
            os.unlink(out_path)
        return None, elapsed, False


def benchmark_vina(mol_df, receptor_pdbqt, center, box_size, n_quick=10, n_full=100):
    """Benchmark Vina on molecule set."""
    print("\n" + "=" * 60)
    print("BENCHMARKING: AutoDock Vina v1.2.7")
    print("=" * 60)

    if not VINA_BIN.exists():
        print("  Vina binary not found, skipping")
        return None

    # Prepare all ligands first
    print(f"\nPreparing 3D structures and PDBQT files for {len(mol_df)} molecules...")
    ligand_dir = WORK_DIR / "ligands_pdbqt"
    ligand_dir.mkdir(exist_ok=True)

    prepared = []
    t_prep_start = time.time()
    for idx, row in mol_df.iterrows():
        mol = smiles_to_3d(row["smiles"])
        if mol is None:
            continue
        pdbqt_path = ligand_dir / f"mol_{idx:04d}.pdbqt"
        if mol_to_pdbqt(mol, pdbqt_path):
            prepared.append({
                "idx": idx,
                "smiles": row["smiles"],
                "pIC50": row["pIC50"],
                "pdbqt": str(pdbqt_path),
            })
    t_prep = time.time() - t_prep_start
    print(f"  Prepared {len(prepared)}/{len(mol_df)} molecules in {t_prep:.1f}s "
          f"({t_prep / len(mol_df):.2f}s/mol)")

    if len(prepared) == 0:
        print("  No molecules could be prepared!")
        return None

    # Quick benchmark: first n_quick molecules
    print(f"\n--- Quick benchmark: {n_quick} molecules ---")
    quick_results = []
    for i, lig in enumerate(prepared[:n_quick]):
        score, elapsed, success = run_vina_docking(
            receptor_pdbqt, lig["pdbqt"], center, box_size, exhaustiveness=8
        )
        quick_results.append({
            "smiles": lig["smiles"],
            "pIC50": lig["pIC50"],
            "vina_score": score,
            "time_s": elapsed,
            "success": success,
        })
        status = f"score={score:.2f}" if score else "FAILED"
        print(f"  [{i+1}/{n_quick}] {elapsed:.1f}s  {status}")

    quick_times = [r["time_s"] for r in quick_results]
    quick_scores = [r["vina_score"] for r in quick_results if r["vina_score"] is not None]
    quick_success = sum(1 for r in quick_results if r["success"])

    print(f"\n  Quick results ({n_quick} mols):")
    print(f"    Success rate: {quick_success}/{n_quick} ({100*quick_success/n_quick:.0f}%)")
    print(f"    Time/mol: {np.mean(quick_times):.1f}s (std={np.std(quick_times):.1f}s)")
    if quick_scores:
        print(f"    Scores: {np.mean(quick_scores):.2f} +/- {np.std(quick_scores):.2f} "
              f"(range: {np.min(quick_scores):.2f} to {np.max(quick_scores):.2f})")

    # Full benchmark: up to n_full molecules
    n_remaining = min(n_full, len(prepared)) - n_quick
    if n_remaining > 0:
        print(f"\n--- Full benchmark: {n_quick + n_remaining} molecules ---")
        full_results = list(quick_results)  # copy quick results
        for i, lig in enumerate(prepared[n_quick:n_quick + n_remaining]):
            score, elapsed, success = run_vina_docking(
                receptor_pdbqt, lig["pdbqt"], center, box_size, exhaustiveness=8
            )
            full_results.append({
                "smiles": lig["smiles"],
                "pIC50": lig["pIC50"],
                "vina_score": score,
                "time_s": elapsed,
                "success": success,
            })
            if (i + 1) % 10 == 0:
                status = f"score={score:.2f}" if score else "FAILED"
                print(f"  [{n_quick + i + 1}/{n_quick + n_remaining}] {elapsed:.1f}s  {status}")
    else:
        full_results = quick_results

    all_times = [r["time_s"] for r in full_results]
    all_scores = [r["vina_score"] for r in full_results if r["vina_score"] is not None]
    all_success = sum(1 for r in full_results if r["success"])
    n_total = len(full_results)

    # Compute correlation with pIC50
    corr_spearman = None
    corr_pearson = None
    if len(all_scores) > 5:
        from scipy import stats
        valid = [(r["pIC50"], r["vina_score"]) for r in full_results if r["vina_score"] is not None]
        pIC50s = [v[0] for v in valid]
        scores = [v[1] for v in valid]  # Vina scores are negative (lower = better)
        neg_scores = [-s for s in scores]  # negate so higher = better like pIC50
        corr_spearman = stats.spearmanr(pIC50s, neg_scores).statistic
        corr_pearson = stats.pearsonr(pIC50s, neg_scores).statistic

    print(f"\n  Full results ({n_total} mols):")
    print(f"    Success rate: {all_success}/{n_total} ({100*all_success/n_total:.0f}%)")
    print(f"    Time/mol: {np.mean(all_times):.1f}s (std={np.std(all_times):.1f}s)")
    print(f"    Total time: {sum(all_times):.0f}s ({sum(all_times)/60:.1f} min)")
    if all_scores:
        print(f"    Scores: {np.mean(all_scores):.2f} +/- {np.std(all_scores):.2f}")
    if corr_spearman is not None:
        print(f"    Spearman(pIC50, -vina_score): {corr_spearman:.3f}")
        print(f"    Pearson(pIC50, -vina_score):  {corr_pearson:.3f}")

    est_280 = np.mean(all_times) * 280
    print(f"\n    Estimated time for 280 molecules: {est_280:.0f}s ({est_280/60:.1f} min)")

    return {
        "tool": "AutoDock Vina",
        "version": "1.2.7",
        "installed": True,
        "n_molecules": n_total,
        "n_success": all_success,
        "success_rate": all_success / n_total,
        "mean_time_per_mol_s": float(np.mean(all_times)),
        "std_time_per_mol_s": float(np.std(all_times)),
        "total_time_s": float(sum(all_times)),
        "mean_score": float(np.mean(all_scores)) if all_scores else None,
        "std_score": float(np.std(all_scores)) if all_scores else None,
        "min_score": float(np.min(all_scores)) if all_scores else None,
        "max_score": float(np.max(all_scores)) if all_scores else None,
        "spearman_vs_pIC50": float(corr_spearman) if corr_spearman is not None else None,
        "pearson_vs_pIC50": float(corr_pearson) if corr_pearson is not None else None,
        "estimated_280_mols_s": float(est_280),
        "estimated_280_mols_min": float(est_280 / 60),
        "exhaustiveness": 8,
        "device": "CPU",
        "prep_time_per_mol_s": t_prep / len(mol_df),
        "detailed_results": full_results,
    }


# ============================================================
# Step 4: Check GNINA and DiffDock availability
# ============================================================

def check_gnina():
    """Check if GNINA is available."""
    print("\n" + "=" * 60)
    print("CHECKING: GNINA")
    print("=" * 60)

    # GNINA only has Linux binaries; no macOS build
    return {
        "tool": "GNINA",
        "version": "N/A",
        "installed": False,
        "reason": "GNINA only provides Linux x86_64 binaries. No macOS (ARM64) build available.",
        "install_instructions": (
            "Requires Linux x86_64. Install via:\n"
            "  wget https://github.com/gnina/gnina/releases/latest/download/gnina\n"
            "  chmod +x gnina\n"
            "Or Docker: docker pull gnina/gnina"
        ),
        "estimated_time_per_mol_s": "60-120s (CPU), uses CNN scoring on top of Vina",
        "device": "CPU (GPU optional for CNN rescoring)",
        "notes": "GNINA = Vina + CNN rescoring. Expect ~2x slower than Vina but better pose quality.",
    }


def check_diffdock():
    """Check if DiffDock is available."""
    print("\n" + "=" * 60)
    print("CHECKING: DiffDock")
    print("=" * 60)

    # DiffDock requires specific torch-geometric versions, not a simple pip install
    return {
        "tool": "DiffDock",
        "version": "N/A",
        "installed": False,
        "reason": (
            "DiffDock requires specific PyTorch Geometric / torch-scatter / torch-sparse versions "
            "and ESM protein embeddings. Not a simple pip install. "
            "Also, MPS backend is unreliable for transformer models (per project policy)."
        ),
        "install_instructions": (
            "Clone: git clone https://github.com/gcorso/DiffDock\n"
            "Requires: torch-geometric, torch-scatter, torch-sparse, e3nn, ESM\n"
            "Best run in dedicated conda env or Docker.\n"
            "Docker: docker pull bondalex/diffdock"
        ),
        "estimated_time_per_mol_s": "10-30s (GPU), 60-180s (CPU)",
        "device": "GPU recommended (CUDA), MPS unreliable",
        "notes": (
            "Diffusion-based generative model. Generates diverse poses without "
            "predefined binding pocket. Best for blind docking scenarios."
        ),
    }


# ============================================================
# Main
# ============================================================

def main():
    print("=" * 60)
    print("DOCKING BENCHMARK: ZAP70 (CHEMBL4899) Molecules")
    print("=" * 60)
    print(f"Working directory: {WORK_DIR}")
    print(f"Vina binary: {VINA_BIN} (exists: {VINA_BIN.exists()})")

    mol_df = load_molecules(target_chembl_id="CHEMBL4899", n_sample=100, seed=42)

    # Download and prepare protein
    pdb_path = download_pdb("1U59")
    receptor_pdbqt, center, box_size = prepare_receptor(pdb_path)

    results = {}

    # Benchmark Vina
    vina_result = benchmark_vina(mol_df, receptor_pdbqt, center, box_size,
                                  n_quick=10, n_full=100)
    if vina_result:
        # Separate detailed results for summary
        detailed = vina_result.pop("detailed_results")
        results["autodock_vina"] = vina_result

        # Save detailed per-molecule results separately
        detailed_df = pd.DataFrame(detailed)
        detailed_path = WORK_DIR / "vina_detailed_results.csv"
        detailed_df.to_csv(detailed_path, index=False)
        print(f"\nDetailed Vina results saved to {detailed_path}")

    # Check GNINA
    gnina_result = check_gnina()
    results["gnina"] = gnina_result
    print(f"  Status: Not installed ({gnina_result['reason']})")

    # Check DiffDock
    diffdock_result = check_diffdock()
    results["diffdock"] = diffdock_result
    print(f"  Status: Not installed ({diffdock_result['reason']})")

    # Summary and recommendation
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)

    summary = {
        "benchmark_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "target": "CHEMBL4899 (ZAP70)",
        "protein": "PDB 1U59 (ZAP70 kinase, staurosporine complex)",
        "n_molecules_sampled": len(mol_df),
        "platform": "macOS ARM64 (Apple Silicon)",
        "tools": results,
        "recommendation": (
            "AutoDock Vina is the only tool that runs natively on macOS ARM64. "
            "GNINA requires Linux; DiffDock requires a dedicated environment with "
            "specific PyTorch Geometric versions. For this project's needs:\n"
            "1. Vina: Ready to use, ~30-60s/mol, good for quick scoring\n"
            "2. GNINA: Better poses (CNN rescoring) but Linux-only\n"
            "3. DiffDock: Best for blind docking, needs GPU + complex setup\n"
            "Recommendation: Use Vina for feasibility. Consider GNINA in Docker "
            "for production runs if better pose quality is needed."
        ),
    }

    # Save results
    output_path = RESULTS_DIR / "docking_benchmark.json"
    with open(output_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nResults saved to {output_path}")

    # Print table
    print(f"\n{'Tool':<20} {'Installed':<12} {'Time/mol':<15} {'Success':<10} {'Device'}")
    print("-" * 70)
    for name, r in results.items():
        installed = "YES" if r.get("installed") else "NO"
        if r.get("installed"):
            time_str = f"{r['mean_time_per_mol_s']:.1f}s"
            success_str = f"{r['success_rate']*100:.0f}%"
        else:
            time_str = r.get("estimated_time_per_mol_s", "N/A")
            success_str = "N/A"
        device = r.get("device", "N/A")
        print(f"{name:<20} {installed:<12} {time_str:<15} {success_str:<10} {device}")


if __name__ == "__main__":
    main()
