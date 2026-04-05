#!/usr/bin/env python3
"""
Overnight MPS REINVENT4 Runner — Sequential Reinvent + Mol2Mol + LibInvent on Apple Silicon.

Runs all three generators sequentially on MPS with extended steps, collecting and
rescoring results after each generator (fault-tolerant). Monitors memory usage
and saves a summary JSON at the end.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_overnight_mps.py [generator]

    generator: reinvent | mol2mol | libinvent | all (default: all)
"""

import sys
import os
import json
import gc
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "reinvent4_configs"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_mps"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Conda executable
CONDA_PATH = "/opt/miniconda3/condabin/conda"
if not os.path.exists(CONDA_PATH):
    for p in ["/opt/miniconda3/bin/conda",
              os.path.expanduser("~/miniconda3/condabin/conda"),
              os.path.expanduser("~/miniconda3/bin/conda")]:
        if os.path.exists(p):
            CONDA_PATH = p
            break
if not os.path.exists(CONDA_PATH):
    raise RuntimeError("Cannot find conda executable")

# Memory monitoring
try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False


def get_memory_gb():
    """Return current process memory usage in GB, or None if psutil unavailable."""
    if not HAS_PSUTIL:
        return None
    proc = psutil.Process(os.getpid())
    return proc.memory_info().rss / (1024 ** 3)


def log_memory(label=""):
    """Print memory usage if psutil is available."""
    mem = get_memory_gb()
    if mem is not None:
        print(f"[memory] {label}: {mem:.2f} GB RSS")


def get_reference_smiles(n=5):
    """Get top N ZAP70 actives for Tanimoto reference in Mol2Mol config."""
    actives_file = PROJECT_ROOT / "data" / "zap70_top_actives_clean.smi"
    if not actives_file.exists():
        actives_file = PROJECT_ROOT / "data" / "zap70_top_actives.smi"
    smiles = [line.strip() for line in open(actives_file) if line.strip()][:n]
    return smiles


def resolve_config(template_path, output_path, generator_name):
    """Replace placeholders in TOML config with actual paths."""
    content = template_path.read_text()

    content = content.replace("__CONDA_PATH__", CONDA_PATH)
    content = content.replace("__PROJECT_ROOT__", str(PROJECT_ROOT))

    # Prior paths for each generator
    priors_dir = REINVENT4_ROOT / "priors"
    content = content.replace("__PRIOR_PATH__", str(priors_dir / "reinvent.prior"))
    content = content.replace("__PRIOR_PATH_MOL2MOL__", str(priors_dir / "mol2mol_medium_similarity.prior"))
    content = content.replace("__PRIOR_PATH_LIBINVENT__", str(priors_dir / "libinvent.prior"))

    # For mol2mol: inject reference SMILES
    if "__ZAP70_REF_SMILES__" in content:
        ref_smiles = get_reference_smiles(5)
        smiles_toml = "[" + ", ".join(f'"{s}"' for s in ref_smiles) + "]"
        content = content.replace("__ZAP70_REF_SMILES__", smiles_toml)

    output_path.write_text(content)
    print(f"Config written: {output_path}")
    return output_path


def prepare_scaffolds():
    """Extract Murcko scaffolds from ZAP70 actives for LibInvent."""
    from rdkit import Chem, RDLogger
    from rdkit.Chem.Scaffolds import MurckoScaffold
    RDLogger.DisableLog('rdApp.*')

    scaffolds_file = PROJECT_ROOT / "data" / "zap70_scaffolds.smi"
    if scaffolds_file.exists():
        n = sum(1 for _ in open(scaffolds_file))
        print(f"Scaffolds file exists: {scaffolds_file} ({n} scaffolds)")
        return scaffolds_file

    actives_file = PROJECT_ROOT / "data" / "zap70_actives.smi"
    if not actives_file.exists():
        raise FileNotFoundError(f"Need {actives_file} -- run reinvent4_film_scorer.py first")

    smiles_list = [line.strip() for line in open(actives_file) if line.strip()]
    print(f"Extracting scaffolds from {len(smiles_list)} ZAP70 actives...")

    scaffolds = set()
    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            scaffold = MurckoScaffold.GetScaffoldForMol(mol)
            scaffold_smi = Chem.MolToSmiles(scaffold)
            if scaffold_smi and scaffold_smi != "" and Chem.MolFromSmiles(scaffold_smi) is not None:
                scaffolds.add(scaffold_smi)
        except Exception:
            continue

    libinvent_scaffolds = []
    for scaffold_smi in scaffolds:
        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is None:
            continue

        from rdkit.Chem import RWMol

        rwmol = RWMol(mol)
        h_positions = []
        for atom in rwmol.GetAtoms():
            if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
                h_positions.append(atom.GetIdx())

        if not h_positions:
            for atom in rwmol.GetAtoms():
                if atom.GetTotalNumHs() > 0 and atom.GetDegree() <= 2:
                    h_positions.append(atom.GetIdx())

        if len(h_positions) < 1:
            continue

        np.random.seed(hash(scaffold_smi) % 2**32)
        n_points = min(2, len(h_positions))
        chosen = np.random.choice(h_positions, n_points, replace=False)

        rwmol2 = RWMol(mol)
        attach_num = 0
        for idx in sorted(chosen):
            atom = rwmol2.GetAtomWithIdx(idx)
            new_idx = rwmol2.AddAtom(Chem.Atom(0))
            rwmol2.GetAtomWithIdx(new_idx).SetIsotope(0)
            rwmol2.GetAtomWithIdx(new_idx).SetAtomMapNum(attach_num)
            rwmol2.AddBond(idx, new_idx, Chem.BondType.SINGLE)
            atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
            attach_num += 1

        try:
            new_mol = rwmol2.GetMol()
            Chem.SanitizeMol(new_mol)
            new_smi = Chem.MolToSmiles(new_mol)
            import re
            new_smi = re.sub(r'\[(\d+)\*:(\d+)\]', r'[*:\2]', new_smi)
            new_smi = re.sub(r'\[\*:(\d+)\]', r'[*:\1]', new_smi)
            if new_smi and '[*' in new_smi:
                libinvent_scaffolds.append(new_smi)
        except Exception:
            continue

    libinvent_scaffolds = list(set(libinvent_scaffolds))
    print(f"Generated {len(libinvent_scaffolds)} LibInvent scaffolds from {len(scaffolds)} Murcko scaffolds")

    with open(scaffolds_file, "w") as f:
        for smi in libinvent_scaffolds:
            f.write(smi + "\n")

    return scaffolds_file


def run_reinvent4(config_path, generator_name, working_dir):
    """Run REINVENT4 with given config on MPS."""
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running REINVENT4 (MPS): {generator_name}")
    print(f"Config: {config_path}")
    print(f"Working dir: {working_dir}")
    print(f"{'='*60}\n")

    log_memory(f"before {generator_name}")

    log_file = working_dir / f"{generator_name}.log"

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "quris",
        "reinvent", str(config_path), "-d", "mps"
    ]

    start_time = time.time()
    try:
        with open(log_file, "w") as lf:
            process = subprocess.Popen(
                cmd,
                cwd=str(working_dir),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,
            )

            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)

            process.wait()

        elapsed = time.time() - start_time
        print(f"\n{generator_name} finished in {elapsed/60:.1f} min (exit code {process.returncode})")
        log_memory(f"after {generator_name}")

        if process.returncode != 0:
            print(f"WARNING: {generator_name} exited with code {process.returncode}")
            print(f"Check log: {log_file}")

        return process.returncode == 0, elapsed

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{generator_name} FAILED after {elapsed/60:.1f} min: {e}")
        return False, elapsed


def collect_results(working_dir, generator_name):
    """Collect generated SMILES from REINVENT4 CSV output."""
    import pandas as pd
    from rdkit import Chem

    csv_files = sorted(working_dir.glob(f"{generator_name}*.csv"))
    if not csv_files:
        csv_files = sorted(working_dir.glob("*.csv"))

    if not csv_files:
        print(f"No CSV output found in {working_dir}")
        return None

    all_smiles = set()
    all_scores = []

    for csv_file in csv_files:
        try:
            # Handle the CSV comma issue with on_bad_lines='skip'
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            print(f"  {csv_file.name}: {len(df)} rows, columns: {list(df.columns)[:8]}")

            # Find SMILES column
            smi_col = None
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
                if col in df.columns:
                    smi_col = col
                    break
            if smi_col is None and len(df.columns) > 0:
                smi_col = df.columns[0]

            # Find score column
            score_col = None
            for col in ["FiLMDelta pIC50 (raw)", "total_score", "Score", "score"]:
                if col in df.columns:
                    score_col = col
                    break

            for _, row in df.iterrows():
                smi = str(row[smi_col]).strip()
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    can = Chem.MolToSmiles(mol)
                    if can not in all_smiles:
                        all_smiles.add(can)
                        entry = {"smiles": can}
                        if score_col and pd.notna(row.get(score_col)):
                            entry["reinvent_score"] = float(row[score_col])
                        all_scores.append(entry)

        except Exception as e:
            print(f"  Error reading {csv_file.name}: {e}")

    print(f"  Total unique valid SMILES: {len(all_smiles)}")
    return all_scores


def rescore_with_film(smiles_list, batch_size=500):
    """Re-score all generated molecules with FiLMDelta for consistent comparison."""
    print(f"\nRe-scoring {len(smiles_list)} molecules with FiLMDelta...")

    all_scores = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        input_str = "\n".join(batch)

        try:
            result = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", "quris",
                 "python", str(PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py")],
                input=input_str, capture_output=True, text=True, timeout=300,
            )

            if result.returncode == 0:
                try:
                    data = json.loads(result.stdout.strip())
                    scores = data["payload"]["pIC50"]
                    all_scores.extend(scores)
                except (json.JSONDecodeError, KeyError) as e:
                    print(f"  Batch {i}: parse error: {e}")
                    all_scores.extend([float('nan')] * len(batch))
            else:
                print(f"  Batch {i}: scorer failed")
                all_scores.extend([float('nan')] * len(batch))
        except subprocess.TimeoutExpired:
            print(f"  Batch {i}: scorer timed out")
            all_scores.extend([float('nan')] * len(batch))

        if (i + batch_size) % 2000 == 0:
            print(f"  Re-scored {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}")

    return all_scores


def save_generator_results(gen_name, results, timing, results_dir):
    """Save results for a single generator (fault-tolerant incremental save)."""
    output_file = results_dir / f"{gen_name}_results.json"

    if results is None:
        data = {"generator": gen_name, "status": "failed", "elapsed_min": timing / 60}
    else:
        smiles = [r["smiles"] for r in results]
        film_scores = [r.get("film_pIC50", float('nan')) for r in results]
        valid_scores = [s for s in film_scores if not np.isnan(s)]

        data = {
            "generator": gen_name,
            "status": "completed",
            "elapsed_min": timing / 60,
            "n_molecules": len(results),
            "n_scored": len(valid_scores),
            "mean_pIC50": float(np.nanmean(valid_scores)) if valid_scores else None,
            "median_pIC50": float(np.nanmedian(valid_scores)) if valid_scores else None,
            "max_pIC50": float(np.nanmax(valid_scores)) if valid_scores else None,
            "n_potent_7plus": sum(1 for s in valid_scores if s >= 7.0),
            "n_potent_8plus": sum(1 for s in valid_scores if s >= 8.0),
            "top_10": sorted(results, key=lambda x: -x.get("film_pIC50", 0))[:10],
        }

    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {gen_name} results: {output_file}")
    return data


def generate_summary(all_results, timings):
    """Generate final summary JSON and print results."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "device": "mps",
        "total_elapsed_min": sum(timings.values()) / 60,
        "generators": {},
        "total_unique_molecules": 0,
    }

    total_smiles = set()
    for gen_name, results in all_results.items():
        elapsed = timings.get(gen_name, 0)

        if results is None:
            summary["generators"][gen_name] = {
                "status": "failed",
                "elapsed_min": elapsed / 60,
            }
            continue

        smiles = [r["smiles"] for r in results]
        scores = [r.get("film_pIC50", float('nan')) for r in results]
        valid_scores = [s for s in scores if not np.isnan(s)]

        total_smiles.update(smiles)

        summary["generators"][gen_name] = {
            "status": "completed",
            "elapsed_min": elapsed / 60,
            "n_molecules": len(results),
            "n_scored": len(valid_scores),
            "mean_pIC50": float(np.nanmean(valid_scores)) if valid_scores else None,
            "median_pIC50": float(np.nanmedian(valid_scores)) if valid_scores else None,
            "max_pIC50": float(np.nanmax(valid_scores)) if valid_scores else None,
            "n_potent_7plus": sum(1 for s in valid_scores if s >= 7.0),
            "n_potent_8plus": sum(1 for s in valid_scores if s >= 8.0),
            "top_10": sorted(results, key=lambda x: -x.get("film_pIC50", 0))[:10],
        }

    summary["total_unique_molecules"] = len(total_smiles)

    # Print timing summary
    print(f"\n{'='*60}")
    print("REINVENT4 MPS Overnight Generation Summary")
    print(f"{'='*60}")

    for gen_name, stats in summary["generators"].items():
        elapsed_min = stats.get("elapsed_min", 0)
        if stats["status"] == "failed":
            print(f"\n{gen_name}: FAILED ({elapsed_min:.1f} min)")
            continue
        print(f"\n{gen_name} ({elapsed_min:.1f} min):")
        print(f"  Molecules: {stats['n_molecules']}")
        if stats['mean_pIC50'] is not None:
            print(f"  Mean pIC50: {stats['mean_pIC50']:.3f}")
        else:
            print(f"  Mean pIC50: N/A")
        if stats['max_pIC50'] is not None:
            print(f"  Max pIC50:  {stats['max_pIC50']:.3f}")
        else:
            print(f"  Max pIC50:  N/A")
        print(f"  Potent (>=7): {stats['n_potent_7plus']}")
        print(f"  Potent (>=8): {stats['n_potent_8plus']}")

    total_min = summary["total_elapsed_min"]
    print(f"\nTotal unique molecules: {summary['total_unique_molecules']}")
    print(f"Total elapsed time: {total_min:.1f} min ({total_min/60:.1f} hr)")

    # Save summary
    summary_file = RESULTS_DIR / "overnight_mps_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"\nSummary saved: {summary_file}")

    return summary


def main():
    # Use extended configs for longer overnight runs (10-12h budget)
    use_extended = "--extended" in sys.argv or "-x" in sys.argv

    # Parse generator arg (skip flags)
    positional_args = [a for a in sys.argv[1:] if not a.startswith("-")]
    generators_arg = positional_args[0] if positional_args else "all"
    if use_extended:
        configs = {
            "reinvent": ("reinvent_denovo_mps_extended.toml", "reinvent_denovo_ext"),
            "mol2mol": ("mol2mol_optimize_mps_extended.toml", "mol2mol_ext"),
            "libinvent": ("libinvent_rgroup_mps_extended.toml", "libinvent_ext"),
        }
        print("Using EXTENDED configs (10-12h overnight)")
    else:
        configs = {
            "reinvent": ("reinvent_denovo_mps.toml", "reinvent_denovo_mps"),
            "mol2mol": ("mol2mol_optimize_mps.toml", "mol2mol_optimize_mps"),
            "libinvent": ("libinvent_rgroup_mps.toml", "libinvent_rgroup_mps"),
        }

    if generators_arg == "all":
        run_list = ["reinvent", "mol2mol", "libinvent"]
    else:
        run_list = [g.strip() for g in generators_arg.split(",")]

    # Validate
    for g in run_list:
        if g not in configs:
            print(f"Unknown generator: {g}. Choose from: {list(configs.keys())}")
            sys.exit(1)

    print(f"Overnight MPS run: {run_list}")
    print(f"Results dir: {RESULTS_DIR}")
    log_memory("startup")

    # Prepare scaffolds for LibInvent
    if "libinvent" in run_list:
        try:
            prepare_scaffolds()
        except Exception as e:
            print(f"WARNING: Failed to prepare scaffolds: {e}")
            print("Skipping libinvent")
            run_list.remove("libinvent")

    # Ensure FiLMDelta model is cached
    model_cache = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_model.pt"
    if not model_cache.exists():
        print("Pre-training FiLMDelta model...")
        subprocess.run(
            ["conda", "run", "--no-capture-output", "-n", "quris", "python", "-u",
             str(PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py")],
            input="c1ccccc1\n", capture_output=True, text=True,
        )

    all_results = {}
    timings = {}
    overall_start = time.time()

    for gen_name in run_list:
        template_name, prefix = configs[gen_name]
        template_path = CONFIGS_DIR / template_name

        working_dir = RESULTS_DIR / gen_name
        resolved_config = working_dir / template_name
        working_dir.mkdir(parents=True, exist_ok=True)

        resolve_config(template_path, resolved_config, gen_name)

        success, elapsed = run_reinvent4(resolved_config, prefix, working_dir)
        timings[gen_name] = elapsed

        if success:
            results = collect_results(working_dir, prefix)
            if results:
                # Re-score with FiLMDelta for consistent comparison
                smiles_list = [r["smiles"] for r in results]
                film_scores = rescore_with_film(smiles_list)
                for r, s in zip(results, film_scores):
                    r["film_pIC50"] = s
            all_results[gen_name] = results
        else:
            all_results[gen_name] = None

        # Save results incrementally after each generator (fault-tolerant)
        save_generator_results(gen_name, all_results[gen_name], elapsed, RESULTS_DIR)

        # Free memory between generators
        gc.collect()
        log_memory(f"after {gen_name} cleanup")

    overall_elapsed = time.time() - overall_start
    print(f"\nAll generators completed in {overall_elapsed/60:.1f} min ({overall_elapsed/3600:.1f} hr)")

    generate_summary(all_results, timings)


if __name__ == "__main__":
    main()
