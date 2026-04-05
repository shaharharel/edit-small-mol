#!/usr/bin/env python3
"""
Memory-Safe MPS Runner: Rescore existing Mol2Mol results, then run
Mol2Mol (purge_memories=true, batch_size=32) + LibInvent sequentially.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_memsafe_mps.py
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


def get_reference_smiles(n=5):
    actives_file = PROJECT_ROOT / "data" / "zap70_top_actives_clean.smi"
    if not actives_file.exists():
        actives_file = PROJECT_ROOT / "data" / "zap70_top_actives.smi"
    return [line.strip() for line in open(actives_file) if line.strip()][:n]


def resolve_config(template_path, output_path):
    content = template_path.read_text()
    priors_dir = REINVENT4_ROOT / "priors"
    content = content.replace("__CONDA_PATH__", CONDA_PATH)
    content = content.replace("__PROJECT_ROOT__", str(PROJECT_ROOT))
    content = content.replace("__PRIOR_PATH__", str(priors_dir / "reinvent.prior"))
    content = content.replace("__PRIOR_PATH_MOL2MOL__", str(priors_dir / "mol2mol_medium_similarity.prior"))
    content = content.replace("__PRIOR_PATH_LIBINVENT__", str(priors_dir / "libinvent.prior"))
    if "__ZAP70_REF_SMILES__" in content:
        ref_smiles = get_reference_smiles(5)
        smiles_toml = "[" + ", ".join(f'"{s}"' for s in ref_smiles) + "]"
        content = content.replace("__ZAP70_REF_SMILES__", smiles_toml)
    output_path.write_text(content)
    print(f"Config written: {output_path}")
    return output_path


def rescore_with_film(smiles_list, batch_size=500):
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
                data = json.loads(result.stdout.strip())
                scores = data["payload"]["pIC50"]
                all_scores.extend(scores)
            else:
                all_scores.extend([float('nan')] * len(batch))
        except Exception:
            all_scores.extend([float('nan')] * len(batch))
        if (i + batch_size) % 5000 == 0:
            print(f"  Re-scored {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}")
    return all_scores


def collect_and_rescore_existing():
    """Rescore the 65K molecules from the killed Mol2Mol run."""
    import pandas as pd
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')

    mol2mol_dir = RESULTS_DIR / "mol2mol"
    csvs = sorted(mol2mol_dir.glob("mol2mol_ext*.csv"))
    if not csvs:
        print("No existing Mol2Mol CSV files found")
        return

    print(f"\n{'='*60}")
    print("Rescoring existing Mol2Mol results (33 steps, killed run)")
    print(f"{'='*60}")

    all_smiles = []
    seen = set()
    for f in csvs:
        try:
            df = pd.read_csv(f, on_bad_lines='skip')
            smi_col = 'SMILES' if 'SMILES' in df.columns else df.columns[0]
            for s in df[smi_col]:
                mol = Chem.MolFromSmiles(str(s).strip())
                if mol:
                    can = Chem.MolToSmiles(mol)
                    if can not in seen:
                        seen.add(can)
                        all_smiles.append(can)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    print(f"Found {len(all_smiles)} unique valid SMILES from killed run")

    if not all_smiles:
        return

    film_scores = rescore_with_film(all_smiles)
    results = [{"smiles": s, "film_pIC50": sc} for s, sc in zip(all_smiles, film_scores)]
    valid_scores = [s for s in film_scores if not np.isnan(s)]

    data = {
        "generator": "mol2mol_killed_run",
        "status": "partial (killed at step 33)",
        "n_molecules": len(results),
        "n_scored": len(valid_scores),
        "mean_pIC50": float(np.nanmean(valid_scores)) if valid_scores else None,
        "median_pIC50": float(np.nanmedian(valid_scores)) if valid_scores else None,
        "max_pIC50": float(np.nanmax(valid_scores)) if valid_scores else None,
        "n_potent_7plus": sum(1 for s in valid_scores if s >= 7.0),
        "n_potent_8plus": sum(1 for s in valid_scores if s >= 8.0),
        "top_10": sorted(results, key=lambda x: -x.get("film_pIC50", 0))[:10],
    }

    output_file = RESULTS_DIR / "mol2mol_killed_run_results.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved killed run results: {output_file}")
    print(f"  Molecules: {data['n_molecules']}")
    print(f"  Mean pIC50: {data['mean_pIC50']:.3f}" if data['mean_pIC50'] else "  Mean pIC50: N/A")
    print(f"  Max pIC50: {data['max_pIC50']:.3f}" if data['max_pIC50'] else "  Max pIC50: N/A")
    print(f"  Potent (>=7): {data['n_potent_7plus']}")
    print(f"  Potent (>=8): {data['n_potent_8plus']}")

    gc.collect()


def run_reinvent4(config_path, prefix, working_dir):
    working_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n{'='*60}")
    print(f"Running REINVENT4 (MPS): {prefix}")
    print(f"Config: {config_path}")
    print(f"{'='*60}\n")

    log_file = working_dir / f"{prefix}.log"
    cmd = ["conda", "run", "--no-capture-output", "-n", "quris",
           "reinvent", str(config_path), "-d", "mps"]

    start_time = time.time()
    try:
        with open(log_file, "w") as lf:
            process = subprocess.Popen(
                cmd, cwd=str(working_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in process.stdout:
                sys.stdout.write(line)
                sys.stdout.flush()
                lf.write(line)
            process.wait()

        elapsed = time.time() - start_time
        print(f"\n{prefix} finished in {elapsed/60:.1f} min (exit code {process.returncode})")
        return process.returncode == 0, elapsed
    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{prefix} FAILED after {elapsed/60:.1f} min: {e}")
        return False, elapsed


def collect_results(working_dir, prefix):
    import pandas as pd
    from rdkit import Chem
    csv_files = sorted(working_dir.glob(f"{prefix}*.csv"))
    if not csv_files:
        csv_files = sorted(working_dir.glob("*.csv"))
    if not csv_files:
        return None

    all_smiles = set()
    all_scores = []
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            smi_col = None
            for col in ["SMILES", "smiles", "Smiles"]:
                if col in df.columns:
                    smi_col = col
                    break
            if smi_col is None:
                smi_col = df.columns[0]

            for _, row in df.iterrows():
                smi = str(row[smi_col]).strip()
                mol = Chem.MolFromSmiles(smi)
                if mol:
                    can = Chem.MolToSmiles(mol)
                    if can not in all_smiles:
                        all_smiles.add(can)
                        all_scores.append({"smiles": can})
        except Exception as e:
            print(f"Error reading {csv_file}: {e}")

    print(f"  Total unique valid SMILES: {len(all_smiles)}")
    return all_scores


def save_results(gen_name, results, elapsed, film_scores=None):
    if results is None:
        data = {"generator": gen_name, "status": "failed", "elapsed_min": elapsed / 60}
    else:
        if film_scores:
            for r, s in zip(results, film_scores):
                r["film_pIC50"] = s
        scores = [r.get("film_pIC50", float('nan')) for r in results]
        valid = [s for s in scores if not np.isnan(s)]
        data = {
            "generator": gen_name,
            "status": "completed",
            "elapsed_min": elapsed / 60,
            "n_molecules": len(results),
            "n_scored": len(valid),
            "mean_pIC50": float(np.nanmean(valid)) if valid else None,
            "median_pIC50": float(np.nanmedian(valid)) if valid else None,
            "max_pIC50": float(np.nanmax(valid)) if valid else None,
            "n_potent_7plus": sum(1 for s in valid if s >= 7.0),
            "n_potent_8plus": sum(1 for s in valid if s >= 8.0),
            "top_10": sorted(results, key=lambda x: -x.get("film_pIC50", 0))[:10],
        }

    output_file = RESULTS_DIR / f"{gen_name}_results.json"
    with open(output_file, "w") as f:
        json.dump(data, f, indent=2, default=str)
    print(f"Saved {gen_name} results: {output_file}")
    return data


def main():
    print(f"Memory-Safe MPS Runner")
    print(f"Results dir: {RESULTS_DIR}")

    # Step 1: Rescore existing Mol2Mol results from killed run
    collect_and_rescore_existing()

    configs = [
        ("mol2mol_optimize_mps_memsafe.toml", "mol2mol_memsafe", "mol2mol_memsafe"),
        ("libinvent_rgroup_mps_extended.toml", "libinvent_ext", "libinvent"),
    ]

    for template_name, prefix, gen_name in configs:
        template_path = CONFIGS_DIR / template_name
        working_dir = RESULTS_DIR / gen_name
        working_dir.mkdir(parents=True, exist_ok=True)
        resolved_config = working_dir / template_name
        resolve_config(template_path, resolved_config)

        success, elapsed = run_reinvent4(resolved_config, prefix, working_dir)

        if success:
            results = collect_results(working_dir, prefix)
            if results:
                smiles_list = [r["smiles"] for r in results]
                film_scores = rescore_with_film(smiles_list)
                save_results(gen_name, results, elapsed, film_scores)
            else:
                save_results(gen_name, None, elapsed)
        else:
            save_results(gen_name, None, elapsed)

        gc.collect()
        print(f"\n[cleanup] gc.collect() after {gen_name}")

    print(f"\nAll generators completed!")


if __name__ == "__main__":
    main()
