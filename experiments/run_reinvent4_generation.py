#!/usr/bin/env python3
"""
REINVENT4 Molecular Generation with FiLMDelta Scoring for ZAP70.

Runs multiple REINVENT4 generators (Reinvent de novo, Mol2Mol, LibInvent)
with FiLMDelta anchor-based pIC50 as the scoring function.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_reinvent4_generation.py [generator]

    generator: reinvent | mol2mol | libinvent | all (default: all)
"""

import sys
import os
import json
import shutil
import subprocess
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
CONFIGS_DIR = PROJECT_ROOT / "experiments" / "reinvent4_configs"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Find conda executable
CONDA_PATH = "/opt/miniconda3/condabin/conda"
if not os.path.exists(CONDA_PATH):
    # Try other common locations
    for p in ["/opt/miniconda3/bin/conda",
              os.path.expanduser("~/miniconda3/condabin/conda"),
              os.path.expanduser("~/miniconda3/bin/conda")]:
        if os.path.exists(p):
            CONDA_PATH = p
            break
if not os.path.exists(CONDA_PATH):
    raise RuntimeError("Cannot find conda executable")


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
        raise FileNotFoundError(f"Need {actives_file} — run reinvent4_film_scorer.py first")

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

    # Convert scaffolds to LibInvent format (add attachment points)
    # For each scaffold, find positions where R-groups can be added
    # Simple approach: replace terminal atoms with attachment points
    libinvent_scaffolds = []
    for scaffold_smi in scaffolds:
        mol = Chem.MolFromSmiles(scaffold_smi)
        if mol is None:
            continue

        # Find atoms that could be decorated (single bond to H or terminal)
        # LibInvent expects [*:N] attachment points
        # Strategy: for aromatic rings, add 1-2 attachment points at available positions
        from rdkit.Chem import AllChem, RWMol

        # Simple: replace each H on aromatic carbons with [*]
        # Pick scaffolds with 1-2 attachment points
        rwmol = RWMol(mol)
        h_positions = []
        for atom in rwmol.GetAtoms():
            if atom.GetIsAromatic() and atom.GetTotalNumHs() > 0:
                h_positions.append(atom.GetIdx())

        if not h_positions:
            # Non-aromatic scaffold: try any CH
            for atom in rwmol.GetAtoms():
                if atom.GetTotalNumHs() > 0 and atom.GetDegree() <= 2:
                    h_positions.append(atom.GetIdx())

        if len(h_positions) < 1:
            continue

        # Take 1-2 attachment points
        np.random.seed(hash(scaffold_smi) % 2**32)
        n_points = min(2, len(h_positions))
        chosen = np.random.choice(h_positions, n_points, replace=False)

        # Add dummy atoms as attachment points
        rwmol2 = RWMol(mol)
        attach_num = 0
        for idx in sorted(chosen):
            # Decrement H count and add [*] neighbor
            atom = rwmol2.GetAtomWithIdx(idx)
            new_idx = rwmol2.AddAtom(Chem.Atom(0))  # Atom 0 = dummy [*]
            rwmol2.GetAtomWithIdx(new_idx).SetIsotope(0)
            rwmol2.GetAtomWithIdx(new_idx).SetAtomMapNum(attach_num)
            rwmol2.AddBond(idx, new_idx, Chem.BondType.SINGLE)
            atom.SetNumExplicitHs(max(0, atom.GetTotalNumHs() - 1))
            attach_num += 1

        try:
            new_mol = rwmol2.GetMol()
            Chem.SanitizeMol(new_mol)
            new_smi = Chem.MolToSmiles(new_mol)
            # Convert [N:M] to [*:M] format
            import re
            new_smi = re.sub(r'\[(\d+)\*:(\d+)\]', r'[*:\2]', new_smi)
            new_smi = re.sub(r'\[\*:(\d+)\]', r'[*:\1]', new_smi)
            if new_smi and '[*' in new_smi:
                libinvent_scaffolds.append(new_smi)
        except Exception:
            continue

    # Deduplicate
    libinvent_scaffolds = list(set(libinvent_scaffolds))
    print(f"Generated {len(libinvent_scaffolds)} LibInvent scaffolds from {len(scaffolds)} Murcko scaffolds")

    with open(scaffolds_file, "w") as f:
        for smi in libinvent_scaffolds:
            f.write(smi + "\n")

    return scaffolds_file


def get_reference_smiles(n=5):
    """Get top N ZAP70 actives for Tanimoto reference in Mol2Mol config."""
    actives_file = PROJECT_ROOT / "data" / "zap70_top_actives.smi"
    smiles = [line.strip() for line in open(actives_file) if line.strip()][:n]
    return smiles


def resolve_config(template_path, output_path, generator_name):
    """Replace placeholders in TOML config with actual paths."""
    content = template_path.read_text()

    content = content.replace("__CONDA_PATH__", CONDA_PATH)
    content = content.replace("__PROJECT_ROOT__", str(PROJECT_ROOT))

    # For mol2mol: inject reference SMILES
    if "__ZAP70_REF_SMILES__" in content:
        ref_smiles = get_reference_smiles(5)
        smiles_toml = "[" + ", ".join(f'"{s}"' for s in ref_smiles) + "]"
        content = content.replace("__ZAP70_REF_SMILES__", smiles_toml)

    # Fix prior paths to absolute
    content = content.replace(
        'prior_file = "priors/',
        f'prior_file = "{REINVENT4_ROOT}/priors/'
    )
    content = content.replace(
        'agent_file = "priors/',
        f'agent_file = "{REINVENT4_ROOT}/priors/'
    )

    output_path.write_text(content)
    print(f"Config written: {output_path}")
    return output_path


def run_reinvent4(config_path, generator_name, working_dir):
    """Run REINVENT4 with given config."""
    working_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'='*60}")
    print(f"Running REINVENT4: {generator_name}")
    print(f"Config: {config_path}")
    print(f"Working dir: {working_dir}")
    print(f"{'='*60}\n")

    log_file = working_dir / f"{generator_name}.log"

    cmd = [
        "conda", "run", "--no-capture-output", "-n", "quris",
        "reinvent", str(config_path), "-d", "cpu"
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

        if process.returncode != 0:
            print(f"WARNING: {generator_name} exited with code {process.returncode}")
            print(f"Check log: {log_file}")

        return process.returncode == 0

    except Exception as e:
        elapsed = time.time() - start_time
        print(f"\n{generator_name} FAILED after {elapsed/60:.1f} min: {e}")
        return False


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
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            print(f"  {csv_file.name}: {len(df)} rows, columns: {list(df.columns)[:8]}")

            # REINVENT4 outputs SMILES in a column (usually 'SMILES' or 'smiles')
            smi_col = None
            for col in ["SMILES", "smiles", "Smiles", "canonical_smiles"]:
                if col in df.columns:
                    smi_col = col
                    break
            if smi_col is None and len(df.columns) > 0:
                smi_col = df.columns[0]

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

    # Use the scorer script via stdin/stdout for consistency
    all_scores = []
    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        input_str = "\n".join(batch)

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

        if (i + batch_size) % 2000 == 0:
            print(f"  Re-scored {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}")

    return all_scores


def generate_summary(all_results):
    """Generate summary JSON and print results."""
    summary = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "generators": {},
        "total_unique_molecules": 0,
    }

    total_smiles = set()
    for gen_name, results in all_results.items():
        if results is None:
            summary["generators"][gen_name] = {"status": "failed"}
            continue

        smiles = [r["smiles"] for r in results]
        scores = [r.get("film_pIC50", float('nan')) for r in results]
        valid_scores = [s for s in scores if not np.isnan(s)]

        total_smiles.update(smiles)

        summary["generators"][gen_name] = {
            "status": "completed",
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

    # Print summary
    print(f"\n{'='*60}")
    print("REINVENT4 Generation Summary")
    print(f"{'='*60}")
    for gen_name, stats in summary["generators"].items():
        if stats["status"] == "failed":
            print(f"\n{gen_name}: FAILED")
            continue
        print(f"\n{gen_name}:")
        print(f"  Molecules: {stats['n_molecules']}")
        print(f"  Mean pIC50: {stats['mean_pIC50']:.3f}" if stats['mean_pIC50'] else "  Mean pIC50: N/A")
        print(f"  Max pIC50:  {stats['max_pIC50']:.3f}" if stats['max_pIC50'] else "  Max pIC50:  N/A")
        print(f"  Potent (≥7): {stats['n_potent_7plus']}")
        print(f"  Potent (≥8): {stats['n_potent_8plus']}")

    print(f"\nTotal unique molecules: {summary['total_unique_molecules']}")

    # Save
    summary_file = RESULTS_DIR / "reinvent4_generation_summary.json"
    with open(summary_file, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    print(f"Summary saved: {summary_file}")

    return summary


def main():
    generators = sys.argv[1] if len(sys.argv) > 1 else "all"

    configs = {
        "reinvent": ("reinvent_denovo.toml", "reinvent_denovo"),
        "mol2mol": ("mol2mol_optimize.toml", "mol2mol_optimize"),
        "libinvent": ("libinvent_rgroup.toml", "libinvent_rgroup"),
    }

    if generators == "all":
        run_list = ["reinvent", "mol2mol", "libinvent"]
    else:
        run_list = [g.strip() for g in generators.split(",")]

    # Validate
    for g in run_list:
        if g not in configs:
            print(f"Unknown generator: {g}. Choose from: {list(configs.keys())}")
            sys.exit(1)

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

    for gen_name in run_list:
        template_name, prefix = configs[gen_name]
        template_path = CONFIGS_DIR / template_name

        working_dir = RESULTS_DIR / gen_name
        resolved_config = working_dir / template_name
        working_dir.mkdir(parents=True, exist_ok=True)

        resolve_config(template_path, resolved_config, gen_name)

        success = run_reinvent4(resolved_config, prefix, working_dir)

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

    generate_summary(all_results)


if __name__ == "__main__":
    main()
