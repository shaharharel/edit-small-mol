#!/usr/bin/env python3
"""
Mol 18 REINVENT4 Expansion — Run all 3 generators (Mol2Mol, De Novo, LibInvent)
focused on Mol 18, then collect, score, merge with Phase 1 results, and generate
a unified HTML report.

Mol 18 (free base): C=CC(=O)N1CCN(c2ncnc(NC3(CC)CCNCC3)n2)CC1
Pyrimidine-piperazine with acrylamide warhead.

Usage:
    conda run --no-capture-output -n quris python -u experiments/run_mol18_reinvent4.py

Phase 2a: REINVENT4 Mol2Mol — analogs of Mol 18
Phase 2b: REINVENT4 De Novo — constrained to similarity to Mol 18
Phase 2c: REINVENT4 LibInvent — R-group decoration on Mol 18 scaffold
Phase 3:  Collect, filter, score with FiLMDelta, merge with Phase 1, generate report
"""

import sys
import os
import gc
import json
import shutil
import subprocess
import time
import warnings
from pathlib import Path
from datetime import datetime

sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
import torch

warnings.filterwarnings("ignore")
os.environ['CUDA_VISIBLE_DEVICES'] = ''
os.environ['RDK_DEPRECATION_WARNING'] = 'off'
torch.backends.mps.is_available = lambda: False

from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors, QED, FilterCatalog, Draw
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
REINVENT4_ROOT = PROJECT_ROOT.parent / "REINVENT4"
RESULTS_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "mol18_expansion"
REINVENT_RESULTS_DIR = RESULTS_DIR / "reinvent4"
PHASE1_RESULTS = RESULTS_DIR / "expansion_results.json"
SCORED_CSV = RESULTS_DIR / "scored_candidates.csv"

MOL18_SMILES = "C=CC(=O)N1CCN(c2ncnc(NC3(CC)CCNCC3)n2)CC1"
MOL18_SEED_FILE = RESULTS_DIR / "mol18_seed.smi"

# Find conda
CONDA_PATH = "/opt/miniconda3/condabin/conda"
for p in ["/opt/miniconda3/bin/conda", "/opt/miniconda3/condabin/conda",
          os.path.expanduser("~/miniconda3/condabin/conda")]:
    if os.path.exists(p):
        CONDA_PATH = p
        break


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


# ═══════════════════════════════════════════════════════════════════════════════
# Config Generation
# ═══════════════════════════════════════════════════════════════════════════════

def create_mol2mol_config(output_path):
    """Create Mol2Mol config focused on Mol 18 as seed."""
    config = f'''# REINVENT4 Mol2Mol: Mol 18 Expansion
run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol18_mol2mol"
json_out_config = "_mol18_mol2mol.json"

[parameters]
summary_csv_prefix = "mol18_mol2mol"
use_checkpoint = false
purge_memories = false

prior_file = "{REINVENT4_ROOT}/priors/mol2mol_medium_similarity.prior"
agent_file = "{REINVENT4_ROOT}/priors/mol2mol_medium_similarity.prior"
smiles_file = "{MOL18_SEED_FILE}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

# Stage 1: Generate close analogs
[[stage]]
chkpt_file = "mol18_mol2mol_s1.chkpt"
termination = "simple"
max_score = 0.7
min_steps = 10
max_steps = 50

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.55
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT}/experiments/reinvent4_film_scorer.py"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.TanimotoDistance]
[[stage.scoring.component.TanimotoDistance.endpoint]]
name = "Similarity to Mol18"
weight = 0.15
params.smiles = ["{MOL18_SMILES}"]
params.radius = 3
params.use_counts = true
params.use_features = true

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.15

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]

# Stage 2: Push potency, relax similarity
[[stage]]
chkpt_file = "mol18_mol2mol_s2.chkpt"
termination = "simple"
max_score = 0.75
min_steps = 10
max_steps = 30

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.7
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT}/experiments/reinvent4_film_scorer.py"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 8.0
transform.low = 6.0
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.15

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]
'''
    output_path.write_text(config)
    log(f"  Mol2Mol config: {output_path}")


def create_denovo_config(output_path):
    """Create De Novo config with similarity constraint to Mol 18."""
    config = f'''# REINVENT4 De Novo: Mol 18-guided generation
run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol18_denovo"
json_out_config = "_mol18_denovo.json"

[parameters]
summary_csv_prefix = "mol18_denovo"
use_checkpoint = false
purge_memories = false

prior_file = "{REINVENT4_ROOT}/priors/reinvent.prior"
agent_file = "{REINVENT4_ROOT}/priors/reinvent.prior"

batch_size = 128
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

# Stage 1: Explore around Mol 18 chemistry
[[stage]]
chkpt_file = "mol18_denovo_s1.chkpt"
termination = "simple"
max_score = 0.65
min_steps = 50
max_steps = 200

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.5
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT}/experiments/reinvent4_film_scorer.py"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.0
transform.low = 5.0
transform.k = 0.5

# Similarity to Mol 18 — keep de novo anchored
[[stage.scoring.component]]
[stage.scoring.component.TanimotoDistance]
[[stage.scoring.component.TanimotoDistance.endpoint]]
name = "Similarity to Mol18"
weight = 0.25
params.smiles = ["{MOL18_SMILES}"]
params.radius = 3
params.use_counts = true
params.use_features = true

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.15

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.1
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]

# Stage 2: Push potency harder
[[stage]]
chkpt_file = "mol18_denovo_s2.chkpt"
termination = "simple"
max_score = 0.75
min_steps = 25
max_steps = 150

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT}/experiments/reinvent4_film_scorer.py"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 8.0
transform.low = 6.0
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.TanimotoDistance]
[[stage.scoring.component.TanimotoDistance.endpoint]]
name = "Similarity to Mol18"
weight = 0.15
params.smiles = ["{MOL18_SMILES}"]
params.radius = 3
params.use_counts = true
params.use_features = true

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.15

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.1
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]
'''
    output_path.write_text(config)
    log(f"  De Novo config: {output_path}")


def create_libinvent_config(output_path, scaffold_file):
    """Create LibInvent config for Mol 18 scaffold decoration."""
    config = f'''# REINVENT4 LibInvent: Mol 18 scaffold decoration
run_type = "staged_learning"
device = "cpu"
tb_logdir = "tb_mol18_libinvent"
json_out_config = "_mol18_libinvent.json"

[parameters]
summary_csv_prefix = "mol18_libinvent"
use_checkpoint = false
purge_memories = false

prior_file = "{REINVENT4_ROOT}/priors/libinvent.prior"
agent_file = "{REINVENT4_ROOT}/priors/libinvent.prior"
smiles_file = "{scaffold_file}"

batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001

[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4

# Stage 1: Decorate Mol 18 scaffold
[[stage]]
chkpt_file = "mol18_libinvent_s1.chkpt"
termination = "simple"
max_score = 0.65
min_steps = 50
max_steps = 200

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.6
params.executable = "{CONDA_PATH}"
params.args = "run --no-capture-output -n quris python {PROJECT_ROOT}/experiments/reinvent4_film_scorer.py"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.2

[[stage.scoring.component]]
[stage.scoring.component.MolecularWeight]
[[stage.scoring.component.MolecularWeight.endpoint]]
name = "Molecular weight"
weight = 0.05
transform.type = "double_sigmoid"
transform.high = 550.0
transform.low = 200.0
transform.coef_div = 500.0
transform.coef_si = 20.0
transform.coef_se = 20.0

[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = 0.15
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C#C",
    "C(=[O,S])[O,S]",
]
'''
    output_path.write_text(config)
    log(f"  LibInvent config: {output_path}")


def prepare_mol18_scaffold():
    """Prepare Mol 18 scaffold with attachment points for LibInvent.

    Mol 18: C=CC(=O)N1CCN(c2ncnc(NC3(CC)CCNCC3)n2)CC1
    Core: pyrimidine-piperazine. We mark the acrylamide and the piperidine
    amine as attachment points [*:0] and [*:1].
    """
    # Manually define scaffold with attachment points based on Mol 18 structure
    # The pyrimidine-piperazine core with two decoration sites:
    # [*:0] at the acrylamide position (N of piperazine)
    # [*:1] at the amine on the piperidine ring
    scaffold_variants = [
        # Core with 2 attachment points
        "[*:0]N1CCN(c2ncnc(NC3(CC)CCNCC3)n2)CC1",
        # Pyrimidine-piperazine core only (both sides open)
        "[*:0]N1CCN(c2ncnc(N[*:1])n2)CC1",
        # Core with piperidine, one attachment
        "[*:0]N1CCN(c2ncnc(NC3CCNCC3)n2)CC1",
    ]

    # Use the 2-attachment-point version for maximum diversity
    chosen = scaffold_variants[1]
    log(f"  LibInvent scaffold: {chosen}")

    # Validate it parses
    mol = Chem.MolFromSmiles(chosen)
    if mol is None:
        log("  WARNING: Scaffold doesn't parse, trying alternatives...")
        for alt in scaffold_variants:
            mol = Chem.MolFromSmiles(alt)
            if mol is not None:
                chosen = alt
                log(f"  Using alternative: {chosen}")
                break

    if mol is None:
        # Fall back to existing scaffold file
        existing = PROJECT_ROOT / "data" / "zap70_scaffolds.smi"
        if existing.exists():
            log(f"  Falling back to existing scaffold file: {existing}")
            return existing
        raise RuntimeError("No valid scaffold could be prepared")

    scaffold_file = REINVENT_RESULTS_DIR / "mol18_scaffold.smi"
    scaffold_file.write_text(chosen + "\n")
    return scaffold_file


# ═══════════════════════════════════════════════════════════════════════════════
# REINVENT4 Runner
# ═══════════════════════════════════════════════════════════════════════════════

def run_reinvent4(config_path, name, working_dir):
    """Run a single REINVENT4 job."""
    working_dir.mkdir(parents=True, exist_ok=True)
    log_file = working_dir / f"{name}.log"

    log(f"Running REINVENT4 {name}...")
    log(f"  Config: {config_path}")
    log(f"  Working dir: {working_dir}")

    cmd = ["conda", "run", "--no-capture-output", "-n", "quris",
           "reinvent", str(config_path), "-d", "cpu"]

    start = time.time()
    try:
        with open(log_file, "w") as lf:
            proc = subprocess.Popen(
                cmd, cwd=str(working_dir),
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            lines_printed = 0
            for line in proc.stdout:
                lf.write(line)
                # Print periodic progress (every 50 lines)
                lines_printed += 1
                if lines_printed % 50 == 0 or "step" in line.lower() or "stage" in line.lower():
                    sys.stdout.write(f"  [{name}] {line}")
                    sys.stdout.flush()
            proc.wait()

        elapsed = time.time() - start
        log(f"  {name} done in {elapsed/60:.1f} min (exit={proc.returncode})")
        return proc.returncode == 0

    except Exception as e:
        log(f"  {name} FAILED: {e}")
        return False


def collect_reinvent_results(working_dir, prefix):
    """Collect SMILES from REINVENT4 CSV output."""
    csv_files = sorted(working_dir.glob(f"{prefix}*.csv"))
    if not csv_files:
        csv_files = sorted(working_dir.glob("*.csv"))
    if not csv_files:
        log(f"  No CSV output in {working_dir}")
        return []

    all_smiles = {}  # smiles -> best_score
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file, on_bad_lines='skip')
            log(f"  {csv_file.name}: {len(df)} rows")

            smi_col = next((c for c in ["SMILES", "smiles", "Smiles"] if c in df.columns), None)
            if smi_col is None and len(df.columns) > 0:
                # Check if first col looks like SMILES
                smi_col = df.columns[0] if 'smi' in df.columns[0].lower() else df.columns[3] if len(df.columns) > 3 else df.columns[0]

            score_col = next((c for c in ["FiLMDelta pIC50 (raw)", "Score", "total_score"] if c in df.columns), None)

            for _, row in df.iterrows():
                smi = str(row.get(smi_col, "")).strip()
                mol = Chem.MolFromSmiles(smi)
                if mol is not None:
                    can = Chem.MolToSmiles(mol)
                    score = float(row[score_col]) if score_col and pd.notna(row.get(score_col)) else None
                    if can not in all_smiles or (score and (all_smiles[can] is None or score > all_smiles[can])):
                        all_smiles[can] = score

        except Exception as e:
            log(f"  Error reading {csv_file.name}: {e}")

    results = [{"smiles": s, "reinvent_score": sc} for s, sc in all_smiles.items()]
    log(f"  Collected {len(results)} unique molecules")
    return results


# ═══════════════════════════════════════════════════════════════════════════════
# Scoring
# ═══════════════════════════════════════════════════════════════════════════════

def score_with_film_batch(smiles_list, batch_size=200):
    """Score molecules using FiLMDelta external scorer in batches."""
    from rdkit.Chem import AllChem

    log(f"  Scoring {len(smiles_list)} molecules with FiLMDelta scorer...")
    all_scores = []

    for i in range(0, len(smiles_list), batch_size):
        batch = smiles_list[i:i+batch_size]
        input_str = "\n".join(batch) + "\n"

        try:
            result = subprocess.run(
                ["conda", "run", "--no-capture-output", "-n", "quris",
                 "python", str(PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py")],
                input=input_str, capture_output=True, text=True, timeout=600,
            )

            if result.returncode == 0:
                # Parse JSON from stdout (ignore stderr)
                stdout_lines = result.stdout.strip().split('\n')
                json_line = [l for l in stdout_lines if l.strip().startswith('{')]
                if json_line:
                    data = json.loads(json_line[-1])
                    scores = data["payload"]["pIC50"]
                    all_scores.extend(scores)
                else:
                    all_scores.extend([float('nan')] * len(batch))
            else:
                log(f"    Batch {i}: scorer failed (exit={result.returncode})")
                all_scores.extend([float('nan')] * len(batch))

        except subprocess.TimeoutExpired:
            log(f"    Batch {i}: timeout")
            all_scores.extend([float('nan')] * len(batch))
        except Exception as e:
            log(f"    Batch {i}: error {e}")
            all_scores.extend([float('nan')] * len(batch))

        if (i + batch_size) % 1000 == 0:
            log(f"    Scored {min(i+batch_size, len(smiles_list))}/{len(smiles_list)}")

    return all_scores


def compute_properties(smiles_list):
    """Compute molecular properties for filtering and reporting."""
    properties = []
    mol18_fp = AllChem.GetMorganFingerprintAsBitVect(
        Chem.MolFromSmiles(MOL18_SMILES), 2, nBits=2048)

    # PAINS filter
    params = FilterCatalog.FilterCatalogParams()
    params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
    pains_catalog = FilterCatalog.FilterCatalog(params)

    for smi in smiles_list:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            properties.append(None)
            continue
        try:
            fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
            tc_mol18 = DataStructs.TanimotoSimilarity(mol18_fp, fp)
            pains_free = pains_catalog.GetFirstMatch(mol) is None

            props = {
                "MW": Descriptors.MolWt(mol),
                "LogP": Descriptors.MolLogP(mol),
                "HBA": Descriptors.NumHAcceptors(mol),
                "HBD": Descriptors.NumHDonors(mol),
                "TPSA": Descriptors.TPSA(mol),
                "RotBonds": Descriptors.NumRotatableBonds(mol),
                "QED": QED.qed(mol),
                "HeavyAtoms": mol.GetNumHeavyAtoms(),
                "Rings": Descriptors.RingCount(mol),
                "Tc_to_Mol18": tc_mol18,
                "PAINS_free": pains_free,
            }
            properties.append(props)
        except Exception:
            properties.append(None)

    return properties


def filter_candidates(smiles_list, properties, scores):
    """Filter candidates by drug-likeness criteria."""
    passed = []
    for i, (smi, props, score) in enumerate(zip(smiles_list, properties, scores)):
        if props is None:
            continue
        if not props.get("PAINS_free", False):
            continue
        if props.get("QED", 0) < 0.3:
            continue
        if props.get("MW", 999) > 600:
            continue
        if np.isnan(score) if isinstance(score, float) else False:
            continue
        passed.append({
            "smiles": smi,
            "pIC50": score,
            **props,
        })
    return passed


# ═══════════════════════════════════════════════════════════════════════════════
# Report Generation
# ═══════════════════════════════════════════════════════════════════════════════

def mol_to_svg(smiles, size=(250, 200)):
    """Render molecule as inline SVG."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return "<em>Invalid</em>"
    try:
        from rdkit.Chem.Draw import rdMolDraw2D
        drawer = rdMolDraw2D.MolDraw2DSVG(size[0], size[1])
        drawer.drawOptions().addStereoAnnotation = True
        drawer.DrawMolecule(mol)
        drawer.FinishDrawing()
        svg = drawer.GetDrawingText()
        # Remove XML header for inline use
        if '<?xml' in svg:
            svg = svg[svg.index('<svg'):]
        return svg
    except Exception:
        return f"<code>{smiles}</code>"


def generate_unified_report(all_method_results, top_candidates, stats):
    """Generate comprehensive HTML report merging Phase 1 + Phase 2 results."""

    report_file = RESULTS_DIR / "expansion_report.html"

    # Sort top candidates by pIC50
    top_candidates.sort(key=lambda x: -(x.get("pIC50") or 0))
    top50 = top_candidates[:50]
    top10 = top_candidates[:10]

    html = f"""<!DOCTYPE html>
<html><head>
<meta charset="utf-8">
<title>Mol 18 Expansion — Unified Report</title>
<style>
body {{ font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       max-width: 1400px; margin: 0 auto; padding: 20px; background: #f8f9fa; }}
h1 {{ color: #1a1a2e; border-bottom: 3px solid #0066cc; padding-bottom: 10px; }}
h2 {{ color: #16213e; border-bottom: 1px solid #ddd; padding-bottom: 5px; margin-top: 30px; }}
h3 {{ color: #0f3460; }}
.card {{ background: white; border-radius: 8px; padding: 20px; margin: 15px 0; box-shadow: 0 2px 8px rgba(0,0,0,0.1); }}
.seed-card {{ background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; padding: 25px; border-radius: 12px; }}
.seed-card svg {{ background: white; border-radius: 8px; padding: 5px; }}
table {{ border-collapse: collapse; width: 100%; margin: 10px 0; }}
th {{ background: #2c3e50; color: white; padding: 10px 8px; text-align: left; font-size: 0.85em; }}
td {{ padding: 8px; border-bottom: 1px solid #eee; font-size: 0.85em; }}
tr:hover {{ background: #f0f7ff; }}
.rank-1 {{ background: #fff3cd !important; }}
.rank-2 {{ background: #ffecd2 !important; }}
.rank-3 {{ background: #ffe8e8 !important; }}
.method-badge {{ display: inline-block; padding: 2px 8px; border-radius: 12px; font-size: 0.75em; font-weight: bold; }}
.badge-mol2mol {{ background: #4a90d9; color: white; }}
.badge-denovo {{ background: #e85d04; color: white; }}
.badge-libinvent {{ background: #2d6a4f; color: white; }}
.badge-mmp {{ background: #7b2cbf; color: white; }}
.badge-crem {{ background: #c77dff; color: white; }}
.badge-brics {{ background: #9d4edd; color: white; }}
.stat-grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 15px; }}
.stat-box {{ background: white; border-radius: 8px; padding: 15px; text-align: center; box-shadow: 0 2px 4px rgba(0,0,0,0.05); }}
.stat-box .value {{ font-size: 2em; font-weight: bold; color: #0066cc; }}
.stat-box .label {{ font-size: 0.85em; color: #666; }}
.mol-grid {{ display: grid; grid-template-columns: repeat(auto-fill, minmax(280px, 1fr)); gap: 15px; }}
.mol-card {{ background: white; border-radius: 8px; padding: 15px; box-shadow: 0 2px 6px rgba(0,0,0,0.08); }}
.mol-card .rank {{ font-size: 1.5em; font-weight: bold; color: #0066cc; }}
.smiles {{ font-family: monospace; font-size: 0.7em; word-break: break-all; color: #555; max-height: 40px; overflow: hidden; }}
.disclaimer {{ background: #fff3cd; border: 1px solid #ffc107; padding: 15px; border-radius: 8px; margin: 20px 0; }}
</style>
</head><body>

<h1>Mol 18 Expansion Pipeline — Unified Report</h1>
<p><em>Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}</em></p>

<div class="disclaimer">
<strong>Extrapolation Warning:</strong> FiLMDelta predictions become unreliable at Tanimoto &lt; 0.3 to the training set
(extrapolation test: MAE=0.865, Spearman=0.13). Candidates with higher similarity to ZAP70 training molecules
have more reliable predictions.
</div>

<h2>1. Seed Compound: Molecule 18</h2>
<div class="seed-card">
    <div style="display:flex; gap:30px; align-items:center;">
        <div>{mol_to_svg(MOL18_SMILES, (300, 250))}</div>
        <div>
            <h3 style="margin-top:0;">Mol 18 (Free Base)</h3>
            <p><code style="font-size:0.9em;">{MOL18_SMILES}</code></p>
            <p>Pyrimidine-piperazine core with acrylamide warhead</p>
            <p>MW: {Descriptors.MolWt(Chem.MolFromSmiles(MOL18_SMILES)):.1f} |
               LogP: {Descriptors.MolLogP(Chem.MolFromSmiles(MOL18_SMILES)):.2f} |
               QED: {QED.qed(Chem.MolFromSmiles(MOL18_SMILES)):.3f} |
               HBA: {Descriptors.NumHAcceptors(Chem.MolFromSmiles(MOL18_SMILES))} |
               HBD: {Descriptors.NumHDonors(Chem.MolFromSmiles(MOL18_SMILES))}</p>
        </div>
    </div>
</div>

<h2>2. Generation Summary</h2>
<div class="stat-grid">
    <div class="stat-box"><div class="value">{stats['total_generated']}</div><div class="label">Total Generated</div></div>
    <div class="stat-box"><div class="value">{stats['total_after_filter']}</div><div class="label">After Filtering</div></div>
    <div class="stat-box"><div class="value">{stats['n_potent_7']}</div><div class="label">Potent (pIC50 ≥ 7.0)</div></div>
    <div class="stat-box"><div class="value">{stats['n_potent_8']}</div><div class="label">Highly Potent (≥ 8.0)</div></div>
    <div class="stat-box"><div class="value">{stats['n_methods']}</div><div class="label">Generation Methods</div></div>
    <div class="stat-box"><div class="value">{stats.get('best_pIC50', 'N/A')}</div><div class="label">Best Predicted pIC50</div></div>
</div>

<h3>Per-Method Breakdown</h3>
<table>
<tr><th>Method</th><th>Generated</th><th>After Filter</th><th>Mean pIC50</th><th>Max pIC50</th><th>Potent (≥7)</th><th>% Potent</th></tr>
"""

    for method, mstats in stats.get('per_method', {}).items():
        badge_class = f"badge-{method.lower().replace(' ', '').replace('_', '')}"
        if 'mol2mol' in method.lower():
            badge_class = 'badge-mol2mol'
        elif 'denovo' in method.lower() or 'de novo' in method.lower():
            badge_class = 'badge-denovo'
        elif 'libinvent' in method.lower():
            badge_class = 'badge-libinvent'
        elif 'mmp' in method.lower():
            badge_class = 'badge-mmp'
        elif 'crem' in method.lower():
            badge_class = 'badge-crem'
        elif 'brics' in method.lower():
            badge_class = 'badge-brics'

        pct = f"{mstats.get('pct_potent', 0):.1f}%" if mstats.get('n_filtered', 0) > 0 else "N/A"
        html += f"""<tr>
<td><span class="method-badge {badge_class}">{method}</span></td>
<td>{mstats.get('n_generated', 0):,}</td>
<td>{mstats.get('n_filtered', 0):,}</td>
<td>{mstats.get('mean_pIC50', 0):.3f}</td>
<td>{mstats.get('max_pIC50', 0):.3f}</td>
<td>{mstats.get('n_potent', 0):,}</td>
<td>{pct}</td>
</tr>"""

    html += """</table>

<h2>3. Top 10 Recommended Candidates</h2>
<div class="mol-grid">
"""
    for i, cand in enumerate(top10):
        badge_class = 'badge-brics'
        method = cand.get('method', 'unknown')
        if 'mol2mol' in method.lower():
            badge_class = 'badge-mol2mol'
        elif 'denovo' in method.lower():
            badge_class = 'badge-denovo'
        elif 'libinvent' in method.lower():
            badge_class = 'badge-libinvent'
        elif 'mmp' in method.lower():
            badge_class = 'badge-mmp'
        elif 'crem' in method.lower():
            badge_class = 'badge-crem'

        rank_class = f"rank-{i+1}" if i < 3 else ""
        html += f"""<div class="mol-card {rank_class}">
    <div style="display:flex; justify-content:space-between; align-items:center;">
        <span class="rank">#{i+1}</span>
        <span class="method-badge {badge_class}">{method}</span>
    </div>
    <div style="text-align:center; margin:10px 0;">{mol_to_svg(cand['smiles'], (240, 180))}</div>
    <div class="smiles">{cand['smiles']}</div>
    <div style="margin-top:8px;">
        <strong>pIC50: {cand.get('pIC50', 0):.3f}</strong> |
        Tc: {cand.get('Tc_to_Mol18', 0):.3f} |
        QED: {cand.get('QED', 0):.3f}<br>
        MW: {cand.get('MW', 0):.1f} |
        LogP: {cand.get('LogP', 0):.2f} |
        TPSA: {cand.get('TPSA', 0):.1f}
    </div>
</div>"""

    html += """</div>

<h2>4. Top 50 Candidates — Master Table</h2>
<table>
<tr><th>#</th><th>Structure</th><th>SMILES</th><th>Method</th><th>pIC50</th><th>Tc to Mol18</th>
<th>QED</th><th>MW</th><th>LogP</th><th>TPSA</th><th>HBA</th><th>HBD</th></tr>
"""

    for i, cand in enumerate(top50):
        badge_class = 'badge-brics'
        method = cand.get('method', 'unknown')
        if 'mol2mol' in method.lower():
            badge_class = 'badge-mol2mol'
        elif 'denovo' in method.lower():
            badge_class = 'badge-denovo'
        elif 'libinvent' in method.lower():
            badge_class = 'badge-libinvent'
        elif 'mmp' in method.lower():
            badge_class = 'badge-mmp'
        elif 'crem' in method.lower():
            badge_class = 'badge-crem'

        rank_class = f"rank-{i+1}" if i < 3 else ""
        html += f"""<tr class="{rank_class}">
<td>{i+1}</td>
<td style="min-width:120px;">{mol_to_svg(cand['smiles'], (150, 120))}</td>
<td class="smiles" style="max-width:200px;">{cand['smiles']}</td>
<td><span class="method-badge {badge_class}">{method}</span></td>
<td><strong>{cand.get('pIC50', 0):.3f}</strong></td>
<td>{cand.get('Tc_to_Mol18', 0):.3f}</td>
<td>{cand.get('QED', 0):.3f}</td>
<td>{cand.get('MW', 0):.1f}</td>
<td>{cand.get('LogP', 0):.2f}</td>
<td>{cand.get('TPSA', 0):.1f}</td>
<td>{cand.get('HBA', 0)}</td>
<td>{cand.get('HBD', 0)}</td>
</tr>"""

    html += """</table>

<h2>5. Method Comparison Analysis</h2>
<div class="card">
<h3>Key Observations</h3>
<ul>
"""
    # Add method-specific observations
    for method, mstats in stats.get('per_method', {}).items():
        if mstats.get('n_filtered', 0) > 0:
            html += f"<li><strong>{method}</strong>: {mstats['n_filtered']:,} candidates, "
            html += f"mean pIC50={mstats['mean_pIC50']:.3f}, "
            html += f"{mstats.get('pct_potent', 0):.1f}% potent (≥7.0)</li>\n"

    html += """</ul>
</div>

<h2>6. Notes & Caveats</h2>
<div class="card">
<ul>
<li>All pIC50 predictions use FiLMDelta with kinase pretraining + ZAP70 fine-tuning (280 molecules, anchor-based)</li>
<li>Extrapolation test shows FiLMDelta is unreliable at Tc &lt; 0.3 (MAE=0.865, Spearman=0.13)</li>
<li>Candidates with higher Tc to Mol 18 and to ZAP70 training set have more trustworthy predictions</li>
<li>PAINS-flagged, low QED (&lt;0.3), and high MW (&gt;600) molecules have been removed</li>
<li>Phase 1: MMP enumeration, CReM fragment replacement, BRICS recombination</li>
<li>Phase 2: REINVENT4 Mol2Mol, De Novo (similarity-constrained), LibInvent (scaffold decoration)</li>
</ul>
</div>

</body></html>"""

    report_file.write_text(html)
    log(f"Report saved: {report_file} ({len(html)//1024} KB)")
    return report_file


# ═══════════════════════════════════════════════════════════════════════════════
# Main Pipeline
# ═══════════════════════════════════════════════════════════════════════════════

def main():
    REINVENT_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    log("=" * 70)
    log("Mol 18 REINVENT4 Expansion Pipeline")
    log("=" * 70)

    t0 = time.time()

    # ── Phase 2a: Mol2Mol ─────────────────────────────────────────────────
    mol2mol_dir = REINVENT_RESULTS_DIR / "mol2mol"
    mol2mol_config = mol2mol_dir / "mol18_mol2mol.toml"
    mol2mol_dir.mkdir(parents=True, exist_ok=True)

    # Check if already completed
    mol2mol_csvs = list(mol2mol_dir.glob("mol18_mol2mol*.csv"))
    if mol2mol_csvs:
        log("  Mol2Mol: SKIPPING (already completed)")
        mol2mol_ok = True
    else:
        create_mol2mol_config(mol2mol_config)
        mol2mol_ok = run_reinvent4(mol2mol_config, "mol18_mol2mol", mol2mol_dir)

    mol2mol_results = []
    if mol2mol_ok:
        mol2mol_results = collect_reinvent_results(mol2mol_dir, "mol18_mol2mol")
        for r in mol2mol_results:
            r["method"] = "Mol2Mol"
    log(f"  Mol2Mol: {len(mol2mol_results)} molecules")

    # ── Phase 2b: De Novo ─────────────────────────────────────────────────
    denovo_dir = REINVENT_RESULTS_DIR / "denovo"
    denovo_config = denovo_dir / "mol18_denovo.toml"
    denovo_dir.mkdir(parents=True, exist_ok=True)

    denovo_csvs = list(denovo_dir.glob("mol18_denovo*.csv"))
    if denovo_csvs:
        log("  De Novo: SKIPPING (already completed)")
        denovo_ok = True
    else:
        create_denovo_config(denovo_config)
        denovo_ok = run_reinvent4(denovo_config, "mol18_denovo", denovo_dir)

    denovo_results = []
    if denovo_ok:
        denovo_results = collect_reinvent_results(denovo_dir, "mol18_denovo")
        for r in denovo_results:
            r["method"] = "De Novo"
    log(f"  De Novo: {len(denovo_results)} molecules")

    # ── Phase 2c: LibInvent ───────────────────────────────────────────────
    scaffold_file = prepare_mol18_scaffold()
    libinvent_dir = REINVENT_RESULTS_DIR / "libinvent"
    libinvent_config = libinvent_dir / "mol18_libinvent.toml"
    libinvent_dir.mkdir(parents=True, exist_ok=True)
    create_libinvent_config(libinvent_config, scaffold_file)
    libinvent_ok = run_reinvent4(libinvent_config, "mol18_libinvent", libinvent_dir)

    libinvent_results = []
    if libinvent_ok:
        libinvent_results = collect_reinvent_results(libinvent_dir, "mol18_libinvent")
        for r in libinvent_results:
            r["method"] = "LibInvent"
    log(f"  LibInvent: {len(libinvent_results)} molecules")

    # ── Combine Phase 2 results ──────────────────────────────────────────
    phase2_results = mol2mol_results + denovo_results + libinvent_results
    log(f"\n  Phase 2 total: {len(phase2_results)} unique molecules")

    # ── Score Phase 2 with FiLMDelta (with cache) ──────────────────────
    phase2_score_cache = REINVENT_RESULTS_DIR / "phase2_scored_cache.json"
    if phase2_score_cache.exists():
        log("  Loading cached Phase 2 scores...")
        with open(phase2_score_cache) as f:
            score_map = json.load(f)
        for r in phase2_results:
            s = score_map.get(r["smiles"])
            r["pIC50"] = s
    elif phase2_results:
        phase2_smiles = [r["smiles"] for r in phase2_results]
        log("  Scoring Phase 2 molecules with FiLMDelta...")
        film_scores = score_with_film_batch(phase2_smiles)
        score_map = {}
        for r, s in zip(phase2_results, film_scores):
            val = s if not (isinstance(s, float) and np.isnan(s)) else None
            r["pIC50"] = val
            score_map[r["smiles"]] = val
        with open(phase2_score_cache, "w") as f:
            json.dump(score_map, f)
        log(f"  Scores cached to {phase2_score_cache}")

    # ── Load Phase 1 results ─────────────────────────────────────────────
    phase1_candidates = []
    if SCORED_CSV.exists():
        log("  Loading Phase 1 scored candidates...")
        p1_df = pd.read_csv(SCORED_CSV)
        # Phase 1 CSV uses film_pIC50/source/tanimoto_to_mol18 columns
        for _, row in p1_df.iterrows():
            pIC50 = row.get("film_pIC50", row.get("pred_pIC50_mean", row.get("pIC50", None)))
            source = row.get("source", row.get("method", "Phase1"))
            phase1_candidates.append({
                "smiles": row.get("smiles", ""),
                "method": source,
                "pIC50": float(pIC50) if pd.notna(pIC50) else None,
                "MW": row.get("MW", None),
                "LogP": row.get("LogP", None),
                "QED": row.get("QED", None),
                "TPSA": row.get("TPSA", None),
                "HBA": row.get("HBA", None),
                "HBD": row.get("HBD", None),
                "Tc_to_Mol18": row.get("tanimoto_to_mol18", row.get("Tc_to_Mol18", None)),
            })
        log(f"  Phase 1: {len(phase1_candidates)} candidates loaded")

    # ── Compute properties for Phase 2 ───────────────────────────────────
    if phase2_results:
        log("  Computing properties for Phase 2...")
        phase2_smiles = [r["smiles"] for r in phase2_results]
        props = compute_properties(phase2_smiles)
        for r, p in zip(phase2_results, props):
            if p:
                r.update(p)

    # ── Filter Phase 2 ───────────────────────────────────────────────────
    phase2_filtered = []
    for r in phase2_results:
        if r.get("pIC50") is None:
            continue
        if isinstance(r["pIC50"], float) and np.isnan(r["pIC50"]):
            continue
        if not r.get("PAINS_free", True):
            continue
        if r.get("QED", 1) < 0.3:
            continue
        if r.get("MW", 0) > 600:
            continue
        phase2_filtered.append(r)

    log(f"  Phase 2 after filtering: {len(phase2_filtered)}")

    # ── Merge all candidates ─────────────────────────────────────────────
    all_candidates = phase1_candidates + phase2_filtered

    # Deduplicate by SMILES, keep best score
    seen = {}
    for c in all_candidates:
        smi = c.get("smiles", "")
        if not smi:
            continue
        if smi not in seen or (c.get("pIC50") or 0) > (seen[smi].get("pIC50") or 0):
            seen[smi] = c
    all_candidates = list(seen.values())

    # Remove Mol 18 itself
    mol18_can = Chem.MolToSmiles(Chem.MolFromSmiles(MOL18_SMILES))
    all_candidates = [c for c in all_candidates if c.get("smiles") != mol18_can]

    log(f"  Total unique candidates after merge: {len(all_candidates)}")

    # ── Compute statistics ───────────────────────────────────────────────
    valid_scores = [c["pIC50"] for c in all_candidates if c.get("pIC50") and not np.isnan(c["pIC50"])]

    # Per-method stats
    method_groups = {}
    for c in all_candidates:
        m = c.get("method", "unknown")
        if m not in method_groups:
            method_groups[m] = []
        method_groups[m].append(c)

    per_method = {}
    for method, cands in method_groups.items():
        scores = [c["pIC50"] for c in cands if c.get("pIC50") and not np.isnan(c["pIC50"])]
        per_method[method] = {
            "n_generated": len(cands),  # after filter for phase2
            "n_filtered": len(scores),
            "mean_pIC50": float(np.mean(scores)) if scores else 0,
            "max_pIC50": float(np.max(scores)) if scores else 0,
            "n_potent": sum(1 for s in scores if s >= 7.0),
            "pct_potent": 100 * sum(1 for s in scores if s >= 7.0) / len(scores) if scores else 0,
        }

    stats = {
        "total_generated": len(all_candidates),
        "total_after_filter": len(valid_scores),
        "n_potent_7": sum(1 for s in valid_scores if s >= 7.0),
        "n_potent_8": sum(1 for s in valid_scores if s >= 8.0),
        "n_methods": len(per_method),
        "best_pIC50": f"{max(valid_scores):.3f}" if valid_scores else "N/A",
        "per_method": per_method,
    }

    # ── Generate report ──────────────────────────────────────────────────
    log("\nGenerating unified report...")
    generate_unified_report(all_candidates, all_candidates, stats)

    # ── Save combined results ────────────────────────────────────────────
    combined_results = {
        "timestamp": datetime.now().isoformat(),
        "seed": MOL18_SMILES,
        "stats": stats,
        "phase2_generators": {
            "mol2mol": {"n_raw": len(mol2mol_results), "success": mol2mol_ok},
            "denovo": {"n_raw": len(denovo_results), "success": denovo_ok},
            "libinvent": {"n_raw": len(libinvent_results), "success": libinvent_ok},
        },
        "total_candidates": len(all_candidates),
        "top_50": [
            {k: v for k, v in c.items() if k != "PAINS_free"}
            for c in sorted(all_candidates, key=lambda x: -(x.get("pIC50") or 0))[:50]
        ],
    }

    combined_file = RESULTS_DIR / "combined_expansion_results.json"
    with open(combined_file, "w") as f:
        json.dump(combined_results, f, indent=2, default=str)
    log(f"Combined results: {combined_file}")

    # ── Save full scored CSV ─────────────────────────────────────────────
    combined_csv = RESULTS_DIR / "all_scored_candidates.csv"
    df = pd.DataFrame(all_candidates)
    df = df.sort_values("pIC50", ascending=False)
    df.to_csv(combined_csv, index=False)
    log(f"All candidates CSV: {combined_csv} ({len(df)} rows)")

    elapsed = time.time() - t0
    log(f"\n{'='*70}")
    log(f"Pipeline complete in {elapsed/60:.1f} min")
    log(f"  Total candidates: {len(all_candidates)}")
    log(f"  Potent (≥7.0): {stats['n_potent_7']}")
    log(f"  Best pIC50: {stats['best_pIC50']}")
    log(f"  Report: {RESULTS_DIR / 'expansion_report.html'}")
    log(f"{'='*70}")


if __name__ == "__main__":
    main()
