#!/usr/bin/env python3
"""
ai-gpu overnight REINVENT4 job runner.

Generates configs + runs 7 REINVENT4 jobs sequentially on the T4 GPU:

  Tier 3 v3 (uncertainty-aware FiLMDelta + warhead gate):
    1. LibInvent locked-warhead scaffold + uncertainty-aware reward
    2. Mol2Mol + warhead gate + uncertainty-aware reward
    3. De Novo + warhead gate + uncertainty-aware reward + Tc-to-Mol1

  Tier 4 (unconstrained, post-filter only):
    4. Big De Novo unconstrained (max_steps=300)
    5. Big Mol2Mol unconstrained (max_steps=300)

  Method A/B (FiLMDelta-driven, no warhead constraint):
    6. Method A: De Novo with FiLMDelta-uncertainty as primary reward (weight=0.7), no warhead
    7. Method B: Mol2Mol with FiLMDelta-uncertainty + Tc-to-Mol1 reward (no warhead)

After each REINVENT4 job, collects + filters + scores SMILES.
Final result aggregated into a single JSON.

Usage (on ai-gpu):
    cd ~/edit-small-mol-rsync
    /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python -u experiments/overnight_aigpu_reinvent_jobs.py
"""

import sys
import os
import json
import subprocess
import warnings
from pathlib import Path
from datetime import datetime
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

REINVENT4_ROOT = Path("/home/shaharh_quris_ai/REINVENT4")
PRIOR_LIBINVENT = REINVENT4_ROOT / "priors" / "libinvent.prior"
PRIOR_MOL2MOL = REINVENT4_ROOT / "priors" / "mol2mol_medium_similarity.prior"
PRIOR_DENOVO = REINVENT4_ROOT / "priors" / "reinvent.prior"

CONDA = "/home/shaharh_quris_ai/miniconda3/condabin/conda"
PYTHON = "/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python"
REINVENT_BIN = "/home/shaharh_quris_ai/miniconda3/envs/quris/bin/reinvent"

# Choose which scorer to use — uncertainty-aware ensemble is preferred when available
ENSEMBLE_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "reinvent4_film_ensemble"
FILM_SCORER_UNCERTAIN = PROJECT_ROOT / "experiments" / "reinvent4_film_scorer_uncertainty.py"
FILM_SCORER_SINGLE = PROJECT_ROOT / "experiments" / "reinvent4_film_scorer.py"

MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
WARHEAD_SMARTS = "[CH2]=[CH]C(=O)[N;!H2]"
SCAFFOLD_LOCKED = "C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1"

OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "aigpu_overnight"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def log(msg):
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def has_uncertainty_ensemble():
    if not ENSEMBLE_DIR.exists():
        return False
    seeds = list(ENSEMBLE_DIR.glob("film_seed*.pt"))
    return len(seeds) >= 3


# ── Config templates ──────────────────────────────────────────────────────────

def base_diversity_filter():
    return '''
[diversity_filter]
type = "IdenticalMurckoScaffold"
bucket_size = 25
minscore = 0.4
minsimilarity = 0.4
'''


def warhead_gate_component():
    return f'''
[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "warhead intact"
weight = 1.0
params.smarts = "{WARHEAD_SMARTS}"
params.use_chirality = false
'''


def film_uncertain_component(weight=0.6):
    """FiLMDelta uncertainty-aware reward. Falls back to single-seed if ensemble missing."""
    scorer = FILM_SCORER_UNCERTAIN if has_uncertainty_ensemble() else FILM_SCORER_SINGLE
    return f'''
[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = {weight}
params.executable = "{CONDA}"
params.args = "run --no-capture-output -n quris python {scorer}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5
'''


def qed_component(weight=0.25):
    return f'''
[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = {weight}
'''


def custom_alerts(weight=0.15):
    return f'''
[[stage.scoring.component]]
[stage.scoring.component.custom_alerts]
[[stage.scoring.component.custom_alerts.endpoint]]
name = "Unwanted SMARTS"
weight = {weight}
params.smarts = [
    "[*;r{{8-17}}]",
    "[#8][#8]",
    "[#6;+]",
    "[#16][#16]",
    "[#7;!n][S;!$(S(=O)=O)]",
    "[#7;!n][#7;!n]",
    "C(=[O,S])[O,S]",
]
'''


def tc_to_mol1_component(weight=0.5, low=0.2, high=0.6, reverse=True):
    return f'''
[[stage.scoring.component]]
[stage.scoring.component.TanimotoDistance]
[[stage.scoring.component.TanimotoDistance.endpoint]]
name = "similarity to Mol 1"
weight = {weight}
params.smiles = ["{MOL1_SMILES}"]
params.radius = 2
params.use_counts = false
params.use_features = false
transform.type = "{"reverse_sigmoid" if reverse else "sigmoid"}"
transform.high = {high}
transform.low = {low}
transform.k = 0.5
'''


# ── Job-specific configs ──────────────────────────────────────────────────────

def write_libinvent_locked(out_dir, max_steps=200):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_file = out_dir / "scaffold.smi"
    seed_file.write_text(SCAFFOLD_LOCKED + "\n")
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_libinvent_locked"
json_out_config = "_libinvent_locked.json"

[parameters]
summary_csv_prefix = "libinvent_locked"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_LIBINVENT}"
agent_file = "{PRIOR_LIBINVENT}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "libinvent_locked.chkpt"
termination = "simple"
max_score = 0.85
min_steps = 30
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{film_uncertain_component(0.7)}
{qed_component(0.2)}
{custom_alerts(0.15)}
'''
    cfg_file = out_dir / "libinvent_locked.toml"
    cfg_file.write_text(cfg)
    return cfg_file


def write_mol2mol_warhead(out_dir, max_steps=200):
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_file = out_dir / "seed.smi"
    seed_file.write_text(MOL1_SMILES + "\n")
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_mol2mol_warhead"
json_out_config = "_mol2mol_warhead.json"

[parameters]
summary_csv_prefix = "mol2mol_warhead"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_MOL2MOL}"
agent_file = "{PRIOR_MOL2MOL}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
distance_threshold = 100
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "mol2mol_warhead.chkpt"
termination = "simple"
max_score = 0.85
min_steps = 30
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{warhead_gate_component()}
{film_uncertain_component(0.6)}
{qed_component(0.25)}
{custom_alerts(0.15)}
'''
    cfg_file = out_dir / "mol2mol_warhead.toml"
    cfg_file.write_text(cfg)
    return cfg_file


def write_denovo_warhead(out_dir, max_steps=200):
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_denovo_warhead"
json_out_config = "_denovo_warhead.json"

[parameters]
summary_csv_prefix = "denovo_warhead"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_DENOVO}"
agent_file = "{PRIOR_DENOVO}"
batch_size = 64
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "denovo_warhead.chkpt"
termination = "simple"
max_score = 0.85
min_steps = 30
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{warhead_gate_component()}
{tc_to_mol1_component(0.4, low=0.2, high=0.6, reverse=True)}
{film_uncertain_component(0.5)}
{qed_component(0.2)}
{custom_alerts(0.15)}
'''
    cfg_file = out_dir / "denovo_warhead.toml"
    cfg_file.write_text(cfg)
    return cfg_file


def write_tier4_denovo_uncon(out_dir, max_steps=300):
    """Tier 4: De Novo unconstrained — FiLMDelta + QED only, no warhead, no Tc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_tier4_denovo"
json_out_config = "_tier4_denovo.json"

[parameters]
summary_csv_prefix = "tier4_denovo"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_DENOVO}"
agent_file = "{PRIOR_DENOVO}"
batch_size = 96
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "tier4_denovo.chkpt"
termination = "simple"
max_score = 0.9
min_steps = 50
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{film_uncertain_component(0.7)}
{qed_component(0.2)}
{custom_alerts(0.15)}
'''
    cfg_file = out_dir / "tier4_denovo.toml"
    cfg_file.write_text(cfg)
    return cfg_file


def write_tier4_mol2mol_uncon(out_dir, max_steps=300):
    """Tier 4: Mol2Mol unconstrained — Tc-to-Mol1 + FiLMDelta only."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_file = out_dir / "seed.smi"
    seed_file.write_text(MOL1_SMILES + "\n")
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_tier4_mol2mol"
json_out_config = "_tier4_mol2mol.json"

[parameters]
summary_csv_prefix = "tier4_mol2mol"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_MOL2MOL}"
agent_file = "{PRIOR_MOL2MOL}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
distance_threshold = 100
batch_size = 96
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "tier4_mol2mol.chkpt"
termination = "simple"
max_score = 0.9
min_steps = 50
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{film_uncertain_component(0.7)}
{qed_component(0.2)}
{custom_alerts(0.15)}
'''
    cfg_file = out_dir / "tier4_mol2mol.toml"
    cfg_file.write_text(cfg)
    return cfg_file


def write_method_a_film_driven(out_dir, max_steps=300):
    """Method A: De Novo, FiLMDelta-uncertainty as primary reward (heavy weight),
    no warhead, no Tc."""
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_method_a"
json_out_config = "_method_a.json"

[parameters]
summary_csv_prefix = "method_a_filmdriven_denovo"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_DENOVO}"
agent_file = "{PRIOR_DENOVO}"
batch_size = 96
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "method_a.chkpt"
termination = "simple"
max_score = 0.95
min_steps = 50
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{film_uncertain_component(0.85)}
{qed_component(0.15)}
{custom_alerts(0.1)}
'''
    cfg_file = out_dir / "method_a.toml"
    cfg_file.write_text(cfg)
    return cfg_file


def write_method_b_film_driven(out_dir, max_steps=300):
    """Method B: Mol2Mol, FiLMDelta-uncertainty + Tc-to-Mol1, no warhead."""
    out_dir.mkdir(parents=True, exist_ok=True)
    seed_file = out_dir / "seed.smi"
    seed_file.write_text(MOL1_SMILES + "\n")
    cfg = f'''run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_method_b"
json_out_config = "_method_b.json"

[parameters]
summary_csv_prefix = "method_b_filmdriven_mol2mol"
use_checkpoint = false
purge_memories = false

prior_file = "{PRIOR_MOL2MOL}"
agent_file = "{PRIOR_MOL2MOL}"
smiles_file = "{seed_file}"
sample_strategy = "multinomial"
distance_threshold = 100
batch_size = 96
unique_sequences = true
randomize_smiles = true

[learning_strategy]
type = "dap"
sigma = 128
rate = 0.0001
{base_diversity_filter()}
[[stage]]
chkpt_file = "method_b.chkpt"
termination = "simple"
max_score = 0.95
min_steps = 50
max_steps = {max_steps}

[stage.scoring]
type = "geometric_mean"
{film_uncertain_component(0.7)}
{tc_to_mol1_component(0.3, low=0.15, high=0.5, reverse=True)}
{qed_component(0.15)}
{custom_alerts(0.1)}
'''
    cfg_file = out_dir / "method_b.toml"
    cfg_file.write_text(cfg)
    return cfg_file


# ── Runner ────────────────────────────────────────────────────────────────────

def run_reinvent(cfg_path, working_dir, name):
    log(f"=== START: {name} ({cfg_path}) ===")
    cmd = [REINVENT_BIN, str(cfg_path), "-d", "cuda"]
    log_file = working_dir / f"{name}.log"
    start = time.time()
    with open(log_file, "w") as lf:
        proc = subprocess.run(cmd, cwd=str(working_dir), stdout=lf, stderr=subprocess.STDOUT,
                              text=True)
    elapsed = (time.time() - start) / 60
    log(f"=== END: {name} exit={proc.returncode} elapsed={elapsed:.1f} min ===")
    return proc.returncode == 0


def collect_smiles_from_csvs(working_dir, prefix):
    smis = set()
    for csv in working_dir.glob(f"{prefix}*.csv"):
        try:
            df = pd.read_csv(csv)
            for col in ["SMILES", "Smiles", "smiles"]:
                if col in df.columns:
                    for s in df[col].dropna().astype(str).tolist():
                        smis.add(s)
                    break
        except Exception as e:
            log(f"  WARN reading {csv}: {e}")
    return list(smis)


def main():
    log("=" * 70)
    log("AI-GPU OVERNIGHT REINVENT4 RUNNER")
    log("=" * 70)
    log(f"Started: {datetime.now().isoformat()}")
    log(f"Uncertainty ensemble available: {has_uncertainty_ensemble()}")

    if not has_uncertainty_ensemble():
        log("WARN: ensemble checkpoints not found; falling back to single-seed scorer")
        log(f"      (looking in {ENSEMBLE_DIR})")

    if not REINVENT4_ROOT.exists() or not PRIOR_LIBINVENT.exists():
        log(f"ERROR: REINVENT4 or priors missing under {REINVENT4_ROOT}")
        return

    # Order matters: Tier 3 (warhead-gated) first since they're most useful;
    # Tier 4 + Methods A/B can be killed if time runs out.
    jobs = [
        ("libinvent_locked", write_libinvent_locked, OUT_DIR / "libinvent_locked", 200),
        ("mol2mol_warhead",  write_mol2mol_warhead,  OUT_DIR / "mol2mol_warhead",  200),
        ("denovo_warhead",   write_denovo_warhead,   OUT_DIR / "denovo_warhead",   200),
        ("tier4_denovo",     write_tier4_denovo_uncon, OUT_DIR / "tier4_denovo",   300),
        ("tier4_mol2mol",    write_tier4_mol2mol_uncon, OUT_DIR / "tier4_mol2mol", 300),
        ("method_a",         write_method_a_film_driven, OUT_DIR / "method_a",     300),
        ("method_b",         write_method_b_film_driven, OUT_DIR / "method_b",     300),
    ]

    job_results = []
    for name, write_fn, work_dir, max_steps in jobs:
        try:
            cfg = write_fn(work_dir, max_steps=max_steps)
            ok = run_reinvent(cfg, work_dir, name)
            smis = collect_smiles_from_csvs(work_dir, name)
            job_results.append({"name": name, "config": str(cfg), "ok": ok,
                                "n_smiles": len(smis), "work_dir": str(work_dir)})
            (work_dir / "smiles.txt").write_text("\n".join(smis))
        except Exception as e:
            log(f"  ERROR in job {name}: {e}")
            job_results.append({"name": name, "ok": False, "error": str(e)})

    # Save aggregated summary
    out_summary = OUT_DIR / "summary.json"
    out_summary.write_text(json.dumps({
        "timestamp": datetime.now().isoformat(),
        "uncertainty_ensemble_used": has_uncertainty_ensemble(),
        "jobs": job_results,
    }, indent=2))
    log(f"Summary → {out_summary}")
    log(f"Finished: {datetime.now().isoformat()}")


if __name__ == "__main__":
    main()
