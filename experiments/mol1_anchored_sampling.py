"""Generate REINVENT4 sampling tomls + launcher for Mol1-anchored Tier 4 regeneration.

For each model (pre-RL FT priors + post-RL chkpts), creates a sampling toml that
inputs Mol1 only and samples N=50K mols. Output goes to
data/mol1_anchored_tier4/{cohort_tag}/sampling.csv.

Two tracks:
  Track A (pre-RL): 3 base FT priors × 50K each
    - exp2_preRL_mol1_50K   ← reinvent4_mol2mol_covalent_ft.prior
    - exp6_preRL_mol1_50K   ← reinvent4_mol2mol_warhead_tokens.prior
    - exp6v3_preRL_mol1_50K ← reinvent4_mol2mol_warhead_tokens_v2.prior

  Track A (post-RL): 5 RL'd chkpts × 50K each
    - exp2_v2_rl_postRL_mol1_50K
    - exp2_v2_rl_v2_postRL_mol1_50K
    - exp6_v3_postRL_mol1_50K
    - exp6_v4_postRL_mol1_50K
    - exp6_v5_postRL_mol1_50K

Total Track A: 8 cohorts × 50K = 400K mols. Track B' (Mol1-RL retraining) is
a separate step.

Usage:
  python experiments/mol1_anchored_sampling.py write-configs    # writes tomls + bash launcher
  python experiments/mol1_anchored_sampling.py local-smoke      # runs 1 cohort × 1K samples on CPU as sanity
"""
from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
DATA = PROJECT / "data"
MODELS = PROJECT / "models"
RL_CHKPTS = MODELS / "rl_checkpoints"
OUT_ROOT = DATA / "mol1_anchored_tier4"
TOMLS_DIR = PROJECT / "experiments" / "mol1_anchored_tomls"

MOL1_ANCHOR = DATA / "mol1_only_anchor.smi"
NUM_SMILES = 50_000

# (cohort_tag, model_path_relative_to_project, source_kind)
COHORTS = [
    # Pre-RL FT priors
    ("exp2_preRL_mol1_50K",      "models/reinvent4_mol2mol_covalent_ft.prior",            "preRL"),
    ("exp6_preRL_mol1_50K",      "models/reinvent4_mol2mol_warhead_tokens.prior",         "preRL"),
    ("exp6v3_preRL_mol1_50K",    "models/reinvent4_mol2mol_warhead_tokens_v2.prior",      "preRL"),
    # Post-RL chkpts
    ("exp2_v2_rl_postRL_mol1_50K",    "models/rl_checkpoints/exp2_v2_rl_stage1.chkpt",     "postRL"),
    ("exp2_v2_rl_v2_postRL_mol1_50K", "models/rl_checkpoints/exp2_v2_rl_v2_stage1.chkpt",  "postRL"),
    ("exp6_v3_postRL_mol1_50K",       "models/rl_checkpoints/exp6_v3_stage1.chkpt",        "postRL"),
    ("exp6_v4_postRL_mol1_50K",       "models/rl_checkpoints/exp6_v4_stage1.chkpt",        "postRL"),
    ("exp6_v5_postRL_mol1_50K",       "models/rl_checkpoints/exp6_v5_stage1.chkpt",        "postRL"),
]


def sampling_toml(cohort_tag: str, model_path: str, anchor_path: str,
                  out_dir: str, num_smiles: int, device: str = "cuda") -> str:
    """Return REINVENT4 Mol2Mol sampling toml content."""
    return f"""# Auto-generated Mol1-anchored sampling for {cohort_tag}
run_type = "sampling"
device = "{device}"
json_out_config = "{out_dir}/_sampling.json"

[parameters]
model_file = "{model_path}"
smiles_file = "{anchor_path}"
sample_strategy = "multinomial"
temperature = 1.0
output_file = "{out_dir}/sampling.csv"
num_smiles = {num_smiles}
unique_molecules = true
randomize_smiles = true
"""


def write_configs(remote_home: str | None = None, device: str = "cuda") -> None:
    """Write per-cohort sampling tomls and a launcher script.

    If `remote_home` is given, paths are rewritten to live under that home
    (so we can scp the tree to ai-gpu and have REINVENT find the files).
    """
    TOMLS_DIR.mkdir(parents=True, exist_ok=True)
    OUT_ROOT.mkdir(parents=True, exist_ok=True)

    if remote_home:
        proj_str = f"{remote_home}/edit-small-mol"
    else:
        proj_str = str(PROJECT)

    anchor_str = f"{proj_str}/data/mol1_only_anchor.smi"
    launcher_lines = [
        "#!/bin/bash",
        "set -eo pipefail",
        f'LOG="$HOME/mol1_anchored_sampling.log"',
        'echo "[$(date -u +%FT%TZ)] Mol1-anchored sampling queue start" >> "$LOG"',
        'source $HOME/miniconda3/etc/profile.d/conda.sh',
        'conda activate quris',
        '',
    ]

    for cohort_tag, rel_model, kind in COHORTS:
        out_dir = OUT_ROOT / cohort_tag
        out_dir.mkdir(parents=True, exist_ok=True)
        model_str = f"{proj_str}/{rel_model}"
        out_str = f"{proj_str}/data/mol1_anchored_tier4/{cohort_tag}"
        toml_text = sampling_toml(cohort_tag, model_str, anchor_str, out_str,
                                  NUM_SMILES, device=device)
        toml_path = TOMLS_DIR / f"{cohort_tag}.toml"
        toml_path.write_text(toml_text)

        launcher_lines.extend([
            f'echo "[$(date -u +%FT%TZ)] === {cohort_tag} ({kind}) ===" >> "$LOG"',
            f'mkdir -p "{out_str}"',
            f'reinvent "{proj_str}/experiments/mol1_anchored_tomls/{cohort_tag}.toml" -d {device} 2>&1 | tee -a "$LOG"',
            f'echo "[$(date -u +%FT%TZ)] === {cohort_tag} done ===" >> "$LOG"',
            '',
        ])
        print(f"wrote {toml_path.relative_to(PROJECT)}")

    launcher_path = TOMLS_DIR / "run_all.sh"
    launcher_path.write_text("\n".join(launcher_lines))
    launcher_path.chmod(0o755)
    print(f"wrote {launcher_path.relative_to(PROJECT)}")
    print()
    print("To deploy:")
    print(f"  gcloud compute scp --recurse experiments/mol2mol_tomls ai-gpu:~/edit-small-mol/experiments/ --zone=us-central1-c")
    print(f"  gcloud compute ssh ai-gpu --zone=us-central1-c --command='bash ~/edit-small-mol/experiments/mol1_anchored_tomls/run_all.sh'")


def local_smoke():
    """Run 1 cohort with 1K samples on CPU as a quick sanity test."""
    cohort_tag = "exp2_preRL_mol1_1K_smoke"
    out_dir = OUT_ROOT / cohort_tag
    out_dir.mkdir(parents=True, exist_ok=True)
    model_str = str(MODELS / "reinvent4_mol2mol_covalent_ft.prior")
    anchor_str = str(MOL1_ANCHOR)
    out_str = str(out_dir)
    toml_text = sampling_toml(cohort_tag, model_str, anchor_str, out_str,
                              num_smiles=1000, device="cpu")
    toml_path = TOMLS_DIR / f"{cohort_tag}.toml"
    TOMLS_DIR.mkdir(parents=True, exist_ok=True)
    toml_path.write_text(toml_text)
    print(f"toml: {toml_path}")
    print(f"running reinvent on CPU (1K samples)...")
    subprocess.check_call(["reinvent", str(toml_path), "-d", "cpu"])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["write-configs", "write-remote", "local-smoke"])
    args = ap.parse_args()
    if args.mode == "write-configs":
        write_configs()
    elif args.mode == "write-remote":
        write_configs(remote_home="/home/shaharh_quris_ai")
    elif args.mode == "local-smoke":
        local_smoke()
