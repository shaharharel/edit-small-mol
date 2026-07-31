"""Build 3 RL tomls for Track B' Mol1-anchored RL on EXP6_V5 config.

Based on experiments/exp6_v5_rl.toml. Only changes:
  - smiles_file: 3 variants (Mol1 only, ZAP70+Mol1, Kinase+ZAP70x5+Mol1x20)
  - prior_file, agent_file, chkpt_file: rewritten with cohort tag
  - summary_csv_prefix, tb_logdir, json_out_config: per-cohort

Writes to experiments/mol1_rl_tomls/{cohort_tag}.toml plus a launcher shard.

Tomls use REMOTE absolute paths (/home/shaharh_quris_ai/...) so REINVENT4 finds
its inputs on the GPU VM.
"""
from __future__ import annotations

import json
from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "experiments" / "mol1_rl_tomls"
REMOTE_HOME = "/home/shaharh_quris_ai"
REMOTE_PROJ = f"{REMOTE_HOME}/edit-small-mol"

# Source toml — exp6_v5 was chosen by user
SOURCE_TOML = PROJECT / "experiments" / "exp6_v5_rl.toml"

# Three RL experiments
COHORTS = [
    {
        "tag": "mol1RL_v5_seed_mol1_only",
        "seed_file": "data/mol1_rl_seeds/seed_mol1_only.smi",
        "description": "Mol1 only (1 seed)",
    },
    {
        "tag": "mol1RL_v5_seed_zap70_all_plus_mol1",
        "seed_file": "data/mol1_rl_seeds/seed_zap70_all_plus_mol1.smi",
        "description": "All 280 ZAP70 + Mol1 (281 unique seeds)",
    },
    {
        "tag": "mol1RL_v5_seed_kinase_zap70x5_mol1x20",
        "seed_file": "data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20.smi",
        "description": "Kinase panel + ZAP70 x5 + Mol1 x20 (~35K lines)",
    },
]


def build_toml(cohort_tag: str, seed_file_rel: str) -> str:
    """Build Mol1-RL toml based on exp6_v5_rl.toml structure."""
    seed_path = f"{REMOTE_PROJ}/{seed_file_rel}"
    prior_path = f"{REMOTE_PROJ}/models/reinvent4_mol2mol_warhead_tokens.prior"
    scorer = f"{REMOTE_PROJ}/experiments/reinvent4_film_scorer.py"
    py = f"{REMOTE_HOME}/miniconda3/envs/quris/bin/python"
    chkpt_name = f"{cohort_tag}_stage1.chkpt"

    return f"""# Auto-generated Mol1-anchored RL: {cohort_tag}
# Based on exp6_v5_rl.toml — only seed file changes.

run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_{cohort_tag}"
json_out_config = "_{cohort_tag}.json"

[parameters]
summary_csv_prefix = "{cohort_tag}"
use_checkpoint = false
purge_memories = false

prior_file = "{prior_path}"
agent_file = "{prior_path}"
smiles_file = "{seed_path}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = 16
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

[[stage]]
chkpt_file = "{chkpt_name}"
termination = "simple"
max_score = 0.7
min_steps = 15
max_steps = 50

[stage.scoring]
type = "geometric_mean"

[[stage.scoring.component]]
[stage.scoring.component.ExternalProcess]
[[stage.scoring.component.ExternalProcess.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.50
params.executable = "{py}"
params.args = "{scorer}"
params.property = "pIC50"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "Acrylamide retained"
weight = 0.40
params.smarts = "[CH2]=[CH]C(=O)N"
params.use_chirality = false

[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for c in COHORTS:
        toml_text = build_toml(c["tag"], c["seed_file"])
        toml_path = OUT_DIR / f"{c['tag']}.toml"
        toml_path.write_text(toml_text)
        # QA: file readable + has expected sections
        assert "smiles_file" in toml_path.read_text()
        assert c["seed_file"] in toml_path.read_text()
        print(f"  wrote {toml_path.relative_to(PROJECT)}  ({c['description']})")

    # Build A100 launcher
    launcher = OUT_DIR / "run_a100.sh"
    launcher.write_text(f"""#!/bin/bash
# Track B' Mol1-anchored RL on A100 — 3 RL configs sequentially
set -eo pipefail
LOG="$HOME/mol1_rl_a100.log"
date -u +"[%FT%TZ] Track B mol1-RL queue start" | tee -a "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

PROJ={REMOTE_PROJ}
TOMLS=$PROJ/experiments/mol1_rl_tomls

COHORTS=(
{chr(10).join('  "' + c['tag'] + '"' for c in COHORTS)}
)

for cohort in "${{COHORTS[@]}}"; do
  work=$PROJ/results/paper_evaluation/mol1_rl/$cohort
  mkdir -p "$work"
  cd "$work"
  cp $TOMLS/$cohort.toml ./$cohort.toml
  date -u +"[%FT%TZ] === $cohort START ===" | tee -a "$LOG"
  reinvent ./$cohort.toml -d cuda 2>&1 | tee -a "$LOG" || {{
    date -u +"[%FT%TZ] === $cohort FAILED (continuing) ===" | tee -a "$LOG"
    continue
  }}
  date -u +"[%FT%TZ] === $cohort DONE ===" | tee -a "$LOG"
done

date -u +"[%FT%TZ] Track B mol1-RL ALL DONE" | tee -a "$LOG"
""")
    launcher.chmod(0o755)
    print(f"  wrote {launcher.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
