"""Rebuild Track B' tomls using REST FiLM scoring (no subprocess overhead).

Produces:
  experiments/mol1_rl_tomls_rest/{cohort}.toml — REST-based RL configs
  experiments/mol1_rl_tomls_rest/run_b2_rest.sh — V100 (ai-gpu) B'-2 launcher
  experiments/mol1_rl_tomls_rest/run_b3_rest.sh — A100 B'-3 launcher
  experiments/mol1_rl_tomls_rest/start_film_server.sh — server launcher

Each cohort's launcher: start FiLM server in background → run reinvent → cleanup.
"""
from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "experiments" / "mol1_rl_tomls_rest"
REMOTE_HOME = "/home/shaharh_quris_ai"
REMOTE_PROJ = f"{REMOTE_HOME}/edit-small-mol"

COHORTS = [
    ("mol1RL_v5_seed_mol1_only_rest", "data/mol1_rl_seeds/seed_mol1_only.smi"),
    ("mol1RL_v5_seed_zap70_all_plus_mol1_rest", "data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi"),
    ("mol1RL_v5_seed_kinase_zap70x5_mol1x20_rest", "data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20_clean.smi"),
]

FILM_SERVER_PORT = 8088


def build_toml(cohort_tag: str, seed_file_rel: str) -> str:
    seed_path = f"{REMOTE_PROJ}/{seed_file_rel}"
    prior_path = f"{REMOTE_PROJ}/models/reinvent4_mol2mol_warhead_tokens.prior"
    chkpt_name = f"{cohort_tag}_stage1.chkpt"

    return f"""# Auto-generated REST-based Mol1-RL toml: {cohort_tag}
# FiLM scoring via persistent HTTP server on localhost:{FILM_SERVER_PORT}
# (replaces ExternalProcess subprocess scoring — ~15x faster RL steps)

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

# FiLMDelta pIC50 — via persistent REST server
[[stage.scoring.component]]
[stage.scoring.component.REST]
[[stage.scoring.component.REST.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.50
params.server_url = "http://127.0.0.1"
params.server_port = {FILM_SERVER_PORT}
params.server_endpoint = "score"
params.predictor_id = "film_delta"
params.predictor_version = "v1"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

# Acrylamide retained — native, fast
[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "Acrylamide retained"
weight = 0.40
params.smarts = "[CH2]=[CH]C(=O)N"
params.use_chirality = false

# QED — native, fast
[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10
"""


def build_runner(cohort_tag: str, log_label: str) -> str:
    return f"""#!/bin/bash
# Single-cohort launcher with persistent FiLM REST server
# (starts server, runs RL, stops server)
set -eo pipefail
LOG="$HOME/{log_label}.log"
date -u +"[%FT%TZ] {cohort_tag} START (REST)" | tee -a "$LOG"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

# Ensure waitress is installed
pip install waitress 2>&1 | tail -1

# Start FiLM server (background)
pkill -f reinvent4_film_rest_server 2>/dev/null || true
sleep 2
nohup python {REMOTE_PROJ}/experiments/reinvent4_film_rest_server.py \\
  --host 127.0.0.1 --port {FILM_SERVER_PORT} \\
  > $HOME/film_rest_server.log 2>&1 &
SERVER_PID=$!
echo "FiLM server PID=$SERVER_PID" | tee -a "$LOG"

# Wait for server to come up (max 60s)
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:{FILM_SERVER_PORT}/health >/dev/null 2>&1; then
    echo "FiLM server READY after ${{i}}*2s" | tee -a "$LOG"
    break
  fi
  sleep 2
done
curl -s http://127.0.0.1:{FILM_SERVER_PORT}/health | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Run RL
work={REMOTE_PROJ}/results/paper_evaluation/mol1_rl/{cohort_tag}
rm -rf "$work"; mkdir -p "$work"
cd "$work"
cp {REMOTE_PROJ}/experiments/mol1_rl_tomls_rest/{cohort_tag}.toml .
date -u +"[%FT%TZ] RL launching" | tee -a "$LOG"
timeout 5400 reinvent ./{cohort_tag}.toml -d cuda 2>&1 | tee -a "$LOG" || {{
  date -u +"[%FT%TZ] RL FAILED/TIMED OUT" | tee -a "$LOG"
  kill $SERVER_PID 2>/dev/null
  exit 1
}}

date -u +"[%FT%TZ] RL DONE — final health: $(curl -s http://127.0.0.1:{FILM_SERVER_PORT}/health)" | tee -a "$LOG"
kill $SERVER_PID 2>/dev/null
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    for cohort_tag, seed_file_rel in COHORTS:
        toml_path = OUT_DIR / f"{cohort_tag}.toml"
        toml_path.write_text(build_toml(cohort_tag, seed_file_rel))
        print(f"wrote {toml_path.relative_to(PROJECT)}")

    # Three per-cohort runners
    for cohort_tag, _ in COHORTS:
        label = cohort_tag.replace("mol1RL_v5_seed_", "rest_")
        runner_path = OUT_DIR / f"run_{label}.sh"
        runner_path.write_text(build_runner(cohort_tag, label))
        runner_path.chmod(0o755)
        print(f"wrote {runner_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
