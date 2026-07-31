"""Build THIQ-acrylamide-reward RL tomls.

Replaces previous B'-2/B'-3 with renamed cohorts that reward the FULL Mol1
pharmacophore (THIQ-acrylamide core SMARTS, not generic acrylamide).

Cohort naming (memorable):
  - thiq_rl_mol1only      : seeds = Mol1 only           (smoke-test, expect 100% THIQ)
  - thiq_rl_zap70         : seeds = 220 ZAP70 + Mol1    (medium pool)
  - thiq_rl_kinase        : seeds = kinase + ZAP70x5 + Mol1x20 (broad pool)

Reward:
  - THIQ-acrylamide MatchingSubstructure : weight 0.40 (was: generic acrylamide)
  - FiLMDelta pIC50 (REST)               : weight 0.50
  - QED                                  : weight 0.10

All use the warhead_tokens prior, REST FiLM scoring, batch_size=16.
"""
from __future__ import annotations

from pathlib import Path

PROJECT = Path(__file__).resolve().parent.parent
OUT_DIR = PROJECT / "experiments" / "thiq_rl_tomls"
REMOTE_HOME = "/home/shaharh_quris_ai"
REMOTE_PROJ = f"{REMOTE_HOME}/edit-small-mol"
FILM_PORT = 8088

# THIQ-acrylamide core SMARTS — Mol1's warhead-bearing pharmacophore
THIQ_ACRYL_SMARTS = "C=CC(=O)N1Cc2ccccc2C1"

COHORTS = [
    # EXP6 base (warhead_tokens prior)
    {"tag": "thiq_rl_mol1only", "seed": "data/mol1_rl_seeds/seed_mol1_only.smi",            "batch": 16, "prior": "warhead_tokens"},
    {"tag": "thiq_rl_zap70",    "seed": "data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi", "batch": 8, "prior": "warhead_tokens"},
    {"tag": "thiq_rl_kinase",   "seed": "data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20_clean.smi", "batch": 16, "prior": "warhead_tokens"},
    # EXP2 base (covalent_ft prior — the one that already gave 98.4% THIQ on Mol1 anchor)
    {"tag": "thiq_rl_exp2_mol1only", "seed": "data/mol1_rl_seeds/seed_mol1_only.smi",       "batch": 16, "prior": "covalent_ft"},
    {"tag": "thiq_rl_exp2_zap70",    "seed": "data/mol1_rl_seeds/seed_zap70_all_plus_mol1_clean.smi", "batch": 8, "prior": "covalent_ft"},
    {"tag": "thiq_rl_exp2_kinase",   "seed": "data/mol1_rl_seeds/seed_kinase_zap70x5_mol1x20_clean.smi", "batch": 16, "prior": "covalent_ft"},
]


def build_toml(tag: str, seed_rel: str, batch: int, prior_kind: str = "warhead_tokens") -> str:
    seed = f"{REMOTE_PROJ}/{seed_rel}"
    prior = f"{REMOTE_PROJ}/models/reinvent4_mol2mol_{prior_kind}.prior"
    return f"""# {tag} — Mol1-anchored RL with THIQ-acrylamide reward
# Reward: FiLM pIC50 (0.50) + THIQ-acryl core (0.40) + QED (0.10)
# SMARTS: {THIQ_ACRYL_SMARTS}

run_type = "staged_learning"
device = "cuda"
tb_logdir = "tb_{tag}"
json_out_config = "_{tag}.json"

[parameters]
summary_csv_prefix = "{tag}"
use_checkpoint = false
purge_memories = false

prior_file = "{prior}"
agent_file = "{prior}"
smiles_file = "{seed}"
sample_strategy = "multinomial"
distance_threshold = 100

batch_size = {batch}
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
chkpt_file = "{tag}_stage1.chkpt"
termination = "simple"
max_score = 0.7
min_steps = 15
max_steps = 50

[stage.scoring]
type = "geometric_mean"

# FiLMDelta pIC50 via REST
[[stage.scoring.component]]
[stage.scoring.component.REST]
[[stage.scoring.component.REST.endpoint]]
name = "FiLMDelta pIC50"
weight = 0.50
params.server_url = "http://127.0.0.1"
params.server_port = {FILM_PORT}
params.server_endpoint = "score"
params.predictor_id = "film_delta"
params.predictor_version = "v1"
transform.type = "sigmoid"
transform.high = 7.5
transform.low = 5.5
transform.k = 0.5

# THIQ-acrylamide core retention — was just "[CH2]=[CH]C(=O)N", now FULL core
[[stage.scoring.component]]
[stage.scoring.component.MatchingSubstructure]
[[stage.scoring.component.MatchingSubstructure.endpoint]]
name = "THIQ-acrylamide core retained"
weight = 0.40
params.smarts = "{THIQ_ACRYL_SMARTS}"
params.use_chirality = false

# QED
[[stage.scoring.component]]
[stage.scoring.component.QED]
[[stage.scoring.component.QED.endpoint]]
name = "QED"
weight = 0.10
"""


def build_runner(tag: str) -> str:
    return f"""#!/bin/bash
# {tag} runner — FiLM REST server + RL
set -eo pipefail
LOG="$HOME/{tag}.log"
date -u +"[%FT%TZ] {tag} START" | tee -a "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

pkill -f reinvent4_film_rest 2>/dev/null || true
sleep 2
nohup python {REMOTE_PROJ}/experiments/reinvent4_film_rest_server.py \\
  --host 127.0.0.1 --port {FILM_PORT} > $HOME/film_rest_server.log 2>&1 &
SERVER_PID=$!
echo "FiLM server PID=$SERVER_PID" | tee -a "$LOG"
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:{FILM_PORT}/health >/dev/null 2>&1 && break
  sleep 2
done
date -u +"[%FT%TZ] FiLM ready" | tee -a "$LOG"

work={REMOTE_PROJ}/results/paper_evaluation/mol1_rl/{tag}
rm -rf "$work"; mkdir -p "$work"; cd "$work"
cp {REMOTE_PROJ}/experiments/thiq_rl_tomls/{tag}.toml .
date -u +"[%FT%TZ] RL launching" | tee -a "$LOG"
timeout 5400 reinvent ./{tag}.toml -d cuda 2>&1 | tee -a "$LOG" || {{
  date -u +"[%FT%TZ] RL FAILED/TIMED OUT" | tee -a "$LOG"
  kill $SERVER_PID 2>/dev/null
  exit 1
}}
date -u +"[%FT%TZ] RL DONE" | tee -a "$LOG"
kill $SERVER_PID 2>/dev/null
"""


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for c in COHORTS:
        toml_path = OUT_DIR / f"{c['tag']}.toml"
        toml_path.write_text(build_toml(c["tag"], c["seed"], c["batch"], c.get("prior", "warhead_tokens")))
        print(f"wrote {toml_path.relative_to(PROJECT)} (batch={c['batch']}, prior={c.get('prior', 'warhead_tokens')})")

        runner_path = OUT_DIR / f"run_{c['tag']}.sh"
        runner_path.write_text(build_runner(c["tag"]))
        runner_path.chmod(0o755)
        print(f"wrote {runner_path.relative_to(PROJECT)}")


if __name__ == "__main__":
    main()
