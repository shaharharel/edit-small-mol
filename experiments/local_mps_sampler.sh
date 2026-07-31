#!/bin/bash
# Local MPS-accelerated sampling daemon.
#
# Watches models/rl_checkpoints_b/ for new chkpts. For each new chkpt,
# samples 100K mols on Mol1 anchor using local REINVENT4 + MPS.
# This frees GPU VMs to immediately move on to next RL training.
#
# Run on Mac: nohup bash experiments/local_mps_sampler.sh > local_sampler.log 2>&1 &
set -o pipefail
PROJ="/Users/shaharharel/Documents/github/edit-small-mol"
CONDA_ENV="quris"
ANCHOR="$PROJ/data/mol1_only_anchor.smi"
CHKPT_DIR="$PROJ/models/rl_checkpoints_b"
OUT_DIR="$PROJ/data/local_sampling"
LOG="$PROJ/local_sampler.log"

mkdir -p "$OUT_DIR"
source /opt/miniconda3/etc/profile.d/conda.sh
conda activate $CONDA_ENV

log() { date +"[%FT%T] $*" | tee -a "$LOG"; }
log "=== Local MPS sampler START ==="

# Daemon loop
while true; do
  if [ ! -d "$CHKPT_DIR" ]; then
    sleep 60
    continue
  fi
  for chkpt in "$CHKPT_DIR"/*.chkpt; do
    [ -f "$chkpt" ] || continue
    name=$(basename "$chkpt" .chkpt)
    tag=${name%_stage1}    # strip trailing _stage1
    sample_csv="$OUT_DIR/${tag}_mps_sample.csv"
    sentinel="$OUT_DIR/${tag}.DONE"
    [ -f "$sentinel" ] && continue
    if [ -f "$sample_csv" ] && [ "$(wc -l < "$sample_csv" 2>/dev/null)" -gt 100 ]; then
      touch "$sentinel"
      continue
    fi
    log "Sampling 100K from $tag on Mol1 (MPS)"
    toml="$OUT_DIR/${tag}.toml"
    cat > "$toml" <<EOF
run_type = "sampling"
device = "mps"
json_out_config = "$OUT_DIR/${tag}_sampling.json"

[parameters]
model_file = "$chkpt"
smiles_file = "$ANCHOR"
sample_strategy = "multinomial"
temperature = 1.0
output_file = "$sample_csv"
num_smiles = 100000
unique_molecules = true
randomize_smiles = true
EOF
    cd "$OUT_DIR"
    /opt/miniconda3/envs/$CONDA_ENV/bin/reinvent "$toml" -d mps 2>&1 | tee -a "$LOG" || {
      log "MPS sampling failed for $tag (falling back to CPU)"
      /opt/miniconda3/envs/$CONDA_ENV/bin/reinvent "$toml" -d cpu 2>&1 | tee -a "$LOG" || log "CPU sampling also failed"
    }
    n=$(wc -l < "$sample_csv" 2>/dev/null || echo 0)
    log "$tag local sample DONE: $n rows"
    touch "$sentinel"
  done
  sleep 300  # check for new chkpts every 5 min
done
