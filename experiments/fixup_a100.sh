#!/bin/bash
# Fixup script: retry cohorts that failed due to missing mol1_only_anchor.smi
# or missing covalent_ft.prior. Runs in a separate tmux, waits for main
# orchestrator's reinvent to release GPU, then samples + retrains as needed.
set -o pipefail
PROJ="$HOME/edit-small-mol"
LOG="$HOME/a100_fixup.log"
ANCHOR="$PROJ/data/mol1_only_anchor.smi"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

log() { date -u +"[%FT%TZ] [fixup] $*" | tee -a "$LOG"; }
log "=== Fixup START — waits for orchestrator DONE flag ==="

# Wait for orchestrator
for i in $(seq 1 600); do
  [ -f "$HOME/a100_overnight.DONE" ] && break
  sleep 30
done

# Cancel pending shutdown
sudo shutdown -c 2>/dev/null || true

# Restart FiLM server
pkill -f reinvent4_film_rest_server 2>/dev/null || true
sleep 2
nohup python $PROJ/experiments/reinvent4_film_rest_server.py --host 127.0.0.1 --port 8088 > $HOME/film_rest_server.log 2>&1 &
for i in $(seq 1 30); do curl -sf http://127.0.0.1:8088/health >/dev/null && break; sleep 2; done

# Helper: sample 100K from existing chkpt on Mol1
sample_only() {
  local tag="$1"
  local chkpt="$PROJ/results/paper_evaluation/mol1_rl/$tag/${tag}_stage1.chkpt"
  if [ ! -f "$chkpt" ]; then
    log "no chkpt for $tag — skip sampling"
    return 1
  fi
  local out_dir="$PROJ/data/mol1_anchored_tier4/${tag}_sample_mol1_100K"
  local existing=$(wc -l < "$out_dir/sampling.csv" 2>/dev/null || echo 0)
  if [ "$existing" -gt 1000 ]; then
    log "$tag already has $existing sample rows — skip"
    return 0
  fi
  mkdir -p "$out_dir"
  cat > "$out_dir/sampling.toml" <<EOF
run_type = "sampling"
device = "cuda"
[parameters]
model_file = "$chkpt"
smiles_file = "$ANCHOR"
sample_strategy = "multinomial"
temperature = 1.0
output_file = "$out_dir/sampling.csv"
num_smiles = 100000
unique_molecules = true
randomize_smiles = true
EOF
  cd "$out_dir"
  log "Sampling $tag 100K"
  timeout 3600 reinvent ./sampling.toml -d cuda 2>&1 | tee -a "$LOG"
  local n=$(wc -l < "$out_dir/sampling.csv" 2>/dev/null || echo 0)
  log "$tag sampled $n rows"
}

# Helper: run RL + sampling for one cohort (used for cohorts that had RL failure)
rl_and_sample() {
  local tag="$1"
  local toml_dir="$2"  # thiq_rl_tomls or murcko_rl_tomls
  local work="$PROJ/results/paper_evaluation/mol1_rl/$tag"
  if [ -f "$work/${tag}_stage1.chkpt" ]; then
    log "$tag chkpt exists — just sampling"
    sample_only "$tag"
    return
  fi
  log "RL re-run $tag"
  rm -rf "$work"; mkdir -p "$work"; cd "$work"
  cp "$PROJ/experiments/$toml_dir/$tag.toml" "./$tag.toml"
  # Start with batch_size=4 since A100 OOMs at 16
  sed -i 's|^batch_size = .*|batch_size = 4|' "./$tag.toml"
  timeout 5400 reinvent "./$tag.toml" -d cuda 2>&1 | tee -a "$LOG"
  if [ -f "$work/${tag}_stage1.chkpt" ]; then
    sample_only "$tag"
  else
    log "$tag RL failed"
  fi
}

# Retry tasks (sequential, single GPU)
sample_only "thiq_rl_zap70"
rl_and_sample "thiq_rl_exp2_zap70" "thiq_rl_tomls"
# Also the mol1only cohorts (replaces mol1only tmux)
rl_and_sample "thiq_rl_mol1only" "thiq_rl_tomls"
rl_and_sample "thiq_rl_exp2_mol1only" "thiq_rl_tomls"

pkill -f reinvent4_film_rest_server 2>/dev/null || true
touch "$HOME/a100_fixup.DONE"
log "=== Fixup DONE ==="
sudo shutdown -c 2>/dev/null || true
