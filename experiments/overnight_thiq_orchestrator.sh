#!/bin/bash
# Per-VM autonomous overnight orchestrator with QA + retry + auto-shutdown.
#
# Usage:  bash overnight_thiq_orchestrator.sh <vm_label>
#   vm_label = ai-gpu | ai-gpu2 | ai-gpu-a100
#
# Per-VM assignment:
#   ai-gpu2       -> thiq_rl_mol1only,  thiq_rl_exp2_mol1only
#   ai-gpu        -> thiq_rl_zap70,     thiq_rl_exp2_zap70
#   ai-gpu-a100   -> thiq_rl_kinase,    thiq_rl_exp2_kinase
#
# Flow per cohort:
#   1. RL training (timeout 5400s)
#   2. QA on chkpt
#   3. Mol1 sampling N=100K
#   4. QA on sample CSV
#   5. Mark DONE flag
# At end: write done marker, shutdown VM in 10 min.

set -o pipefail   # do NOT set -e — we want retry/continue logic per step
VM_LBL="${1:-unknown}"
PROJ="$HOME/edit-small-mol"
LOG="$HOME/overnight_${VM_LBL}.log"
DONE_FILE="$HOME/overnight_${VM_LBL}.DONE"
ANCHOR="$PROJ/data/mol1_only_anchor.smi"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

case "$VM_LBL" in
  ai-gpu2)      COHORTS="thiq_rl_mol1only thiq_rl_exp2_mol1only" ;;
  ai-gpu)       COHORTS="thiq_rl_zap70 thiq_rl_exp2_zap70" ;;
  ai-gpu-a100)  COHORTS="thiq_rl_kinase thiq_rl_exp2_kinase" ;;
  *) echo "Unknown VM: $VM_LBL"; exit 1 ;;
esac

log() { date -u +"[%FT%TZ] [${VM_LBL}] $*" | tee -a "$LOG"; }
log "=== ORCHESTRATOR START — cohorts: $COHORTS ==="

# ------------------------------------------------------------------
# Start FiLM REST server (will outlive RL runs)
start_film_server() {
  pkill -f reinvent4_film_rest_server 2>/dev/null || true
  sleep 2
  nohup python $PROJ/experiments/reinvent4_film_rest_server.py \
      --host 127.0.0.1 --port 8088 > $HOME/film_rest_server.log 2>&1 &
  for i in $(seq 1 30); do
    if curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1; then
      log "FiLM server READY"
      return 0
    fi
    sleep 2
  done
  log "FiLM server failed to start"
  return 1
}

# ------------------------------------------------------------------
# Run RL for one cohort, retry once with smaller batch if OOM
run_rl() {
  local tag="$1"
  local work="$PROJ/results/paper_evaluation/mol1_rl/$tag"
  mkdir -p "$work"
  cd "$work"
  cp "$PROJ/experiments/thiq_rl_tomls/$tag.toml" "./$tag.toml"

  log "RL launching: $tag"
  timeout 5400 reinvent ./"$tag".toml -d cuda 2>&1 | tee -a "$LOG"
  local rv=${PIPESTATUS[0]}
  if [ "$rv" = "0" ]; then
    log "RL DONE: $tag"
    return 0
  fi
  log "RL failed/timeout (rv=$rv) for $tag — retrying with batch_size=4"
  # Reduce batch_size and retry
  sed -i 's|^batch_size = .*|batch_size = 4|' ./"$tag".toml
  timeout 5400 reinvent ./"$tag".toml -d cuda 2>&1 | tee -a "$LOG"
  rv=${PIPESTATUS[0]}
  if [ "$rv" = "0" ]; then
    log "RL DONE on retry: $tag"
    return 0
  fi
  log "RL FAILED on retry: $tag"
  return 1
}

# ------------------------------------------------------------------
# Sample N=100000 on Mol1 from a chkpt
sample_chkpt() {
  local tag="$1"
  local chkpt="$PROJ/results/paper_evaluation/mol1_rl/${tag}/${tag}_stage1.chkpt"
  local out_dir="$PROJ/data/mol1_anchored_tier4/${tag}_sample_mol1_100K"
  mkdir -p "$out_dir"
  cat > "$out_dir/sampling.toml" <<EOF
run_type = "sampling"
device = "cuda"
json_out_config = "$out_dir/_sampling.json"

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
  log "SAMPLING 100K on Mol1 from $tag"
  cd "$out_dir"
  timeout 5400 reinvent ./sampling.toml -d cuda 2>&1 | tee -a "$LOG"
  local rv=${PIPESTATUS[0]}
  local n=$(wc -l < "$out_dir/sampling.csv" 2>/dev/null || echo 0)
  if [ "$n" -gt 100 ]; then
    log "SAMPLING DONE: $tag — $n unique rows"
    return 0
  fi
  log "SAMPLING FAILED: $tag — only $n rows"
  return 1
}

# ------------------------------------------------------------------
# QA helper
qa_step() {
  local tag="$1"; local mode="$2"
  log "QA $mode on $tag"
  python $PROJ/experiments/qa_thiq_cohort.py "$tag" --mode "$mode" 2>&1 | tee -a "$LOG"
  return ${PIPESTATUS[0]}
}

# ------------------------------------------------------------------
# Start FiLM server
start_film_server || log "FiLM server failed — RL may still run with broken REST"

# Process each cohort sequentially
for tag in $COHORTS; do
  log "=== START $tag ==="
  if run_rl "$tag"; then
    qa_step "$tag" rl || log "QA[rl] failed for $tag (non-fatal)"
    if sample_chkpt "$tag"; then
      qa_step "$tag" sample || log "QA[sample] failed for $tag (non-fatal)"
    else
      log "Sampling failed for $tag — continuing"
    fi
  else
    log "RL completely failed for $tag — skipping sampling"
  fi
  log "=== DONE $tag ==="
done

# Cleanup FiLM server
pkill -f reinvent4_film_rest_server 2>/dev/null || true

# Write done marker (master polls this)
touch "$DONE_FILE"
log "=== ORCHESTRATOR DONE — wrote $DONE_FILE ==="

# Auto-shutdown in 10 min (gives master station time to pull artifacts)
log "Scheduling shutdown in 10 min..."
sudo shutdown -h +10 2>&1 | tee -a "$LOG" || log "shutdown -h failed (no sudo?). Manual stop needed."
