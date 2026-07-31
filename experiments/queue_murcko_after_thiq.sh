#!/bin/bash
# Wait for the overnight_thiq_orchestrator's DONE flag, then run Murcko RL +
# sampling for 1-2 Murcko cohorts on this VM. Auto-shutdown after.
#
# Per-VM Murcko cohort assignment:
#   ai-gpu      -> murcko_rl_zap70, murcko_rl_exp2_zap70   (2 cohorts, V100)
#   ai-gpu-a100 -> murcko_rl_kinase, murcko_rl_exp2_kinase (2 cohorts, A100)
#   (ai-gpu2 not used for Murcko — mol1only dropped from Murcko set)
set -o pipefail
VM_LBL="${1:-unknown}"
PROJ="$HOME/edit-small-mol"
LOG="$HOME/murcko_${VM_LBL}.log"
DONE_FILE_PREV="$HOME/overnight_${VM_LBL}.DONE"
DONE_FILE_MURCKO="$HOME/murcko_${VM_LBL}.DONE"
ANCHOR="$PROJ/data/mol1_only_anchor.smi"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

case "$VM_LBL" in
  ai-gpu)       MURCKO_COHORTS="murcko_rl_zap70 murcko_rl_exp2_zap70" ;;
  ai-gpu-a100)  MURCKO_COHORTS="murcko_rl_kinase murcko_rl_exp2_kinase" ;;
  ai-gpu2)      log() { date -u +"[%FT%TZ] [${VM_LBL}] mol1only doesn't get Murcko cohorts — exiting" | tee -a "$LOG"; }; touch "$DONE_FILE_MURCKO"; exit 0 ;;
  *) echo "Unknown VM: $VM_LBL"; exit 1 ;;
esac

log() { date -u +"[%FT%TZ] [${VM_LBL}] $*" | tee -a "$LOG"; }

# Cancel pending shutdown from previous orchestrator
sudo shutdown -c 2>/dev/null || true

log "=== Murcko queue start — cohorts: $MURCKO_COHORTS ==="
log "Waiting for $DONE_FILE_PREV"
for i in $(seq 1 600); do
  [ -f "$DONE_FILE_PREV" ] && break
  sleep 30
done

sudo shutdown -c 2>/dev/null || true

# Start FiLM REST server
pkill -f reinvent4_film_rest_server 2>/dev/null || true
sleep 2
nohup python $PROJ/experiments/reinvent4_film_rest_server.py \
    --host 127.0.0.1 --port 8088 > $HOME/film_rest_server.log 2>&1 &
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1 && break
  sleep 2
done
log "FiLM server ready"

# Per-cohort: RL → QA → sampling → QA
for TAG in $MURCKO_COHORTS; do
  log "=== START Murcko cohort: $TAG ==="
  work="$PROJ/results/paper_evaluation/mol1_rl/$TAG"
  rm -rf "$work"; mkdir -p "$work"
  cd "$work"
  cp $PROJ/experiments/murcko_rl_tomls/$TAG.toml ./$TAG.toml
  timeout 5400 reinvent ./$TAG.toml -d cuda 2>&1 | tee -a "$LOG"
  rv=${PIPESTATUS[0]}
  if [ "$rv" != "0" ]; then
    log "RL failed (rv=$rv) — retrying batch_size=4"
    sed -i 's|^batch_size = .*|batch_size = 4|' ./$TAG.toml
    timeout 5400 reinvent ./$TAG.toml -d cuda 2>&1 | tee -a "$LOG"
    rv=${PIPESTATUS[0]}
  fi
  if [ "$rv" != "0" ]; then
    log "RL completely failed — skipping sampling for $TAG"
    continue
  fi
  python $PROJ/experiments/qa_thiq_cohort.py "$TAG" --mode rl 2>&1 | tee -a "$LOG"

  out_dir="$PROJ/data/mol1_anchored_tier4/${TAG}_sample_mol1_100K"
  mkdir -p "$out_dir"
  cat > "$out_dir/sampling.toml" <<EOF
run_type = "sampling"
device = "cuda"
json_out_config = "$out_dir/_sampling.json"

[parameters]
model_file = "$work/${TAG}_stage1.chkpt"
smiles_file = "$ANCHOR"
sample_strategy = "multinomial"
temperature = 1.0
output_file = "$out_dir/sampling.csv"
num_smiles = 100000
unique_molecules = true
randomize_smiles = true
EOF
  cd "$out_dir"
  log "Sampling 100K on Mol1 from $TAG"
  timeout 5400 reinvent ./sampling.toml -d cuda 2>&1 | tee -a "$LOG"
  python $PROJ/experiments/qa_thiq_cohort.py "$TAG" --mode sample 2>&1 | tee -a "$LOG"
  log "=== DONE Murcko cohort: $TAG ==="
done

pkill -f reinvent4_film_rest_server 2>/dev/null || true
touch "$DONE_FILE_MURCKO"
log "=== Murcko ALL DONE — wrote $DONE_FILE_MURCKO ==="
log "Scheduling shutdown in 10 min..."
sudo shutdown -h +10 2>&1 | tee -a "$LOG" || log "shutdown failed"
