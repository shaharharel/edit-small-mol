#!/bin/bash
# A100 multi-cohort overnight orchestrator. Replaces failed V100 work.
# Sequence: thiq_zap70 → thiq_exp2_zap70 → thiq_kinase(reduced) → thiq_exp2_kinase(reduced)
#          → murcko_zap70 → murcko_exp2_zap70 → murcko_kinase(reduced) → murcko_exp2_kinase(reduced)
# After each: QA chkpt → sample 100K on Mol1 → QA sample. At end: VM shutdown.
set -o pipefail
PROJ="$HOME/edit-small-mol"
LOG="$HOME/a100_overnight.log"
DONE_FILE="$HOME/a100_overnight.DONE"
ANCHOR="$PROJ/data/mol1_only_anchor.smi"
KINASE_REDUCED_SEED="$PROJ/data/mol1_rl_seeds/seed_zap70_mol1heavy.smi"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# Cohorts in priority order. Format: tag:toml_file
COHORTS=(
  "thiq_rl_zap70:thiq_rl_zap70.toml"
  "thiq_rl_exp2_zap70:thiq_rl_exp2_zap70.toml"
  "thiq_rl_kinase:thiq_rl_kinase.toml"
  "thiq_rl_exp2_kinase:thiq_rl_exp2_kinase.toml"
  "murcko_rl_zap70:murcko_rl_zap70.toml"
  "murcko_rl_exp2_zap70:murcko_rl_exp2_zap70.toml"
  "murcko_rl_kinase:murcko_rl_kinase.toml"
  "murcko_rl_exp2_kinase:murcko_rl_exp2_kinase.toml"
)

log() { date -u +"[%FT%TZ] [a100] $*" | tee -a "$LOG"; }

# Cancel any stale shutdown
sudo shutdown -c 2>/dev/null || true
log "=== A100 overnight START — ${#COHORTS[@]} cohorts ==="

# Kill any stalled reinvent
pkill -f "reinvent.*toml" 2>/dev/null || true
sleep 3
# Restart FiLM server
pkill -f reinvent4_film_rest_server 2>/dev/null || true
sleep 2
nohup python $PROJ/experiments/reinvent4_film_rest_server.py \
    --host 127.0.0.1 --port 8088 > $HOME/film_rest_server.log 2>&1 &
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1 && break
  sleep 2
done
log "FiLM server ready"

for entry in "${COHORTS[@]}"; do
  TAG="${entry%%:*}"
  TOML_FILE="${entry##*:}"

  # For *kinase cohorts, swap to reduced seed (240 lines) to avoid stalling
  TOML_DIR=$(echo "$TOML_FILE" | grep -q thiq && echo "thiq_rl_tomls" || echo "murcko_rl_tomls")
  SRC_TOML="$PROJ/experiments/$TOML_DIR/$TOML_FILE"
  if [ ! -f "$SRC_TOML" ]; then
    log "SKIP $TAG — toml $SRC_TOML missing"
    continue
  fi

  work="$PROJ/results/paper_evaluation/mol1_rl/$TAG"
  rm -rf "$work"; mkdir -p "$work"
  cd "$work"
  cp "$SRC_TOML" "./$TAG.toml"

  # If this is a kinase cohort, swap seed file in-place
  if [[ "$TAG" == *kinase* ]]; then
    sed -i "s|seed_kinase_zap70x5_mol1x20_clean.smi|seed_zap70_mol1heavy.smi|" "./$TAG.toml"
    log "Using reduced seed pool for $TAG"
  fi

  log "=== START $TAG ==="
  timeout 5400 reinvent "./$TAG.toml" -d cuda 2>&1 | tee -a "$LOG"
  rv=${PIPESTATUS[0]}
  if [ "$rv" != "0" ]; then
    log "RL failed (rv=$rv) for $TAG — retrying batch_size=4"
    sed -i 's|^batch_size = .*|batch_size = 4|' "./$TAG.toml"
    timeout 5400 reinvent "./$TAG.toml" -d cuda 2>&1 | tee -a "$LOG"
    rv=${PIPESTATUS[0]}
  fi
  if [ "$rv" != "0" ]; then
    log "RL completely failed for $TAG — skipping sampling"
    continue
  fi
  python $PROJ/experiments/qa_thiq_cohort.py "$TAG" --mode rl 2>&1 | tee -a "$LOG"

  # Sample 100K on Mol1
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
  log "=== DONE $TAG ==="
done

pkill -f reinvent4_film_rest_server 2>/dev/null || true
touch "$DONE_FILE"
log "=== A100 overnight ALL DONE ==="
log "Scheduling shutdown in 10 min..."
sudo shutdown -h +10 2>&1 | tee -a "$LOG" || log "shutdown failed"
