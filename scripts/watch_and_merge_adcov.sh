#!/bin/bash
# Watch ai-chem2 for adcov DONE sentinels, SCP each CSV down, merge into local scored CSV.
# Idempotent: skips cohorts already merged.
set -e
PROJECT_ROOT=/Users/shaharharel/Documents/github/edit-small-mol
SCORED_DIR=$PROJECT_ROOT/data/tier4_scored
MERGE_PY=$PROJECT_ROOT/scripts/merge_adcov_local.py
COHORTS=(
  murcko_rl_exp2_zap70
  thiq_rl_mol1only
  murcko_rl_kinase
  murcko_rl_zap70
  mol1RL_v5_seed_mol1_only
  thiq_rl_exp2_zap70
  murcko_rl_exp2_kinase
  thiq_rl_exp2_kinase
  thiq_rl_zap70
  thiq_rl_kinase
)
# Local sentinel dir to avoid re-merging
MERGED_DIR=$PROJECT_ROOT/data/tier4_scored/.adcov_merged
mkdir -p "$MERGED_DIR"

source /opt/miniconda3/etc/profile.d/conda.sh
conda activate quris

for tag in "${COHORTS[@]}"; do
  echo "=== [$(date -Is)] tag=$tag ==="
  if [ -f "$MERGED_DIR/$tag.MERGED" ]; then
    echo "  already merged locally — skip"
    continue
  fi
  # Wait for DONE on remote
  while true; do
    rc=$(gcloud compute ssh ai-chem2 --zone=us-east1-b --command="test -f ~/adcov_runs/adcov_${tag}.DONE && echo DONE || echo PENDING" 2>/dev/null | tail -1)
    if [ "$rc" = "DONE" ]; then break; fi
    sleep 60
  done
  echo "  remote DONE detected — SCP + merge"
  ADCOV_LOCAL=$SCORED_DIR/adcov_${tag}.csv
  # SCP
  gcloud compute scp ai-chem2:~/adcov_runs/adcov_${tag}.csv "$ADCOV_LOCAL" --zone=us-east1-b 2>&1 | tail -2
  SCORED_LOCAL=$SCORED_DIR/${tag}_scored.csv
  if [ ! -f "$SCORED_LOCAL" ]; then
    echo "  WARN: no local scored CSV for $tag — skipping merge"
    touch "$MERGED_DIR/$tag.MERGED"
    continue
  fi
  python "$MERGE_PY" "$ADCOV_LOCAL" "$SCORED_LOCAL"
  touch "$MERGED_DIR/$tag.MERGED"
done

echo "=== [$(date -Is)] ALL COHORTS MERGED LOCALLY ==="
