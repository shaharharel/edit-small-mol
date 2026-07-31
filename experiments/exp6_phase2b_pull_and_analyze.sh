#!/bin/bash
# Pull cohort CSVs + rl_results.json from a100-b, then run local Phase 3.
set -uo pipefail
PROJ="/Users/shaharharel/Documents/github/edit-small-mol"
VM="ai-gpu-a100-b"
ZONE="us-central1-b"
USER="shaharh_quris_ai"
KEY="$HOME/.ssh/google_compute_engine"
REMOTE_ROOT="/home/$USER/edit-small-mol"

LOG="$PROJ/results/paper_evaluation/exp6_phase2b_pull.log"
mkdir -p "$(dirname "$LOG")"
echo "[$(date -u +%FT%TZ)] === pull-and-analyze START ===" | tee -a "$LOG"

# pull only the cohort CSVs + rl_results.json + per-target summary (don't pull the
# 80MB agent.prior files)
rsync -avz --quiet --include="*/" \
  --include="iter*_film_cohort.csv" \
  --include="iter*_dabs_cohort.csv" \
  --include="rl_results.json" \
  --include="phase3_analysis.md" \
  --include="phase0_*.json" \
  --include="_PHASE2B_DONE" \
  --exclude="*" \
  -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$USER@$(gcloud compute instances describe $VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):$REMOTE_ROOT/data/exp6_retrospective/" \
  "$PROJ/data/exp6_retrospective/" 2>&1 | tail -10 | tee -a "$LOG"

# pull the orch log too
rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$USER@$(gcloud compute instances describe $VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)'):/home/$USER/exp6_phase2b_orch.log" \
  "$PROJ/results/paper_evaluation/exp6_phase2b_orch.log" 2>&1 | tee -a "$LOG"

echo "[$(date -u +%FT%TZ)] running Phase 3 analyzer..." | tee -a "$LOG"
cd "$PROJ"
~/miniconda3/envs/quris/bin/python experiments/exp6_retrospective_phase3_analyze.py 2>&1 | tail -50 | tee -a "$LOG"

echo "[$(date -u +%FT%TZ)] === pull-and-analyze DONE ===" | tee -a "$LOG"
