#!/bin/bash
# EXP7 — periodically pull cohorts from VM, score, and write incremental reports.
# Designed to run continuously. Exits when sentinel file _stop appears.
#
# Usage:
#   bash exp7_periodic_pull_score.sh [interval_sec=1200]
#
INTERVAL=${1:-1200}
PROJ=/Users/shaharharel/Documents/github/edit-small-mol
LOCAL_RL=$PROJ/data/exp7_lo_benchmark/_rl
LOCAL_RL_FROM_VM=$PROJ/data/exp7_lo_benchmark/_rl_FROM_VM
STOP_FILE=$PROJ/data/exp7_lo_benchmark/_stop
LOG=$PROJ/data/exp7_lo_benchmark/_pull_log.txt

mkdir -p $LOCAL_RL_FROM_VM

while true; do
  if [ -f $STOP_FILE ]; then
    echo "[$(date +%H:%M:%S)] Stop sentinel found — exiting" | tee -a $LOG
    exit 0
  fi

  ts=$(date +%H:%M:%S)
  # Check VM status (skip pull if stopped)
  VM=$(gcloud compute instances describe ai-gpu-a100-b --zone=us-central1-b --format='value(status)' 2>&1 | head -1)
  if [ "$VM" != "RUNNING" ]; then
    echo "[$ts] VM=$VM — skipping pull" | tee -a $LOG
    sleep $INTERVAL
    continue
  fi

  # rsync pull (incremental)
  echo "[$ts] Pulling cohorts..." | tee -a $LOG
  rsync -auz --include='*/' --include='sampled.csv' --include='*.log' --exclude='*' \
      -e "gcloud compute ssh ai-gpu-a100-b --zone=us-central1-b --command" \
      ai-gpu-a100-b:/home/shaharh_quris_ai/edit-small-mol/data/exp7_lo_benchmark/_rl/ \
      $LOCAL_RL_FROM_VM/ 2>&1 | tail -3 | tee -a $LOG || true

  # Fallback: gcloud scp incremental (rsync over gcloud is fragile)
  gcloud compute scp --recurse \
      ai-gpu-a100-b:/home/shaharh_quris_ai/edit-small-mol/data/exp7_lo_benchmark/_rl \
      $LOCAL_RL_FROM_VM/ --zone=us-central1-b 2>&1 | tail -2 | tee -a $LOG || true

  # Merge pulled cohorts into canonical location
  if [ -d $LOCAL_RL_FROM_VM/_rl ]; then
    for d in $LOCAL_RL_FROM_VM/_rl/*/; do
      name=$(basename $d)
      mkdir -p $LOCAL_RL/$name
      if [ -f $d/sampled.csv ]; then
        cp -u $d/sampled.csv $LOCAL_RL/$name/sampled.csv 2>/dev/null || true
      fi
    done
  fi

  N_LOCAL=$(find $LOCAL_RL -name sampled.csv 2>/dev/null | wc -l)
  echo "[$ts] cohorts_local=$N_LOCAL" | tee -a $LOG

  # Incremental Phase 3 scoring (only if at least one new cohort)
  echo "[$ts] Running Phase 3 scoring..." | tee -a $LOG
  /opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_phase3_score.py 2>&1 | tail -3 | tee -a $LOG || true

  # Incremental QA audit
  echo "[$ts] Running QA audit..." | tee -a $LOG
  /opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_qa_audit.py 2>&1 | tail -10 | tee -a $LOG || true

  sleep $INTERVAL
done
