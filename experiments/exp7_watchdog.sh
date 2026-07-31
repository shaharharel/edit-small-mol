#!/bin/bash
# EXP7 watchdog: waits for VM driver to finish (either by budget hit, error, or
# completion), then runs the finalize step automatically.
#
# Polls every 5 min. Exits + finalizes when:
#   - driver process is not in `ps` on VM
#   - OR driver.log contains "BUDGET HIT"
#   - OR total elapsed > 8h
#   - OR sentinel file _stop appears
#
# Sentinel paths:
#   data/exp7_lo_benchmark/_stop        — manual stop
#   data/exp7_lo_benchmark/_finalized   — sentinel set after finalize runs (prevents re-finalize)

PROJ=/Users/shaharharel/Documents/github/edit-small-mol
STOP_FILE=$PROJ/data/exp7_lo_benchmark/_stop
FINAL_FILE=$PROJ/data/exp7_lo_benchmark/_finalized
LOG=$PROJ/data/exp7_lo_benchmark/_watchdog_log.txt

START_EPOCH=$(date +%s)
MAX_RUNTIME_SEC=28800  # 8h hard cap

echo "[$(date +%H:%M:%S)] Watchdog started, max 8h" | tee -a $LOG

while true; do
    NOW=$(date +%s)
    ELAPSED=$(( NOW - START_EPOCH ))
    ts=$(date +%H:%M:%S)

    if [ -f $FINAL_FILE ]; then
        echo "[$ts] Already finalized — exit" | tee -a $LOG
        exit 0
    fi

    if [ -f $STOP_FILE ]; then
        echo "[$ts] Stop sentinel detected — finalizing" | tee -a $LOG
        bash $PROJ/experiments/exp7_finalize.sh 2>&1 | tee -a $LOG
        touch $FINAL_FILE
        exit 0
    fi

    if [ $ELAPSED -gt $MAX_RUNTIME_SEC ]; then
        echo "[$ts] 8h cap hit — force finalize" | tee -a $LOG
        bash $PROJ/experiments/exp7_finalize.sh 2>&1 | tee -a $LOG
        touch $FINAL_FILE
        exit 0
    fi

    # Check VM/driver state
    VM=$(gcloud compute instances describe ai-gpu-a100-b --zone=us-central1-b --format='value(status)' 2>&1 | head -1)
    if [ "$VM" != "RUNNING" ]; then
        echo "[$ts] VM=$VM (already stopped) — finalize locally only" | tee -a $LOG
        bash $PROJ/experiments/exp7_finalize.sh 2>&1 | tee -a $LOG
        touch $FINAL_FILE
        exit 0
    fi

    # Check driver process on VM
    DRIVER_RUNNING=$(gcloud compute ssh ai-gpu-a100-b --zone=us-central1-b --command='pgrep -fc exp7_rl_driver' 2>/dev/null | tr -d '[:space:]')
    BUDGET_HIT=$(gcloud compute ssh ai-gpu-a100-b --zone=us-central1-b --command='grep -c "BUDGET HIT" ~/edit-small-mol/data/exp7_lo_benchmark/_logs/driver.log 2>/dev/null' 2>/dev/null | tr -d '[:space:]')
    [ -z "$DRIVER_RUNNING" ] && DRIVER_RUNNING=0
    [ -z "$BUDGET_HIT" ] && BUDGET_HIT=0

    if [ "$DRIVER_RUNNING" = "0" ] || [ "$BUDGET_HIT" != "0" ]; then
        echo "[$ts] Driver finished (running=$DRIVER_RUNNING budget_hit=$BUDGET_HIT) — finalizing" | tee -a $LOG
        bash $PROJ/experiments/exp7_finalize.sh 2>&1 | tee -a $LOG
        touch $FINAL_FILE
        exit 0
    fi

    echo "[$ts] watchdog: VM=$VM driver_running=$DRIVER_RUNNING elapsed=${ELAPSED}s" | tee -a $LOG
    sleep 600  # poll every 10 min
done
