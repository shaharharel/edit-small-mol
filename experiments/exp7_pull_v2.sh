#!/bin/bash
# EXP7 simple pull: gcloud scp + Phase 3 + QA loop, no rsync hangs.
INTERVAL=${1:-1800}
PROJ=/Users/shaharharel/Documents/github/edit-small-mol
LOCAL_RL=$PROJ/data/exp7_lo_benchmark/_rl
LOCAL_PULL=$PROJ/data/exp7_lo_benchmark/_rl_FROM_VM
STOP_FILE=$PROJ/data/exp7_lo_benchmark/_stop
LOG=$PROJ/data/exp7_lo_benchmark/_pull_log.txt

mkdir -p $LOCAL_PULL

while true; do
    if [ -f $STOP_FILE ]; then
        echo "[$(date +%H:%M:%S)] Stop sentinel — exit" | tee -a $LOG
        exit 0
    fi

    ts=$(date +%H:%M:%S)
    VM=$(gcloud compute instances describe ai-gpu-a100-b --zone=us-central1-b --format='value(status)' 2>&1 | head -1)
    if [ "$VM" != "RUNNING" ]; then
        echo "[$ts] VM=$VM — skip pull" | tee -a $LOG
        sleep $INTERVAL
        continue
    fi

    # List cell dirs with sampled.csv on VM
    echo "[$ts] Listing remote cohorts..." | tee -a $LOG
    REMOTE_LIST=$(gcloud compute ssh ai-gpu-a100-b --zone=us-central1-b --command='find ~/edit-small-mol/data/exp7_lo_benchmark/_rl -name sampled.csv 2>/dev/null' 2>/dev/null)
    REMOTE_COUNT=$(echo "$REMOTE_LIST" | grep -c sampled.csv)
    echo "[$ts] Remote cohorts: $REMOTE_COUNT" | tee -a $LOG

    if [ $REMOTE_COUNT -gt 0 ]; then
        # Pull only new cohorts
        for REMOTE_PATH in $REMOTE_LIST; do
            CELL=$(basename $(dirname $REMOTE_PATH))
            LOCAL_PATH=$LOCAL_RL/$CELL/sampled.csv
            mkdir -p $LOCAL_RL/$CELL
            if [ ! -f $LOCAL_PATH ] || [ "$REMOTE_PATH" -nt "$LOCAL_PATH" ]; then
                # not present locally or remote is newer — pull
                gcloud compute scp ai-gpu-a100-b:$REMOTE_PATH $LOCAL_PATH --zone=us-central1-b 2>/dev/null || true
            fi
        done
        N_LOCAL=$(find $LOCAL_RL -name sampled.csv 2>/dev/null | wc -l | tr -d ' ')
        echo "[$ts] Local cohorts after pull: $N_LOCAL" | tee -a $LOG

        # Phase 3 scoring + QA
        /opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_phase3_score.py 2>&1 | tail -3 | tee -a $LOG || true
        /opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_qa_audit.py 2>&1 | tail -5 | tee -a $LOG || true
    fi

    sleep $INTERVAL
done
