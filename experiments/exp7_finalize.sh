#!/bin/bash
# EXP7 final wrap-up: pull everything, score, build reports, stop VM.
#
# Usage:
#   bash exp7_finalize.sh

set -e
PROJ=/Users/shaharharel/Documents/github/edit-small-mol
LOCAL_RL=$PROJ/data/exp7_lo_benchmark/_rl
LOCAL_RL_FROM_VM=$PROJ/data/exp7_lo_benchmark/_rl_FROM_VM
STOP_FILE=$PROJ/data/exp7_lo_benchmark/_stop

# Signal pull orchestrator to stop
touch $STOP_FILE

# Final pull
echo "=== Final pull from VM ==="
mkdir -p $LOCAL_RL_FROM_VM
gcloud compute scp --recurse \
    ai-gpu-a100-b:/home/shaharh_quris_ai/edit-small-mol/data/exp7_lo_benchmark/_rl \
    $LOCAL_RL_FROM_VM/ --zone=us-central1-b 2>&1 | tail -3

# Merge
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
echo "Cohorts pulled: $N_LOCAL"

# Stop VM
echo "=== Stopping VM ==="
gcloud compute instances stop ai-gpu-a100-b --zone=us-central1-b --quiet 2>&1 | tail -3

# Final scoring
echo "=== Phase 3 scoring ==="
/opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_phase3_score.py 2>&1 | tail -20

# Final eval table + reports
echo "=== Eval table + final reports ==="
/opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_eval_table.py 2>&1 | tail -15

# Final QA
echo "=== Final QA ==="
/opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7_qa_audit.py 2>&1 | tail -10

# Cleanup orchestrator
pkill -f exp7_periodic_pull_score 2>/dev/null
rm -f $STOP_FILE

# Verify VM stopped
sleep 5
VM=$(gcloud compute instances describe ai-gpu-a100-b --zone=us-central1-b --format='value(status)' 2>&1 | head -1)
echo "VM final status: $VM"

echo "=== DONE ==="
