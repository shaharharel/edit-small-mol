#!/bin/bash
# Pull v2 _rl + driver result CSVs from a100-b. Run periodically.
set -euo pipefail
DEST=/Users/shaharharel/Documents/github/edit-small-mol/data/exp7_v2_benchmark/_rl
mkdir -p "$DEST"
gcloud compute scp --recurse --tunnel-through-iap --zone=us-central1-b \
    ai-gpu-a100-b:/home/shaharh_quris_ai/edit-small-mol/data/exp7_v2_benchmark/_rl \
    /Users/shaharharel/Documents/github/edit-small-mol/data/exp7_v2_benchmark/ 2>&1 | tail -5
LOCAL_DIR=/Users/shaharharel/Documents/github/edit-small-mol/data/exp7_v2_benchmark/_rl
N_CELLS=$(ls "$LOCAL_DIR" 2>/dev/null | wc -l)
N_SAMPLED=$(find "$LOCAL_DIR" -name 'sampled.csv' -size +500c | wc -l)
echo "[pull] dirs=$N_CELLS sampled.csv>500B=$N_SAMPLED"
