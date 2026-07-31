#!/bin/bash
# EXP6 Phase 2 finisher — pulls remote artifacts then STOPS a100-b.
#
# Run AFTER the RL driver is confirmed done (rl_results.json contains all
# expected cells, or time budget exhausted). Always stops the VM at the end.

set -uo pipefail
VM="ai-gpu-a100-b"
ZONE="us-central1-b"
USER="shaharh_quris_ai"
PROJ_LOCAL="/Users/shaharharel/Documents/github/edit-small-mol"
PROJ_REMOTE="/home/$USER/edit-small-mol"
KEY="$HOME/.ssh/google_compute_engine"

log() { date -u +"[%FT%TZ] $*"; }

log "=== exp6 pull + stop ==="

# Get latest IP
IP=$(gcloud compute instances describe $VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)' 2>/dev/null)
log "VM IP: $IP"

# Pull rl_results.json + cohort CSVs + run logs
log "Pulling rl_results.json + cohorts + logs"
rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$USER@$IP:$PROJ_REMOTE/data/exp6_retrospective/" \
  "$PROJ_LOCAL/data/exp6_retrospective/" 2>&1 | tail -10 || log "rsync data failed"

rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$USER@$IP:~/exp6_rl_film.log" \
  "$USER@$IP:~/exp6_rl_dabs.log" \
  "$PROJ_LOCAL/results/paper_evaluation/" 2>&1 | tail -5 || log "rsync logs failed"

# Build summary report
log "Building summary"
cd "$PROJ_LOCAL"
source /opt/miniconda3/etc/profile.d/conda.sh && conda activate quris
python experiments/exp6_rl_summary.py 2>&1 | tail -10

# STOP a100-b unconditionally
log "Stopping $VM"
gcloud compute instances stop $VM --zone=$ZONE --quiet 2>&1 | tail -5

# Verify
status=$(gcloud compute instances describe $VM --zone=$ZONE --format='value(status)')
log "Final VM status: $status"
if [ "$status" != "TERMINATED" ] && [ "$status" != "STOPPING" ]; then
  log "WARN: VM not TERMINATED, retrying stop"
  gcloud compute instances stop $VM --zone=$ZONE --quiet
fi
log "=== DONE ==="
