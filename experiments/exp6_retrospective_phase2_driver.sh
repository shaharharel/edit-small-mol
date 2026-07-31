#!/bin/bash
# EXP6 Retrospective LO — Phase 2 driver (local orchestrator).
#
# Boots a100-b, syncs Phase 1 outputs, runs (target, strategy, iter)
# REINVENT4 RL + sampling + scoring across the matrix, then stops a100-b.
#
# Strategy A = "single anchor" (the official anchor only, seeds=anchor)
# Strategy B = "100-anchor pool" (seeds drawn from anchor_pool_strategy_b.csv)
#
# After iter 1, promoted-50 cohort (by ensemble pIC50 with Tc>=0.3 to original
# anchor) is added back as seeds for iter 2; ditto iter 3.

set -uo pipefail
PROJ="/Users/shaharharel/Documents/github/edit-small-mol"
LOG="$PROJ/results/paper_evaluation/exp6_phase2_driver.log"
mkdir -p "$(dirname "$LOG")"
VM="ai-gpu-a100-b"
ZONE="us-central1-b"
USER="shaharh_quris_ai"
KEY="$HOME/.ssh/google_compute_engine"
REMOTE_ROOT="/home/$USER/edit-small-mol"

log() { date -u +"[%FT%TZ] [phase2] $*" | tee -a "$LOG"; }

log "=== EXP6 Phase 2 driver START ==="

# --- 0) Boot a100-b ---
STATUS=$(gcloud compute instances describe $VM --zone=$ZONE --format='value(status)' 2>/dev/null)
log "Pre-boot VM status: $STATUS"
if [ "$STATUS" != "RUNNING" ]; then
  log "Starting $VM..."
  gcloud compute instances start $VM --zone=$ZONE --quiet 2>&1 | tee -a "$LOG"
fi
# Wait for SSH (up to 5 min)
IP=$(gcloud compute instances describe $VM --zone=$ZONE --format='value(networkInterfaces[0].accessConfigs[0].natIP)')
log "VM IP: $IP — waiting for SSH..."
for i in $(seq 1 60); do
  if ssh -i $KEY -o ConnectTimeout=5 -o StrictHostKeyChecking=no $USER@$IP "echo OK" >/dev/null 2>&1; then
    log "SSH up after ${i} attempts"
    break
  fi
  sleep 5
done

# Sanity-check SSH again (and abort if fail)
if ! ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $USER@$IP "echo OK" >/dev/null 2>&1; then
  log "FATAL: SSH never came up to $VM. Stopping VM and aborting."
  gcloud compute instances stop $VM --zone=$ZONE --quiet 2>&1 | tee -a "$LOG"
  exit 1
fi

# --- 1) Sync Phase 1 outputs + scorer + driver script to VM ---
log "Syncing files to VM..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$IP \
  "mkdir -p $REMOTE_ROOT/data/exp6_retrospective $REMOTE_ROOT/experiments $REMOTE_ROOT/src $REMOTE_ROOT/models" 2>&1 | tee -a "$LOG"

rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$PROJ/data/exp6_retrospective/" "$USER@$IP:$REMOTE_ROOT/data/exp6_retrospective/" 2>&1 | tail -3 | tee -a "$LOG"

rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$PROJ/experiments/exp6_retrospective_phase2_scorer.py" \
  "$PROJ/experiments/exp6_retrospective_phase2_runner.py" \
  "$USER@$IP:$REMOTE_ROOT/experiments/" 2>&1 | tee -a "$LOG"

rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$PROJ/src/" "$USER@$IP:$REMOTE_ROOT/src/" 2>&1 | tail -3 | tee -a "$LOG"

# Make sure prior model is on VM (~80 MB)
PRIOR_NAME="reinvent4_mol2mol_warhead_tokens_v2.prior"
PRIOR_REMOTE_OK=$(ssh -i $KEY -o StrictHostKeyChecking=no $USER@$IP "[ -s $REMOTE_ROOT/models/$PRIOR_NAME ] && echo Y || echo N")
if [ "$PRIOR_REMOTE_OK" = "N" ]; then
  log "Uploading prior model..."
  rsync -avz --progress -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
    "$PROJ/models/$PRIOR_NAME" "$USER@$IP:$REMOTE_ROOT/models/" 2>&1 | tail -5 | tee -a "$LOG"
else
  log "Prior already on VM."
fi

# --- 2) Kick off the per-target Python runner on the VM ---
log "Launching remote runner..."
ssh -i $KEY -o StrictHostKeyChecking=no $USER@$IP \
  "cd $REMOTE_ROOT && nohup /home/$USER/miniconda3/envs/quris/bin/python \
    experiments/exp6_retrospective_phase2_runner.py \
    > $REMOTE_ROOT/exp6_phase2_runner.log 2>&1 &
   sleep 1
   echo started PID=\$!" 2>&1 | tee -a "$LOG"

# --- 3) Poll: stream remote log + check for DONE flag every 60 s ---
log "Polling remote runner..."
while true; do
  TAIL=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $USER@$IP \
    "tail -n 3 $REMOTE_ROOT/exp6_phase2_runner.log 2>/dev/null" 2>/dev/null | head -3)
  if [ -n "$TAIL" ]; then
    echo "$TAIL" | sed "s|^|  REMOTE: |" | tee -a "$LOG"
  fi
  DONE=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $USER@$IP \
    "[ -f $REMOTE_ROOT/data/exp6_retrospective/_PHASE2_DONE ] && echo Y || echo N" 2>/dev/null)
  STILL_RUNNING=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no $USER@$IP \
    "pgrep -fc exp6_retrospective_phase2_runner.py 2>/dev/null || echo 0" 2>/dev/null)
  if [ "$DONE" = "Y" ]; then
    log "Remote DONE flag detected."
    break
  fi
  if [ "$STILL_RUNNING" = "0" ]; then
    log "WARN: runner not running and no DONE flag. Likely crashed; pulling log."
    break
  fi
  sleep 60
done

# --- 4) Pull artifacts ---
log "Pulling remote artifacts..."
rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$USER@$IP:$REMOTE_ROOT/data/exp6_retrospective/" "$PROJ/data/exp6_retrospective/" 2>&1 | tail -5 | tee -a "$LOG"
rsync -avz --quiet -e "ssh -i $KEY -o StrictHostKeyChecking=no" \
  "$USER@$IP:$REMOTE_ROOT/exp6_phase2_runner.log" "$PROJ/results/paper_evaluation/exp6_phase2_runner.log" 2>&1 | tee -a "$LOG"

# --- 5) Stop a100-b unconditionally ---
log "Stopping $VM..."
gcloud compute instances stop $VM --zone=$ZONE --quiet 2>&1 | tee -a "$LOG"

log "=== EXP6 Phase 2 driver DONE ==="
