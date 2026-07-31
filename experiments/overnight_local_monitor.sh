#!/bin/bash
# Local poll monitor: every 15 min, pulls new chkpts + sampling.csvs from all 3
# VMs to local. Stops VM when its DONE flag appears.
#
# Run on Mac: bash experiments/overnight_local_monitor.sh
set -o pipefail
PROJ="/Users/shaharharel/Documents/github/edit-small-mol"
LOG="$PROJ/overnight_local_monitor.log"

VMS=(
  "ai-gpu       us-central1-c 136.119.139.94"
  "ai-gpu2      us-central1-a 34.59.164.180"
  "ai-gpu-a100  us-central1-b 34.10.172.236"
)
COHORTS=(
  "thiq_rl_zap70 thiq_rl_exp2_zap70"
  "thiq_rl_mol1only thiq_rl_exp2_mol1only"
  "thiq_rl_kinase thiq_rl_exp2_kinase"
)

log() { date -u +"[%FT%TZ] $*" | tee -a "$LOG"; }
KEY="$HOME/.ssh/google_compute_engine"
log "=== Local monitor START ==="

mkdir -p "$PROJ/models/rl_checkpoints_b" "$PROJ/data/overnight_pull"

while true; do
  for i in 0 1 2; do
    read VM ZONE IP <<<"${VMS[$i]}"
    COHORT_PAIR="${COHORTS[$i]}"
    for tag in $COHORT_PAIR; do
      # Pull chkpt if not already local + nonzero
      LOCAL_CHKPT="$PROJ/models/rl_checkpoints_b/${tag}_stage1.chkpt"
      if [ ! -f "$LOCAL_CHKPT" ] || [ ! -s "$LOCAL_CHKPT" ]; then
        REMOTE_CHKPT="~/edit-small-mol/results/paper_evaluation/mol1_rl/${tag}/${tag}_stage1.chkpt"
        SIZE=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no shaharh_quris_ai@$IP "stat -c %s $REMOTE_CHKPT 2>/dev/null" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 50000000 ]; then
          log "Pulling chkpt $tag from $VM ($SIZE bytes)"
          scp -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
            shaharh_quris_ai@$IP:$REMOTE_CHKPT "$LOCAL_CHKPT" >/dev/null 2>&1 \
            && log "  ✓ pulled $tag" || log "  ✗ failed to pull $tag"
        fi
      fi
      # Pull sampling.csv if newer
      LOCAL_CSV="$PROJ/data/overnight_pull/${tag}_sample_mol1_100K.csv"
      REMOTE_CSV="~/edit-small-mol/data/mol1_anchored_tier4/${tag}_sample_mol1_100K/sampling.csv"
      REMOTE_LINES=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no shaharh_quris_ai@$IP "wc -l < $REMOTE_CSV 2>/dev/null" 2>/dev/null || echo 0)
      LOCAL_LINES=$(wc -l < "$LOCAL_CSV" 2>/dev/null || echo 0)
      if [ "$REMOTE_LINES" -gt 100 ] && [ "$REMOTE_LINES" -gt "$LOCAL_LINES" ]; then
        log "Pulling sampling $tag from $VM (remote=$REMOTE_LINES, local=$LOCAL_LINES)"
        scp -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
          shaharh_quris_ai@$IP:$REMOTE_CSV "$LOCAL_CSV" >/dev/null 2>&1 \
          && log "  ✓ pulled $tag sampling" || log "  ✗ pull failed"
      fi
    done
    # Check DONE flag — shut down VM if done
    DONE=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no shaharh_quris_ai@$IP "ls ~/overnight_${VM}.DONE 2>/dev/null" 2>/dev/null || echo "")
    if [ -n "$DONE" ]; then
      log "$VM is DONE. Final chkpt pull + shutdown VM..."
      # One final sync
      for tag in $COHORT_PAIR; do
        REMOTE_CHKPT="~/edit-small-mol/results/paper_evaluation/mol1_rl/${tag}/${tag}_stage1.chkpt"
        scp -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
          shaharh_quris_ai@$IP:$REMOTE_CHKPT "$PROJ/models/rl_checkpoints_b/${tag}_stage1.chkpt" >/dev/null 2>&1
        REMOTE_CSV="~/edit-small-mol/data/mol1_anchored_tier4/${tag}_sample_mol1_100K/sampling.csv"
        scp -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
          shaharh_quris_ai@$IP:$REMOTE_CSV "$PROJ/data/overnight_pull/${tag}_sample_mol1_100K.csv" >/dev/null 2>&1
      done
      # Stop VM via gcloud
      gcloud compute instances stop $VM --zone=$ZONE --quiet 2>&1 | tee -a "$LOG"
      VMS[$i]=""   # disable further polling
    fi
  done
  # All 3 done?
  ACTIVE_COUNT=$(for v in "${VMS[@]}"; do [ -n "$v" ] && echo x; done | wc -l)
  if [ "$ACTIVE_COUNT" = "0" ]; then
    log "All VMs DONE. Monitor exiting."
    break
  fi
  sleep 900   # 15 min between polls
done
