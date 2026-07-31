#!/bin/bash
# Safe local monitor: ONLY pulls chkpts + sampling.csvs from the A100.
# Does NOT shut down VMs — user controls that.
#
# Run on Mac: nohup bash experiments/safe_local_monitor.sh > safe_monitor.log 2>&1 &
set -o pipefail
PROJ="/Users/shaharharel/Documents/github/edit-small-mol"
LOG="$PROJ/safe_monitor.log"
KEY="$HOME/.ssh/google_compute_engine"
A100_IP="34.10.172.236"

ALL_TAGS=(
  thiq_rl_zap70 thiq_rl_exp2_zap70 thiq_rl_kinase thiq_rl_exp2_kinase
  murcko_rl_zap70 murcko_rl_exp2_zap70 murcko_rl_kinase murcko_rl_exp2_kinase
  thiq_rl_mol1only thiq_rl_exp2_mol1only
)

log() { date +"[%FT%T] $*" | tee -a "$LOG"; }
log "=== Safe local monitor START (pulls only, no shutdown) ==="
mkdir -p "$PROJ/models/rl_checkpoints_b" "$PROJ/data/overnight_pull"

while true; do
  for tag in "${ALL_TAGS[@]}"; do
    local_chkpt="$PROJ/models/rl_checkpoints_b/${tag}_stage1.chkpt"
    remote_chkpt="~/edit-small-mol/results/paper_evaluation/mol1_rl/${tag}/${tag}_stage1.chkpt"
    if [ ! -f "$local_chkpt" ] || [ ! -s "$local_chkpt" ]; then
      size=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no shaharh_quris_ai@$A100_IP "stat -c %s $remote_chkpt 2>/dev/null" 2>/dev/null || echo 0)
      if [ "$size" -gt 50000000 ]; then
        log "pulling chkpt $tag ($size bytes)"
        scp -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=20 \
          shaharh_quris_ai@$A100_IP:$remote_chkpt "$local_chkpt" >/dev/null 2>&1 \
          && log "  ✓ pulled $tag chkpt"
      fi
    fi
    # Sampling csv
    local_csv="$PROJ/data/overnight_pull/${tag}_sample_mol1_100K.csv"
    remote_csv="~/edit-small-mol/data/mol1_anchored_tier4/${tag}_sample_mol1_100K/sampling.csv"
    remote_lines=$(ssh -i $KEY -o ConnectTimeout=10 -o StrictHostKeyChecking=no shaharh_quris_ai@$A100_IP "wc -l < $remote_csv 2>/dev/null" 2>/dev/null || echo 0)
    local_lines=$(wc -l < "$local_csv" 2>/dev/null || echo 0)
    if [ "$remote_lines" -gt 100 ] && [ "$remote_lines" -gt "$local_lines" ]; then
      log "pulling sample $tag (remote=$remote_lines, local=$local_lines)"
      scp -i $KEY -o StrictHostKeyChecking=no -o ConnectTimeout=30 \
        shaharh_quris_ai@$A100_IP:$remote_csv "$local_csv" >/dev/null 2>&1 \
        && log "  ✓ pulled $tag sample"
    fi
  done
  sleep 600   # 10 min
done
