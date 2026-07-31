#!/usr/bin/env bash
# Orchestrator: run C2 → upload, C3 → upload, C4 → upload sequentially.
# Skips cohorts whose ckpt already exists AND filtered samples exist.
# Resilient: idempotent per-cohort so re-runs pick up where they left off.
set -uo pipefail

PROJECT_ROOT="$HOME/edit-small-mol"
CAMPAIGN_DIR="$PROJECT_ROOT/data/paper_pair_training/boltz_dpo_campaign"
CKPT_ROOT="$PROJECT_ROOT/models/dpo_checkpoints"
ERR_LOG="$CAMPAIGN_DIR/trainer_errors.log"

mkdir -p "$CAMPAIGN_DIR"
mkdir -p "$PROJECT_ROOT/logs"

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate quris

run_cohort() {
  local cohort_id="$1"
  local cohort_name="$2"
  local ckpt="$CKPT_ROOT/$cohort_name/dpo_latest.chkpt"
  local samples="$CAMPAIGN_DIR/samples_${cohort_name}_400.csv"
  local log="$PROJECT_ROOT/logs/${cohort_id}_run.log"

  if [ -f "$ckpt" ] && [ -f "$samples" ] && [ "$(wc -l < "$samples")" -gt 50 ]; then
    echo "[$(date -u +%H:%M:%S)] [$cohort_id] chkpt+samples exist → skip"
    return 0
  fi

  echo "[$(date -u +%H:%M:%S)] [$cohort_id] launching train+sample → $log"
  python -u "$PROJECT_ROOT/experiments/dpo_geometry/run_boltz_dpo.py" \
    --cohort "$cohort_id" \
    --action both \
    --n-samples 3000 \
    --sample-batch 32 \
    --log-every 25 \
    --checkpoint-every 100 \
    > "$log" 2>&1
  rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date -u +%H:%M:%S)] [$cohort_id] FAILED rc=$rc — check $log" | tee -a "$ERR_LOG"
    tail -50 "$log" | tee -a "$ERR_LOG"
    # attempt a re-run once for resilience
    echo "[$(date -u +%H:%M:%S)] [$cohort_id] RETRY once"
    python -u "$PROJECT_ROOT/experiments/dpo_geometry/run_boltz_dpo.py" \
      --cohort "$cohort_id" \
      --action both \
      --n-samples 3000 \
      --sample-batch 32 \
      --log-every 25 \
      --checkpoint-every 100 \
      >> "$log" 2>&1
    rc=$?
    if [ $rc -ne 0 ]; then
      echo "[$(date -u +%H:%M:%S)] [$cohort_id] RETRY FAILED rc=$rc" | tee -a "$ERR_LOG"
      return $rc
    fi
  fi

  # Upload
  echo "[$(date -u +%H:%M:%S)] [$cohort_id] uploading"
  bash "$PROJECT_ROOT/experiments/dpo_geometry/upload_cohort.sh" "$cohort_name" 2>&1 | tee -a "$log"
  return 0
}

echo "[$(date -u +%H:%M:%S)] [orchestrator] start"

for cohort_pair in "C2:boltz_geom_dpo_only" "C3:boltz_geom_dpo_plus_dap" "C4:boltz_geom_dpo_regularized"; do
  cid="${cohort_pair%%:*}"
  cname="${cohort_pair##*:}"
  run_cohort "$cid" "$cname" || {
    echo "[$(date -u +%H:%M:%S)] [orchestrator] cohort $cid failed; continuing to next"
  }
done

# Emit DONE flag
DONE="$PROJECT_ROOT/data/agent_coord/from_dpo_trainer_DONE.txt"
mkdir -p "$(dirname "$DONE")"
cat > "$DONE" <<EOF
DPO campaign done: $(date -u +%Y-%m-%dT%H:%M:%SZ)
Cohorts (chkpt|samples_400):
$(for c in boltz_geom_dpo_only boltz_geom_dpo_plus_dap boltz_geom_dpo_regularized; do
    ckpt="$CKPT_ROOT/$c/dpo_latest.chkpt"
    samples="$CAMPAIGN_DIR/samples_${c}_400.csv"
    ce="$([ -f "$ckpt" ] && echo YES || echo NO)"
    se="$([ -f "$samples" ] && echo "$(wc -l < "$samples")" || echo NO)"
    echo "  $c: chkpt=$ce samples=$se"
  done)
GCS: gs://quris-shahar-dev/boltz_dpo_campaign/
EOF
echo "[$(date -u +%H:%M:%S)] [orchestrator] done → $DONE"
cat "$DONE"
