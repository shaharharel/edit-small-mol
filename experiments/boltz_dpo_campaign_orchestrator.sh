#!/bin/bash
# Full campaign orchestrator: waits for C1 to complete, then loops through C2/C3/C4
# pulling artifacts from GCS as they arrive.
#
# Idempotent: re-invoke to resume. Uses per-cohort DONE marker.

set -u
ROOT=/home/shaharh_quris_ai/edit-small-mol
CAMPAIGN=$ROOT/data/paper_pair_training/boltz_dpo_campaign
SAMPLES=$CAMPAIGN/samples
LOGS=$CAMPAIGN/logs
CKPT_DIR=$ROOT/models/dpo_checkpoints
GCS=gs://quris-shahar-dev/boltz_dpo_campaign

mkdir -p $SAMPLES $LOGS

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

log() { echo "[$(date -u +%FT%TZ)] $*" | tee -a $LOGS/orchestrator.log; }

wait_for_cofold_done() {
  # Wait until no boltz predict processes are running AND no driver instance for cohort
  local cohort=$1
  log "waiting for cofolds of $cohort to finish..."
  while pgrep -f "boltz_dpo_campaign_driver.py.*$cohort" > /dev/null || pgrep -f "boltz predict" > /dev/null; do
    sleep 60
  done
  log "cofold barrier passed for $cohort"
}

process_cohort() {
  local cohort=$1
  local gcs_folder=$2
  local target_n=$3

  local done_flag=$CAMPAIGN/${cohort}.DONE
  if [ -f "$done_flag" ]; then
    log "$cohort already DONE ($done_flag) — skip"
    return 0
  fi

  # Poll GCS until artifacts arrive
  log "polling GCS for $cohort ($gcs_folder)..."
  local samples_csv=$SAMPLES/${cohort}_400.csv
  while true; do
    if gsutil -q stat $GCS/$gcs_folder/samples_${gcs_folder}_400.csv 2>/dev/null; then
      log "$cohort artifacts ready in GCS"
      break
    fi
    sleep 300
  done

  # Fetch
  mkdir -p $CKPT_DIR/$gcs_folder
  gsutil cp $GCS/$gcs_folder/dpo_latest.chkpt $CKPT_DIR/$gcs_folder/dpo_latest.chkpt
  gsutil cp $GCS/$gcs_folder/samples_${gcs_folder}_400.csv $samples_csv
  gsutil cp $GCS/$gcs_folder/samples_${gcs_folder}_raw.csv $SAMPLES/${cohort}_raw.csv || true
  gsutil cp $GCS/$gcs_folder/training_summary.json $CKPT_DIR/$gcs_folder/ || true
  log "$cohort artifacts fetched"

  # Cofold — reuse the driver, feed the pre-filtered samples CSV
  log "$cohort starting cofold pipeline (n=$target_n)"
  nohup python $ROOT/experiments/boltz_dpo_campaign_driver.py \
    --cohort $cohort \
    --smiles_csv $samples_csv \
    --n_target $target_n \
    --max_workers 2 \
    > $LOGS/driver_${cohort}.log 2>&1 &
  local pid=$!
  log "$cohort driver PID=$pid"

  # Wait for driver to finish
  wait $pid
  local rc=$?
  log "$cohort driver rc=$rc"

  # Mark DONE
  touch $done_flag
  log "$cohort DONE"
}

log "orchestrator started"

# Wait for C1 to fully finish (both first batch of 70 AND merged batch)
log "waiting for C1 to finish..."
while pgrep -f "boltz_dpo_campaign_driver.py.*c1_composite_v1" > /dev/null; do
  sleep 60
done
# Also wait for the restart helper's v2 driver to spawn and finish
sleep 30
while pgrep -f "boltz_dpo_campaign_driver.py.*c1_composite_v1" > /dev/null; do
  sleep 60
done
# Also wait for restart helper itself
while pgrep -f "boltz_dpo_c1_restart_when_ready" > /dev/null; do
  sleep 60
done
log "C1 pipeline complete — marking DONE"
touch $CAMPAIGN/c1_composite_v1.DONE

# Now serialize C2, C3, C4 — each polls GCS, fetches, cofolds
process_cohort c2_boltz_geom_dpo_only          boltz_geom_dpo_only          400
process_cohort c3_boltz_geom_dpo_plus_dap      boltz_geom_dpo_plus_dap      400
process_cohort c4_boltz_geom_dpo_regularized   boltz_geom_dpo_regularized   400

# Generate final report
log "generating final report"
python $ROOT/experiments/boltz_dpo_campaign_report.py
mkdir -p $ROOT/data/agent_coord
touch $ROOT/data/agent_coord/from_boltz_dpo_campaign_DONE.txt
log "campaign DONE — flag written"
