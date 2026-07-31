#!/usr/bin/env bash
# Upload a cohort's checkpoint + samples to GCS, then write handoff coord file.
# Usage: upload_cohort.sh <cohort_name>   (e.g. boltz_geom_dpo_only)
set -euo pipefail

COHORT="${1:?cohort name required (e.g. boltz_geom_dpo_only)}"
PROJECT_ROOT="$HOME/edit-small-mol"
CKPT="$PROJECT_ROOT/models/dpo_checkpoints/$COHORT/dpo_latest.chkpt"
CAMPAIGN_DIR="$PROJECT_ROOT/data/paper_pair_training/boltz_dpo_campaign"
SAMPLES_400="$CAMPAIGN_DIR/samples_${COHORT}_400.csv"
SAMPLES_RAW="$CAMPAIGN_DIR/samples_${COHORT}_raw.csv"

GCS_BASE="gs://quris-shahar-dev/boltz_dpo_campaign/$COHORT"

if [ ! -f "$CKPT" ]; then
  echo "[ERROR] missing checkpoint $CKPT" >&2
  exit 1
fi
if [ ! -f "$SAMPLES_400" ]; then
  echo "[ERROR] missing filtered samples $SAMPLES_400" >&2
  exit 1
fi

echo "[upload] uploading $COHORT to $GCS_BASE"
gsutil -m cp "$CKPT" "$GCS_BASE/dpo_latest.chkpt"
gsutil -m cp "$SAMPLES_400" "$GCS_BASE/samples_${COHORT}_400.csv"
if [ -f "$SAMPLES_RAW" ]; then
  gsutil -m cp "$SAMPLES_RAW" "$GCS_BASE/samples_${COHORT}_raw.csv"
fi

# Upload training + summary for completeness
TRAIN_HIST="$PROJECT_ROOT/results/paper_evaluation/boltz_dpo_campaign/$COHORT/training_history.csv"
TRAIN_SUM="$PROJECT_ROOT/results/paper_evaluation/boltz_dpo_campaign/$COHORT/training_summary.json"
[ -f "$TRAIN_HIST" ] && gsutil -m cp "$TRAIN_HIST" "$GCS_BASE/training_history.csv"
[ -f "$TRAIN_SUM" ] && gsutil -m cp "$TRAIN_SUM" "$GCS_BASE/training_summary.json"

echo "[upload] listing:"
gsutil ls "$GCS_BASE/"

# Write coord file locally for pull-back
mkdir -p "$PROJECT_ROOT/data/agent_coord"
COORD_FILE="$PROJECT_ROOT/data/agent_coord/from_dpo_trainer_${COHORT}_READY.txt"
cat > "$COORD_FILE" <<EOF
Cohort: $COHORT
Ready: $(date -u +%Y-%m-%dT%H:%M:%SZ)
GCS: $GCS_BASE/
Files:
  - dpo_latest.chkpt
  - samples_${COHORT}_400.csv
  - samples_${COHORT}_raw.csv
  - training_history.csv
  - training_summary.json
Message: "$COHORT ready at $GCS_BASE/ — start cofolding"
EOF
echo "[coord] wrote $COORD_FILE"

echo "[done] $COHORT upload complete"
