#!/bin/bash
# Serial per-cohort Boltz cofold for v2_curriculum_clean.
# Runs 4 cohorts one at a time, each with 2 workers = 2 concurrent boltz predicts (safe from OOM).
# Existing CIFs are skipped by driver's `if cif_path.exists()` guard.
set -euo pipefail
ROOT=/home/shaharh_quris_ai/edit-small-mol
CLEAN=$ROOT/data/paper_pair_training/v2_curriculum_clean
STEER=$CLEAN/steering_samples
COFOLD=$CLEAN/cofold_track
LOGS=$ROOT/logs
mkdir -p $COFOLD $LOGS

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

echo "[boltz-serial] PATH check: $(which boltz)"
echo "[boltz-serial] python: $(which python)"
echo "[boltz-serial] start $(date -u +%H:%M:%S)"

FAIL=0
for COHORT in theta_90 theta_105 theta_130 null_pose; do
  LOG=$LOGS/boltz_v2curr_clean_${COHORT}_serial.out
  echo "[boltz-serial] ==== cohort=$COHORT start $(date -u +%H:%M:%S) ===="
  # Run in foreground (serial), 2 workers per cohort
  python $ROOT/experiments/boltz_dpo_campaign_driver.py \
    --cohort v2curr_clean_$COHORT \
    --smiles_csv $STEER/samples_$COHORT.csv \
    --n_target 100 --max_workers 2 > $LOG 2>&1 || {
      echo "[boltz-serial] cohort=$COHORT FAILED (rc=$?)"
      FAIL=$((FAIL + 1))
    }
  echo "[boltz-serial] ==== cohort=$COHORT end $(date -u +%H:%M:%S) ===="
done
echo "[boltz-serial] all cohorts done, fail_count=$FAIL"

echo '=== Collect track_A CSVs ==='
for COHORT in theta_90 theta_105 theta_130 null_pose; do
  SRC=$ROOT/data/paper_pair_training/boltz_dpo_campaign/track_A_v2curr_clean_$COHORT.csv
  DST=$COFOLD/track_A_$COHORT.csv
  if [ -f "$SRC" ]; then
    cp $SRC $DST
    echo "copied $SRC -> $DST ($(wc -l < $DST) rows)"
  else
    echo "WARN: missing $SRC"
  fi
done

echo '=== Re-run head-to-head comparison ==='
cd $ROOT
python experiments/m1a_pocket_decoder/v2/compare_dirty_vs_clean.py 2>&1 | tee $LOGS/compare_v2curr_v3.out

echo '=== Write DONE flag placeholder ==='
touch $CLEAN/.pipeline_boltz_serial_complete
# Remove old broken flag from parallel run
rm -f $CLEAN/.pipeline_boltz_relaunch_complete

echo "[boltz-serial] pipeline COMPLETE $(date -u +%H:%M:%S)"
