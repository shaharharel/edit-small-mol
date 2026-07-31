#!/bin/bash
# Relaunch Phase D Boltz cofolds for v2_curriculum_clean with proper conda env.
# Runs 4 cohorts in parallel, each with 2 Boltz workers = 8 concurrent Boltz predicts.
# Ensures conda env is sourced so `boltz` binary is on PATH.
set -euo pipefail
ROOT=/home/shaharh_quris_ai/edit-small-mol
CLEAN=$ROOT/data/paper_pair_training/v2_curriculum_clean
STEER=$CLEAN/steering_samples
COFOLD=$CLEAN/cofold_track
LOGS=$ROOT/logs
mkdir -p $COFOLD $LOGS

# Ensure conda is available in this shell too (though we won't use its python here)
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

echo "[boltz-relaunch] PATH check: $(which boltz)"
echo "[boltz-relaunch] python: $(which python)"

# Launch each cohort in background with proper env
declare -a PIDS=()
for COHORT in theta_90 theta_105 theta_130 null_pose; do
  LOG=$LOGS/boltz_v2curr_clean_${COHORT}.out
  echo "[boltz-relaunch] launching cohort=$COHORT log=$LOG"
  # Each cohort gets 2 workers. bash -lc ensures the child processes inherit env.
  bash -lc "
    source ~/miniconda3/etc/profile.d/conda.sh
    conda activate quris
    cd $ROOT
    python experiments/boltz_dpo_campaign_driver.py \
      --cohort v2curr_clean_$COHORT \
      --smiles_csv $STEER/samples_$COHORT.csv \
      --n_target 100 --max_workers 2
  " > $LOG 2>&1 &
  PIDS+=($!)
  echo "[boltz-relaunch] cohort=$COHORT pid=${PIDS[-1]}"
  sleep 3  # small stagger so YAML build doesn't race
done

echo "[boltz-relaunch] waiting on PIDs: ${PIDS[@]}"
# Wait for all to complete
FAIL=0
for i in "${!PIDS[@]}"; do
  PID=${PIDS[$i]}
  if wait $PID; then
    echo "[boltz-relaunch] pid=$PID ok"
  else
    echo "[boltz-relaunch] pid=$PID FAILED (rc=$?)"
    FAIL=$((FAIL + 1))
  fi
done
echo "[boltz-relaunch] all cohorts done, fail_count=$FAIL"

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
python experiments/m1a_pocket_decoder/v2/compare_dirty_vs_clean.py 2>&1 | tee $LOGS/compare_v2curr_v2.out

echo '=== Write DONE flag placeholder ==='
touch $CLEAN/.pipeline_boltz_relaunch_complete

echo '=== v2_curriculum_clean Boltz re-launch pipeline COMPLETE ==='
