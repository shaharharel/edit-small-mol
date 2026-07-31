#!/bin/bash
# v2_curriculum_clean pipeline (Phase C + D). Runs on ai-gpu-a100-b.
# Assumes CLEAN pairs already built by build_pairs_v2_curriculum.py with
# --require_both_geoms 1 (see data/paper_pair_training/v2_curriculum_clean/pair_stats.md).
set -euo pipefail
ROOT=/home/shaharh_quris_ai/edit-small-mol
CLEAN=$ROOT/data/paper_pair_training/v2_curriculum_clean
MODELS=$ROOT/models/v2_curriculum_clean
LOGS=$ROOT/logs
PY=/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python
mkdir -p $MODELS $LOGS

echo '=== Phase C: train v2_curriculum_clean (max 10000 steps) ==='
cd $ROOT
$PY experiments/m1a_pocket_decoder/v2/train_v2_curriculum.py \
  --train_npz $CLEAN/pairs_train.npz \
  --val_npz   $CLEAN/pairs_val.npz \
  --warm_start $ROOT/models/m1a_v2.ckpt \
  --out_dir $MODELS \
  --curve_path $CLEAN/training_curve.json \
  --lr 5e-5 --bs 24 --weight_decay 1e-2 \
  --warmup_steps 500 --max_steps 10000 \
  --ckpt_interval 1000 --val_interval 500 --log_every 50 \
  --seed 0 2>&1 | tee $LOGS/train_v2_curriculum_clean.out

echo '=== Phase D Step 1: sample steering diagnostic (100 per cohort) ==='
STEER_DIR=$CLEAN/steering_samples
mkdir -p $STEER_DIR
$PY experiments/m1a_pocket_decoder/v2/sample_v2_curriculum_diagnostic.py \
  --ckpt $MODELS/best.chkpt \
  --out_dir $STEER_DIR \
  --n 100 --batch_size 64 --temperature 1.0 2>&1 | tee $LOGS/sample_v2_curriculum_clean.out

echo '=== Phase D Step 2: Boltz cofold each cohort ==='
COFOLD_DIR=$CLEAN/cofold_track
mkdir -p $COFOLD_DIR
for COHORT in theta_90 theta_105 theta_130 null_pose; do
  echo "--- Cohort: $COHORT ---"
  $PY experiments/boltz_dpo_campaign_driver.py \
    --cohort v2curr_clean_$COHORT \
    --smiles_csv $STEER_DIR/samples_$COHORT.csv \
    --n_target 100 --max_workers 3 2>&1 | tee -a $LOGS/boltz_v2curr_clean.out \
    || echo "cohort $COHORT had errors — continuing"
done

echo '=== Phase D Step 3: collect track_A CSVs ==='
for COHORT in theta_90 theta_105 theta_130 null_pose; do
  SRC=$ROOT/data/paper_pair_training/boltz_dpo_campaign/track_A_v2curr_clean_$COHORT.csv
  DST=$COFOLD_DIR/track_A_$COHORT.csv
  if [ -f "$SRC" ]; then
    cp $SRC $DST
    echo "copied $SRC -> $DST"
  else
    echo "WARN: missing $SRC"
  fi
done

echo '=== Phase E: head-to-head comparison analysis ==='
$PY experiments/m1a_pocket_decoder/v2/compare_dirty_vs_clean.py 2>&1 | tee $LOGS/compare_v2curr.out

echo '=== v2_curriculum_clean pipeline COMPLETE ==='
