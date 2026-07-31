#!/bin/bash
# Fetch noconstraint cofold harvest CSVs from a100-b + run analysis + build report.
set -e

REMOTE_ROOT=/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/v2_curriculum_clean
LOCAL_ROOT=/Users/shaharharel/Documents/github/edit-small-mol/data/paper_pair_training/v2_curriculum_clean
mkdir -p "$LOCAL_ROOT/noconstraint_cofolds"

echo "[fetch] pulling track_A CSVs..."
for c in theta_90 theta_105 theta_130 null_pose; do
  gcloud compute scp \
    "ai-gpu-a100-b:$REMOTE_ROOT/noconstraint_cofolds/track_A_v2curr_clean_NOCONSTR_${c}.csv" \
    "$LOCAL_ROOT/noconstraint_cofolds/" \
    --zone=us-central1-b 2>&1 | tail -3 || echo "  (missing ${c})"
done

echo "[fetch] pulling driver logs..."
mkdir -p "$LOCAL_ROOT/noconstraint_cofolds/logs"
gcloud compute scp --recurse \
  "ai-gpu-a100-b:$REMOTE_ROOT/noconstraint_cofolds/logs" \
  "$LOCAL_ROOT/noconstraint_cofolds/" \
  --zone=us-central1-b 2>&1 | tail -5 || true

echo "[analyze] running analysis..."
/opt/miniconda3/envs/quris/bin/python /Users/shaharharel/Documents/github/edit-small-mol/experiments/analyze_noconstraint_cofolds.py \
  --cofold_dir "$LOCAL_ROOT/noconstraint_cofolds" \
  --out_summary_json "$LOCAL_ROOT/noconstraint_summary.json" \
  --out_merged_csv "$LOCAL_ROOT/noconstraint_cofolds_merged.csv" \
  --constrained_baseline_dir "$LOCAL_ROOT/constrained_baseline"

echo "[report] building report..."
/opt/miniconda3/envs/quris/bin/python /Users/shaharharel/Documents/github/edit-small-mol/experiments/build_constraint_free_report.py \
  --root "$LOCAL_ROOT"

echo "[done]"
