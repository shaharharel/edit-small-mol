#!/bin/bash
set -e
LOG=results/paper_evaluation/cohort_comparison/local_run.log
mkdir -p "$(dirname $LOG)"
exec > >(tee -a "$LOG") 2>&1
echo "=== START $(date -u +%FT%TZ) ==="

# Smallest -> largest to fail fast (now includes LibInvent + Amine for load balance)
for COHORT in DeNovo_warhead_gate L_locked C5 H1 Mol2Mol_warhead_gate LibInvent_locked Amine_Replacements; do
  echo "=== $COHORT === $(date -u +%FT%TZ)"
  conda run --no-capture-output -n quris python -u experiments/cohort_comparison_dock.py \
    --cohort "$COHORT" \
    --input-sdf "data/cohort_comparison/cohorts/$COHORT/input.sdf" \
    --out-dir results/paper_evaluation/cohort_comparison/per_cohort \
    --workers 4 --threads 2 \
    || echo "$COHORT failed (continuing)"
done

echo "=== ALL LOCAL DONE $(date -u +%FT%TZ) ==="
touch results/paper_evaluation/cohort_comparison/LOCAL_DONE
