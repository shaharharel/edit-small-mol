#!/bin/bash
# Run after the primary local docking finishes. Picks up the bigger baseline cohorts.
set -e
LOG=results/paper_evaluation/cohort_comparison/local_followon.log
mkdir -p "$(dirname $LOG)"
exec > >(tee -a "$LOG") 2>&1
echo "=== FOLLOWON START $(date -u +%FT%TZ) ==="

for COHORT in LibInvent_locked Amine_Replacements; do
  echo "=== $COHORT === $(date -u +%FT%TZ)"
  conda run --no-capture-output -n quris python -u experiments/cohort_comparison_dock.py \
    --cohort "$COHORT" \
    --input-sdf "data/cohort_comparison/cohorts/$COHORT/input.sdf" \
    --out-dir results/paper_evaluation/cohort_comparison/per_cohort \
    --workers 4 --threads 2 \
    || echo "$COHORT failed"
done
echo "=== FOLLOWON DONE $(date -u +%FT%TZ) ==="
touch results/paper_evaluation/cohort_comparison/FOLLOWON_DONE
