#!/bin/bash
# One-shot: pull every available per_mol.csv from ai-chem2, then run stats + report.
# Idempotent.
set -e
PROJECT=/Users/shaharharel/Documents/github/edit-small-mol
cd "$PROJECT"

LOCAL_BASE=results/paper_evaluation/cohort_comparison/per_cohort
mkdir -p "$LOCAL_BASE"

# Pull from ai-chem2
for COHORT in H2 H3 L1_FT_H2 LibInvent_locked Amine_Replacements; do
  REMOTE_CSV="~/cohort_dock/results/per_cohort/$COHORT/per_mol.csv"
  REMOTE_SDF="~/cohort_dock/results/per_cohort/$COHORT/top10.sdf"
  LOCAL_DIR="$LOCAL_BASE/$COHORT"
  # Check existence on remote via stat
  exists=$(gcloud compute ssh ai-chem2 --zone=us-east1-b --command="test -f $REMOTE_CSV && echo Y || echo N" 2>/dev/null | tail -1 || echo N)
  if [ "$exists" != "Y" ]; then
    echo "[skip] $COHORT — remote CSV not yet present"
    continue
  fi
  # If local CSV already exists and has more rows than expected, skip (avoid clobber from Mac's parallel completion)
  if [ -f "$LOCAL_DIR/per_mol.csv" ]; then
    local_rows=$(wc -l < "$LOCAL_DIR/per_mol.csv" 2>/dev/null || echo 0)
    if [ "$local_rows" -gt "100" ]; then
      echo "[skip] $COHORT — local CSV already populated ($local_rows rows)"
      continue
    fi
  fi
  mkdir -p "$LOCAL_DIR"
  echo "[fetch] $COHORT"
  gcloud compute scp ai-chem2:$REMOTE_CSV "$LOCAL_DIR/per_mol.csv" --zone=us-east1-b 2>/dev/null || echo "  CSV scp failed"
  gcloud compute scp ai-chem2:$REMOTE_SDF "$LOCAL_DIR/top10.sdf" --zone=us-east1-b 2>/dev/null || true
done

echo ""
echo "--- Running stats ---"
conda run --no-capture-output -n quris python -u experiments/cohort_comparison_stats.py

echo ""
echo "--- Running report ---"
conda run --no-capture-output -n quris python -u experiments/cohort_comparison_report.py
