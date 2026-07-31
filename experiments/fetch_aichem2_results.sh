#!/bin/bash
# Pull per_mol.csv and top10.sdf from ai-chem2 for each cohort that has a complete CSV.
# Idempotent: only fetches if local copy is missing or smaller.
set -e
LOCAL_BASE=results/paper_evaluation/cohort_comparison/per_cohort
mkdir -p "$LOCAL_BASE"

for COHORT in H2 H3 L1_FT_H2 LibInvent_locked Amine_Replacements; do
  REMOTE_CSV="~/cohort_dock/results/per_cohort/$COHORT/per_mol.csv"
  REMOTE_SDF="~/cohort_dock/results/per_cohort/$COHORT/top10.sdf"
  LOCAL_CSV="$LOCAL_BASE/$COHORT/per_mol.csv"
  LOCAL_SDF="$LOCAL_BASE/$COHORT/top10.sdf"
  # Check existence
  exists=$(gcloud compute ssh ai-chem2 --zone=us-east1-b --command="ls -la $REMOTE_CSV 2>/dev/null | wc -l" 2>&1 | tail -1)
  if [ "$exists" != "1" ]; then
    echo "[skip] $COHORT: remote CSV not yet present"
    continue
  fi
  mkdir -p "$LOCAL_BASE/$COHORT"
  echo "[fetch] $COHORT"
  gcloud compute scp ai-chem2:$REMOTE_CSV "$LOCAL_CSV" --zone=us-east1-b 2>/dev/null || echo "  CSV scp failed"
  gcloud compute scp ai-chem2:$REMOTE_SDF "$LOCAL_SDF" --zone=us-east1-b 2>/dev/null || echo "  SDF scp failed"
done
echo "fetch done"
