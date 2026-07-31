#!/bin/bash
# Stops ai-chem2 once all Lingo priority cohorts (H2, H3, L1_FT_H2) finish.
# Run AFTER fetching all cohorts to local. Cost-saving for the n2-standard-16
# n2-standard-16 STANDARD instance (~$0.78/hr).
set -e
echo "Waiting for ai-chem2 priority cohorts (H2, H3, L1_FT_H2) to finish..."
while true; do
  done_priority=$(gcloud compute ssh ai-chem2 --zone=us-east1-b --command='ls ~/cohort_dock/results/per_cohort/H2/per_mol.csv ~/cohort_dock/results/per_cohort/H3/per_mol.csv ~/cohort_dock/results/per_cohort/L1_FT_H2/per_mol.csv 2>/dev/null | wc -l' 2>/dev/null | tail -1 || echo 0)
  if [ "$done_priority" -ge "3" ]; then
    echo "All 3 priority cohorts done. Stopping ai-chem2."
    gcloud compute instances stop ai-chem2 --zone=us-east1-b
    gcloud compute instances describe ai-chem2 --zone=us-east1-b --format='value(status)'
    exit 0
  fi
  sleep 120
done
