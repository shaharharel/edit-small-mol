#!/bin/bash
# Final pipeline: re-metric, enrich, stats, report.
set -e
cd /Users/shaharharel/Documents/github/edit-small-mol

echo "=== STEP 1: Pull remote results from ai-chem2 ==="
for COHORT in H2 H3 L1_FT_H2 LibInvent_locked Amine_Replacements; do
  REMOTE_CSV="~/cohort_dock/results/per_cohort/$COHORT/per_mol.csv"
  REMOTE_SDF="~/cohort_dock/results/per_cohort/$COHORT/top10.sdf"
  REMOTE_POSES="~/cohort_dock/results/per_cohort/$COHORT/poses"
  REMOTE_LIGS="~/cohort_dock/results/per_cohort/$COHORT/ligands_pdbqt"
  LOCAL_DIR="results/paper_evaluation/cohort_comparison/per_cohort/$COHORT"
  # Skip if local already populated with >=100 rows (prefer Mac result)
  if [ -f "$LOCAL_DIR/per_mol.csv" ]; then
    local_rows=$(wc -l < "$LOCAL_DIR/per_mol.csv" 2>/dev/null || echo 0)
    if [ "$local_rows" -gt "100" ]; then
      echo "[skip] $COHORT — already populated locally ($local_rows rows)"
      continue
    fi
  fi
  exists=$(gcloud compute ssh ai-chem2 --zone=us-east1-b --command="test -f $REMOTE_CSV && echo Y || echo N" 2>/dev/null | tail -1 || echo N)
  if [ "$exists" != "Y" ]; then
    echo "[skip] $COHORT — remote CSV not present"
    continue
  fi
  mkdir -p "$LOCAL_DIR/poses" "$LOCAL_DIR/ligands_pdbqt"
  echo "[fetch] $COHORT"
  gcloud compute scp ai-chem2:$REMOTE_CSV "$LOCAL_DIR/per_mol.csv" --zone=us-east1-b 2>/dev/null || true
  gcloud compute scp ai-chem2:$REMOTE_SDF "$LOCAL_DIR/top10.sdf" --zone=us-east1-b 2>/dev/null || true
  # Bulk fetch pose PDBQTs and ligand PDBQTs for re-metric.
  echo "  fetching poses (this may take a moment)..."
  gcloud compute scp --recurse "ai-chem2:$REMOTE_POSES" "$LOCAL_DIR/" --zone=us-east1-b 2>/dev/null || true
  gcloud compute scp --recurse "ai-chem2:$REMOTE_LIGS" "$LOCAL_DIR/" --zone=us-east1-b 2>/dev/null || true
done

echo ""
echo "=== STEP 2: Re-metric cohorts (consistent thresholds) ==="
conda run --no-capture-output -n quris python -u experiments/cohort_comparison_remetric.py

echo ""
echo "=== STEP 3: Enrich with input-pose geometry ==="
conda run --no-capture-output -n quris python -u experiments/enrich_input_geometry.py

echo ""
echo "=== STEP 4: Compute stats ==="
conda run --no-capture-output -n quris python -u experiments/cohort_comparison_stats.py

echo ""
echo "=== STEP 5: Generate report ==="
conda run --no-capture-output -n quris python -u experiments/cohort_comparison_report.py

echo ""
echo "=== FINALIZE DONE $(date -u +%FT%TZ) ==="
echo "Outputs:"
echo "  results/paper_evaluation/cohort_comparison/all_cohorts_metrics.csv"
echo "  results/paper_evaluation/cohort_comparison/per_cohort_summary.csv"
echo "  results/paper_evaluation/cohort_comparison/comparison_pvalues.csv"
echo "  results/paper_evaluation/cohort_comparison/comparison_effect_sizes.csv"
echo "  results/paper_evaluation/cohort_comparison/report.md"
echo "  results/paper_evaluation/cohort_comparison/figures/"
