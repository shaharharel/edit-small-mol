#!/bin/bash
# Run pubTc + xTB enrichment over all 7 tier4_scored cohorts on ai-chem.
# Designed to be invoked inside a tmux session: ~30 CPU workers, sequential per-cohort.
set -e
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris
cd ~/edit-small-mol

COHORTS=(
  "murcko_rl_exp2_zap70_scored.csv"
  "thiq_rl_mol1only_scored.csv"
  "murcko_rl_zap70_scored.csv"
  "mol1RL_v5_seed_mol1_only_scored.csv"
  "thiq_rl_exp2_kinase_scored.csv"
  "thiq_rl_zap70_scored.csv"
  "thiq_rl_kinase_scored.csv"
)

mkdir -p logs/enrich
T0=$(date +%s)
for c in "${COHORTS[@]}"; do
  echo "============================================================"
  echo "[$(date +%H:%M:%S)] starting $c"
  echo "============================================================"
  python experiments/enrich_tier4_pubtc_xtb.py \
    --cohort_csv data/tier4_scored/$c \
    --workers 30 \
    2>&1 | tee logs/enrich/$c.log
  echo "[$(date +%H:%M:%S)] finished $c"
done
T1=$(date +%s)
echo "============================================================"
echo "ALL DONE in $(( (T1-T0)/60 )) minutes"
echo "============================================================"
