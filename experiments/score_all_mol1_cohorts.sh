#!/bin/bash
# Score all sampled Mol1-anchored cohorts with fast metrics (no FiLM)
# Outputs to data/tier4_scored/{tag}_scored.csv
set -o pipefail
PROJ="/Users/shaharharel/Documents/github/edit-small-mol"
PY="/opt/miniconda3/envs/quris/bin/python"
SCORER="$PROJ/experiments/score_mol1_cohort_fast.py"
LOG="$PROJ/score_all.log"

MOL1="C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

echo "[$(date +%H:%M:%S)] === SCORING ALL COHORTS (no FiLM, fast pass) ===" | tee -a "$LOG"

COHORTS=(
  "mol1RL_v5_seed_mol1_only:$PROJ/data/local_sampling/mol1RL_v5_seed_mol1_only_mps_sample.csv"
  "thiq_rl_zap70:$PROJ/data/local_sampling/thiq_rl_zap70_mps_sample.csv"
  "thiq_rl_kinase:$PROJ/data/local_sampling/thiq_rl_kinase_mps_sample.csv"
  "thiq_rl_exp2_kinase:$PROJ/data/local_sampling/thiq_rl_exp2_kinase_mps_sample.csv"
  "murcko_rl_zap70:$PROJ/data/local_sampling/murcko_rl_zap70_mps_sample.csv"
)

for entry in "${COHORTS[@]}"; do
  TAG="${entry%%:*}"
  INPUT="${entry##*:}"
  OUTPUT="$PROJ/data/tier4_scored/${TAG}_scored.csv"
  if [ ! -f "$INPUT" ]; then
    echo "[$(date +%H:%M:%S)] SKIP $TAG: no input" | tee -a "$LOG"
    continue
  fi
  if [ -f "$OUTPUT" ] && [ "$(wc -l < "$OUTPUT")" -gt 100 ]; then
    echo "[$(date +%H:%M:%S)] SKIP $TAG: output exists" | tee -a "$LOG"
    continue
  fi
  echo "[$(date +%H:%M:%S)] >>> $TAG" | tee -a "$LOG"
  $PY "$SCORER" "$INPUT" "$OUTPUT" --tag "$TAG" --seed "$MOL1" --no-film 2>&1 | tee -a "$LOG"
  echo "[$(date +%H:%M:%S)] <<< $TAG done" | tee -a "$LOG"
done

echo "[$(date +%H:%M:%S)] === ALL DONE ===" | tee -a "$LOG"
