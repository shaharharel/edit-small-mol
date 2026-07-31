#!/bin/bash
# Track B' Mol1-anchored RL on A100 — 3 RL configs sequentially
set -eo pipefail
LOG="$HOME/mol1_rl_a100.log"
date -u +"[%FT%TZ] Track B mol1-RL queue start" | tee -a "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

PROJ=/home/shaharh_quris_ai/edit-small-mol
TOMLS=$PROJ/experiments/mol1_rl_tomls

COHORTS=(
  "mol1RL_v5_seed_mol1_only"
  "mol1RL_v5_seed_zap70_all_plus_mol1"
  "mol1RL_v5_seed_kinase_zap70x5_mol1x20"
)

for cohort in "${COHORTS[@]}"; do
  work=$PROJ/results/paper_evaluation/mol1_rl/$cohort
  mkdir -p "$work"
  cd "$work"
  cp $TOMLS/$cohort.toml ./$cohort.toml
  date -u +"[%FT%TZ] === $cohort START ===" | tee -a "$LOG"
  reinvent ./$cohort.toml -d cuda 2>&1 | tee -a "$LOG" || {
    date -u +"[%FT%TZ] === $cohort FAILED (continuing) ===" | tee -a "$LOG"
    continue
  }
  date -u +"[%FT%TZ] === $cohort DONE ===" | tee -a "$LOG"
done

date -u +"[%FT%TZ] Track B mol1-RL ALL DONE" | tee -a "$LOG"
