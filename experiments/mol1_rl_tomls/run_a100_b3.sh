#!/bin/bash
# Track B'-3 (kinase + ZAP70x5 + Mol1x20 seeds) on A100
set -eo pipefail
LOG="$HOME/mol1_rl_a100_b3.log"
date -u +"[%FT%TZ] A100 Track B'-3 start" | tee -a "$LOG"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

PROJ=$HOME/edit-small-mol
TOMLS=$PROJ/experiments/mol1_rl_tomls
COHORT="mol1RL_v5_seed_kinase_zap70x5_mol1x20"

work=$PROJ/results/paper_evaluation/mol1_rl/$COHORT
mkdir -p "$work"
cd "$work"
cp $TOMLS/$COHORT.toml ./$COHORT.toml
date -u +"[%FT%TZ] === $COHORT START on A100 ===" | tee -a "$LOG"
# 2h cap; A100 should finish faster
timeout 7200 reinvent ./$COHORT.toml -d cuda 2>&1 | tee -a "$LOG" || {
  date -u +"[%FT%TZ] === $COHORT FAILED/TIMED OUT ===" | tee -a "$LOG"
  exit 1
}
date -u +"[%FT%TZ] === $COHORT DONE on A100 ===" | tee -a "$LOG"
