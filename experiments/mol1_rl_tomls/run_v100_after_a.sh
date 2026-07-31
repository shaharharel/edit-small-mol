#!/bin/bash
# Track B' Mol1-anchored RL on a V100. Waits for Track A (run_aigpu*.sh) to finish
# (= no `reinvent` process running), then runs the 3 RL configs sequentially.
#
# Run as: nohup bash run_v100_after_a.sh > ~/mol1_rl_v100.out 2>&1 &
set -eo pipefail
LOG="$HOME/mol1_rl_v100.log"
date -u +"[%FT%TZ] Track B' launcher armed; waiting for Track A to release GPU" | tee -a "$LOG"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

# Wait for any running reinvent process to finish (Track A)
while pgrep -f 'reinvent.*\.toml' >/dev/null; do
  sleep 30
done
date -u +"[%FT%TZ] No reinvent running. Track A done. First, retry any missing Track A cohorts." | tee -a "$LOG"

PROJ=$HOME/edit-small-mol
TOMLS=$PROJ/experiments/mol1_rl_tomls
A_TOMLS=$PROJ/experiments/mol1_anchored_tomls

# Retry any Track A cohort that has empty or missing sampling.csv
A_COHORTS=(
  "exp2_preRL_mol1_50K"
  "exp6_preRL_mol1_50K"
  "exp6v3_preRL_mol1_50K"
  "exp2_v2_rl_postRL_mol1_50K"
  "exp2_v2_rl_v2_postRL_mol1_50K"
  "exp6_v3_postRL_mol1_50K"
  "exp6_v4_postRL_mol1_50K"
  "exp6_v5_postRL_mol1_50K"
)
for cohort in "${A_COHORTS[@]}"; do
  out_csv=$PROJ/data/mol1_anchored_tier4/$cohort/sampling.csv
  if [ -f "$out_csv" ] && [ "$(wc -l <"$out_csv")" -gt 100 ]; then
    continue
  fi
  toml=$A_TOMLS/$cohort.toml
  model=$(grep -E '^model_file' $toml | head -1 | sed 's/.*= "\(.*\)"/\1/')
  if [ ! -f "$model" ]; then
    date -u +"[%FT%TZ] [retry] $cohort: model $model NOT FOUND, skipping" | tee -a "$LOG"
    continue
  fi
  date -u +"[%FT%TZ] [retry] $cohort: missing or empty -> running" | tee -a "$LOG"
  mkdir -p "$(dirname $out_csv)"
  timeout 3600 reinvent $toml -d cuda 2>&1 | tee -a "$LOG" || {
    date -u +"[%FT%TZ] [retry] $cohort: FAILED" | tee -a "$LOG"
    continue
  }
  date -u +"[%FT%TZ] [retry] $cohort: DONE" | tee -a "$LOG"
done

date -u +"[%FT%TZ] Track A retries complete. Starting Track B'." | tee -a "$LOG"

COHORTS=(
  "mol1RL_v5_seed_mol1_only"
  "mol1RL_v5_seed_zap70_all_plus_mol1"
  # B'-3 (kinase) runs on A100 (ai-gpu-a100) — skipped here
)

for cohort in "${COHORTS[@]}"; do
  work=$PROJ/results/paper_evaluation/mol1_rl/$cohort
  mkdir -p "$work"
  cd "$work"
  cp $TOMLS/$cohort.toml ./$cohort.toml
  date -u +"[%FT%TZ] === $cohort START ===" | tee -a "$LOG"
  # Per-cohort timeout: 7200s (2h) — kill if it doesn't finish
  timeout 7200 reinvent ./$cohort.toml -d cuda 2>&1 | tee -a "$LOG" || {
    date -u +"[%FT%TZ] === $cohort FAILED or TIMED OUT (continuing) ===" | tee -a "$LOG"
    continue
  }
  date -u +"[%FT%TZ] === $cohort DONE ===" | tee -a "$LOG"
done

date -u +"[%FT%TZ] Track B' ALL DONE on V100" | tee -a "$LOG"
