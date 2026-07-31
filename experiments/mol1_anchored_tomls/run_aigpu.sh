#!/bin/bash
# ai-gpu shard: 4 cohorts
set -eo pipefail
LOG="$HOME/mol1_anchored_sampling_aigpu.log"
date -u +"[%FT%TZ] ai-gpu shard start" | tee -a "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

PROJ=$HOME/edit-small-mol
TOMLS=$PROJ/experiments/mol1_anchored_tomls

COHORTS=(
  "exp2_preRL_mol1_50K"
  "exp6_preRL_mol1_50K"
  "exp2_v2_rl_postRL_mol1_50K"
  "exp6_v3_postRL_mol1_50K"
)

for cohort in "${COHORTS[@]}"; do
  out_dir=$PROJ/data/mol1_anchored_tier4/$cohort
  out_csv=$out_dir/sampling.csv
  mkdir -p "$out_dir"
  if [ -f "$out_csv" ] && [ "$(wc -l <"$out_csv")" -gt 1000 ]; then
    date -u +"[%FT%TZ] SKIP $cohort (sampling.csv has $(wc -l <"$out_csv") lines)" | tee -a "$LOG"
    continue
  fi
  date -u +"[%FT%TZ] === $cohort START ===" | tee -a "$LOG"
  reinvent "$TOMLS/$cohort.toml" -d cuda 2>&1 | tee -a "$LOG" || {
    date -u +"[%FT%TZ] === $cohort FAILED (continuing) ===" | tee -a "$LOG"
    continue
  }
  n=$(wc -l <"$out_csv" 2>/dev/null || echo 0)
  date -u +"[%FT%TZ] === $cohort DONE, $n lines ===" | tee -a "$LOG"
done

date -u +"[%FT%TZ] ai-gpu shard ALL DONE" | tee -a "$LOG"
