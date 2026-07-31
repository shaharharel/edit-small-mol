#!/bin/bash
set -eo pipefail
LOG="$HOME/mol1_anchored_sampling.log"
echo "[$(date -u +%FT%TZ)] Mol1-anchored sampling queue start" >> "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

echo "[$(date -u +%FT%TZ)] === exp2_preRL_mol1_50K (preRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp2_preRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp2_preRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp2_preRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp6_preRL_mol1_50K (preRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp6_preRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp6_preRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp6_preRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp6v3_preRL_mol1_50K (preRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp6v3_preRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp6v3_preRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp6v3_preRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp2_v2_rl_postRL_mol1_50K (postRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp2_v2_rl_postRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp2_v2_rl_postRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp2_v2_rl_postRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp2_v2_rl_v2_postRL_mol1_50K (postRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp2_v2_rl_v2_postRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp2_v2_rl_v2_postRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp2_v2_rl_v2_postRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp6_v3_postRL_mol1_50K (postRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp6_v3_postRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp6_v3_postRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp6_v3_postRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp6_v4_postRL_mol1_50K (postRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp6_v4_postRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp6_v4_postRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp6_v4_postRL_mol1_50K done ===" >> "$LOG"

echo "[$(date -u +%FT%TZ)] === exp6_v5_postRL_mol1_50K (postRL) ===" >> "$LOG"
mkdir -p "/home/shaharh_quris_ai/edit-small-mol/data/mol1_anchored_tier4/exp6_v5_postRL_mol1_50K"
reinvent "/home/shaharh_quris_ai/edit-small-mol/experiments/mol1_anchored_tomls/exp6_v5_postRL_mol1_50K.toml" -d cuda 2>&1 | tee -a "$LOG"
echo "[$(date -u +%FT%TZ)] === exp6_v5_postRL_mol1_50K done ===" >> "$LOG"
