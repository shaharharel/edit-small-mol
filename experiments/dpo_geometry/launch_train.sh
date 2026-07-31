#!/usr/bin/env bash
# Stream C DPO launcher for V100 ai-gpu2.
# Designed to survive SSH disconnect via systemd-run --scope --collect.
set -uo pipefail
cd "$HOME/edit-small-mol"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate quris

LOG_DIR="$HOME/edit-small-mol/results/paper_evaluation/dpo_geometry"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] launching Stream C DPO training (PID $$) → $LOG"
exec python -u experiments/dpo_geometry/run_stream_c_dpo.py \
  --train \
  --epochs 5 \
  --batch-size 8 \
  --beta 0.1 \
  --lr 5e-5 \
  --wd 1e-2 \
  --warmup-steps 500 \
  --checkpoint-every 500 \
  --max-val-pairs 800 \
  --log-every 50 \
  --device cuda \
  >> "$LOG" 2>&1
