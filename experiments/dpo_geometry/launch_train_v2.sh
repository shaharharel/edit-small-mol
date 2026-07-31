#!/usr/bin/env bash
# Stream C v2 DPO launcher: gentler hyperparams + Mol1-similar pairs only.
# - lr 1e-5 (was 5e-5; v1 collapsed scaffold)
# - epochs 3 (was 5)
# - 12.7K pairs (was 52K; filtered to Mol1-similar)
set -uo pipefail
cd "$HOME/edit-small-mol"
source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate quris

LOG_DIR="$HOME/edit-small-mol/results/paper_evaluation/dpo_geometry"
mkdir -p "$LOG_DIR"
LOG="$LOG_DIR/train_v2_$(date +%Y%m%d_%H%M%S).log"

echo "[$(date)] launching Stream C v2 DPO training (PID $$) → $LOG"
exec python -u experiments/dpo_geometry/run_stream_c_dpo.py \
  --train \
  --pairs $HOME/edit-small-mol/data/dpo_pairs/geometry_only_v2.parquet \
  --trained-ckpt $LOG_DIR/stream_c_dpo_v2.chkpt \
  --epochs 3 \
  --batch-size 8 \
  --beta 0.1 \
  --lr 1e-5 \
  --wd 1e-2 \
  --warmup-steps 200 \
  --checkpoint-every 500 \
  --max-val-pairs 600 \
  --log-every 50 \
  --device cuda \
  >> "$LOG" 2>&1
