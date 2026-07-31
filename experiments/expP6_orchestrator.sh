#!/bin/bash
# P6 overnight orchestrator on ai-gpu.
# Chains: RL sample+score → P6-DIR train → P6-DIVERSE train → P6-POSENOISE train.
# Emits progress to data/exp_P6/orchestrator.log.
set -u
cd /home/shaharh_quris_ai/edit-small-mol
source /home/shaharh_quris_ai/miniconda3/etc/profile.d/conda.sh
conda activate quris

LOG=data/exp_P6/orchestrator.log
mkdir -p data/exp_P6/{samples,models,rl_logs}
: > $LOG

log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }

wait_for_file() {
  local f="$1"; local timeout="${2:-3600}"
  local waited=0
  while [[ ! -f "$f" && $waited -lt $timeout ]]; do
    sleep 30; waited=$((waited+30))
  done
  [[ -f "$f" ]]
}

# 1) Wait for RL-MINSTRICT to finish.
log "Waiting for RL ckpt models/exp_P6/RL_MINSTRICT_from_P51v2.ckpt..."
while pgrep -f 'train_m1a_v2_dap_MIN_planar_STRICT' > /dev/null; do sleep 30; done
if [[ -f models/exp_P6/RL_MINSTRICT_from_P51v2.ckpt ]]; then
  log "RL done. Skipping RL sample+score (deferred to bulk scoring later)."
else
  log "RL ckpt NOT found — training may have failed. Check RL log."
fi

# 2) Wait for P6-DIR mine to finish (should be ~1 min).
log "Waiting for P6-DIR mine data/exp_P6/pairs/dir_wh_planar.pkl..."
wait_for_file data/exp_P6/pairs/dir_wh_planar.pkl 600 || { log "DIR mine timeout"; }

# 3) Launch P6-DIR training (uses Exp-N base, 10 epochs).
if [[ -f data/exp_P6/pairs/dir_wh_planar.pkl ]]; then
  log "Launching P6-DIR training..."
  python experiments/train_m1a_v2_pairs.py \
    --pairs_pkl data/exp_P6/pairs/dir_wh_planar.pkl \
    --base_ckpt models/m1a_v2.ckpt \
    --out_dir data/exp_P6/models/P6-DIR \
    --epochs 10 --lr 3e-5 --batch_size 16 --seed 42 \
    --ckpt_interval 1000 \
    --progress_path data/exp_P6/rl_logs/P6-DIR_progress.json \
    > data/exp_P6/rl_logs/P6-DIR.stdout 2>&1
  log "P6-DIR training done."
else
  log "P6-DIR pkl missing — SKIP."
fi

# 4) Launch P6-POSENOISE training (uses P5-1 pkl + pose noise).
if [[ -f data/exp_P5/pairs_P5-1_within.pkl ]]; then
  log "Launching P6-POSENOISE training..."
  python experiments/train_m1a_v2_pairs_posenoise.py \
    --pairs_pkl data/exp_P5/pairs_P5-1_within.pkl \
    --base_ckpt models/m1a_v2.ckpt \
    --out_dir data/exp_P6/models/P6-POSENOISE \
    --epochs 10 --lr 3e-5 --batch_size 16 --seed 42 \
    --pose_noise_std 0.3 \
    --ckpt_interval 1000 \
    --progress_path data/exp_P6/rl_logs/P6-POSENOISE_progress.json \
    > data/exp_P6/rl_logs/P6-POSENOISE.stdout 2>&1
  log "P6-POSENOISE training done."
else
  log "P5-1 pkl missing at data/exp_P5/pairs_P5-1_within.pkl — SKIP."
fi

# 5) Wait for P6-DIVERSE mine.
log "Waiting for P6-DIVERSE mine data/exp_P6/pairs/diverse_wh.pkl..."
wait_for_file data/exp_P6/pairs/diverse_wh.pkl 3600 || { log "DIVERSE mine timeout"; }

# 6) Launch P6-DIVERSE training.
if [[ -f data/exp_P6/pairs/diverse_wh.pkl ]]; then
  log "Launching P6-DIVERSE training..."
  python experiments/train_m1a_v2_pairs.py \
    --pairs_pkl data/exp_P6/pairs/diverse_wh.pkl \
    --base_ckpt models/m1a_v2.ckpt \
    --out_dir data/exp_P6/models/P6-DIVERSE \
    --epochs 10 --lr 3e-5 --batch_size 16 --seed 42 \
    --ckpt_interval 1000 \
    --progress_path data/exp_P6/rl_logs/P6-DIVERSE_progress.json \
    > data/exp_P6/rl_logs/P6-DIVERSE.stdout 2>&1
  log "P6-DIVERSE training done."
else
  log "P6-DIVERSE pkl missing — SKIP."
fi

log "ALL P6 TRAINING DONE. Writing ALL_DONE.flag."
touch data/exp_P6/ALL_DONE.flag
