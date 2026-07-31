#!/bin/bash
# thiq_rl_zap70 runner — FiLM REST server + RL
set -eo pipefail
LOG="$HOME/thiq_rl_zap70.log"
date -u +"[%FT%TZ] thiq_rl_zap70 START" | tee -a "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

pkill -f reinvent4_film_rest 2>/dev/null || true
sleep 2
nohup python /home/shaharh_quris_ai/edit-small-mol/experiments/reinvent4_film_rest_server.py \
  --host 127.0.0.1 --port 8088 > $HOME/film_rest_server.log 2>&1 &
SERVER_PID=$!
echo "FiLM server PID=$SERVER_PID" | tee -a "$LOG"
for i in $(seq 1 30); do
  curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1 && break
  sleep 2
done
date -u +"[%FT%TZ] FiLM ready" | tee -a "$LOG"

work=/home/shaharh_quris_ai/edit-small-mol/results/paper_evaluation/mol1_rl/thiq_rl_zap70
rm -rf "$work"; mkdir -p "$work"; cd "$work"
cp /home/shaharh_quris_ai/edit-small-mol/experiments/thiq_rl_tomls/thiq_rl_zap70.toml .
date -u +"[%FT%TZ] RL launching" | tee -a "$LOG"
timeout 5400 reinvent ./thiq_rl_zap70.toml -d cuda 2>&1 | tee -a "$LOG" || {
  date -u +"[%FT%TZ] RL FAILED/TIMED OUT" | tee -a "$LOG"
  kill $SERVER_PID 2>/dev/null
  exit 1
}
date -u +"[%FT%TZ] RL DONE" | tee -a "$LOG"
kill $SERVER_PID 2>/dev/null
