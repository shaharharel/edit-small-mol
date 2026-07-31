#!/bin/bash
# Single-cohort launcher with persistent FiLM REST server
# (starts server, runs RL, stops server)
set -eo pipefail
LOG="$HOME/rest_mol1_only_rest.log"
date -u +"[%FT%TZ] mol1RL_v5_seed_mol1_only_rest START (REST)" | tee -a "$LOG"

source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris

# Ensure waitress is installed
pip install waitress 2>&1 | tail -1

# Start FiLM server (background)
pkill -f reinvent4_film_rest_server 2>/dev/null || true
sleep 2
nohup python /home/shaharh_quris_ai/edit-small-mol/experiments/reinvent4_film_rest_server.py \
  --host 127.0.0.1 --port 8088 \
  > $HOME/film_rest_server.log 2>&1 &
SERVER_PID=$!
echo "FiLM server PID=$SERVER_PID" | tee -a "$LOG"

# Wait for server to come up (max 60s)
for i in $(seq 1 30); do
  if curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1; then
    echo "FiLM server READY after ${i}*2s" | tee -a "$LOG"
    break
  fi
  sleep 2
done
curl -s http://127.0.0.1:8088/health | tee -a "$LOG"
echo "" | tee -a "$LOG"

# Run RL
work=/home/shaharh_quris_ai/edit-small-mol/results/paper_evaluation/mol1_rl/mol1RL_v5_seed_mol1_only_rest
rm -rf "$work"; mkdir -p "$work"
cd "$work"
cp /home/shaharh_quris_ai/edit-small-mol/experiments/mol1_rl_tomls_rest/mol1RL_v5_seed_mol1_only_rest.toml .
date -u +"[%FT%TZ] RL launching" | tee -a "$LOG"
timeout 5400 reinvent ./mol1RL_v5_seed_mol1_only_rest.toml -d cuda 2>&1 | tee -a "$LOG" || {
  date -u +"[%FT%TZ] RL FAILED/TIMED OUT" | tee -a "$LOG"
  kill $SERVER_PID 2>/dev/null
  exit 1
}

date -u +"[%FT%TZ] RL DONE — final health: $(curl -s http://127.0.0.1:8088/health)" | tee -a "$LOG"
kill $SERVER_PID 2>/dev/null
