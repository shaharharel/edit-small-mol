#!/bin/bash
# thiq_rl_exp2_zap70_distilled runner
# Same as run_thiq_rl_exp2_zap70.sh but adds the distilled-validator REST server on port 8089
# and uses the 4-component reward TOML.
set -eo pipefail
LOG="$HOME/thiq_rl_exp2_zap70_distilled.log"
date -u +"[%FT%TZ] thiq_rl_exp2_zap70_distilled START" | tee "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── 1. FiLMDelta REST server on port 8088 ─────────────────────────────────
pkill -f reinvent4_film_rest_server 2>/dev/null || true
pkill -f distilled_validator_server 2>/dev/null || true
sleep 2
nohup python /home/shaharh_quris_ai/edit-small-mol/experiments/reinvent4_film_rest_server.py \
  --host 127.0.0.1 --port 8088 > $HOME/film_rest_server.log 2>&1 &
FILM_PID=$!
echo "FiLM PID=$FILM_PID" | tee -a "$LOG"

# ── 2. Distilled validator REST server on port 8089 ───────────────────────
nohup python /home/shaharh_quris_ai/edit-small-mol/experiments/distilled_validator_server.py \
  --host 127.0.0.1 --port 8089 > $HOME/distilled_server.log 2>&1 &
DIST_PID=$!
echo "Distilled PID=$DIST_PID" | tee -a "$LOG"

# Wait for both
for i in $(seq 1 30); do
  ok1=$(curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1 && echo 1 || echo 0)
  ok2=$(curl -sf http://127.0.0.1:8089/health >/dev/null 2>&1 && echo 1 || echo 0)
  [ "$ok1" = "1" ] && [ "$ok2" = "1" ] && break
  sleep 2
done
date -u +"[%FT%TZ] Servers ready (FiLM=$ok1, Distilled=$ok2)" | tee -a "$LOG"
[ "$ok1$ok2" != "11" ] && { echo "SERVER FAIL — aborting"; kill $FILM_PID $DIST_PID 2>/dev/null; exit 1; }

# ── 3. Launch RL with the 4-component reward TOML ─────────────────────────
work=/home/shaharh_quris_ai/edit-small-mol/results/paper_evaluation/mol1_rl/thiq_rl_exp2_zap70_distilled
rm -rf "$work"; mkdir -p "$work"; cd "$work"
cp /home/shaharh_quris_ai/edit-small-mol/experiments/thiq_rl_tomls/thiq_rl_exp2_zap70_distilled.toml .
date -u +"[%FT%TZ] RL launching" | tee -a "$LOG"
timeout 7200 reinvent ./thiq_rl_exp2_zap70_distilled.toml -d cuda 2>&1 | tee -a "$LOG" || {
  date -u +"[%FT%TZ] RL FAILED or TIMED OUT" | tee -a "$LOG"
  kill $FILM_PID $DIST_PID 2>/dev/null
  exit 1
}
date -u +"[%FT%TZ] RL DONE" | tee -a "$LOG"
ls -la "$work" | tee -a "$LOG"

# ── 4. Cleanup REST servers ───────────────────────────────────────────────
kill $FILM_PID $DIST_PID 2>/dev/null
date -u +"[%FT%TZ] thiq_rl_exp2_zap70_distilled END" | tee -a "$LOG"
