#!/bin/bash
# Exp 3 — reward ablation runner. Launches FiLM REST server (port 8088) on CPU,
# then runs 3 RL ablations serially (drop-film does NOT need FiLM server but we
# leave it up for simplicity; geometric_mean ignores missing components anyway).
set -eo pipefail
LOG=$HOME/exp3_ablations.log
date -u +"[%FT%TZ] EXP3 ABLATIONS START" | tee "$LOG"
source $HOME/miniconda3/etc/profile.d/conda.sh
conda activate quris
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTORCH_ALLOC_CONF=expandable_segments:True

# ── 1. FiLM REST server on CPU port 8088 ──────────────────────────────
pkill -f reinvent4_film_rest_server 2>/dev/null || true
sleep 2
CUDA_VISIBLE_DEVICES="" nohup python /home/shaharh_quris_ai/edit-small-mol/experiments/reinvent4_film_rest_server.py \
  --host 127.0.0.1 --port 8088 > $HOME/film_rest_server_exp3.log 2>&1 &
FILM_PID=$!
echo "FiLM PID=$FILM_PID" | tee -a "$LOG"

for i in $(seq 1 30); do
  ok=$(curl -sf http://127.0.0.1:8088/health >/dev/null 2>&1 && echo 1 || echo 0)
  [ "$ok" = "1" ] && break
  sleep 2
done
date -u +"[%FT%TZ] FiLM server ready=$ok" | tee -a "$LOG"
[ "$ok" != "1" ] && { echo "SERVER FAIL — aborting"; kill $FILM_PID 2>/dev/null; exit 1; }

# ── 2. Run 3 ablations serially ───────────────────────────────────────
for tag in drop_film drop_smarts drop_qed; do
  work=/home/shaharh_quris_ai/edit-small-mol/data/exp_reward_ablation/$tag
  rm -rf "$work"; mkdir -p "$work"; cd "$work"
  cp /home/shaharh_quris_ai/edit-small-mol/experiments/exp_reward_ablation/thiq_rl_ablation_${tag}.toml .
  date -u +"[%FT%TZ] ABLATION=$tag launching" | tee -a "$LOG"
  timeout 7200 reinvent ./thiq_rl_ablation_${tag}.toml -d cuda 2>&1 | tee -a "$LOG" || {
    date -u +"[%FT%TZ] ABLATION=$tag FAILED/TIMED OUT — continuing" | tee -a "$LOG"
  }
  date -u +"[%FT%TZ] ABLATION=$tag DONE" | tee -a "$LOG"
  ls -la "$work" | tee -a "$LOG"
done

# ── 3. Cleanup ────────────────────────────────────────────────────────
kill $FILM_PID 2>/dev/null || true
date -u +"[%FT%TZ] EXP3 ABLATIONS END" | tee -a "$LOG"
