#!/bin/bash
# EXP6 Phase 2b orchestrator: KRAS_A first, then B for all 3 targets.
# Designed to run in ONE tmux session on a100-b. NO restart loops.
# Logs to ~/exp6_phase2b_orch.log.
set -uo pipefail
export PATH=/home/shaharh_quris_ai/miniconda3/envs/quris/bin:$PATH
cd /home/shaharh_quris_ai/edit-small-mol

LOG=~/exp6_phase2b_orch.log
echo "[$(date -u +%FT%TZ)] === Phase 2b orchestrator START ===" | tee -a "$LOG"

# Pre-flight: REST servers must be alive
for p in 8088 8089 8090; do
  code=$(curl -sf -o /dev/null -w "%{http_code}" http://127.0.0.1:$p/health)
  if [ "$code" != "200" ]; then
    echo "[$(date -u +%FT%TZ)] FATAL: REST $p down (http=$code)" | tee -a "$LOG"
    exit 2
  fi
  echo "[$(date -u +%FT%TZ)] REST $p OK" | tee -a "$LOG"
done

# Phase A: KRAS_A — 3 iters
echo "[$(date -u +%FT%TZ)] === PHASE A: KRAS_A iter1-3 ===" | tee -a "$LOG"
python experiments/exp6_rl_driver.py \
  --targets kras_g12c \
  --strategies A \
  --scorers film \
  --iters 1 2 3 \
  --time_budget_h 1.5 \
  --out_json /home/shaharh_quris_ai/edit-small-mol/data/exp6_retrospective/_rl/rl_results.json \
  2>&1 | tee -a "$LOG"

# Phase B: Strategy B for all 3 targets, 3 iters each
echo "[$(date -u +%FT%TZ)] === PHASE B: ALL_B iter1-3 ===" | tee -a "$LOG"
python experiments/exp6_rl_driver.py \
  --targets egfr_t790m btk kras_g12c \
  --strategies B \
  --scorers film \
  --iters 1 2 3 \
  --time_budget_h 3.5 \
  --out_json /home/shaharh_quris_ai/edit-small-mol/data/exp6_retrospective/_rl/rl_results.json \
  2>&1 | tee -a "$LOG"

echo "[$(date -u +%FT%TZ)] === Phase 2b orchestrator DONE ===" | tee -a "$LOG"
touch /home/shaharh_quris_ai/edit-small-mol/data/exp6_retrospective/_PHASE2B_DONE
