#!/bin/bash
# EXP6 resume on a100-b after preemption / disconnect.
# Brings REST servers + driver back up via tmux. Driver auto-resumes from rl_results.json.

set -uo pipefail
PROJ="$HOME/edit-small-mol"

# Detect what's running
n_servers=$(pgrep -c -f exp6_rest_server || echo 0)
n_drivers=$(pgrep -c -f exp6_rl_driver || echo 0)
echo "[resume] current state: servers=$n_servers drivers=$n_drivers"

if [ "$n_servers" -lt 6 ]; then
  echo "[resume] starting REST servers via tmux"
  tmux kill-session -t exp6_servers 2>/dev/null || true
  tmux new-session -d -s exp6_servers "bash $PROJ/experiments/exp6_boot_servers.sh > $HOME/exp6_boot.log 2>&1; sleep 1000000"
  sleep 10
  pgrep -c -f exp6_rest_server
fi

if [ "$n_drivers" -lt 1 ]; then
  echo "[resume] starting driver via tmux"
  tmux kill-session -t exp6_rl 2>/dev/null || true
  scorers="${SCORERS:-film}"
  budget="${BUDGET_H:-4.5}"
  log="${LOG:-$HOME/exp6_rl_${scorers// /_}.log}"
  tmux new-session -d -s exp6_rl "source $HOME/miniconda3/etc/profile.d/conda.sh && conda activate quris && cd $PROJ && python -u experiments/exp6_rl_driver.py --scorers $scorers --time_budget_h $budget 2>&1 | tee $log; echo FINISHED_AT_\$(date +%s) >> $log; sleep 600"
  sleep 4
fi

echo "[resume] tmux:"
tmux ls 2>/dev/null
echo "[resume] procs:"
pgrep -af exp6 | head -10
