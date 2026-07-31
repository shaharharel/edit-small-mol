#!/bin/bash
# Wait for warhead_tokens-base THIQ RL to finish, then launch covalent_ft-base
# variants on the same VM. Run once per VM.
#
# Usage: bash queue_thiq_exp2_after.sh <vm_label>
#   vm_label = ai-gpu | ai-gpu2 | ai-gpu-a100
set -eo pipefail
LBL=${1:-unknown}
LOG="$HOME/queue_thiq_exp2.log"
date -u +"[%FT%TZ] [$LBL] queue start — waiting for current reinvent to finish" | tee -a "$LOG"

# Wait until no reinvent process running
while pgrep -f 'reinvent.*\.toml' >/dev/null; do
  sleep 30
done
date -u +"[%FT%TZ] [$LBL] no reinvent running — launching exp2 THIQ variant" | tee -a "$LOG"

# Decide which exp2 toml to run based on VM
case "$LBL" in
  ai-gpu2)        SCRIPT=run_thiq_rl_exp2_mol1only.sh ;;
  ai-gpu)         SCRIPT=run_thiq_rl_exp2_zap70.sh ;;
  ai-gpu-a100)    SCRIPT=run_thiq_rl_exp2_kinase.sh ;;
  *) echo "unknown vm $LBL"; exit 1 ;;
esac

bash $HOME/edit-small-mol/experiments/thiq_rl_tomls/$SCRIPT
date -u +"[%FT%TZ] [$LBL] exp2 THIQ DONE" | tee -a "$LOG"
