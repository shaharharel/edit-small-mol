#!/bin/bash
# EXP7-v2 LO eval — runs ON a100-b. Boots 9 REST servers (one per v2 target) then
# drives through queue_covft.csv + queue_mol2mol.csv using two drivers in sequence.
#
# Usage (on a100-b):
#   bash ~/edit-small-mol/experiments/exp7v2_run_on_vm.sh
#
# Expects:
#   data/exp7_v2_benchmark/_phase1/{SOS1,KRAS_G12D,...}/{filmdelta.pt,anchor_pool_strategy_b.csv,warhead_smarts.json}
#   data/exp7_v2_benchmark/_rl/queue_covft.csv
#   data/exp7_v2_benchmark/_rl/queue_mol2mol.csv
#   models/reinvent4_mol2mol_covalent_ft.prior
#   models/reinvent4_mol2mol_prior.prior

set -euo pipefail

PROJ=/home/shaharh_quris_ai/edit-small-mol
cd "$PROJ"
export PATH=/home/shaharh_quris_ai/miniconda3/envs/quris/bin:$PATH

# Kill any leftover REST servers and drivers from previous runs
# IMPORTANT: do NOT use 'exp7v2' as a pkill pattern — our own script name contains it!
pkill -9 -f exp6_rest_server 2>/dev/null || true
pkill -9 -f exp7_rl_driver 2>/dev/null || true
pkill -9 -f exp7v2_rl_driver 2>/dev/null || true
pkill -9 -f exp7v2_mol2mol_driver 2>/dev/null || true
pkill -9 -f reinvent 2>/dev/null || true
sleep 2

REST=$PROJ/experiments/exp6_rest_server.py
LOG_DIR=$PROJ/data/exp7_v2_benchmark/_logs
mkdir -p $LOG_DIR

declare -A PORTS=(
    [SOS1]=8101 [KRAS_G12D]=8102 [KRAS_G12C]=8103
    [CDK7]=8104 [BCL2]=8105 [FGFR1]=8106
    [EGFR_T790M]=8107 [BTK]=8108 [BTK_Cys481]=8109
)

# Launch REST servers
for TK in "${!PORTS[@]}"; do
    PORT=${PORTS[$TK]}
    TD=$PROJ/data/exp7_v2_benchmark/_phase1/$TK
    if [ ! -f "$TD/filmdelta.pt" ]; then
        echo "[$TK] SKIP — no filmdelta.pt"
        continue
    fi
    setsid nohup env CUDA_VISIBLE_DEVICES= /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python \
        $REST --target_dir $TD --src_root $PROJ --scorer film --port $PORT \
        > $LOG_DIR/rest_$TK.log 2>&1 < /dev/null &
    echo "[$TK] REST server launching on port $PORT (PID $!)"
done

echo "Waiting 15s for REST servers..."
sleep 15

for TK in "${!PORTS[@]}"; do
    PORT=${PORTS[$TK]}
    if curl -sf http://127.0.0.1:$PORT/health >/dev/null; then
        echo "[$TK] REST READY at :$PORT"
    else
        echo "[$TK] REST NOT responding at :$PORT — check $LOG_DIR/rest_$TK.log"
    fi
done

# Launch covFT driver in background (108 cells: 54 RL + 54 baseline)
COVFT_DRIVER=$PROJ/experiments/exp7v2_rl_driver.py
COVFT_QUEUE=$PROJ/data/exp7_v2_benchmark/_rl/queue_covft.csv
COVFT_RESULTS=$PROJ/data/exp7_v2_benchmark/_rl/driver_covft_results.csv

# Launch mol2mol driver in parallel (NO — serialise to avoid GPU contention)
MOL_DRIVER=$PROJ/experiments/exp7v2_mol2mol_driver.py
MOL_QUEUE=$PROJ/data/exp7_v2_benchmark/_rl/queue_mol2mol.csv
MOL_RESULTS=$PROJ/data/exp7_v2_benchmark/_rl/driver_mol2mol_results.csv

# Run covFT and mol2mol drivers in PARALLEL — single A100 has 40GB,
# each reinvent process uses ~2GB so both fit comfortably.
nohup bash -c "
  /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python $COVFT_DRIVER \
    --pairs_csv $COVFT_QUEUE --out_results $COVFT_RESULTS --budget_sec 28800 \
    > $LOG_DIR/driver_covft.log 2>&1 &
  COVFT_PID=\$!
  echo \"covFT driver PID \$COVFT_PID\"
  sleep 30  # let covFT start first
  /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python $MOL_DRIVER \
    --pairs_csv $MOL_QUEUE --out_results $MOL_RESULTS --budget_sec 28800 \
    > $LOG_DIR/driver_mol2mol.log 2>&1 &
  MOL_PID=\$!
  echo \"mol2mol driver PID \$MOL_PID\"
  wait \$COVFT_PID
  echo '=== covFT driver done ==='
  wait \$MOL_PID
  echo '=== mol2mol driver done ==='
  echo '=== ALL DRIVERS DONE ==='
" > $LOG_DIR/driver_master.log 2>&1 &
DRIVER_PID=$!
echo "Master driver launched PID $DRIVER_PID — log: $LOG_DIR/driver_master.log"
echo "$DRIVER_PID" > $LOG_DIR/driver.pid

sleep 5
echo "=== First minute of covFT log ==="
sleep 30
tail -20 $LOG_DIR/driver_covft.log 2>&1 || true
echo "Driver continues in background. PID $DRIVER_PID"
