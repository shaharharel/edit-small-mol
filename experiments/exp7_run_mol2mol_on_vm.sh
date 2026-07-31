#!/bin/bash
# EXP7 LO eval — mol2mol arm. Runs ON a100-b.
# Boots 5 REST servers (one per target) if not already up, then drives mol2mol_queue.csv
# using exp7_mol2mol_driver.py.
#
# Usage (on a100-b):
#   cd ~/edit-small-mol && bash experiments/exp7_run_mol2mol_on_vm.sh
#
# Expects:
#   data/exp7_lo_benchmark/_phase1/{egfr_t790m,btk,jak3,her2,fgfr}/{filmdelta.pt,anchor_pool_strategy_b.csv,warhead_smarts.json}
#   data/exp7_lo_benchmark/_rl/mol2mol_queue.csv
#   models/reinvent4_mol2mol_prior.prior

set -euo pipefail

PROJ=/home/shaharh_quris_ai/edit-small-mol
cd "$PROJ"
export PATH=/home/shaharh_quris_ai/miniconda3/envs/quris/bin:$PATH

REST=$PROJ/experiments/exp6_rest_server.py
LOG_DIR=$PROJ/data/exp7_lo_benchmark/_logs
mkdir -p $LOG_DIR

# Ensure anchor_pool_strategy_b.csv exists per target
for TK in egfr_t790m btk jak3 her2 fgfr; do
    TD=$PROJ/data/exp7_lo_benchmark/_phase1/$TK
    if [ ! -f "$TD/anchor_pool_strategy_b.csv" ]; then
        FIRST_POOL=$(ls $TD/anchors/*_b_pool.csv 2>/dev/null | head -1)
        if [ -n "$FIRST_POOL" ]; then
            cp "$FIRST_POOL" "$TD/anchor_pool_strategy_b.csv"
            echo "[$TK] Copied $(basename $FIRST_POOL) -> anchor_pool_strategy_b.csv"
        fi
    fi
done

# Kill any leftover servers + drivers (specific patterns to avoid killing our own SSH)
pkill -9 -f exp6_rest_server 2>/dev/null || true
pkill -9 -f exp7_rl_driver 2>/dev/null || true
pkill -9 -f exp7_mol2mol_driver 2>/dev/null || true
pkill -9 -f 'reinvent ' 2>/dev/null || true
sleep 3

declare -A PORTS=([egfr_t790m]=8088 [btk]=8089 [jak3]=8090 [her2]=8091 [fgfr]=8092)
for TK in "${!PORTS[@]}"; do
    PORT=${PORTS[$TK]}
    TD=$PROJ/data/exp7_lo_benchmark/_phase1/$TK
    setsid nohup env CUDA_VISIBLE_DEVICES= /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python \
        $REST --target_dir $TD --src_root $PROJ --scorer film --port $PORT \
        > $LOG_DIR/rest_$TK.log 2>&1 < /dev/null &
    echo "[$TK] REST launching on :$PORT (PID $!)"
done

sleep 15
ALL_OK=true
for TK in "${!PORTS[@]}"; do
    PORT=${PORTS[$TK]}
    if curl -sf http://127.0.0.1:$PORT/health >/dev/null; then
        echo "[$TK] REST READY :$PORT"
    else
        echo "[$TK] REST NOT READY :$PORT — check $LOG_DIR/rest_$TK.log"
        ALL_OK=false
    fi
done

if [ "$ALL_OK" != "true" ]; then
    echo "WARNING: Not all REST servers READY. Sleeping 10s more..."
    sleep 10
fi

# Launch driver
DRIVER=$PROJ/experiments/exp7_mol2mol_driver.py
QUEUE=$PROJ/data/exp7_lo_benchmark/_rl/mol2mol_queue.csv
RESULTS=$PROJ/data/exp7_lo_benchmark/_rl/mol2mol_driver_results.csv

# Budget: 4.5h (16200 sec) — Phase 1 cap is 5h, leave 30 min for tail+pull
nohup /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python $DRIVER \
    --pairs_csv $QUEUE --out_results $RESULTS --budget_sec 16200 \
    > $LOG_DIR/mol2mol_driver.log 2>&1 &
DRIVER_PID=$!
echo "Driver launched PID $DRIVER_PID, log: $LOG_DIR/mol2mol_driver.log"
echo "$DRIVER_PID" > $LOG_DIR/mol2mol_driver.pid

echo ""
echo "First 20s log tail:"
sleep 20
tail -30 $LOG_DIR/mol2mol_driver.log
echo ""
echo "Done with bootstrap; driver continues in background. PID $DRIVER_PID"
