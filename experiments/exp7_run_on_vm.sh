#!/bin/bash
# EXP7 LO eval — runs ON a100-b. Boots 5 REST servers (one per target) then
# drives through the queue.csv using exp7_rl_driver.py.
#
# Usage (on a100-b):
#   cd ~/edit-small-mol && bash experiments/exp7_run_on_vm.sh
#
# Expects:
#   data/exp7_lo_benchmark/_phase1/{egfr_t790m,btk,jak3,her2,fgfr}/{filmdelta.pt,anchor_pool_strategy_b.csv,warhead_smarts.json}
#   data/exp7_lo_benchmark/_rl/queue.csv
#   models/reinvent4_mol2mol_covalent_ft.prior

set -euo pipefail

PROJ=/home/shaharh_quris_ai/edit-small-mol
cd "$PROJ"
export PATH=/home/shaharh_quris_ai/miniconda3/envs/quris/bin:$PATH

# Kill any leftover servers/drivers
pkill -9 -f exp6_rest_server 2>/dev/null || true
pkill -9 -f exp7_rl_driver 2>/dev/null || true
pkill -9 -f reinvent 2>/dev/null || true
sleep 2

REST=$PROJ/experiments/exp6_rest_server.py   # reuse — same contract
LOG_DIR=$PROJ/data/exp7_lo_benchmark/_logs
mkdir -p $LOG_DIR

# Synthesize anchor_pool_strategy_b.csv per target dir for REST server (it expects this exact filename)
# Strategy: use the FIRST pair's B pool csv per target — REST scoring uses these as anchors for FiLM ensemble
# (For per-pair-tailored REST scoring we'd need per-pair servers; this is good-enough since FiLM ensemble averages over the pool)
for TK in egfr_t790m btk jak3 her2 fgfr; do
    TD=$PROJ/data/exp7_lo_benchmark/_phase1/$TK
    FIRST_POOL=$(ls $TD/anchors/*_b_pool.csv 2>/dev/null | head -1)
    if [ -n "$FIRST_POOL" ]; then
        cp "$FIRST_POOL" "$TD/anchor_pool_strategy_b.csv"
        echo "[$TK] Using $(basename $FIRST_POOL) as REST scoring anchor pool"
    fi
done

# Launch REST servers (ports 8088-8092)
declare -A PORTS=([egfr_t790m]=8088 [btk]=8089 [jak3]=8090 [her2]=8091 [fgfr]=8092)
for TK in "${!PORTS[@]}"; do
    PORT=${PORTS[$TK]}
    TD=$PROJ/data/exp7_lo_benchmark/_phase1/$TK
    setsid nohup env CUDA_VISIBLE_DEVICES= /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python \
        $REST --target_dir $TD --src_root $PROJ --scorer film --port $PORT \
        > $LOG_DIR/rest_$TK.log 2>&1 < /dev/null &
    echo "[$TK] REST server launching on port $PORT (PID $!)"
done

# Wait for servers
sleep 12
for TK in "${!PORTS[@]}"; do
    PORT=${PORTS[$TK]}
    if curl -sf http://127.0.0.1:$PORT/health >/dev/null; then
        echo "[$TK] REST server READY at :$PORT"
    else
        echo "[$TK] REST server NOT responding at :$PORT — check $LOG_DIR/rest_$TK.log"
    fi
done

# Launch driver
DRIVER=$PROJ/experiments/exp7_rl_driver.py
QUEUE=$PROJ/data/exp7_lo_benchmark/_rl/queue.csv
RESULTS=$PROJ/data/exp7_lo_benchmark/_rl/driver_results.csv

# Budget: 7 hours (25200 sec). Driver stops cleanly when budget exceeded.
nohup /home/shaharh_quris_ai/miniconda3/envs/quris/bin/python $DRIVER \
    --pairs_csv $QUEUE --out_results $RESULTS --budget_sec 25200 \
    > $LOG_DIR/driver.log 2>&1 &
DRIVER_PID=$!
echo "Driver launched PID $DRIVER_PID, log: $LOG_DIR/driver.log"
echo "$DRIVER_PID" > $LOG_DIR/driver.pid

echo ""
echo "Tailing driver log (Ctrl-C OK; driver keeps running):"
sleep 5
tail -f $LOG_DIR/driver.log &
TAIL_PID=$!

# Print summary every 60s for first 5 min so SSH session has signal
sleep 300
kill $TAIL_PID 2>/dev/null || true
echo ""
echo "=== After 5 min ==="
tail -20 $LOG_DIR/driver.log
echo ""
echo "Done with bootstrap; driver continues in background. PID $DRIVER_PID"
