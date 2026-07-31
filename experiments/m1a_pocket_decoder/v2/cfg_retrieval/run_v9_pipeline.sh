#!/usr/bin/env bash
# v9 multi-target: run each target sequentially.
set -u -o pipefail
export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval_v9_multitarget
export STAGE_LOG=$OUT_DIR/run_v9.log

mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

for TARGET in BTK EGFR JAK3 KRAS_G12C; do
    echo "[$(date -u +%H:%M:%S)] === $TARGET ===" >> "$STAGE_LOG"
    bash "$CFG_DIR/run_v9_target.sh" "$TARGET" >> "$STAGE_LOG" 2>&1
done

echo "[$(date -u +%H:%M:%S)] Aggregate summary..." >> "$STAGE_LOG"
python "$CFG_DIR/build_v9_summary.py" >> "$STAGE_LOG" 2>&1

{
    echo "v9 MULTI-TARGET DPO COMPLETE (BTK, EGFR, JAK3, KRAS-G12C)."
    echo ""
    echo "Summary: $OUT_DIR/multi_target_summary.md"
    echo "CSV: $OUT_DIR/multi_target_summary.csv"
    echo ""
    echo "Per-target outputs in $OUT_DIR/{BTK,EGFR,JAK3,KRAS_G12C}/"
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_v9_DONE.txt"

echo "[$(date -u +%H:%M:%S)] v9 DONE." >> "$STAGE_LOG"
