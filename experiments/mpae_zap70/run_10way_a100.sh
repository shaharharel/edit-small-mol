#!/bin/bash
# Drive the 10-way ZAP70 factorial on A100.
# Sequentially trains 5 targets × 2 variants = 10 models, samples + evaluates each,
# then generates the verdict report.
#
# Total budget: ~5 hours

set -u
export PYTHONUNBUFFERED=1
ROOT=/home/shaharh_quris_ai/edit-small-mol
LOG_DIR=$ROOT/data/paper_pair_training/mpae_zap70/logs
mkdir -p "$LOG_DIR"
STAGE_LOG=$LOG_DIR/stage_progress.log
echo "[$(date -Is)] START 10-way factorial" > "$STAGE_LOG"

# 0) Build labeled_mols + esm_cache (skip if pre-built artifacts already present)
if [ -f $ROOT/data/paper_pair_training/mpae_zap70/labeled_mols_zap70.npz ] && \
   [ -f $ROOT/data/paper_pair_training/mpae_zap70/esm_cache_zap70.npz ] && \
   [ -f $ROOT/data/paper_pair_training/mpae_zap70/pose_stats_zap70.json ]; then
    echo "[$(date -Is)] STAGE 0: SKIP build (pre-built artifacts present)" >> "$STAGE_LOG"
else
    echo "[$(date -Is)] STAGE 0: build labeled data" >> "$STAGE_LOG"
    python3 $ROOT/experiments/mpae_zap70/build_labeled_zap70.py 2>&1 | tee "$LOG_DIR/00_build.log"
    if [ $? -ne 0 ]; then
        echo "[$(date -Is)] FATAL: build failed" >> "$STAGE_LOG"
        exit 1
    fi
fi

TARGETS=(d theta phi mpae composite)
VARIANTS=(3 4)

# 1) Train + eval each of the 10 models sequentially
for tgt in "${TARGETS[@]}"; do
    for v in "${VARIANTS[@]}"; do
        echo "[$(date -Is)] STAGE 1: train v${v}_${tgt}" >> "$STAGE_LOG"
        python3 $ROOT/experiments/mpae_zap70/train_zap70_target.py \
            --target "$tgt" --variant "$v" \
            2>&1 | tee "$LOG_DIR/train_v${v}_${tgt}.log"
        rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            echo "[$(date -Is)] WARN train v${v}_${tgt} rc=$rc" >> "$STAGE_LOG"
        fi

        echo "[$(date -Is)] STAGE 2: sample+eval v${v}_${tgt}" >> "$STAGE_LOG"
        python3 $ROOT/experiments/mpae_zap70/sample_and_proxy_eval_zap70.py \
            --target "$tgt" --variant "$v" \
            2>&1 | tee "$LOG_DIR/sample_v${v}_${tgt}.log"
        rc=${PIPESTATUS[0]}
        if [ $rc -ne 0 ]; then
            echo "[$(date -Is)] WARN sample v${v}_${tgt} rc=$rc" >> "$STAGE_LOG"
        fi
    done
done

# 2) Verdict report
echo "[$(date -Is)] STAGE 3: verdict report" >> "$STAGE_LOG"
python3 $ROOT/experiments/mpae_zap70/write_10way_verdict.py \
    2>&1 | tee "$LOG_DIR/verdict.log"

echo "[$(date -Is)] DONE 10-way factorial" >> "$STAGE_LOG"
