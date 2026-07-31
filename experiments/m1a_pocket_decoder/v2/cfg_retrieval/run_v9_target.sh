#!/usr/bin/env bash
# v9 per-target DPO pipeline. Usage: run_v9_target.sh <TARGET_NAME>
set -u -o pipefail
TARGET="$1"
export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval_v9_multitarget/$TARGET
export MODEL_DIR=$PROJECT_ROOT/models/cfg_retrieval_v9_multitarget/$TARGET
export STAGE_LOG=$OUT_DIR/run.log

mkdir -p "$OUT_DIR" "$MODEL_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

echo "[$(date -u +%H:%M:%S)] [$TARGET] Baseline sampling..." >> "$STAGE_LOG"
python "$CFG_DIR/sample_v9_target.py" \
    --target "$TARGET" \
    --out_csv "$OUT_DIR/samples_${TARGET}_baseline.csv" \
    --progress_path "$OUT_DIR/baseline_progress.json" \
    --n 200 --batch_size 32 \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] [$TARGET] Baseline 2D panel..." >> "$STAGE_LOG"
mkdir -p "$OUT_DIR/samples_baseline"
cp "$OUT_DIR/samples_${TARGET}_baseline.csv" "$OUT_DIR/samples_baseline/"
python "$CFG_DIR/covalent_metric_panel.py" \
    --samples_dir "$OUT_DIR/samples_baseline" \
    --out_csv "$OUT_DIR/covalent_metric_panel_${TARGET}_baseline.csv" \
    --vina_topn 0 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] [$TARGET] Baseline Vina cov..." >> "$STAGE_LOG"
python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_${TARGET}_baseline.csv" \
    --out_csv "$OUT_DIR/covalent_vina_cov_panel_${TARGET}_baseline.csv" \
    --progress_path "$OUT_DIR/baseline_vina_progress.json" \
    --work_dir "$OUT_DIR/vina_baseline_work" \
    --workers 4 --incremental_save_every 20 --resume \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] [$TARGET] Build DPO pairs..." >> "$STAGE_LOG"
python "$CFG_DIR/build_dpo_pairs_v9_target.py" \
    --target "$TARGET" \
    --panel_2d "$OUT_DIR/covalent_metric_panel_${TARGET}_baseline.csv" \
    --panel_vina "$OUT_DIR/covalent_vina_cov_panel_${TARGET}_baseline.csv" \
    --out_csv "$OUT_DIR/dpo_pairs_${TARGET}.csv" \
    --n_pairs 300 --winner_quantile 0.40 --loser_quantile 0.60 --min_gap 10.0 \
    >> "$STAGE_LOG" 2>&1

# Check pair count. If <20, skip DPO training/sampling for this target.
N_PAIRS=$(( $(wc -l < "$OUT_DIR/dpo_pairs_${TARGET}.csv") - 1 ))
if [ "$N_PAIRS" -lt 20 ]; then
    echo "[$(date -u +%H:%M:%S)] [$TARGET] SKIP DPO — only $N_PAIRS pairs" >> "$STAGE_LOG"
    exit 0
fi

DPO_CKPT=$MODEL_DIR/dpo_final.ckpt
if [ ! -s "$DPO_CKPT" ]; then
    echo "[$(date -u +%H:%M:%S)] [$TARGET] DPO training..." >> "$STAGE_LOG"
    python "$CFG_DIR/train_v6_dpo.py" \
        --pairs_csv "$OUT_DIR/dpo_pairs_${TARGET}.csv" \
        --out_dir "$MODEL_DIR" \
        --log_dir "$OUT_DIR" \
        --beta 0.1 --epochs 3 --batch_size 8 \
        --lr_new 5e-6 --lr_base 1e-6 --warmup_steps 20 --ckpt_interval 50 \
        >> "$STAGE_LOG" 2>&1
fi

echo "[$(date -u +%H:%M:%S)] [$TARGET] DPO sampling..." >> "$STAGE_LOG"
python "$CFG_DIR/sample_v9_target.py" \
    --target "$TARGET" \
    --dpo_ckpt "$DPO_CKPT" \
    --out_csv "$OUT_DIR/samples_${TARGET}_dpo.csv" \
    --progress_path "$OUT_DIR/dpo_progress.json" \
    --n 200 --batch_size 32 \
    >> "$STAGE_LOG" 2>&1

mkdir -p "$OUT_DIR/samples_dpo"
cp "$OUT_DIR/samples_${TARGET}_dpo.csv" "$OUT_DIR/samples_dpo/"
python "$CFG_DIR/covalent_metric_panel.py" \
    --samples_dir "$OUT_DIR/samples_dpo" \
    --out_csv "$OUT_DIR/covalent_metric_panel_${TARGET}_dpo.csv" \
    --vina_topn 0 >> "$STAGE_LOG" 2>&1

python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_${TARGET}_dpo.csv" \
    --out_csv "$OUT_DIR/covalent_vina_cov_panel_${TARGET}_dpo.csv" \
    --progress_path "$OUT_DIR/dpo_vina_progress.json" \
    --work_dir "$OUT_DIR/vina_dpo_work" \
    --workers 4 --incremental_save_every 20 --resume \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] [$TARGET] DONE" >> "$STAGE_LOG"
