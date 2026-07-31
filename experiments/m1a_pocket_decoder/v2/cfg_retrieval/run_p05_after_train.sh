#!/usr/bin/env bash
# Post-training pipeline for the p_drop=0.05 CFG variant (insurance run).
# Sequenced: sample → 2D panel → xTB Fukui → Vina BD → report_p05
# Outputs go to data/paper_pair_training/cfg_retrieval_p05/ (parallel dir).

set -u -o pipefail

export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval_p05
export STAGE_LOG=$OUT_DIR/run_p05.log

mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

FINAL_CKPT=$PROJECT_ROOT/models/cfg_retrieval_p05/cfg_final.ckpt
if [ ! -s "$FINAL_CKPT" ]; then
    echo "[$(date -u +%H:%M:%S)] ERROR: p05 checkpoint missing: $FINAL_CKPT" >> "$STAGE_LOG"
    exit 1
fi

# 1. Sample 8 cells x 200.
echo "[$(date -u +%H:%M:%S)] Sampling p05 8 cells..." >> "$STAGE_LOG"
python "$CFG_DIR/sample_cfg_retrieval.py" \
    --ckpt "$FINAL_CKPT" \
    --retrieval_json "$OUT_DIR/retrieval_top5.json" \
    --out_dir "$OUT_DIR" \
    --n 200 --batch_size 32 --temperature 1.0 \
    --cfg_scales 1.0,1.5,2.0,3.0 --retrieval_modes off,on \
    --tag p05 \
    >> "$STAGE_LOG" 2>&1

# 2. 2D covalent panel.
echo "[$(date -u +%H:%M:%S)] 2D panel..." >> "$STAGE_LOG"
python "$CFG_DIR/covalent_metric_panel.py" \
    --samples_dir "$OUT_DIR" \
    --out_csv "$OUT_DIR/covalent_metric_panel_p05.csv" \
    --vina_topn 0 >> "$STAGE_LOG" 2>&1

# 3. xTB Fukui batch.
echo "[$(date -u +%H:%M:%S)] xTB Fukui batch..." >> "$STAGE_LOG"
python "$CFG_DIR/xtb_fukui_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_p05.csv" \
    --out_csv "$OUT_DIR/covalent_xtb_panel_p05.csv" \
    --progress_path "$OUT_DIR/xtb_batch_progress.json" \
    --workers 4 \
    >> "$STAGE_LOG" 2>&1

# 4. Vina batch (uses workers=2 while ai-gpu p_drop=0.10 Vina may still be running).
echo "[$(date -u +%H:%M:%S)] Vina batch..." >> "$STAGE_LOG"
python "$CFG_DIR/vina_dock_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_p05.csv" \
    --out_csv "$OUT_DIR/covalent_vina_panel_p05.csv" \
    --progress_path "$OUT_DIR/vina_batch_progress.json" \
    --workers 2 \
    >> "$STAGE_LOG" 2>&1

# 5. Report v3 for p05.
echo "[$(date -u +%H:%M:%S)] Report..." >> "$STAGE_LOG"
python "$CFG_DIR/build_cfg_retrieval_report_v3.py" \
    --panel_2d_csv "$OUT_DIR/covalent_metric_panel_p05.csv" \
    --panel_xtb_csv "$OUT_DIR/covalent_xtb_panel_p05.csv" \
    --panel_vina_csv "$OUT_DIR/covalent_vina_panel_p05.csv" \
    --panel_cofold_csv "$OUT_DIR/covalent_cofold_panel_p05_MISSING.csv" \
    --out_md "$OUT_DIR/cfg_retrieval_report_p05.md" \
    --out_summary_csv "$OUT_DIR/cfg_retrieval_report_p05_summary.csv" \
    --baseline_cell "cfg_s1_retrieval_off_p05" \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] p05 pipeline DONE." >> "$STAGE_LOG"
