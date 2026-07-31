#!/usr/bin/env bash
# v4 FiLM: sample -> 2D panel -> xTB -> cov-Vina -> top-25 local_only -> report
# Assumes film_final.ckpt already exists in models/cfg_retrieval_v4_film/.

set -u -o pipefail
export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval_v4_film
export STAGE_LOG=$OUT_DIR/run_v4.log

mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

FINAL_CKPT=$PROJECT_ROOT/models/cfg_retrieval_v4_film/film_final.ckpt
if [ ! -s "$FINAL_CKPT" ]; then
    echo "[$(date -u +%H:%M:%S)] ERROR: no ckpt at $FINAL_CKPT" >> "$STAGE_LOG"
    exit 1
fi

# 1. Sampling (4 cells × 200).
echo "[$(date -u +%H:%M:%S)] Sampling..." >> "$STAGE_LOG"
python "$CFG_DIR/sample_film.py" \
    --ckpt "$FINAL_CKPT" \
    --out_dir "$OUT_DIR" \
    --n 200 --batch_size 32 --temperature 1.0 \
    >> "$STAGE_LOG" 2>&1

# 2. 2D panel.
echo "[$(date -u +%H:%M:%S)] 2D panel..." >> "$STAGE_LOG"
python "$CFG_DIR/covalent_metric_panel.py" \
    --samples_dir "$OUT_DIR" \
    --out_csv "$OUT_DIR/covalent_metric_panel_v4.csv" \
    --vina_topn 0 >> "$STAGE_LOG" 2>&1

# 3. xTB Fukui.
echo "[$(date -u +%H:%M:%S)] xTB batch..." >> "$STAGE_LOG"
python "$CFG_DIR/xtb_fukui_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v4.csv" \
    --out_csv "$OUT_DIR/covalent_xtb_panel_v4.csv" \
    --progress_path "$OUT_DIR/xtb_progress.json" \
    --workers 8 >> "$STAGE_LOG" 2>&1

# 4. Vina cov (--score_only).
echo "[$(date -u +%H:%M:%S)] Vina cov batch (--score_only)..." >> "$STAGE_LOG"
python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v4.csv" \
    --out_csv "$OUT_DIR/covalent_vina_cov_panel_v4.csv" \
    --progress_path "$OUT_DIR/vina_cov_progress.json" \
    --work_dir "$OUT_DIR/vina_cov_work" \
    --workers 4 --incremental_save_every 20 --resume \
    >> "$STAGE_LOG" 2>&1

# 5. Plan top-25/cell and Vina --local_only.
echo "[$(date -u +%H:%M:%S)] Planning top-25 + local_only..." >> "$STAGE_LOG"
python "$CFG_DIR/plan_top25_vina_cov.py" \
    --in_csv "$OUT_DIR/covalent_vina_cov_panel_v4.csv" \
    --out_csv "$OUT_DIR/cofold_plan_top25_v4.csv" \
    --per_cell 25 >> "$STAGE_LOG" 2>&1

python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v4.csv" \
    --plan_csv "$OUT_DIR/cofold_plan_top25_v4.csv" \
    --out_csv "$OUT_DIR/covalent_vina_local_panel_v4.csv" \
    --progress_path "$OUT_DIR/vina_local_progress.json" \
    --work_dir "$OUT_DIR/vina_local_work" \
    --workers 4 --mode local_only --incremental_save_every 20 \
    >> "$STAGE_LOG" 2>&1

# 6. Report.
echo "[$(date -u +%H:%M:%S)] Report..." >> "$STAGE_LOG"
python "$CFG_DIR/build_cfg_retrieval_v4_report.py" \
    >> "$STAGE_LOG" 2>&1

# 7. DONE marker.
{
    echo "v4 FiLM pipeline COMPLETE."
    echo ""
    echo "Report: $OUT_DIR/cfg_retrieval_report_v4_film.md"
    echo "Summary: $OUT_DIR/cfg_retrieval_report_v4_film_summary.csv"
    echo "Effects: $OUT_DIR/cfg_retrieval_report_v4_film_effects.csv"
    echo ""
    echo "Panels:"
    echo "  covalent_metric_panel_v4.csv"
    echo "  covalent_xtb_panel_v4.csv"
    echo "  covalent_vina_cov_panel_v4.csv"
    echo "  covalent_vina_local_panel_v4.csv"
    echo ""
    echo "Training log: $OUT_DIR/train_log.csv"
    echo "Model checkpoint: $FINAL_CKPT"
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_v4_DONE.txt"

echo "[$(date -u +%H:%M:%S)] v4 DONE." >> "$STAGE_LOG"
