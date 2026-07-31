#!/usr/bin/env bash
# v6 DPO: build pairs -> train DPO -> sample 2x2 -> panels -> report.
set -u -o pipefail
export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval_v6_dpo
export MODEL_DIR=$PROJECT_ROOT/models/cfg_retrieval_v6_dpo
export STAGE_LOG=$OUT_DIR/run_v6.log

mkdir -p "$OUT_DIR" "$MODEL_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

echo "[$(date -u +%H:%M:%S)] Build DPO pairs from v3+v4 Vina data..." >> "$STAGE_LOG"
python "$CFG_DIR/build_dpo_pairs_from_vina.py" \
    --out_csv "$OUT_DIR/dpo_pairs.csv" \
    --per_cell_cap 25 >> "$STAGE_LOG" 2>&1

DPO_CKPT=$MODEL_DIR/dpo_final.ckpt
if [ ! -s "$DPO_CKPT" ]; then
    echo "[$(date -u +%H:%M:%S)] DPO training..." >> "$STAGE_LOG"
    python "$CFG_DIR/train_v6_dpo.py" \
        --pairs_csv "$OUT_DIR/dpo_pairs.csv" \
        --out_dir "$MODEL_DIR" \
        --log_dir "$OUT_DIR" \
        --beta 0.1 --epochs 3 --batch_size 8 \
        --lr_new 5e-6 --lr_base 1e-6 --warmup_steps 50 --ckpt_interval 100 \
        >> "$STAGE_LOG" 2>&1
fi

echo "[$(date -u +%H:%M:%S)] Sampling v6..." >> "$STAGE_LOG"
python "$CFG_DIR/sample_v6_dpo.py" \
    --dpo_ckpt "$DPO_CKPT" \
    --out_dir "$OUT_DIR" \
    --n 200 --batch_size 32 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] 2D panel..." >> "$STAGE_LOG"
python "$CFG_DIR/covalent_metric_panel.py" \
    --samples_dir "$OUT_DIR" \
    --out_csv "$OUT_DIR/covalent_metric_panel_v6.csv" \
    --vina_topn 0 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] xTB Fukui..." >> "$STAGE_LOG"
python "$CFG_DIR/xtb_fukui_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v6.csv" \
    --out_csv "$OUT_DIR/covalent_xtb_panel_v6.csv" \
    --progress_path "$OUT_DIR/xtb_progress.json" \
    --workers 8 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Vina cov --score_only..." >> "$STAGE_LOG"
python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v6.csv" \
    --out_csv "$OUT_DIR/covalent_vina_cov_panel_v6.csv" \
    --progress_path "$OUT_DIR/vina_cov_progress.json" \
    --work_dir "$OUT_DIR/vina_cov_work" \
    --workers 4 --incremental_save_every 20 --resume \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Plan top-25 + local_only..." >> "$STAGE_LOG"
python "$CFG_DIR/plan_top25_vina_cov.py" \
    --in_csv "$OUT_DIR/covalent_vina_cov_panel_v6.csv" \
    --out_csv "$OUT_DIR/cofold_plan_top25_v6.csv" \
    --per_cell 25 >> "$STAGE_LOG" 2>&1
python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v6.csv" \
    --plan_csv "$OUT_DIR/cofold_plan_top25_v6.csv" \
    --out_csv "$OUT_DIR/covalent_vina_local_panel_v6.csv" \
    --progress_path "$OUT_DIR/vina_local_progress.json" \
    --work_dir "$OUT_DIR/vina_local_work" \
    --workers 4 --mode local_only --incremental_save_every 20 \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Report..." >> "$STAGE_LOG"
python "$CFG_DIR/build_v6_report.py" >> "$STAGE_LOG" 2>&1

{
    echo "v6 DPO pipeline COMPLETE."
    echo ""
    echo "Report:  $OUT_DIR/cfg_retrieval_report_v6_dpo.md"
    echo "Summary: $OUT_DIR/cfg_retrieval_report_v6_dpo_summary.csv"
    echo "Effects: $OUT_DIR/cfg_retrieval_report_v6_dpo_effects.csv"
    echo ""
    echo "Panels:"
    echo "  covalent_metric_panel_v6.csv"
    echo "  covalent_xtb_panel_v6.csv"
    echo "  covalent_vina_cov_panel_v6.csv"
    echo "  covalent_vina_local_panel_v6.csv"
    echo ""
    echo "Training pairs: $OUT_DIR/dpo_pairs.csv"
    echo "Training log:   $OUT_DIR/train_log.csv"
    echo "Model: $DPO_CKPT"
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_v6_DONE.txt"

echo "[$(date -u +%H:%M:%S)] v6 DONE." >> "$STAGE_LOG"
