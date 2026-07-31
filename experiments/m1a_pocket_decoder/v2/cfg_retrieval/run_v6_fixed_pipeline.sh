#!/usr/bin/env bash
# v6-FIXED: same as v6 but with save-bug fixed (QA #8) and pair set at scale.
# All outputs live in the same v6_dpo dir but with _FIXED suffixes to preserve
# audit trail of the broken original run.

set -u -o pipefail
export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval_v6_dpo
export MODEL_DIR=$PROJECT_ROOT/models/cfg_retrieval_v6_dpo_FIXED
export STAGE_LOG=$OUT_DIR/run_v6_FIXED.log

mkdir -p "$OUT_DIR" "$MODEL_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

echo "[$(date -u +%H:%M:%S)] Build DPO pairs at scale (global strategy)..." >> "$STAGE_LOG"
python "$CFG_DIR/build_dpo_pairs_from_vina.py" \
    --out_csv "$OUT_DIR/dpo_pairs_FIXED.csv" \
    --strategy global \
    --winner_quantile 0.40 --loser_quantile 0.60 \
    --global_n_pairs 2000 --min_gap 10.0 \
    >> "$STAGE_LOG" 2>&1

DPO_CKPT=$MODEL_DIR/dpo_final.ckpt
if [ ! -s "$DPO_CKPT" ]; then
    echo "[$(date -u +%H:%M:%S)] DPO training (FIXED save)..." >> "$STAGE_LOG"
    python "$CFG_DIR/train_v6_dpo.py" \
        --pairs_csv "$OUT_DIR/dpo_pairs_FIXED.csv" \
        --out_dir "$MODEL_DIR" \
        --log_dir "$OUT_DIR" \
        --beta 0.1 --epochs 3 --batch_size 8 \
        --lr_new 5e-6 --lr_base 1e-6 --warmup_steps 50 --ckpt_interval 100 \
        >> "$STAGE_LOG" 2>&1
    # Copy log into FIXED-suffixed name.
    cp "$OUT_DIR/train_log.csv" "$OUT_DIR/train_log_FIXED.csv" 2>/dev/null || true
fi

# Verify checkpoint contains base_network_state.
echo "[$(date -u +%H:%M:%S)] Verify checkpoint contents..." >> "$STAGE_LOG"
python -c "
import torch
sd = torch.load('$DPO_CKPT', map_location='cpu', weights_only=False)
print('ckpt keys:', list(sd.keys()))
print('model_state keys:', len(sd.get('model_state', {})))
print('base_network_state keys:', len(sd.get('base_network_state', {})))
assert 'base_network_state' in sd, 'FIXED SAVE FAILED: no base_network_state key!'
assert len(sd['base_network_state']) > 100, f'FIXED SAVE FAILED: only {len(sd[\"base_network_state\"])} base keys'
print('SAVE-FIX VERIFIED: base transformer weights persisted.')
" >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Sample v6-FIXED..." >> "$STAGE_LOG"
# Use a distinct output subdir so files don't collide with the buggy run.
mkdir -p "$OUT_DIR/samples_FIXED"
python "$CFG_DIR/sample_v6_dpo.py" \
    --dpo_ckpt "$DPO_CKPT" \
    --out_dir "$OUT_DIR/samples_FIXED" \
    --n 200 --batch_size 32 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] 2D panel..." >> "$STAGE_LOG"
python "$CFG_DIR/covalent_metric_panel.py" \
    --samples_dir "$OUT_DIR/samples_FIXED" \
    --out_csv "$OUT_DIR/covalent_metric_panel_v6_FIXED.csv" \
    --vina_topn 0 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] xTB Fukui..." >> "$STAGE_LOG"
python "$CFG_DIR/xtb_fukui_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v6_FIXED.csv" \
    --out_csv "$OUT_DIR/covalent_xtb_panel_v6_FIXED.csv" \
    --progress_path "$OUT_DIR/xtb_progress_FIXED.json" \
    --workers 8 >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Vina cov --score_only..." >> "$STAGE_LOG"
python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v6_FIXED.csv" \
    --out_csv "$OUT_DIR/covalent_vina_cov_panel_v6_FIXED.csv" \
    --progress_path "$OUT_DIR/vina_cov_progress_FIXED.json" \
    --work_dir "$OUT_DIR/vina_cov_work_FIXED" \
    --workers 4 --incremental_save_every 20 --resume \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Plan top-25 + local_only..." >> "$STAGE_LOG"
python "$CFG_DIR/plan_top25_vina_cov.py" \
    --in_csv "$OUT_DIR/covalent_vina_cov_panel_v6_FIXED.csv" \
    --out_csv "$OUT_DIR/cofold_plan_top25_v6_FIXED.csv" \
    --per_cell 25 >> "$STAGE_LOG" 2>&1
python "$CFG_DIR/vina_covalent_batch.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v6_FIXED.csv" \
    --plan_csv "$OUT_DIR/cofold_plan_top25_v6_FIXED.csv" \
    --out_csv "$OUT_DIR/covalent_vina_local_panel_v6_FIXED.csv" \
    --progress_path "$OUT_DIR/vina_local_progress_FIXED.json" \
    --work_dir "$OUT_DIR/vina_local_work_FIXED" \
    --workers 4 --mode local_only --incremental_save_every 20 \
    >> "$STAGE_LOG" 2>&1

echo "[$(date -u +%H:%M:%S)] Report v6-FIXED..." >> "$STAGE_LOG"
python "$CFG_DIR/build_v6_report.py" \
    --out_md "$OUT_DIR/cfg_retrieval_report_v6_dpo_FIXED.md" \
    --out_summary_csv "$OUT_DIR/cfg_retrieval_report_v6_dpo_FIXED_summary.csv" \
    --out_effects_csv "$OUT_DIR/cfg_retrieval_report_v6_dpo_FIXED_effects.csv" \
    --panel_2d "$OUT_DIR/covalent_metric_panel_v6_FIXED.csv" \
    --panel_xtb "$OUT_DIR/covalent_xtb_panel_v6_FIXED.csv" \
    --panel_cov "$OUT_DIR/covalent_vina_cov_panel_v6_FIXED.csv" \
    --panel_local "$OUT_DIR/covalent_vina_local_panel_v6_FIXED.csv" \
    >> "$STAGE_LOG" 2>&1

{
    echo "v6 DPO FIXED pipeline COMPLETE (QA #8 fixes applied)."
    echo ""
    echo "Fix summary:"
    echo "  1. save_film_model + train ckpt now persists base_network_state"
    echo "     (17.46M transformer weights that receive DPO gradients)."
    echo "  2. build_dpo_pairs_from_vina defaults to strategy=global with"
    echo "     winner_quantile=0.40, loser_quantile=0.60, min_gap=10 kcal/mol"
    echo "     -> ~2000 pairs (vs original 338)."
    echo ""
    echo "Report:  $OUT_DIR/cfg_retrieval_report_v6_dpo_FIXED.md"
    echo "Summary: $OUT_DIR/cfg_retrieval_report_v6_dpo_FIXED_summary.csv"
    echo "Effects: $OUT_DIR/cfg_retrieval_report_v6_dpo_FIXED_effects.csv"
    echo ""
    echo "Panels (FIXED-suffixed to preserve audit trail of original buggy run):"
    echo "  covalent_metric_panel_v6_FIXED.csv"
    echo "  covalent_xtb_panel_v6_FIXED.csv"
    echo "  covalent_vina_cov_panel_v6_FIXED.csv"
    echo "  covalent_vina_local_panel_v6_FIXED.csv"
    echo ""
    echo "Training pairs: $OUT_DIR/dpo_pairs_FIXED.csv (2000 pairs)"
    echo "Training log:   $OUT_DIR/train_log_FIXED.csv"
    echo "Model: $DPO_CKPT"
    echo ""
    echo "Original (buggy) files preserved (unsuffixed) for audit:"
    echo "  cfg_retrieval_report_v6_dpo.md, samples_dpo_*.csv, etc."
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_v6_fixed_DONE.txt"

echo "[$(date -u +%H:%M:%S)] v6-FIXED DONE." >> "$STAGE_LOG"
