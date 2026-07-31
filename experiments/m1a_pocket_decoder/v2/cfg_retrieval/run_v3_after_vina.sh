#!/usr/bin/env bash
# v3 continuation: after Vina batch finishes, plan top-25 cofolds and run them.
# Then run the final v3 report.
#
# Reuses `experiments/m1a_pocket_decoder/v2/cfg_retrieval/` scripts:
#   - plan_top25_cofolds.py       -> cofold_plan_top25.csv
#   - batch_boltz_cofold.py       -> boltz_cofolds/{cell}/pred_s####/
#   - cofold_metric_panel.py      -> covalent_cofold_panel.csv
#   - build_cfg_retrieval_report_v3.py -> cfg_retrieval_report_v3.md

set -u -o pipefail

export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval
export STAGE_LOG=$OUT_DIR/run_v3_after_vina.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

# 1. Plan top-25 per cell.
echo "[$(date -u +%H:%M:%S)] Planning top-25 cofolds per cell..." >> "$STAGE_LOG"
python "$CFG_DIR/plan_top25_cofolds.py" --per_cell 25 >> "$STAGE_LOG" 2>&1

# 2. Boltz cofold batch (consume plan).
echo "[$(date -u +%H:%M:%S)] Boltz-2 cofolding top-25 per cell..." >> "$STAGE_LOG"
# Clear stale cofold dirs from the aborted v2 run (they were geometry-only,
# not Vina-ranked, so results are stale).  Keep the _smoke folder alone.
find "$OUT_DIR/boltz_cofolds" -mindepth 1 -maxdepth 1 -type d ! -name "_smoke" \
    -exec rm -rf {} + 2>/dev/null || true
python "$CFG_DIR/batch_boltz_cofold.py" \
    --plan_csv "$OUT_DIR/cofold_plan_top25.csv" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
    --sampling_steps 200 --timeout_s 900 \
    >> "$STAGE_LOG" 2>&1

# 3. Cofold metric panel (real BD + xTB + Vina rescore per cofold).
echo "[$(date -u +%H:%M:%S)] Cofold metric panel..." >> "$STAGE_LOG"
python "$CFG_DIR/cofold_metric_panel.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
    --out_csv "$OUT_DIR/covalent_cofold_panel.csv" \
    >> "$STAGE_LOG" 2>&1

# 4. Final v3 report.
echo "[$(date -u +%H:%M:%S)] v3 report..." >> "$STAGE_LOG"
python "$CFG_DIR/build_cfg_retrieval_report_v3.py" \
    --panel_2d_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
    --panel_xtb_csv "$OUT_DIR/covalent_xtb_panel.csv" \
    --panel_vina_csv "$OUT_DIR/covalent_vina_panel.csv" \
    --panel_cofold_csv "$OUT_DIR/covalent_cofold_panel.csv" \
    --out_md "$OUT_DIR/cfg_retrieval_report_v3.md" \
    --out_summary_csv "$OUT_DIR/cfg_retrieval_report_v3_summary.csv" \
    >> "$STAGE_LOG" 2>&1

# 5. Coord DONE marker.
{
    echo "CFG + retrieval-prefix v3 pipeline COMPLETE."
    echo ""
    echo "Report v3:  $OUT_DIR/cfg_retrieval_report_v3.md"
    echo "Summary:    $OUT_DIR/cfg_retrieval_report_v3_summary.csv"
    echo ""
    echo "Panels:"
    echo "  covalent_metric_panel_v2.csv (2D per-sample)"
    echo "  covalent_xtb_panel.csv        (REAL xTB Fukui f+ on 570 valid+acryl)"
    echo "  covalent_vina_panel.csv       (REAL Vina global dock on 570 valid+acryl)"
    echo "  covalent_cofold_panel.csv     (Boltz-2 cofold TOP-25 per cell verification)"
    echo ""
    echo "v1 smoke (frozen base + Gasteiger proxy) archived at:"
    echo "  cfg_retrieval_report_v1_smoke.md"
    echo ""
    echo "Training log: train_log.csv (10-row CSV, step 0..3945)"
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_DONE.txt"

echo "[$(date -u +%H:%M:%S)] v3 DONE." >> "$STAGE_LOG"
