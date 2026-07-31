#!/usr/bin/env bash
# CFG + retrieval-prefix v2 pipeline: real Boltz cofolds + real xTB + real BD.
#
# Stages:
#  0. rename existing (smoke) report -> cfg_retrieval_report_v1_smoke.md
#  1. retrieval_top5.json  (reuse if present)
#  2. train_m1a_v2_cfg_unfrozen.py  --p_drop 0.10 --epochs 15 (v2 ckpt)
#  3. sample_cfg_retrieval.py  s ∈ {1.0, 1.5, 2.0, 3.0} × {off, on} × N=200
#  4. covalent_metric_panel.py (2D per-sample metrics, all 1600)
#  5. batch_boltz_cofold.py  top 50 per cell (400 cofolds)
#  6. cofold_metric_panel.py (real BD + real xTB + Vina) on those 400
#  7. build_cfg_retrieval_report_v2.py -> cfg_retrieval_report_v2.md

set -u -o pipefail

export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval
export PROGRESS_MD=$OUT_DIR/progress.md
export STAGE_LOG=$OUT_DIR/run_all_v2.log

mkdir -p "$OUT_DIR"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

t_start=$(date +%s)

heartbeat() {
    local stage="$1"; shift
    local msg="$*"
    local now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local el=$(( $(date +%s) - t_start ))
    {
        echo "## CFG+Retrieval v2 Pipeline (updated $now)"
        echo ""
        echo "**Current stage**: $stage"
        echo "**Elapsed**: ${el}s ($((el/60)) min)"
        echo ""
        echo "$msg"
        echo ""
        echo "### Stage timeline"
        if [ -f "$OUT_DIR/.stage_timeline_v2" ]; then
            cat "$OUT_DIR/.stage_timeline_v2"
        else
            echo "(no completed stages yet)"
        fi
    } > "$PROGRESS_MD" || true
}

record_stage() {
    local stage="$1"; local status="$2"
    local now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "- $now  **$stage** — $status" >> "$OUT_DIR/.stage_timeline_v2"
}

# 0. Rename smoke report if present.
if [ -f "$OUT_DIR/cfg_retrieval_report.md" ] && \
     [ ! -f "$OUT_DIR/cfg_retrieval_report_v1_smoke.md" ]; then
    mv "$OUT_DIR/cfg_retrieval_report.md" \
        "$OUT_DIR/cfg_retrieval_report_v1_smoke.md" || true
    mv "$OUT_DIR/cfg_retrieval_report_summary.csv" \
        "$OUT_DIR/cfg_retrieval_report_v1_smoke_summary.csv" 2>/dev/null || true
    mv "$OUT_DIR/covalent_metric_panel.csv" \
        "$OUT_DIR/covalent_metric_panel_v1_smoke.csv" 2>/dev/null || true
fi
# Also archive any v1 sample CSVs so we don't confuse the v2 pipeline.
mkdir -p "$OUT_DIR/v1_smoke_samples" 2>/dev/null || true
for f in "$OUT_DIR"/samples_cfg_s*_retrieval_*.csv; do
    if [ -f "$f" ] && [ ! -f "$OUT_DIR/v1_smoke_samples/$(basename "$f")" ]; then
        mv "$f" "$OUT_DIR/v1_smoke_samples/" 2>/dev/null || true
    fi
done

# 1. Retrieval prefix.
heartbeat "1_retrieval_prefix" "Reusing existing retrieval_top5.json if present..."
if [ ! -s "$OUT_DIR/retrieval_top5.json" ]; then
    record_stage "1_retrieval_prefix" "STARTED"
    python "$CFG_DIR/build_retrieval_prefix.py" --k 5 \
        --out "$OUT_DIR/retrieval_top5.json" >> "$STAGE_LOG" 2>&1
    record_stage "1_retrieval_prefix" "DONE"
else
    record_stage "1_retrieval_prefix" "SKIP (reuse existing)"
fi

# 2. CFG training v2 (p_drop=0.10, unfrozen base).
FINAL_CKPT=$PROJECT_ROOT/models/cfg_retrieval_v2/cfg_final.ckpt
if [ ! -s "$FINAL_CKPT" ]; then
    heartbeat "2_cfg_training_v2" "Unfrozen CFG training p=0.10, 15 epochs, LR_new=5e-5 LR_base=1e-5..."
    record_stage "2_cfg_training_v2" "STARTED"
    python "$CFG_DIR/train_m1a_v2_cfg_unfrozen.py" \
        --p_drop 0.10 --epochs 15 --warmup_steps 500 --ckpt_interval 500 \
        --lr_new 5e-5 --lr_base 1e-5 \
        --out_dir "$PROJECT_ROOT/models/cfg_retrieval_v2" \
        --log_csv_name train_log_v2.csv \
        --resume >> "$STAGE_LOG" 2>&1
    record_stage "2_cfg_training_v2" "DONE"
else
    record_stage "2_cfg_training_v2" "SKIP (ckpt exists)"
fi

# 3. Sample matrix (s ∈ {1.0, 1.5, 2.0, 3.0} × {off, on}).
heartbeat "3_sampling_v2" "Sampling 8 cells × N=200 mols..."
need=0
for s in 1 1p5 2 3; do
    for r in off on; do
        f=$OUT_DIR/samples_cfg_s${s}_retrieval_${r}_v2.csv
        if [ ! -s "$f" ] || [ "$(wc -l < "$f")" -lt 201 ]; then
            need=1
        fi
    done
done
if [ "$need" -eq 1 ]; then
    record_stage "3_sampling_v2" "STARTED"
    python "$CFG_DIR/sample_cfg_retrieval.py" \
        --ckpt "$FINAL_CKPT" \
        --n 200 --batch_size 32 --temperature 1.0 \
        --cfg_scales 1.0,1.5,2.0,3.0 \
        --retrieval_modes off,on \
        --tag v2 >> "$STAGE_LOG" 2>&1
    record_stage "3_sampling_v2" "DONE"
else
    record_stage "3_sampling_v2" "SKIP (all 8 present)"
fi

# 4. 2D covalent metric panel.
heartbeat "4_covalent_panel_2d" "Per-sample RDKit metrics for all 1600..."
if [ ! -s "$OUT_DIR/covalent_metric_panel_v2.csv" ]; then
    record_stage "4_covalent_panel_2d" "STARTED"
    python "$CFG_DIR/covalent_metric_panel.py" \
        --samples_dir "$OUT_DIR" \
        --out_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
        --vina_topn 0 >> "$STAGE_LOG" 2>&1
    record_stage "4_covalent_panel_2d" "DONE"
else
    record_stage "4_covalent_panel_2d" "SKIP"
fi

# 5. Batch Boltz cofolds (top 50 per cell × 8 = 400).
heartbeat "5_boltz_cofolds" "Cofolding top 50 valid+acryl per cell (400 total, ~17h)..."
record_stage "5_boltz_cofolds" "STARTED"
python "$CFG_DIR/batch_boltz_cofold.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
    --per_cell 50 --sampling_steps 200 --timeout_s 900 \
    >> "$STAGE_LOG" 2>&1
record_stage "5_boltz_cofolds" "DONE"

# 6. Cofold metric panel: real BD, real xTB, Vina.
heartbeat "6_cofold_panel" "Scoring cofolded samples with xTB + Vina + real BD gate..."
record_stage "6_cofold_panel" "STARTED"
python "$CFG_DIR/cofold_metric_panel.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
    --out_csv "$OUT_DIR/covalent_cofold_panel.csv" \
    >> "$STAGE_LOG" 2>&1
record_stage "6_cofold_panel" "DONE"

# 7. Report v2.
heartbeat "7_report_v2" "Writing v2 report..."
record_stage "7_report_v2" "STARTED"
python "$CFG_DIR/build_cfg_retrieval_report_v2.py" \
    --panel_2d_csv "$OUT_DIR/covalent_metric_panel_v2.csv" \
    --panel_cofold_csv "$OUT_DIR/covalent_cofold_panel.csv" \
    --out_md "$OUT_DIR/cfg_retrieval_report_v2.md" \
    --out_summary_csv "$OUT_DIR/cfg_retrieval_report_v2_summary.csv" \
    >> "$STAGE_LOG" 2>&1
record_stage "7_report_v2" "DONE"

heartbeat "PIPELINE_V2_DONE" "See cfg_retrieval_report_v2.md."
{
    echo "CFG + retrieval-prefix v2 pipeline complete."
    echo ""
    echo "Report v2: $OUT_DIR/cfg_retrieval_report_v2.md"
    echo "Panel 2D:  $OUT_DIR/covalent_metric_panel_v2.csv"
    echo "Cofold panel: $OUT_DIR/covalent_cofold_panel.csv"
    echo ""
    echo "Smoke (v1): $OUT_DIR/cfg_retrieval_report_v1_smoke.md"
    echo ""
    echo "See progress.md for full stage timeline."
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_DONE.txt"

echo "PIPELINE V2 DONE @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
