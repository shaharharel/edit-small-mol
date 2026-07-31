#!/usr/bin/env bash
# CFG + retrieval-prefix full pipeline for ai-gpu.
#
# Stages:
#   1. build_retrieval_prefix.py     — top-5 similar CovInDB pockets, LOO ZAP70
#   2. train_m1a_v2_cfg.py           — CFG fine-tune warm-started from m1a_v2.ckpt
#   3. sample_cfg_retrieval.py       — 8 cells × 200 samples
#   4. covalent_metric_panel.py      — per-sample scoring
#   5. build_cfg_retrieval_report.py — bootstrap CIs + markdown report
#
# All output goes to data/paper_pair_training/cfg_retrieval/.
# Progress heartbeat: progress.md updated by each stage; also written to
# progress.json for machine-readable monitoring.
set -u -o pipefail   # NOT set -e — one stage failing shouldn't drop the tmux

export PROJECT_ROOT=/home/shaharh_quris_ai/edit-small-mol
export CFG_DIR=$PROJECT_ROOT/experiments/m1a_pocket_decoder/v2/cfg_retrieval
export OUT_DIR=$PROJECT_ROOT/data/paper_pair_training/cfg_retrieval
export PROGRESS_MD=$OUT_DIR/progress.md
export STAGE_LOG=$OUT_DIR/run_all.log

mkdir -p "$OUT_DIR"

# Activate quris env.
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

t_start=$(date +%s)

heartbeat() {
    local stage="$1"; shift
    local msg="$*"
    local now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    local el=$(( $(date +%s) - t_start ))
    {
        echo "## Progress (updated $now)"
        echo ""
        echo "**Current stage**: $stage"
        echo "**Elapsed**: ${el}s ($((el/60)) min)"
        echo ""
        echo "$msg"
        echo ""
        echo "### Stage timeline"
        if [ -f "$OUT_DIR/.stage_timeline" ]; then
            cat "$OUT_DIR/.stage_timeline"
        else
            echo "(no completed stages yet)"
        fi
    } > "$PROGRESS_MD" || true
}

record_stage() {
    local stage="$1"; local status="$2"
    local now=$(date -u +"%Y-%m-%dT%H:%M:%SZ")
    echo "- $now  **$stage** — $status" >> "$OUT_DIR/.stage_timeline"
}

# ------------------------------------------------------------------
# Stage 1: retrieval prefix
# ------------------------------------------------------------------
heartbeat "1_retrieval_prefix" "Ranking top-5 CovInDB pockets by cosine to ZAP70..."
if [ ! -s "$OUT_DIR/retrieval_top5.json" ]; then
    record_stage "1_retrieval_prefix" "STARTED"
    python "$CFG_DIR/build_retrieval_prefix.py" \
        --k 5 --out "$OUT_DIR/retrieval_top5.json" \
        >> "$STAGE_LOG" 2>&1
    record_stage "1_retrieval_prefix" "DONE"
else
    record_stage "1_retrieval_prefix" "SKIP (already exists)"
fi

# ------------------------------------------------------------------
# Stage 2: CFG training (main compute)
# ------------------------------------------------------------------
FINAL_CKPT=$PROJECT_ROOT/models/cfg_retrieval/cfg_final.ckpt
if [ ! -s "$FINAL_CKPT" ]; then
    heartbeat "2_cfg_training" "Warm-start CFG training from m1a_v2.ckpt (3 epochs)..."
    record_stage "2_cfg_training" "STARTED"
    python "$CFG_DIR/train_m1a_v2_cfg.py" \
        --p_drop 0.15 \
        --batch_size 32 \
        --lr 5e-5 \
        --epochs 30 \
        --warmup_steps 500 \
        --ckpt_interval 500 \
        --val_frac 0.10 \
        --seed 0 \
        --resume \
        >> "$STAGE_LOG" 2>&1
    record_stage "2_cfg_training" "DONE"
else
    record_stage "2_cfg_training" "SKIP (final ckpt exists)"
fi

# ------------------------------------------------------------------
# Stage 3: sampling matrix (8 cells × 200)
# ------------------------------------------------------------------
heartbeat "3_sampling" "Sampling 8 cells × 200 mols each..."
# Check if all 8 cells present with 200+ rows.
need_sample=0
for s in 1 2 3 5; do
    for r in off on; do
        f=$OUT_DIR/samples_cfg_s${s}_retrieval_${r}.csv
        if [ ! -s "$f" ] || [ "$(wc -l < "$f")" -lt 201 ]; then
            need_sample=1
        fi
    done
done
if [ "$need_sample" -eq 1 ]; then
    record_stage "3_sampling" "STARTED"
    python "$CFG_DIR/sample_cfg_retrieval.py" \
        --n 200 --batch_size 32 --temperature 1.0 \
        --cfg_scales 1.0,2.0,3.0,5.0 \
        --retrieval_modes off,on \
        >> "$STAGE_LOG" 2>&1
    record_stage "3_sampling" "DONE"
else
    record_stage "3_sampling" "SKIP (all 8 cells present)"
fi

# ------------------------------------------------------------------
# Stage 4: covalent metric panel (CPU; ~30 min)
# ------------------------------------------------------------------
heartbeat "4_covalent_panel" "Computing per-sample covalent metrics..."
if [ ! -s "$OUT_DIR/covalent_metric_panel.csv" ]; then
    record_stage "4_covalent_panel" "STARTED"
    python "$CFG_DIR/covalent_metric_panel.py" \
        --samples_dir "$OUT_DIR" \
        --out_csv "$OUT_DIR/covalent_metric_panel.csv" \
        --vina_topn 0 \
        >> "$STAGE_LOG" 2>&1
    record_stage "4_covalent_panel" "DONE"
else
    record_stage "4_covalent_panel" "SKIP (panel exists)"
fi

# ------------------------------------------------------------------
# Stage 5: report
# ------------------------------------------------------------------
heartbeat "5_report" "Writing markdown report with bootstrap CIs..."
record_stage "5_report" "STARTED"
python "$CFG_DIR/build_cfg_retrieval_report.py" \
    --panel_csv "$OUT_DIR/covalent_metric_panel.csv" \
    --out_md "$OUT_DIR/cfg_retrieval_report.md" \
    --out_summary_csv "$OUT_DIR/cfg_retrieval_report_summary.csv" \
    >> "$STAGE_LOG" 2>&1
record_stage "5_report" "DONE"

heartbeat "PIPELINE_DONE" "Full pipeline finished. See cfg_retrieval_report.md"

# Emit DONE marker for coordinator.
{
    echo "CFG + retrieval-prefix pipeline complete."
    echo "Report: $OUT_DIR/cfg_retrieval_report.md"
    echo "Panel:  $OUT_DIR/covalent_metric_panel.csv"
    echo "Summary: $OUT_DIR/cfg_retrieval_report_summary.csv"
    echo ""
    echo "See progress.md for stage timeline."
} > "$PROJECT_ROOT/data/agent_coord/from_cfg_retrieval_DONE.txt"

echo "PIPELINE DONE @ $(date -u +%Y-%m-%dT%H:%M:%SZ)"
