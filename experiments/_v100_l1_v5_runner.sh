#!/bin/bash
# V100 orchestrator for L1 v5 (6 experiments × 20 epochs × 200 H2-anchor samples).
# Designed to be SPOT-preemption resilient: each train+sample is idempotent based
# on summary.json / samples.sdf state on disk. tmux re-attach to "l1v5" session.
#
# Usage on V100:
#     tmux new -d -s l1v5 'bash ~/edit-small-mol/experiments/_v100_l1_v5_runner.sh 2>&1 | tee ~/l1v5.log'
#     tmux attach -t l1v5     # to view
#
# Local-side checks: nightly poll ~/l1v5.log via gcloud ssh; final flag = /tmp/l1v5_all_done.

set -u
set -o pipefail

ROOT="$HOME/edit-small-mol"
cd "$ROOT" || { echo "[orch] cannot cd to $ROOT"; exit 1; }

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lingo3dmol

EPOCHS=20
BATCH=16
SEQ_LEN=80
LR=1e-5
PRETRAINED="external/Lingo3DMol/checkpoint/gen_mol.pkl"
POCKET="data/lingo3dmol_smoke/zap70_pocket_cys346.pdb"
PDB_DIR="data/covbinder/raw_covindb2/PDB"
CSV_ACR="data/covbinder/exp4_michael_acceptor.csv"
CSV_MW="data/covbinder/exp2_multi_warhead.csv"

# Each entry: name | csv | extra train flags | extra-tag (for log)
# Use a tab as the field separator so single-quoted args with spaces survive.
# Note: exp1 = soft (lower CE weight); exp4 = hard (default ce=0.1); exp7 = longer-train control (40 ep)
declare -a EXPS=(
"soft_anchor_geom|$CSV_ACR|--enable_anchor_geometry_loss --anchor_geom_weight 0.05 --anchor_geom_ce_w 0.05|exp1_soft|$EPOCHS"
"multi_warhead|$CSV_MW|--enable_anchor_geometry_loss --anchor_geom_weight 0.05 --anchor_geom_ce_w 0.1|exp2_multiwarhead|$EPOCHS"
"lora_r8|$CSV_ACR|--enable_anchor_geometry_loss --anchor_geom_weight 0.05 --anchor_geom_ce_w 0.1 --lora_rank 8 --lora_alpha 16|exp3_lora|$EPOCHS"
"hard_anchor_geom|$CSV_ACR|--enable_anchor_geometry_loss --anchor_geom_weight 0.05 --anchor_geom_ce_w 0.1|exp4_hard|$EPOCHS"
"pose_aware_weight|$CSV_ACR|--pose_aware_token_weight 5.0 --pose_aware_n_tokens 8 --enable_anchor_geometry_loss --anchor_geom_weight 0.05 --anchor_geom_ce_w 0.1|exp5_poseaware|$EPOCHS"
"long_control|$CSV_ACR|--enable_anchor_geometry_loss --anchor_geom_weight 0.05 --anchor_geom_ce_w 0.1|exp7_long40|40"
)

ts() { date '+%Y-%m-%d %H:%M:%S'; }

log_section () {
    echo
    echo "================================================================================"
    echo "[$(ts)] $*"
    echo "================================================================================"
}

run_train () {
    local name="$1"; local csv="$2"; local extra="$3"; local tag="$4"; local epochs="$5"
    local out="data/lingo3dmol_L1_v5_${name}"
    local summary="$out/summary.json"

    if [ -f "$summary" ]; then
        local done_ep
        done_ep=$(python -c "import json,sys; d=json.load(open('$summary')); print(int(d.get('epochs',0)))" 2>/dev/null || echo 0)
        if [ "$done_ep" -ge "$epochs" ]; then
            log_section "[$tag] TRAIN already done (epochs=$done_ep), skipping"
            return 0
        fi
    fi

    log_section "[$tag] TRAIN start (csv=$csv epochs=$epochs)"
    mkdir -p "$out"
    # shellcheck disable=SC2086
    python experiments/run_lingo3dmol_l1_train.py \
        --pretrained_ckpt "$PRETRAINED" \
        --out_dir "$out" \
        --epochs "$epochs" \
        --batch_size $BATCH \
        --seq_len $SEQ_LEN \
        --lr $LR \
        --device cuda \
        --complex_csv "$csv" \
        --pocket_pdb_dir "$PDB_DIR" \
        --pocket_radius 15.0 \
        $extra 2>&1 | tee "$out/train.log"
    local rc=${PIPESTATUS[0]}
    log_section "[$tag] TRAIN done (rc=$rc)"
    return $rc
}

run_sample () {
    local name="$1"; local tag="$2"
    local out="data/lingo3dmol_L1_v5_${name}"
    local ckpt="$out/ckpt_phase2_dev.pt"
    local sample_dir="$out/samples_zap70_h2_anchor"
    local sdf="$sample_dir/samples.sdf"

    if [ ! -f "$ckpt" ]; then
        log_section "[$tag] SAMPLE skipped — ckpt missing $ckpt"
        return 1
    fi

    if [ -f "$sdf" ]; then
        local n_mol
        n_mol=$(grep -c '\$\$\$\$' "$sdf" 2>/dev/null || echo 0)
        if [ "$n_mol" -ge 100 ]; then
            log_section "[$tag] SAMPLE already done (n_mol=$n_mol), skipping"
            return 0
        fi
    fi

    log_section "[$tag] SAMPLE start (target=200 mols H2 anchor)"
    mkdir -p "$sample_dir"
    python experiments/run_lingo3dmol_l2_extended_anchor.py \
        --pocket_pdb "$POCKET" \
        --anchor_id H2 \
        --caption_path "$ckpt" \
        --output "$sdf" \
        --gennums 200 \
        --min_acceptable 100 \
        --max_run_seconds 1800 \
        --tempture 1.0 2>&1 | tee "$sample_dir/sample.log"
    local rc=${PIPESTATUS[0]}
    log_section "[$tag] SAMPLE done (rc=$rc)"
    return $rc
}

# ============================================================================
# main loop
# ============================================================================
log_section "[orch] START — 6 exps, epochs default=$EPOCHS, batch=$BATCH"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader || true

for entry in "${EXPS[@]}"; do
    IFS='|' read -r name csv extra tag epochs <<< "$entry"
    log_section "[orch] === $tag ($name) ==="
    if run_train "$name" "$csv" "$extra" "$tag" "$epochs"; then
        run_sample "$name" "$tag" || true
    else
        log_section "[orch] TRAIN FAILED for $tag — continuing to next exp"
    fi
done

log_section "[orch] ALL DONE"
touch /tmp/l1v5_all_done
ls -la data/lingo3dmol_L1_v5_*/summary.json data/lingo3dmol_L1_v5_*/samples_zap70_h2_anchor/samples.sdf 2>&1 | head -40

echo "[$(ts)] orchestrator exit 0"
