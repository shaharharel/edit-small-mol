#!/bin/bash
# Eval the 6 L1 v5 sampled cohorts using eval_generation_quality.py.
# Run on V100 AFTER sampling completes (sequential; CPU work, ~30s/exp).
set -u
set -o pipefail

ROOT="$HOME/edit-small-mol"
cd "$ROOT" || exit 1

source ~/miniconda3/etc/profile.d/conda.sh
conda activate lingo3dmol

EXPS=(soft_anchor_geom multi_warhead lora_r8 hard_anchor_geom pose_aware_weight long_control)

for name in "${EXPS[@]}"; do
    sdf="data/lingo3dmol_L1_v5_${name}/samples_zap70_h2_anchor/samples.sdf"
    out_dir="data/lingo3dmol_L1_v5_${name}/eval_h2_anchor"
    if [ ! -f "$sdf" ]; then
        echo "[eval] $name SKIP (no sdf at $sdf)"
        continue
    fi
    echo "[eval] $name → $out_dir"
    python experiments/eval_generation_quality.py \
        --sdf "$sdf" --method "L1v5_${name}" --out-dir "$out_dir" 2>&1 | tail -20
done

echo
echo "=== ALL EVALS DONE ==="
ls -la data/lingo3dmol_L1_v5_*/eval_h2_anchor/*.json 2>&1 | head -20
