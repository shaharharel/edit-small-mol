#!/usr/bin/env bash
# Boltz-2 cofold runner for 401 unique mols on ai-gpu-a100.
# Resumable: skips YAMLs whose output directory already exists with confidence JSON.
# N_PARALLEL=2 by default (A100 40GB has ~40GB VRAM; one cofold ~12-16GB).

set -uo pipefail

N_PARALLEL=${N_PARALLEL:-2}
RUN_DIR="$HOME/boltz_run"
YAML_DIR="$RUN_DIR/yamls"
OUT_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"
PROGRESS="$RUN_DIR/progress.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Activate conda env where boltz is installed
source "$HOME/.bashrc" >/dev/null 2>&1 || true
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null; then
    eval "$(conda shell.bash hook)"
fi

# Try multiple env names
for env in boltz boltz2 quris; do
    if conda env list 2>/dev/null | grep -q "^${env} "; then
        conda activate "$env"
        echo "[runner] activated conda env: $env"
        break
    fi
done

if ! command -v boltz >/dev/null; then
    echo "[runner] ERROR: boltz not found in PATH"
    exit 1
fi

echo "[runner] boltz: $(boltz --version 2>&1 | head -1)"
echo "[runner] N_PARALLEL=$N_PARALLEL"
echo "[runner] YAML_DIR=$YAML_DIR"
echo "[runner] OUT_DIR=$OUT_DIR"

# Build queue: skip done
QUEUE_FILE="$RUN_DIR/queue.txt"
: > "$QUEUE_FILE"
total=0
done_=0
for y in "$YAML_DIR"/*.yaml; do
    uid=$(basename "$y" .yaml)
    total=$((total + 1))
    # Boltz writes results to <out_dir>/boltz_results_<uid>/predictions/<uid>/confidence_<uid>_model_0.json
    cjson="$OUT_DIR/boltz_results_${uid}/predictions/${uid}/confidence_${uid}_model_0.json"
    if [ -f "$cjson" ]; then
        done_=$((done_ + 1))
        continue
    fi
    echo "$y" >> "$QUEUE_FILE"
done

remaining=$(wc -l < "$QUEUE_FILE")
echo "[runner] total=$total done=$done_ remaining=$remaining"

if [ "$remaining" -eq 0 ]; then
    echo "[runner] nothing to do, all cofolds complete"
    exit 0
fi

# Worker: process one YAML
boltz_one() {
    local yaml="$1"
    local uid
    uid=$(basename "$yaml" .yaml)
    local log="$LOG_DIR/${uid}.log"
    local out="$OUT_DIR/boltz_results_${uid}"
    if [ -f "$out/predictions/${uid}/confidence_${uid}_model_0.json" ]; then
        echo "[skip] $uid"
        return 0
    fi
    echo "[start $(date -u +%H:%M:%S)] $uid"
    # use msa_server for one-shot MSAs (single ColabFold call per protein); cached after first run.
    boltz predict "$yaml" \
        --out_dir "$OUT_DIR" \
        --use_msa_server \
        --use_potentials \
        --output_format mmcif \
        --override \
        --accelerator gpu \
        --num_workers 0 \
        > "$log" 2>&1
    rc=$?
    if [ $rc -eq 0 ] && [ -f "$out/predictions/${uid}/confidence_${uid}_model_0.json" ]; then
        echo "[done  $(date -u +%H:%M:%S)] $uid"
    else
        echo "[FAIL  $(date -u +%H:%M:%S)] $uid (rc=$rc) — see $log"
    fi
}
export -f boltz_one
export OUT_DIR LOG_DIR

# Run with xargs -P
start_ts=$(date +%s)
xargs -n1 -P"$N_PARALLEL" -I{} bash -c 'boltz_one "$@"' _ {} < "$QUEUE_FILE"
end_ts=$(date +%s)
wall=$((end_ts - start_ts))

# Final progress
done_after=$(find "$OUT_DIR" -name "confidence_*_model_0.json" | wc -l | tr -d ' ')
echo "[runner] FINAL done=$done_after / $total  wall=${wall}s"
python3 -c "import json,sys; json.dump({'total':$total,'done':$done_after,'wall_s':$wall}, open('$PROGRESS','w'))"
