#!/usr/bin/env bash
# Boltz-2 cofold runner for cohort 3597. Uses ~/boltz3597 (not ~/boltz_run)
# to avoid the QA agent's clean_remote.sh wiping work.
# Resumable: skips YAMLs whose output directory already exists with confidence JSON.

set -uo pipefail

N_PARALLEL=${N_PARALLEL:-2}
RUN_DIR="$HOME/boltz3597"
YAML_DIR="$RUN_DIR/yamls"
OUT_DIR="$RUN_DIR/results"
LOG_DIR="$RUN_DIR/logs"
PROGRESS="$RUN_DIR/progress.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Activate conda env
source "$HOME/.bashrc" >/dev/null 2>&1 || true
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [ -f "/opt/conda/etc/profile.d/conda.sh" ]; then
    source "/opt/conda/etc/profile.d/conda.sh"
elif command -v conda >/dev/null; then
    eval "$(conda shell.bash hook)"
fi

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

echo "[runner] N_PARALLEL=$N_PARALLEL  YAML_DIR=$YAML_DIR  OUT_DIR=$OUT_DIR"

# Build queue: skip done. Filter out macOS resource forks (._*).
QUEUE_FILE="$RUN_DIR/queue.txt"
: > "$QUEUE_FILE"
total=0
done_=0
for y in "$YAML_DIR"/*.yaml; do
    base=$(basename "$y")
    # Skip macOS resource forks
    case "$base" in ._*) continue ;; esac
    uid="${base%.yaml}"
    total=$((total + 1))
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
        echo "[FAIL  $(date -u +%H:%M:%S)] $uid (rc=$rc) -- see $log"
    fi
}
export -f boltz_one
export OUT_DIR LOG_DIR

start_ts=$(date +%s)
xargs -n1 -P"$N_PARALLEL" -I{} bash -c 'boltz_one "$@"' _ {} < "$QUEUE_FILE"
end_ts=$(date +%s)
wall=$((end_ts - start_ts))

done_after=$(find "$OUT_DIR" -name "confidence_*_model_0.json" 2>/dev/null | wc -l | tr -d ' ')
echo "[runner] FINAL done=$done_after / $total  wall=${wall}s"
python3 -c "import json,sys; json.dump({'total':$total,'done':$done_after,'wall_s':$wall}, open('$PROGRESS','w'))" 2>/dev/null || true
