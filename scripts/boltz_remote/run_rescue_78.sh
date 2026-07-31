#!/usr/bin/env bash
# Boltz-2 cofold runner for rescue-78 cohort.
# Uses isolated dirs: ~/boltz_run/rescue_78_yamls and ~/boltz_run/rescue_78_results
# so it never interferes with the historic 401-mol Lingo3DMol set already in
# ~/boltz_run/yamls and ~/boltz_run/results.
#
# Resumable: skips YAMLs whose confidence JSON already exists.
# N_PARALLEL=2 by default (A100-40GB; each cofold ~12-16GB VRAM).

set -uo pipefail

N_PARALLEL=${N_PARALLEL:-2}
RUN_DIR="$HOME/boltz_run"
YAML_DIR="$RUN_DIR/rescue_78_yamls"
OUT_DIR="$RUN_DIR/rescue_78_results"
LOG_DIR="$RUN_DIR/rescue_78_logs"
QUEUE_FILE="$RUN_DIR/rescue_78_queue.txt"
PROGRESS="$RUN_DIR/rescue_78_progress.json"
mkdir -p "$OUT_DIR" "$LOG_DIR"

# Activate conda
source "$HOME/.bashrc" >/dev/null 2>&1 || true
if [ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]; then
    source "$HOME/miniconda3/etc/profile.d/conda.sh"
fi
for env in boltz boltz2 quris; do
    if conda env list 2>/dev/null | grep -q "^${env} "; then
        conda activate "$env"
        echo "[rescue-78] activated env: $env"
        break
    fi
done
if ! command -v boltz >/dev/null; then
    echo "[rescue-78] ERROR: boltz not found"; exit 1
fi
echo "[rescue-78] boltz: $(boltz --version 2>&1 | head -1)"

# Build queue
: > "$QUEUE_FILE"
total=0; done_=0
for y in "$YAML_DIR"/*.yaml; do
    [ -f "$y" ] || continue
    uid=$(basename "$y" .yaml)
    total=$((total + 1))
    cjson="$OUT_DIR/boltz_results_${uid}/predictions/${uid}/confidence_${uid}_model_0.json"
    if [ -f "$cjson" ]; then
        done_=$((done_ + 1))
        continue
    fi
    echo "$y" >> "$QUEUE_FILE"
done
remaining=$(wc -l < "$QUEUE_FILE")
echo "[rescue-78] total=$total done=$done_ remaining=$remaining"

if [ "$remaining" -eq 0 ]; then
    echo "[rescue-78] nothing to do"
    : > "$RUN_DIR/rescue_78_done"
    exit 0
fi

boltz_one() {
    local yaml="$1"; local uid; uid=$(basename "$yaml" .yaml)
    local log="$LOG_DIR/${uid}.log"
    local out="$OUT_DIR/boltz_results_${uid}"
    if [ -f "$out/predictions/${uid}/confidence_${uid}_model_0.json" ]; then
        echo "[skip] $uid"; return 0
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
        echo "[FAIL  $(date -u +%H:%M:%S)] $uid (rc=$rc) — see $log"
    fi
}
export -f boltz_one
export OUT_DIR LOG_DIR

start_ts=$(date +%s)
xargs -n1 -P"$N_PARALLEL" -I{} bash -c 'boltz_one "$@"' _ {} < "$QUEUE_FILE"
end_ts=$(date +%s)
wall=$((end_ts - start_ts))

done_after=$(find "$OUT_DIR" -name "confidence_*_model_0.json" | wc -l | tr -d ' ')
echo "[rescue-78] FINAL done=$done_after / $total  wall=${wall}s"
python3 -c "import json; json.dump({'total':$total,'done':$done_after,'wall_s':$wall}, open('$PROGRESS','w'))"

# Sentinel
: > "$RUN_DIR/rescue_78_done"
