#!/bin/bash
# Run Boltz cofold over a slice of COValid YAMLs.
# Args: SLICE_START  SLICE_END  OUT_DIR
# Caller hands one slice to each machine (A100, ai-gpu2).

set -e
YAML_ROOT=~/covalid_yamls/covalid_top100
OUT=${3:-~/covalid_results}
S=${1:-0}
E=${2:-9999}
LOG=~/logs/covalid_boltz_${S}_${E}.log

source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris

mkdir -p "$OUT" ~/logs

# Build a list of YAMLs sorted by name (deterministic) and slice.
mapfile -t ALL < <(find "$YAML_ROOT" -name '*.yaml' | sort)
TOTAL=${#ALL[@]}
END=$(( E < TOTAL ? E : TOTAL ))

echo "BOLTZ COFOLD BATCH" | tee -a "$LOG"
echo "  total YAMLs: $TOTAL" | tee -a "$LOG"
echo "  slice [$S, $END)" | tee -a "$LOG"
echo "  out: $OUT" | tee -a "$LOG"
echo "  log: $LOG" | tee -a "$LOG"

t0=$(date +%s)
n_done=0
for ((i = S; i < END; i++)); do
    Y=${ALL[$i]}
    NAME=$(basename "$Y" .yaml)
    TARGET=$(basename "$(dirname "$Y")")
    OUT_SUB="$OUT/$TARGET"
    mkdir -p "$OUT_SUB"
    if [ -f "$OUT_SUB/boltz_results_${NAME}/predictions/${NAME}/${NAME}_model_0.cif" ]; then
        echo "[$i/$END] $NAME already done — skip" | tee -a "$LOG"
        continue
    fi
    t1=$(date +%s)
    boltz predict "$Y" \
        --out_dir "$OUT_SUB" \
        --accelerator gpu --diffusion_samples 1 --sampling_steps 100 \
        --output_format mmcif --override --use_msa_server \
        >> "$LOG" 2>&1 || echo "[$i/$END] $NAME FAILED" | tee -a "$LOG"
    n_done=$((n_done+1))
    dt=$(( $(date +%s) - t1 ))
    et=$(( $(date +%s) - t0 ))
    echo "[$i/$END] $NAME done in ${dt}s  (cumulative ${n_done} in ${et}s)" | tee -a "$LOG"
done

echo "BATCH COMPLETE: $n_done cofolds in $(( $(date +%s) - t0 ))s" | tee -a "$LOG"
