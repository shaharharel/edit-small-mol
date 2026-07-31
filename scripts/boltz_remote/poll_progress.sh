#!/usr/bin/env bash
# Poll all 3 A100 machines for boltz cofold progress.
# Outputs JSON to /tmp/boltz_progress.json and prints a one-line summary.

set -u

declare -A ZONES=( [ai-gpu-a100]=us-central1-a [ai-gpu-a100-b]=us-central1-b [ai-gpu-a100-c]=us-central1-c )
TOTAL_PER=1199

OUT=/tmp/boltz_progress.json
printf '{\n  "ts_utc": "%s",\n  "machines": {\n' "$(date -u +%FT%TZ)" > "$OUT"

first=1
for m in ai-gpu-a100 ai-gpu-a100-b ai-gpu-a100-c; do
    z=${ZONES[$m]}
    info=$(gcloud compute ssh "$m" --zone="$z" --command='
      n_done=$(find ~/boltz_run/results -name "confidence_*_model_0.json" 2>/dev/null | wc -l)
      n_running=$(pgrep -fc "boltz predict" 2>/dev/null || echo 0)
      gpu_util=$(nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader 2>/dev/null | tr -d " ")
      gpu_mem=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader 2>/dev/null | awk "{print \$1}")
      last_log=$(tail -1 /tmp/boltz_run.log 2>/dev/null | tr -d "\""\\)
      finished=$(test -f /tmp/boltz_run_done && echo true || echo false)
      printf "%s|%s|%s|%s|%s|%s" "$n_done" "$n_running" "$gpu_util" "$gpu_mem" "$last_log" "$finished"
    ' --quiet 2>/dev/null)
    IFS='|' read -r ndone nrun gpu_util gpu_mem last_log finished <<<"$info"
    if [ $first -eq 0 ]; then echo "    ," >> "$OUT"; fi
    first=0
    cat >> "$OUT" <<EOF
    "$m": {"done": ${ndone:-0}, "total": $TOTAL_PER, "running": ${nrun:-0}, "gpu_util_pct": "${gpu_util:-?}", "gpu_mem_mib": "${gpu_mem:-?}", "finished_sentinel": ${finished:-false}, "last_log": "${last_log:-}"}
EOF
done

printf '  }\n}\n' >> "$OUT"
echo "$(date -u +%H:%M:%SZ) progress: $(grep -oE '"done": [0-9]+' "$OUT" | awk -F: '{s+=$2} END {print s}') / 3597"
