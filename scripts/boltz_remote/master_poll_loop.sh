#!/usr/bin/env bash
# Master polling loop for the Vina + cov-Vina rescore pipeline.
# Polls all 6 Boltz machines every 15 min, fires threshold_fire.sh at
# 2000 / 3000 / 3597 cumulative CIFs across the fleet.
#
# Usage: master_poll_loop.sh [start_time_epoch]
# Stops when 3597 is reached, or after 18h, or via TERM signal.

set -u
ROOT="/Users/shaharharel/Documents/github/edit-small-mol"
LOG="$ROOT/logs/master_poll_loop.log"
STATE="/tmp/vina_pipeline_thresholds_fired.txt"
HB="/tmp/vina_pipeline_master.json"
mkdir -p "$ROOT/logs"
touch "$STATE"

START_EPOCH=${1:-$(date +%s)}
MAX_RUN_S=$((18 * 3600))
POLL_INTERVAL_S=900   # 15 min

machines=(
    "ai-gpu-a100:us-central1-a"
    "ai-gpu-a100-b:us-central1-b"
    "ai-gpu-a100-c:us-central1-c"
    "ai-gpu-a100-d:us-east1-b"
    "ai-gpu-a100-e:us-west1-b"
    "ai-gpu-a100-f:us-west4-b"
    "ai-gpu-a100-g:europe-west4-a"
    "ai-gpu-a100-h:us-central1-f"
)

fired() {
    grep -qx "$1" "$STATE" 2>/dev/null
}
mark_fired() {
    echo "$1" >> "$STATE"
}

update_heartbeat_polling() {
    local total="$1"
    local next="$2"
    local eta="$3"
    /opt/miniconda3/envs/quris/bin/python -c "
import json, datetime
hb = json.load(open('$HB'))
hb.update({
    'ts': datetime.datetime.utcnow().isoformat() + 'Z',
    'phase': 'waiting_for_threshold',
    'boltz_total': $total,
    'next_threshold': $next,
    'next_action_eta_sec': $eta,
})
json.dump(hb, open('$HB', 'w'), indent=2)
"
}

while true; do
    now=$(date +%s)
    elapsed=$((now - START_EPOCH))
    if [ "$elapsed" -ge "$MAX_RUN_S" ]; then
        echo "[poll $(date -u +%H:%M:%S)] 18h elapsed, exiting" >> "$LOG"
        echo "POLL_LOOP_EXIT max_runtime"
        exit 0
    fi

    # Probe all 6 machines in parallel.
    total=0
    per_machine=""
    for m in "${machines[@]}"; do
        name="${m%:*}"; zone="${m#*:}"
        c=$(gcloud compute ssh "$name" --zone="$zone" \
              --command='cd ~/boltz_run && find results -name "*.cif" 2>/dev/null | wc -l' \
              2>/dev/null | tr -d ' \r\n')
        c=${c:-0}
        per_machine="$per_machine $name=$c"
        total=$((total + c))
    done
    iso=$(date -u +%Y-%m-%dT%H:%M:%SZ)
    echo "[poll $iso] total=$total/3597$per_machine" >> "$LOG"
    # Emit one line to stdout so the Monitor wrapper records it.
    echo "POLL_PROGRESS total=$total $per_machine"

    # Determine next threshold.
    if [ "$total" -ge 3597 ]; then
        next=3597
    elif [ "$total" -ge 3000 ]; then
        next=3597
    elif [ "$total" -ge 2000 ]; then
        next=3000
    else
        next=2000
    fi
    update_heartbeat_polling "$total" "$next" "$POLL_INTERVAL_S"

    # Fire thresholds (in order).
    for thr in 2000 3000 3597; do
        if [ "$total" -ge "$thr" ] && ! fired "$thr"; then
            echo "[poll $iso] FIRE threshold $thr" >> "$LOG"
            echo "THRESHOLD_FIRE $thr  total=$total"
            mark_fired "$thr"
            bash "$ROOT/scripts/boltz_remote/threshold_fire.sh" "$thr" >> "$LOG" 2>&1
            echo "THRESHOLD_DONE $thr"
        fi
    done

    # All thresholds done?
    if fired 2000 && fired 3000 && fired 3597; then
        echo "[poll $iso] all thresholds fired, exiting" >> "$LOG"
        echo "POLL_LOOP_EXIT all_thresholds_done"
        exit 0
    fi

    # Stall detector: if we've been at the same total for > 90 min beyond
    # 3500, accept that and fire the 3597 threshold anyway.
    if [ "$total" -ge 3500 ] && ! fired 3597; then
        last_total_file="/tmp/vina_pipeline_last_total.txt"
        last_total_ts_file="/tmp/vina_pipeline_last_total_ts.txt"
        if [ -f "$last_total_file" ]; then
            last_total=$(cat "$last_total_file")
            last_ts=$(cat "$last_total_ts_file")
            if [ "$last_total" = "$total" ] && [ $((now - last_ts)) -ge 5400 ]; then
                echo "[poll $iso] STALL at $total (>=3500) for 90 min — firing 3597" >> "$LOG"
                echo "THRESHOLD_FIRE 3597 (stall at $total)"
                mark_fired 3597
                bash "$ROOT/scripts/boltz_remote/threshold_fire.sh" 3597 >> "$LOG" 2>&1
                echo "THRESHOLD_DONE 3597"
                exit 0
            fi
            if [ "$last_total" != "$total" ]; then
                echo "$total" > "$last_total_file"
                echo "$now" > "$last_total_ts_file"
            fi
        else
            echo "$total" > "$last_total_file"
            echo "$now" > "$last_total_ts_file"
        fi
    fi

    sleep "$POLL_INTERVAL_S"
done
