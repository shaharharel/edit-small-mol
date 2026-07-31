#!/usr/bin/env bash
# rolling_backup.sh — make a versioned local snapshot + GCS push of generated
# molecule cohorts and reproducibility artifacts. Runs forever in a 2h loop
# (cap at 12 snapshots → 24h rolling history).
#
# Usage:   nohup bash experiments/rolling_backup.sh > logs/rolling_backup.log 2>&1 &
# Stop:    kill <pid>   (or pkill -f rolling_backup.sh)
#
# Free-disk guard: aborts a cycle if < 5GB free.

set -uo pipefail

REPO="/Users/shaharharel/Documents/github/edit-small-mol"
BACKUP_ROOT="$REPO/data/_backups"
LOG_DIR="$REPO/logs"
BUCKET="gs://edit-small-mol-cohort-backups-202606"
MAX_SNAPSHOTS=12
SLEEP_SECS=$((2 * 60 * 60))   # 2h
MANIFEST_BUILDER="/tmp/build_manifest.py"

mkdir -p "$BACKUP_ROOT" "$LOG_DIR"

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

one_snapshot() {
    local ts snap free_gb
    ts="$(date +%Y%m%d_%H%M%S)"
    snap="$BACKUP_ROOT/$ts"

    free_gb=$(df -g "$REPO" | awk 'NR==2 {print $4}')
    if [ "$free_gb" -lt 5 ]; then
        log "ABORT cycle: only ${free_gb}GB free (<5GB threshold)"
        return 1
    fi

    log "Snapshot $ts starting (free=${free_gb}GB)"
    mkdir -p "$snap"/{tier1_scored_cohorts,tier2_rl_checkpoints,tier3_local_sampling,tier4_analysis_artifacts}

    # Tier 1: scored cohort CSVs (read-only rsync, checksum-based; --no-times so
    # re-snapshots only copy CHANGED files vs the prev snapshot — but each
    # snapshot is its own dir, so this is mostly defensive vs concurrent writers).
    rsync -a --checksum --no-times "$REPO"/data/tier4_scored/*_scored.csv "$snap/tier1_scored_cohorts/" 2>>"$LOG_DIR/rolling_backup.err" || true
    # Tier 2: RL checkpoints
    rsync -a --checksum --no-times "$REPO"/models/rl_checkpoints_b/*.chkpt "$snap/tier2_rl_checkpoints/" 2>>"$LOG_DIR/rolling_backup.err" || true
    # Tier 3: raw sampling outputs
    rsync -a --checksum --no-times "$REPO"/data/local_sampling/*.csv "$snap/tier3_local_sampling/" 2>>"$LOG_DIR/rolling_backup.err" || true
    # Tier 4: analysis artifacts
    rsync -a --checksum --no-times "$REPO"/results/paper_evaluation/mol1_pharmacophore_*.png "$REPO"/data/tier4_scored/mol1_pharmacophore_coverage.csv "$snap/tier4_analysis_artifacts/" 2>>"$LOG_DIR/rolling_backup.err" || true

    # Manifest
    if [ -f "$MANIFEST_BUILDER" ]; then
        python3 "$MANIFEST_BUILDER" "$ts" >>"$LOG_DIR/rolling_backup.log" 2>&1 || log "manifest builder failed"
    fi

    local size_h
    size_h=$(du -sh "$snap" | awk '{print $1}')
    log "Local snapshot complete: $snap ($size_h)"

    # GCS push (parallel)
    if command -v gsutil >/dev/null 2>&1; then
        log "Pushing to $BUCKET/$ts ..."
        if gsutil -m cp -r "$snap" "$BUCKET/" >>"$LOG_DIR/rolling_backup.log" 2>>"$LOG_DIR/rolling_backup.err"; then
            log "GCS upload complete: $BUCKET/$ts"
        else
            log "GCS upload FAILED (continuing — local snapshot still intact)"
        fi
    else
        log "gsutil not available — skipping GCS push"
    fi

    # Rotate: keep only the most recent MAX_SNAPSHOTS local dirs
    local n_snaps
    n_snaps=$(ls -1 "$BACKUP_ROOT" | grep -E '^[0-9]{8}_[0-9]{6}$' | wc -l | tr -d ' ')
    if [ "$n_snaps" -gt "$MAX_SNAPSHOTS" ]; then
        local to_drop
        to_drop=$(ls -1 "$BACKUP_ROOT" | grep -E '^[0-9]{8}_[0-9]{6}$' | sort | head -n $((n_snaps - MAX_SNAPSHOTS)))
        for d in $to_drop; do
            log "Rotating out old snapshot: $d"
            rm -rf "$BACKUP_ROOT/$d"
        done
    fi
}

log "rolling_backup.sh started (PID $$, cadence ${SLEEP_SECS}s, cap=$MAX_SNAPSHOTS)"
while true; do
    one_snapshot || log "snapshot cycle returned non-zero"
    log "Sleeping ${SLEEP_SECS}s until next snapshot..."
    sleep "$SLEEP_SECS"
done
