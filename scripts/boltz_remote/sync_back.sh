#!/usr/bin/env bash
# Incremental sync of Boltz cofold outputs from each A100 to local Mac.
# Usage: ./sync_back.sh [a|b|c|all]  (default: all)
# Pulls only NEW boltz_results_<row_id>/predictions/<row_id>/{*.cif,confidence_*.json,*.pdb}

set -u

LOCAL_BASE="/Users/shaharharel/Documents/github/edit-small-mol/data/boltz_results/cohort_3597_full"
WHICH="${1:-all}"

sync_machine() {
    local letter="$1"
    local name="ai-gpu-a100"
    [ "$letter" != "a" ] && name="ai-gpu-a100-${letter}"
    local zone="us-central1-${letter}"
    local dest="${LOCAL_BASE}/from_${letter}/results"
    mkdir -p "$dest"
    echo "[sync $(date +%H:%M:%S)] $name ($zone) -> $dest"

    # Pull entire boltz_results dirs that have a confidence JSON
    # Build remote tarball of relevant files only
    local tarball="/tmp/boltz_sync_${letter}_$$.tgz"
    gcloud compute ssh "$name" --zone="$zone" --command="
        cd ~/boltz_run/results 2>/dev/null || exit 0
        # List dirs with confidence file
        finds=\$(find . -maxdepth 4 -name 'confidence_*_model_0.json' 2>/dev/null | sed 's|/predictions/.*||;s|^./||' | sort -u)
        if [ -z \"\$finds\" ]; then echo 'no confidence files yet'; exit 0; fi
        echo \"\$finds\" | wc -l | xargs -I{} echo 'dirs to pack: {}'
        # Pack only the prediction subdir
        echo \"\$finds\" | while read d; do
            uid=\${d#boltz_results_}
            if [ -d \"\$d/predictions/\$uid\" ]; then
                echo \"\$d/predictions/\$uid\"
            fi
        done > /tmp/sync_${letter}_list.txt
        tar czf /tmp/sync_${letter}.tgz -T /tmp/sync_${letter}_list.txt 2>/dev/null
        ls -la /tmp/sync_${letter}.tgz
    " --quiet 2>&1 | tail -10

    # Pull tarball
    gcloud compute scp "${name}:/tmp/sync_${letter}.tgz" "$tarball" --zone="$zone" --quiet 2>&1 | tail -3
    if [ -f "$tarball" ]; then
        tar xzf "$tarball" -C "$dest"
        rm "$tarball"
        n=$(find "$dest" -name 'confidence_*_model_0.json' 2>/dev/null | wc -l | tr -d ' ')
        echo "[sync] $name: $n confidence files local now"
    fi
}

if [ "$WHICH" = "all" ]; then
    for letter in a b c; do
        sync_machine "$letter" &
    done
    wait
else
    sync_machine "$WHICH"
fi

# Summary
total=$(find "${LOCAL_BASE}/from_a" "${LOCAL_BASE}/from_b" "${LOCAL_BASE}/from_c" -name 'confidence_*_model_0.json' 2>/dev/null | wc -l | tr -d ' ')
echo "TOTAL local confidence files: $total / 3597"
