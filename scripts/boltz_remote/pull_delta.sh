#!/usr/bin/env bash
# Delta pull of Boltz cofold outputs from a single machine to local Mac.
# Pulls only cofold dirs that are NOT already present locally.
#
# Layout:
#   remote: ~/boltz_run/results/boltz_results_<id>/predictions/<id>/{<id>_model_0.cif,confidence_*.json,...}
#   local : data/boltz_results/cohort_3597_full/from_<letter>/<id>/{<id>_model_0.cif,...}
#
# Per cofold pulled, this script grabs only the five files needed by the rescore
# pipelines:
#   <id>_model_0.cif
#   <id>_model_0.pdb            (may not exist remotely — we generate from CIF)
#   <id>_model_0.lig.sdf        (split-output if remote produced it)
#   <id>_model_0.prot.pdb       (split-output if remote produced it)
#   confidence_<id>_model_0.json
#
# Usage: pull_delta.sh <letter> <name> <zone>
#   letter = a|b|c|d|e|f
#   name   = ai-gpu-a100[-x]
#   zone   = us-central1-a / us-east1-b / etc.

set -u

LETTER="${1:?letter}"
NAME="${2:?name}"
ZONE="${3:?zone}"
LOCAL_BASE="/Users/shaharharel/Documents/github/edit-small-mol/data/boltz_results/cohort_3597_full"
DEST="${LOCAL_BASE}/from_${LETTER}"
mkdir -p "$DEST"

# 1) Build "already-have" list (cofold IDs already on disk with a CIF).
HAVE_FILE=$(mktemp)
ls -1 "$DEST" 2>/dev/null | while read d; do
    if [ -f "$DEST/$d/${d}_model_0.cif" ]; then
        echo "$d"
    fi
done | sort -u > "$HAVE_FILE"
N_HAVE=$(wc -l < "$HAVE_FILE" | tr -d ' ')

# 2) Ask remote for all cofold IDs that have produced a CIF.
REMOTE_LIST=$(mktemp)
gcloud compute ssh "$NAME" --zone="$ZONE" --command="
    cd ~/boltz_run/results 2>/dev/null || exit 0
    find . -maxdepth 4 -name '*_model_0.cif' 2>/dev/null \
        | awk -F'/' '{print \$NF}' \
        | sed 's/_model_0.cif$//' \
        | sort -u
" --quiet 2>/dev/null > "$REMOTE_LIST"
N_REMOTE=$(wc -l < "$REMOTE_LIST" | tr -d ' ')

# 3) Compute delta (remote minus have).
DELTA_FILE=$(mktemp)
comm -23 "$REMOTE_LIST" "$HAVE_FILE" > "$DELTA_FILE"
N_DELTA=$(wc -l < "$DELTA_FILE" | tr -d ' ')

echo "[pull-delta $LETTER] remote=$N_REMOTE  local=$N_HAVE  delta=$N_DELTA"
if [ "$N_DELTA" -eq 0 ]; then
    rm -f "$HAVE_FILE" "$REMOTE_LIST" "$DELTA_FILE"
    exit 0
fi

# 4) Tar only the 5 wanted files per delta cofold remotely, then scp.
REMOTE_LIST_BASENAME="sync_${LETTER}_delta_$$.list"
TAR_NAME="sync_${LETTER}_delta_$$.tgz"
LOCAL_TAR="/tmp/${TAR_NAME}"

# Upload the delta list to the remote machine.
gcloud compute scp "$DELTA_FILE" "${NAME}:/tmp/${REMOTE_LIST_BASENAME}" --zone="$ZONE" --quiet 2>&1 | tail -2

gcloud compute ssh "$NAME" --zone="$ZONE" --command="
    cd ~/boltz_run/results 2>/dev/null || { echo 'no results dir'; exit 1; }
    LIST=/tmp/${REMOTE_LIST_BASENAME}
    : > /tmp/${TAR_NAME}.files
    while read uid; do
        d=\"boltz_results_\${uid}/predictions/\${uid}\"
        if [ -d \"\$d\" ]; then
            for f in \"\${uid}_model_0.cif\" \"\${uid}_model_0.pdb\" \"\${uid}_model_0.lig.sdf\" \"\${uid}_model_0.prot.pdb\" \"confidence_\${uid}_model_0.json\"; do
                if [ -f \"\$d/\$f\" ]; then echo \"\$d/\$f\" >> /tmp/${TAR_NAME}.files; fi
            done
        fi
    done < \$LIST
    n_files=\$(wc -l < /tmp/${TAR_NAME}.files | tr -d ' ')
    echo \"files-to-pack=\$n_files\"
    tar czf /tmp/${TAR_NAME} -T /tmp/${TAR_NAME}.files 2>/dev/null
    ls -la /tmp/${TAR_NAME}
    rm -f \$LIST /tmp/${TAR_NAME}.files
" --quiet 2>&1 | tail -5

gcloud compute scp "${NAME}:/tmp/${TAR_NAME}" "$LOCAL_TAR" --zone="$ZONE" --quiet 2>&1 | tail -2

if [ ! -f "$LOCAL_TAR" ]; then
    echo "[pull-delta $LETTER] ERROR: tar pull failed"
    rm -f "$HAVE_FILE" "$REMOTE_LIST" "$DELTA_FILE"
    exit 1
fi

# 5) Extract into per-cofold subdirs.
STAGE=$(mktemp -d)
tar xzf "$LOCAL_TAR" -C "$STAGE" 2>/dev/null || { echo "[pull-delta $LETTER] ERROR: tar extract failed"; exit 1; }
N_MOVED=0
for src in "$STAGE"/boltz_results_*/predictions/*/; do
    [ -d "$src" ] || continue
    uid=$(basename "$src")
    target_dir="$DEST/$uid"
    mkdir -p "$target_dir"
    cp -n "$src"*_model_0.cif "$target_dir/" 2>/dev/null || true
    cp -n "$src"*_model_0.pdb "$target_dir/" 2>/dev/null || true
    cp -n "$src"*_model_0.lig.sdf "$target_dir/" 2>/dev/null || true
    cp -n "$src"*_model_0.prot.pdb "$target_dir/" 2>/dev/null || true
    cp -n "$src"confidence_*_model_0.json "$target_dir/" 2>/dev/null || true
    N_MOVED=$((N_MOVED+1))
done
rm -rf "$STAGE" "$LOCAL_TAR"

# 6) Clean up remote tar so it doesn't keep growing on the GPU box.
gcloud compute ssh "$NAME" --zone="$ZONE" --command="rm -f /tmp/${TAR_NAME}" --quiet 2>/dev/null

echo "[pull-delta $LETTER] moved $N_MOVED cofold dirs into $DEST"
rm -f "$HAVE_FILE" "$REMOTE_LIST" "$DELTA_FILE"
