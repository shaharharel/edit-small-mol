#!/usr/bin/env bash
# Pull rescue-78 Boltz cofold outputs from A100 ai-gpu-a100 to the local Mac.
# Pulls the 5 files per cofold needed for the rescore pipelines.
#
# Layout (remote): ~/boltz_run/rescue_78_results/boltz_results_<rid>/predictions/<rid>/...
# Layout (local) : data/boltz_rescue_78/<rid>/...

set -uo pipefail

ZONE=us-central1-a
NAME=ai-gpu-a100
LOCAL_BASE="/Users/shaharharel/Documents/github/edit-small-mol/data/boltz_rescue_78"
mkdir -p "$LOCAL_BASE"

# 1) Build local "already-have" list
HAVE_FILE=$(mktemp)
ls -1 "$LOCAL_BASE" 2>/dev/null | while read d; do
    if [ -f "$LOCAL_BASE/$d/${d}_model_0.cif" ]; then echo "$d"; fi
done | sort -u > "$HAVE_FILE"
N_HAVE=$(wc -l < "$HAVE_FILE" | tr -d ' ')

# 2) Ask remote for all cofold IDs that have produced a CIF
REMOTE_LIST=$(mktemp)
gcloud compute ssh "$NAME" --zone="$ZONE" --tunnel-through-iap --command="
    cd \$HOME/boltz_run/rescue_78_results 2>/dev/null || exit 0
    find . -maxdepth 4 -name '*_model_0.cif' 2>/dev/null \
        | awk -F'/' '{print \$NF}' \
        | sed 's/_model_0.cif\$//' \
        | sort -u
" --quiet 2>/dev/null > "$REMOTE_LIST"
N_REMOTE=$(wc -l < "$REMOTE_LIST" | tr -d ' ')

# 3) Delta
DELTA_FILE=$(mktemp)
comm -23 "$REMOTE_LIST" "$HAVE_FILE" > "$DELTA_FILE"
N_DELTA=$(wc -l < "$DELTA_FILE" | tr -d ' ')

echo "[rescue-78 pull] remote=$N_REMOTE  local=$N_HAVE  delta=$N_DELTA"
if [ "$N_DELTA" -eq 0 ]; then
    rm -f "$HAVE_FILE" "$REMOTE_LIST" "$DELTA_FILE"
    exit 0
fi

# 4) Tar the wanted files remotely
REMOTE_LIST_BASENAME="rescue_78_delta_$$.list"
TAR_NAME="rescue_78_delta_$$.tgz"
LOCAL_TAR="/tmp/${TAR_NAME}"

gcloud compute scp "$DELTA_FILE" "${NAME}:/tmp/${REMOTE_LIST_BASENAME}" \
    --zone="$ZONE" --tunnel-through-iap --quiet 2>&1 | tail -2

gcloud compute ssh "$NAME" --zone="$ZONE" --tunnel-through-iap --command="
    cd \$HOME/boltz_run/rescue_78_results 2>/dev/null || { echo 'no results dir'; exit 1; }
    LIST=/tmp/${REMOTE_LIST_BASENAME}
    : > /tmp/${TAR_NAME}.files
    while read uid; do
        d=\"boltz_results_\${uid}/predictions/\${uid}\"
        if [ -d \"\$d\" ]; then
            for f in \"\${uid}_model_0.cif\" \"\${uid}_model_0.pdb\" \"\${uid}_model_0.lig.sdf\" \"\${uid}_model_0.prot.pdb\" \"confidence_\${uid}_model_0.json\" \"pae_\${uid}_model_0.npz\" \"pde_\${uid}_model_0.npz\" \"plddt_\${uid}_model_0.npz\"; do
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

gcloud compute scp "${NAME}:/tmp/${TAR_NAME}" "$LOCAL_TAR" \
    --zone="$ZONE" --tunnel-through-iap --quiet 2>&1 | tail -2

if [ ! -f "$LOCAL_TAR" ]; then
    echo "[rescue-78 pull] ERROR: tar pull failed"
    rm -f "$HAVE_FILE" "$REMOTE_LIST" "$DELTA_FILE"
    exit 1
fi

# 5) Extract into per-cofold subdirs (flatten predictions/<rid>/* into <rid>/*)
STAGE=$(mktemp -d)
tar xzf "$LOCAL_TAR" -C "$STAGE" 2>/dev/null || { echo "[rescue-78 pull] ERROR: tar extract failed"; exit 1; }
N_MOVED=0
while read uid; do
    src_dir="$STAGE/boltz_results_${uid}/predictions/${uid}"
    if [ ! -d "$src_dir" ]; then continue; fi
    mkdir -p "$LOCAL_BASE/$uid"
    cp -p "$src_dir"/* "$LOCAL_BASE/$uid/" 2>/dev/null
    N_MOVED=$((N_MOVED + 1))
done < "$DELTA_FILE"

rm -rf "$STAGE" "$LOCAL_TAR"
rm -f "$HAVE_FILE" "$REMOTE_LIST" "$DELTA_FILE"
echo "[rescue-78 pull] moved=$N_MOVED  total_local=$(ls "$LOCAL_BASE" | wc -l | tr -d ' ')"
