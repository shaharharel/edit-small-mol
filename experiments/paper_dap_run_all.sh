#!/bin/bash
# Run all three DAP attempts sequentially. Writes progress to a top-level log.
set -u
REPO=/home/shaharh_quris_ai/edit-small-mol
WORK=/home/shaharh_quris_ai/paper_dap_repro
mkdir -p $WORK
TOP_LOG=$WORK/run_all.log
STATUS=$WORK/status.txt

echo "[$(date -u +%FT%TZ)] START run_all" > $TOP_LOG
echo "STARTED $(date -u +%FT%TZ)" > $STATUS

for TAG in C A B; do
    REPORT=$WORK/report_${TAG}.json
    if [[ -f "$REPORT" ]]; then
        MEAN=$(python3 -c "import json; print(json.load(open('$REPORT'))['pIC50_mean'])" 2>/dev/null || echo "NaN")
        echo "[$(date -u +%FT%TZ)] Skipping $TAG — already has report (mean=$MEAN)" >> $TOP_LOG
        continue
    fi
    echo "" >> $TOP_LOG
    echo "==============================================================" >> $TOP_LOG
    echo "[$(date -u +%FT%TZ)] ATTEMPT $TAG" >> $TOP_LOG
    echo "==============================================================" >> $TOP_LOG
    bash $REPO/experiments/paper_dap_run_attempt.sh $TAG 2>&1 | tee -a $TOP_LOG

    # Check if the report file exists and if attempt hit the 7.0 threshold
    REPORT=$WORK/report_${TAG}.json
    if [[ -f "$REPORT" ]]; then
        MEAN=$(python3 -c "import json; print(json.load(open('$REPORT'))['pIC50_mean'])" 2>/dev/null || echo "NaN")
        MEDIAN=$(python3 -c "import json; print(json.load(open('$REPORT'))['pIC50_median'])" 2>/dev/null || echo "NaN")
        echo "[$(date -u +%FT%TZ)] Attempt $TAG: mean=$MEAN median=$MEDIAN" >> $STATUS

        # Early-stop if we hit >= 7.0 on mean or median
        WINNER=$(python3 -c "import json; r=json.load(open('$REPORT')); print(1 if (r['pIC50_mean']>=7.0 or r['pIC50_median']>=7.0) else 0)" 2>/dev/null || echo "0")
        if [[ "$WINNER" == "1" ]]; then
            echo "WINNER $TAG at $(date -u +%FT%TZ)" >> $STATUS
            echo "" >> $TOP_LOG
            echo "*** ATTEMPT $TAG HIT 7.0 THRESHOLD — STOPPING ***" >> $TOP_LOG
            break
        fi
    else
        echo "[$(date -u +%FT%TZ)] Attempt $TAG: NO REPORT" >> $STATUS
    fi
done

echo "[$(date -u +%FT%TZ)] END run_all" >> $TOP_LOG
echo "FINISHED $(date -u +%FT%TZ)" >> $STATUS
