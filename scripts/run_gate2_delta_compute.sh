#!/bin/bash
# Gate 2 delta-compute orchestration.
#
# Trigger: /tmp/BOLTZ_DONE exists. New A+B CIFs have landed (~260 row_ids).
#
# Pipeline:
#   1. Identify delta = (CIFs on disk now) - (CIFs that existed at last orchestrator
#      compute pass). Save to /tmp/boltz_final_delta_rids.json.
#   2. Run all six compute pipelines on the delta rows:
#       - compute_full_boltz_metrics  (geom/contacts/mPAE-proxy)
#       - xTB (HOMO/LUMO/q_Cb/fukui/omega/gap)  — actually already 100% on full 3,597
#       - rdkit cofold pose strain
#       - PROPKA pKa_Cys346
#       - Vina rescore (vanilla --score_only on Boltz pose)
#       - covvina rescore (AD-CovDock --local_only on Boltz pose)
#   3. Re-run merge_boltz_metrics_into_cohort.py (which now also reads xTB and
#      v2 strain/PROPKA CSVs via the iter-4 patch).
#   4. Restart backend.
#
# Designed to be idempotent / resume-safe (every batch script supports `--limit`
# or has built-in skip-on-done logic).

set -e
cd "$(dirname "$0")/.."

PATH=/opt/miniconda3/envs/quris/bin:$PATH
PY=/opt/miniconda3/envs/quris/bin/python3

ts() { date -u +%FT%TZ; }

echo "[$(ts)] Gate 2 delta compute starting"

# Step 1: count CIFs now
N_NOW=$(find data/boltz_results/cohort_3597_full -name "*_model_0.cif" 2>/dev/null \
  | awk -F/ '{print $(NF-1)}' | sort -u | wc -l | tr -d ' ')
echo "[$(ts)] CIFs on disk now: $N_NOW"

# Save the delta list (everything on disk now minus what was in the last
# boltz_full_metrics_3597.csv pass)
$PY -c "
from pathlib import Path
import pandas as pd, json
prev_csv = Path('data/tier4_scored/boltz_full_metrics_3597.csv')
prev_rids = set()
if prev_csv.exists():
    prev_rids = set(pd.read_csv(prev_csv)['row_id'].astype(int))
root = Path('data/boltz_results/cohort_3597_full')
cur_rids = set()
for sh in 'abcdefgh':
    for cif in root.glob(f'from_{sh}/*/*_model_0.cif'):
        try: cur_rids.add(int(cif.parent.name))
        except: pass
delta = sorted(cur_rids - prev_rids)
print(f'prev: {len(prev_rids)}; current: {len(cur_rids)}; delta: {len(delta)}')
with open('/tmp/boltz_final_delta_rids.json', 'w') as f:
    json.dump({'delta_row_ids': delta, 'n_delta': len(delta),
               'n_prev_cifs': len(prev_rids), 'n_cur_cifs': len(cur_rids)}, f)
print('wrote /tmp/boltz_final_delta_rids.json')
"

# Step 2a: compute_full_boltz_metrics (resume picks up new CIFs)
echo "[$(ts)] STEP 2a: compute_full_boltz_metrics"
$PY scripts/compute_full_boltz_metrics.py --workers 8 2>&1 | tail -10

# Step 2b: xTB (already 100% on full 3,597 from earlier orchestrator pass — skip)
echo "[$(ts)] STEP 2b: xTB already 100% on full 3597 cohort. Skipping."

# Step 2c: pose-strain (resume picks up new CIFs)
echo "[$(ts)] STEP 2c: pose-strain"
$PY experiments/score_rdkit_strain_batch.py --mode cohort3597 --workers 8 2>&1 | tail -10

# Step 2d: PROPKA (resume picks up new CIFs)
echo "[$(ts)] STEP 2d: PROPKA"
$PY experiments/propka_cys346_batch_cohort3597.py --workers 8 2>&1 | tail -10

# Step 2e: Vina rescore (resume picks up new CIFs)
echo "[$(ts)] STEP 2e: Vina rescore"
$PY experiments/vina_rescore_cohort_3597.py --workers 5 2>&1 | tail -10

# Step 2f: covvina rescore (resume picks up new CIFs)
echo "[$(ts)] STEP 2f: covvina rescore"
$PY experiments/covvina_rescore_cohort_3597.py --workers 5 2>&1 | tail -10

# Step 3: merge
echo "[$(ts)] STEP 3: merge_boltz_metrics_into_cohort"
cp data/tier4_scored/boltz2_cohort_A_relaxed.csv \
   data/tier4_scored/boltz2_cohort_A_relaxed.csv.bak.gate2_pre_merge
$PY scripts/merge_boltz_metrics_into_cohort.py 2>&1 | tail -20

# Step 4: restart backend
echo "[$(ts)] STEP 4: restart backend"
pkill -f 'backend.py' 2>/dev/null || true
sleep 3
nohup $PY experiments/server/backend.py > /tmp/backend.log 2>&1 &
echo "[$(ts)] backend PID=$!"

# Wait for it to listen
for i in 1 2 3 4 5 6 7 8 9 10; do
    if lsof -i :5001 >/dev/null 2>&1; then
        echo "[$(ts)] backend listening on :5001"
        break
    fi
    sleep 5
done

echo "[$(ts)] Gate 2 delta compute COMPLETE"
