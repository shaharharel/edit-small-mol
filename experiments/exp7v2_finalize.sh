#!/bin/bash
# EXP7-v2 finalize: pull all cohorts from VM, score them, generate report + summary.
set -euo pipefail

PROJ=/Users/shaharharel/Documents/github/edit-small-mol
DEST=$PROJ/data/exp7_v2_benchmark/_rl

echo "=== 1. Final pull from a100-b ==="
mkdir -p $PROJ/data/exp7_v2_benchmark/_rl
gcloud compute scp --recurse --tunnel-through-iap --zone=us-central1-b --compress \
    ai-gpu-a100-b:/home/shaharh_quris_ai/edit-small-mol/data/exp7_v2_benchmark/_rl \
    "$PROJ/data/exp7_v2_benchmark/" 2>&1 | tail -5

N=$(find $DEST -name 'sampled.csv' -size +500c | wc -l)
echo "[finalize] pulled — sampled.csv > 500B: $N (expected 216)"

echo "=== 2. Score cohorts ==="
/opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7v2_score_cohorts.py 2>&1 | tail -25

echo "=== 3. Generate HTML + JSON report ==="
/opt/miniconda3/envs/quris/bin/python $PROJ/experiments/exp7v2_4way_report.py 2>&1 | tail -15

echo "=== 4. Display top-line summary ==="
/opt/miniconda3/envs/quris/bin/python -c "
import json
with open('$PROJ/data/exp7_v2_benchmark/exp7_v2_4way_summary.json') as f:
    d = json.load(f)
print('Methods x Metrics — median values:')
print(f\"{'metric':30s} \" + ' '.join(f'{m:>16s}' for m in d['methods']))
for metric in d['metrics']:
    line = f\"{metric:30s} \"
    for m in d['methods']:
        v = d['aggregate'][m][metric]['median']
        line += f'{v:>16.4f} '
    print(line)
print()
print('Wins (out of 54):')
print(f\"{'method':20s} \" + ' '.join(f'{m:>22s}' for m in d['metrics']))
for m in d['methods']:
    line = f'{m:20s} '
    for metric in d['metrics']:
        line += f'{d[\"win_counts\"][m][metric]:>22d} '
    print(line)
"

echo "=== 5. STOP a100-b ==="
gcloud compute instances stop ai-gpu-a100-b --zone=us-central1-b 2>&1 | tail -3
gcloud compute instances describe ai-gpu-a100-b --zone=us-central1-b --format='value(status)'
