#!/bin/bash
# Auto-restart C1 cofolding with merged sample pool once extra sampling completes.
# Idempotent, resumable — the driver skips already-cofolded YAMLs.

set -u
ROOT=/home/shaharh_quris_ai/edit-small-mol
CAMPAIGN=$ROOT/data/paper_pair_training/boltz_dpo_campaign
SAMPLES=$CAMPAIGN/samples
LOGS=$CAMPAIGN/logs

# Wait for extra sampler to finish
echo "[wait] extra sampler..."
while pgrep -f sample_dpo_composite_cohort > /dev/null; do
  sleep 30
done
echo "[wait] extra sampler done at $(date -u +%FT%TZ)"

# Merge raw + extra (dedup on SMILES)
MERGED=$SAMPLES/c1_composite_v1_merged.csv
python3 - <<'PY'
import csv
from pathlib import Path
root = Path("/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/boltz_dpo_campaign/samples")
seen = set()
rows = []
for name in ["c1_composite_v1_raw.csv", "c1_composite_v1_extra.csv"]:
    p = root / name
    if not p.exists():
        continue
    with p.open() as fh:
        rd = csv.reader(fh)
        header = next(rd, None)
        for row in rd:
            if not row:
                continue
            s = row[0].strip()
            if not s or s in seen:
                continue
            seen.add(s)
            rows.append(row)
out = root / "c1_composite_v1_merged.csv"
with out.open("w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(header or ["SMILES", "Input_SMILES", "NLL"])
    for r in rows:
        w.writerow(r)
print(f"merged {len(rows)} unique SMILES to {out}")
PY

# Wait for current driver to finish current 70 batch (or restart it directly)
echo "[wait] first driver instance..."
while pgrep -f "boltz_dpo_campaign_driver.py.*c1_composite_v1" > /dev/null; do
  sleep 30
done
echo "[wait] first driver done at $(date -u +%FT%TZ)"

# Relaunch driver with merged pool; existing YAMLs will be re-inspected, existing CIFs skipped
source ~/miniconda3/etc/profile.d/conda.sh
conda activate quris
cd $ROOT
nohup python experiments/boltz_dpo_campaign_driver.py \
  --cohort c1_composite_v1 \
  --smiles_csv $MERGED \
  --n_target 400 \
  --max_workers 2 \
  > $LOGS/driver_c1_v2.log 2>&1 &
echo "[start] driver v2 PID=$! at $(date -u +%FT%TZ)"
