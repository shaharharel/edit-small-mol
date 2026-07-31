#!/usr/bin/env bash
# Threshold-fire: pull deltas from all 6 Boltz machines, then run vanilla Vina
# + cov-Vina --score_only on the newly pulled cofolds, then QA.
#
# Usage: threshold_fire.sh <threshold>
#   threshold = 2000 | 3000 | 3597 (used only for log labelling)

set -u
THRESH="${1:?threshold}"
ROOT="/Users/shaharharel/Documents/github/edit-small-mol"
LOG_DIR="$ROOT/logs"
mkdir -p "$LOG_DIR"
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
LOG="$LOG_DIR/threshold_${THRESH}_${STAMP}.log"

echo "[fire $THRESH $STAMP] start" | tee -a "$LOG"

# 1) Pull deltas from all 8 machines in parallel.
echo "[fire $THRESH] pulling deltas..." | tee -a "$LOG"
for m in "ai-gpu-a100:us-central1-a:a" \
         "ai-gpu-a100-b:us-central1-b:b" \
         "ai-gpu-a100-c:us-central1-c:c" \
         "ai-gpu-a100-d:us-east1-b:d" \
         "ai-gpu-a100-e:us-west1-b:e" \
         "ai-gpu-a100-f:us-west4-b:f" \
         "ai-gpu-a100-g:europe-west4-a:g" \
         "ai-gpu-a100-h:us-central1-f:h"; do
    name="${m%%:*}"
    rest="${m#*:}"
    zone="${rest%:*}"
    letter="${rest##*:}"
    bash "$ROOT/scripts/boltz_remote/pull_delta.sh" "$letter" "$name" "$zone" >> "$LOG" 2>&1 &
done
wait
echo "[fire $THRESH] all pulls done" | tee -a "$LOG"

# Count what's on disk now.
TOTAL_LOCAL=$(find "$ROOT/data/boltz_results/cohort_3597_full/from_"* -maxdepth 2 -name '*_model_0.cif' 2>/dev/null | wc -l | tr -d ' ')
echo "[fire $THRESH] total cofolds local now: $TOTAL_LOCAL" | tee -a "$LOG"

# 2) Run vanilla Vina --score_only (resumable; appends new rows).
echo "[fire $THRESH] vanilla Vina rescore..." | tee -a "$LOG"
/opt/miniconda3/envs/quris/bin/python "$ROOT/experiments/vina_rescore_cohort_3597.py" \
    --workers 6 >> "$LOG" 2>&1
RC1=$?
echo "[fire $THRESH] vanilla rc=$RC1" | tee -a "$LOG"

# 3) Run cov-Vina --score_only (resumable; appends new rows).
echo "[fire $THRESH] cov-Vina rescore..." | tee -a "$LOG"
/opt/miniconda3/envs/quris/bin/python "$ROOT/experiments/covvina_rescore_cohort_3597.py" \
    --workers 6 >> "$LOG" 2>&1
RC2=$?
echo "[fire $THRESH] cov-Vina rc=$RC2" | tee -a "$LOG"

# 4) QA — schema, success rate, distribution, file integrity.
/opt/miniconda3/envs/quris/bin/python - "$THRESH" "$LOG" <<'PYEOF' 2>&1 | tee -a "$LOG"
import sys, json, datetime, random
from pathlib import Path
import pandas as pd

THRESH = int(sys.argv[1])
LOG_PATH = sys.argv[2]
ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
ts = datetime.datetime.utcnow().isoformat() + "Z"
qa_lines = []

# Vanilla
v = pd.read_csv(ROOT / "data/tier4_scored/vina_rescore_boltz_intermediate.csv")
c = pd.read_csv(ROOT / "data/tier4_scored/covvina_rescore_boltz_intermediate.csv")
print(f"vanilla rows={len(v)}  success={(v.success_flag==1).sum()}")
print(f"cov-Vina rows={len(c)}  success={(c.success_flag==1).sum()}")

# Schema
v_cols_ok = list(v.columns) == [
    "yaml_name","row_id","vina_affinity_kcalmol","vina_inter_kcalmol",
    "vina_intra_kcalmol","vina_torsions_kcalmol","vina_unbound_kcalmol",
    "success_flag","error","boltz_machine","n_protein_atoms","n_lig_atoms","ts",
]
c_cols_ok = list(c.columns) == [
    "yaml_name","row_id","covvina_affinity_kcalmol","covvina_inter_kcalmol",
    "covvina_intra_kcalmol","covvina_torsions_kcalmol","covvina_unbound_kcalmol",
    "success_flag","error","boltz_machine","n_protein_atoms","n_lig_atoms","ts",
]

# NaN gate
v_nan = v.yaml_name.isna().any() or v.row_id.isna().any()
c_nan = c.yaml_name.isna().any() or c.row_id.isna().any()

# Success rate ≥ 90%
v_succ = (v.success_flag==1).mean()
c_succ = (c.success_flag==1).mean()

# Distribution: median should not shift > 2 kcal/mol from baseline.
v_med = v[v.success_flag==1].vina_affinity_kcalmol.median()
c_med = c[c.success_flag==1].covvina_affinity_kcalmol.median()
V_BASELINE_MED = 0.255
C_BASELINE_MED = -3.348
v_drift_ok = abs(v_med - V_BASELINE_MED) <= 2.0
c_drift_ok = abs(c_med - C_BASELINE_MED) <= 2.0

# Affinity in [-15, +200] for ok rows
v_range_ok = v[v.success_flag==1].vina_affinity_kcalmol.between(-15, 200).all()
c_range_ok = c[c.success_flag==1].covvina_affinity_kcalmol.between(-15, 200).all()

# Inter/intra not all zero
v_inter_ok = not (v[v.success_flag==1].vina_inter_kcalmol == 0).all()
c_inter_ok = not (c[c.success_flag==1].covvina_inter_kcalmol == 0).all()

# File integrity spot-check on 5 random cofolds
random.seed(THRESH)
cofs = list((ROOT / "data/boltz_results/cohort_3597_full").glob("from_*/*"))
cofs = [d for d in cofs if d.is_dir() and (d / f"{d.name}_model_0.cif").exists()]
sample = random.sample(cofs, min(5, len(cofs)))
file_ok_n = 0
for d in sample:
    cif = d / f"{d.name}_model_0.cif"
    lig = d / f"{d.name}_model_0.lig.sdf"
    if cif.exists() and cif.stat().st_size > 1000 and lig.exists() and lig.stat().st_size > 200:
        # Quick acrylamide check: contains C=C-C(=O)-N pattern by SDF heuristic
        # (just check the file is readable and not empty).
        file_ok_n += 1

passes = sum([
    v_cols_ok, c_cols_ok,
    not v_nan, not c_nan,
    v_succ >= 0.90, c_succ >= 0.90,
    v_drift_ok, c_drift_ok,
    v_range_ok, c_range_ok,
    v_inter_ok, c_inter_ok,
    file_ok_n >= 4,
])

note = (
    f"thresh={THRESH}  vanilla {len(v)} rows ({100*v_succ:.1f}% ok, med={v_med:.2f})  "
    f"covvina {len(c)} rows ({100*c_succ:.1f}% ok, med={c_med:.2f})  "
    f"schema_ok=({v_cols_ok},{c_cols_ok})  drift_ok=({v_drift_ok},{c_drift_ok})  "
    f"file_spot={file_ok_n}/5"
)
verdict = "PASS" if passes == 13 else "PARTIAL"
print(f"QA {verdict}  passes={passes}/13  {note}")
with open("/tmp/vina_pipeline_qa_log.txt", "a") as f:
    f.write(f"{ts}  threshold_{THRESH}  {verdict}  {note}\n")

# Heartbeat update
hb_path = Path("/tmp/vina_pipeline_master.json")
hb = json.loads(hb_path.read_text())
hb.update({
    "ts": ts,
    "phase": f"qa_{THRESH}",
    "boltz_total": int((v.success_flag==1).sum()),
    "vanilla_rescored": int((v.success_flag==1).sum()),
    "covvina_rescored": int((c.success_flag==1).sum()),
    "qa_passes": hb.get("qa_passes", 0) + (1 if verdict == "PASS" else 0),
    "qa_failures": hb.get("qa_failures", 0) + (0 if verdict == "PASS" else 1),
    "last_qa_note": note,
    "next_threshold": {2000: 3000, 3000: 3597, 3597: None}.get(THRESH, hb.get("next_threshold")),
})
hb_path.write_text(json.dumps(hb, indent=2))
print(f"heartbeat updated: phase={hb['phase']}")
PYEOF

echo "[fire $THRESH] done" | tee -a "$LOG"
