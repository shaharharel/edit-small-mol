#!/usr/bin/env bash
# Driver: run 7 BD-corrected N=500 cohorts sequentially on T4.
# Logs each cohort's status, BD-angle smoke, and eval.
set -uo pipefail
cd ~/edit-small-mol
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lingo3dmol

POCKET=data/lingo3dmol_smoke/zap70_pocket_cys346.pdb
OUTROOT=data/lingo3dmol_BD_corrected
MASTER_LOG=${OUTROOT}/_driver.log
SUMMARY=${OUTROOT}/_summary.json
L1_FT_CKPT=$HOME/edit-small-mol/data/lingo3dmol_L1_v3_FIXED/ckpt_phase2_dev.pt

mkdir -p "$OUTROOT"
echo "[driver] start $(date -Iseconds)" | tee -a "$MASTER_LOG"

run_cohort () {
  local NAME="$1"
  local CMD="$2"
  local DIR="${OUTROOT}/${NAME}"
  local LOG="${DIR}/run.log"
  local SDF="${DIR}/samples.sdf"
  local EVALJSON="${DIR}/eval.json"
  mkdir -p "$DIR"
  echo "[driver] === ${NAME} start $(date -Iseconds) ===" | tee -a "$MASTER_LOG"
  T0=$(date +%s)
  bash -c "$CMD" > "$LOG" 2>&1
  RC=$?
  T1=$(date +%s)
  DUR=$((T1-T0))
  BD_LINE=$(grep -E "BD angle at atom0" "$LOG" | head -1 || echo "NO_BD_LINE")
  N_MOLS=0
  if [ -f "$SDF" ]; then
    N_MOLS=$(grep -c "^\\$\\$\\$\\$" "$SDF" 2>/dev/null || echo 0)
  fi
  echo "[driver] ${NAME} rc=${RC} dur=${DUR}s n_mols=${N_MOLS} ${BD_LINE}" | tee -a "$MASTER_LOG"
  # eval
  if [ -f "$SDF" ] && [ "${N_MOLS}" -gt 0 ]; then
    python experiments/eval_lingo3dmol_plans.py --input "$SDF" --tag "$NAME" --out "$EVALJSON" > "${DIR}/eval.log" 2>&1
    EVAL_RC=$?
    if [ ! -f "$EVALJSON" ] && [ -f "results/lingo3dmol/${NAME}_eval.json" ]; then
      cp "results/lingo3dmol/${NAME}_eval.json" "$EVALJSON"
    fi
    echo "[driver] ${NAME} eval rc=${EVAL_RC} json=$([ -f "$EVALJSON" ] && echo OK || echo MISSING)" | tee -a "$MASTER_LOG"
  else
    echo "[driver] ${NAME} no SDF or empty; skipping eval" | tee -a "$MASTER_LOG"
  fi
  echo "[driver] === ${NAME} end ===" | tee -a "$MASTER_LOG"
}

# Cohort 1: L2_scaff_C5_BD_N500
run_cohort "L2_scaff_C5_BD_N500" "python experiments/run_lingo3dmol_l2_scaffold_C5_inpaint.py \
  --pocket_pdb ${POCKET} \
  --output ${OUTROOT}/L2_scaff_C5_BD_N500/samples.sdf \
  --gennums 500 --gen_frag_set 10 --prod_time 3 \
  --coc_dis 0.5 --min_acceptable 200 --max_run_seconds 3600 --tempture 1.0"

# Cohort 2: L2_ext_H2_BD_N500
run_cohort "L2_ext_H2_BD_N500" "python experiments/run_lingo3dmol_l2_extended_anchor.py \
  --pocket_pdb ${POCKET} \
  --anchor_id H2 --output ${OUTROOT}/L2_ext_H2_BD_N500/samples.sdf \
  --tempture 1.0 --gennums 500 --min_acceptable 200 --max_run_seconds 3600"

# Cohort 3: L2_ext_H1_BD_N500
run_cohort "L2_ext_H1_BD_N500" "python experiments/run_lingo3dmol_l2_extended_anchor.py \
  --pocket_pdb ${POCKET} \
  --anchor_id H1 --output ${OUTROOT}/L2_ext_H1_BD_N500/samples.sdf \
  --tempture 1.0 --gennums 500 --min_acceptable 200 --max_run_seconds 3600"

# Cohort 4: L2_ext_H3_BD_N500
run_cohort "L2_ext_H3_BD_N500" "python experiments/run_lingo3dmol_l2_extended_anchor.py \
  --pocket_pdb ${POCKET} \
  --anchor_id H3 --output ${OUTROOT}/L2_ext_H3_BD_N500/samples.sdf \
  --tempture 1.0 --gennums 500 --min_acceptable 200 --max_run_seconds 3600"

# Cohort 5: L2_ext_L_BD_N500
run_cohort "L2_ext_L_BD_N500" "python experiments/run_lingo3dmol_l2_extended_anchor.py \
  --pocket_pdb ${POCKET} \
  --anchor_id L --output ${OUTROOT}/L2_ext_L_BD_N500/samples.sdf \
  --tempture 1.0 --gennums 500 --min_acceptable 200 --max_run_seconds 3600"

# Cohort 6: L2_scaff_C1_BD_N500
run_cohort "L2_scaff_C1_BD_N500" "python experiments/run_lingo3dmol_l2_scaffold_inpaint.py \
  --pocket_pdb ${POCKET} \
  --output ${OUTROOT}/L2_scaff_C1_BD_N500/samples.sdf \
  --gennums 500 --gen_frag_set 10 --prod_time 3 \
  --coc_dis 0.5 --min_acceptable 200 --max_run_seconds 3600 --tempture 1.0"

# Cohort 7: L1_FT_H2_BD_N500 — uses L1 FT ckpt
run_cohort "L1_FT_H2_BD_N500" "python experiments/run_lingo3dmol_l2_extended_anchor.py \
  --pocket_pdb ${POCKET} \
  --anchor_id H2 \
  --caption_path ${L1_FT_CKPT} \
  --output ${OUTROOT}/L1_FT_H2_BD_N500/samples.sdf \
  --tempture 1.0 --gennums 500 --min_acceptable 200 --max_run_seconds 3600"

echo "[driver] all cohorts done $(date -Iseconds)" | tee -a "$MASTER_LOG"
