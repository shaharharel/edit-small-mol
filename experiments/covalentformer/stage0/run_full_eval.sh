#!/bin/bash
# FULL EVALUATION SWEEP -- 5k cohorts for every steering arm, one shared baseline.
#
# THE POINT: produce ONE table covering every steering param, where each row answers
# "does this instruction move the generated cohort, and at what cost to the molecules".
#
# WHY THE BASELINE IS GENERATED ONCE. The mol2mol PRIOR cohort does not depend on which
# param is being steered -- it is the same 5,000 anchors through the same unconditioned
# model. Generating it per-param would be 8x the compute for identical numbers, and worse,
# a per-param baseline drawn with a different seed would make cross-param comparison a
# seed contrast rather than a param contrast. One baseline, one seed, reused everywhere.
#
# THREE REFERENCE POINTS PER PARAM, because "better than baseline" is ambiguous:
#   prior  raw mol2mol                -- no covalent fine-tuning at all
#   none   covalent-FT, no token      -- isolates the INSTRUCTION from the FINE-TUNING
#   instr  steered, UP and DOWN       -- the thing being claimed
# The manuscript's planarity effect was measured against the PRIOR. An effect that only
# beats the prior may be entirely the covalent fine-tuning that `none` already has.
#
# ORDER MATTERS: the proven param goes FIRST. If the harness is broken, role is where it
# shows up, because role is the one param with a known answer to check against
# (82.0%/77.1% obedience vs a ~27% floor). A harness that cannot reproduce role is not
# trustworthy on anything else.
set -u
cd "$(dirname "$0")/.."
P=${PY:-python3}
N=${N:-5000}
SEED=${SEED:-20260916}
OUT=results/full_eval
mkdir -p $OUT logs

PRIOR=../../models/reinvent4_mol2mol_warhead_tokens.prior

echo "############ FULL EVAL  n=$N  seed=$SEED  $(date +%T) ############"

# ---------------------------------------------------------------- 1. ROLE (proven first)
# Per-role cohorts vs the A0 matched control. NEVER POOLED: train is 45x skewed
# (WARHEAD 149,953 vs LINKER 3,335) and valid is deliberately rebalanced, so a pooled
# role figure measures the MIX, not the model.
if [ -f ckpt_A_role_strat/ep2.ckpt ]; then
  echo "=== role (per-role, vs A0 control) ==="
  $P -u stage0/cohort_shift.py --param role \
     --instr-ckpt ckpt_A_role_strat/ep2.ckpt \
     --none-ckpt  ckpt_A0_strat/ep2.ckpt \
     --valid data/roles/valid_strat.csv --n $N --seed $SEED \
     --out $OUT/role.json > logs/eval_role.log 2>&1
  tail -14 logs/eval_role.log
fi

# ------------------------------------------------- 2. LIGAND PARAMS (high data, UP/DOWN)
for PARAM in linker_atom_count acyl_N_motif warhead_planarity; do
  I=ckpt_steer_v5/${PARAM}_instr_best.pt; [ -f "$I" ] || I=ckpt_steer_v5/${PARAM}_instr_ep1.pt
  Nn=ckpt_steer_v5/${PARAM}_none_best.pt;  [ -f "$Nn" ] || Nn=ckpt_steer_v5/${PARAM}_none_ep1.pt
  [ -f "$I" ] || { echo "SKIP $PARAM (no ckpt)"; continue; }
  echo "=== $PARAM ==="
  $P -u stage0/cohort_shift.py --param $PARAM \
     --instr-ckpt "$I" --none-ckpt "$Nn" --prior-ckpt "$PRIOR" \
     --valid data/steer_v5/${PARAM}_valid.csv --n $N --seed $SEED \
     --out $OUT/${PARAM}.json > logs/eval_${PARAM}.log 2>&1
  tail -12 logs/eval_${PARAM}.log
done

# ----------------------------------------------- 3. POCKET PARAMS (small, v7 checkpoints)
for PARAM in buried_sasa pocket_occupancy theta_bd d_cys_scaffold; do
  I=ckpt_v7/${PARAM}_over_instr_best.pt
  Nn=ckpt_v7/${PARAM}_over_none_best.pt
  [ -f "$I" ] || { echo "SKIP $PARAM (no v7 ckpt)"; continue; }
  echo "=== $PARAM (oversample arm) ==="
  $P -u stage0/cohort_shift.py --param $PARAM \
     --instr-ckpt "$I" --none-ckpt "$Nn" --prior-ckpt "$PRIOR" \
     --valid data/steer_v5/${PARAM}_valid.csv --n $N --seed $SEED \
     --out $OUT/${PARAM}.json > logs/eval_${PARAM}.log 2>&1
  tail -12 logs/eval_${PARAM}.log
done

# ------------------------------------ 4. GENERATION + MANUSCRIPT PANEL on the same cohorts
for PARAM in linker_atom_count acyl_N_motif warhead_planarity; do
  I=ckpt_steer_v5/${PARAM}_instr_best.pt; [ -f "$I" ] || I=ckpt_steer_v5/${PARAM}_instr_ep1.pt
  [ -f "$I" ] || continue
  $P -u stage0/generate_and_score.py --param $PARAM --ckpt "$I" \
     --valid data/steer_v5/${PARAM}_valid.csv --train data/steer_v5/${PARAM}_train.csv \
     --n $N --seed $SEED --out $OUT/${PARAM}_panel.json > logs/panel_${PARAM}.log 2>&1
  echo "  panel $PARAM: $(grep -E 'validity|warhead_retention' logs/panel_${PARAM}.log | tr '\n' ' ')"
done

echo "############ DONE $(date +%T) ############"
echo "Summarise with:  python3 stage0/summarise_eval.py $OUT"
