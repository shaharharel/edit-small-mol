#!/bin/bash
# Run Fukui f+ on 10 P5 methods × 500 Mol1 mols on ai-chem.
# Called on remote. Uses quris env's xtb.
set -e
cd /home/shaharh_quris_ai/edit-small-mol
mkdir -p data/exp_P5/fukui
LOG=data/exp_P5/fukui/run.log
: > $LOG
source /home/shaharh_quris_ai/miniconda3/etc/profile.d/conda.sh
conda activate quris

METHODS=(
  mol2mol_baseline
  v2_cond_denoise
  v2_cond_denoise_dap
  v2_cond_with_zap70
  v2_cond_dap_min_strict
  v2_cond_P5-1-v2
  v2_cond_P5-CROSSPDB-v2
  v2_cond_P5-A-v2
  v2_cond_P5-4-v2
  v2_cond_P5-HYBRID-v2
)

for m in "${METHODS[@]}"; do
  SMI=data/exp_P5/covvina/${m}_input.smi
  OUT=data/exp_P5/fukui/${m}_fukui.csv
  if [[ ! -f "$SMI" ]]; then
    echo "MISSING input: $SMI" | tee -a $LOG
    continue
  fi
  if [[ -f "$OUT" ]]; then
    echo "SKIP done: $OUT ($(wc -l < $OUT))" | tee -a $LOG
    continue
  fi
  echo ">>> $m  ($(date -u +%H:%M:%S))" | tee -a $LOG
  python experiments/fukui_smi_batch.py "$SMI" "$OUT" --workers 30 2>&1 | tee -a $LOG
  echo "<<< $m done  ($(date -u +%H:%M:%S)) rows=$(wc -l < $OUT)" | tee -a $LOG
done

touch data/exp_P5/fukui/DONE.flag
echo "ALL DONE ($(date -u +%H:%M:%S))" | tee -a $LOG
