#!/bin/bash
# Sequential arm queue. One param per arm, each fine-tuning the mol2mol prior.
# Order: the two conditioning modes of a param run ADJACENTLY so that if the queue is
# interrupted we still hold a matched (instr, none) pair rather than an unpaired arm --
# an unpaired instr arm cannot be read as steering, only as "a model exists".
cd "$(dirname "$0")/.."
P=/opt/miniconda3/envs/quris/bin/python
mkdir -p ckpt_steer_v5 logs
for PARAM in warhead_planarity acyl_N_motif linker_atom_count; do
  for MODE in instr none; do
    OUT=ckpt_steer_v5/${PARAM}_${MODE}
    if [ -f "${OUT}_history.json" ]; then echo "SKIP $OUT (done)"; continue; fi
    echo "=== $(date +%T) $PARAM / $MODE ==="
    $P -u stage0/train_steer_v5.py --param "$PARAM" --mode "$MODE" \
       --out "$OUT" --epochs 2 --bs 64 --seed 20260916 \
       > logs/arm_${PARAM}_${MODE}.log 2>&1
    tail -3 logs/arm_${PARAM}_${MODE}.log
  done
done
echo "=== LIGAND ARMS DONE $(date +%T) ==="
