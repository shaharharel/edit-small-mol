#!/bin/bash
# v4 covalent-only: five task types in one re-weighted corpus, leak-free siamese dpIC50 head.
#
# CORPUS: v4_train_14h.jsonl -- 235,795 rows, residue-level pocket block (20 residues x 4 numbers,
# 1,377 tok/row vs 2,469 for atom-level), contacts capped at 10k, affinity at 80k, rung3 x3.
# 67.8M tokens. Rung 3 is now the MOLECULE-DISJOINT TRAIN SPLIT (4,501 of 5,518) and every other
# corpus is decontaminated against the 1,017-row test fold with a working filter -- build_joint's
# clean() silently kept everything because it parsed formatted outputs as whole SMILES. Priced on THIS run's measured throughput (374M tok <-> 77.3 h), that is ~14 h --
# the only package that yields a complete, LR-annealed model in one night on one A100-40GB.
#
# bs 3 x accum 11 = effective batch 33 (bs 4 OOM'd in backward at maxlen 3072), unchanged from v3 so the LR schedule stays comparable.
# bs was 2 while the trainer requested output_hidden_states=True and retained all 29 hidden states
# (36.7 of 41 GB at maxlen 3072, >=11.7 s/step, 56 h per epoch). With the final-norm forward hook
# only one hidden state is kept, which is what buys the larger micro-batch.
# expandable_segments cuts the 2.45 GB that sat reserved-but-unallocated in the OOM report.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd /home/shaharh_quris_ai
python3 train_v4_head.py \
  --train v4/v4_train_14h_new.jsonl \
  --out v4_out \
  --maxlen 3072 \
  --bs 3 --accum 11 \
  --epochs 1.0 \
  --ckpt-every 250 \
  --auto-resume \
  > v4_train.log 2>&1
