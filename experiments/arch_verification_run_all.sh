#!/usr/bin/env bash
# Run all 8 arch_verification recipes sequentially on ai-gpu.
# Each recipe writes ONE row to data/arch_verification/results.csv.
# NOTE: we do NOT set -e — a single recipe failure should not halt the batch.
set -uo pipefail

cd ~/edit-small-mol
PY=~/miniconda3/envs/quris/bin/python
LOGDIR=data/arch_verification/logs
mkdir -p "$LOGDIR"

# Shared args
SAMPLE_N=500
TEMP=1.0

run_recipe() {
  local name="$1"; shift
  echo "=== [$(date '+%H:%M:%S')] START recipe=$name ==="
  $PY experiments/arch_verification_runner.py --recipe "$name" "$@" 2>&1 | tee "$LOGDIR/${name}.log" || echo "RECIPE_FAILED: $name"
  echo "=== [$(date '+%H:%M:%S')] END   recipe=$name ==="
}

# Recipe 1: v2_pairs_only (20 ep, LR 5e-5, bs 32) — replicate Claude's baseline
run_recipe v2_pairs_only \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/optionA/v2_pairs_scheme_A_1286.csv \
  --epochs 20 --lr 5e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "replicate Claude's baseline"

# Recipe 2: v2_pairs_more_epochs (60 ep, LR 5e-5, bs 32)
run_recipe v2_pairs_more_epochs \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/optionA/v2_pairs_scheme_A_1286.csv \
  --epochs 60 --lr 5e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "3x longer training"

# Recipe 3: v2_pairs_higher_lr (20 ep, LR 1e-4, bs 32)
run_recipe v2_pairs_higher_lr \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/optionA/v2_pairs_scheme_A_1286.csv \
  --epochs 20 --lr 1e-4 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "2x LR"

# Recipe 4: v2_pairs_lower_lr_longer (100 ep, LR 1e-5, bs 32)
run_recipe v2_pairs_lower_lr_longer \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/optionA/v2_pairs_scheme_A_1286.csv \
  --epochs 100 --lr 1e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "gentle long training"

# Recipe 5: v2_pairs_random_scaffold_pairing (20 ep, LR 5e-5)
run_recipe v2_pairs_random_scaffold_pairing \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/arch_verification/pairs/recipe5_scaffold_hop_pairs.csv \
  --epochs 20 --lr 5e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "scaffold-hop pairs (not Tc)"

# Recipe 6: v1_pairs_only (base mol2mol_medium_similarity, no covFT)
run_recipe v1_pairs_only \
  --prior_ckpt /home/shaharh_quris_ai/REINVENT4/priors/mol2mol_medium_similarity.prior \
  --train_csv data/optionA/v2_pairs_scheme_A_1286.csv \
  --epochs 20 --lr 5e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "base mol2mol prior (no covFT)"

# Recipe 7: v1_covaFT + v2_pairs + Mol1_anchored_augmentation (30 ep, LR 5e-5)
run_recipe v1_covaFT_mol1_aug \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/arch_verification/pairs/recipe7_v2_plus_mol1_aug.csv \
  --epochs 30 --lr 5e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "v2+500 (src, Mol1) aug pairs, 30ep"

# Recipe 8: massive Mol1-aug (30 ep, LR 5e-5)
run_recipe v1_covaFT_massive_mol1_aug \
  --prior_ckpt models/reinvent4_mol2mol_covalent_ft.prior \
  --train_csv data/arch_verification/pairs/recipe8_v2_plus_massive_mol1.csv \
  --epochs 30 --lr 5e-5 --bs 32 --n_samples $SAMPLE_N --temperature $TEMP \
  --notes "v2+5000 top-Tc (src, Mol1) pairs, 30ep"

echo "ALL RECIPES DONE at $(date)"
