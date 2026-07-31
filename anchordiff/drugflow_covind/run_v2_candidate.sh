#!/bin/bash
# Run one v2 candidate end-to-end on V100:
#   $1 = candidate tag (e.g. F, F_unfrozen, G)
#   $2 = extra train flags (quoted)
#   $3 = N samples (default 25)
#
# Trains 1 epoch, samples N mols, copies SDF to results dir.
# Usage:
#   run_v2_candidate.sh F "--freeze_bond_head --no_jitter" 25
#   run_v2_candidate.sh F_unfrozen "--no_jitter" 25
#   run_v2_candidate.sh G "--freeze_bond_head --jitter_sigma 0.03" 25

set -e
TAG="$1"
EXTRA="$2"
N="${3:-25}"
RUN_DIR="$HOME/runs/drugflow_dc_v2_$TAG"
RESULTS_DIR="$HOME/results/covalent_gen_day1/drugflow_dc_v2_$TAG"

source ~/miniconda3/etc/profile.d/conda.sh
conda activate drugflow
export LD_PRELOAD=/home/shaharh_quris_ai/miniconda3/envs/drugflow/lib/libstdc++.so.6

mkdir -p "$RUN_DIR" "$RESULTS_DIR"
cd ~/edit-small-mol

if [ ! -f "$RUN_DIR/best.pt" ]; then
  echo "[$TAG] Training..."
  python -m anchordiff.drugflow_covind.train_dc_v2 \
    --epochs 1 --batch_size 4 --lr 5e-6 \
    $EXTRA \
    --out_dir "$RUN_DIR" 2>&1 | tail -30
else
  echo "[$TAG] Using existing $RUN_DIR/best.pt"
fi

if [ ! -f "$RUN_DIR/best.pt" ]; then
  echo "[$TAG] Training did not produce best.pt — aborting"
  exit 1
fi

echo "[$TAG] Sampling N=$N..."
python -m anchordiff.drugflow_covind.sample \
  --ft_ckpt "$RUN_DIR/best.pt" \
  --receptor ~/edit-small-mol/anchordiff/pockets/zap70_cys346/receptor.pdb \
  --ref_ligand ~/edit-small-mol/anchordiff/pockets/zap70_cys346/ref_ligand.sdf \
  --warhead ~/edit-small-mol/anchordiff/pockets/zap70_cys346/warhead.sdf \
  --datadir ~/DrugFlow/src/default \
  --n_samples $N --batch_size 25 \
  --inpaint --strength_pos 0.7 --strength_h 5.0 --strength_e 5.0 \
  --output "$RESULTS_DIR/samples.sdf" 2>&1 | tail -10

echo "[$TAG] Done. SDF: $RESULTS_DIR/samples.sdf"
ls -la "$RESULTS_DIR/" 2>&1
