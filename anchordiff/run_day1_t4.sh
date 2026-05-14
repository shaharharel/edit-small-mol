#!/bin/bash
# Day 1 generation on T4 (ai-gpu).
# Produces 4 cohorts × 100 mols per target (ZAP70 + BTK).
# Cohorts:
#   D0_vanilla       — vanilla DiffSBDD generate_ligands (no covalent awareness)
#   D1_inpaint       — DiffSBDD inpaint with warhead atoms FIXED (anchor mask)
#   D1_inpaint_rank  — D1_inpaint sampled 5x, top-100 ranked by FiLMDelta (=C1)
#   D1_proj          — D1_inpaint + post-hoc covalent geometry projection (=C2)
#
# Outputs: ~/anchordiff_results/day1/{target}/{cohort}/*.sdf

set -e
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source /opt/conda/etc/profile.d/conda.sh
conda activate diffsbdd

BASE=~/anchordiff_results/day1
mkdir -p $BASE
cd ~/DiffSBDD

CKPT=~/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt
N_VANILLA=50
N_INPAINT=100
N_INPAINT_RANK_OVERSAMPLE=200  # sample 200, will rank to 100 with FiLMDelta locally

run_target() {
  TGT=$1; PDB=$2; REF_LIG=$3; FIX_ATOMS=$4; CYS_RESI=$5

  OUT=$BASE/$TGT
  mkdir -p $OUT/D0_vanilla $OUT/D1_inpaint $OUT/D1_inpaint_oversample

  echo ""
  echo "=================================================="
  echo "TARGET: $TGT  (Cys$CYS_RESI)"
  echo "=================================================="

  # Cohort 1: vanilla DiffSBDD (no constraint at all)
  echo "[$(date +%H:%M:%S)] D0_vanilla: generating $N_VANILLA mols"
  python generate_ligands.py $CKPT \
    --pdbfile $PDB \
    --ref_ligand $REF_LIG \
    --outfile $OUT/D0_vanilla/mols.sdf \
    --n_samples $N_VANILLA \
    --batch_size $N_VANILLA \
    --sanitize 2>&1 | tail -5

  # Cohort 2: inpaint with warhead fixed
  echo "[$(date +%H:%M:%S)] D1_inpaint: generating $N_INPAINT mols with fixed atoms: $FIX_ATOMS"
  python inpaint.py $CKPT \
    --pdbfile $PDB \
    --ref_ligand $REF_LIG \
    --fix_atoms $FIX_ATOMS \
    --outfile $OUT/D1_inpaint/mols.sdf \
    --n_samples $N_INPAINT \
    --batch_size $N_INPAINT \
    --sanitize 2>&1 | tail -5

  # Cohort 3 (oversample for FiLMDelta ranking): inpaint with N=200
  echo "[$(date +%H:%M:%S)] D1_inpaint_oversample: generating $N_INPAINT_RANK_OVERSAMPLE mols"
  python inpaint.py $CKPT \
    --pdbfile $PDB \
    --ref_ligand $REF_LIG \
    --fix_atoms $FIX_ATOMS \
    --outfile $OUT/D1_inpaint_oversample/mols.sdf \
    --n_samples $N_INPAINT_RANK_OVERSAMPLE \
    --batch_size 50 \
    --sanitize 2>&1 | tail -5

  echo "[$(date +%H:%M:%S)] $TGT done"
}

# ── ZAP70 ─────────────────────────────────────────────────────────────────
# Mol 1 warhead atoms in DiffSBDD's SDF: the first 4 atoms in the embedded SDF
# are the acrylamide (assuming canonical SMILES order C=CC(=O)N...).
# DiffSBDD reads atom names from the SDF via OpenBabel; for a SMILES-embedded
# SDF the names default to "C1", "C2", "C3", "O1", etc. We'll specify by name.
run_target "zap70" \
  ~/anchordiff/pockets/zap70/receptor.pdb \
  ~/anchordiff/pockets/zap70/ref_ligand.sdf \
  "C1 C2 C3 O1 N1" \
  560

# ── BTK ───────────────────────────────────────────────────────────────────
run_target "btk" \
  ~/anchordiff/pockets/btk/receptor.pdb \
  ~/anchordiff/pockets/btk/ref_ligand.sdf \
  "C1 C2 C3 O1 N1" \
  481

echo ""
echo "[$(date +%H:%M:%S)] === Day 1 generation COMPLETE ==="
echo "Outputs:"
find $BASE -name '*.sdf' | xargs -I {} sh -c 'echo "  {} : $(grep -c \"\$\$\$\$\" {}) mols"'
