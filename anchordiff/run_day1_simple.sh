#!/bin/bash
# Day 1 (simplified) — vanilla DiffSBDD generation on ZAP70 + BTK pockets.
# This is the BASELINE cohort. Inpaint + constraint-projected variants will
# follow once we verify the basic pipeline works and the outputs look sensible.
set -e
source ~/miniconda3/etc/profile.d/conda.sh 2>/dev/null || \
  source /opt/conda/etc/profile.d/conda.sh
conda activate diffsbdd

BASE=~/anchordiff_results/day1_simple
mkdir -p $BASE
cd ~/DiffSBDD

CKPT=~/DiffSBDD/checkpoints/crossdocked_fullatom_cond.ckpt
N_SAMPLES=100

# Verify checkpoint loaded
if [ ! -f "$CKPT" ] || [ "$(stat -c%s $CKPT 2>/dev/null || echo 0)" -lt 100000000 ]; then
  echo "ERROR: checkpoint missing or incomplete: $CKPT"
  ls -lh $CKPT 2>&1
  exit 1
fi

# ── ZAP70 ─────────────────────────────────────────────────────────────────
echo "[$(date +%H:%M:%S)] ZAP70 (4K2R) — vanilla DiffSBDD, $N_SAMPLES mols"
mkdir -p $BASE/zap70
# Use imatinib (residue STI in 4K2R, chain A) as the reference ligand to define the pocket.
# imatinib's residue number in 4K2R: A:701 (HETATM STI 701 from earlier inspection)
python generate_ligands.py $CKPT \
  --pdbfile ~/anchordiff/pockets/zap70/receptor.pdb \
  --ref_ligand A:701 \
  --outfile $BASE/zap70/vanilla.sdf \
  --n_samples $N_SAMPLES \
  --batch_size 50 \
  --sanitize 2>&1 | tee $BASE/zap70/vanilla.log | tail -10

echo "[$(date +%H:%M:%S)] ZAP70 vanilla: $(grep -c '\$\$\$\$' $BASE/zap70/vanilla.sdf 2>/dev/null || echo 0) mols generated"

# ── BTK ───────────────────────────────────────────────────────────────────
# BTK 5P9J binds to ibrutinib (residue 1AZ in 5P9J). Find the residue and chain.
echo "[$(date +%H:%M:%S)] BTK (5P9J) — vanilla DiffSBDD, $N_SAMPLES mols"
mkdir -p $BASE/btk
# Search 5P9J for the ibrutinib residue
LIG_RES=$(grep "^HETATM" ~/anchordiff/pockets/btk/receptor.pdb | awk '{print $4 ":" $5 ":" $6}' | sort -u | grep -v "HOH\|SO4\|NA\| MG\|CL " | head -1)
echo "  detected BTK ligand: $LIG_RES"
# Extract chain:resi format
CHAIN_RES=$(echo $LIG_RES | awk -F: '{print $2 ":" $3}')

python generate_ligands.py $CKPT \
  --pdbfile ~/anchordiff/pockets/btk/receptor.pdb \
  --ref_ligand $CHAIN_RES \
  --outfile $BASE/btk/vanilla.sdf \
  --n_samples $N_SAMPLES \
  --batch_size 50 \
  --sanitize 2>&1 | tee $BASE/btk/vanilla.log | tail -10

echo "[$(date +%H:%M:%S)] BTK vanilla: $(grep -c '\$\$\$\$' $BASE/btk/vanilla.sdf 2>/dev/null || echo 0) mols generated"

echo ""
echo "[$(date +%H:%M:%S)] === DAY 1 SIMPLE COMPLETE ==="
ls -la $BASE/zap70/ $BASE/btk/
