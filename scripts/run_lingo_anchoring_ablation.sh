#!/bin/bash
# Lingo3DMol anchoring ablation: test less-aggressive anchoring strategies
# at fixed T=1.5 (best diversity from temperature ablation), H2 anchor,
# L1 FT v1 ckpt. Each config produces 100 mols (min_acceptable) of up to
# 200 (gennums).
#
# Configs (5; SOFT_COORD is skipped — see report for rationale):
#   FULL              — baseline (pin 50 prefix tokens + coords)
#   BC_ONLY           — pin only [start_0, C_0] (β-C voxel)
#   BC_PLUS_4         — pin acrylamide core (atoms 0..4 = C=C-C(=O)-N)
#   SMI_ONLY          — pin all 50 token codes, coords = -1 sentinel (float)
#   BC_NOCONSTRAINT   — 2-token code pin, coords = -1 sentinel (no spatial bias)

set -uo pipefail
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lingo3dmol
cd ~/edit-small-mol

ROOT=/home/shaharh_quris_ai/edit-small-mol
POCKET=$ROOT/data/lingo3dmol_smoke/zap70_pocket_cys346.pdb
ANCHOR=$ROOT/data/lingo3dmol_anchor_zap70_cys346.json
CONTACT=$ROOT/external/Lingo3DMol/checkpoint/contact.pkl
# L1 FT v1 ckpt (matches what the temperature ablation uses; the
# `~/edit-small-mol/data/lingo3dmol_L1_full_ft/ckpt_phase2_dev.pt` path
# referenced in the spec does not exist on V100 — covlingo_full_v1 is the
# v1 of the same model in production here).
FT_CKPT=$ROOT/data/covlingo_full_v1/ckpt_phase2_dev.pt
OUT_BASE=$ROOT/data/lingo_anchoring_ablation
LOG=$HOME/lingo_anchoring_ablation.log

mkdir -p $OUT_BASE
echo "[anchor-ablation] start $(date)" | tee -a $LOG

# Flatten checkpoint once (reuse temperature-ablation flat if it exists)
FLAT_CKPT=$ROOT/data/lingo_sampling_ablation/flat_ft.pkl
if [ ! -f "$FLAT_CKPT" ]; then
  FLAT_CKPT=$OUT_BASE/flat_ft.pkl
  python - "$FT_CKPT" "$FLAT_CKPT" << 'PYEOF'
import sys, torch
src, dst = sys.argv[1], sys.argv[2]
ck = torch.load(src, map_location='cpu', weights_only=False)
sd = ck['model'] if isinstance(ck, dict) and 'model' in ck else ck
torch.save(sd, dst)
print('[flat]', len(sd), 'keys ->', dst)
PYEOF
fi
echo "[anchor-ablation] using flat ckpt $FLAT_CKPT" | tee -a $LOG

run_one() {
  local MODE=$1
  local OUT_DIR=$OUT_BASE/$MODE
  local OUT=$OUT_DIR/samples.sdf
  mkdir -p $OUT_DIR

  if [ -f "$OUT" ]; then
    local n=$(grep -c '^\$\$\$\$' "$OUT" 2>/dev/null || echo 0)
    if [ "$n" -ge 100 ]; then
      echo "[anchor-ablation $MODE] SKIP (already has $n mols)" | tee -a $LOG
      return 0
    fi
    echo "[anchor-ablation $MODE] resume — found $n existing mols" | tee -a $LOG
  fi

  echo "[anchor-ablation $MODE] starting at $(date)" | tee -a $LOG
  timeout 3900 python -u experiments/run_lingo3dmol_l2_extended_anchor.py \
    --pocket_pdb $POCKET \
    --output $OUT \
    --anchor_id H2 \
    --anchor_mode $MODE \
    --anchor_json $ANCHOR \
    --contact_path $CONTACT \
    --caption_path $FLAT_CKPT \
    --gennums 200 \
    --min_acceptable 100 \
    --gen_frag_set 10 \
    --prod_time 2 \
    --topk 5 \
    --tempture 1.5 \
    --max_run_seconds 3600 \
    > $OUT_DIR/sample.log 2>&1 || echo "[anchor-ablation $MODE] timeout/err exit=$?" | tee -a $LOG

  if [ -f "$OUT" ]; then
    local nf=$(grep -c '^\$\$\$\$' "$OUT" 2>/dev/null || echo 0)
    local sz=$(stat -c%s "$OUT" 2>/dev/null || echo 0)
    echo "[anchor-ablation $MODE] done at $(date) — produced $nf mols, sdf_size=$sz" | tee -a $LOG
  else
    echo "[anchor-ablation $MODE] done at $(date) — NO SDF" | tee -a $LOG
  fi
}

# 5 configs (sequential).
run_one FULL
run_one BC_ONLY
run_one BC_PLUS_4
run_one SMI_ONLY
run_one BC_NOCONSTRAINT

echo "[anchor-ablation] all done $(date)" | tee -a $LOG
