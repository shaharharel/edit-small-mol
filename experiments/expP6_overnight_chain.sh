#!/bin/bash
# P6 overnight chain: 4 more RL/POSENOISE variants, sample, and score.
set -u
cd /home/shaharh_quris_ai/edit-small-mol
source /home/shaharh_quris_ai/miniconda3/etc/profile.d/conda.sh
conda activate quris

LOG=data/exp_P6/rl_logs/overnight_chain.log
: > $LOG
log() { echo "[$(date -u +%H:%M:%S)] $*" | tee -a $LOG; }

# =====================================================
# STAGE 1: RL variants (train then sample) — sequential
# =====================================================

# RL-SHORT (100 steps): early-stopping to see if it's more diverse
log "STAGE 1a: RL-SHORT (100 steps) from P5-1-v2"
python experiments/m1a_pocket_decoder/v2/fixes/train_m1a_v2_dap_MIN_planar_STRICT.py \
    --base_ckpt data/exp_P5/models/P5-1_v2/m1a_v2_final.ckpt \
    --out_ckpt models/exp_P6/RL_SHORT_from_P51v2.ckpt \
    --log_json data/exp_P6/rl_logs/RL_SHORT.json --max_steps 100 \
    >> data/exp_P6/rl_logs/RL_SHORT.stdout 2>&1
python experiments/m1a_pocket_decoder/v2/fixes/sample_m1a_v2_dap.py \
    --ckpt models/exp_P6/RL_SHORT_from_P51v2.ckpt --n 500 --batch_size 32 \
    --out_csv data/exp_P6/samples/RL_SHORT_from_P51v2_mol1.csv >> $LOG 2>&1
log "STAGE 1a done"

# RL-GEOMEAN (base variant, no MIN — softer aggregation)
log "STAGE 1b: RL-GEOMEAN (base DAP, 250 steps) from P5-1-v2"
python experiments/m1a_pocket_decoder/v2/fixes/train_m1a_v2_dap.py \
    --base_ckpt data/exp_P5/models/P5-1_v2/m1a_v2_final.ckpt \
    --out_ckpt models/exp_P6/RL_GEOMEAN_from_P51v2.ckpt \
    --log_json data/exp_P6/rl_logs/RL_GEOMEAN.json --max_steps 250 \
    >> data/exp_P6/rl_logs/RL_GEOMEAN.stdout 2>&1
python experiments/m1a_pocket_decoder/v2/fixes/sample_m1a_v2_dap.py \
    --ckpt models/exp_P6/RL_GEOMEAN_from_P51v2.ckpt --n 500 --batch_size 32 \
    --out_csv data/exp_P6/samples/RL_GEOMEAN_from_P51v2_mol1.csv >> $LOG 2>&1
log "STAGE 1b done"

# =====================================================
# STAGE 2: P6-POSENOISE variants (σ=0.5, σ=1.0)
# =====================================================
for SIGMA in 0.5 1.0; do
    LABEL="P6-POSENOISE-s${SIGMA/./}"
    log "STAGE 2: $LABEL (pose_noise_std=$SIGMA)"
    python experiments/train_m1a_v2_pairs_posenoise.py \
        --pairs_pkl data/exp_P5/pairs_P5-1_within.pkl \
        --base_ckpt models/m1a_v2.ckpt \
        --out_dir data/exp_P6/models/$LABEL \
        --epochs 10 --lr 3e-5 --batch_size 16 --pose_noise_std $SIGMA \
        --ckpt_interval 5000 \
        --progress_path data/exp_P6/rl_logs/${LABEL}_progress.json \
        >> data/exp_P6/rl_logs/${LABEL}.stdout 2>&1
    python experiments/m1a_pocket_decoder/v2/fixes/sample_m1a_v2_dap.py \
        --ckpt data/exp_P6/models/$LABEL/m1a_v2_final.ckpt --n 500 --batch_size 32 \
        --out_csv data/exp_P6/samples/${LABEL}_mol1.csv >> $LOG 2>&1
    log "STAGE 2 $LABEL done"
done

# =====================================================
# STAGE 3: Cov-Vina + Fukui on all new cohorts (deferred)
# For now just prep SMI files for downstream cov-Vina + Fukui.
# =====================================================
mkdir -p data/exp_P6/smi_inputs
for f in data/exp_P6/samples/*_mol1*.csv; do
    base=$(basename $f .csv)
    python -c "
import pandas as pd
df = pd.read_csv('$f')
with open('data/exp_P6/smi_inputs/${base}.smi','w') as out:
    for s in df['SMILES'].dropna().astype(str):
        out.write(s.strip() + '\n')
" 2>/dev/null
done

log "ALL P6 OVERNIGHT VARIANTS DONE"
touch data/exp_P6/ALL_DONE.flag
