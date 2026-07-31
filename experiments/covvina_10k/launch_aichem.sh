#!/usr/bin/env bash
# Push runner + SMILES to both ai-chem VMs and launch cov-Vina 10k for 4 cohorts.
#   VM1 (ai-chem):  base (10k), rl (10k)
#   VM2 (ai-chem2): covft (10k), v2fixed (5k)
# Runs the two cohorts sequentially per VM (32 workers each).
# Total expected wall time ~35-45 min per VM at ~1.5 s/mol × 32 workers.
set -euo pipefail

VM1_IP="34.24.15.172"   # ai-chem
VM2_IP="34.24.176.40"   # ai-chem2
SSH="ssh -o StrictHostKeyChecking=no -i /Users/shaharharel/.ssh/google_compute_engine shaharh_quris_ai"
SCP="scp -o StrictHostKeyChecking=no -i /Users/shaharharel/.ssh/google_compute_engine"

REMOTE_ROOT="/home/shaharh_quris_ai/edit-small-mol"
REMOTE_DIR="${REMOTE_ROOT}/experiments/covvina_10k"

echo "=== Ensure remote dirs ==="
for IP in $VM1_IP $VM2_IP; do
  $SSH@$IP "mkdir -p ${REMOTE_DIR}/inputs ${REMOTE_ROOT}/results/paper_evaluation ${REMOTE_ROOT}/logs"
done

echo "=== Push runner + inputs ==="
LOCAL_ROOT="/Users/shaharharel/Documents/github/edit-small-mol"
BASE_CSV="${LOCAL_ROOT}/experiments/exp_covft_value/samples_base.csv"
COVFT_CSV="${LOCAL_ROOT}/experiments/exp_covft_value/samples_covft.csv"
RL_CSV="${LOCAL_ROOT}/experiments/exp_geom_bc/samples_rl.csv"
V2_CSV="${LOCAL_ROOT}/data/m1a_v2_ablation/cohort_A.csv"
RUNNER="${LOCAL_ROOT}/experiments/covvina_10k/run_covvina_10k.py"

for IP in $VM1_IP $VM2_IP; do
  $SCP $RUNNER shaharh_quris_ai@$IP:${REMOTE_DIR}/run_covvina_10k.py
done

# VM1 inputs
$SCP $BASE_CSV shaharh_quris_ai@$VM1_IP:${REMOTE_DIR}/inputs/samples_base.csv
$SCP $RL_CSV   shaharh_quris_ai@$VM1_IP:${REMOTE_DIR}/inputs/samples_rl.csv

# VM2 inputs
$SCP $COVFT_CSV shaharh_quris_ai@$VM2_IP:${REMOTE_DIR}/inputs/samples_covft.csv
$SCP $V2_CSV    shaharh_quris_ai@$VM2_IP:${REMOTE_DIR}/inputs/cohort_A.csv

echo "=== Launch VM1 (ai-chem): base + rl ==="
$SSH@$VM1_IP "cd ${REMOTE_DIR} && nohup bash -c '\
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate quris && \
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && \
  python run_covvina_10k.py --smiles inputs/samples_base.csv --cohort base \
    --out ${REMOTE_ROOT}/results/paper_evaluation/covft_geometric_covvina_base_10k.csv \
    --workers 32 --checkpoint-every 200 > ${REMOTE_ROOT}/logs/covvina_10k_base.log 2>&1 && \
  python run_covvina_10k.py --smiles inputs/samples_rl.csv --cohort rl \
    --out ${REMOTE_ROOT}/results/paper_evaluation/covft_geometric_covvina_rl_10k.csv \
    --workers 32 --checkpoint-every 200 > ${REMOTE_ROOT}/logs/covvina_10k_rl.log 2>&1 \
' > /dev/null 2>&1 < /dev/null &"

echo "=== Launch VM2 (ai-chem2): covft + v2fixed ==="
$SSH@$VM2_IP "cd ${REMOTE_DIR} && nohup bash -c '\
  source ~/miniconda3/etc/profile.d/conda.sh && conda activate quris && \
  export LD_LIBRARY_PATH=\$CONDA_PREFIX/lib:\${LD_LIBRARY_PATH:-} && \
  python run_covvina_10k.py --smiles inputs/samples_covft.csv --cohort covft \
    --out ${REMOTE_ROOT}/results/paper_evaluation/covft_geometric_covvina_covft_10k.csv \
    --workers 32 --checkpoint-every 200 > ${REMOTE_ROOT}/logs/covvina_10k_covft.log 2>&1 && \
  python run_covvina_10k.py --smiles inputs/cohort_A.csv --cohort v2fixed \
    --out ${REMOTE_ROOT}/results/paper_evaluation/covft_geometric_covvina_v2fixed_10k.csv \
    --workers 32 --checkpoint-every 200 > ${REMOTE_ROOT}/logs/covvina_10k_v2fixed.log 2>&1 \
' > /dev/null 2>&1 < /dev/null &"

echo "=== Launched. Tail logs with:"
echo "  ssh ai-chem  'tail -f ~/edit-small-mol/logs/covvina_10k_base.log'"
echo "  ssh ai-chem2 'tail -f ~/edit-small-mol/logs/covvina_10k_covft.log'"
