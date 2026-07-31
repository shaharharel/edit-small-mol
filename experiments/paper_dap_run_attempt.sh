#!/bin/bash
# Run one paper_dap_repro attempt: train → sample 10k → score → report.
# Usage: bash paper_dap_run_attempt.sh <TAG:A|B|C>
set -e

TAG=$1
if [[ -z "$TAG" ]]; then
    echo "Usage: $0 <TAG>"
    exit 1
fi

REPO=/home/shaharh_quris_ai/edit-small-mol
WORK=/home/shaharh_quris_ai/paper_dap_repro
PY=/home/shaharh_quris_ai/miniconda3/envs/quris/bin/python
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p $WORK
cd $WORK

CONFIG=$REPO/experiments/paper_dap_repro_${TAG}.toml
TRAIN_LOG=$WORK/train_${TAG}.log
SAMPLE_TOML=$WORK/sample_${TAG}.toml
SAMPLE_LOG=$WORK/sample_${TAG}.log
SAMPLE_CSV=$WORK/samples_${TAG}.csv
SCORE_JSON=$WORK/scores_${TAG}.json
REPORT_JSON=$WORK/report_${TAG}.json

echo "=============================================================="
echo "PAPER DAP REPRO — ATTEMPT $TAG"
echo "=============================================================="
date

# Canary before training - to confirm scorer/model consistency
CANARY_SMI="Nc1ncnc2c1cc(-c1ccc(NC(=O)/C=C/CN(C)C)cc1)cn2"
echo "[canary-pre] $CANARY_SMI" > $WORK/canary_${TAG}.log
echo "$CANARY_SMI" | $PY $REPO/experiments/reinvent4_film_scorer.py 2>>$WORK/canary_${TAG}.log | tee -a $WORK/canary_${TAG}.log

# --- TRAIN ---
echo "[train] starting $CONFIG"
t0=$(date +%s)
$PY -m reinvent -l $TRAIN_LOG -d cuda "$CONFIG" 2>&1 | tail -5
t1=$(date +%s)
echo "[train] elapsed=$((t1-t0))s"

# Find the checkpoint produced (the .chkpt file mentioned in stage)
CHKPT=$WORK/${TAG}_stage1.chkpt
if [[ ! -f "$CHKPT" ]]; then
    echo "[train] ERROR: no checkpoint at $CHKPT"
    ls -la $WORK/*.chkpt 2>&1 | head
    exit 2
fi
echo "[train] chkpt=$CHKPT  size=$(du -h $CHKPT | cut -f1)"

# --- CANARY POST-TRAIN (sanity: same molecule via the same scorer must still work) ---
echo "[canary-post] $CANARY_SMI" >> $WORK/canary_${TAG}.log
echo "$CANARY_SMI" | $PY $REPO/experiments/reinvent4_film_scorer.py 2>>$WORK/canary_${TAG}.log | tee -a $WORK/canary_${TAG}.log

# --- SAMPLE 10K ---
echo "[sample] building TOML"
sed -e "s|{MODEL}|$CHKPT|g" \
    -e "s|{OUT}|$SAMPLE_CSV|g" \
    -e "s|{TAG}|$TAG|g" \
    $REPO/experiments/paper_dap_sample_template.toml > $SAMPLE_TOML

echo "[sample] starting"
t0=$(date +%s)
$PY -m reinvent -l $SAMPLE_LOG -d cuda "$SAMPLE_TOML" 2>&1 | tail -5
t1=$(date +%s)
echo "[sample] elapsed=$((t1-t0))s"

if [[ ! -f "$SAMPLE_CSV" ]]; then
    echo "[sample] ERROR: no samples at $SAMPLE_CSV"
    exit 3
fi
echo "[sample] rows=$(wc -l < $SAMPLE_CSV)"

# --- SCORE 10k with FiLM (batched, single-model-load) ---
echo "[score] scoring cohort"
t0=$(date +%s)
$PY $REPO/experiments/paper_dap_batched_scorer.py \
    --tag $TAG \
    --samples $SAMPLE_CSV \
    --report $REPORT_JSON \
    --scored-csv $WORK/scored_${TAG}.csv \
    --target-batch 128
t1=$(date +%s)
echo "[score] elapsed=$((t1-t0))s"
echo "=============================================================="
echo "ATTEMPT $TAG DONE — see $REPORT_JSON"
cat $REPORT_JSON
echo "=============================================================="
