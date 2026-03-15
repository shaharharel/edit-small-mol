#!/bin/bash
# Overnight evaluation pipeline
# Run: nohup bash experiments/run_overnight.sh > results/overnight.log 2>&1 &

set -e

echo "=== Starting overnight evaluation: $(date) ==="

# Phase 1: Embedder selection (if not already done)
echo ""
echo "=== Phase 1: Embedder Selection ==="
conda run -n quris python -u experiments/run_paper_evaluation.py --phase 1

echo ""
echo "=== Phase 1 complete: $(date) ==="

# Phase 2: Architecture comparison (uses best embedder from Phase 1)
echo ""
echo "=== Phase 2: Architecture Comparison ==="
conda run -n quris python -u experiments/run_paper_evaluation.py --phase 2

echo ""
echo "=== Phase 2 complete: $(date) ==="

# Phase 3: Generalization splits
echo ""
echo "=== Phase 3: Generalization ==="
conda run -n quris python -u experiments/run_paper_evaluation.py --phase 3

echo ""
echo "=== Phase 3 complete: $(date) ==="

# Generate report
echo ""
echo "=== Generating report ==="
conda run -n quris python -u experiments/run_paper_evaluation.py --report-only

echo ""
echo "=== ALL DONE: $(date) ==="
