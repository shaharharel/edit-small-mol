#!/bin/bash
# FULL CORPUS REBUILD under PANEL_VERSION cf-warheads-v1-2026-09-15.
# Writes to *_v2 dirs; the contaminated v1 artifacts are left in place as evidence.
set -e
cd "$(dirname "$0")/.."
P=/opt/miniconda3/envs/quris/bin/python
echo "### STEP 1/3  union  $(date +%T)"
$P -u stage0/build_covalent_union.py --out data/covalent_union_v2/pairs.jsonl
echo "### STEP 2/3  all-vs-all  $(date +%T)"
$P -u stage0/pair_covalent_allvsall.py --src data/covalent_union_v2/pairs.jsonl \
     --tc 0.4 --windows 150 --out data/covalent_allvsall_v2
echo "### STEP 3/3  filter TC>=0.4 dMW<=100 MW<=800  $(date +%T)"
$P -u stage0/filter_covalent_pairs.py \
     --pairs data/covalent_allvsall_v2/pairs_tc0.40_mw150.jsonl \
     --union data/covalent_union_v2/pairs.jsonl \
     --max-dmw 100 --max-mw 800 --out data/covalent_final_v2/pairs.jsonl
echo "### REBUILD DONE  $(date +%T)"
