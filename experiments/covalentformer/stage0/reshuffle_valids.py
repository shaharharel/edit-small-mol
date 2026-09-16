#!/usr/bin/env python
"""Reshuffle every valid/train CSV in place. SEEDED, header preserved, verified after.

WHY. The split builders write rows in CORPUS order, and the corpus is MW-ASCENDING (the
all-vs-all used an MW-sorted sliding window). Any code that takes the head of a valid file
-- every `--n` flag in this repo -- therefore samples the SMALLEST molecules. That single
fact produced three separate wrong conclusions in one session: a phantom truncation bug
(generated MW 261 vs a corpus median of 482), a bogus 'correction' of it computed on the
same bad sample, and a 34/37 tie rate that was measured on 60 tiny fragment-like epoxides.

Shuffling at WRITE time is the fix; cohort_shift.py also shuffles defensively at READ time.
Both, because the next script to slice a head will not remember to.
"""
import csv, glob, os, random, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
RDLogger.DisableLog('rdApp.*')

SEED = 20260916
rng = random.Random(SEED)

def mw_trend(rows, n=400):
    """Correlation of MW with ROW INDEX. ~0 after shuffling; strongly + before."""
    sub = rows[:n]
    xs, ys = [], []
    for i, r in enumerate(sub):
        m = Chem.MolFromSmiles(r.get('anchor', ''))
        if m is not None:
            xs.append(i); ys.append(Descriptors.MolWt(m))
    if len(xs) < 30:
        return None
    return float(np.corrcoef(xs, ys)[0, 1])

files = sorted(glob.glob('data/steer_v*/*.csv'))
print('%-46s %7s %9s %9s' % ('file', 'rows', 'r_before', 'r_after'))
for f in files:
    with open(f) as fh:
        rd = csv.DictReader(fh); cols = rd.fieldnames; rows = list(rd)
    if len(rows) < 20:
        continue
    before = mw_trend(rows)
    rng.shuffle(rows)
    tmp = f + '.tmp'
    with open(tmp, 'w', newline='') as fo:
        w = csv.DictWriter(fo, fieldnames=cols); w.writeheader(); w.writerows(rows)
    os.replace(tmp, f)          # atomic: never leave a half-written split on disk
    with open(f) as fh:
        after = mw_trend(list(csv.DictReader(fh)))
    flag = ''
    if before is not None and after is not None:
        if abs(before) > 0.3 and abs(after) < 0.15:
            flag = '  <- ORDERING REMOVED'
        elif abs(after) > 0.3:
            flag = '  <- STILL ORDERED, INVESTIGATE'
    print('%-46s %7d %9s %9s%s' % (f, len(rows),
          ('%.3f' % before) if before is not None else '  n/a',
          ('%.3f' % after) if after is not None else '  n/a', flag))
print('\nseed %d. r = corr(MW, row index) over the first 400 rows.' % SEED)
print('Near zero after shuffling means a head slice is now a representative sample.')
