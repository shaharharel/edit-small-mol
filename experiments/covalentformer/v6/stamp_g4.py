#!/usr/bin/env python3
"""Stamp the g4 records with their generation, now that the corpus is complete.

dataset_manifest.py resolves a generation from the BATCH NUMBER, which works while every
generation occupies a distinct batch range inside v6/label_out. g4 re-labels batches 0001-0071 --
the same numbers as g1/g2/g3 -- in a separate directory, so the batch-number scheme cannot
distinguish it and is deliberately not reused here.

Stamping was held until verify_output.py passed on all 7,069 pairs, so the field cannot assert a
completeness that did not exist.
"""
import glob
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
DIR = os.path.join(HERE, 'label_out_g4')
GEN = 'g4_v2_input_anchored_envelope_pass2'

n = stamped = already = 0
for f in sorted(glob.glob(os.path.join(DIR, 'batch_*.json'))):
    d = json.load(open(f))
    dirty = False
    for pid, v in d.items():
        n += 1
        if v.get('label_generation') == GEN:
            already += 1
        else:
            v['label_generation'] = GEN
            stamped += 1
            dirty = True
    if dirty:
        json.dump(d, open(f, 'w'), indent=2)
print('records %s   stamped %s   already stamped %s' % (format(n, ','), format(stamped, ','),
                                                        format(already, ',')))
# re-read and confirm
import collections
c = collections.Counter()
for f in sorted(glob.glob(os.path.join(DIR, 'batch_*.json'))):
    for v in json.load(open(f)).values():
        c[v.get('label_generation')] += 1
print('verified on re-read: %s' % dict(c))
