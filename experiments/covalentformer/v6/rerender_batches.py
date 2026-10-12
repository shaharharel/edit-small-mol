#!/usr/bin/env python3
"""Re-render the EXISTING batch packets from the current corpus, keeping the same pairs.

make_label_batches.py re-samples, so running it after a rebuild would hand the chairs a different
fifty pairs than the three labellers read, and every reading would be orphaned. This renders exactly
the ids already in label_batches.json.

It exists because I rebuilt the corpus and left the .txt packets stale, so a chair adjudicated
against evidence two builds old and reported two already-fixed bugs as live. Deriving an artifact
and then regenerating its source without regenerating the artifact is the same mistake as the
truncated column and the dead RESIDUE SUSPECT annotation: the fix was real, the thing people read
was not.
"""
import json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SP)
from render_v7 import render

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'

ev = {}
for line in open(EV):
    r = json.loads(line)
    ev[r['pair']] = r

man = json.load(open(os.path.join(SP, 'label_batches.json')))
missing = []
for fname, ids in sorted(man.items()):
    chunk = []
    for pid in ids:
        if pid not in ev:
            missing.append(pid)
            continue
        chunk.append(ev[pid])
    p = os.path.join(SP, fname)
    with open(p, 'w') as f:
        f.write('LABELLING BATCH %s — %d pairs. One sentence per pair.\n\n'
                % (fname.replace('label_batch_', '').replace('.txt', ''), len(chunk)))
        for e in chunk:
            f.write(render(e) + '\n\n')
    print('%-20s %2d of %2d pairs re-rendered  (%d KB)'
          % (fname, len(chunk), len(ids), os.path.getsize(p) / 1024))

if missing:
    print('\n*** %d pairs are no longer in the corpus and were dropped from their packet: %s'
          % (len(missing), ', '.join(missing[:8])))
    sys.exit(1)
print('\nall pairs present; packets now match the live corpus')
