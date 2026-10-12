#!/usr/bin/env python3
"""Render N pairs into K batch files for the labelling panel, and verify the five fixes from the
medchem re-audit on the pairs actually being labelled.

Pairs are drawn uniformly at random (no stratification) so the 20-example test measures what the
full run would look like, not a curated best case.
"""
import collections, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_v7 import render

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SP = os.path.dirname(os.path.abspath(__file__))
N = int(sys.argv[1]) if len(sys.argv) > 1 else 20
K = int(sys.argv[2]) if len(sys.argv) > 2 else 4

rows = [json.loads(l) for l in open(D + '/v7_evidence.jsonl')]
random.seed(2026)
pick = random.sample(rows, N)

# ---- verify the re-audit fixes on the sample we are about to label, not on the corpus at large
chk = collections.Counter()
for e in pick:
    lo = e['LABEL_ONLY']
    names = {a['atom'] for a in e['MODEL_INPUT']['atoms']}
    chk['pairs'] += 1
    if lo.get('room_contradicts_edit') is None:
        chk['BLOCKER room_contradicts_edit is None'] += 1
    for c in (lo.get('conserved_interactions') or []):
        if c.get('input_interaction', '').startswith('no contact within') or \
           c.get('analogue_interaction', '').startswith('no contact within'):
            chk['BLOCKER conserved row that is not a real contact'] += 1
        if c['input_atom'] not in names:
            chk['BLOCKER conserved row names an absent input atom'] += 1
    ac = lo.get('analogue_contacts') or []
    if len([c['atom'] for c in ac]) != len({c['atom'] for c in ac}):
        chk['BLOCKER duplicate analogue atoms'] += 1
    chk['conserved rows'] += len(lo.get('conserved_interactions') or [])
    chk['rule-7 pairs'] += bool(lo.get('room_contradicts_edit'))
    chk['pairs with a salt bridge'] += any('SALT BRIDGE' in a['interaction']
                                           for a in e['MODEL_INPUT']['atoms'])
    chk['pairs with a covalent bond'] += any(a['interaction'].startswith('COVALENT BOND')
                                             for a in e['MODEL_INPUT']['atoms'])

print('sample of %d pairs for the labelling test' % N)
for k in sorted(chk):
    print('   %-50s %d' % (k, chk[k]))
blockers = [k for k in chk if k.startswith('BLOCKER')]
print('\n%s' % ('BLOCKERS PRESENT: ' + ', '.join(blockers) if blockers else 'no blockers in sample'))

per = (N + K - 1) // K
paths = []
for b in range(K):
    chunk = pick[b * per:(b + 1) * per]
    if not chunk:
        continue
    p = os.path.join(SP, 'label_batch_%d.txt' % (b + 1))
    with open(p, 'w') as f:
        f.write('LABELLING BATCH %d — %d pairs. One sentence per pair.\n\n' % (b + 1, len(chunk)))
        for e in chunk:
            f.write(render(e) + '\n\n')
    paths.append((p, [e['pair'] for e in chunk]))

with open(os.path.join(SP, 'label_batches.json'), 'w') as f:
    json.dump({os.path.basename(p): ids for p, ids in paths}, f, indent=1)
print()
for p, ids in paths:
    print('%s  %d pairs  %d KB' % (os.path.basename(p), len(ids), os.path.getsize(p) / 1024))
