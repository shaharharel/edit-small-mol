#!/usr/bin/env python3
"""Collect the panel's output and run every label through gate.py.

The gate is mechanical, so a rule violation surfaces here rather than after training -- which is
how v5 shipped a 24% phantom-residue rate. A high reject rate is a finding about the brief or the
evidence, not about the labellers.
"""
import collections, glob, json, os, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from gate import check

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SP = os.path.dirname(os.path.abspath(__file__))

ev = {}
for l in open(D + '/v7_evidence.jsonl'):
    r = json.loads(l)
    ev[r['pair']] = r

labels = {}
for p in sorted(glob.glob(os.path.join(SP, 'label_out', 'batch_*.json'))):
    try:
        labels.update(json.load(open(p)))
    except Exception as e:
        print('could not parse %s: %s' % (os.path.basename(p), e))
print('collected %d labels from %d batch files\n'
      % (len(labels), len(glob.glob(os.path.join(SP, 'label_out', 'batch_*.json')))))

verd = collections.Counter()
reasons = collections.Counter()
npass = nrej = 0
wc = []
rows = []
for pid, rec in sorted(labels.items()):
    v = (rec.get('verdict') or '?').split()[0].upper()
    verd[v] += 1
    final = (rec.get('final') or '').strip()
    if v == 'REJECT' or not final:
        rows.append((pid, 'REJECTED BY CHAIR', rec.get('why', ''), []))
        continue
    e = ev.get(pid)
    if e is None:
        rows.append((pid, 'PAIR NOT IN EVIDENCE', '', []))
        continue
    bad = check(final, e)
    wc.append(len(final.split()))
    if bad:
        nrej += 1
        for b in bad:
            reasons[b.split(',')[0][:62]] += 1
    else:
        npass += 1
    rows.append((pid, 'PASS' if not bad else 'GATE REJECT', final, bad))

print('chair verdicts: %s' % dict(verd))
n = npass + nrej
if n:
    print('gate: %d/%d written sentences pass (%.0f%%)' % (npass, n, 100.0 * npass / n))
if wc:
    wc.sort()
    print('sentence length: median %d words, max %d (limit 35)' % (wc[len(wc) // 2], wc[-1]))
if reasons:
    print('\ngate rejection reasons:')
    for k, v in reasons.most_common():
        print('   %-64s %d' % (k, v))

print('\n' + '=' * 100)
for pid, status, final, bad in rows:
    print('\n%-24s %s' % (pid, status))
    if final:
        print('   "%s"' % final)
    for b in bad:
        print('     ! %s' % b)

with open(os.path.join(SP, 'label20_scored.json'), 'w') as f:
    json.dump({'labels': labels,
               'gate': {'pass': npass, 'reject': nrej,
                        'verdicts': dict(verd), 'reasons': dict(reasons)}}, f, indent=1)
print('\nwrote label20_scored.json')
