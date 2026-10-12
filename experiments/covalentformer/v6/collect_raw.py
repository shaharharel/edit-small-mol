#!/usr/bin/env python3
"""Assemble the three independent readings of each pair into one file per batch, for its chair.

The three labellers never see each other, so this is the first place their readings meet. What the
chair needs is not three sentences to pick from but the DISAGREEMENT: where two labellers read the
same evidence differently, one of them has usually found something -- a caveat the other missed, a
rule applied differently, or a defect in our data. Three rounds running, every real bug in this
pipeline surfaced as a disagreement or a complaint, never as a consensus.
"""
import collections, glob, json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
RAW = os.path.join(SP, 'label_out', 'raw')

manifest = json.load(open(os.path.join(SP, 'label_batches.json')))
out_dir = os.path.join(SP, 'label_out', 'for_chair')
os.makedirs(out_dir, exist_ok=True)

report = collections.Counter()
for fname, pair_ids in sorted(manifest.items()):
    b = fname.replace('label_batch_', '').replace('.txt', '')
    reads = {}
    for tag in ('labA', 'labB', 'labC'):
        p = os.path.join(RAW, 'batch_%s_%s.json' % (b, tag))
        if not os.path.exists(p):
            report['missing %s' % os.path.basename(p)] += 1
            continue
        try:
            reads[tag] = json.load(open(p))
        except Exception as e:
            report['unparseable %s: %s' % (os.path.basename(p), e)] += 1
    if not reads:
        continue

    bundle = {}
    for pid in pair_ids:
        rows = {}
        for tag, d in reads.items():
            r = d.get(pid)
            if isinstance(r, dict):
                rows[tag] = {'sentence': (r.get('sentence') or '').strip(),
                             'basis': r.get('basis') or '',
                             'confidence': r.get('confidence') or '',
                             'concern': r.get('concern') or ''}
        if not rows:
            report['no labeller covered %s' % pid] += 1
            continue
        sents = [v['sentence'] for v in rows.values()]
        blanks = sum(1 for s in sents if not s)
        # A cheap, honest disagreement signal: do the three sentences name the same residues? Two
        # labellers citing different residues for the same edit have read the pocket differently,
        # which is exactly what the chair should look at first.
        import re
        resets = [frozenset(re.findall(r'\b[A-Z]{3}\d{1,5}\b', s)) for s in sents if s]
        agree_res = bool(resets) and all(x == resets[0] for x in resets)
        bundle[pid] = {'readings': rows,
                       'n_blank': blanks,
                       'residues_agree': agree_res,
                       'residue_sets': [sorted(x) for x in resets]}
        report['pairs where the three name DIFFERENT residues' if not agree_res
               else 'pairs where all three name the same residues'] += 1
        if blanks:
            report['pairs at least one labeller refused'] += 1

    with open(os.path.join(out_dir, 'batch_%s.json' % b), 'w') as f:
        json.dump(bundle, f, indent=1)
    print('batch %-3s %2d pairs, %d readings each -> for_chair/batch_%s.json'
          % (b, len(bundle), len(reads), b))

print()
for k, v in report.most_common():
    print('  %-56s %4d' % (k, v))
