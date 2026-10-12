#!/usr/bin/env python3
"""Collapse the evidence rows to one row per CHEMICAL EDIT, keeping the best-evidenced crystal.

Why this is needed before labelling and not after:

The row count (13,168) counts (hit crystal, lead crystal) combinations. The same chemical edit
recurs across many PDB entries -- the worst case is a phosphatidylcholine acyl-chain change that
appears 154 times in membrane-protein structures. Paying three reviewers and a chair to write 154
independent rationales for one lipid edit is wasted effort, and it would tilt the corpus toward
chemistry that is not hit-to-lead at all.

This is deduplication, NOT filtering: every distinct edit survives. Only redundant crystals of an
edit already represented are set aside, and they are written to a sidecar file so the extra
structural evidence for an edit is never actually lost.

Choosing which crystal represents an edit, in order:
  1. numbering_consistent -- hit and lead residue numbering agree, so lead_contacts are citable
  2. more atoms in the MODEL_INPUT table -- a more complete pocket description
  3. a site_of_change that is a believable size for the edit (see score_site)
  4. pair string, purely so the choice is deterministic across runs
"""
import json, collections, os

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SRC = D + '/v6_evidence_aligned.jsonl'
OUT = D + '/v6_labelset.jsonl'
EXTRA = D + '/v6_evidence_redundant.jsonl'

rows = [json.loads(l) for l in open(SRC)]


def edit_key(r):
    return (r['MODEL_INPUT']['hit_smiles'], r['ANSWER']['analogue_smiles'])


def score_site(r):
    """Prefer a site_of_change whose size is plausible for the edit it claims to describe."""
    soc = len(r['LABEL_ONLY']['site_of_change'])
    n = len(r['MODEL_INPUT']['atoms'])
    if not soc or not n:
        return -1
    # a site covering nearly the whole molecule explains nothing; nor does a 1-atom site for a
    # large edit. 10-40% of the molecule is the useful band.
    frac = soc / n
    return 1 if 0.05 <= frac <= 0.45 else 0


def rank(r):
    lo = r['LABEL_ONLY']
    return (1 if lo.get('numbering_consistent') else 0,
            len(r['MODEL_INPUT']['atoms']),
            score_site(r),
            r['pair'])


groups = collections.defaultdict(list)
for r in rows:
    groups[edit_key(r)].append(r)

kept, extra = [], []
for k, g in groups.items():
    g.sort(key=rank, reverse=True)
    best = g[0]
    # keep a pointer to the crystals we set aside, so the evidence is recoverable
    if len(g) > 1:
        best = dict(best)
        best['other_crystals'] = [{'pair': o['pair'], 'lead_pdb': o['LABEL_ONLY']['lead_pdb']}
                                  for o in g[1:]][:20]
        best['n_other_crystals'] = len(g) - 1
        extra.extend(g[1:])
    kept.append(best)

kept.sort(key=lambda r: r['pair'])
with open(OUT, 'w') as f:
    for r in kept:
        f.write(json.dumps(r) + '\n')
with open(EXTRA, 'w') as f:
    for r in extra:
        f.write(json.dumps(r) + '\n')

sizes = collections.Counter(len(g) for g in groups.values())
print('rows in                    %6d' % len(rows))
print('unique chemical edits      %6d   -> written to %s' % (len(kept), os.path.basename(OUT)))
print('redundant crystals set aside %4d   -> %s' % (len(extra), os.path.basename(EXTRA)))
print()
print('crystals per edit:')
for n in sorted(sizes)[:6]:
    print('   %3d crystal(s)  %6d edits' % (n, sizes[n]))
big = sorted(groups.items(), key=lambda kv: -len(kv[1]))[:3]
print('   worst case      %6d crystals for one edit' % len(big[0][1]))
print()
print('numbering_consistent among the kept rows: %d/%d (%.0f%%)'
      % (sum(1 for r in kept if r['LABEL_ONLY'].get('numbering_consistent')), len(kept),
         100 * sum(1 for r in kept if r['LABEL_ONLY'].get('numbering_consistent')) / max(len(kept), 1)))
tg = collections.Counter(r['target'] for r in kept)
print('unique targets among the kept rows: %d ; largest series %d rows (%.1f%% of corpus)'
      % (len(tg), tg.most_common(1)[0][1], 100 * tg.most_common(1)[0][1] / max(len(kept), 1)))
