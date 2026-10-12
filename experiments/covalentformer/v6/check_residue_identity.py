#!/usr/bin/env python3
"""Does a residue NUMBER mean the same RESIDUE in the hit's and the lead's crystal?

`fix_numbering.py` finds the author-numbering offset between the two entries by mode-voting over
residue-type name matches, and accepts it when it explains a majority of the lead's contacts. What
it never checks is the consequence: that at the accepted offset the residue TYPES actually agree.

They do not always. 255 pairs carry `numbering_consistent: True` while some shared residue number
is a different amino acid on each side -- GLN110 in the hit and THR110 in the lead at confidence
0.97. Those are point mutants of the same protein, crystallised separately. A labeller comparing
hit contacts to `lead_contacts` would then write "the lead's group now reaches THR110" when the
hit's protein has a glutamine there, which is a protein difference masquerading as an edit effect.

Annotate, do not drop: a mutant pair is still a real pair, and for some targets the mutation is the
point. But the labeller has to be told which residue numbers are not comparable.
"""
import collections, json, os, re, sys

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
SRC = sys.argv[1] if len(sys.argv) > 1 else 'v6_labelcohort.jsonl'
OUT = sys.argv[2] if len(sys.argv) > 2 else 'v6_labelcohort_checked.jsonl'
RES = re.compile(r'([A-Z]{1,3})(-?\d+)\.')


def typed(faces):
    """('LEU', '137') for an amino acid; None for a metal or het group (ZN301.ZN)."""
    m = RES.match(faces)
    return (m.group(1), m.group(2)) if m and len(m.group(1)) == 3 else None


def conflicts(e):
    hit = collections.defaultdict(set)
    for a in e['MODEL_INPUT']['atoms']:
        t = typed(a['faces'])
        if t:
            hit[t[1]].add(t[0])
    out = {}
    for c in (e['LABEL_ONLY'].get('lead_contacts') or []):
        t = typed(c['faces'])
        if t and t[1] in hit and t[0] not in hit[t[1]]:
            out[t[1]] = (sorted(hit[t[1]])[0], t[0])
    return out


if __name__ == '__main__':
    rows = [json.loads(l) for l in open(os.path.join(D, SRC))]
    n = 0
    tally = collections.Counter()
    with open(os.path.join(D, OUT), 'w') as f:
        for e in rows:
            cf = conflicts(e)
            if cf:
                n += 1
                e['LABEL_ONLY']['residue_identity_conflicts'] = {
                    k: '%s in the hit, %s in the lead' % v for k, v in sorted(cf.items())}
                e['LABEL_ONLY']['residue_identity_warning'] = (
                    'residue number(s) %s are a DIFFERENT amino acid in the two crystals (a point '
                    'mutant). Do not compare hit and lead contacts at those numbers.'
                    % ', '.join(sorted(cf)))
                for k, v in cf.items():
                    tally['%s->%s' % v] += 1
            f.write(json.dumps(e) + '\n')
    print('pairs in                          %6d' % len(rows))
    print('pairs with a residue-identity clash %4d  (%.1f%%)' % (n, 100.0 * n / len(rows)))
    print('pairs dropped                          0   (annotation only)')
    print('\nmost common substitutions:')
    for k, v in tally.most_common(10):
        print('   %-14s %4d' % (k, v))
    print('\n-> %s' % os.path.join(D, OUT))
