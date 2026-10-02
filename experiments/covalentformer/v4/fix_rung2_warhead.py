#!/usr/bin/env python
"""Repair rung-2 rows whose instruction claims "keeps the warhead" while the target changes it.

MEASURED: 9,650 of 79,241 rows (12.18%) change the molecule-level warhead class between hit and
analogue, yet every rung-2 prompt ends "Propose a close analogue that keeps the warhead and the
binding scaffold." Those rows train the model to ignore an explicit constraint in its own prompt,
and they are 4% of the full v4 mixture.

These are REAL CHEMISTRY, not corrupt rows -- cyanamide (N-C#N) to activated nitrile (C-C#N) is a
genuine medicinal-chemistry move with different reactivity and mechanism. So the fix corrects the
INSTRUCTION rather than deleting the pair: drop the warhead clause for those rows and keep the
scaffold clause, which still holds.

WHY NOT NAME THE TARGET WARHEAD. "...change the warhead to activated nitrile" would read more
naturally, but it puts the answer's warhead class in the prompt, making those 9,650 rows markedly
easier than the other 69,591 and inflating any aggregate generation metric. Removing the false
clause costs nothing and leaks nothing.

MEASUREMENT LEVEL MATTERS AND COST TWO WRONG NUMBERS HERE. A first pass tested the truthiness of
classify()'s return -- a dict, always truthy, even {'ok': False} -- and reported 82.6%. A second
pass classified the detached MCS fragments and reported 10.9%, but a fragment has no molecular
context, so it both misses warheads and invents them (74.6% of its "swaps" preserved the warhead at
the molecule level). The instruction's claim is about the MOLECULE, so the molecule is what gets
checked.
"""
from __future__ import annotations
import argparse, collections, json, sys

sys.path.insert(0, '/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/stage0')
from rdkit import Chem, RDLogger                                   # noqa: E402
RDLogger.DisableLog('rdApp.*')
from covalent_filter import classify                               # noqa: E402

KEEP = 'keeps the warhead and the binding scaffold'
ONLY_SCAFFOLD = 'keeps the binding scaffold'


def wset(smi):
    if not smi:
        return None
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    r = classify(m)
    return frozenset(r.get('accepted') or []) if r.get('ok') else frozenset()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--inp', required=True)
    ap.add_argument('--out', required=True)
    a = ap.parse_args()

    c = collections.Counter()
    out = []
    for line in open(a.inp):
        r = json.loads(line)
        h, an = wset(r.get('hit')), wset(r.get('analogue'))
        if h is None or an is None:
            c['unparseable']+= 1
            out.append(r)
            continue
        if h and an is not None and h != an and not (h & an):
            ins = r.get('instruction') or ''
            if KEEP in ins:
                r['instruction'] = ins.replace(KEEP, ONLY_SCAFFOLD)
                r['warhead_changed'] = True
                c['instruction_corrected'] += 1
            else:
                c['changed_but_clause_absent'] += 1
        else:
            c['warhead_preserved'] += 1
        out.append(r)

    tot = sum(c.values())
    print('rows %d' % tot)
    for k, v in c.most_common():
        print('  %-30s %6d  %5.2f%%' % (k, v, 100.0 * v / max(tot, 1)))

    # GATE: after the repair, no row may both claim the warhead is kept and change it.
    bad = 0
    for r in out:
        h, an = wset(r.get('hit')), wset(r.get('analogue'))
        if h and an is not None and h != an and not (h & an) and KEEP in (r.get('instruction') or ''):
            bad += 1
    print('\nrows still claiming "keeps the warhead" while changing it: %d (must be 0)' % bad)
    if bad:
        sys.exit('GATE FAILED -- nothing written')
    with open(a.out, 'w') as f:
        for r in out:
            f.write(json.dumps(r) + '\n')
    print('wrote %s  %d rows' % (a.out, len(out)))
    print('FIX_DONE')


if __name__ == '__main__':
    main()
