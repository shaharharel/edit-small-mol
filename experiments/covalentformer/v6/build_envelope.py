#!/usr/bin/env python3
"""Compute the 8 A pocket envelope for every pair, as a SIDECAR file. Does not touch the corpus.

    python build_envelope.py --probe 60          # sample, verify, change nothing
    python build_envelope.py --workers 12        # full pass -> v7_envelope.json

WHY THIS EXISTS

The input atom table lists, for each ligand atom, its NEAREST protein atom. That samples the pocket
through nearest-neighbour relationships only, so a residue 4 A from the ligand that happens not to
be anybody's nearest neighbour appears NOWHERE. Measured on the first 2,000 labels, 24% of sentences
name a residue absent from the input side, and 100% of those mentions trace to the analogue's own
contact table with 0% invented -- the labellers were reading real evidence that the input side could
not express. The v2 rules forbid naming those residues, which makes rule 1 strict in a way the
evidence cannot satisfy. The envelope is what makes it satisfiable: measured on 70 pairs, an 8 A cut
recovers 94% of exactly those mentions (6 A recovers 75%, 10 A 96%, for a larger prompt).

8.0 A is therefore not a round number chosen for comfort -- it is where the recovery curve flattens.

WHAT IT IS NOT

It is not a contact list. A residue at 7.5 A touches nothing; it is somewhere a group COULD reach.
The renderer must present it that way or labellers will cite envelope distances as interactions,
which would be a worse failure than the one this fixes.

THE RISK THIS CARRIES, AND THE FLAGS THAT DISCLOSE IT

An mmCIF holds the asymmetric unit. A residue 8 A away can be a CRYSTAL PACKING NEIGHBOUR rather
than part of the binding site, and a labeller has no way to tell. That is the one failure mode here
that produces a confidently WRONG label rather than a thin one, and the corpus already carries a
`CHAIN TERMINUS CONTACT` caveat for the same reason ("a chain end is usually a neighbouring copy in
the crystal"). So every envelope row carries:

  `other_chain`  the residue is not on the ligand's own chain. Real multi-chain pockets exist (the
                 corpus has a MULTI-CHAIN POCKET flag), so this is disclosed, not filtered.
  `terminus`     the residue has an OXT atom, i.e. it is a chain end -- the strongest cheap signal
                 of a lattice neighbour rather than a site residue.
  `n_atoms`      how many of its atoms are inside the cut. A residue reaching in with one atom is a
                 different proposition from one presenting a whole face.

NECESSARY CONDITION, CHECKED ON EVERY PAIR

Every residue already in that pair's contact table must appear in its envelope. A contact is by
definition within 4 A, so an envelope at 8 A that misses one is a bug in this script -- wrong chain
bookkeeping, wrong ligand copy, a parsing slip. `--probe` asserts it and the full pass counts it.
"""
import argparse
import collections
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import evidence_v4 as E  # noqa: E402  (its reader and residue tables are the source of truth)

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
OUT = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_envelope.json'
CUT = 8.0
_RES = re.compile(r'([A-Z0-9]+?)(\d+)\.')


def envelope_for(pdb, lig3):
    """Residues with any atom within CUT of any atom of the input ligand copy."""
    P, L, _W = E.read_cx(pdb, lig3)
    if not P or not L:
        return None
    Px = np.array([a['xyz'] for a in P])
    Lx = np.array([a['xyz'] for a in L])
    d = np.linalg.norm(Px[:, None, :] - Lx[None, :, :], axis=2).min(axis=1)
    # the ligand copy's own chain, for the other_chain flag
    lig_ch = collections.Counter(a['ch'] for a in L).most_common(1)[0][0]
    by = {}
    term = set()
    for i in np.where(d <= CUT)[0]:
        a = P[i]
        key = (a['ch'], a['rn'], a['res'])
        cur = by.get(key)
        if cur is None or d[i] < cur['min_dist']:
            by[key] = {'res': a['res'], 'rn': a['rn'], 'ch': a['ch'],
                       'min_dist': round(float(d[i]), 2),
                       'n_atoms': 1,
                       'hetero': bool(a.get('hetero')),
                       'other_chain': a['ch'] != lig_ch}
        else:
            cur['n_atoms'] += 1
        if a.get('name') == 'OXT':
            term.add(key)
    for key, row in by.items():
        row['terminus'] = key in term
    rows = sorted(by.values(), key=lambda r: r['min_dist'])
    return {'cut_A': CUT, 'ligand_chain': lig_ch, 'n_residues': len(rows), 'residues': rows}


def table_residues(rec):
    """Residues the pair's contact table names, each with the SHORTEST distance the table gives it.

    The distance matters, and assuming it did not cost me a false alarm on 36 pairs. The table is
    not a 4 A contact list: it reports each ligand atom's NEAREST protein atom whatever the
    distance, so a row can read `CM2 faces MG3260.MG at 8.08` when nothing is closer. An 8.00 A
    envelope then excludes that residue correctly, and a condition that demanded its presence was
    testing something false. 7M4V is the clean example -- an RNA structure with 159 magnesium ions,
    where the ligand's nearest Mg sits past the cut -- and 6PC5 is the other shape of it, a pair
    whose ligand is tens of angstroms from any protein at all.
    """
    out = {}
    for a in rec['MODEL_INPUT']['atoms']:
        for k, dk in (('faces', 'distance_A'), ('polar_partner', 'polar_distance_A')):
            m = _RES.match(str(a.get(k) or ''))
            if m and a.get(dk) is not None:
                key = (m.group(1), m.group(2))
                out[key] = min(out.get(key, 1e9), float(a[dk]))
    return out


def _job(args):
    pair, pdb, lig3 = args
    try:
        return pair, envelope_for(pdb, lig3)
    except Exception as exc:                                   # noqa: BLE001
        return pair, {'error': repr(exc)[:160]}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--probe', type=int, default=0, help='sample N pairs, verify, write nothing')
    ap.add_argument('--workers', type=int, default=max(1, (os.cpu_count() or 4) - 2))
    ap.add_argument('--out', default=OUT)
    a = ap.parse_args()

    recs = []
    for line in open(EV):
        r = json.loads(line)
        pdb = r['pair'][:4]
        lig3 = r['pair'].split()[1].split('->')[0]
        recs.append((r['pair'], pdb, lig3, r))
    print('corpus %s pairs' % format(len(recs), ','))

    if a.probe:
        import random
        random.seed(11)
        recs = random.sample(recs, min(a.probe, len(recs)))
        print('probing %d' % len(recs))

    jobs = [(p, pdb, lig) for p, pdb, lig, _ in recs]
    by_pair = {}
    if a.workers > 1 and not a.probe:
        import multiprocessing as mp
        with mp.Pool(a.workers) as pool:
            for i, (pair, env) in enumerate(pool.imap_unordered(_job, jobs, chunksize=8), 1):
                by_pair[pair] = env
                if i % 500 == 0:
                    print('  %s / %s' % (format(i, ','), format(len(jobs), ',')), flush=True)
    else:
        for j in jobs:
            pair, env = _job(j)
            by_pair[pair] = env

    S = collections.Counter()
    sizes = []
    missing_examples = []
    for pair, pdb, lig, rec in recs:
        env = by_pair.get(pair)
        if env is None:
            S['no structure readable'] += 1
            continue
        if 'error' in env:
            S['error'] += 1
            continue
        S['ok'] += 1
        sizes.append(env['n_residues'])
        S['rows_total'] += env['n_residues']
        S['rows_other_chain'] += sum(1 for x in env['residues'] if x['other_chain'])
        S['rows_terminus'] += sum(1 for x in env['residues'] if x['terminus'])
        S['rows_hetero'] += sum(1 for x in env['residues'] if x['hetero'])
        # NECESSARY CONDITION, correctly scoped: a table residue recorded INSIDE the cut must be
        # in the envelope. One recorded beyond the cut must not be, and demanding it was the bug.
        have = {(x['res'], x['rn']) for x in env['residues']}
        tbl = table_residues(rec)
        inside = {k for k, d in tbl.items() if d <= CUT + 0.01}
        S['table_residues_beyond_cut'] += len(tbl) - len(inside)
        miss = inside - have
        if miss:
            S['pairs MISSING a table residue inside the cut'] += 1
            if len(missing_examples) < 5:
                missing_examples.append((pair, sorted((k, round(tbl[k], 2)) for k in miss)[:4]))
        if not env['n_residues']:
            S['pairs with an EMPTY envelope'] += 1

    print()
    for k in ('ok', 'no structure readable', 'error', 'pairs with an EMPTY envelope',
              'table_residues_beyond_cut', 'pairs MISSING a table residue inside the cut'):
        print('  %-32s %s' % (k, format(S[k], ',')))
    if sizes:
        sizes.sort()
        print('\n  envelope size: median %d, p90 %d, max %d  (mean %.1f)'
              % (sizes[len(sizes) // 2], sizes[int(len(sizes) * 0.9)], sizes[-1],
                 sum(sizes) / len(sizes)))
        print('  of %s envelope rows: %.1f%% other chain, %.1f%% chain terminus, %.1f%% hetero'
              % (format(S['rows_total'], ','),
                 100.0 * S['rows_other_chain'] / S['rows_total'],
                 100.0 * S['rows_terminus'] / S['rows_total'],
                 100.0 * S['rows_hetero'] / S['rows_total']))
    for pair, miss in missing_examples:
        print('  !! %s misses %s' % (pair, miss))

    if S['pairs MISSING a table residue inside the cut']:
        print('\nFAILED on %d pairs: a contact-table residue recorded INSIDE %.1f A is absent from '
              'the envelope. That is a bookkeeping bug here -- wrong ligand copy, wrong chain, a '
              'parsing slip -- not a property of the data. Do not merge this.'
              % (S['pairs MISSING a table residue inside the cut'], CUT))
        return 2
    print('\nnecessary condition holds: every contact-table residue recorded within %.1f A is in '
          'its pair\'s envelope (%s table residues lie beyond the cut and are correctly absent).'
          % (CUT, format(S['table_residues_beyond_cut'], ',')))

    if a.probe:
        print('\n(probe only -- nothing written)')
        return 0
    json.dump(by_pair, open(a.out, 'w'))
    print('wrote %s  (%.1f MB)' % (a.out, os.path.getsize(a.out) / 1e6))
    return 0


if __name__ == '__main__':
    sys.exit(main())
