#!/usr/bin/env python
"""PRE-REGISTERED FALSIFIER for theta_bd, stated before the numbers are read.

THE CHALLENGE (from the covalent medchem consult, and it is a good one): in a crystal
structure of a FORMED adduct the attacked carbon is sp3, so nucleophile-C(sp3)-C(alpha) is
just a tetrahedral bond angle. The Burgi-Dunitz angle describes APPROACH to an sp2 carbonyl
on the way to sp3 -- a pre-reaction quantity that an adduct structure cannot contain. If that
is right, theta_bd is refinement geometry, not design geometry, and it dies the way d_cys did.

Supporting evidence already in hand: pooled mean 111.77 deg against tetrahedral 109.47, with
79.3% of values inside one sd of tetrahedral.

THE TEST. Bond angles in a crystal structure are RESTRAINED to dictionary ideals, and the
restraint bites harder at low resolution where the data cannot overrule it. So:

  H1 (theta_bd is refinement noise): spread SHRINKS on high-resolution structures, because
      better data means tighter geometry around the same ideal value. Predict sd <= 6 deg
      at <= 2.0 A, against 10.1 deg pooled.

  H2 (theta_bd is real conformational variation): spread is FLAT or GROWS with resolution,
      because better data resolves genuine differences that were being averaged away.

These predict OPPOSITE directions, so the test can fail. Whichever fires, it is reported.

A third outcome is possible and must not be narrated as either: if high-resolution structures
are too few, the answer is UNDERPOWERED, not H1.
"""
from __future__ import annotations
import os, sys, json, statistics, collections

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_param_gate import PDB_DIR   # noqa: E402


def resolution(pdb):
    """Parse RESOLUTION from REMARK 2. Returns None for NMR / unparseable."""
    path = os.path.join(PDB_DIR, pdb + '.pdb')
    if not os.path.exists(path):
        return None
    with open(path, errors='ignore') as fh:
        for line in fh:
            if line.startswith('REMARK   2 RESOLUTION.'):
                # PARSE AFTER THE WORD 'RESOLUTION.', NOT line.split().
                # line.split() picks up the '2' of "REMARK   2" first, so EVERY structure
                # came back 2.0 A and all 904 complexes landed in one bin -- which is what
                # gave it away, since no real PDB set has a single resolution.
                tail = line.split('RESOLUTION.', 1)[1]
                for tok in tail.replace('ANGSTROMS', ' ').split():
                    try:
                        v = float(tok)
                    except ValueError:
                        continue
                    if 0.3 < v < 10.0:
                        return v
                return None
            if line.startswith('ATOM'):
                break
    return None


def main():
    rows = [json.loads(l) for l in open('data/instructions/pocket_pairs_v2.jsonl')]
    per_pdb = {}
    for r in rows:
        if r.get('theta_bd_a') is not None:
            per_pdb[r['pdb_a']] = r['theta_bd_a']
    print('complexes with a theta_bd value: %d' % len(per_pdb)); sys.stdout.flush()

    res = {}
    for p in per_pdb:
        res[p] = resolution(p)
    have = {p: v for p, v in res.items() if v is not None}
    print('with a parseable resolution: %d' % len(have))

    bins = [('<= 1.8 A', 0.0, 1.8), ('1.8-2.2', 1.8, 2.2),
            ('2.2-2.8', 2.2, 2.8), ('> 2.8 A', 2.8, 99.0)]
    print('\n%-10s %6s %9s %9s %9s' % ('bin', 'n', 'mean', 'sd', '|mean-109.47|'))
    out = {}
    for name, lo, hi in bins:
        v = [per_pdb[p] for p, r in have.items() if lo <= r < hi]
        if len(v) < 8:
            print('%-10s %6d   (too few to report)' % (name, len(v)))
            continue
        m, s = statistics.mean(v), statistics.pstdev(v)
        out[name] = (len(v), m, s)
        print('%-10s %6d %9.2f %9.3f %9.2f' % (name, len(v), m, s, abs(m - 109.47)))

    allv = list(per_pdb.values())
    print('\npooled     %6d %9.2f %9.3f %9.2f'
          % (len(allv), statistics.mean(allv), statistics.pstdev(allv),
             abs(statistics.mean(allv) - 109.47)))

    hi_res = out.get('<= 1.8 A')
    print('\n=== VERDICT ===')
    if hi_res is None:
        print('UNDERPOWERED: fewer than 8 complexes at <= 1.8 A. Neither H1 nor H2 fires.')
    else:
        n, m, s = hi_res
        print('high-res (<=1.8 A) sd = %.3f deg on n=%d, against %.3f pooled'
              % (s, n, statistics.pstdev(allv)))
        if s <= 6.0:
            print('H1 FIRES -- theta_bd is REFINEMENT GEOMETRY. Kill it: the spread that')
            print('   passed the original gate was low-resolution restraint slop, not')
            print('   conformational range. Same fate as d_cys, one level less obvious.')
        elif s >= statistics.pstdev(allv) * 0.9:
            print('H2 FIRES -- spread SURVIVES at high resolution, so it is not restraint')
            print('   slop. theta_bd stays, and the tetrahedral-mean objection is answered:')
            print('   a mean near 109.5 with real spread is a soft angle, not a fixed one.')
        else:
            print('INTERMEDIATE: sd %.2f is between the H1 bar (6.0) and the H2 bar (%.2f).'
                  % (s, statistics.pstdev(allv) * 0.9))
            print('   Partially restraint-driven. Not a clean kill and not a clean keep.')


if __name__ == '__main__':
    main()
