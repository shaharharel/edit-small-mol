#!/usr/bin/env python
"""TWO EXTRA pocket params chosen for FULL COVERAGE of the complex set.

WHY THESE TWO. theta_bd and d_cys_scaffold both begin by locating THE ATTACKED CARBON, which
silently excludes every warhead whose electrophile is not carbon -- boronic acid attacks at
boron, sulfonyl fluoride at sulfur. That is why they label 4,510 / 4,516 of 5,689 complexes
instead of all of them, and it is a COVERAGE loss that no deadband tuning can recover.
pocket_occupancy covers everything but moves in only 37.7% of pairs, and its denominator
(cavity volume) is the noisiest quantity in the set.

So both params here are deliberately defined WITHOUT reference to the electrophile:

  polar_contacts     ligand N/O atoms within 3.5 A of a protein N/O, counted.
                     The only parameter on the list that speaks to the K_I (recognition) side
                     rather than the reactivity side. An integer, so its deadband is 1 and
                     there is no measurement-noise question at all.
                     EXCLUDES the covalent partner residue, otherwise every complex scores a
                     free point for the bond it is defined by.

  buried_sasa_per_ha buried SASA divided by ligand heavy-atom count.
                     Addresses a real objection to raw buried_sasa: absolute burial rises
                     monotonically with size, so "increase burial" has the cheapest-edit
                     solution ADD GREASY BULK -- ligand efficiency down, logP up. Dividing by
                     heavy-atom count asks the ligand to bury MORE PER ATOM, which is the
                     instruction a chemist actually means. Same discipline that was applied to
                     warhead_reach_3d, which was killed by its size-matched control; applying
                     it to one param and not another is the inconsistency worth fixing.
                     Derived from quantities already computed, so it costs one pass.

Both are emitted ALONGSIDE the existing params, replacing nothing.
"""
from __future__ import annotations
import os, sys, json, math, argparse, collections, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_param_gate import parse_pdb, dist, sasa, NUC, PDB_DIR, VDW   # noqa: E402
from label_pocket_pairs import ligand_of                                 # noqa: E402

EXTRA_VERSION = 'pocketextra-v1-2026-09-16'
DEADBAND = {'polar_contacts': 1.0, 'buried_sasa_per_ha': 0.30}
POLAR = {'N', 'O'}


def measure(pdb):
    path = os.path.join(PDB_DIR, pdb + '.pdb')
    if not os.path.exists(path):
        return None
    try:
        prot, het = parse_pdb(path)
    except Exception:
        return None
    lig = ligand_of(het)
    if not prot or not lig:
        return None
    out = {'polar_contacts': None, 'buried_sasa_per_ha': None}

    near = [p for p in prot if any(dist(p[4:7], l[4:7]) < 12.0 for l in lig[:6])]

    # --- which residue carries the covalent bond? EXCLUDE it from the polar count.
    cov_res = None
    for p in prot:
        if p[1] in NUC and p[0] == NUC[p[1]]:
            if any(dist(p[4:7], l[4:7]) < 2.2 for l in lig):
                cov_res = (p[1], p[2], p[3]); break

    n_pol = 0
    for l in lig:
        if l[7].upper() not in POLAR:
            continue
        for p in near:
            if p[7].upper() not in POLAR:
                continue
            if cov_res is not None and (p[1], p[2], p[3]) == cov_res:
                continue
            if dist(l[4:7], p[4:7]) <= 3.5:
                n_pol += 1
                break          # count LIGAND ATOMS engaged, not atom PAIRS: pairs would
                               # double-count a single donor seeing two acceptors and turn
                               # the param into a proxy for local protein density.
    out['polar_contacts'] = float(n_pol)

    free = sasa(lig)
    if free > 1.0:
        bound = sasa(lig, context=near)
        buried = 100.0 * (free - bound) / free
        out['buried_sasa_per_ha'] = buried / max(len(lig), 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/pocket_pairs/pairs.jsonl')
    ap.add_argument('--out', default='data/instructions/pocket_extra_v1.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs)]
    if a.limit:
        rows = rows[:a.limit]
    need = sorted(set([r['pdb_a'] for r in rows] + [r['pdb_b'] for r in rows]))
    print('pairs %d   complexes %d' % (len(rows), len(need))); sys.stdout.flush()

    cache = {}
    for i, p in enumerate(need):
        cache[p] = measure(p)
        if (i + 1) % 100 == 0:
            print('  %d/%d  usable %d'
                  % (i + 1, len(need), sum(1 for v in cache.values() if v)))
            sys.stdout.flush()

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    stats = collections.defaultdict(collections.Counter)
    cov = collections.Counter()
    n = 0
    with open(a.out, 'w') as fo:
        for r in rows:
            ma, mb = cache.get(r['pdb_a']), cache.get(r['pdb_b'])
            if not ma or not mb:
                continue
            row = {'a': r['a'], 'b': r['b'], 'protein': r['protein'],
                   'pdb_a': r['pdb_a'], 'pdb_b': r['pdb_b'],
                   'extra_version': EXTRA_VERSION}
            for p in ('polar_contacts', 'buried_sasa_per_ha'):
                va, vb = ma[p], mb[p]
                if va is None or vb is None:
                    row[p + '_dir'] = None; continue
                cov[p] += 1
                d = vb - va
                dirn = 'SAME' if abs(d) < DEADBAND[p] else ('UP' if d > 0 else 'DOWN')
                row[p + '_dir'] = dirn
                row[p + '_delta'] = round(d, 4)
                row[p + '_a'] = round(va, 4)
                stats[p][dirn] += 1
            fo.write(json.dumps(row) + '\n'); n += 1

    print('\n=== EXTRA POCKET PARAMS %s ===' % EXTRA_VERSION)
    print('  pair rows written %d / %d' % (n, len(rows)))
    print('\n  %-20s %8s %8s %8s %8s %8s   %s'
          % ('param', 'COVERAGE', 'UP', 'DOWN', 'SAME', 'MOVED%', 'vs the attacked-C params'))
    for p in ('polar_contacts', 'buried_sasa_per_ha'):
        c = stats[p]; tot = sum(c.values())
        up, dn, sm = c.get('UP', 0), c.get('DOWN', 0), c.get('SAME', 0)
        mv = up + dn
        bal = (min(up, dn) / max(up, dn)) if max(up, dn) else 0.0
        print('  %-20s %8d %8d %8d %8d %7.1f%%   balance %.2f'
              % (p, tot, up, dn, sm, 100.0 * mv / max(tot, 1), bal))
    print('\n  (theta_bd covered 4510 and d_cys_scaffold 4516 of 5689 -- both lose the')
    print('   non-carbon electrophiles. These two never look for the electrophile.)')
    print('  wrote %s' % a.out)


if __name__ == '__main__':
    main()
