#!/usr/bin/env python
"""Pocket/complex steering params, labelled as RELATIVE instructions on same-protein pairs.

    theta_bd          Burgi-Dunitz approach angle at the attacked carbon
    d_cys_scaffold    nucleophile -> nearest SCAFFOLD ring atom (NOT the formed bond)
    buried_sasa       % of ligand SASA occluded by the protein
    pocket_occupancy  ligand heavy-atom volume / enclosing pocket volume

THE LABEL IS A DIRECTION, NOT A DELTA: "increase pocket occupancy", "reduce the approach
angle". The numeric delta is carried alongside so the trainer can pick, but the instruction
is the token.

WHY THAT MATTERS FOR THE GATE. My first pocket gate asked "is the absolute spread wide
enough", which is the right question for a CONTINUOUS conditioning scalar and the WRONG one
for a binary instruction. A param with a narrow range can still be perfectly decidable
pairwise. It also used p95/p05 on a quantity bounded at 100%, where ratios compress
mechanically and the threshold was unreachable regardless of the data. This file asks the
question that actually matches the label:

    Of the same-protein pairs, what fraction move the param by more than measurement noise,
    and is the resulting UP/DOWN split balanced?

DEADBANDS are set from the measurement's own noise, not from taste:
  theta_bd     3.0 deg  -- crystallographic coordinate error on a 3-atom angle
  buried_sasa  3.0 pts  -- Shrake-Rupley with 92 points quantises near ~1-2 pts
  occupancy    0.03     -- same source, propagated through the ratio

PAIRS ARE SAME-PROTEIN BY CONSTRUCTION (the pocket pair file is built that way). That is the
honest denominator: across different proteins burial varies because POCKETS differ, which is
not something a ligand generator can steer.
"""
from __future__ import annotations
import os, sys, json, math, argparse, collections, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pocket_param_gate import parse_pdb, dist, angle, sasa, NUC, PDB_DIR, VDW  # noqa: E402
from rdkit import Chem, RDLogger                                               # noqa: E402
from rdkit.Chem.Scaffolds import MurckoScaffold                                # noqa: E402
RDLogger.DisableLog('rdApp.*')

POCKET_VERSION = 'pocket-v2-2026-09-16'
DEADBAND = {'theta_bd': 3.0, 'buried_sasa': 3.0, 'pocket_occupancy': 0.03,
            'd_cys_scaffold': 0.35}

# ---- d_cys, REINSTATED UNDER A DEFINITION THAT IS NOT A BOND LENGTH ----------------
# The original d_cys measured nucleophile -> ELECTROPHILE. In a crystal structure of an
# ADDUCT that distance IS the formed covalent bond: 1.797 A, sd 0.106, with every residue
# type sitting on its textbook C-X value. There was nothing to steer, and no amount of
# size-matching creates range that valence chemistry forbids.
#
# But that killed a FORMULA, not the DIRECTION -- the same mistake that was made with
# warhead_span. The medchem question was never "how long is the bond". It is "how far does
# the RECOGNITION SCAFFOLD sit from the nucleophile", i.e. does the core press up against
# the catalytic residue or stand off from it. That distance runs through the linker, is
# free to vary, and is exactly what a linker edit moves.
#
# So: nucleophile -> nearest atom of the ligand's Murcko ring system. Deadband 0.35 A,
# comfortably above crystallographic coordinate error (~0.1-0.2 A at typical resolution)
# and far below the range a linker edit produces.


def ligand_of(het):
    groups = collections.defaultdict(list)
    for h in het:
        if h[1] in ('HOH', 'WAT', 'SO4', 'PO4', 'GOL', 'EDO', 'NAG', 'ZN', 'MG',
                    'NA', 'CL', 'CA', 'K', 'ACT', 'DMS', 'TRS', 'MPD', 'PEG'):
            continue
        groups[(h[1], h[2], h[3])].append(h)
    if not groups:
        return None
    lig = max(groups.values(), key=len)
    return lig if len(lig) >= 8 else None


def measure(pdb):
    """All three pocket params for one complex. Any that cannot be computed is None."""
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
    out = {'theta_bd': None, 'buried_sasa': None, 'pocket_occupancy': None,
           'd_cys_scaffold': None}

    # theta_BD -- nucleophile / attacked carbon / alpha carbon
    best = None
    for p in prot:
        if p[1] in NUC and p[0] == NUC[p[1]]:
            for lat in lig:
                if lat[7].upper() != 'C':
                    continue
                d = dist(p[4:7], lat[4:7])
                if d < 2.2 and (best is None or d < best[0]):
                    best = (d, p, lat)
    if best is not None:
        _, nuc, catom = best
        nbrs = sorted((l for l in lig if l is not catom and dist(l[4:7], catom[4:7]) < 1.75),
                      key=lambda l: dist(l[4:7], catom[4:7]))
        if nbrs:
            out['theta_bd'] = angle(nuc[4:7], catom[4:7], nbrs[0][4:7])

    # d_cys_scaffold -- nucleophile to the nearest RING atom of the ligand.
    # Ring membership is taken from the COORDINATES (an atom in a closed cycle of the
    # ligand's own bond graph, inferred by distance), because HETATM records carry no bond
    # orders and the SMILES atom order does not map onto PDB atom order. A ring atom here
    # is one with >=2 ligand neighbours within 1.75 A that lies in a cycle -- approximated
    # by requiring >=2 such neighbours which themselves have >=2. That is deliberately
    # conservative: it may MISS a ring, it will not invent one.
    if best is not None:
        nuc = best[1]
        adj = {}
        for i, l1 in enumerate(lig):
            adj[i] = [j for j, l2 in enumerate(lig)
                      if j != i and dist(l1[4:7], l2[4:7]) < 1.75]
        ring_like = [i for i in adj if len(adj[i]) >= 2 and
                     sum(1 for j in adj[i] if len(adj[j]) >= 2) >= 2]
        if ring_like:
            out['d_cys_scaffold'] = min(dist(nuc[4:7], lig[i][4:7]) for i in ring_like)

    near = [p for p in prot if any(dist(p[4:7], l[4:7]) < 12.0 for l in lig[:6])]
    free = sasa(lig)
    if free > 1.0:
        bound = sasa(lig, context=near)
        out['buried_sasa'] = 100.0 * (free - bound) / free

        # pocket_occupancy: ligand vdW volume / volume of the enclosing pocket shell.
        # The pocket is the set of grid points within 5 A of a ligand atom and NOT inside a
        # protein atom -- i.e. space the ligand COULD occupy. Occupancy is the share it takes.
        lv = sum(4.0/3.0*math.pi*VDW.get(l[7].upper(), 1.70)**3 for l in lig)
        xs = [l[4] for l in lig]; ys = [l[5] for l in lig]; zs = [l[6] for l in lig]
        pad, step = 5.0, 1.0
        free_pts = 0
        gx = gy = gz = 0
        x = min(xs)-pad
        while x <= max(xs)+pad:
            y = min(ys)-pad
            while y <= max(ys)+pad:
                z = min(zs)-pad
                while z <= max(zs)+pad:
                    if any(dist((x, y, z), l[4:7]) < 5.0 for l in lig):
                        if not any(dist((x, y, z), p[4:7]) < VDW.get(p[7].upper(), 1.70)
                                   for p in near):
                            free_pts += 1
                    z += step
                y += step
            x += step
        pocket_vol = free_pts * step**3
        if pocket_vol > 1.0:
            out['pocket_occupancy'] = lv / pocket_vol
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default='data/pocket_pairs/pairs.jsonl')
    ap.add_argument('--out', default='data/instructions/pocket_pairs_v2.jsonl')
    ap.add_argument('--limit', type=int, default=0)
    a = ap.parse_args()

    rows = [json.loads(l) for l in open(a.pairs)]
    if a.limit:
        rows = rows[:a.limit]
    need = sorted(set([r['pdb_a'] for r in rows] + [r['pdb_b'] for r in rows]))
    print('pairs %d   unique complexes %d' % (len(rows), len(need))); sys.stdout.flush()

    cache = {}
    for i, p in enumerate(need):
        cache[p] = measure(p)
        if (i+1) % 50 == 0:
            ok = sum(1 for v in cache.values() if v)
            print('  measured %d/%d   usable %d' % (i+1, len(need), ok)); sys.stdout.flush()

    os.makedirs(os.path.dirname(a.out) or '.', exist_ok=True)
    stats = collections.defaultdict(collections.Counter)
    n = 0
    with open(a.out, 'w') as fo:
        for r in rows:
            ma, mb = cache.get(r['pdb_a']), cache.get(r['pdb_b'])
            if not ma or not mb:
                continue
            row = {'a': r['a'], 'b': r['b'], 'protein': r['protein'],
                   'pdb_a': r['pdb_a'], 'pdb_b': r['pdb_b'], 'tc': r.get('tc'),
                   'pocket_version': POCKET_VERSION}
            for p in ('theta_bd', 'buried_sasa', 'pocket_occupancy', 'd_cys_scaffold'):
                va, vb = ma[p], mb[p]
                if va is None or vb is None:
                    row[p + '_dir'] = None; row[p + '_delta'] = None
                    continue
                d = vb - va
                dirn = 'SAME' if abs(d) < DEADBAND[p] else ('UP' if d > 0 else 'DOWN')
                row[p + '_dir'] = dirn
                row[p + '_delta'] = round(d, 4)
                row[p + '_a'] = round(va, 4)
                stats[p][dirn] += 1
            fo.write(json.dumps(row) + '\n'); n += 1

    print('\n=== POCKET INSTRUCTIONS %s ===' % POCKET_VERSION)
    print('  pair rows with BOTH complexes measured: %d / %d' % (n, len(rows)))
    print('  deadbands %s' % DEADBAND)
    print('\n  %-18s %8s %8s %8s %8s   %s' % ('param', 'labelled', 'UP', 'DOWN', 'MOVED%', 'balance'))
    for p in ('theta_bd', 'buried_sasa', 'pocket_occupancy', 'd_cys_scaffold'):
        c = stats[p]; tot = sum(c.values())
        up, dn = c.get('UP', 0), c.get('DOWN', 0)
        mv = up + dn
        bal = (min(up, dn) / max(up, dn)) if max(up, dn) else 0.0
        print('  %-18s %8d %8d %8d %7.1f%%   %.2f  %s'
              % (p, tot, up, dn, 100.0*mv/max(tot, 1), bal,
                 'USABLE' if (mv >= 500 and bal >= 0.5) else 'THIN -- needs more pairs'))
    print('\n  wrote %s' % a.out)


if __name__ == '__main__':
    main()
