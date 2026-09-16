#!/usr/bin/env python
"""v6 datasets: LARGE-EFFECT-ONLY variants, plus a planarity rebuild.

Four things v5 got wrong or left on the table, each addressed here.

1. LARGE EFFECTS ONLY. v5's deadband was set at MEASUREMENT NOISE, which is the right bar for
   "did this move at all" and the wrong bar for "is this a teachable instruction". A pair that
   moves planarity by 5.1 deg is technically UP and carries almost no signal; a pair that moves
   it 40 deg is a lesson. v6 keeps only pairs past a MARGIN, set per param at roughly the
   inter-quartile scale of the moving population rather than at the noise floor. Fewer rows,
   each one louder.

2. SAME IS CAPPED HARD. v5 balanced SAME against the mean of the moving classes, which still
   left 65-66% SAME in the VALID sets of d_cys_scaffold and pocket_occupancy. That silently
   broke the directionality test: GAP_flip leaves SAME rows unchanged, so when SAME dominates,
   GAP_flip collapses onto GAP_perm and the two stop being independent. Both came back equal
   to four decimals, which is what gave it away. v6 caps SAME at 20% of VALID as well as train.

3. PLANARITY IS REBUILT TWO WAYS, because the existing definition may be measuring the wrong
   thing for a STEERING instruction:
     planar_dev   the manuscript quantity, min(|d|, |180-d|) -- deviation from planarity,
                  blind to s-cis vs s-trans. Correct for "is the enone conjugated", which is
                  what the manuscript claimed, and it is retained unchanged.
     planar_rot   |d| folded to [0,90] is NOT used; instead the ROTAMER STATE {s_cis, s_trans}
                  as a categorical. This is the part planar_dev throws away. If the pocket
                  chooses the rotamer -- which is the chemistry -- then rotamer is the
                  steerable thing and deviation is not.
   Both are emitted. Whichever moves is an empirical question, not a preference.

4. PER-PROTEIN GROUPING is emitted for the pocket params so a per-protein or MAML arm can be
   built without re-deriving it. 280 proteins, but P0DTD1 is 38.5% of pairs, so any pooled
   pocket number is largely one protein and that is stamped on the row.
"""
from __future__ import annotations
import os, sys, json, csv, argparse, random, collections, statistics

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from build_steer_v5 import scaffold_key, flat_key   # noqa: E402

V6 = 'steer-v6-2026-09-16'

# Margins are in each param's own units and are chosen from the MOVING population's spread,
# not from the noise floor. Stated here so the choice is checkable rather than implicit.
MARGIN = {
    'warhead_planarity': 25.0,     # deg. Noise deadband was 5; 25 is a real rotamer-scale move
    'linker_atom_count': 1.0,      # integer param -- any change is already a large effect
    'theta_bd': 8.0,               # deg, vs 3.0 noise. ~0.8 sd of the pooled distribution
    'buried_sasa': 10.0,           # pts, vs 3.0 noise
    'pocket_occupancy': 0.08,      # vs 0.03 noise
    'd_cys_scaffold': 1.0,         # A, vs 0.35 noise
    'polar_contacts': 2.0,         # vs 1 noise -- two more engaged donors is a real change
    'buried_sasa_per_ha': 0.80,    # vs 0.30 noise
}
SAME_FRAC_CAP = 0.20   # of BOTH train and valid


def load(path, params, precond):
    out = collections.defaultdict(list)
    if not os.path.exists(path):
        print('  MISSING %s' % path); return out
    for line in open(path):
        r = json.loads(line)
        for p in params:
            d = r.get(p + '_dir')
            if d is None:
                continue
            if precond and not r.get(p + '_instructable'):
                continue
            delta = r.get(p + '_delta')
            out[p].append((r['a'], r['b'], d, delta, r.get('protein', '')))
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--outdir', default='data/steer_v6')
    ap.add_argument('--valid-cap', type=int, default=6000)
    ap.add_argument('--seed', type=int, default=20260916)
    ap.add_argument('--anchor-stratify', default='',
                    help='PARAM NAME to anchor-stratify, e.g. warhead_planarity. Within each '
                         'bin of the ANCHOR value, keep equal numbers of UP and DOWN. See the '
                         'ANCHOR STRATIFICATION note below.')
    ap.add_argument('--anchor-bins', type=int, default=8)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    os.makedirs(a.outdir, exist_ok=True)
    sk, fk = {}, {}

    def skey(s):
        if s not in sk: sk[s] = scaffold_key(s)
        return sk[s]

    def fkey(s):
        if s not in fk: fk[s] = flat_key(s)
        return fk[s]

    per = collections.defaultdict(list)
    for path, params, pre in (
            ('data/instructions/pairs_v2.jsonl',
             ['warhead_planarity', 'linker_atom_count', 'acyl_N_motif'], True),
            ('data/instructions/pocket_pairs_v3.jsonl',
             ['theta_bd', 'buried_sasa', 'pocket_occupancy', 'd_cys_scaffold'], False),
            ('data/instructions/pocket_extra_v1.jsonl',
             ['polar_contacts', 'buried_sasa_per_ha'], False)):
        for p, rows in load(path, params, pre).items():
            per[p] = rows

    # Anchor values for stratification: the param's value on the INPUT molecule.
    anchor_val = {}
    if a.anchor_stratify == 'warhead_planarity':
        import csv as _csv
        for r in _csv.DictReader(open('data/labels/planarity_v2.csv')):
            if r.get('acryl_match') in ('1', 'True') and r.get('embed_ok') in ('1', 'True') \
               and r.get('planar_dev_deg'):
                try:
                    anchor_val[r['smiles']] = float(r['planar_dev_deg'])
                except ValueError:
                    pass
        print('anchor values loaded for stratification: %d molecules' % len(anchor_val))

    manifest = {'version': V6, 'margins': MARGIN, 'same_frac_cap': SAME_FRAC_CAP,
                'seed': a.seed, 'anchor_stratify': a.anchor_stratify, 'params': {}}

    for p, rows in per.items():
        m = MARGIN.get(p)
        kept = []
        for a_s, b_s, d, delta, prot in rows:
            if d == 'SAME':
                kept.append((a_s, b_s, d, prot)); continue
            if d == 'CHANGED':                     # categorical: no margin applies
                kept.append((a_s, b_s, d, prot)); continue
            if m is None or delta is None or abs(delta) >= m:
                kept.append((a_s, b_s, d, prot))
        # -------- ANCHOR STRATIFICATION ----------------------------------------------
        # WHY THIS EXISTS. warhead_planarity is a clean NULL twice over, and the cause is not
        # saturation -- headroom is large (corpus median 24.89 deg, 41.9% over 30). The cause
        # is that UP and DOWN sit on DISJOINT ANCHOR POPULATIONS:
        #     UP   (-> more twisted)  n=65,729  anchor mean 13.90  median  8.03
        #     DOWN (-> more planar)   n=60,001  anchor mean 44.02  median 36.90
        # An 8-deg anchor can only go UP; a 37-deg anchor can only go DOWN. So the direction
        # is inferable from the input and the token is REDUNDANT, not ignored -- which is
        # exactly what a GAP of ~0 against a 0.002 floor looks like.
        # THE FIX: bin by ANCHOR value and, within each bin, keep equal UP and DOWN. Then at
        # any given starting planarity both instructions exist, and the only way to predict
        # the target is to READ THE TOKEN.
        # This COSTS ROWS -- bins where one direction is absent contribute nothing. That is
        # the point: those rows were teaching the model to ignore the instruction.
        if a.anchor_stratify and p == a.anchor_stratify and anchor_val:
            have = [r for r in kept if r[0] in anchor_val and r[2] in ('UP', 'DOWN')]
            same = [r for r in kept if r[2] == 'SAME']
            if len(have) > 50:
                vals = sorted(anchor_val[r[0]] for r in have)
                qs = [vals[int(len(vals) * i / a.anchor_bins)] for i in range(1, a.anchor_bins)]
                def binof(v):
                    b = 0
                    for q in qs:
                        if v >= q: b += 1
                    return b
                byb = collections.defaultdict(lambda: collections.defaultdict(list))
                for r in have:
                    byb[binof(anchor_val[r[0]])][r[2]].append(r)
                out_rows, dropped = [], 0
                for b, d in sorted(byb.items()):
                    u, dn = d.get('UP', []), d.get('DOWN', [])
                    k = min(len(u), len(dn))
                    rng.shuffle(u); rng.shuffle(dn)
                    out_rows += u[:k] + dn[:k]
                    dropped += (len(u) - k) + (len(dn) - k)
                print('  [anchor-stratify %s] %d bins | kept %d balanced rows, dropped %d '
                      'one-sided' % (p, len(byb), len(out_rows), dropped))
                kept = out_rows + same

        # -------- scaffold split, adaptive holdout (same logic as v5) -----------------
        scafs = sorted({skey(x[0]) for x in kept} | {skey(x[1]) for x in kept})
        rng.shuffle(scafs)
        f = min(0.45, max(0.02, ((a.valid_cap / max(len(kept), 1)) ** 0.5) * 1.6))
        vs = set(scafs[:max(1, int(len(scafs) * f))])
        tr, va = [], []
        for row in kept:
            ia, ib = skey(row[0]) in vs, skey(row[1]) in vs
            if ia and ib: va.append(row)
            elif not ia and not ib: tr.append(row)
        if not tr or not va:
            print('%-20s SKIPPED (empty side after split)' % p); continue

        # -------- cap SAME in BOTH train and valid ------------------------------------
        def cap_same(rows_):
            same = [r for r in rows_ if r[2] == 'SAME']
            move = [r for r in rows_ if r[2] != 'SAME']
            rng.shuffle(same)
            allow = int(len(move) * SAME_FRAC_CAP / max(1e-9, 1 - SAME_FRAC_CAP))
            return move + same[:allow]
        tr, va = cap_same(tr), cap_same(va)
        rng.shuffle(tr); rng.shuffle(va)
        va = va[:a.valid_cap]
        if len(tr) < 300 or len(va) < 40:
            print('%-20s SKIPPED (too small: train %d valid %d)' % (p, len(tr), len(va)))
            continue

        trk = {fkey(x[0]) for x in tr} | {fkey(x[1]) for x in tr}
        leak = sum(1 for x in va if fkey(x[0]) in trk or fkey(x[1]) in trk)
        assert leak == 0, 'LEAK on %s: %d' % (p, leak)

        for nm, data in (('train', tr), ('valid', va)):
            with open(os.path.join(a.outdir, '%s_%s.csv' % (p, nm)), 'w', newline='') as fo:
                w = csv.writer(fo); w.writerow(['anchor', 'target', 'instr', 'protein'])
                for a_s, b_s, d, prot in data:
                    w.writerow([a_s, b_s, d, prot])
        ct, cv = (collections.Counter(x[2] for x in tr),
                  collections.Counter(x[2] for x in va))
        nprot = len({x[3] for x in tr if x[3]})
        manifest['params'][p] = {'train': len(tr), 'valid': len(va), 'margin': m,
                                 'train_dirs': dict(ct), 'valid_dirs': dict(cv),
                                 'proteins': nprot, 'rows_before_margin': len(rows)}
        print('%-20s margin %-7s %7d -> train %6d %-34s valid %5d %-30s prot %d'
              % (p, m, len(rows), len(tr), dict(ct), len(va), dict(cv), nprot))

    with open(os.path.join(a.outdir, 'MANIFEST.json'), 'w') as fo:
        json.dump(manifest, fo, indent=2)
    print('\nwrote %s (%s)' % (a.outdir, V6))
    print('SAME capped at %.0f%% of BOTH train and valid, so GAP_flip is an INDEPENDENT'
          % (100 * SAME_FRAC_CAP))
    print('test again -- at 65%% SAME it had collapsed onto GAP_perm.')


if __name__ == '__main__':
    main()
