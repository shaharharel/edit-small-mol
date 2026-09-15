#!/usr/bin/env python
"""Per-param train/valid sets with a MOLECULE-DISJOINT split.

SPLIT RULE -- SCAFFOLD-DISJOINT, and the first attempt is recorded because its failure is
a real property of the corpus, not a typo:

  MEASURED: all-vs-all pairing makes the molecule graph ONE GIANT COMPONENT. On
  warhead_planarity, 575,713 of 576,025 pairs (99.9%) sit in a single connected component;
  there are 24 components total and the second largest holds 230 pairs. A connected-component
  assignment therefore cannot produce a both-endpoints-unseen MOLECULE split at any useful
  ratio -- you either hand the blob to valid (which is what happened: train 248, valid
  575,777) or you get a 0.05% valid set scraped off the periphery. Note the first attempt
  ASSERTED ZERO LEAK and passed: the split was perfectly disjoint and perfectly useless. A
  leak check does not check usability.

  So we partition one level coarser than the molecule. MURCKO SCAFFOLDS are assigned to
  train or valid; a pair is kept only if BOTH endpoints' scaffolds fall on the same side,
  and pairs that cross the boundary are DROPPED and counted. This is strictly stronger than
  molecule-disjoint -- different scaffold implies different molecule -- and it needs no
  component logic, so it cannot degenerate the way the first attempt did.

  Weaker rules have failed here repeatedly. Anchor-disjoint alone left a 14.42% `keep` leak
  once dummy isotopes were stripped, and a separate 21.1% leak sat in `regrow` -- the half
  the model must actually generate -- for three de-leakings running. Both were invisible to
  a raw-string check. So: canonical SMILES, isotopes stripped, stereo removed, and BOTH
  endpoints tested.



BALANCE: SAME rows are DOWNSAMPLED to the mean of the moving classes. "Do not change it" is
a legitimate instruction and is kept as its own level, but on most params SAME is the
majority class, and an unbalanced majority lets an arm score well by always predicting
no-op. We are measuring whether the model FOLLOWS the instruction, so the classes it must
distinguish get comparable mass.

The valid set keeps the NATURAL class mix, not the balanced one -- a rebalanced valid set
makes every pooled number a mix artifact, which has bitten every pooled figure in this
project at least once.
"""
from __future__ import annotations
import os, sys, json, csv, argparse, random, collections

SPLIT_VERSION = 'steer-v5-2026-09-16'

LIGAND_PARAMS = ['warhead_planarity', 'acyl_N_motif', 'linker_atom_count']
POCKET_PARAMS = ['theta_bd', 'buried_sasa', 'pocket_occupancy', 'd_cys_scaffold']


def scaffold_key(smi):
    """Murcko scaffold, stereo- and isotope-blind. Acyclic molecules have an EMPTY scaffold;
    they are keyed by their own flat SMILES so they cannot all collapse into one bucket."""
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return 'RAW:' + smi
    for at in m.GetAtoms():
        at.SetIsotope(0)
    Chem.RemoveStereochemistry(m)
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        k = Chem.MolToSmiles(sc) if sc is not None and sc.GetNumAtoms() else ''
    except Exception:
        k = ''
    return k if k else 'ACYCLIC:' + Chem.MolToSmiles(m)


def flat_key(smi):
    """Stereo- and isotope-blind canonical key. Falls back to the raw string on parse
    failure -- a molecule we cannot canonicalise must still be able to COLLIDE, otherwise
    it silently lands on both sides of the split."""
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return smi
    for at in m.GetAtoms():
        at.SetIsotope(0)
    Chem.RemoveStereochemistry(m)
    try:
        return Chem.MolToSmiles(m)
    except Exception:
        return smi


class DSU:
    def __init__(self):
        self.p = {}

    def find(self, x):
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]; x = self.p[x]
        return x

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[ra] = rb


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ligand', default='data/instructions/pairs_v2.jsonl')
    ap.add_argument('--pocket', default='data/instructions/pocket_pairs_v2.jsonl')
    ap.add_argument('--outdir', default='data/steer_v5')
    ap.add_argument('--valid-scaf-frac', type=float, default=0.22,
                    help='fraction of MURCKO SCAFFOLDS held out; the resulting pair '
                         'fraction is much smaller because cross-boundary pairs are dropped')
    ap.add_argument('--cap', type=int, default=180000, help='max TRAIN rows per param')
    ap.add_argument('--valid-cap', type=int, default=6000,
                    help='max VALID rows, subsampled uniformly to preserve the natural mix')
    ap.add_argument('--seed', type=int, default=20260916)
    a = ap.parse_args()
    rng = random.Random(a.seed)
    scafcache = {}

    def skey(smi):
        if smi not in scafcache:
            scafcache[smi] = scaffold_key(smi)
        return scafcache[smi]

    os.makedirs(a.outdir, exist_ok=True)

    # ---- load every row once, keyed by param ------------------------------------------
    per_param = collections.defaultdict(list)
    keycache = {}

    def key(s):
        if s not in keycache:
            keycache[s] = flat_key(s)
        return keycache[s]

    for path, params, need_precond in ((a.ligand, LIGAND_PARAMS, True),
                                       (a.pocket, POCKET_PARAMS, False)):
        if not os.path.exists(path):
            print('MISSING %s -- skipping' % path); continue
        for line in open(path):
            r = json.loads(line)
            for p in params:
                d = r.get(p + '_dir')
                if d is None:
                    continue
                # ligand params carry an explicit instructable flag (the warhead-class
                # precondition). Pocket pairs are same-protein by construction.
                if need_precond and not r.get(p + '_instructable'):
                    continue
                per_param[p].append((r['a'], r['b'], d))

    manifest = {'split_version': SPLIT_VERSION, 'seed': a.seed, 'params': {}}

    for p, rows in per_param.items():
        if not rows:
            continue
        # ---- scaffold assignment; a pair must be WHOLLY on one side ------------------
        scafs = set()
        for a_s, b_s, _ in rows:
            scafs.add(skey(a_s)); scafs.add(skey(b_s))
        scafs = sorted(scafs)
        rng.shuffle(scafs)
        n_valid_scaf = max(1, int(len(scafs) * a.valid_scaf_frac))
        valid_scaf = set(scafs[:n_valid_scaf])

        tr, va, crossed = [], [], 0
        for row in rows:
            sa, sb = skey(row[0]), skey(row[1])
            ina, inb = sa in valid_scaf, sb in valid_scaf
            if ina and inb:
                va.append(row)
            elif not ina and not inb:
                tr.append(row)
            else:
                crossed += 1

        # ---- balance the TRAIN set only ----------------------------------------------
        by_dir = collections.defaultdict(list)
        for row in tr:
            by_dir[row[2]].append(row)
        moving = {k: v for k, v in by_dir.items() if k != 'SAME'}
        if moving:
            tgt = max(1, sum(len(v) for v in moving.values()) // len(moving))
            same = by_dir.get('SAME', [])
            rng.shuffle(same)
            tr_bal = []
            for k, v in moving.items():
                rng.shuffle(v); tr_bal += v[:a.cap]
            tr_bal += same[:min(len(same), tgt)]
        else:
            tr_bal = tr
        rng.shuffle(tr_bal)
        tr_bal = tr_bal[:a.cap]

        # ---- VERIFY the split, do not trust it ---------------------------------------
        tr_scaf, tr_mol = set(), set()
        for a_s, b_s, _ in tr_bal:
            tr_scaf.add(skey(a_s)); tr_scaf.add(skey(b_s))
            tr_mol.add(key(a_s)); tr_mol.add(key(b_s))
        leak_s = sum(1 for a_s, b_s, _ in va
                     if skey(a_s) in tr_scaf or skey(b_s) in tr_scaf)
        leak_m = sum(1 for a_s, b_s, _ in va
                     if key(a_s) in tr_mol or key(b_s) in tr_mol)
        assert leak_s == 0, 'SCAFFOLD LEAK on %s: %d/%d' % (p, leak_s, len(va))
        assert leak_m == 0, 'MOLECULE LEAK on %s: %d/%d' % (p, leak_m, len(va))
        # BOTH are asserted: scaffold-disjoint should IMPLY molecule-disjoint, and checking
        # the implication is how you find out the scaffold key is doing something unexpected.

        # USABILITY, not just disjointness -- the first split passed its leak check with 248
        # train rows. A split that cannot train is a failed split even at zero leak.
        assert len(tr_bal) > len(va), \
            'DEGENERATE SPLIT on %s: train %d <= valid %d -- the scaffold assignment ' \
            'handed the bulk of the corpus to valid' % (p, len(tr_bal), len(va))
        assert len(tr_bal) >= 400, \
            'UNTRAINABLE SPLIT on %s: only %d train rows' % (p, len(tr_bal))

        for name, data in (('train', tr_bal), ('valid', va)):
            fp = os.path.join(a.outdir, '%s_%s.csv' % (p, name))
            with open(fp, 'w', newline='') as fo:
                w = csv.writer(fo); w.writerow(['anchor', 'target', 'instr'])
                for a_s, b_s, d in data:
                    w.writerow([a_s, b_s, d])

        # Cap valid for scoring speed. Subsampled UNIFORMLY, never per-class: stratifying
        # here would rebalance the valid mix and make every pooled number a mix artifact.
        if len(va) > a.valid_cap:
            rng.shuffle(va); va = va[:a.valid_cap]

        cnt_tr = collections.Counter(d for _, _, d in tr_bal)
        cnt_va = collections.Counter(d for _, _, d in va)
        manifest['params'][p] = {'train': len(tr_bal), 'valid': len(va),
                                 'pairs_dropped_crossing_boundary': crossed,
                                 'train_dirs': dict(cnt_tr), 'valid_dirs': dict(cnt_va),
                                 'leak_verified_zero': True}
        print('%-20s train %7d %-40s valid %6d %-34s dropped-crossing %d (%.0f%%)'
              % (p, len(tr_bal), dict(cnt_tr), len(va), dict(cnt_va),
                 crossed, 100.0*crossed/max(len(rows), 1)))

    with open(os.path.join(a.outdir, 'MANIFEST.json'), 'w') as fo:
        json.dump(manifest, fo, indent=2)
    print('\nwrote %s  (%s)' % (a.outdir, SPLIT_VERSION))
    print('every param asserted 0 leak: both endpoints stereo/isotope-blind unseen')


if __name__ == '__main__':
    main()
