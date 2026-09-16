"""FEASIBILITY, MEASURED BEFORE BUILDING ANYTHING.

Can wclass/path/flex be revived at all? My recorded answer blamed CReM ("CReM swaps a scaffold
core and never touches the warhead"). That is an OBSERVATION on 295 pairs, not a mechanism --
and build_steer_pairs.py:144 is `if not vm.HasSubstructMatch(ACR): continue  # G2: warhead must
survive the edit`, which DISCARDS every variant whose warhead changed before anything counts it.
So the 295/295 constancy is partly OUR FILTER measuring itself.

This asks the only question that decides the direction, with G2 relaxed from "must still be an
acrylamide" to "must still bear SOME recognised electrophile":
    of the variants CReM actually proposes, what fraction change the warhead class / the
    attachment->electrophile bond path / the warhead-internal rotatable-bond count?

If that fraction is ~0, CReM really is the blocker and only a new operator or mined pairs help.
If it is material, the blocker was our own gate and the three "dead" params are recoverable here.
Floor and ceiling both reported. No conclusion is hardcoded in any print below.
"""
import os, sys, json, argparse, random
from collections import Counter
from rdkit import Chem, RDLogger
from rdkit.Chem import Descriptors
from crem.crem import mutate_mol
RDLogger.DisableLog('rdApp.*')

# SAME ROOT AS build_steer_pairs.py:34. My first version derived it from __file__ two levels
# up, which lands on covalentformer/ and made the pool glob match ZERO files -- pandas then
# raised 'No objects to concatenate', i.e. an empty pool crashing rather than silently scoring 0.
ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
DB = os.path.join(ROOT, 'data/crem_db/chembl33_sa2_f5.db')

# The nine classes measured in CovInDB. ORDER MATTERS for first-match labelling, so the most
# specific patterns come first; a molecule matching several is recorded as the joined tuple.
WARHEADS = [
    ('acrylamide',      '[CH2]=[CH]C(=O)N'),
    ('propiolamide',    'C#CC(=O)N'),
    ('chloroacetamide', 'ClCC(=O)N'),
    ('vinylsulfone',    'C=CS(=O)(=O)'),
    ('fluorosulfate',   'OS(=O)(=O)F'),
    ('epoxide',         'C1CO1'),
    ('boronic',         'B(O)O'),
    ('aldehyde',        '[CX3H1](=O)'),
    ('nitrile',         '[NX1]#[CX2]'),
]
PATS = [(k, Chem.MolFromSmarts(v)) for k, v in WARHEADS]
assert all(p is not None for _, p in PATS), 'a SMARTS failed to compile'


def wclass(m):
    """Tuple of every recognised electrophile class present. () means none -- NOT a class."""
    return tuple(k for k, p in PATS if m.HasSubstructMatch(p))


def elec_atom(m):
    """Index of the electrophilic carbon of the FIRST matching warhead, or None."""
    for _, p in PATS:
        hit = m.GetSubstructMatch(p)
        if hit:
            return hit[0]
    return None


def wpath_and_flex(m):
    """(bond path electrophile -> nearest ring atom, rotatable bonds within 3 bonds of the
    electrophile). Both are WARHEAD-INTERNAL.

    MY FIRST VERSION USED max(dm[a]) -- the electrophile to the molecule's FARTHEST atom. That is
    a WHOLE-MOLECULE size measure, not warhead topology: every scaffold edit moves it, so it
    reported 49.39% "path changes" and would have read as reviving a dead parameter. It measured
    the thing CReM is known to edit, not the thing `path` names. The scaffold attachment point is
    the nearest ring atom, so the electrophile->ring distance is the warhead-internal path length.
    Returns (None, None) when there is no electrophile, or no ring to attach to."""
    a = elec_atom(m)
    if a is None:
        return None, None
    dm = Chem.GetDistanceMatrix(m)
    ring_atoms = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    if not ring_atoms:
        return None, None
    path = int(min(dm[a][i] for i in ring_atoms))
    near = {i for i in range(m.GetNumAtoms()) if dm[a][i] <= 3}
    flex = sum(1 for b in m.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
               and b.GetBeginAtomIdx() in near and b.GetEndAtomIdx() in near
               and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)
    return path, flex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--n-parents', type=int, default=120)
    ap.add_argument('--max-repl', type=int, default=100)
    ap.add_argument('--seed', type=int, default=20260915)
    ap.add_argument('--out', default='/tmp/warhead_mutability.json')
    a = ap.parse_args()

    import glob, pandas as pd
    df = pd.concat([pd.read_csv(f) for f in
                    glob.glob(os.path.join(ROOT, 'paper/reproducibility/metrics/planar_2d/*.csv'))])
    pool = df[(df.acryl_match == True) & df.smi.notna()].drop_duplicates('smi').smi.tolist()
    rng = random.Random(a.seed)          # SEEDED: this samples parents
    rng.shuffle(pool)
    parents = pool[:a.n_parents]
    print('parents sampled: %d (seed %d) from pool %d' % (len(parents), a.seed, len(pool)),
          flush=True)

    tot = kept_g2 = 0
    d_wclass = d_path = d_flex = 0
    lost_all = 0
    trans = Counter()
    path_deltas, flex_deltas = [], []

    for i, smi in enumerate(parents):
        m = Chem.MolFromSmiles(smi)
        if m is None:
            continue
        p_w = wclass(m)
        p_path, p_flex = wpath_and_flex(m)
        try:
            variants = list(mutate_mol(m, db_name=DB, radius=3, min_size=1, max_size=8,
                                       min_inc=-2, max_inc=2, max_replacements=a.max_repl,
                                       ncores=1))
        except Exception as e:
            print('  parent %d mutate failed: %s' % (i, e), flush=True)
            continue
        for v in variants:
            vm = Chem.MolFromSmiles(v)
            if vm is None:
                continue
            tot += 1
            v_w = wclass(vm)
            # G2 AS SHIPPED: acrylamide must survive. Counted so the two gates are comparable.
            if vm.HasSubstructMatch(PATS[0][1]):
                kept_g2 += 1
            if not v_w:
                lost_all += 1            # no recognised electrophile at all -- a real destruction
                continue
            if v_w != p_w:
                d_wclass += 1
                trans[('+'.join(p_w) or 'none', '+'.join(v_w) or 'none')] += 1
            v_path, v_flex = wpath_and_flex(vm)
            if p_path is not None and v_path is not None and v_path != p_path:
                d_path += 1
                path_deltas.append(v_path - p_path)
            if p_flex is not None and v_flex is not None and v_flex != p_flex:
                d_flex += 1
                flex_deltas.append(v_flex - p_flex)
        if (i + 1) % 20 == 0:
            print('  %d/%d parents, %d variants so far' % (i + 1, len(parents), tot), flush=True)

    def pct(x):
        return 100.0 * x / tot if tot else float('nan')

    res = dict(n_parents=len(parents), n_variants=tot, seed=a.seed,
               kept_by_shipped_G2=kept_g2, kept_by_shipped_G2_pct=pct(kept_g2),
               lost_all_warheads=lost_all, lost_all_pct=pct(lost_all),
               d_wclass=d_wclass, d_wclass_pct=pct(d_wclass),
               d_path=d_path, d_path_pct=pct(d_path),
               d_flex=d_flex, d_flex_pct=pct(d_flex),
               transitions={'%s->%s' % k: v for k, v in trans.most_common(20)})
    print('\n================ MEASURED ================')
    print('variants proposed by CReM          : %d' % tot)
    print('kept by SHIPPED G2 (acrylamide)    : %d (%.2f%%)' % (kept_g2, pct(kept_g2)))
    print('lose EVERY recognised warhead      : %d (%.2f%%)' % (lost_all, pct(lost_all)))
    print('-- with G2 relaxed to "any warhead", among ALL proposed variants --')
    print('change WCLASS                      : %d (%.2f%%)' % (d_wclass, pct(d_wclass)))
    print('change PATH  (elec->nearest ring)     : %d (%.2f%%)' % (d_path, pct(d_path)))
    print('change FLEX  (rot bonds <=3 away)  : %d (%.2f%%)' % (d_flex, pct(d_flex)))
    if trans:
        print('\ntop wclass transitions:')
        for k, v in trans.most_common(10):
            print('   %-34s %d' % ('%s -> %s' % k, v))
    json.dump(res, open(a.out, 'w'), indent=2)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
