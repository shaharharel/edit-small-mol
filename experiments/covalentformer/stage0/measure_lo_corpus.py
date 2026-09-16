"""WHAT THE REAL ChEMBL PAIR CORPUS CAN CARRY -- and whether the acrylamide requirement is real.

data/lo_corpus/lo_pairs_v1.csv is 307,555 REAL ChEMBL lead-optimisation pairs across 691 targets,
63,860 unique molecules, already TC-filtered (mean 0.628). It has existed since June and nothing
in the steering work has ever read it. The steering pool instead is 17,409 of OUR OWN model's
generations, 100% acrylamide (#194).

THE QUESTION THAT DECIDES THE ACRYLAMIDE REQUIREMENT.
Every one of the six geom params was defined FROM THE ELECTROPHILE. That is why a warhead is
mandatory -- not because the chemistry demands it, but because the reference atom was chosen to
be the electrophilic carbon. #137 already showed attachment-point reach works on 40/40
NON-warhead fragments, so the alternative reference exists and is tested.

So this measures BOTH definitions side by side on the same 307k pairs:
  ELECTROPHILE-REFERENCED : needs a warhead. Coverage is the cost of the current definition.
  ATTACHMENT-REFERENCED   : needs only a ring and an exit vector. Coverage is what dropping the
                            acrylamide requirement would buy.
The gap between those two coverages IS the price of the acrylamide constraint, in rows.

Both ends of the range are reported: COVERAGE (how many pairs can be labelled at all) and SIGNAL
(how many of those actually CHANGE the param). A param with high coverage and no signal is as
useless as one with no coverage, and both numbers must be visible before anyone builds on this.
"""
import os, json, argparse
from collections import Counter
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

SRC = '/Users/shaharharel/Documents/github/edit-small-mol/data/lo_corpus/lo_pairs_v1.csv'
WARHEADS = [
    ('acrylamide', '[CH2]=[CH]C(=O)N'), ('propiolamide', 'C#CC(=O)N'),
    ('chloroacetamide', 'ClCC(=O)N'), ('vinylsulfone', 'C=CS(=O)(=O)'),
    ('fluorosulfate', 'OS(=O)(=O)F'), ('epoxide', 'C1CO1'),
    ('boronic', 'B(O)O'), ('aldehyde', '[CX3H1](=O)'), ('nitrile', '[NX1]#[CX2]'),
]
_P = None


def _pats():
    global _P
    if _P is None:
        _P = [(k, Chem.MolFromSmarts(v)) for k, v in WARHEADS]
    return _P


def wclass(m):
    return tuple(k for k, p in _pats() if m.HasSubstructMatch(p))


def elec_atom(m):
    for _, p in _pats():
        h = m.GetSubstructMatch(p)
        if h:
            return h[0]
    return None


def _from_ref(m, ref):
    """(path to nearest ring, rotatable bonds within 3 bonds) measured FROM atom `ref`."""
    dm = Chem.GetDistanceMatrix(m)
    ring = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    if not ring:
        return None, None
    path = int(min(dm[ref][i] for i in ring))
    near = {i for i in range(m.GetNumAtoms()) if dm[ref][i] <= 3}
    flex = sum(1 for b in m.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
               and b.GetBeginAtomIdx() in near and b.GetEndAtomIdx() in near
               and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)
    return path, flex


def attach_atom(m):
    """ATTACHMENT-REFERENCED anchor: the acyclic terminal heavy atom farthest from the ring
    system -- the exit vector of the longest substituent. Defined for any molecule with a ring,
    warhead or not. This is the #137 reference, which worked on 40/40 non-warhead fragments."""
    ring = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    if not ring:
        return None
    dm = Chem.GetDistanceMatrix(m)
    best, bd = None, -1
    for i in range(m.GetNumAtoms()):
        a = m.GetAtomWithIdx(i)
        if a.IsInRing():
            continue
        d = min(dm[i][r] for r in ring)
        if d > bd:
            best, bd = i, d
    return best


def one(rec):
    s1, s2 = rec
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return None
    out = {}
    w1, w2 = wclass(m1), wclass(m2)
    out['both_warhead'] = int(bool(w1) and bool(w2))
    out['any_warhead'] = int(bool(w1) or bool(w2))
    out['acryl_both'] = int('acrylamide' in w1 and 'acrylamide' in w2)
    if w1 and w2:
        out['d_wclass'] = int(w1 != w2)
        a1, b1 = tuple(x for x in w1 if x != 'nitrile'), tuple(x for x in w2 if x != 'nitrile')
        out['d_wclass_strict'] = int(bool(a1 and b1 and a1 != b1))
        e1, e2 = elec_atom(m1), elec_atom(m2)
        if e1 is not None and e2 is not None:
            p1, f1 = _from_ref(m1, e1)
            p2, f2 = _from_ref(m2, e2)
            if p1 is not None and p2 is not None:
                out['elec_cov'] = 1
                out['elec_d_path'] = int(p1 != p2)
                out['elec_d_flex'] = int(f1 != f2)
    t1, t2 = attach_atom(m1), attach_atom(m2)
    if t1 is not None and t2 is not None:
        p1, f1 = _from_ref(m1, t1)
        p2, f2 = _from_ref(m2, t2)
        if p1 is not None and p2 is not None:
            out['att_cov'] = 1
            out['att_d_path'] = int(p1 != p2)
            out['att_d_flex'] = int(f1 != f2)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--limit', type=int, default=0, help='0 = all pairs')
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--out', default='/tmp/lo_corpus_params.json')
    a = ap.parse_args()

    df = pd.read_csv(SRC, usecols=['input_smiles', 'output_smiles'])
    if a.limit:
        df = df.head(a.limit)
    recs = list(zip(df.input_smiles, df.output_smiles))
    print('pairs to screen: %d  on %d workers' % (len(recs), a.workers), flush=True)

    c = Counter()
    n_ok = 0
    with Pool(a.workers) as pool:
        for n, r in enumerate(pool.imap_unordered(one, recs, chunksize=256), 1):
            if r is None:
                continue
            n_ok += 1
            for k, v in r.items():
                c[k] += v
            if n % 50000 == 0:
                print('  %d/%d' % (n, len(recs)), flush=True)

    def pc(x, d):
        return 100.0 * x / d if d else float('nan')

    ec, ac = c['elec_cov'], c['att_cov']
    print('\n================ MEASURED ================')
    print('pairs parsed                       : %d' % n_ok)
    print('\n-- COVERAGE: how many pairs can be LABELLED at all --')
    print('both sides bear ANY warhead        : %d (%.2f%%)' % (c['both_warhead'], pc(c['both_warhead'], n_ok)))
    print('both sides bear an ACRYLAMIDE      : %d (%.2f%%)   <-- what the current gate demands'
          % (c['acryl_both'], pc(c['acryl_both'], n_ok)))
    print('ELECTROPHILE-referenced labelable  : %d (%.2f%%)' % (ec, pc(ec, n_ok)))
    print('ATTACHMENT-referenced labelable    : %d (%.2f%%)   <-- #137 reference, no warhead needed'
          % (ac, pc(ac, n_ok)))
    if ec:
        print('   coverage ratio attach/elec      : %.1fx' % (ac / ec))
    print('\n-- SIGNAL: of the LABELABLE pairs, how many CHANGE the param --')
    if c['both_warhead']:
        print('WCLASS  (as labelled)  : %d / %d (%.2f%%)'
              % (c['d_wclass'], c['both_warhead'], pc(c['d_wclass'], c['both_warhead'])))
        print('WCLASS  (nitrile strip): %d / %d (%.2f%%)   <-- artifact-hardened'
              % (c['d_wclass_strict'], c['both_warhead'], pc(c['d_wclass_strict'], c['both_warhead'])))
    print('PATH  elec-ref   : %d / %d (%.2f%%)' % (c['elec_d_path'], ec, pc(c['elec_d_path'], ec)))
    print('PATH  attach-ref : %d / %d (%.2f%%)' % (c['att_d_path'], ac, pc(c['att_d_path'], ac)))
    print('FLEX  elec-ref   : %d / %d (%.2f%%)' % (c['elec_d_flex'], ec, pc(c['elec_d_flex'], ec)))
    print('FLEX  attach-ref : %d / %d (%.2f%%)' % (c['att_d_flex'], ac, pc(c['att_d_flex'], ac)))
    print('\nFOR REFERENCE -- USABLE ROW COUNTS, the number that actually matters:')
    print('  current steer_k8 dataset        : ~8,957 rows, 100%% acrylamide, 100%% our own generations')
    print('  attach-ref PATH rows here       : %d' % c['att_d_path'])
    print('  attach-ref FLEX rows here       : %d' % c['att_d_flex'])
    print('  elec-ref   WCLASS rows here     : %d (strict)' % c['d_wclass_strict'])
    json.dump({k: int(v) for k, v in c.items()} | dict(n_ok=n_ok), open(a.out, 'w'), indent=2)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
