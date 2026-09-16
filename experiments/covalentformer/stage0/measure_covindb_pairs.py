"""THE GATE ON THE WHOLE DIRECTION, measured before building anything.

#194 established that CReM cannot revive wclass/path/flex: with the warhead gate FULLY relaxed it
changes them in 1.20% / 0.74% / 0.04% of 9,416 variants, and the wclass figure is mostly my own
SMARTS counting incidental scaffold nitriles. So a different PAIR SOURCE is the only route, and
the obvious candidate is matched pairs mined from real covalent ligands.

This counts them. A pair qualifies when two DISTINCT real CovInDB ligands share a Bemis-Murcko
scaffold and differ in the parameter of interest. That is the same shape as the CReM task --
parent, variant, delta -- but sourced from chemistry instead of an operator.

BOTH ENDS OF THE RANGE ARE REPORTED, because a raw pair count means nothing on its own:
  FLOOR   how many pairs exist at all (share a scaffold)                -> the denominator
  SIGNAL  how many of those actually CHANGE each parameter              -> the usable rows
A parameter whose signal count is ~0 is dead under this source too, and that must be visible.

Nothing here is conditional on the answer. No conclusion is printed before its number.
"""
import os, json, argparse
from collections import defaultdict, Counter
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
SRC = os.path.join(ROOT, 'data/covbinder/raw_covindb2/CovInDB_All.csv')

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


def wclass(m):
    return tuple(k for k, p in PATS if m.HasSubstructMatch(p))


def elec_atom(m):
    for _, p in PATS:
        hit = m.GetSubstructMatch(p)
        if hit:
            return hit[0]
    return None


def wpath_flex(m):
    a = elec_atom(m)
    if a is None:
        return None, None
    dm = Chem.GetDistanceMatrix(m)
    ring = [i for i in range(m.GetNumAtoms()) if m.GetAtomWithIdx(i).IsInRing()]
    if not ring:
        return None, None
    path = int(min(dm[a][i] for i in ring))
    near = {i for i in range(m.GetNumAtoms()) if dm[a][i] <= 3}
    flex = sum(1 for b in m.GetBonds()
               if b.GetBondType() == Chem.BondType.SINGLE and not b.IsInRing()
               and b.GetBeginAtomIdx() in near and b.GetEndAtomIdx() in near
               and b.GetBeginAtom().GetDegree() > 1 and b.GetEndAtom().GetDegree() > 1)
    return path, flex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', default='/tmp/covindb_pairs.json')
    ap.add_argument('--max-per-scaffold', type=int, default=40,
                    help='cap pairs enumerated per scaffold; a huge congeneric series would '
                         'otherwise dominate the count. Dropped pairs are REPORTED, not hidden.')
    a = ap.parse_args()

    df = pd.read_csv(SRC)
    smis = [s for s in df['SMILES'].dropna().unique()]
    print('CovInDB unique SMILES: %d' % len(smis), flush=True)

    by_scaf = defaultdict(list)
    n_ok = n_nowarhead = 0
    for s in smis:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        w = wclass(m)
        if not w:
            n_nowarhead += 1
            continue
        try:
            scaf = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            continue
        if not scaf:
            continue
        p, f = wpath_flex(m)
        n_ok += 1
        by_scaf[scaf].append((str(s), w, p, f))

    print('with a recognised warhead AND a Murcko scaffold: %d' % n_ok)
    print('no recognised warhead (excluded)               : %d' % n_nowarhead)
    multi = {k: v for k, v in by_scaf.items() if len(v) > 1}
    print('scaffolds with >=2 ligands                     : %d' % len(multi))

    tot_pairs = capped = 0
    d_wclass = d_path = d_flex = 0
    trans = Counter()
    for scaf, members in multi.items():
        n = len(members)
        full = n * (n - 1) // 2
        lim = a.max_per_scaffold
        pairs = []
        for i in range(n):
            for j in range(i + 1, n):
                pairs.append((members[i], members[j]))
        if len(pairs) > lim:
            capped += len(pairs) - lim
            pairs = pairs[:lim]
        for (s1, w1, p1, f1), (s2, w2, p2, f2) in pairs:
            tot_pairs += 1
            if w1 != w2:
                d_wclass += 1
                trans[tuple(sorted(['+'.join(w1), '+'.join(w2)]))] += 1
            if p1 is not None and p2 is not None and p1 != p2:
                d_path += 1
            if f1 is not None and f2 is not None and f1 != f2:
                d_flex += 1

    def pct(x):
        return 100.0 * x / tot_pairs if tot_pairs else float('nan')

    print('\n================ MEASURED ================')
    print('scaffold-matched pairs enumerated : %d' % tot_pairs)
    print('pairs dropped by the per-scaffold cap (%d): %d' % (a.max_per_scaffold, capped))
    print('-- of those pairs, how many CHANGE each parameter --')
    print('change WCLASS : %d (%.2f%%)' % (d_wclass, pct(d_wclass)))
    print('change PATH   : %d (%.2f%%)' % (d_path, pct(d_path)))
    print('change FLEX   : %d (%.2f%%)' % (d_flex, pct(d_flex)))
    print('\nFOR REFERENCE, the same three under CReM (#194, 9,416 variants):')
    print('   wclass 1.20%  path 0.74%  flex 0.04%')
    if trans:
        print('\ntop wclass pairings:')
        for k, v in trans.most_common(10):
            print('   %-40s %d' % (' <-> '.join(k), v))
    json.dump(dict(n_unique=len(smis), n_ok=n_ok, n_nowarhead=n_nowarhead,
                   n_scaffolds_multi=len(multi), n_pairs=tot_pairs, n_capped=capped,
                   d_wclass=d_wclass, d_wclass_pct=pct(d_wclass),
                   d_path=d_path, d_path_pct=pct(d_path),
                   d_flex=d_flex, d_flex_pct=pct(d_flex),
                   transitions={' <-> '.join(k): v for k, v in trans.most_common(20)}),
              open(a.out, 'w'), indent=2)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
