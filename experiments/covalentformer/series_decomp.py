"""FEASIBILITY GATE for the retrospective hit-to-lead claim: WHERE did real series actually improve?

THE CLAIM WE WANT. Take a covalent program, anchor on an early compound, generate, and show the
generated population is better than what mol2mol/LibInvent produce from the same anchor. CovInDB
supports this with real power: 8,132 exact nM IC50 rows over 219 targets, and 244 (target, paper)
series carrying >=10 compounds -- so n for the cluster bootstrap is 244 PROGRAMS, not 244 molecules
and emphatically not the number of molecules generated.

THE GATE. Our training cuts are WARHEAD-DIRECTED: every example puts the electrophile in the
generated half, so the model edits the reactive end and leaves the core alone. That is a real scope
limit, not a tunable. If a program's potency gain came from swapping the hinge binder, this
architecture structurally cannot reproduce it, and a null result there would say nothing about
geometry -- it would only say we picked the wrong programs.

So before committing to case studies, measure, per program: between the WEAKEST and STRONGEST
compounds, did the change land at the reactive end or on the core?

    same Bemis-Murcko scaffold, different warhead/linker  -> REACTIVE-END program. Addressable.
    different Bemis-Murcko scaffold                       -> CORE-HOP program.     Not addressable.

Programs are ranked by how much potency moved, so the case studies are chosen on evidence rather
than on which targets are famous.

WHAT THIS DOES NOT DO. It does not tell us our model will win -- only which contests it is entitled
to enter. Reporting a win on programs selected AFTER seeing generation results would be selection on
the outcome; the addressable list has to be fixed here, before any model is scored against it.
"""
import os, sys, csv, json, argparse, collections
import numpy as np
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem, Descriptors
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

SRC = 'data/covbinder/raw_covindb2/CovInDB_All.csv'
MIN_N = 10
WARHEADS = [
    ('michael_terminal', '[CH2]=[CH]C(=O)[NX3]'),
    ('michael_sub', '[CX3]=[CX3][CX3](=O)[NX3]'),
    ('haloacetamide', '[F,Cl,Br,I][CH2]C(=O)[NX3]'),
    ('vinylsulfone', '[CX3]=[CX3][SX4](=O)(=O)'),
    ('propiolamide', 'C#CC(=O)[NX3]'),
]
PATS = [(n, Chem.MolFromSmarts(s)) for n, s in WARHEADS]


def wclass(m):
    for n, p in PATS:
        if p is not None and m.HasSubstructMatch(p):
            return n
    return 'none'


def bm(m):
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m)
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default=SRC)
    ap.add_argument('--out', default='experiments/covalentformer/data/series_decomp.json')
    a = ap.parse_args()

    import pandas as pd
    d = pd.read_csv(a.src, low_memory=False)
    # Activity_type is stored with HTML tags: 'IC<sub>50</sub>'. A plain contains('IC50') matches
    # 18 of 9560 rows and silently produces a "there is no data" conclusion.
    d['act'] = d.Activity_type.astype(str).str.replace(r'<[^>]+>', '', regex=True).str.upper().str.strip()
    q = d[(d.act == 'IC50') & (d.Relation.astype(str).str.strip() == '=')
          & (d.Unit.astype(str).str.strip() == 'nM')].copy()
    q['v'] = pd.to_numeric(q.Value, errors='coerce')
    q = q[q.v > 0].copy()
    q['pIC50'] = 9 - np.log10(q.v)
    print('usable IC50 rows %d | targets %d | unique SMILES %d'
          % (len(q), q.Target_Gene.nunique(), q.SMILES.nunique()))

    cache = {}

    def prep(smi):
        if smi not in cache:
            m = Chem.MolFromSmiles(smi)
            cache[smi] = None if m is None else (
                m, bm(m), wclass(m), AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048))
        return cache[smi]

    out, stats = [], collections.Counter()
    for (tgt, ref), g in q.groupby(['Target_Gene', 'Reference']):
        g = g.dropna(subset=['SMILES'])
        if len(g) < MIN_N:
            continue
        g = g.sort_values('pIC50')
        lo, hi = g.iloc[0], g.iloc[-1]
        a_, b_ = prep(lo.SMILES), prep(hi.SMILES)
        if a_ is None or b_ is None:
            stats['unparseable'] += 1
            continue
        span = float(hi.pIC50 - lo.pIC50)
        same_core = (a_[1] is not None and a_[1] == b_[1])
        same_wh = (a_[2] == b_[2])
        tc = float(DataStructs.TanimotoSimilarity(a_[3], b_[3]))
        if same_core and not same_wh:
            kind = 'reactive_end'          # core kept, warhead class changed
        elif same_core and same_wh:
            kind = 'decoration'            # core and warhead class kept; linker/substituents moved
        else:
            kind = 'core_hop'              # scaffold changed
        stats[kind] += 1
        out.append(dict(target=tgt, reference=str(ref)[:110], n=int(len(g)), span=round(span, 2),
                        kind=kind, tanimoto=round(tc, 3),
                        wh_lo=a_[2], wh_hi=b_[2],
                        pIC50_lo=round(float(lo.pIC50), 2), pIC50_hi=round(float(hi.pIC50), 2)))

    print('\nPROGRAMS with >=%d compounds: %d' % (MIN_N, len(out)))
    tot = sum(stats[k] for k in ('reactive_end', 'decoration', 'core_hop'))
    for k in ('reactive_end', 'decoration', 'core_hop'):
        print('  %-13s %4d  (%.0f%%)' % (k, stats[k], 100 * stats[k] / max(tot, 1)))
    addressable = [r for r in out if r['kind'] in ('reactive_end', 'decoration')]
    print('\n  ADDRESSABLE by a warhead-directed editor: %d / %d programs (%.0f%%)'
          % (len(addressable), len(out), 100 * len(addressable) / max(len(out), 1)))
    print('  (reactive_end + decoration keep the Bemis-Murcko core, which is what our cuts edit)')

    addressable.sort(key=lambda r: -r['span'])
    print('\n  TOP CANDIDATE CASE STUDIES (largest potency gain, core retained):')
    print('  %-12s %5s %6s %-13s %7s  %s' % ('target', 'n', 'span', 'kind', 'Tc', 'pIC50 lo->hi'))
    for r in addressable[:14]:
        print('  %-12s %5d %6.1f %-13s %7.2f  %.1f -> %.1f'
              % (r['target'][:12], r['n'], r['span'], r['kind'], r['tanimoto'],
                 r['pIC50_lo'], r['pIC50_hi']))

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    json.dump({'programs': out, 'counts': dict(stats), 'min_n': MIN_N,
               'n_addressable': len(addressable)}, open(a.out, 'w'), indent=1)
    print('\n  wrote %s' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
