"""#195's NEXT STEP: tighten scaffold-matched pairs to EDIT-SIZED pairs, and re-count.

#195 counted pairs that merely share a Bemis-Murcko scaffold. That is a weaker relation than the
parent->variant edit the model is trained on. This applies an MMP-style constraint -- a large
common core (MCS) and a bounded size change -- and re-counts. Counts are EXPECTED to fall; #195
said so before this was run, so a drop is not a surprise to be explained away afterwards.

WHICH NULL APPLIES, stated before any number, because I nearly ran the wrong one.
#143/#176 killed delta-geometry with a DESCRIPTOR NULL: that label claimed to carry 3D conformer
information and an ECFP4 fingerprint with no conformer input reproduced it, so it was a
topological restatement. wclass/path/flex make NO such claim -- they are topological BY
CONSTRUCTION (a SMARTS match; a bond-path integer; a rotatable-bond count). Asking whether ECFP4
predicts them is guaranteed to answer yes and would prove nothing here. The live risk is the #26
shape instead -- a conditioning label that is a deterministic function of the input -- and the
null that bites that is the PERMUTATION null on the steering task, which score_steer_arms.py
already implements. No descriptor null is run here; its absence is deliberate and stated.

PARALLEL: the serial version pegged one core of sixteen and its progress line only fired between
scaffolds, so it reported nothing for a minute and could not be estimated. Pairs are independent,
so they go through a Pool with a real progress counter.
"""
import os, json, argparse
from collections import defaultdict, Counter
from multiprocessing import Pool
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
SRC = os.path.join(ROOT, 'data/covbinder/raw_covindb2/CovInDB_All.csv')
WARHEADS = [
    ('acrylamide', '[CH2]=[CH]C(=O)N'), ('propiolamide', 'C#CC(=O)N'),
    ('chloroacetamide', 'ClCC(=O)N'), ('vinylsulfone', 'C=CS(=O)(=O)'),
    ('fluorosulfate', 'OS(=O)(=O)F'), ('epoxide', 'C1CO1'),
    ('boronic', 'B(O)O'), ('aldehyde', '[CX3H1](=O)'), ('nitrile', '[NX1]#[CX2]'),
]
_P = None
_ARGS = None


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


def _init(args):
    global _ARGS
    _ARGS = args


def one_pair(rec):
    """(s1,w1,p1,f1,h1, s2,w2,p2,f2,h2) -> dict of flags, or None if it fails a gate."""
    s1, w1, p1, f1, h1, s2, w2, p2, f2, h2 = rec
    if abs(h1 - h2) > _ARGS['max_heavy_delta']:
        return dict(size_ok=0, mmp=0)
    m1, m2 = Chem.MolFromSmiles(s1), Chem.MolFromSmiles(s2)
    if m1 is None or m2 is None:
        return dict(size_ok=1, mmp=0)
    try:
        res = rdFMCS.FindMCS([m1, m2], timeout=_ARGS['mcs_timeout'],
                             ringMatchesRingOnly=True, completeRingsOnly=True)
    except Exception:
        return dict(size_ok=1, mmp=0)
    if res.numAtoms < _ARGS['min_core_frac'] * min(h1, h2):
        return dict(size_ok=1, mmp=0, timeout=int(bool(res.canceled)))
    w1, w2 = tuple(w1), tuple(w2)
    a2 = tuple(x for x in w1 if x != 'nitrile')
    b2 = tuple(x for x in w2 if x != 'nitrile')
    strict = bool(a2 and b2 and a2 != b2)
    return dict(size_ok=1, mmp=1, timeout=int(bool(res.canceled)),
                dw=int(w1 != w2), dw_strict=int(strict),
                dp=int(p1 is not None and p2 is not None and p1 != p2),
                df=int(f1 is not None and f2 is not None and f1 != f2),
                trans=(tuple(sorted(['+'.join(a2), '+'.join(b2)])) if strict else None))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--max-heavy-delta', type=int, default=10)
    ap.add_argument('--min-core-frac', type=float, default=0.70)
    ap.add_argument('--mcs-timeout', type=int, default=2)
    ap.add_argument('--workers', type=int, default=14)
    ap.add_argument('--out', default='/tmp/covindb_mmp.json')
    a = ap.parse_args()

    df = pd.read_csv(SRC)
    by = defaultdict(list)
    for s in df['SMILES'].dropna().unique():
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        w = wclass(m)
        if not w:
            continue
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:
            continue
        if not sc:
            continue
        p, f = wpath_flex(m)
        by[sc].append((str(s), w, p, f, m.GetNumHeavyAtoms()))

    recs = []
    for sc, ms in by.items():
        for i in range(len(ms)):
            for j in range(i + 1, len(ms)):
                recs.append(ms[i] + ms[j])
    print('scaffolds with >=2 ligands : %d' % sum(1 for v in by.values() if len(v) > 1), flush=True)
    print('scaffold-matched pairs     : %d' % len(recs), flush=True)
    print('screening on %d workers...' % a.workers, flush=True)

    cfg = dict(max_heavy_delta=a.max_heavy_delta, min_core_frac=a.min_core_frac,
               mcs_timeout=a.mcs_timeout)
    size_ok = mmp = dw = dws = dp = dfx = touts = 0
    trans = Counter()
    with Pool(a.workers, initializer=_init, initargs=(cfg,)) as pool:
        for n, r in enumerate(pool.imap_unordered(one_pair, recs, chunksize=32), 1):
            size_ok += r.get('size_ok', 0); mmp += r.get('mmp', 0)
            touts += r.get('timeout', 0)
            dw += r.get('dw', 0); dws += r.get('dw_strict', 0)
            dp += r.get('dp', 0); dfx += r.get('df', 0)
            if r.get('trans'):
                trans[r['trans']] += 1
            if n % 2000 == 0:
                print('  %d/%d screened, %d edit-sized' % (n, len(recs), mmp), flush=True)

    def pct(x):
        return 100.0 * x / mmp if mmp else float('nan')

    print('\n================ MEASURED ================')
    print('scaffold-matched pairs (#195)      : %d' % len(recs))
    print('survive heavy-atom delta <= %-2d     : %d' % (a.max_heavy_delta, size_ok))
    print('survive MCS core >= %.0f%% of both   : %d   <-- EDIT-SIZED PAIRS'
          % (100 * a.min_core_frac, mmp))
    print('MCS timeouts (result kept anyway)  : %d' % touts)
    print('\n-- of the %d edit-sized pairs --' % mmp)
    print('change WCLASS (as labelled)        : %d (%.2f%%)' % (dw, pct(dw)))
    print('change WCLASS (nitrile stripped)   : %d (%.2f%%)   <-- artifact-hardened' % (dws, pct(dws)))
    print('change PATH                        : %d (%.2f%%)' % (dp, pct(dp)))
    print('change FLEX                        : %d (%.2f%%)' % (dfx, pct(dfx)))
    print('\nFOR REFERENCE:')
    print('  #195 scaffold-only (15,614)      : wclass 42.90%% / 19.32%% strict, path 36.17%%, flex 23.63%%')
    print('  #194 CReM (9,416 variants)       : wclass  1.20%%,  path  0.74%%,  flex  0.04%%')
    if trans:
        print('\ntop GENUINE warhead swaps among edit-sized pairs:')
        for k, v in trans.most_common(12):
            print('   %-44s %d' % (' <-> '.join(k), v))
    json.dump(dict(n_scaffold_pairs=len(recs), n_size_ok=size_ok, n_mmp=mmp, mcs_timeouts=touts,
                   d_wclass=dw, d_wclass_pct=pct(dw),
                   d_wclass_strict=dws, d_wclass_strict_pct=pct(dws),
                   d_path=dp, d_path_pct=pct(dp), d_flex=dfx, d_flex_pct=pct(dfx),
                   params=cfg, transitions={' <-> '.join(k): v for k, v in trans.most_common(20)}),
              open(a.out, 'w'), indent=2)
    print('\nwrote %s' % a.out)


if __name__ == '__main__':
    main()
