#!/usr/bin/env python3
"""Is Tanimoto >= 0.70 enough to mean "same series", or is a scaffold criterion needed?

Answered empirically against the corpus we already trust: whatever our 8,359 same-paper pairs look
like on scaffold measures IS our working definition of a series. A candidate filter that would
reject a large share of those pairs is too strict, however reasonable it sounds.

Three measures, because they disagree in informative ways:
  identical Murcko  the strict reading of "same scaffold"
  scaffold Tanimoto similarity of the two Murcko scaffolds, so a ring swap is near-miss not fail
  MCS coverage      maximum common substructure heavy atoms / heavy atoms of the smaller molecule.
                    This is the closest thing to "one edit on a shared core" without fragmenting.
"""
import itertools, json, random, sys, collections, os
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog('rdApp.*')

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
HERE = os.path.dirname(os.path.abspath(__file__))
src = open(os.path.join(HERE, 'build_pairs_v7.py')).read()
ns = {}; exec(compile(src.split('def ')[0], 'c', 'exec'), ns)
COF, NOTMED, HMIN, HMAX = ns['COFACTOR'], ns['NOT_MEDCHEM'], ns['HEAVY_MIN'], ns['HEAVY_MAX']

def scaf(m):
    try: return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception: return None

def fp(m): return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)

def mcs_cov(a, b):
    r = rdFMCS.FindMCS([a, b], timeout=5, ringMatchesRingOnly=True, completeRingsOnly=False)
    if r.canceled or not r.smartsString: return None
    q = Chem.MolFromSmarts(r.smartsString)
    if q is None: return None
    n = q.GetNumAtoms()
    return n / min(a.GetNumHeavyAtoms(), b.GetNumHeavyAtoms())

def profile(name, pairs, n=400, seed=0):
    rng = random.Random(seed)
    sample = rng.sample(pairs, min(n, len(pairs)))
    same, stan, cov, tc = [], [], [], []
    for sa, sb in sample:
        a, b = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
        if a is None or b is None: continue
        tc.append(DataStructs.TanimotoSimilarity(fp(a), fp(b)))
        ka, kb = scaf(a), scaf(b)
        if ka is not None and kb is not None:
            same.append(1.0 if ka == kb else 0.0)
            ma, mb = Chem.MolFromSmiles(ka), Chem.MolFromSmiles(kb)
            if ma is not None and mb is not None and ma.GetNumAtoms() and mb.GetNumAtoms():
                stan.append(DataStructs.TanimotoSimilarity(fp(ma), fp(mb)))
        c = mcs_cov(a, b)
        if c is not None: cov.append(c)
    m = lambda x: sum(x)/len(x) if x else float('nan')
    q = lambda x,p: sorted(x)[int(p*(len(x)-1))] if x else float('nan')
    print('%-34s n=%-4d  TC %.3f | identical Murcko %5.1f%% | scaffold-TC %.3f | MCS cov %.3f (p10 %.3f)'
          % (name, len(sample), m(tc), 100*m(same), m(stan), m(cov), q(cov,0.10)))

# our existing corpus
have = []
for line in open(D + '/v7_pairs.jsonl'):
    r = json.loads(line); have.append((r['input'], r['analogue']))
print('existing same-paper corpus: %s pairs' % format(len(have), ','))

# candidate cross-paper pairs at TC thresholds
meta = json.load(open(D + '/meta.json')); lig = json.load(open(D + '/ligand_smiles.json'))
by_t = collections.defaultdict(set)
for pid, v in meta.items():
    for u in (v.get('unp') or []):
        for l in (v.get('lig') or []):
            if l not in COF and l not in NOTMED: by_t[u].add(l)
cache = {}
def prep(c):
    if c in cache: return cache[c]
    s = lig.get(c); mm = Chem.MolFromSmiles(s) if s else None
    if mm is None or not (HMIN <= mm.GetNumHeavyAtoms() <= HMAX): cache[c] = None
    else: cache[c] = (mm, fp(mm), Chem.MolToSmiles(mm))
    return cache[c]
havekey = set()
for sa, sb in have:
    a, b = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    if a and b: havekey.add(frozenset((Chem.MolToSmiles(a), Chem.MolToSmiles(b))))

bands = {0.70: [], 0.60: []}
for t, ls in by_t.items():
    ok = [c for c in sorted(ls) if prep(c)]
    if len(ok) < 2: continue
    for x, y in itertools.combinations(ok, 2):
        mx, fx, cx = cache[x]; my, fy, cy = cache[y]
        s = DataStructs.TanimotoSimilarity(fx, fy)
        if s < 0.60: continue
        if frozenset((cx, cy)) in havekey: continue
        for thr in (0.70, 0.60):
            if s >= thr: bands[thr].append((cx, cy))
print('new cross-paper candidates: TC>=0.70 %s | TC>=0.60 %s\n'
      % (format(len(bands[0.70]), ','), format(len(bands[0.60]), ',')))

profile('EXISTING corpus (same paper)', have)
profile('NEW cross-paper, TC>=0.70', bands[0.70])
profile('NEW cross-paper, TC>=0.60', bands[0.60])
