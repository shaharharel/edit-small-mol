#!/usr/bin/env python3
"""Apply the h2l pair filter stack with per-step attrition, and test three direction proxies.

Every filter is reported as a survivor count so we can see which one does the damage, rather than
only the final number.
"""
import json, collections, random, itertools
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdFMCS
RDLogger.DisableLog('rdApp.*')
D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
M = json.load(open(D + '/meta.json'))
SMI = json.load(open(D + '/ligand_smiles.json'))
AFF = set(json.load(open(D + '/affinity_ids.json')))

ent = {k: v for k, v in M.items() if v['unp'] and (v['doi'] or v['pm'])}
series = collections.defaultdict(dict)          # (unp,paper) -> lig -> set(pdb)
for pid, v in ent.items():
    for u in v['unp']:
        for l in v['lig']:
            series[(u, v['doi'] or v['pm'])].setdefault(l, set()).add(pid)

raw = []
for key, ligs in series.items():
    if len(ligs) < 2:
        continue
    for a, b in itertools.combinations(sorted(ligs), 2):
        raw.append((key, a, b))
print('0. same target + same paper                      %s' % format(len(raw), ','))

mol, fp, hv = {}, {}, {}
def get(l):
    if l in mol: return mol[l]
    s = SMI.get(l)
    m = Chem.MolFromSmiles(s) if s else None
    mol[l] = m
    if m is not None:
        fp[l] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        hv[l] = m.GetNumHeavyAtoms()
    return m

s1 = [r for r in raw if get(r[1]) is not None and get(r[2]) is not None]
print('1. both ligands parse from the CCD               %s   (-%.0f%%)' % (format(len(s1), ','), 100*(1-len(s1)/len(raw))))
s2 = [r for r in s1 if 10 <= hv[r[1]] <= 70 and 10 <= hv[r[2]] <= 70]
print('2. both 10-70 heavy atoms (drops buffers/junk)   %s   (-%.0f%%)' % (format(len(s2), ','), 100*(1-len(s2)/len(s1))))
s3 = [r for r in s2 if Chem.MolToSmiles(mol[r[1]], isomericSmiles=False) != Chem.MolToSmiles(mol[r[2]], isomericSmiles=False)]
print('3. not identical / stereoisomer-only             %s   (-%.0f%%)' % (format(len(s3), ','), 100*(1-len(s3)/len(s2))))
s4 = []
for r in s3:
    t = DataStructs.TanimotoSimilarity(fp[r[1]], fp[r[2]])
    if 0.40 <= t <= 0.95:
        s4.append((r, t))
print('4. Tanimoto 0.40-0.95 (an EDIT, not a swap)      %s   (-%.0f%%)' % (format(len(s4), ','), 100*(1-len(s4)/len(s3))))

# MCS is expensive: estimate its extra attrition on a sample
random.seed(0)
samp = random.sample(s4, min(400, len(s4)))
ok = 0
for (key, a, b), t in samp:
    try:
        res = rdFMCS.FindMCS([mol[a], mol[b]], timeout=5, completeRingsOnly=False)
        if res.numAtoms >= 0.5 * min(hv[a], hv[b]):
            ok += 1
    except Exception:
        pass
frac = ok / len(samp)
print('5. MCS >=50%% of the smaller ligand (sampled)     ~%s   (-%.0f%%)  [est. from %d samples]'
      % (format(int(len(s4)*frac), ','), 100*(1-frac), len(samp)))

cap = collections.Counter()
s6 = []
for (key, a, b), t in s4:
    if cap[key[0]] < 500:            # per-TARGET cap
        cap[key[0]] += 1
        s6.append(((key, a, b), t))
print('6. cap 500 pairs per target (anti-Mpro)          %s   (-%.0f%%)' % (format(len(s6), ','), 100*(1-len(s6)/len(s4))))
print('\n   ESTIMATED USABLE PAIRS after all filters:     ~%s' % format(int(len(s6)*frac), ','))
print('   distinct targets: %d  | distinct series: %d'
      % (len({k[0] for (k, a, b), t in s6}), len({k for (k, a, b), t in s6})))

# ---------------- direction proxies ----------------
print('\n=== DIRECTION PROXIES on the surviving pairs ===')
dat = {p: (v.get('date') or '') for p, v in ent.items()}
nd = nh = na = 0
for (key, a, b), t in s6:
    pa, pb = series[key][a], series[key][b]
    da = min((dat[p] for p in pa if dat.get(p)), default='')
    db = min((dat[p] for p in pb if dat.get(p)), default='')
    if da and db and da != db: nd += 1
    if hv[a] != hv[b]: nh += 1
    if (pa & AFF) and (pb & AFF): na += 1
n = len(s6)
print('  (a) deposition date differs        %6d / %d  (%.0f%%)' % (nd, n, 100.0*nd/n))
print('  (b) heavy-atom count differs       %6d / %d  (%.0f%%)' % (nh, n, 100.0*nh/n))
print('  (c) BOTH have measured affinity    %6d / %d  (%.0f%%)' % (na, n, 100.0*na/n))