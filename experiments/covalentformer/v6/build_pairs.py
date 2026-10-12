#!/usr/bin/env python3
"""Build the v6 pair corpus: surviving same-paper analogue pairs, each with its pocket and evidence.

No hit-to-lead DIRECTION exists for 97% of pairs (affinity on both sides covers 3%), so the task is
stated honestly as analogue generation in a pocket; ranking is left to the potency head at inference.
Pairs are oriented smaller -> larger, the common elaboration motion, with ties broken deterministically.

Evidence per pair mirrors what the v4 chair panel received -- property deltas plus per-atom burial
and contacts on the atoms that CHANGED -- because that is the information a reviewer needs to write
a rationale that is true of this pair rather than plausible in general.
"""
import collections, gzip, itertools, json, math, os, sys
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFMCS
RDLogger.DisableLog('rdApp.*')

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
M = json.load(open(D + '/meta.json'))
SMI = json.load(open(D + '/ligand_smiles.json'))
STD = set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
LIMIT = int(sys.argv[1]) if len(sys.argv) > 1 else 0

ent = {k: v for k, v in M.items() if v['unp'] and (v['doi'] or v['pm'])}
series = collections.defaultdict(dict)
for pid, v in ent.items():
    for u in v['unp']:
        for l in v['lig']:
            series[(u, v['doi'] or v['pm'])].setdefault(l, set()).add(pid)

mol, fp, hv = {}, {}, {}
def get(l):
    if l in mol: return mol[l]
    s = SMI.get(l); m = Chem.MolFromSmiles(s) if s else None; mol[l] = m
    if m is not None:
        fp[l] = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048); hv[l] = m.GetNumHeavyAtoms()
    return m

cand, cap = [], collections.Counter()
for key, ligs in sorted(series.items()):
    if len(ligs) < 2: continue
    for a, b in itertools.combinations(sorted(ligs), 2):
        if get(a) is None or get(b) is None: continue
        if not (10 <= hv[a] <= 70 and 10 <= hv[b] <= 70): continue
        if Chem.MolToSmiles(mol[a], isomericSmiles=False) == Chem.MolToSmiles(mol[b], isomericSmiles=False): continue
        t = DataStructs.TanimotoSimilarity(fp[a], fp[b])
        if t < 0.30: continue
        if cap[key[0]] >= 500: continue
        cap[key[0]] += 1
        lo, hi = (a, b) if (hv[a], a) <= (hv[b], b) else (b, a)   # smaller -> larger, deterministic
        cand.append((key, lo, hi, round(t, 3)))
print('candidate pairs: %s' % format(len(cand), ','), flush=True)
if LIMIT: cand = cand[:LIMIT]


def pocket_of(pid, lig3):
    """Protein atoms within 8 A of the named ligand, plus that ligand's atoms, from one entry."""
    f = os.path.join(D, 'cif', '%s.cif.gz' % pid)
    if not os.path.exists(f): return None, None
    P, L = [], []
    try:
        with gzip.open(f, 'rt', errors='ignore') as fh:
            for ln in fh:
                if not ln.startswith(('ATOM', 'HETATM')): continue
                p = ln.split()
                if len(p) < 13: continue
                el, comp = p[2], p[5]
                if el == 'H': continue
                try: xyz = [float(p[10]), float(p[11]), float(p[12])]
                except Exception: continue
                if comp == lig3: L.append({'el': el, 'xyz': xyz, 'name': p[3]})
                elif comp in STD:
                    P.append({'el': el, 'xyz': xyz, 'res': comp, 'rn': p[8] if p[8].isdigit() else p[16], 'name': p[3]})
    except Exception:
        return None, None
    if not L or not P: return None, None
    Lx = np.array([a['xyz'] for a in L]); Px = np.array([a['xyz'] for a in P])
    d = np.linalg.norm(Px[:, None, :] - Lx[None, :, :], axis=2)
    near = d.min(1) <= 8.0
    return [P[i] for i in np.where(near)[0]], L


def changed_atoms(ma, mb):
    try:
        r = rdFMCS.FindMCS([ma, mb], timeout=5, completeRingsOnly=False)
        core = Chem.MolFromSmarts(r.smartsString)
        ka = set(ma.GetSubstructMatch(core)); kb = set(mb.GetSubstructMatch(core))
        return [i for i in range(ma.GetNumAtoms()) if i not in ka], \
               [i for i in range(mb.GetNumAtoms()) if i not in kb], r.numAtoms
    except Exception:
        return [], [], 0


rows, miss = [], 0
for n, (key, lo, hi, t) in enumerate(cand):
    if n % 200 == 0 and n: print('  %d / %d  (%d skipped)' % (n, len(cand), miss), flush=True)
    pid_lo = sorted(series[key][lo])[0]
    P, L = pocket_of(pid_lo, lo)
    if P is None:
        miss += 1; continue
    ca, cb, ncore = changed_atoms(mol[lo], mol[hi])
    Px = np.array([a['xyz'] for a in P]); Lx = np.array([a['xyz'] for a in L])
    # per-atom evidence on the ligand as crystallised (indices are PDB order, not SMILES order --
    # the index alignment fix belongs to the prompt renderer, not here)
    ev = []
    for i, a in enumerate(L):
        d = np.linalg.norm(Px - np.array(a['xyz']), axis=1)
        j = int(np.argmin(d))
        ev.append({'el': a['el'], 'name': a['name'],
                   'burial': int((d <= 8.0).sum()),
                   'nearest': '%s%s.%s' % (P[j]['res'], P[j]['rn'], P[j]['name']),
                   'd': round(float(d[j]), 2)})
    A, B = mol[lo], mol[hi]
    rows.append({'target': key[0], 'paper': key[1], 'pdb': pid_lo,
                 'hit_code': lo, 'ana_code': hi,
                 'hit': Chem.MolToSmiles(A), 'analogue': Chem.MolToSmiles(B),
                 'tanimoto': t, 'mcs_atoms': ncore,
                 'heavy': [hv[lo], hv[hi]],
                 'clogp': [round(Descriptors.MolLogP(A), 2), round(Descriptors.MolLogP(B), 2)],
                 'tpsa': [round(Descriptors.TPSA(A), 1), round(Descriptors.TPSA(B), 1)],
                 'rotb': [Descriptors.NumRotatableBonds(A), Descriptors.NumRotatableBonds(B)],
                 'hbd': [Descriptors.NumHDonors(A), Descriptors.NumHDonors(B)],
                 'hba': [Descriptors.NumHAcceptors(A), Descriptors.NumHAcceptors(B)],
                 'changed_hit': len(ca), 'changed_ana': len(cb),
                 'pocket_residues': sorted({'%s%s' % (p['res'], p['rn']) for p in P}),
                 'ligand_evidence': ev})
out = os.path.join(D, 'v6_pairs%s.jsonl' % ('_sample' if LIMIT else ''))
with open(out, 'w') as f:
    for r in rows: f.write(json.dumps(r) + '\n')
print('\nwrote %s rows -> %s   (%d skipped for missing structure/pocket)' % (format(len(rows), ','), out, miss))
if rows:
    r = rows[0]
    print('\n--- sample row ---')
    print('  %s  %s -> %s   Tc %.2f  MCS %d atoms' % (r['pdb'], r['hit_code'], r['ana_code'], r['tanimoto'], r['mcs_atoms']))
    print('  heavy %s  clogp %s  tpsa %s  rotb %s' % (r['heavy'], r['clogp'], r['tpsa'], r['rotb']))
    print('  pocket residues: %d   ligand atoms with evidence: %d' % (len(r['pocket_residues']), len(r['ligand_evidence'])))
    print('  first 3 atoms: %s' % r['ligand_evidence'][:3])