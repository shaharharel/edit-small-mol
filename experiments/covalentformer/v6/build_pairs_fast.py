#!/usr/bin/env python3
"""v6 pair corpus, parallel build.

The serial version re-opened and re-parsed a gzipped mmCIF for EVERY pair, which put the full run at
~19 hours. Two changes: pairs are grouped by the structure they need so each file is parsed once,
and groups are spread over all cores. Same output, same filters.
"""
import collections, gzip, itertools, json, os, sys
import numpy as np
from multiprocessing import Pool, cpu_count
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, Descriptors, rdFMCS
RDLogger.DisableLog('rdApp.*')

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
STD = set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
COFACTOR = set('NAD NAP NDP NAI NDE FAD FMN FDA ATP ADP AMP GTP GDP GMP ANP ACP AGS GNP GSP SAM '
               'SAH COA ACO MLC HEM HEC HEA DHE UDP UTP CTP TTP UMP CMP UGA PLP TPP TDP BTN FES '
               'SF4 MGD GSH GDN PQQ F43 COM B12 CLA BCL PHO'.split())


def pocket_of(pid, lig3):
    f = os.path.join(D, 'cif', '%s.cif.gz' % pid)
    if not os.path.exists(f):
        return None, None
    P, L, hdr, idx = [], [], [], None
    try:
        with gzip.open(f, 'rt', errors='ignore') as fh:
            for ln in fh:
                if ln.startswith('_atom_site.'):
                    hdr.append(ln.strip().split('.', 1)[1]); idx = None; continue
                if hdr and idx is None:
                    try:
                        idx = {k: hdr.index(k) for k in ('type_symbol', 'label_atom_id',
                               'label_comp_id', 'auth_seq_id', 'Cartn_x', 'Cartn_y', 'Cartn_z')}
                    except ValueError:
                        return None, None
                if not ln.startswith(('ATOM', 'HETATM')):
                    if P or L: break
                    continue
                p = ln.split()
                if idx is None or len(p) <= max(idx.values()): continue
                if p[idx['type_symbol']] in ('H', 'D'): continue
                comp = p[idx['label_comp_id']]
                try:
                    xyz = [float(p[idx['Cartn_x']]), float(p[idx['Cartn_y']]), float(p[idx['Cartn_z']])]
                except ValueError:
                    continue
                if comp == lig3:
                    L.append({'el': p[idx['type_symbol']], 'xyz': xyz, 'name': p[idx['label_atom_id']]})
                elif comp in STD:
                    P.append({'el': p[idx['type_symbol']], 'xyz': xyz, 'res': comp,
                              'rn': p[idx['auth_seq_id']], 'name': p[idx['label_atom_id']]})
    except Exception:
        return None, None
    if len(L) < 5 or len(P) < 20:
        return None, None
    Lx = np.array([a['xyz'] for a in L]); Px = np.array([a['xyz'] for a in P])
    near = np.where(np.linalg.norm(Px[:, None, :] - Lx[None, :, :], axis=2).min(1) <= 8.0)[0]
    if len(near) < 10:
        return None, None
    return [P[i] for i in near], L


def work(job):
    """One structure, all the pairs that need it."""
    pid, lig_lo, items = job
    P, L = pocket_of(pid, lig_lo)
    if not P:
        return []
    Px = np.array([a['xyz'] for a in P])
    ev = []
    for a in L:
        d = np.linalg.norm(Px - np.array(a['xyz']), axis=1)
        j = int(np.argmin(d))
        ev.append({'el': a['el'], 'name': a['name'].strip('"'), 'burial': int((d <= 8.0).sum()),
                   'nearest': '%s%s.%s' % (P[j]['res'], P[j]['rn'], P[j]['name'].strip('"')),
                   'd': round(float(d[j]), 2)})
    poc = sorted({'%s%s' % (p['res'], p['rn']) for p in P})
    out = []
    for (target, paper, hi, t, smi_lo, smi_hi) in items:
        A, B = Chem.MolFromSmiles(smi_lo), Chem.MolFromSmiles(smi_hi)
        if A is None or B is None: continue
        try:
            r = rdFMCS.FindMCS([A, B], timeout=5, completeRingsOnly=False)
            ncore = r.numAtoms
            core = Chem.MolFromSmarts(r.smartsString)
            ca = A.GetNumAtoms() - len(A.GetSubstructMatch(core))
            cb = B.GetNumAtoms() - len(B.GetSubstructMatch(core))
        except Exception:
            ncore, ca, cb = 0, 0, 0
        out.append({'target': target, 'paper': paper, 'pdb': pid,
                    'hit_code': lig_lo, 'ana_code': hi,
                    'hit': Chem.MolToSmiles(A), 'analogue': Chem.MolToSmiles(B),
                    'tanimoto': t, 'mcs_atoms': ncore,
                    'heavy': [A.GetNumHeavyAtoms(), B.GetNumHeavyAtoms()],
                    'clogp': [round(Descriptors.MolLogP(A), 2), round(Descriptors.MolLogP(B), 2)],
                    'tpsa': [round(Descriptors.TPSA(A), 1), round(Descriptors.TPSA(B), 1)],
                    'rotb': [Descriptors.NumRotatableBonds(A), Descriptors.NumRotatableBonds(B)],
                    'hbd': [Descriptors.NumHDonors(A), Descriptors.NumHDonors(B)],
                    'hba': [Descriptors.NumHAcceptors(A), Descriptors.NumHAcceptors(B)],
                    'changed_hit': ca, 'changed_ana': cb,
                    'pocket_residues': poc, 'ligand_evidence': ev})
    return out


if __name__ == '__main__':
    M = json.load(open(D + '/meta.json')); SMI = json.load(open(D + '/ligand_smiles.json'))
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
    jobs = collections.defaultdict(list); cap = collections.Counter()
    for key, ligs in sorted(series.items()):
        if len(ligs) < 2: continue
        for a, b in itertools.combinations(sorted(ligs), 2):
            if a in COFACTOR or b in COFACTOR: continue
            if get(a) is None or get(b) is None: continue
            if not (10 <= hv[a] <= 70 and 10 <= hv[b] <= 70): continue
            if Chem.MolToSmiles(mol[a], isomericSmiles=False) == Chem.MolToSmiles(mol[b], isomericSmiles=False): continue
            t = DataStructs.TanimotoSimilarity(fp[a], fp[b])
            if t < 0.30: continue
            if cap[key[0]] >= 500: continue
            cap[key[0]] += 1
            lo, hi = (a, b) if (hv[a], a) <= (hv[b], b) else (b, a)
            pid = sorted(series[key][lo])[0]
            jobs[(pid, lo)].append((key[0], key[1], hi, round(t, 3), SMI[lo], SMI[hi]))
    joblist = [(p, l, v) for (p, l), v in jobs.items()]
    npairs = sum(len(v) for v in jobs.values())
    print('pairs %s over %s distinct structures  (%.1f pairs/structure)'
          % (format(npairs, ','), format(len(joblist), ','), npairs / len(joblist)), flush=True)
    nproc = max(1, cpu_count() - 1)
    print('using %d processes' % nproc, flush=True)
    out = open(D + '/v6_pairs.jsonl', 'w')
    kept = done = 0
    with Pool(nproc) as pool:
        for res in pool.imap_unordered(work, joblist, chunksize=16):
            done += 1
            for r in res:
                out.write(json.dumps(r) + '\n'); kept += 1
            if done % 1000 == 0:
                print('  structures %d/%d   pairs kept %s' % (done, len(joblist), format(kept, ',')), flush=True)
    out.close()
    print('\nDONE: %s pairs -> v6_pairs.jsonl' % format(kept, ','))