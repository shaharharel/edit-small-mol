#!/usr/bin/env python3
"""Same-series analogue pairs inside a covalent programme, from the local mirror.

Takes the output of find_covalent_programmes.py and turns it into candidate pairs using the
corpus's own pairing criteria -- same protein, same covalent site, Tanimoto >= 0.50, a bounded
heavy-atom change -- so a pair built here is the same KIND of object the model trained on, while the
target is one the corpus has never seen. That separation is the point: mol1 confounds an unseen
target with a predicted pose, and a pair from a deposited crystal does not.

LIGAND STRUCTURES COME FROM THE FULL LOCAL CCD, not from the project's ccd_atoms.json cache. That
cache stores an atom/bond graph (no SMILES) and covers only the ligands the corpus already used, so
it found 0 of 95 ligands for these five programmes. components.cif.gz has all 95.
"""
import argparse
import collections
import gzip
import itertools
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'pdb_mirror'))


def ccd_smiles(codes, path=None):
    """{code: SMILES} for the given CCD codes, from components.cif.gz in one pass."""
    path = path or os.path.join(MIRROR, 'components.cif.gz')
    assert os.path.exists(path), 'no CCD at %s' % path
    want, out, cur = set(codes), {}, None
    with gzip.open(path, 'rt', errors='ignore') as fh:
        for ln in fh:
            if ln.startswith('data_'):
                cur = ln[5:].strip()
            elif cur in want and 'SMILES' in ln:
                p = ln.split()
                if len(p) >= 5 and p[1] in ('SMILES_CANONICAL', 'SMILES'):
                    q = ln.split('"')
                    cand = q[-2] if len(q) >= 3 else p[-1]
                    # several programs deposit a descriptor each; keep the longest, which is the
                    # one that survives round-tripping most often in practice
                    if cur not in out or len(cand) > len(out[cur]):
                        out[cur] = cand
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--programmes', default=os.path.join(HERE, 'covalent_programmes_full.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'programme_pairs.json'))
    ap.add_argument('--tc', type=float, default=0.50, help='corpus pairing threshold')
    ap.add_argument('--max-dha', type=int, default=10)
    a = ap.parse_args()
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')

    prog = json.load(open(a.programmes))
    codes = {h['ligand'] for v in prog.values() for h in v['covalent']}
    smi = ccd_smiles(codes)
    print('ligand codes %d, SMILES recovered %d' % (len(codes), len(smi)))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    out = {}
    for uni, v in prog.items():
        hits = v['covalent']
        if not hits:
            out[uni] = {'label': v['label'], 'site': None, 'n_ligands': 0, 'pairs': []}
            print('%-12s no covalent complex' % v['label'])
            continue
        # the programme's real site is the nucleophile most of its complexes use; a stray hit on
        # another residue is usually a second chain or a crystallisation artefact
        site = collections.Counter('%s%s' % (h['residue'], h['resnum'])
                                   for h in hits).most_common(1)[0][0]
        ligs = collections.defaultdict(set)
        for h in hits:
            if '%s%s' % (h['residue'], h['resnum']) == site:
                ligs[h['ligand']].add(h['pdb'])
        mols = {c: Chem.MolFromSmiles(smi[c]) for c in ligs if c in smi}
        mols = {c: m for c, m in mols.items() if m is not None}
        fps = {c: gen.GetFingerprint(m) for c, m in mols.items()}
        pairs = []
        for x, y in itertools.combinations(sorted(fps), 2):
            tc = DataStructs.TanimotoSimilarity(fps[x], fps[y])
            dha = abs(mols[x].GetNumHeavyAtoms() - mols[y].GetNumHeavyAtoms())
            # Tc 1.0 is the same graph deposited under two codes (stereoisomers, tautomers): there
            # is no edit to describe, so it is not a pair for this purpose.
            if a.tc <= tc < 1.0 and dha <= a.max_dha:
                pairs.append({'tc': round(tc, 3), 'a': x, 'b': y, 'dha': dha,
                              'pdb_a': sorted(ligs[x])[0], 'pdb_b': sorted(ligs[y])[0],
                              'smiles_a': smi[x], 'smiles_b': smi[y],
                              'ha_a': mols[x].GetNumHeavyAtoms(),
                              'ha_b': mols[y].GetNumHeavyAtoms()})
        pairs.sort(key=lambda p: -p['tc'])
        out[uni] = {'label': v['label'], 'site': site, 'n_ligands': len(mols), 'pairs': pairs}
        print('%-12s site %-8s %2d ligands -> %3d pairs' % (v['label'], site, len(mols), len(pairs)))
        for p in pairs[:4]:
            print('    Tc %.3f  %-6s %s -> %-6s %s   |dHA| %d'
                  % (p['tc'], p['a'], p['pdb_a'], p['b'], p['pdb_b'], p['dha']))
    json.dump(out, open(a.out, 'w'), indent=1)
    json.dump(smi, open(os.path.join(HERE, 'ligand_smiles.json'), 'w'), indent=1)
    print('\n%d pairs over %d programmes -> %s'
          % (sum(len(v['pairs']) for v in out.values()), len(out), a.out))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
