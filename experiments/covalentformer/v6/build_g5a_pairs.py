#!/usr/bin/env python3
"""Turn the g5a (same-paper) candidates into a v7_pairs-format file the evidence builder accepts.

The candidate list carries ligand codes and SMILES but no STRUCTURES, and evidence needs two
specific crystals. For a same-paper pair both ligands co-occur in at least one publication, so the
structures are chosen from that shared paper and both must carry the pair's UniProt accession --
the same rule build_pairs_v7.py applies, and the one whose absence caused the v6 bug where 12% of
rows paired two unrelated proteins.

Descriptor fields (clogp/tpsa/rotb/hbd/hba/heavy) are recomputed here rather than copied, so a row
cannot carry a descriptor that disagrees with its own SMILES.
"""
import collections
import json
import os

from rdkit import Chem, RDLogger
from rdkit.Chem import Crippen, Descriptors, rdMolDescriptors
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
OUT = os.path.join(D, 'g5a_pairs.jsonl')

meta = json.load(open(D + '/meta.json'))
cands = [json.loads(l) for l in open(os.path.join(HERE, 'label_queue_g5', 'candidates.jsonl'))]
g5a = [c for c in cands if c.get('same_paper')]
print('g5a same-paper candidates: %s' % format(len(g5a), ','))

# paper -> ligand -> entries, restricted to entries carrying that accession
paper_lig_pdb = collections.defaultdict(lambda: collections.defaultdict(list))
for pid, v in meta.items():
    p = v.get('doi') or v.get('pm')
    if not p:
        continue
    for l in (v.get('lig') or []):
        paper_lig_pdb[p][l].append((pid, set(v.get('unp') or [])))

def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return dict(heavy=m.GetNumHeavyAtoms(), clogp=round(Crippen.MolLogP(m), 2),
                tpsa=round(rdMolDescriptors.CalcTPSA(m), 1),
                rotb=rdMolDescriptors.CalcNumRotatableBonds(m),
                hbd=rdMolDescriptors.CalcNumHBD(m), hba=rdMolDescriptors.CalcNumHBA(m))

rows, drop = [], collections.Counter()
for c in g5a:
    t, paper = c['target'], c.get('shared_paper')
    a, b = c['input_code'], c['analogue_code']
    if not paper:
        drop['no shared paper recorded'] += 1
        continue
    # both structures from that paper, both carrying the target accession
    ca = [pid for pid, unps in paper_lig_pdb[paper].get(a, []) if t in unps]
    cb = [pid for pid, unps in paper_lig_pdb[paper].get(b, []) if t in unps]
    if not ca or not cb:
        # Entries lacking any UniProt annotation (11,831 of the mirror) also land here. They are
        # NOT rescued: without an accession we cannot verify both structures are the same protein,
        # and that is precisely the v6 failure where 12% of rows paired unrelated proteins. 46
        # pairs are given up to keep that guarantee.
        drop['no structure in the shared paper carries the target accession'] += 1
        continue
    # Choose two DISTINCT structures. Taking sorted()[0] on each side independently collided on
    # the same PDB for 91 pairs that had a perfectly good distinct pairing available -- the two
    # ligands simply shared their alphabetically first entry. Search the candidates instead, and
    # only give up when no distinct combination exists.
    pa = pb = None
    for x in sorted(ca):
        for y in sorted(cb):
            if x != y:
                pa, pb = x, y
                break
        if pa:
            break
    if pa is None:
        drop['only one crystal in this paper holds both ligands'] += 1
        continue
    qa, qb = props(c['input_smiles']), props(c['analogue_smiles'])
    if qa is None or qb is None:
        drop['unparseable SMILES'] += 1
        continue
    if abs(qb['heavy'] - qa['heavy']) > 15:
        drop['edit larger than 15 heavy atoms (gen_ev5 would reject)'] += 1
        continue
    rows.append({
        'target': t, 'paper': paper, 'pdb': pa, 'analogue_pdb': pb,
        'input_code': a, 'analogue_code': b,
        'input': c['input_smiles'], 'analogue': c['analogue_smiles'],
        'tanimoto': c['tanimoto'],
        'heavy': [qa['heavy'], qb['heavy']], 'clogp': [qa['clogp'], qb['clogp']],
        'tpsa': [qa['tpsa'], qb['tpsa']], 'rotb': [qa['rotb'], qb['rotb']],
        'hbd': [qa['hbd'], qb['hbd']], 'hba': [qa['hba'], qb['hba']],
        'n_crystal_realisations': min(len(ca), len(cb)), 'other_crystals': [],
        'g5_tier': c['tier'],
    })

print('\nbuilt %s rows -> %s' % (format(len(rows), ','), OUT))
print('dropped %s:' % format(sum(drop.values()), ','))
for k, n in drop.most_common():
    print('   %-62s %s' % (k, format(n, ',')))
have_cif = sum(1 for r in rows
               if os.path.exists(os.path.join(D, 'cif', r['pdb'] + '.cif.gz'))
               and os.path.exists(os.path.join(D, 'cif', r['analogue_pdb'] + '.cif.gz')))
print('\nrows whose BOTH cif files are already on disk: %s / %s'
      % (format(have_cif, ','), format(len(rows), ',')))
with open(OUT, 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + '\n')
