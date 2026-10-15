#!/usr/bin/env python3
"""g5, rebuilt: target-grouped pairs at Tanimoto >= 0.70 that share a Murcko scaffold.

The first g5 was described as same-paper and was not: 1,135 of its 1,602 pairs had no DOI and no
PubMed on either structure, so same-paper was unverifiable. Rather than pretend, this build drops
the paper requirement outright and uses a threshold CALIBRATED against the cases where provenance
IS verifiable -- pairs whose two ligands both appear in cited structures:

    P(same paper)      any scaffold    same Murcko
    Tanimoto >= 0.65      79.7%           83.9%
    Tanimoto >= 0.70      81.4%         **85.0%**   <- this build

So the pairs here are not asserted to be one series; they are drawn from a population that is one
series 85% of the time, which is a measured claim rather than a hopeful one.

What is NOT relaxed is the protein identity check: both structures must carry the target's UniProt
accession. That check is what stopped v6 pairing two unrelated proteins, and no threshold argument
substitutes for it. Entries with no UniProt at all are therefore still not rescued.

Structures are chosen to be DISTINCT, and from a shared paper when one exists -- searching for a
distinct combination rather than taking sorted()[0] on each side independently, which collided on
91 pairs in the first build when both ligands shared their alphabetically first entry.

Writes g5_pairs.jsonl, the file the rest of the pipeline already reads. Run:

    python build_g5b_pairs.py
    python gen_ev5_g5.py && python clean_g5.py && python apply_completeness_g5.py
    python build_envelope_g5.py --workers 8
    COVF_ENVELOPE=.../g5_envelope.json python make_full_batches_g5.py
"""
import collections
import itertools
import json
import os
import sys

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Crippen, DataStructs, Descriptors, rdMolDescriptors
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import build_pairs_v7 as B  # noqa: E402  -- EXCLUDE, NULLS, cite_of and the size cuts, reused

D = B.D
OUT = os.path.join(D, 'g5_pairs.jsonl')
TC_FLOOR = float(os.environ.get('COVF_TC', '0.70'))
REQUIRE_SCAFFOLD = os.environ.get('COVF_SCAFFOLD', '1') == '1'
MAX_EDIT = 15            # gen_ev5 rejects anything larger, so dropping it here is honest bookkeeping

M = json.load(open(D + '/meta.json'))
SMI = json.load(open(D + '/ligand_smiles.json'))

# ---- indices -------------------------------------------------------------------------------
lig_entries = collections.defaultdict(list)      # ligand -> [(pdb, accessions, paper)]
by_target = collections.defaultdict(set)         # accession -> {ligand}
for pid, v in M.items():
    unp = {u for u in (v.get('unp') or ()) if str(u).strip() not in B.NULLS}
    paper = B.cite_of(v)
    for l in (v.get('lig') or []):
        if l in B.EXCLUDE:
            continue
        lig_entries[l].append((pid, unp, paper))
        for u in unp:
            by_target[u].add(l)

cache = {}


def prep(l):
    if l in cache:
        return cache[l]
    s = SMI.get(l)
    m = Chem.MolFromSmiles(s) if s else None
    if (m is None or not (B.HEAVY_MIN <= m.GetNumHeavyAtoms() <= B.HEAVY_MAX)
            or Descriptors.NumRotatableBonds(m) > B.ROTB_MAX):
        cache[l] = None
    else:
        try:
            sc = MurckoScaffold.MurckoScaffoldSmiles(mol=m)
        except Exception:                                          # noqa: BLE001
            sc = None
        cache[l] = (AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048),
                    Chem.MolToSmiles(m, isomericSmiles=False), sc, m.GetNumHeavyAtoms())
    return cache[l]


def props(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return dict(heavy=m.GetNumHeavyAtoms(), clogp=round(Crippen.MolLogP(m), 2),
                tpsa=round(rdMolDescriptors.CalcTPSA(m), 1),
                rotb=rdMolDescriptors.CalcNumRotatableBonds(m),
                hbd=rdMolDescriptors.CalcNumHBD(m), hba=rdMolDescriptors.CalcNumHBA(m))


# ---- SMILES pairs the built corpus already holds, so nothing is relabelled -------------------
# BOTH files, and this is not belt-and-braces: v7_evidence.jsonl holds the 7,069 pairs whose
# evidence BUILT, while v7_pairs.jsonl holds all 8,359 that were SELECTED. Deduping against the
# evidence alone let 143 pairs back in -- ones g1-g4 already chose and whose evidence failed, so
# they are neither new nor likely to build this time either.
have = set()
have_codes = set()
for line in open(D + '/v7_evidence.jsonl'):
    r = json.loads(line)
    a = r['MODEL_INPUT'].get('input_smiles')
    b = (r.get('ANSWER') or {}).get('analogue_smiles')
    try:
        have.add(frozenset((Chem.MolToSmiles(Chem.MolFromSmiles(a), isomericSmiles=False),
                            Chem.MolToSmiles(Chem.MolFromSmiles(b), isomericSmiles=False))))
    except Exception:                                              # noqa: BLE001
        pass
for line in open(D + '/v7_pairs.jsonl'):
    r = json.loads(line)
    have_codes.add(frozenset((r['input_code'], r['analogue_code'])))
    try:
        have.add(frozenset((Chem.MolToSmiles(Chem.MolFromSmiles(r['input']), isomericSmiles=False),
                            Chem.MolToSmiles(Chem.MolFromSmiles(r['analogue']),
                                             isomericSmiles=False))))
    except Exception:                                              # noqa: BLE001
        pass
print('already taken by g1-g4: %s SMILES pairs, %s ligand-code pairs'
      % (format(len(have), ','), format(len(have_codes), ',')))

# ---- enumerate -------------------------------------------------------------------------------
drop = collections.Counter()
rows, seen = [], set()
for t, ligs in sorted(by_target.items()):
    ok = [l for l in sorted(ligs) if prep(l)]
    if len(ok) < 2:
        continue
    for a, b in itertools.combinations(ok, 2):
        fa, ca, sa, ha = cache[a]
        fb, cb, sb, hb = cache[b]
        if ca == cb:
            drop['identical ignoring stereochemistry'] += 1
            continue
        if REQUIRE_SCAFFOLD and not (sa and sb and sa == sb):
            drop['different Murcko scaffold'] += 1
            continue
        if DataStructs.TanimotoSimilarity(fa, fb) < TC_FLOOR:
            drop['Tanimoto < %.2f' % TC_FLOOR] += 1
            continue
        if abs(hb - ha) > MAX_EDIT:
            drop['edit larger than %d heavy atoms' % MAX_EDIT] += 1
            continue
        key = frozenset((ca, cb))
        if key in have or frozenset((a, b)) in have_codes:
            drop['already selected by g1-g4 (built corpus or pair list)'] += 1
            continue
        if key in seen:
            drop['duplicate SMILES pair (another target or ligand code)'] += 1
            continue

        ea = [(pid, p) for pid, unps, p in lig_entries[a] if t in unps]
        eb = [(pid, p) for pid, unps, p in lig_entries[b] if t in unps]
        if not ea or not eb:
            # No accession, no pair. Entries with no UniProt at all land here and are NOT rescued:
            # without it we cannot show the two crystals are the same protein, which is the v6 bug.
            drop['no structure carries the target accession'] += 1
            continue
        # distinct structures, preferring a shared paper when one exists
        pick = None
        for x, px in sorted(ea):
            for y, py in sorted(eb):
                if x == y:
                    continue
                if px and py and px == py:
                    pick = (x, y, px)
                    break
                if pick is None:
                    pick = (x, y, None)
            if pick and pick[2]:
                break
        if pick is None:
            drop['only one crystal carries both ligands'] += 1
            continue
        pa, pb, paper = pick
        if not (os.path.exists(os.path.join(D, 'cif', pa + '.cif.gz'))
                and os.path.exists(os.path.join(D, 'cif', pb + '.cif.gz'))):
            drop['cif not on disk'] += 1
            continue
        qa, qb = props(SMI[a]), props(SMI[b])
        if qa is None or qb is None:
            drop['unparseable SMILES'] += 1
            continue
        seen.add(key)
        rows.append({
            'target': t, 'paper': paper, 'pdb': pa, 'analogue_pdb': pb,
            'input_code': a, 'analogue_code': b,
            'input': SMI[a], 'analogue': SMI[b],
            'tanimoto': round(DataStructs.TanimotoSimilarity(fa, fb), 3),
            'murcko': sa,
            'heavy': [qa['heavy'], qb['heavy']], 'clogp': [qa['clogp'], qb['clogp']],
            'tpsa': [qa['tpsa'], qb['tpsa']], 'rotb': [qa['rotb'], qb['rotb']],
            'hbd': [qa['hbd'], qb['hbd']], 'hba': [qa['hba'], qb['hba']],
            'n_crystal_realisations': min(len(ea), len(eb)), 'other_crystals': [],
            'provenance': 'same_paper' if paper else 'no_shared_citation',
        })

print('\nbuilt %s rows -> %s' % (format(len(rows), ','), OUT))
for k, n in drop.most_common():
    print('   dropped  %-56s %s' % (k, format(n, ',')))
prov = collections.Counter(r['provenance'] for r in rows)
print('\nprovenance: %s' % dict(prov))
print('targets %s   mean Tanimoto %.3f'
      % (format(len({r['target'] for r in rows}), ','),
         sum(r['tanimoto'] for r in rows) / max(len(rows), 1)))
with open(OUT, 'w') as fh:
    for r in rows:
        fh.write(json.dumps(r) + '\n')
