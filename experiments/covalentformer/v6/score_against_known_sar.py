#!/usr/bin/env python3
"""Did the model propose a move the programme actually made? Ground truth, no chemist needed.

THERE IS NOT ONE REAL MOVE. Each programme made many, so the comparison pool is every known
compound for that target, not the single withheld reference. Two sources, both local:

  ChEMBL actives for the target  (verified by target_name, NOT by recalled id -- two of five ids
                                  guessed from memory were wrong: CHEMBL4618 is Leukotriene A-4
                                  hydrolase, not PSMB5, and CHEMBL4681 is AKR1C3, not NS3, so
                                  pooling them would have compared proteasome generations against
                                  2,676 leukotriene compounds)
      JAK3        CHEMBL2148      2,108 unique SMILES
      HER2        CHEMBL1824      1,350
      CathepsinK  CHEMBL268         761
      PSMB5       CHEMBL4662        723   "Proteasome Macropain subunit MB1"
      NS3         CHEMBL4296325      25
  every ligand deposited in any PDB entry of that target

AND THE INPUT LIGAND IS EXCLUDED FROM THE POOL. The first version left it in, so a generation that
reproduced its own input scored as "found a known compound": all five of its KNOWN hits were
7KX->7KX, A1IJC->A1IJC, INA->INA, and its NEAR hits were Tc 0.85-0.90 to the input. Echoes, not
SAR. A hit now also has to be CLOSER to the known compound than to the input, so moving away from
the input and toward something real is what counts.

Every programme prompt is one half of a real same-series pair, and the other half -- the compound
the programme actually advanced to -- was withheld from the prompt. So for these five programmes,
unlike mol1, there is an answer key:

  EXACT   the proposed analogue canonicalises to the reference analogue. The model proposed the
          move that was made.
  KNOWN   it canonicalises to some OTHER deposited ligand of the same target. Not the move in this
          pair, but a compound that programme really made -- which is a hit, not a miss, and is why
          this script looks past the single reference.
  NEAR    high Tanimoto to either, without being identical.

This is worth more than a panel for this question. A chemist can say a proposal is plausible; only
the deposited record can say it is what happened. It is also the one thing mol1 could never
provide: mol1 is a designed seed with no reference analogue, so nothing it proposed could be scored
against reality at all.

READ THE NEGATIVE RESULT CAREFULLY IF IT COMES. Not reproducing the reference is weak evidence of
failure: a pocket admits many reasonable edits and the programme made one of them, often for
reasons outside the structure (synthesis, patent, selectivity, pharmacokinetics) that the model is
given no access to. An EXACT hit is strong evidence; a miss is almost none. The honest headline is
the hit count, never a "miss rate".
"""
import argparse
import collections
import gzip
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))
MIRROR = os.path.abspath(os.path.join(HERE, '..', '..', '..', 'data', 'pdb_mirror'))
# Verified against the target_name column of overlapping_assay_pairs_minimal_mmp.csv, not recalled.
CHEMBL_OF = {
    'P52333': ('CHEMBL2148', 'Tyrosine-protein kinase JAK3'),
    'P04626': ('CHEMBL1824', 'Receptor tyrosine-protein kinase erbB-2'),
    'P43235': ('CHEMBL268', 'Cathepsin K'),
    'P28074': ('CHEMBL4662', 'Proteasome Macropain subunit MB1'),
    'P26664': ('CHEMBL4296325', 'NS3 protease'),
}
ACTIVITIES = os.path.abspath(os.path.join(
    HERE, '..', '..', '..', 'data', 'overlapping_assays', 'extracted', 'activities_minimal.csv'))

SKIP = {'HOH', 'DOD', 'SO4', 'PO4', 'GOL', 'EDO', 'PEG', 'ACT', 'CL', 'NA', 'MG', 'ZN', 'CA',
        'MN', 'K', 'IOD', 'BR', 'NO3', 'DMS', 'TRS', 'MES', 'EPE', 'FMT', 'ACY', 'NAG', 'BMA',
        'MAN', 'FUC', 'ATP', 'ADP', 'AMP', 'GTP', 'GDP', 'NAD', 'NAP', 'FAD', 'SAH', 'SAM',
        'COA', 'HEM', 'MYR', 'PLM', 'OLA', 'STE', 'UNX', 'UNL'}


def ligand_codes(pdb):
    """Every non-trivial ligand code in one entry, from the local mirror."""
    path = os.path.join(MIRROR, 'cif', '%s.cif.gz' % pdb)
    if not os.path.exists(path):
        return set()
    out, hdr, idx = set(), [], None
    with gzip.open(path, 'rt', errors='ignore') as fh:
        for ln in fh:
            if ln.startswith('_atom_site.'):
                hdr.append(ln.strip().split('.', 1)[1]); idx = None; continue
            if hdr and idx is None:
                try:
                    idx = {k: hdr.index(k) for k in ('group_PDB', 'label_comp_id')}
                except ValueError:
                    return out
            if not ln.startswith('HETATM'):
                if out:
                    continue
                continue
            p = ln.split()
            if idx is None or len(p) <= max(idx.values()):
                continue
            c = p[idx['label_comp_id']]
            if c not in SKIP and len(c) >= 2 and not c.startswith('UNK'):
                out.add(c)
    return out


def ccd_smiles(codes):
    """{code: SMILES} from the full local CCD, one pass."""
    path = os.path.join(MIRROR, 'components.cif.gz')
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
                    if cur not in out or len(cand) > len(out[cur]):
                        out[cur] = cand
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', nargs='+', required=True)
    ap.add_argument('--prompts', default=os.path.join(HERE, 'programme_prompts.jsonl'))
    ap.add_argument('--programmes', default=os.path.join(HERE, 'covalent_programmes_full.json'))
    ap.add_argument('--near', type=float, default=0.85)
    ap.add_argument('--out', default=os.path.join(HERE, 'known_sar_hits.json'))
    a = ap.parse_args()
    from rdkit import Chem, DataStructs, RDLogger
    from rdkit.Chem import rdFingerprintGenerator
    RDLogger.DisableLog('rdApp.*')
    canon = lambda s: (lambda m: Chem.MolToSmiles(m) if m else None)(Chem.MolFromSmiles(s or ''))
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    prompts = {r['pair']: r for r in (json.loads(l) for l in open(a.prompts))}
    prog = json.load(open(a.programmes))

    # every ligand ever deposited with each target, across the entries RCSB listed for it
    uni_of = {r['pair']: r['target'] for r in prompts.values()}
    entries = collections.defaultdict(set)
    for uni, v in prog.items():
        for h in v['covalent']:
            entries[uni].add(h['pdb'])
    known_codes = collections.defaultdict(set)
    for uni, pdbs in entries.items():
        for pdb in pdbs:
            known_codes[uni] |= ligand_codes(pdb)
    smi = ccd_smiles({c for s in known_codes.values() for c in s})
    known = {}
    for uni, codes in known_codes.items():
        d = {}
        for c in codes:
            cs = canon(smi.get(c, ''))
            if cs:
                d[cs] = 'PDB:%s' % c
        known[uni] = d
    # ChEMBL actives for the same target: thousands of real compounds against the tens the PDB has
    if os.path.exists(ACTIVITIES):
        import csv
        wanted = {v[0]: k for k, v in CHEMBL_OF.items()}
        pool = collections.defaultdict(set)
        with open(ACTIVITIES) as fh:
            for row in csv.DictReader(fh):
                uni = wanted.get(row.get('target_chembl_id'))
                if uni:
                    pool[uni].add(row.get('canonical_smiles'))
        for uni, smis in pool.items():
            d = known.setdefault(uni, {})
            for x in smis:
                cs = canon(x)
                if cs and cs not in d:
                    d[cs] = 'ChEMBL'
    print('comparison pool per target:')
    for uni, d in sorted(known.items()):
        lab = prog.get(uni, {}).get('label', uni)
        npdb = sum(1 for v in d.values() if v.startswith('PDB:'))
        print('  %-12s %5d compounds (%d PDB ligands, %d ChEMBL)'
              % (lab, len(d), npdb, len(d) - npdb))

    rows = []
    for path in a.gens:
        for r in (json.loads(l) for l in open(path)):
            pr = prompts.get(r['pair'])
            if pr is None:
                continue
            uni = pr['target']
            inp = canon(pr['input_smiles'])
            ref = canon(pr['reference_smiles'])
            reffp = gen.GetFingerprint(Chem.MolFromSmiles(ref)) if ref else None
            # the INPUT is not a discovery. Leaving it in the pool made every "hit" an echo.
            kmap = {k: v for k, v in known.get(uni, {}).items() if k != inp}
            kfps = {k: gen.GetFingerprint(Chem.MolFromSmiles(k)) for k in kmap}
            inpfp = gen.GetFingerprint(Chem.MolFromSmiles(inp)) if inp else None
            for g in r['generations']:
                cs = canon(g.get('gen_smiles'))
                if not cs:
                    continue
                if cs == inp:
                    continue                     # echo of the input: not a proposal at all
                hit = None
                fp0 = gen.GetFingerprint(Chem.MolFromSmiles(cs))
                tc_inp = (DataStructs.TanimotoSimilarity(fp0, inpfp) if inpfp is not None else 0.0)
                if ref and cs == ref:
                    hit = ('EXACT_reference', kmap.get(cs, pr['analogue_code']), 1.0)
                elif cs in kmap:
                    hit = ('KNOWN_ligand_same_target', kmap[cs], 1.0)
                else:
                    best, code = 0.0, None
                    fp = gen.GetFingerprint(Chem.MolFromSmiles(cs))
                    if reffp is not None:
                        best = DataStructs.TanimotoSimilarity(fp, reffp); code = pr['analogue_code']
                    for k, kfp in kfps.items():
                        t = DataStructs.TanimotoSimilarity(fp, kfp)
                        if t > best:
                            best, code = t, kmap[k]
                    # a NEAR hit must be closer to the known compound than to its own input,
                    # otherwise it is just "looks like what it was given"
                    if best >= a.near and best > tc_inp:
                        hit = ('NEAR', code, round(best, 3))
                if hit:
                    rows.append({'pair': r['pair'], 'programme': pr['programme'],
                                 'kind': hit[0], 'matched_ligand': hit[1], 'tanimoto': hit[2],
                                 'proposed_smiles': cs, 'decode': g['decode'],
                                 'tanimoto_to_input': round(tc_inp, 3),
                                 'sentence': ' '.join((g.get('gen_sentence') or '').split())})
    by = collections.Counter(r['kind'] for r in rows)
    print('\n%s' % dict(by))
    json.dump(rows, open(a.out, 'w'), indent=1)
    for kind in ('EXACT_reference', 'KNOWN_ligand_same_target', 'NEAR'):
        sel = [r for r in rows if r['kind'] == kind]
        if not sel:
            continue
        print('\n=== %s: %d ===' % (kind, len(sel)))
        for r in sel[:10]:
            print('  %-22s %-11s -> %-6s Tc %.2f  %s'
                  % (r['pair'], r['programme'], r['matched_ligand'], r['tanimoto'],
                     r['sentence'][:80]))
    print('\nwrote %s' % a.out)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
