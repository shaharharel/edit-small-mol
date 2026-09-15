"""THE COVALENT-ONLY UNION CORPUS: CovInDB + the covalent subset of ChEMBL.

WHY. Every arm trained tonight used v4e, which is 4.77% acrylamide and ~6% covalent by a strict
warhead panel -- generic ChEMBL matched-pair medicinal chemistry. A steering claim on it is about
linker length in med-chem, not about covalent ligands. CovInDB is genuinely covalent but smaller.
The union is the honest substrate: keep every CovInDB pair, and keep the ChEMBL pairs where BOTH
sides carry a real warhead.

STRICT MEANS NITRILE IS EXCLUDED ON ITS OWN. A bare [N]#[C] matches every benzonitrile; counted
that way ChEMBL reads 14.22% covalent, and 8.62 of those points are nitriles. Excluded, the honest
figure is 6.05%. Nitrile IS a real warhead in context (cathepsin, DPP-4), so it is kept when it
co-occurs with another electrophile, and dropped when it is the only match.
"""
import os, sys, json, argparse
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

WARHEADS = {
    'acrylamide_michael': '[CX3]=[CX3][CX3](=O)[NX3,OX2]',
    'vinyl_sulfonyl':     '[CX3]=[CX3][SX4](=O)(=O)',
    'haloacetamide':      '[Cl,Br,I][CH2][CX3](=O)[NX3]',
    'epoxide':            'C1OC1',
    'aziridine':          'C1NC1',
    'boronic':            '[BX3]([OX2H])[OX2H]',
    'aldehyde':           '[CX3H1](=O)[#6]',
    'beta_lactam':        'O=C1CCN1',
    'sulfonyl_fluoride':  '[SX4](=O)(=O)[F]',
    'alpha_ketoamide':    '[CX3](=O)[CX3](=O)[NX3]',
}
WEAK = {'nitrile': '[NX1]#[CX2]'}          # only counts alongside a strong match
PAT = {k: Chem.MolFromSmarts(v) for k, v in {**WARHEADS, **WEAK}.items()}
_C = {}


def warheads(smi):
    if smi in _C:
        return _C[smi]
    m = Chem.MolFromSmiles(smi)
    if m is None:
        _C[smi] = None; return None
    strong = [k for k in WARHEADS if PAT[k] is not None and m.HasSubstructMatch(PAT[k])]
    weak = [k for k in WEAK if m.HasSubstructMatch(PAT[k])]
    _C[smi] = strong + (weak if strong else [])
    return _C[smi]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--covindb', default='data/covindb_pairs/pairs.jsonl')
    ap.add_argument('--chembl', default='data/chembl36_pairs_v3/balanced_dedup.jsonl')
    ap.add_argument('--out', default='data/covalent_union/pairs.jsonl')
    ap.add_argument('--limit-chembl', type=int, default=0)
    a = ap.parse_args()
    os.makedirs(os.path.dirname(a.out), exist_ok=True)

    seen_pair, mols, kept, src = set(), set(), 0, {'covindb': 0, 'chembl_covalent': 0}
    n_cov = n_chem = chem_drop = 0
    with open(a.out, 'w') as out:
        for line in open(a.covindb):
            r = json.loads(line); n_cov += 1
            k = (r['input_smiles'], r['output_smiles'])
            if k in seen_pair: continue
            seen_pair.add(k); mols.update(k); kept += 1; src['covindb'] += 1
            r['source'] = 'covindb'
            out.write(json.dumps(r) + '\n')
        for i, line in enumerate(open(a.chembl)):
            if a.limit_chembl and i >= a.limit_chembl: break
            r = json.loads(line); n_chem += 1
            wa, wb = warheads(r['input_smiles']), warheads(r['output_smiles'])
            if not wa or not wb:          # BOTH sides must be covalent
                chem_drop += 1; continue
            k = (r['input_smiles'], r['output_smiles'])
            if k in seen_pair: continue
            seen_pair.add(k); mols.update(k); kept += 1; src['chembl_covalent'] += 1
            r['source'] = 'chembl_covalent'; r['warheads_a'] = wa; r['warheads_b'] = wb
            out.write(json.dumps(r) + '\n')
            if kept % 20000 == 0:
                print('  ... %d pairs, %d unique molecules' % (kept, len(mols)), flush=True)

    print('=== COVALENT UNION CORPUS ===')
    print('  CovInDB pairs read      %8d   kept %8d' % (n_cov, src['covindb']))
    print('  ChEMBL  pairs read      %8d   kept %8d   dropped (a side not covalent) %8d'
          % (n_chem, src['chembl_covalent'], chem_drop))
    print('  UNIQUE PAIRS            %8d' % kept)
    print('  UNIQUE MOLECULES        %8d' % len(mols))
    print('  wrote %s' % a.out)
    json.dump(dict(unique_pairs=kept, unique_molecules=len(mols), by_source=src,
                   chembl_read=n_chem, chembl_dropped_not_covalent=chem_drop,
                   covindb_read=n_cov, strong_warheads=sorted(WARHEADS),
                   weak_only_excluded=sorted(WEAK)),
              open(a.out.replace('.jsonl', '_meta.json'), 'w'), indent=2)
    return 0


if __name__ == '__main__':
    sys.exit(main())
