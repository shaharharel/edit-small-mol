#!/usr/bin/env python3
"""Build inference prompts for the five unseen covalent programmes.

Uses the CORPUS'S OWN path end to end -- evidence_v4.build() for the contact table and
sft_data.render_prompt() for the text -- so a prompt here is byte-identical in structure to the
ones the model was fine-tuned on. That matters because the question these programmes answer is
about the TARGET being unseen, and any difference in prompt format would confound it.

This is what mol1 cannot give us. mol1 is an unseen target AND a predicted pose AND element+index
atom names AND sampled decoding, all at once, and the ownership drop from 70% (held-out crystals)
to 29% (mol1) cannot be apportioned among them. These are unseen targets with MEASURED geometry,
CCD atom names and a real reference analogue, so they isolate the target shift by itself.

The reference analogue is written to the output but NEVER into the prompt: at inference the model
gets the input side only, exactly as in training. It is carried so that a proposal can afterwards
be compared against the edit the programme actually made.
"""
import argparse
import importlib.util
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def pair_record(p, uniprot):
    """The record shape evidence_v4.build() reads, matching build_pairs_v7's output.

    build() wants more than the four obvious fields: it reads r['heavy'], r['clogp'], r['tpsa'],
    r['rotb'], r['hbd'], r['hba'], r['tanimoto'], r['target'], r['analogue_pdb'] and
    r['analogue_code'] as well, each descriptor being an [input, analogue] pair. A missing one
    raises KeyError from deep inside the table assembly, which is how the second run of this script
    failed on all 41 pairs after the CCD fix. Computed here with the same RDKit calls
    build_pairs_v7 uses, so a prompt built here is descriptor-identical to a corpus prompt.
    """
    from rdkit import Chem
    from rdkit.Chem import Descriptors
    a = Chem.MolFromSmiles(p['smiles_a'])
    b = Chem.MolFromSmiles(p['smiles_b'])
    if a is None or b is None:
        return None
    two = lambda f: [f(a), f(b)]
    return {
        'pdb': p['pdb_a'], 'input_code': p['a'], 'input': p['smiles_a'],
        'analogue': p['smiles_b'], 'analogue_code': p['b'], 'analogue_pdb': p['pdb_b'],
        'target': uniprot, 'tanimoto': p['tc'],
        'heavy': two(lambda m: m.GetNumHeavyAtoms()),
        'clogp': [round(Descriptors.MolLogP(a), 2), round(Descriptors.MolLogP(b), 2)],
        'tpsa': [round(Descriptors.TPSA(a), 1), round(Descriptors.TPSA(b), 1)],
        'rotb': two(Descriptors.NumRotatableBonds),
        'hbd': two(Descriptors.NumHDonors),
        'hba': two(Descriptors.NumHAcceptors),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', default=os.path.join(HERE, 'programme_pairs.json'))
    ap.add_argument('--programmes', default=os.path.join(HERE, 'covalent_programmes_full.json'))
    ap.add_argument('--out', default=os.path.join(HERE, 'programme_prompts.jsonl'))
    ap.add_argument('--per-programme', type=int, default=6,
                    help='pairs per programme, taken in descending Tanimoto')
    a = ap.parse_args()

    ev4 = load('ev4', 'evidence_v4.py')
    # evidence_v4 loads ccd_atoms.json at import, and that cache covers only the ligands the
    # corpus already uses: 91 of these 95 are absent. Without a CCD definition ccd_mol() returns
    # (None, None), no ligand atom NAMES can be assigned, and build() returns None while saying
    # nothing about why -- which is exactly how the first run of this script produced 0 prompts
    # from 41 valid pairs. Merge the extension rather than editing the corpus cache in place.
    ext_path = os.path.join(ev4.D, 'ccd_atoms_ext.json')
    if os.path.exists(ext_path):
        ext = json.load(open(ext_path))
        before = len(ev4.CCD)
        ev4.CCD.update({k: v for k, v in ext.items() if k not in ev4.CCD})
        print('CCD cache %d -> %d (+%d from ccd_atoms_ext.json)'
              % (before, len(ev4.CCD), len(ev4.CCD) - before))
    sd = load('sft_data', 'sft_data.py')
    pairs = json.load(open(a.pairs))
    prog = json.load(open(a.programmes))

    # a ligand code -> the entries it appears in, so a pair can be built from either side
    site_of = {}
    for uni, v in prog.items():
        for h in v['covalent']:
            site_of.setdefault((uni, h['ligand']), set()).add(
                (h['pdb'], '%s%s' % (h['residue'], h['resnum']), h['dist']))

    rows, fails = [], []
    for uni, v in pairs.items():
        picked = 0
        for p in v['pairs']:
            if picked >= a.per_programme:
                break
            # build from the a-side crystal; the b-side ligand is the reference analogue
            rec = pair_record(p, uni)
            if rec is None:
                fails.append((uni, p['pdb_a'], p['a'], 'SMILES would not parse'))
                continue
            try:
                mi = ev4.build(rec, with_label_side=False)
            except Exception as e:
                fails.append((uni, p['pdb_a'], p['a'], 'build raised %s' % type(e).__name__))
                continue
            if not mi:
                fails.append((uni, p['pdb_a'], p['a'], 'build returned None'))
                continue
            block = mi.get('MODEL_INPUT') if isinstance(mi, dict) and 'MODEL_INPUT' in mi else mi
            try:
                # render_prompt returns only the BODY. The model is trained on the Alpaca wrapper
                # -- '### Instruction:' + the task statement, then '### Input:', then a trailing
                # '### Response:\n' -- and sft_data.alpaca_prompt() is what adds it. Passing the
                # bare body gave the model a contact table with no task statement and no response
                # cue: 450 generations produced a parseable SMILES 1% of the time, against 97% for
                # the same arm on held-out crystals.
                prompt = sd.alpaca_prompt(sd.render_prompt(uni, block, pocket=True))
                assert prompt.lstrip().startswith('### Instruction:')
                assert prompt.endswith('### Response:\n')
            except Exception as e:
                fails.append((uni, p['pdb_a'], p['a'], 'render raised %s' % type(e).__name__))
                continue
            rows.append({
                'pair': '%s %s->%s' % (p['pdb_a'], p['a'], p['b']),
                'programme': v['label'], 'target': uni, 'site': v['site'],
                'pdb': p['pdb_a'], 'input_code': p['a'], 'analogue_code': p['b'],
                'input_smiles': p['smiles_a'], 'reference_smiles': p['smiles_b'],
                'tanimoto': p['tc'], 'dha': p['dha'],
                'prompt': prompt, 'MODEL_INPUT': block,
                'n_atom_rows': len(block.get('atoms') or []),
                'UNSEEN_TARGET': True, 'split': 'programme'})
            picked += 1
        print('%-12s %d prompts' % (v['label'], picked))

    with open(a.out, 'w') as fh:
        for r in rows:
            fh.write(json.dumps(r) + '\n')
    print('\nwrote %d prompts -> %s' % (len(rows), a.out))
    if fails:
        print('%d pairs could not be built:' % len(fails))
        for f in fails[:12]:
            print('  %-9s %s %-6s %s' % f)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
