#!/usr/bin/env python3
"""Blind SINGLE-SLOT panel for the BASE models' sensible/reasoning cells.

A matched panel is impossible here and that is a measurement, not a choice: of the 262 benchmark
pairs, Llama-Inst base yields 136 rows with both a sentence and a parseable molecule, ether0 base
59 and ChemDFM base 31 -- and all three overlap on exactly ONE pair. So each arm is sampled from
its OWN eligible rows and every case shows ONE proposal, scored absolutely.

SmileyLlama base is excluded on a measurement too: its outputs average 1.0 word, so there is no
rationale to judge and any rate would describe the parser, not the model.

The 48 items (16 per arm) are shuffled into one file so a single set of reviewers applies one
scale with arm identity hidden -- the drift that forced the dagger footnote on the ChemDFM row
comes from scoring arms in separate panels, and this avoids it.
"""
import argparse, json, os, random
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
SCR = ('/private/tmp/claude-501/-Users-shaharharel-Documents-github-edit-small-mol/'
       'a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/scratchpad/alltab/sft_runs')
MODELS = {'llama_base': SCR + '/zeroshot_base_bench/generations.jsonl',
          'chemdfm_base': SCR + '/zeroshot_chemdfm_bench/generations.jsonl',
          'ether0_base': SCR + '/zeroshot_ether0_bench/generations.jsonl'}
import importlib.util as _il
_s = _il.spec_from_file_location('_p5', os.path.join(HERE, 'build_selfconsist_panel_5arm.py'))
_m = _il.module_from_spec(_s)


def smi(r):
    """The generated SMILES, or None. The field holds the STRING 'None' when nothing parsed, which
    is truthy -- reading it with `or` silently counts failures as molecules."""
    for k in ('gen_smiles_canonical', 'gen_smiles'):
        v = r.get(k)
        if isinstance(v, str) and v.strip() and v.strip() != 'None':
            return v.strip()
    return None


def fmt_atoms(atoms, limit=40):
    out = []
    for a in atoms[:limit]:
        bits = ['%-5s %-2s' % (a.get('atom', '?'), a.get('element', '?'))]
        if a.get('faces'):
            bits.append('faces %-12s' % a['faces'])
        if a.get('distance_A') is not None:
            bits.append('%.2f A' % a['distance_A'])
        if a.get('interaction'):
            bits.append(str(a['interaction']))
        out.append('    ' + '  '.join(bits))
    if len(atoms) > limit:
        out.append('    ... %d more atom rows' % (len(atoms) - limit))
    return '\n'.join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--per-arm', type=int, default=16)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=41)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    items = []
    for name, path in MODELS.items():
        rows = [json.loads(l) for l in open(path)]
        el = [r for r in rows
              if smi(r) and Chem.MolFromSmiles(smi(r)) and (r.get('gen_sentence') or '').strip()]
        print('%-13s eligible %3d of %d' % (name, len(el), len(rows)))
        for r in rng.sample(el, min(a.per_arm, len(el))):
            items.append((name, r))
    rng.shuffle(items)

    want = {r['pair'] for _, r in items}
    recs = {}
    for line in open(EV):
        x = json.loads(line)
        if x['pair'] in want:
            recs[x['pair']] = x

    os.makedirs(a.out, exist_ok=True)
    key, blocks = {}, []
    for i, (name, r) in enumerate(items, 1):
        rec = recs.get(r['pair'])
        if rec is None:
            continue
        mi = rec['MODEL_INPUT']
        key['item_%03d' % i] = {'arm': name, 'pair': r['pair']}
        blocks.append("""
================================================================================
ITEM {i} of {n}   item_id item_{i:03d}   id {pid}   target {tgt}
================================================================================
INPUT LIGAND
  SMILES: {ins}

INPUT CONTACT TABLE -- each atom's NEAREST protein atom, measured in the input crystal
{atoms}

--------------------------------------------------------------------------------
ONE PROPOSAL. This is a model's OWN proposed molecule and its OWN rationale. There is
no reference answer, and you are NOT asked whether this is the analogue that was
actually crystallised -- many analogues are valid.

  PROPOSED MOLECULE: {smi}
  RATIONALE: {sent}
--------------------------------------------------------------------------------
Judge ONLY self-consistency and groundedness:

  edit_identified_correctly  Compare the INPUT SMILES with the proposal's OWN molecule. Does the
                             rationale describe the change that actually separates them?
  atom_attribution_ok        Every distance or contact cited: a real row of the input contact
                             table, AND attached to the atom that owns it?
  invented_chemistry         Does it assert a contact, bond or interaction the table does not
                             support?  (true = bad)
  chemically_sensible_edit   Is the proposed molecule a reasonable medicinal-chemistry analogue of
                             the input at all (not a different scaffold, not nonsense)?
  would_show_a_chemist       Would you put this molecule-plus-rationale in front of a medicinal
                             chemistry audience?
  one_line_why
""".format(i=i, n=len(items), pid=r['pair'], tgt=(rec.get('target') or '?'),
           ins=mi.get('input_smiles', '?'), atoms=fmt_atoms(mi.get('atoms') or []),
           smi=smi(r), sent=(r.get('gen_sentence') or '').strip()))

    open(os.path.join(a.out, 'panel.txt'), 'w').write('\n'.join(blocks))
    json.dump(key, open(os.path.join(a.out, 'KEY_DO_NOT_SHOW_REVIEWERS.json'), 'w'), indent=2)
    print('wrote %s/panel.txt: %d items' % (a.out, len(blocks)))


if __name__ == '__main__':
    main()
