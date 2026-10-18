#!/usr/bin/env python3
"""Blind review panel over candidate g3 TRAINING rows, to pick report exemplars.

The report shows five g3 training rows as examples of what the model is fine-tuned on. Those rows
were picked by eye (recognisable target, a sentence in the house style, chair verdict ADOPT) and
never independently checked. A chair ADOPT means three labellers and a chair accepted the row at
labelling time; it does not mean the sentence survives a fresh audit against the contact table,
and the chemist panels on model output have repeatedly found real distances attached to the wrong
atom. An exemplar in the report should clear a higher bar than an average training row.

So: draw a pool of candidates, have three reviewers audit each against its own contact table, and
keep only the rows that pass unanimously.

The panel shows the FULL prompt (input SMILES + contact table) and the chair sentence, which is the
training target. Reviewers never see which rows are currently in the report.
"""
import argparse, json, os, random, re, sys

HERE = os.path.dirname(os.path.abspath(__file__))
G3 = os.path.join(HERE, 'sft_out_g3/sft_pocket.jsonl')

# the five currently in the report, mixed into the pool and not flagged
IN_REPORT = ['8AI0 M79->O4I', '7B9Z T58->R3B', '6EAA J1V->J2D',
             '6P6G LUP->O41', '9Q66 A1CQP->A1CQZ']


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pool', type=int, default=18)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=19)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    rows = [json.loads(l) for l in open(G3)]
    tr = {r['pair']: r for r in rows if r['split'] == 'train'}
    cur = [tr[p] for p in IN_REPORT if p in tr]
    assert len(cur) == len(IN_REPORT), 'report rows missing from g3 train'

    # candidates: ADOPT, a sentence in the house style (a measured distance and a residue), and a
    # contact table small enough that a reviewer can actually audit every row of it
    AA = {'ALA','ARG','ASN','ASP','CYS','GLN','GLU','GLY','HIS','ILE','LEU','LYS','MET',
          'PHE','PRO','SER','THR','TRP','TYR','VAL'}
    RES = re.compile(r'\b([A-Z]{3})(\d+)')
    pool = []
    for p, r in tr.items():
        if p in IN_REPORT or r.get('verdict') != 'ADOPT':
            continue
        s = r.get('reference_sentence') or ''
        if not re.search(r'\d+\.\d+', s):
            continue
        if not any(m.group(1) in AA for m in RES.finditer(s.upper())):
            continue
        if not (18 <= len(s.split()) <= 38) or (r.get('n_atom_rows') or 99) > 40:
            continue
        pool.append(r)
    extra = rng.sample(pool, min(a.pool - len(cur), len(pool)))
    items = cur + extra
    rng.shuffle(items)
    print('pool %d eligible, panel %d items (%d currently in report, unflagged)'
          % (len(pool), len(items), len(cur)))

    os.makedirs(a.out, exist_ok=True)
    key, blocks = {}, []
    for i, r in enumerate(items, 1):
        key['item_%03d' % i] = {'pair': r['pair'], 'in_report': r['pair'] in IN_REPORT}
        # the prompt already contains the input SMILES and the measured contact table verbatim
        body = r['prompt'].split('### Input:', 1)[1].split('### Response:')[0].strip()
        blocks.append("""
================================================================================
ITEM {i} of {n}   item_id item_{i:03d}   pair {pair}   PDB {pdb}
================================================================================
{body}

PROPOSED ANALOGUE (what the pair actually changed to):
  {ana}

CHAIR SENTENCE -- this is the training target, the text the model is fine-tuned to emit:
  {sent}
--------------------------------------------------------------------------------
Audit the chair sentence against the contact table above. Score:

  every_number_real          Every distance the sentence cites is a real row of that table.
  attribution_correct        Every cited distance is attached to the atom/residue that owns it.
  edit_described_correctly   Compare INPUT SMILES with the PROPOSED ANALOGUE. Does the sentence
                             describe the change that actually separates them, with nothing
                             silently extra?
  no_invented_chemistry      It asserts no contact, bond or interaction the table does not support.
  exemplar_quality           Would you put THIS row in a paper as an example of the training data?
  one_line_why
""".format(i=i, n=len(items), pair=r['pair'], pdb=r.get('pdb'), body=body,
           ana=r.get('analogue_smiles'), sent=(r.get('reference_sentence') or '').strip()))

    open(os.path.join(a.out, 'panel.txt'), 'w').write('\n'.join(blocks))
    json.dump(key, open(os.path.join(a.out, 'KEY_DO_NOT_SHOW_REVIEWERS.json'), 'w'), indent=2)
    print('wrote %s/panel.txt' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
