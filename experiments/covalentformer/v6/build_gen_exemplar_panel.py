#!/usr/bin/env python3
"""Blind audit of GENERATED proposals, to pick the report's generation-sample exemplars.

Section 4 shows three held-out pairs answered by three ChemDFM arms. Those three pairs were picked
by eye, to illustrate the base model's failure modes. That is a defensible editorial purpose and an
indefensible way to choose published evidence: nothing checked that the trained arms' sentences in
those rows are actually correct, and the chemist panels have found that roughly a third of
model-cited distances are real numbers attached to the wrong atom. A figure that shows a trained
arm "doing it right" must not be a row where it is quietly wrong.

So every proposal in a candidate pool is audited against its own input contact table by three
reviewers, with **arm identity hidden**: each pair shows its three proposals as unlabelled slots in
a per-pair shuffled order. Grading cannot then be anchored on knowing which arm is untrained.

Reviewers also answer one per-pair question -- whether the pair is a fair illustration -- so the
selection criterion is explicit rather than editorial. The pool deliberately includes the three
pairs currently published, unflagged.
"""
import argparse, json, os, random, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, '..', '..', '..'))
EVIDENCE = os.path.join(ROOT, 'data/pdb_mirror/v7_evidence.jsonl')
G3_DATA = os.path.join(HERE, 'sft_out_g3/sft_pocket.jsonl')
SCRATCH = ('/private/tmp/claude-501/-Users-shaharharel-Documents-github-edit-small-mol/'
           'a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/scratchpad')

ARMS = {'base': SCRATCH + '/alltab/sft_runs/zeroshot_chemdfm_bench/generations.jsonl',
        'g3': os.path.join(HERE, 'sft_2k/chemdfm_pocket_g3/generations.jsonl'),
        'g3g4': SCRATCH + '/g3g4res/chemdfm/generations.jsonl'}

CURRENT = ['6PGE OJM->OHJ', '5OIW 9WN->A0K', '6G9H ERW->EQZ']


def gen_smiles(r):
    for k in ('gen_smiles_canonical', 'gen_smiles'):
        v = r.get(k)
        if isinstance(v, str) and v.strip() and v.strip() != 'None':
            return v.strip()
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', type=int, default=12)
    ap.add_argument('--out', required=True)
    ap.add_argument('--seed', type=int, default=23)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    data = {k: {r['pair']: r for r in (json.loads(l) for l in open(p))}
            for k, p in ARMS.items()}
    # the prompt body (input SMILES + measured contact table) comes from the val rows of the
    # training file, so the reviewer audits against exactly what the model was shown
    val = {r['pair']: r for r in (json.loads(l) for l in open(G3_DATA)) if r['split'] == 'val'}

    common = [p for p in data['base']
              if p in val
              and all(gen_smiles(data[k].get(p, {}))
                      and Chem.MolFromSmiles(gen_smiles(data[k][p]))
                      and (data[k][p].get('gen_sentence') or '').strip() for k in ARMS)]
    print('pairs answered by all three arms: %d' % len(common))
    rest = [p for p in common if p not in CURRENT]
    chosen = [p for p in CURRENT if p in common] + rng.sample(
        rest, min(a.pairs - len([p for p in CURRENT if p in common]), len(rest)))
    rng.shuffle(chosen)
    print('panel pairs: %d (%d currently published, unflagged)'
          % (len(chosen), len([p for p in chosen if p in CURRENT])))

    os.makedirs(a.out, exist_ok=True)
    key, blocks = {}, []
    for i, pair in enumerate(chosen, 1):
        body = val[pair]['prompt'].split('### Input:', 1)[1].split('### Response:')[0].strip()
        slots = list(ARMS)
        rng.shuffle(slots)
        key['pair_%02d' % i] = {'pair': pair, 'published_now': pair in CURRENT,
                                'slots': {chr(65 + j): s for j, s in enumerate(slots)}}
        props = []
        for j, armk in enumerate(slots):
            r = data[armk][pair]
            props.append('  PROPOSAL %s\n    molecule: %s\n    reasoning: %s\n'
                         % (chr(65 + j), gen_smiles(r),
                            (r.get('gen_sentence') or '').strip()))
        blocks.append("""
================================================================================
PAIR {i} of {n}   pair_id pair_{i:02d}   {pair}
================================================================================
{body}

THREE PROPOSALS from three different models. The letters are reshuffled for every pair; do not try
to infer which model is which. One of the three is an UNTRAINED model, but not reliably the same
letter, so judge each slot only against the table above.

{props}
--------------------------------------------------------------------------------
For EACH of A, B, C score, in `slots`:

  molecule_parses_and_is_an_analogue   A real molecule, and a single-step analogue of the input --
                                       not a different scaffold, not the input handed back unchanged.
  every_number_real                    Every distance cited is a real row of the table above.
  attribution_correct                  Every cited distance is attached to the atom and residue
                                       that own it. A real number on the wrong atom is FALSE.
  edit_described_correctly             Does the reasoning describe the change that actually
                                       separates the input from THAT slot's own molecule?
  no_invented_chemistry                It asserts no contact, bond or interaction the table does
                                       not support, and does not contradict the table's own typing.
  publishable                          Would you be willing to see THIS slot printed in a paper,
                                       as an accurate example of what this model produced?
  one_line_why

Then for the PAIR as a whole, in `pair_verdict`:

  fair_illustration   Is this pair a fair, non-cherry-picked illustration of how these three models
                      differ? False if the differences are trivial, or if the pair is unusually
                      easy or unusually degenerate.
  best_slot           A, B, C, or "none".
  pair_why
""".format(i=i, n=len(chosen), pair=pair, body=body, props='\n'.join(props)))

    open(os.path.join(a.out, 'panel.txt'), 'w').write('\n'.join(blocks))
    json.dump(key, open(os.path.join(a.out, 'KEY_DO_NOT_SHOW_REVIEWERS.json'), 'w'), indent=2)
    print('wrote %s/panel.txt' % a.out)
    return 0


if __name__ == '__main__':
    sys.exit(main())
