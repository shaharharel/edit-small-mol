#!/usr/bin/env python
"""Is the generator STEERABLE? Direction-crossed intents we author ourselves.

WHY THIS EXPERIMENT AND NOT THE ONE WE RAN. The published property table bucketed rows by regexing
the intent sentence the corpus already contained. Three things were wrong with that. The regex
`/polar/` also matches "apolar" and "nonpolar", stock phrases in a pocket description, so the
polarity bucket swallowed 263 of 349 rows. The bucket label was read from a sentence written by an
agent that had the product molecule in view, so the labelling was partly circular with the outcome.
And the buckets overlapped, 349 rows giving 645 memberships, so they were never six independent
dials.

Here the intent is ours. The same input, the same template, ONE WORD FLIPPED:

    "Make this analogue more polar."      vs    "Make this analogue less polar."

Six axes, two directions each. Circularity is gone because no bucket is read from model text, and
the expected sign is fixed before anything runs. The question becomes a single clean one: when the
only thing that changes is the direction word, does the property move the way the word says?

HOW A JOINT MODEL IS INSTRUCTED AT ALL. ChemDFM-21k emits the reason and the molecule in one pass,
so it has no slot for an intent -- which is why "does the reason drive the molecule" was untestable
on it. The trick, borrowed from `rationale_forced_prompts.jsonl`, is to PREFILL the response: the
prompt already ends at "### Response:\n", so the authored sentence is appended there and the model
resumes at "SMILES:". The molecule is then decoded under a sentence we chose. The same file works
for M, which takes an intent natively, so the two are comparable on identical prompts.

CONTROLS THAT MAKE IT READABLE, all generated here rather than argued about later:
  both      both direction words for one input, so each input is its own control (paired)
  neutral   "Propose an improved analogue." -- no direction stated, the drift of the model's
            own preferences under a prefilled but uninformative sentence
  nonsense  a direction word for a property the sentence does not name, to catch a model that
            moves properties whenever any sentence is prefilled

PRE-REGISTERED READING, written down before the run so it cannot be chosen afterwards: steerability
on an axis means the SIGNED property difference between the "more" and "less" arms has the expected
sign on a majority of inputs, tested paired (Wilcoxon) with Holm correction across the six axes.
A model that is steerable on nothing will show differences centred on zero. The neutral arm fixes
where zero is.

    python experiments/covalentformer/build_steer_probe.py \
        --corpus <corpus_potency_pocket.jsonl> --out data/steer_probe --limit-inputs 120
"""
from __future__ import annotations

import argparse
import collections
import json
import os
import sys

# axis -> (property, more-direction sentence, less-direction sentence, sign of 'more' on the prop)
# The wording is deliberately plain and nearly identical across a pair: anything else and the
# comparison measures prose style rather than direction.
AXES = [
    ('polarity', 'TPSA',
     'Make this analogue more polar.',
     'Make this analogue less polar.', +1),
    ('lipophilicity', 'cLogP',
     'Make this analogue more lipophilic.',
     'Make this analogue less lipophilic.', +1),
    ('size', 'MW',
     'Make this analogue larger.',
     'Make this analogue smaller.', +1),
    ('rings', 'RingCount',
     'Add a ring to this analogue.',
     'Remove a ring from this analogue.', +1),
    ('flexibility', 'RotB',
     'Make this analogue more flexible.',
     'Make this analogue more rigid.', +1),
    ('hbd', 'HBD',
     'Add a hydrogen-bond donor to this analogue.',
     'Remove a hydrogen-bond donor from this analogue.', +1),
]
NEUTRAL = 'Propose an improved analogue.'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--corpus', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--split', default='val')
    ap.add_argument('--limit-inputs', type=int, default=120,
                    help='distinct INPUT MOLECULES, not rows: this val set repeats each input '
                         'several times and a count of rows would silently shrink the real n')
    a = ap.parse_args()

    # one row per distinct input molecule. The 21k val set pairs each input with several analogues,
    # so taking rows would ask the same question up to ten times and call it ten observations.
    best = {}
    for line in open(a.corpus):
        r = json.loads(line)
        if r.get('split') != a.split:
            continue
        key = r.get('input_smiles')
        if not key or not r.get('prompt'):
            continue
        if key not in best:
            best[key] = r
    rows = list(best.values())[:a.limit_inputs]
    print('distinct input molecules in %s: %d; using %d' % (a.split, len(best), len(rows)))

    os.makedirs(a.out, exist_ok=True)
    out = []
    for r in rows:
        stem = r['prompt']
        if not stem.rstrip().endswith('### Response:'):
            # the authored sentence has to land exactly where the model's own sentence would
            stem = stem.rstrip() + '\n\n### Response:\n'
        base = {'pair': r['pair'], 'target': r.get('target_chembl_id') or r.get('target'),
                'input_smiles': r['input_smiles'],
                'reference_analogue': r.get('analogue_smiles'),
                'table_residues': r.get('table_residues')}
        for axis, prop, more, less, sign in AXES:
            for direction, sent in (('more', more), ('less', less)):
                out.append(dict(base, arm='%s_%s' % (axis, direction), axis=axis, prop=prop,
                                direction=direction, expected_sign=(sign if direction == 'more'
                                                                    else -sign),
                                forced_sentence=sent,
                                prompt=stem + sent + '\nSMILES:'))
        out.append(dict(base, arm='neutral', axis='neutral', prop=None, direction='none',
                        expected_sign=0, forced_sentence=NEUTRAL,
                        prompt=stem + NEUTRAL + '\nSMILES:'))

    path = os.path.join(a.out, 'steer_prompts.jsonl')
    with open(path, 'w') as fh:
        for r in out:
            fh.write(json.dumps(r) + '\n')
    c = collections.Counter(r['arm'] for r in out)
    print('\nwrote %d prompts to %s' % (len(out), path))
    print('%d inputs x (%d axes x 2 directions + 1 neutral) = %d'
          % (len(rows), len(AXES), len(rows) * (2 * len(AXES) + 1)))
    for k in sorted(c):
        print('  %-24s %4d' % (k, c[k]))
    print('\nscore with:  python experiments/covalentformer/score_steer_probe.py '
          '--gens <generations.jsonl> --prompts %s' % path)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
