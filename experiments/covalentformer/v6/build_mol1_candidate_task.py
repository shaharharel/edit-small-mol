#!/usr/bin/env python3
"""Build the external-chemist candidate-pick task for mol1, over the WHOLE generation pool.

WHY THIS REPLACES THE 24-CANDIDATE VERSION
------------------------------------------
The first `mol1_candidate_pick/` held 24 candidates in a single `candidates.txt`. That number was
not the output of any filter -- 181 unique generations survive the automated checks -- it was the
size a one-file, one-sitting review could carry. Every other task in the labelling folder
(`four_arm_eval/`, `chemdfm_g3g4_eval/`, `metrics_eval/`) is batched: 25 items per file, a manifest,
and `score_out/batch_NN_rev{A,B,C}.json`. This script puts mol1 on that same footing so the pool
size stops being bounded by the file layout.

WHAT COUNTS AS A CANDIDATE
--------------------------
A (molecule, sentence) pair, deduplicated on both. 125 distinct molecules carry 181 distinct
sentences, because the same analogue is often proposed with different reasoning -- and the reasoning
is what this task is judging. Collapsing to one sentence per molecule would throw away the only
head-to-head reasoning comparisons in the set, so duplicates of a MOLECULE are kept and flagged to
the reviewer; exact duplicate GENERATIONS are dropped.

BASE-ARM GENERATIONS ARE EXCLUDED on instruction. The base arm also could not fill a pool of this
size: 120 of its generations yield 23 that pass, against 93 (g4) and 88 (g3+g4).

THE KEY NEVER ENTERS THE TASK FOLDER. It is written next to this script, gitignored, and holds the
arm and the automated reward per item so that chemist picks can be scored against both.
"""
import argparse
import collections
import importlib.util
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
GEN = os.path.join(HERE, 'mol1_cand')
PROMPT = os.path.join(HERE, 'mol1_chemdfm', 'mol1_prompt_covalent.jsonl')
ARMS = ('g4', 'g3g4')          # base excluded deliberately; see module docstring
MIN_CITED = 2                  # the automated pre-filter the brief promises reviewers


def load_reward():
    spec = importlib.util.spec_from_file_location('rw', os.path.join(HERE, 'reward_verifiers.py'))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def norm(s):
    return ' '.join((s or '').split())


def collect():
    """Unique (molecule, sentence) candidates from the non-base arms, after the stated filter."""
    rows = []
    for arm in ARMS:
        path = os.path.join(GEN, 'gen_%s.jsonl' % arm)
        for r in (json.loads(l) for l in open(path)):
            r['arm'] = arm
            rows.append(r)
    kept = [r for r in rows
            if r.get('valid') and not r.get('echo')
            and r.get('groundedness', 0) == 1.0
            and len(r.get('named_residues') or []) >= MIN_CITED]
    uniq = {}
    for r in kept:
        uniq.setdefault((r.get('gen_smiles_canonical'), norm(r.get('gen_sentence'))), r)
    return len(rows), len(kept), list(uniq.values())


def contact_table(prompt):
    """The table block of the prompt, verbatim, for the batch headers.

    Reviewers see one batch file and nothing else, so the table and the mol1 caveats have to be
    repeated in every batch rather than living once in START_HERE.md.
    """
    lines = prompt.split('\n')
    start = next(i for i, l in enumerate(lines) if l.startswith('MEASURED POCKET CONTACTS'))
    end = len(lines)
    for i in range(start + 1, len(lines)):
        if lines[i].strip() == '' and i > start + 3:
            end = i
            break
    return '\n'.join(lines[start:end])


TABLE_ROW = re.compile(
    r'^(?P<atom>[A-Z][a-z]?\d+)\s+(?P<el>[A-Za-z]{1,2})\s+(?P<faces>\S+)\s+'
    r'(?P<chain>\S+)\s+(?P<dist>[\d.]+)\s+(?P<ang>\S+)\s*\|\s*(?P<inter>.+)$')


def model_input_from_prompt(prompt, input_smiles):
    """Reconstruct the MODEL_INPUT dict that reward_verifiers.score() expects.

    mol1 is not in v7_evidence.jsonl -- it is a designed seed with no deposited complex, so there
    is no evidence row to look up, and build_mol1_prompt.py writes the rendered prompt without the
    structured block. The table printed in the prompt IS that block rendered, one row per heavy
    atom, so parsing it back is exact rather than an approximation: scoring then sees precisely the
    rows the model saw.

    suspect_residues is derived from the rows that carry their own uncertainty flag, which is how
    the prompt communicates it to the model and therefore what 'uncertainty_respect' must be
    measured against.
    """
    atoms, suspect = [], set()
    for line in prompt.split('\n'):
        m = TABLE_ROW.match(line.rstrip())
        if not m:
            continue
        inter = m.group('inter').strip()
        atoms.append({'atom': m.group('atom'), 'faces': m.group('faces'),
                      'distance_A': float(m.group('dist')), 'interaction': inter})
        if 'uncertain' in inter.lower() or 'unreliable' in inter.lower():
            suspect.add(m.group('faces').split('.')[0])
    assert atoms, 'no contact rows parsed out of the mol1 prompt'
    return {'atoms': atoms, 'input_smiles': input_smiles,
            'suspect_residues': sorted(suspect)}


HEADER = """\
================================================================================
ZAP70 / mol1 -- CANDIDATE ANALOGUES, batch %(batch)02d of %(nbatch)d (%(n)d items)
================================================================================
TARGET: P43403
INPUT SMILES: %(smiles)s

%(table)s

READ BEFORE SCORING -- the same three facts hold for every item in every batch.

1. mol1 IS A COVALENT INHIBITOR. It carries an acrylamide Michael acceptor that bonds CYS346. A
   proposal that removes that warhead, or alpha-methylates it to a methacrylamide (far less
   thiol-reactive), fails no matter what else it gets right.

2. THE TABLE IS PREDICTED, NOT MEASURED. mol1 has no crystal structure; the table comes from a
   cofolded model. Treat the geometry as approximate. The warhead C-S bond is modelled at 1.40 A
   where a real C-S bond is ~1.81 A: the BOND is asserted by the pose setup, its LENGTH is
   unreliable. Citing the bond is correct. Quoting 1.40 A as a measured fact is not.

3. ATOM NAMES are element + running index over heavy atoms in molecule order (C1, C2, ... O1, N1).
   Derive the mapping from the input SMILES yourself with RDKit -- which atoms are the acrylamide,
   the isoindoline, the carboxamide, the imidazole, the N-isopropyl -- and therefore which rows each
   group owns. Earlier rounds found that this derivation, rather than trusting the prose, settled
   most judgements.

ALREADY CHECKED, so you do not have to: every candidate below parses, is not the input unchanged,
cites at least two distances, and every cited (residue, distance) pair is a real row of the table.
Fabricated numbers are not the failure mode here.

Candidates are shuffled across the whole pool and model identity is hidden. They span the full range
of an automated score, so expect a mix; position carries no information.

%(dups)s
--------------------------------------------------------------------------------

"""

ITEM = """\
  %(id)s
    molecule:  %(smiles)s
    reasoning: %(sentence)s

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--task-dir', required=True,
                    help='e.g. ~/Downloads/covalentformer_labelling_task/mol1_candidate_pick')
    ap.add_argument('--per-batch', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    task = os.path.expanduser(a.task_dir)

    rec = json.loads(open(PROMPT).readline())
    table = contact_table(rec['prompt'])
    n_raw, n_kept, cands = collect()

    rw = load_reward()
    mi = model_input_from_prompt(rec['prompt'], rec['input_smiles'])
    assert len(mi['atoms']) == rec['n_atom_rows'], (
        'parsed %d contact rows but the prompt record declares %d'
        % (len(mi['atoms']), rec['n_atom_rows']))
    for c in cands:
        sc = rw.score(mi, c['gen_sentence'], c['gen_smiles'], rec['input_smiles'])
        c['_reward'] = sc['total']
        c['_components'] = sc['components']

    # Deterministic shuffle: the order must be reproducible from the seed alone, so that the key
    # and the batches can be rebuilt independently and still agree.
    random.Random(a.seed).shuffle(cands)
    for i, c in enumerate(cands, 1):
        c['_id'] = 'item_%03d' % i

    bymol = collections.defaultdict(list)
    for c in cands:
        bymol[c['gen_smiles_canonical']].append(c['_id'])
    dupgroups = sorted([v for v in bymol.values() if len(v) > 1])

    os.makedirs(os.path.join(task, 'batches'), exist_ok=True)
    os.makedirs(os.path.join(task, 'score_out'), exist_ok=True)
    os.makedirs(os.path.join(task, 'tools'), exist_ok=True)

    # Even split, not fixed-size chunks: 181 items at 25 each leaves a final batch of 6, and a
    # reviewer's calibration drifts with batch length, so the short batch is not comparable to the
    # others. nb is the batch COUNT implied by --per-batch; sizes then differ by at most one.
    nb_target = max(1, -(-len(cands) // a.per_batch))
    batches, lo = [], 0
    for k in range(nb_target):
        size = len(cands) // nb_target + (1 if k < len(cands) % nb_target else 0)
        batches.append(cands[lo:lo + size])
        lo += size
    nb = len(batches)
    dup_note = (
        'DUPLICATE MOLECULES: %d molecules in this pool appear more than once, each time with a\n'
        'DIFFERENT rationale, and the repeats are spread across batches. That is deliberate -- the\n'
        'rationale is what is being judged. Score each item on its own sentence.' % len(dupgroups)
    ) if dupgroups else ''

    manifest = {}
    for bi, batch in enumerate(batches, 1):
        name = 'batch_%02d.txt' % bi
        manifest[name] = [c['_id'] for c in batch]
        body = HEADER % {'batch': bi, 'nbatch': nb, 'n': len(batch),
                         'smiles': rec['input_smiles'], 'table': table, 'dups': dup_note}
        for c in batch:
            # Printed exactly as the JSON key the reviewer must use. The 24-candidate round
            # printed 'CANDIDATE 01' against a 'cand_01' key and relied on the reviewer to
            # transliterate; there is no reason to leave that gap open.
            body += ITEM % {'id': c['_id'],
                            'smiles': c['gen_smiles'], 'sentence': norm(c['gen_sentence'])}
        open(os.path.join(task, 'batches', name), 'w').write(body)
    json.dump(manifest, open(os.path.join(task, 'batch_manifest.json'), 'w'), indent=1)

    key = {c['_id']: {'arm': c['arm'], 'smiles': c['gen_smiles_canonical'],
                      'reward': round(c['_reward'], 4),
                      'components': {k: round(v, 4) for k, v in c['_components'].items()},
                      'n_cited': len(c.get('named_residues') or []),
                      'cited': c.get('named_residues')} for c in cands}
    for grp in dupgroups:
        for i in grp:
            key[i]['duplicate_molecule_with'] = [x for x in grp if x != i]
    keypath = os.path.join(HERE, 'mol1_candidate_task_KEY_DO_NOT_SHIP.json')
    json.dump(key, open(keypath, 'w'), indent=1)

    open(os.path.join(task, 'START_HERE.md'), 'w').write(start_here(nb, len(cands), len(dupgroups)))
    open(os.path.join(task, 'tools', 'verify_output.py'), 'w').write(VERIFY)

    old = os.path.join(task, 'candidates.txt')
    if os.path.exists(old):
        os.rename(old, os.path.join(task, 'candidates_SUPERSEDED_24.txt'))

    print('raw generations (g4+g3g4) %d -> pass filter %d -> unique candidates %d'
          % (n_raw, n_kept, len(cands)))
    print('arms: %s' % dict(collections.Counter(c['arm'] for c in cands)))
    print('distinct molecules %d, of which %d carry more than one rationale'
          % (len(bymol), len(dupgroups)))
    rr = sorted(c['_reward'] for c in cands)
    print('automated reward: min %.3f  median %.3f  max %.3f  zero-scored %d'
          % (rr[0], rr[len(rr) // 2], rr[-1], sum(1 for x in rr if x == 0)))
    print('wrote %d batches of <=%d to %s' % (nb, a.per_batch, os.path.join(task, 'batches')))
    print('key (NOT in the task folder): %s' % keypath)
    return 0


def start_here(nbatch, ntotal, ndup):
    return """\
# Kick-off prompt — pick the mol1 candidates worth making

Paste everything below the line to the agent that will run this task.

---

You are a **medicinal chemist on a covalent-inhibitor programme**, not a labeller. Nothing you write
becomes training data. We have a designed covalent seed against ZAP70 and two fine-tuned models have
proposed analogues of it. We need to know **which ones a chemist would actually make, and why** —
and the *reasoning* matters as much as the molecule.

## What to do

1. Read this file in full.
2. Work through the **{nbatch} files in `batches/`** — **{ntotal} candidates** in total, up to 25 per
   file. Each batch file repeats the input molecule, the full contact table, and the three standing
   caveats, so a batch is self-contained.
3. **Three reviewers independently per batch.** They must not read each other's output.
4. Each reviewer writes `score_out/batch_NN_rev{{A,B,C}}.json`.

That is 3 agents per batch, {nruns} runs, {ntotal} items, {nslots} slot judgements.

## What this pool is

Every candidate is an analogue of the same input molecule, from **two fine-tuned models in shuffled
order with model identity hidden**. We hold the key; do not try to infer which model wrote what.

This is the **whole** pool of generations that passed the automated filter — not a pre-selected
shortlist. An earlier round showed only 24 of these, which was a limit of the file layout rather
than of the candidates, and it meant the best analogue may never have been seen. Expect a wide
range of quality.

**{ndup} molecules appear more than once with different reasoning**, spread across batches. That is
deliberate: when two rationales defend the same molecule, which one is honest is exactly what we
want to learn. Score each item on its own sentence, and if you notice a repeat, say so in the
`one_line_why`.

## What we have already checked, so you don't have to

Every candidate parses, is not the input unchanged, cites at least two distances, and **every cited
(residue, distance) pair is a real row of the table**. Fabricated numbers are not the failure mode.

## What the filter cannot check — this is the job

1. **Ownership.** Is each cited contact owned by the group the edit actually touches? The recurring
   failure is a *real* distance belonging to a group the edit **leaves alone**, credited to the group
   that changed. In an earlier round of 10 candidates, **7 of 10 failed exactly this**.
2. **Does the molecule match the claim?** Sentences have promised a hydroxyl and delivered a plain
   hydrogen, called an appended ester a "methylene insertion", and silently swapped one heteroaryl
   for another. Diff input against candidate in RDKit: formula, MCS, ring sizes, stereocentres.

## Scoring

Per candidate, five booleans plus a one-line reason. **The rubric is unchanged from the 24-candidate
round**, deliberately, so the two can be compared.

| field | true when |
|---|---|
| `molecule_is_sound_analogue` | a single-step, synthesisable analogue that would survive water, with the acrylamide intact and **not** α-methylated |
| `claim_matches_molecule` | the molecule does what the sentence says it does |
| `ownership_correct` | every cited contact is owned by an atom the edit touched, or is correctly described as retained |
| `caveat_respected` | it does not assert the flagged 1.40 Å length as established fact |
| `would_make_it` | you would put this on a synthesis list for this programme |

Then, per batch, a `best_of_batch` block: `top_three` (item ids, ranked, best first),
`any_worth_making` (bool), and `why` — a short paragraph on what separated the good from the bad.

**There is a second stage.** We pool every `top_three` across all {nbatch} batches and all three
reviewers, then run one short finals round on the nominated items to produce a single ranking. So
`top_three` is a nomination, not a final verdict — nominate generously rather than conservatively.

## Rules

1. Score every candidate in your batch. A skipped one is indistinguishable from a false one.
2. Never assert a number you did not look up in the table.
3. Three independent reviewers per batch, no peeking.
4. There is no answer key in this folder. Do not look for one.
5. Run `python tools/verify_output.py` when you are done; it checks every batch's output parses and
   covers exactly the items in `batch_manifest.json`.

## Output

`score_out/batch_NN_rev{{A,B,C}}.json` — one JSON object per reviewer per batch:

```json
{{"candidates": {{"item_007": {{"molecule_is_sound_analogue": true, "claim_matches_molecule": false,
                             "ownership_correct": false, "caveat_respected": true,
                             "would_make_it": false, "one_line_why": "..."}},
                "...": {{}}}},
 "best_of_batch": {{"top_three": ["item_007", "item_019", "item_002"],
                   "any_worth_making": true, "why": "..."}}}}
```
""".format(nbatch=nbatch, ntotal=ntotal, ndup=ndup,
           nruns=nbatch * 3, nslots=ntotal * 5 * 3)


VERIFY = '''#!/usr/bin/env python3
"""Check every reviewer file parses and covers exactly its batch. Run from the task folder."""
import json, os, sys

FIELDS = ['molecule_is_sound_analogue', 'claim_matches_molecule', 'ownership_correct',
          'caveat_respected', 'would_make_it']
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(HERE, 'batch_manifest.json')))

bad = 0
for batch, items in sorted(man.items()):
    nn = batch.split('_')[1].split('.')[0]
    for rev in 'ABC':
        p = os.path.join(HERE, 'score_out', 'batch_%s_rev%s.json' % (nn, rev))
        if not os.path.exists(p):
            print('MISSING  %s' % os.path.relpath(p, HERE)); bad += 1; continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print('UNPARSEABLE %s: %s' % (os.path.relpath(p, HERE), e)); bad += 1; continue
        got = set(d.get('candidates') or {})
        want = set(items)
        if got != want:
            print('COVERAGE %s: missing %s, unexpected %s'
                  % (os.path.relpath(p, HERE), sorted(want - got) or '-', sorted(got - want) or '-'))
            bad += 1
        for k, v in (d.get('candidates') or {}).items():
            miss = [f for f in FIELDS if f not in v]
            if miss:
                print('FIELDS   %s %s missing %s' % (os.path.relpath(p, HERE), k, miss)); bad += 1
            for f in FIELDS:
                if f in v and not isinstance(v[f], bool):
                    print('TYPE     %s %s.%s is %r, expected true/false'
                          % (os.path.relpath(p, HERE), k, f, v[f])); bad += 1
        bob = d.get('best_of_batch') or {}
        if not isinstance(bob.get('top_three'), list) or not bob.get('top_three'):
            print('BEST     %s: best_of_batch.top_three missing or empty'
                  % os.path.relpath(p, HERE)); bad += 1
        else:
            stray = [x for x in bob['top_three'] if x not in want]
            if stray:
                print('BEST     %s: top_three names items not in this batch: %s'
                      % (os.path.relpath(p, HERE), stray)); bad += 1

print('\\n%s' % ('OK -- %d batches x 3 reviewers complete and consistent' % len(man)
                 if not bad else '%d problem(s) above' % bad))
sys.exit(1 if bad else 0)
'''


if __name__ == '__main__':
    raise SystemExit(main())
