#!/usr/bin/env python3
"""Build the round-2 mol1 chemist task: everything that clears the automated filters.

Round 1 sent all 181 candidates and 3 survived -- 543 slot judgements, most of them spent on
failures that funnel_mol1.py now detects without a chemist, each rule validated against that same
panel's majority (caveat 94%, ownership 83%). So this round sends only what the rules keep, and
asks the chemists the question no rule answers: is this edit worth making.

Funnel, from 502 generations over two prompt variants:
    463 unique and parsing
    415 verifier score > 0
    166 caveat-clean
     83 ownership-clean          <- what this task contains

THE TWO PROMPT VARIANTS ARE THE EXPERIMENT. mol1's pose is a cofold whose CYS346 bond length is not
measurable, and the original prompt printed "1.40" four times while the brief forbids asserting it.
Round 1 failed 142 of 181 on exactly that. The `no_length` variant prints no distance for that bond.
Keep rate went 11% -> 30%. Reviewers are not told which variant a candidate came from, and the
variant is recorded only in the key.

WHAT THE RULES ALREADY GUARANTEE, so the brief can tell reviewers not to re-check it:
every candidate parses, is not the input, cites >= 2 real table rows, does not quote a length for
the CYS346 bond, does not build on the row the table flags uncertain, does not call a no-angle polar
contact a hydrogen bond, and cites in its edit clause only contacts owned by atoms the edit touches.

WHAT THE RULES CANNOT GUARANTEE, and so is the job: whether the molecule is synthesisable and
water-stable, whether the claim matches the molecule in detail (the rules check the edited GROUP,
not the exact substituent -- round 1's item_100 called a 2-(diethylamino)ethyl arm "a
diethylamino"), and whether a chemist would make it.
"""
import argparse
import collections
import json
import os

HERE = os.path.dirname(os.path.abspath(__file__))

HEADER = """\
================================================================================
ZAP70 / mol1 -- ROUND 2, batch %(b)02d of %(nb)d (%(n)d candidates)
================================================================================
TARGET: P43403
INPUT SMILES: %(inp)s

%(table)s

READ BEFORE SCORING

1. mol1 IS A COVALENT INHIBITOR. Its acrylamide bonds CYS346. A proposal that removes that warhead,
   or alpha-methylates it to a methacrylamide (far less thiol-reactive), fails regardless.

2. THE TABLE IS PREDICTED, NOT MEASURED. mol1 has no crystal; this is a cofold. Treat the geometry
   as approximate. No distance is given for the CYS346 bond because a cofold cannot measure it.

3. ATOM NAMES are element + running index over heavy atoms in molecule order. The mapping, published
   by the round-1 finals reviewer and used by our filters, is:
       acrylamide   C1-C3, O1          isoindoline  N1, C4-C9, C17-C18
       carboxamide  C10, O2, N2        imidazole    C11, C12, N3, C16, N4
       N-isopropyl  C13, C14, C15
   and the N-isopropyl's own contacts are C14 -> LYS369.CG 3.67 and C13/C15 -> MET414.SD 3.63/3.53.

ALREADY CHECKED BY RULE -- do not spend time re-checking these. Every candidate below: parses; is
not the input; cites at least two real rows of the table; quotes no length for the CYS346 bond;
builds on no row the table flags uncertain; calls no angle-less polar contact a hydrogen bond; and
in its edit clause cites only contacts owned by atoms its edit touches.

STILL YOUR JOB, because no rule covers it:
  * is the molecule a synthesisable, water-stable, single-step analogue;
  * does the claim match the molecule in DETAIL -- our ownership rule checks the edited GROUP, not
    the exact substituent, and a round-1 candidate called a 2-(diethylamino)ethyl arm "a
    diethylamino";
  * would you make it.

Order is shuffled. Some molecules appear more than once with different reasoning; that is
deliberate, and which rationale is honest is part of what we want to know.

--------------------------------------------------------------------------------

"""

CASE = """\
  %(id)s
    molecule:  %(smiles)s
    reasoning: %(sentence)s

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--funnel', default=os.path.join(HERE, 'mol1_funnel.json'))
    ap.add_argument('--task-dir', required=True)
    ap.add_argument('--per-batch', type=int, default=21)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import random
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    import importlib.util as u
    spec = u.spec_from_file_location('tr', os.path.join(HERE, 'build_mol1_candidate_task.py'))
    tr = u.module_from_spec(spec); spec.loader.exec_module(tr)

    task = os.path.expanduser(a.task_dir)
    kept = json.load(open(a.funnel))['kept']
    base = json.loads(open(os.path.join(HERE, 'mol1_prompt_variants.jsonl')).readline())
    # the no_length table is what reviewers should see: it is the prompt we are standardising on
    nl = [json.loads(l) for l in open(os.path.join(HERE, 'mol1_prompt_variants.jsonl'))]
    table = tr.contact_block = None
    prompt = [r for r in nl if r.get('variant') == 'no_length'][0]['prompt']
    i = prompt.index('MEASURED POCKET CONTACTS')
    j = prompt.index('### Response')
    table = prompt[i:j].strip()

    rng = random.Random(a.seed)
    rng.shuffle(kept)
    for n, c in enumerate(kept, 1):
        c['id'] = 'r2_%03d' % n
    bymol = collections.defaultdict(list)
    for c in kept:
        m = Chem.MolFromSmiles(c['smiles'])
        bymol[Chem.MolToSmiles(m)].append(c['id'])

    os.makedirs(os.path.join(task, 'batches'), exist_ok=True)
    os.makedirs(os.path.join(task, 'score_out'), exist_ok=True)
    os.makedirs(os.path.join(task, 'tools'), exist_ok=True)

    nb = max(1, -(-len(kept) // a.per_batch))
    batches, lo = [], 0
    for k in range(nb):
        size = len(kept) // nb + (1 if k < len(kept) % nb else 0)
        batches.append(kept[lo:lo + size]); lo += size

    manifest, key = {}, {}
    for bi, batch in enumerate(batches, 1):
        name = 'batch_%02d.txt' % bi
        manifest[name] = [c['id'] for c in batch]
        body = HEADER % {'b': bi, 'nb': nb, 'n': len(batch),
                         'inp': base['input_smiles'], 'table': table}
        for c in batch:
            body += CASE % {'id': c['id'], 'smiles': c['smiles'], 'sentence': c['sentence']}
            key[c['id']] = {'variant': c['variant'], 'decode': c['decode'],
                            'reward': c['reward'], 'smiles': c['smiles'],
                            'edited_groups': c['edited_groups'],
                            'cited_owners': c.get('cited_owners')}
        open(os.path.join(task, 'batches', name), 'w').write(body)
    for mol, ids in bymol.items():
        if len(ids) > 1:
            for i_ in ids:
                key[i_]['duplicate_molecule_with'] = [x for x in ids if x != i_]

    json.dump(manifest, open(os.path.join(task, 'batch_manifest.json'), 'w'), indent=1)
    kp = os.path.join(HERE, 'mol1_round2_KEY_DO_NOT_SHIP.json')
    json.dump(key, open(kp, 'w'), indent=1)
    open(os.path.join(task, 'START_HERE.md'), 'w').write(
        start_here(nb, len(kept), len(bymol), collections.Counter(c['variant'] for c in kept)))
    open(os.path.join(task, 'tools', 'verify_output.py'), 'w').write(VERIFY)

    print('candidates: %d in %d batches (<=%d each)' % (len(kept), nb, a.per_batch))
    print('distinct molecules: %d' % len(bymol))
    print('variant split (KEY ONLY): %s' % dict(collections.Counter(c['variant'] for c in kept)))
    print('task -> %s' % task)
    print('key  -> %s' % kp)
    return 0


def start_here(nb, n, nmol, variants):
    return """\
# Kick-off prompt — mol1 round 2, which of these would you make?

Paste everything below the line to the agent that will run this task.

---

You are a **medicinal chemist on a covalent-inhibitor programme**, not a labeller. Nothing you write
becomes training data.

We have a designed covalent seed against ZAP70 and a fine-tuned model proposed analogues of it.
**{n} candidates** ({nmol} distinct molecules) are below. They are the survivors of an automated
filter, not a raw pool.

## Why this round is smaller than the last one

Round 1 sent you **181** candidates and **3** survived — 543 slot judgements, most of them spent on
failures we can now detect by rule. Those rules were built from your round-1 verdicts and validated
against them: the caveat rule agrees with your majority on **94%** of the 181, the ownership rule on
**83%**. So this round sends only what the rules keep, and asks you the part no rule answers.

## What to do

1. Read this file in full.
2. Work through the **{nb} files in `batches/`** — each is self-contained, with the input molecule
   and the full contact table.
3. **Three reviewers independently per batch.** They must not read each other's output.
4. Each writes `score_out/batch_NN_rev{{A,B,C}}.json`.

## Already checked by rule — please don't re-check

Every candidate parses, is not the input unchanged, cites at least two real rows of the table,
**quotes no length for the CYS346 bond**, builds on no row the table flags `uncertain`, calls no
angle-less polar contact a hydrogen bond, and in its edit clause cites only contacts owned by atoms
its edit touches.

## Still your job

| field | true when |
|---|---|
| `molecule_is_sound_analogue` | a single-step, synthesisable analogue that would survive water, acrylamide intact and **not** α-methylated |
| `claim_matches_molecule` | the molecule does what the sentence says **in detail**. Our rule checks the edited GROUP, not the exact substituent — a round-1 candidate called a 2-(diethylamino)ethyl arm "a diethylamino" and passed every rule |
| `would_make_it` | you would put this on a synthesis list for this programme |
| `one_line_why` | what decided it |

Then per batch a `summary` block: `n_would_make` (int), and `pattern` — a short paragraph on what
separated the good from the bad, **including any failure our rules let through**, which is the most
useful thing you can tell us.

## Rules

1. Score every candidate. A skipped one is indistinguishable from a false one.
2. Never assert a distance you did not look up in the table.
3. Three independent reviewers per batch, no peeking.
4. There is no answer key in this folder. Do not look for one.
5. Some molecules appear more than once with different reasoning. Score each on its own sentence,
   and say if you notice a repeat.
6. Run `python tools/verify_output.py` when done.

## Output

`score_out/batch_NN_rev{{A,B,C}}.json`:

```json
{{"candidates": {{"r2_007": {{"molecule_is_sound_analogue": true, "claim_matches_molecule": true,
                           "would_make_it": false, "one_line_why": "..."}},
                "...": {{}}}},
 "summary": {{"n_would_make": 4, "pattern": "..."}}}}
```
""".format(nb=nb, n=n, nmol=nmol)


VERIFY = '''#!/usr/bin/env python3
"""Check every reviewer file parses and covers exactly its batch. Run from the task folder."""
import json, os, sys
BOOL = ['molecule_is_sound_analogue', 'claim_matches_molecule', 'would_make_it']
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
man = json.load(open(os.path.join(HERE, 'batch_manifest.json')))
bad = 0
for batch, ids in sorted(man.items()):
    nn = batch.split('_')[1].split('.')[0]
    for rev in 'ABC':
        p = os.path.join(HERE, 'score_out', 'batch_%s_rev%s.json' % (nn, rev))
        rel = os.path.relpath(p, HERE)
        if not os.path.exists(p):
            print('MISSING  %s' % rel); bad += 1; continue
        try:
            d = json.load(open(p))
        except Exception as e:
            print('UNPARSEABLE %s: %s' % (rel, e)); bad += 1; continue
        got, want = set(d.get('candidates') or {}), set(ids)
        if got != want:
            print('COVERAGE %s: missing %s, unexpected %s'
                  % (rel, sorted(want - got) or '-', sorted(got - want) or '-')); bad += 1
        for k, v in (d.get('candidates') or {}).items():
            for f in BOOL:
                if not isinstance(v.get(f), bool):
                    print('TYPE     %s %s.%s=%r' % (rel, k, f, v.get(f))); bad += 1
            if not str(v.get('one_line_why') or '').strip():
                print('EMPTY    %s %s.one_line_why' % (rel, k)); bad += 1
        s = d.get('summary') or {}
        if not isinstance(s.get('n_would_make'), int):
            print('SUMMARY  %s: n_would_make must be an integer' % rel); bad += 1
        if not str(s.get('pattern') or '').strip():
            print('SUMMARY  %s: pattern is blank' % rel); bad += 1
print('\\n%s' % ('OK -- %d batches x 3 reviewers complete' % len(man)
                 if not bad else '%d problem(s) above' % bad))
sys.exit(1 if bad else 0)
'''


if __name__ == '__main__':
    raise SystemExit(main())
