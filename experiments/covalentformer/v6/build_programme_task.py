#!/usr/bin/env python3
"""Build the chemist task for the five unseen covalent programmes.

WHY THIS TASK IS DIFFERENT FROM THE mol1 ONE
--------------------------------------------
mol1's ownership could be filtered automatically because the round-1 finals reviewer published its
atom-order mapping. Here that is impossible and the reason is recorded in reward_verifiers.py: the
contact tables are keyed by the deposition's own CCD atom names, rows are sorted alphabetically
rather than in molecule order, and mapping a CCD name back to a molecule position matches
AMBIGUOUSLY for 79% of ligands. That is why the `directional` verifier carries weight 0.00 after
three attempts. So OWNERSHIP IS THE QUESTION THIS TASK EXISTS TO ANSWER, not something it can
pre-filter.

The number we are after: an external panel put atom_attribution_ok at 70% on 150 held-out crystal
generations and the mol1 panel put ownership_correct at 29% on 181. mol1 confounds three shifts at
once -- unseen target, predicted pose, sampled decoding -- so that gap cannot be apportioned. These
programmes are unseen targets with MEASURED crystal geometry and CCD atom names, so if ownership
lands near 70% the pose was mol1's problem, and if it lands near 29% the unseen target was.

EACH CANDIDATE CARRIES ITS OWN TABLE. Unlike mol1, the 18 inputs are different ligands in different
proteins, so a batch cannot share one table and every case prints its own.

THE REFERENCE ANALOGUE IS WITHHELD. Every input here is one half of a real same-series pair, and the
other half is a compound the programme actually made. It is in the key, never in the task: showing
it would turn a judgement about reasoning into a comparison against an answer.
"""
import argparse
import collections
import importlib.util
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, path))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def mcs_groups(inp, gen):
    """(removed, added) heavy-atom element multisets, for the group-naming check."""
    from rdkit import Chem
    from rdkit.Chem import rdFMCS
    a, b = Chem.MolFromSmiles(inp), Chem.MolFromSmiles(gen)
    if a is None or b is None:
        return None
    m = rdFMCS.FindMCS([a, b], timeout=5, ringMatchesRingOnly=True)
    core = Chem.MolFromSmarts(m.smartsString) if m.smartsString else None
    if core is None:
        return None
    ka, kb = set(a.GetSubstructMatch(core)), set(b.GetSubstructMatch(core))
    rm = collections.Counter(a.GetAtomWithIdx(i).GetSymbol()
                             for i in range(a.GetNumAtoms()) if i not in ka)
    ad = collections.Counter(b.GetAtomWithIdx(i).GetSymbol()
                             for i in range(b.GetNumAtoms()) if i not in kb)
    return rm, ad


HEADER = """\
================================================================================
UNSEEN COVALENT PROGRAMMES -- batch %(b)02d of %(nb)d (%(n)d candidates)
================================================================================

Each case below is a DIFFERENT ligand in a DIFFERENT protein, so every case prints its own input
SMILES and its own measured contact table. Read each case against its own table only.

These are five covalent programmes the model has never seen: JAK3 (Cys909), the 20S proteasome
subunit PSMB5 (Thr1), cathepsin K (Cys25), HER2 (Cys805) and HCV NS3/4A (Ser139). Every contact
table here is MEASURED from a deposited crystal, not predicted.

HOW TO READ A TABLE. Columns are atom name, element, the nearest protein atom it faces
(RES123.ATOM), chain, distance in Angstrom, the angle where one is defined, and the interaction
typing. Atom names are the deposition's own CCD names (C01, N10, O13) and rows are sorted
ALPHABETICALLY, not in molecule order -- so you cannot map a name to a molecule position by
counting. Recovering which group owns a row needs the chemistry. A row typed `polar/apolar packing`
or `hydrophobic contact` is proximity, not an interaction. Some rows carry explicit refusals such as
`donor/acceptor INCOMPATIBLE ... not a hydrogen bond`.

THE MAIN QUESTION IS OWNERSHIP, and we cannot pre-check it here. For mol1 we could, because its
atom mapping was published; CCD names defeat it for these. So: is every contact a sentence cites
owned by an atom the edit actually touches, or is it a real distance belonging to a group the edit
leaves alone? Derive the edit yourself in RDKit from the two SMILES; do not trust the sentence's own
account of which group it changes.

ALREADY CHECKED: every candidate parses, is not the input unchanged, and every cited
(residue, distance) pair is a real row of its own table. Fabricated numbers are not the failure mode.

Order is shuffled and the model identity is hidden. Some molecules may appear more than once with
different reasoning.

--------------------------------------------------------------------------------

"""

CASE = """\
%(id)s   programme %(programme)s   covalent site %(site)s   pdb %(pdb)s
INPUT SMILES: %(inp)s

%(table)s

  PROPOSED ANALOGUE: %(smiles)s
  MODEL REASONING  : %(sentence)s

--------------------------------------------------------------------------------

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--gens', nargs='+', required=True)
    ap.add_argument('--prompts', default=os.path.join(HERE, 'programme_prompts.jsonl'))
    ap.add_argument('--task-dir', required=True)
    ap.add_argument('--per-batch', type=int, default=20)
    ap.add_argument('--per-input', type=int, default=4,
                    help='cap per input molecule, so one easy pocket cannot dominate the batch')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    import random
    from rdkit import Chem, RDLogger
    RDLogger.DisableLog('rdApp.*')
    rw = load('rw', 'reward_verifiers.py')

    prompts = {r['pair']: r for r in (json.loads(l) for l in open(a.prompts))}
    rng = random.Random(a.seed)

    pool = collections.defaultdict(list)
    seen = set()
    stage = collections.Counter()
    for path in a.gens:
        for r in (json.loads(l) for l in open(path)):
            pr = prompts.get(r['pair'])
            if pr is None:
                continue
            mi = pr['MODEL_INPUT']
            for g in r['generations']:
                sent = ' '.join((g.get('gen_sentence') or '').split())
                smi = g.get('gen_smiles') or ''
                m = Chem.MolFromSmiles(smi) if smi else None
                if m is None:
                    continue
                key = (r['pair'], Chem.MolToSmiles(m), sent)
                if key in seen:
                    continue
                seen.add(key)
                stage['unique_parsing'] += 1
                s = rw.score(mi, sent, smi, pr['input_smiles'])
                if s['total'] == 0.0:
                    continue
                stage['verifier_nonzero'] += 1
                gr = mcs_groups(pr['input_smiles'], smi)
                if not gr or not (sum(gr[0].values()) or sum(gr[1].values())):
                    continue
                stage['has_detectable_edit'] += 1
                pool[r['pair']].append({'pair': r['pair'], 'sentence': sent, 'smiles': smi,
                                        'reward': round(s['total'], 3), 'decode': g['decode'],
                                        'removed': dict(gr[0]), 'added': dict(gr[1])})

    picked = []
    for pair, cands in pool.items():
        cands.sort(key=lambda c: -c['reward'])
        uniq, got = set(), 0
        for c in cands:
            cs = Chem.MolToSmiles(Chem.MolFromSmiles(c['smiles']))
            if cs in uniq:
                continue
            uniq.add(cs); picked.append(c); got += 1
            if got >= a.per_input:
                break
    rng.shuffle(picked)
    for n, c in enumerate(picked, 1):
        c['id'] = 'prog_%03d' % n

    task = os.path.expanduser(a.task_dir)
    os.makedirs(os.path.join(task, 'batches'), exist_ok=True)
    os.makedirs(os.path.join(task, 'score_out'), exist_ok=True)
    os.makedirs(os.path.join(task, 'tools'), exist_ok=True)

    nb = max(1, -(-len(picked) // a.per_batch))
    batches, lo = [], 0
    for k in range(nb):
        size = len(picked) // nb + (1 if k < len(picked) % nb else 0)
        batches.append(picked[lo:lo + size]); lo += size

    manifest, key = {}, {}
    for bi, batch in enumerate(batches, 1):
        name = 'batch_%02d.txt' % bi
        manifest[name] = [c['id'] for c in batch]
        body = HEADER % {'b': bi, 'nb': nb, 'n': len(batch)}
        for c in batch:
            pr = prompts[c['pair']]
            p = pr['prompt']
            i = p.index('MEASURED POCKET CONTACTS'); j = p.index('### Response')
            body += CASE % {'id': c['id'], 'programme': pr['programme'], 'site': pr['site'],
                            'pdb': pr['pdb'], 'inp': pr['input_smiles'],
                            'table': p[i:j].strip(), 'smiles': c['smiles'],
                            'sentence': c['sentence']}
            key[c['id']] = {'pair': c['pair'], 'programme': pr['programme'],
                            'reward': c['reward'], 'decode': c['decode'],
                            'removed': c['removed'], 'added': c['added'],
                            'reference_smiles_WITHHELD': pr['reference_smiles'],
                            'tanimoto_to_reference': pr['tanimoto']}
        open(os.path.join(task, 'batches', name), 'w').write(body)

    json.dump(manifest, open(os.path.join(task, 'batch_manifest.json'), 'w'), indent=1)
    kp = os.path.join(HERE, 'programme_task_KEY_DO_NOT_SHIP.json')
    json.dump(key, open(kp, 'w'), indent=1)
    byprog = collections.Counter(prompts[c['pair']]['programme'] for c in picked)
    open(os.path.join(task, 'START_HERE.md'), 'w').write(
        start_here(nb, len(picked), len(pool), dict(byprog)))
    open(os.path.join(task, 'tools', 'verify_output.py'), 'w').write(VERIFY)

    for k_, v in stage.items():
        print('  %-22s %d' % (k_, v))
    print('selected %d (<=%d per input) over %d inputs -> %d batches'
          % (len(picked), a.per_input, len(pool), nb))
    print('by programme: %s' % dict(byprog))
    print('task -> %s\nkey  -> %s' % (task, kp))
    return 0


def start_here(nb, n, ninputs, byprog):
    return """\
# Kick-off prompt — five covalent programmes the model has never seen

Paste everything below the line to the agent that will run this task.

---

You are a **senior medicinal chemist**. Nothing you write becomes training data.

A model fine-tuned on covalent analogue design was run on **five programmes absent from its
training data**: JAK3 (Cys909), the 20S proteasome subunit PSMB5 (Thr1), cathepsin K (Cys25), HER2
(Cys805) and HCV NS3/4A (Ser139). **{n} candidates** over **{ninputs} input ligands** are below:
{byprog}.

Every contact table here is **measured from a deposited crystal** — unlike the earlier mol1 round,
where the geometry was predicted from a cofold.

## The question we need answered

An external panel scored **70%** ownership-correct on held-out generations from the model's own
training distribution. The mol1 panel scored **29%**. mol1 differed in three ways at once — unseen
target, predicted pose, sampled decoding — so we cannot say which caused the drop. These candidates
are unseen targets with **measured** geometry. Where ownership lands tells us which it was.

**We cannot pre-filter ownership here.** For mol1 we could, because its atom mapping was published.
These tables use the deposition's own CCD atom names, sorted alphabetically rather than in molecule
order, and mapping a CCD name back to a molecule position is ambiguous for most ligands. So
ownership is the core of this task.

## What to do

1. Read this file in full.
2. Work through the **{nb} files in `batches/`**. Each case prints its own input SMILES and its own
   contact table — different ligands in different proteins, so never read one case against another's
   table.
3. **Three reviewers independently per batch.** They must not read each other's output.
4. Each writes `score_out/batch_NN_rev{{A,B,C}}.json`.

## Per candidate

| field | true when |
|---|---|
| `edit_identified_correctly` | diff the two SMILES in RDKit: does the sentence describe the change that actually separates them? Describing a group present in both, or ignoring the one that changed, is false |
| `ownership_correct` | every cited contact is owned by an atom the edit touches, or is correctly described as retained |
| `chemically_sensible_edit` | a synthesisable, water-stable single-step analogue that keeps the warhead |
| `reasoning_helpful` | would this sentence help you decide whether to make the compound, beyond restating the table? A correct but purely descriptive sentence is **not** helpful — mark it false and say so |
| `would_make_it` | you would put this on a synthesis list |
| `one_line_why` | the atom, row or distance that decided it |

Then per batch a `summary`: `n_ownership_ok`, `n_would_make` (ints) and `pattern` — in particular
whether these sentences ever say **why** an edit might help, or only which contact changes.

## Rules

1. Score every candidate. A skipped one is indistinguishable from a false one.
2. Never assert a distance you did not look up in that case's own table.
3. A row typed `polar/apolar packing` or `hydrophobic contact` is proximity, not chemistry.
4. There is no answer key in this folder. Do not look for one.
5. Run `python tools/verify_output.py` when done.

## Output

```json
{{"candidates": {{"prog_007": {{"edit_identified_correctly": true, "ownership_correct": false,
                             "chemically_sensible_edit": true, "reasoning_helpful": false,
                             "would_make_it": false, "one_line_why": "..."}}}},
 "summary": {{"n_ownership_ok": 9, "n_would_make": 3, "pattern": "..."}}}}
```
""".format(nb=nb, n=n, ninputs=ninputs, byprog=byprog)


VERIFY = '''#!/usr/bin/env python3
"""Check every reviewer file parses and covers exactly its batch. Run from the task folder."""
import json, os, sys
BOOL = ['edit_identified_correctly', 'ownership_correct', 'chemically_sensible_edit',
        'reasoning_helpful', 'would_make_it']
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
        for f in ('n_ownership_ok', 'n_would_make'):
            if not isinstance(s.get(f), int):
                print('SUMMARY  %s: %s must be an integer' % (rel, f)); bad += 1
        if not str(s.get('pattern') or '').strip():
            print('SUMMARY  %s: pattern is blank' % rel); bad += 1
print('\\n%s' % ('OK -- %d batches x 3 reviewers complete' % len(man)
                 if not bad else '%d problem(s) above' % bad))
sys.exit(1 if bad else 0)
'''


if __name__ == '__main__':
    raise SystemExit(main())
