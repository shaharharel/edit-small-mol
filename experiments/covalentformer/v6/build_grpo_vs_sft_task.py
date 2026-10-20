#!/usr/bin/env python3
"""Build a blind head-to-head chemist task: SFT reasoning vs GRPO reasoning, same prompt.

WHY A HEAD-TO-HEAD AND NOT AN ABSOLUTE SCORE
--------------------------------------------
On the 262 held-out pairs the automated reward puts the two arms at 0.8421 (GRPO, after the
trailing-SMILES defect is repaired) against 0.8443 (SFT) -- a 0.002 difference, i.e. it cannot tell
them apart. It also cannot see the failure mode the chemist panels keep finding: a contact that is
real, and cited, but owned by a group the edit leaves alone. `directional`, the verifier meant to
catch that, carries weight 0.00 because it agreed with the chemist majority on 2 of 4 measurable
cases. So the question "did GRPO change the reasoning" is not answerable from our metrics, and the
only instrument left is a chemist reading both and saying which is better.

Pairing the two arms on the SAME prompt removes prompt difficulty from the comparison, which an
absolute score per arm cannot do at this sample size.

WHAT IS EXCLUDED, AND WHY
-------------------------
  * 60 of the 262 sentences are character-identical between the arms. There is nothing to judge.
  * The GRPO arm appends junk to the SMILES on 30 of 262 generations ('replace', 'markdown',
    'argo'), which breaks RDKit and accounts for the entire apparent 0.109 reward gap. That is a
    generation-termination defect, not a chemistry or reasoning defect, and asking chemists to
    re-discover it wastes the panel. The SMILES is repaired here -- longest parsing prefix -- and
    the repair is disclosed per item so nobody scores a molecule we edited without knowing.

The arm labels A and B are shuffled per item, so position carries no information, and the key is
written outside the task folder.
"""
import argparse
import collections
import difflib
import json
import os
import random
import re

HERE = os.path.dirname(os.path.abspath(__file__))
SFT = os.path.join(HERE, 'g3g4_eval_results', 'chemdfm', 'generations.jsonl')
DATA = os.path.join(HERE, 'sft_out_g3g4', 'sft_pocket.jsonl')


def strip_specials(t):
    return re.sub(r'<\|[^|>]*\|>', '', t or '').replace('</s>', '')


def split_completion(t):
    t = strip_specials(t)
    sent = t.split('SMILES:')[0].strip()
    smi = t.split('SMILES:')[-1].strip().split()[0] if 'SMILES:' in t else ''
    return sent, smi


def repair(smi):
    """Longest prefix that RDKit parses, to undo a fused junk suffix. Returns (smiles, trimmed)."""
    from rdkit import Chem
    if not smi or Chem.MolFromSmiles(smi) is not None:
        return smi, ''
    for n in range(len(smi) - 1, max(len(smi) - 16, 0), -1):
        if Chem.MolFromSmiles(smi[:n]) is not None:
            return smi[:n], smi[n:]
    return smi, ''


def _canon(smi):
    from rdkit import Chem
    m = Chem.MolFromSmiles(smi) if smi else None
    return Chem.MolToSmiles(m) if m is not None else None


def table_block(prompt):
    i = prompt.find('MEASURED POCKET CONTACTS')
    j = prompt.find('### Response')
    assert i >= 0 and j > i, 'prompt has no contact-table block'
    return prompt[i:j].strip()


HEADER = """\
================================================================================
REASONING HEAD-TO-HEAD -- batch %(b)02d of %(nb)d (%(n)d cases)
================================================================================

Each case gives one crystallised input ligand, its MEASURED contact table, and TWO proposals from
two different fine-tuned models. Same prompt, same table, model identity hidden, A/B order shuffled
per case. We hold the key.

YOUR JOB: say which proposal's REASONING a medicinal chemist would rather have, and why. The
molecules matter only insofar as the sentence describes them truthfully.

The two models score 0.8421 and 0.8443 on our automated reward -- it cannot tell them apart. It also
cannot check the failure mode that matters most here, so that check is the core of this task:

  OWNERSHIP. Is each cited contact owned by an atom the edit actually touches? The recurring failure
  is a REAL distance, correctly quoted from the table, belonging to a group the edit LEAVES ALONE
  and credited to the group that changed. In an earlier round on a different target, 7 of 10
  proposals failed exactly this. Derive the atom-to-group mapping yourself with RDKit from the input
  SMILES rather than trusting either sentence.

Also worth your attention, because we suspect both models share it: these sentences describe WHAT
contact changes but rarely WHY a chemist would want that -- no potency, selectivity or property
rationale. If that is your read too, say so in `why`; if one arm does better at it, that is exactly
the kind of difference we are looking for.

NOTE ON MOLECULES: where a model's SMILES carried a trailing junk string that broke parsing, we
trimmed it to the longest parsing prefix and the case says so. That defect is understood and is not
what we are asking you to judge.

--------------------------------------------------------------------------------

"""

CASE = """\
%(id)s   target %(target)s   pdb %(pdb)s
INPUT SMILES: %(inp)s

%(table)s

  PROPOSAL A
    reasoning: %(a_sent)s
    molecule:  %(a_smi)s%(a_note)s

  PROPOSAL B
    reasoning: %(b_sent)s
    molecule:  %(b_smi)s%(b_note)s

--------------------------------------------------------------------------------

"""


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--grpo-gens', required=True, help='generations.jsonl from the GRPO arm')
    ap.add_argument('--task-dir', required=True)
    ap.add_argument('--per-batch', type=int, default=25)
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    task = os.path.expanduser(a.task_dir)
    rng = random.Random(a.seed)

    meta = {}
    for line in open(DATA):
        r = json.loads(line)
        if r['split'] == 'val':
            meta[r['pair']] = r
    g = {r['pair']: r for r in (json.loads(l) for l in open(os.path.expanduser(a.grpo_gens)))}
    s = {r['pair']: r for r in (json.loads(l) for l in open(SFT))}

    cases, identical, repaired = [], 0, 0
    for pair in sorted(set(g) & set(s) & set(meta)):
        gs, gm = split_completion(g[pair].get('completion'))
        ss, sm = split_completion(s[pair].get('completion'))
        gm2, gtrim = repair(gm)
        sm2, strim = repair(sm)
        # Drop a case only when there is nothing to compare: the sentences are near-identical AND
        # the molecule is the same. Exact-match alone is far too weak a filter -- 60 of 262 are
        # character-identical, but another 59 differ by a word or two ('a hydrophobic methyl'
        # against 'a hydrophobic group') over an identical molecule, and putting those in front of
        # a chemist spends the panel's attention on wording variants.
        gn, sn = ' '.join(gs.split()), ' '.join(ss.split())
        sim = difflib.SequenceMatcher(None, gn, sn).ratio()
        same_mol = (_canon(gm2) is not None and _canon(gm2) == _canon(sm2))
        if sim >= 0.90 and same_mol:
            identical += 1
            continue
        repaired += bool(gtrim) + bool(strim)
        arms = [('grpo', gs, gm2, gtrim), ('sft', ss, sm2, strim)]
        rng.shuffle(arms)                      # which arm is A is decided per case
        cases.append({'pair': pair, 'meta': meta[pair], 'A': arms[0], 'B': arms[1]})
    rng.shuffle(cases)
    for i, c in enumerate(cases, 1):
        c['id'] = 'case_%03d' % i

    os.makedirs(os.path.join(task, 'batches'), exist_ok=True)
    os.makedirs(os.path.join(task, 'score_out'), exist_ok=True)
    os.makedirs(os.path.join(task, 'tools'), exist_ok=True)

    nb = max(1, -(-len(cases) // a.per_batch))
    batches, lo = [], 0
    for k in range(nb):
        sz = len(cases) // nb + (1 if k < len(cases) % nb else 0)
        batches.append(cases[lo:lo + sz]); lo += sz

    manifest, key = {}, {}
    for bi, batch in enumerate(batches, 1):
        name = 'batch_%02d.txt' % bi
        manifest[name] = [c['id'] for c in batch]
        body = HEADER % {'b': bi, 'nb': nb, 'n': len(batch)}
        for c in batch:
            m = c['meta']
            note = lambda tr: ('    [SMILES repaired: trailing %r trimmed]' % tr) if tr else ''
            body += CASE % {
                'id': c['id'], 'target': m.get('target', '?'), 'pdb': m.get('pdb', '?'),
                'inp': m['input_smiles'], 'table': table_block(m['prompt']),
                'a_sent': ' '.join(c['A'][1].split()), 'a_smi': c['A'][2],
                'a_note': note(c['A'][3]),
                'b_sent': ' '.join(c['B'][1].split()), 'b_smi': c['B'][2],
                'b_note': note(c['B'][3])}
            key[c['id']] = {'pair': c['pair'], 'A': c['A'][0], 'B': c['B'][0],
                            'A_trimmed': c['A'][3], 'B_trimmed': c['B'][3]}
        open(os.path.join(task, 'batches', name), 'w').write(body)

    json.dump(manifest, open(os.path.join(task, 'batch_manifest.json'), 'w'), indent=1)
    keypath = os.path.join(HERE, 'grpo_vs_sft_task_KEY_DO_NOT_SHIP.json')
    json.dump(key, open(keypath, 'w'), indent=1)
    open(os.path.join(task, 'START_HERE.md'), 'w').write(start_here(nb, len(cases), identical))
    open(os.path.join(task, 'tools', 'verify_output.py'), 'w').write(VERIFY)

    sides = collections.Counter(v['A'] for v in key.values())
    print('val pairs in both arms: %d' % len(set(g) & set(s) & set(meta)))
    print('dropped as nothing-to-compare (>=0.90 similar AND same molecule): %d' % identical)
    print('cases built: %d   SMILES repaired: %d' % (len(cases), repaired))
    print('arm placed at A: %s  (should be ~half each)' % dict(sides))
    print('wrote %d batches to %s' % (nb, os.path.join(task, 'batches')))
    print('key (NOT in the task folder): %s' % keypath)
    return 0


def start_here(nb, n, identical):
    return """\
# Kick-off prompt — which model reasons better?

Paste everything below the line to the agent that will run this task.

---

You are a **medicinal chemist** comparing two models, not a labeller. Nothing you write becomes
training data.

Two fine-tuned models each proposed one analogue, with a rationale, for the same crystallised input
ligand. We need to know **whose reasoning a chemist would rather have, and why**.

## Why we are asking you rather than measuring it

Our automated reward scores the two arms **0.8421** and **0.8443** — it cannot distinguish them. It
also cannot check the failure mode we care most about (ownership, below): the verifier built for it
agreed with the chemist majority on 2 of 4 measurable cases, so it is switched off. You are the only
instrument we have for this question.

## What to do

1. Read this file in full.
2. Work through the **{nb} files in `batches/`** — **{n} cases** total. Each batch file is
   self-contained: every case carries its own input SMILES and its own measured contact table.
3. **Three reviewers independently per batch.** They must not read each other's output.
4. Each writes `score_out/batch_NN_rev{{A,B,C}}.json`.

{identical} of the 262 held-out cases were dropped because there was nothing to compare: the two
sentences were at least 90% identical AND both models proposed the same molecule. Every case you
see differs in the molecule, in the reasoning, or in both.

## The two checks that matter

**1. Ownership — the core of this task.** Is each cited contact owned by an atom the edit actually
touches? The recurring failure is a *real* distance, quoted correctly from the table, that belongs
to a group the edit **leaves alone**. Derive the atom-to-group mapping yourself with RDKit from the
input SMILES; do not trust either sentence's own account of which group it is editing. In an earlier
round on a different target, 7 of 10 proposals failed exactly this.

**2. Is it a hypothesis or just bookkeeping?** We suspect both arms describe *what contact changes*
without saying *why a chemist would want that* — no potency, selectivity or property rationale. If
that is your read, say so. If one arm does it better, that is precisely the difference we are after.

## Scoring

Per case:

| field | meaning |
|---|---|
| `better_reasoning` | `"A"`, `"B"` or `"tie"` — whose rationale you would rather have |
| `a_ownership_ok` | every contact A cites is owned by an atom A's edit touches (or is correctly described as retained) |
| `b_ownership_ok` | same for B |
| `a_would_use` | you would put A's molecule-plus-rationale in front of a chemistry audience |
| `b_would_use` | same for B |
| `why` | one line: what separated them, or why they tie |

Then per batch, a `summary` block: `a_wins`, `b_wins`, `ties` (ints), and `pattern` — a short
paragraph on any systematic difference you noticed, including whether either arm offers real design
rationale.

## Rules

1. Score every case in your batch. A skipped case is indistinguishable from a tie.
2. Never assert a distance you did not look up in that case's own table.
3. A and B are shuffled per case. Position carries no information; do not try to infer which model
   is which.
4. There is no answer key in this folder. Do not look for one.
5. Where a case says the SMILES was repaired, a trailing junk string was trimmed to the longest
   parsing prefix. That defect is understood and is **not** what we are asking you to judge.
6. Run `python tools/verify_output.py` when done.

## Output

`score_out/batch_NN_rev{{A,B,C}}.json`:

```json
{{"cases": {{"case_007": {{"better_reasoning": "A", "a_ownership_ok": true,
                         "b_ownership_ok": false, "a_would_use": true,
                         "b_would_use": false, "why": "..."}},
            "...": {{}}}},
 "summary": {{"a_wins": 9, "b_wins": 11, "ties": 5, "pattern": "..."}}}}
```
""".format(nb=nb, n=n, identical=identical)


VERIFY = '''#!/usr/bin/env python3
"""Check every reviewer file parses and covers exactly its batch. Run from the task folder."""
import json, os, sys

BOOL = ['a_ownership_ok', 'b_ownership_ok', 'a_would_use', 'b_would_use']
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
        got, want = set(d.get('cases') or {}), set(ids)
        if got != want:
            print('COVERAGE %s: missing %s, unexpected %s'
                  % (rel, sorted(want - got) or '-', sorted(got - want) or '-')); bad += 1
        for k, v in (d.get('cases') or {}).items():
            if v.get('better_reasoning') not in ('A', 'B', 'tie'):
                print('VALUE    %s %s.better_reasoning=%r, expected A/B/tie'
                      % (rel, k, v.get('better_reasoning'))); bad += 1
            for f in BOOL:
                if not isinstance(v.get(f), bool):
                    print('TYPE     %s %s.%s=%r, expected true/false'
                          % (rel, k, f, v.get(f))); bad += 1
            if not str(v.get('why') or '').strip():
                print('EMPTY    %s %s.why is blank' % (rel, k)); bad += 1
        s = d.get('summary') or {}
        if not all(isinstance(s.get(x), int) for x in ('a_wins', 'b_wins', 'ties')):
            print('SUMMARY  %s: a_wins/b_wins/ties must be integers' % rel); bad += 1
        elif s['a_wins'] + s['b_wins'] + s['ties'] != len(ids):
            print('SUMMARY  %s: counts sum to %d, batch has %d cases'
                  % (rel, s['a_wins'] + s['b_wins'] + s['ties'], len(ids))); bad += 1

print('\\n%s' % ('OK -- %d batches x 3 reviewers complete and consistent' % len(man)
                 if not bad else '%d problem(s) above' % bad))
sys.exit(1 if bad else 0)
'''


if __name__ == '__main__':
    raise SystemExit(main())
