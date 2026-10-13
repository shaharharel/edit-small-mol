#!/usr/bin/env python3
"""Reference-residue recall against an analytic chance baseline, from saved generations alone.

    python ref_recall.py sft_runs/<cell>/eval/generations.jsonl [more.jsonl ...] \
                         --data sft_out/sft_pocket.jsonl

WHY THIS EXISTS SEPARATELY FROM eval_axes.py
--------------------------------------------
eval_axes.py (the canonical scorer) reports `groundedness` and `follows_pocket_frac`, which ask
whether the residues a sentence names EXIST in the pocket table it was shown. Both are trivially
gamed: the table is in the prompt, so a model that copies any three residue names out of it scores
1.0 -- above the reference ceiling of 0.90, which is the tell that the metric is measuring copying
rather than understanding. The chat-template pilot did exactly that, scoring a perfect 1.0.

This asks the harder question: of the residues the CHAIR named, how many did the model name? That is
recall against the chemist's own choice, and it cannot be satisfied by copying arbitrary rows.

Recall alone is still not interpretable, because a model that names more residues scores higher by
luck. So it is reported against the ANALYTIC CHANCE BASELINE for each row: if the model drew its
`k = |named ∩ table|` residues uniformly at random from the `N = |table|` rows available, the
expected fraction of the chair's `R` grounded residues it would hit is just k/N -- the hypergeometric
mean, which needs no sampling. The honest number is the LIFT, recall / chance.

Both are macro-averaged over rows that have at least one grounded reference residue, since a row
where the chair named nothing checkable has no recall to measure.

The reference row is scored through the same code path and must come out at 1.0 by construction; if
it does not, this metric is broken rather than the model being superhuman.
"""
import argparse
import json
import os
import statistics
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sft_data import residue_tokens  # noqa: E402


def load_data(path):
    """pair -> {table, reference_residues}. The table must come from the DATA, never from the run.

    Taking the pocket table from the generations file would let the denominator move with whatever
    the model happened to emit, so the chance baseline would be defined by the thing being measured.
    """
    out = {}
    for line in open(path):
        r = json.loads(line)
        if r['split'] in ('train', 'val'):
            out[r['pair']] = {'table': set(r['table_residues']),
                              'ref': set(r['reference_residues'])}
    return out


def score(gen_path, data):
    rows = [json.loads(l) for l in open(gen_path)]
    recalls, chances, lifts = [], [], []
    ref_recalls = []
    n_skipped = 0
    for r in rows:
        d = data.get(r['pair'])
        if d is None:
            n_skipped += 1
            continue
        table = d['table']
        # Only the chair's residues that are IN the table are recallable: the ~10% the analogue's
        # own crystal supplied are unknowable from MODEL_INPUT and scoring them would push the
        # ceiling below 1.0 and make the metric unreadable.
        ref = d['ref'] & table
        if not ref:
            n_skipped += 1
            continue
        gen = residue_tokens(r.get('gen_sentence') or '') & table
        recall = len(gen & ref) / len(ref)
        chance = len(gen) / len(table) if table else 0.0
        recalls.append(recall)
        chances.append(chance)
        if chance > 0:
            lifts.append(recall / chance)
        # the ceiling: the reference scored by the identical path
        ref_gen = residue_tokens(r.get('reference_sentence') or '') & table
        ref_recalls.append(len(ref_gen & ref) / len(ref))
    mr = statistics.mean(recalls) if recalls else None
    mc = statistics.mean(chances) if chances else None
    return {
        'run': gen_path,
        'n_scored': len(recalls),
        'n_skipped': n_skipped,
        'ref_residue_recall': round(mr, 4) if mr is not None else None,
        'chance_baseline': round(mc, 4) if mc is not None else None,
        'lift_over_chance': round(mr / mc, 3) if (mr and mc) else None,
        'mean_per_row_lift': round(statistics.mean(lifts), 3) if lifts else None,
        'CEILING_reference_recall': round(statistics.mean(ref_recalls), 4) if ref_recalls else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('runs', nargs='+')
    ap.add_argument('--data', required=True)
    ap.add_argument('--json', default=None)
    a = ap.parse_args()
    data = load_data(a.data)
    out = []
    print('%-52s %7s %7s %6s %8s' % ('run', 'recall', 'chance', 'lift', 'ceiling'))
    print('-' * 86)
    for g in a.runs:
        s = score(g, data)
        out.append(s)
        label = g.replace('/eval/generations.jsonl', '').replace('sft_runs/', '')
        print('%-52s %7s %7s %6s %8s'
              % (label[-52:], s['ref_residue_recall'], s['chance_baseline'],
                 s['lift_over_chance'], s['CEILING_reference_recall']))
    if a.json:
        json.dump(out, open(a.json, 'w'), indent=2)
        print('\nwrote %s' % a.json)


if __name__ == '__main__':
    main()
