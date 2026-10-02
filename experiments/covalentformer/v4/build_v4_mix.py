#!/usr/bin/env python
"""Assemble v4_train.jsonl: all five task types in ONE corpus, re-weighted by sampling budget.

WHY ONE CORPUS. v3 established that the rungs train together rather than sequentially, with the
mixture controlled by per-rung budgets instead of by ordering. v4 keeps that and adds two row types
v3 did not have as separate tasks: pocket-conditioned rung 3 on the fixed frame, and contacts.

WHERE THE AFFINITY LABEL GOES -- MEASURED, NOT ASSUMED. The obvious design is to hang dpIC50 on the
rung-3 rows so one example teaches both the edit and its potency consequence. That is impossible
here and the number says so: of 5,518 rung-3 pairs exactly ONE has a measured dpIC50 (0.0%), and of
79,241 rung-2 pairs, 2,031 do (2.6%). Rung 3 requires BOTH members to have a solved covalent
co-crystal; potency lives in ChEMBL/covindb for compounds that were mostly never crystallised. So:
  - affinity rides as its OWN row type (143,075 rows), which is what v3 did
  - the 2,031 rung-2 rows that DO have a label get it attached, since that is free
  - every other row carries dpic50=None and the head's masked loss gives it weight 0
That last point is the mechanism for "we don't need it on all examples": no fabricated targets.

THE dpIC50 TEXT IS STRIPPED FROM THE RESPONSE. v3's affinity rows write "Predicted dpIC50: +0.92"
into `output`, inside the span a response-pooled head reads -- the leak v3b had to strip at train
time. Stripping it in the DATA means no future trainer can reintroduce it by forgetting to.

RUNG 3 IS REPEATED. At x1 it is 5,518 of ~760k rows = 0.73%, i.e. ~172 optimizer steps, which
cannot teach a pocket-conditioned task. r3_repeat gives it exposure comparable to what a sequential
arm would deliver, the same device v3 used.
"""
from __future__ import annotations
import argparse, collections, json, os, random, re, sys

sys.path.insert(0, os.path.expanduser('~'))
from build_joint import read                                      # noqa: E402
# NOT build_joint.clean / load_eval_analogues -- both are broken. clean() canonicalised r['input']
# and r['output'] as whole SMILES; v4 rows have no bare 'input' and their output is
# "Change: REPLACE\nAnalogue: CC...", which MolFromSmiles rejects, so both lookups returned None and
# the filter kept everything. It printed "79241 rows -> 79241 kept, 0 dropped" every run.
# decon.py extracts the analogue from wherever it actually lives and compares that.
from decon import canon, clean, row_molecules                      # noqa: E402

DPIC_LINE = re.compile(r'^\s*Predicted dpIC50:.*$\n?', re.M)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--v3-train', default=os.path.expanduser('~/ladder_v3/v3_train.jsonl'))
    ap.add_argument('--v4-dir', default=os.path.expanduser('~/v4'))
    ap.add_argument('--ladder-dir', default=os.path.expanduser('~/ladder_v3'))
    ap.add_argument('--out', default=os.path.expanduser('~/v4/v4_train.jsonl'))
    ap.add_argument('--rung1-n', type=int, default=150000)
    ap.add_argument('--r3-repeat', type=int, default=8)
    ap.add_argument('--contacts-n', type=int, default=0, help='cap contacts rows (0 = all)')
    ap.add_argument('--affinity-n', type=int, default=0, help='cap affinity rows (0 = all)')
    ap.add_argument('--seed', type=int, default=20261001)
    a = ap.parse_args()
    rng = random.Random(a.seed)

    # DECONTAMINATE AGAINST THE FRESH rung-3 TEST FOLD, not the old rung3_R3_*_{test,valid,mpro}
    # files. Those were derived from cdl.jsonl, which is also where v4's rung-3 corpus comes from:
    # 99.5% of the old test fold sits in v4's training rows, and filtering against it would leave
    # 1,083 of 5,518. split_rung3.py instead cut a connected-component split of the molecule graph,
    # so no hit and no analogue crosses the boundary. That fold is the thing the rest of the corpus
    # must now be clean of.
    test_rows = read(os.path.join(a.v4_dir, 'rung3_v4_test.jsonl'))
    if not test_rows:
        sys.exit('FATAL: rung3_v4_test.jsonl missing -- run split_rung3.py first')
    evalset = set()
    for r in test_rows:
        evalset |= row_molecules(r)
    eval_pdbs = {r.get('pdb_a') for r in test_rows} | {r.get('pdb_b') for r in test_rows}
    eval_pdbs.discard(None)
    print('rung-3 TEST fold: %d rows -> %d molecules, %d PDB ids to exclude everywhere else'
          % (len(test_rows), len(evalset), len(eval_pdbs)), flush=True)

    # ---- rung1 and affinity come from v3_train, which build_joint already decontaminated ----
    r1, aff = [], []
    for l in open(a.v3_train):
        r = json.loads(l)
        if r.get('dpic50') is not None:
            o = r.get('output') or ''
            n0 = len(o)
            r['output'] = DPIC_LINE.sub('', o)
            r['_stripped'] = int(len(r['output']) != n0)
            r['task'] = 'AFFINITY'
            aff.append(r)
        elif r.get('rung') == 1 or r.get('source') == '1':
            r1.append(r)
    strip_n = sum(x.pop('_stripped') for x in aff)
    print('rung1 pool %d | affinity %d (dpIC50 text stripped from %d responses)'
          % (len(r1), len(aff), strip_n), flush=True)
    if strip_n != len(aff):
        print('  NOTE: %d affinity rows had no dpIC50 line to strip' % (len(aff) - strip_n), flush=True)

    # equal draw per rung-1 task so dedup's warhead collapse does not skew the mixture (v3's rule)
    bytask = collections.defaultdict(list)
    for r in r1:
        bytask[r.get('task', '?')].append(r)
    per = max(1, a.rung1_n // max(len(bytask), 1))
    r1s = []
    for t in sorted(bytask):
        pool = bytask[t]; rng.shuffle(pool)
        r1s.extend(pool[:per])
        print('  rung1 task %-14s available %7d -> taking %6d' % (t, len(pool), min(per, len(pool))))

    # ---- the new v4 corpora ----
    r2 = read(os.path.join(a.v4_dir, 'rung2_mmp.jsonl'))
    r3 = read(os.path.join(a.v4_dir, 'rung3_v4_train.jsonl'))
    con = read(os.path.join(a.v4_dir, 'contacts_v4.jsonl'))
    r2 = clean(r2, evalset, 'rung2')
    r1s = clean(r1s, evalset, 'rung1')
    aff = clean(aff, evalset, 'affinity')
    # contacts carry no SMILES at all (the ligand is coordinates), so a molecule filter cannot see
    # them -- but a contacts row from a TEST complex still shows the model that pocket. Exclude by
    # PDB id, which is the key that actually links them.
    n0 = len(con)
    con = [r for r in con if r.get('pdb') not in eval_pdbs]
    print('  %-10s %7d rows -> %7d kept, %7d DROPPED as test-complex contact (%.2f%%)'
          % ('contacts', n0, len(con), n0 - len(con), 100.0 * (n0 - len(con)) / max(n0, 1)),
          flush=True)
    # CAPS EXIST BECAUSE OF MEASURED COMPUTE, NOT TASTE. Priced on the observed 18.6 s/step, the
    # uncapped atom-mode corpus is 365M tokens = 75 h on one A100-40GB. Pocket rows dominate even
    # after the residue-block rewrite (1,377 tok vs 2,469), so hitting a single overnight run means
    # cutting ROWS as well. Sampling is uniform, so class balance and the contacts baselines
    # measured at build time still describe what is trained on.
    if a.contacts_n and len(con) > a.contacts_n:
        rng.shuffle(con); con = con[:a.contacts_n]
        print('  contacts capped to %d' % len(con), flush=True)
    if a.affinity_n and len(aff) > a.affinity_n:
        rng.shuffle(aff); aff = aff[:a.affinity_n]
        print('  affinity capped to %d' % len(aff), flush=True)
    # rung 3 IS the supervision for the pocket task and is never filtered against itself.
    print('  %-10s %7d rows (supervision, not filtered)' % ('rung3', len(r3)), flush=True)
    # contacts have no analogue to leak -- the target is an interaction type, so the analogue-based
    # filter does not apply. Assert that rather than silently skipping the check.
    assert not any('analogue' in r for r in con[:2000]), 'contacts rows unexpectedly carry an analogue'
    print('  %-10s %7d rows (no analogue field -> analogue leak filter N/A)' % ('contacts', len(con)),
          flush=True)

    # ---- attach the 2,031 measurable dpic50 labels onto rung-2 rows ----
    key2y = {}
    for r in aff:
        h = r.get('input')
        m = re.search(r'Analogue:\s*(\S+)', r.get('output') or '')
        if h and m:
            ch, ca = canon(h), canon(m.group(1))
            if ch and ca:
                key2y[(ch, ca)] = r['dpic50']
    tagged = 0
    for r in r2:
        y = key2y.get((canon(r.get('hit')), canon(r.get('analogue'))))
        if y is not None:
            r['dpic50'] = y; tagged += 1
    print('  rung2 rows given a measured dpic50: %d (%.2f%%)'
          % (tagged, 100.0 * tagged / max(len(r2), 1)), flush=True)

    for rows, rg in ((r1s, 1), (r2, 2), (con, 'contacts'), (aff, 'affinity')):
        for r in rows:
            r['rung'] = rg
    r3s = []
    for _ in range(a.r3_repeat):
        for r in r3:
            q = dict(r); q['rung'] = 3; r3s.append(q)

    mix = r1s + r2 + r3s + con + aff
    rng.shuffle(mix)

    # ---- QA GATE: nothing is written unless this passes ----
    fails = collections.Counter()
    for r in mix:
        if not r.get('instruction'):
            fails['missing_instruction'] += 1
        if not r.get('output'):
            fails['missing_output'] += 1
    # the defect that would have crashed training: rung3 shipped with no output field at all
    n_no_out = sum(1 for r in r3s if not r.get('output'))
    if n_no_out:
        fails['rung3_missing_output'] += n_no_out
    # the leak: no response may still state its own regression target
    for r in mix:
        if r.get('dpic50') is not None and 'dpIC50' in (r.get('output') or ''):
            fails['dpic50_STILL_IN_RESPONSE'] += 1
    ny = sum(1 for r in mix if r.get('dpic50') is not None)
    if ny < 1000:
        fails['too_few_affinity_labels_%d' % ny] += 1
    print('\n=== QA GATE ===')
    c = collections.Counter(str(r.get('rung')) for r in mix)
    for k in sorted(c):
        print('  rung %-10s %7d  (%.1f%%)' % (k, c[k], 100.0 * c[k] / len(mix)))
    print('  rows with a dpic50 label: %d (%.1f%%)' % (ny, 100.0 * ny / len(mix)))
    for k, v in fails.most_common():
        print('  FAIL %-34s %d' % (k, v))
    if fails:
        sys.exit('QA FAILED -- nothing written')
    print('  QA PASSED')

    with open(a.out, 'w') as fh:
        for r in mix:
            fh.write(json.dumps(r) + '\n')
    eff = 32
    steps = len(mix) // eff
    print('\nV4 MIX -> %s' % a.out)
    print('  TOTAL %d rows -> %d optimizer steps at effective batch %d' % (len(mix), steps, eff))
    print('  rung-3 gradient steps: %d (%d unique x%d)' % (len(r3s) // eff, len(r3), a.r3_repeat))
    print('  at the MEASURED 2.39 s/step this is %.1f h for one epoch' % (steps * 2.39 / 3600.0))
    print('V4_MIX_DONE')
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
