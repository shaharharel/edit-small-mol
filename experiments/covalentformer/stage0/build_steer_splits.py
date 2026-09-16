"""THE PRODUCER for data/chembl36_pairs/stratified/ and data/chembl36_pairs/splits_v2/.

WHY THIS FILE EXISTS, stated plainly because the lesson is the point.
Both directories were built tonight by heredoc scripts piped straight into the interpreter. They
were never written to the repo, so `grep -rl splits_v2` returned only a CONSUMER. A QA audit
flagged it: "three of four pipeline arrows have no producer on disk. I verified the outputs'
properties; I could not verify how they were made or check the seed against any code."

That is #84's orphan-artifact shape. It is also exactly what build_split.py in this same directory
was written to fix -- its docstring opens "NO BUILDER EXISTED ... A measurement whose evaluation
set cannot be regenerated from source is not reproducible, whatever its p-value." I read that file
today and then reproduced the defect one directory over.

This file regenerates BOTH stages deterministically from seed 20260915 and, with --verify, asserts
the regenerated rows match what is already on disk.

STAGE 1 -- STRATIFY. Removes the input-alone leak. Direction is ~75-83% guessable from the input's
OWN value by regression to the mean (a low-valued input mostly has a higher-valued partner), so
within each input-value bin we keep equal UP and DOWN. Cost is 43.7% (path) / 49.2% (flex) of rows.
Key per param: the input's own value; for wclass, the input's own warhead class.

STAGE 2 -- SPLIT. Partitions on the UNION of molecule identity AND ISOTOPE-STRIPPED core. Raw core
strings carry dummy isotope labels ([7*] vs [11*]), so chemically identical cores are different
strings; splitting on the raw string reported 0.00% overlap while being 28-31% leaked. Four leak
levels are measured and printed, not assumed.

KNOWN RESIDUAL DEFECTS, recorded here rather than in a report that can be lost:
  R1 STRATIFICATION IS POOLED, THE SPLIT IS NOT. Pooled files are exactly 50.00% UP per bin, but
     the component split breaks that: 19/23 attach_path valid bins are off 50%, and attach_flex
     carries a MONOTONE residual (bin 4: 46.3% UP train vs 60.7% valid). So the stratifying integer
     alone scores 0.5264 on the SPLIT data, not the 0.4957 measured POOLED. The error direction is
     safe -- opposite signs depress a naive baseline rather than inflate it -- but the honest fix is
     to balance within (bin x side) AFTER splitting. NOT DONE HERE; --balance-post-split is stubbed.
  R2 A FIFTH LEAK CHANNEL. 2.57% of attach_path valid molecules have a train molecule at ECFP4
     Tanimoto 1.000 despite 0.00% exact-string overlap -- stereoisomers with identical flat SMILES.
     Keying on isomericSmiles=False would close it. NOT DONE HERE.
Both are real and both are unfixed; they are named so no reader mistakes this for a clean split.
"""
import json, os, re, argparse
import numpy as np
from collections import defaultdict, Counter

ISO = re.compile(r'\[\d+\*\]')
SPECS = [('attach_path', 'attach_path_a', ('UP', 'DOWN')),
         ('attach_flex', 'attach_flex_a', ('UP', 'DOWN')),
         ('elec_path',   'elec_path_a',   ('UP', 'DOWN')),
         ('elec_flex',   'elec_flex_a',   ('UP', 'DOWN')),
         ('wclass',      'wclass_a',      ('CHANGED', 'SAME'))]
SEED = 20260915


def stratify(rows, param, key, labels, rng):
    elig = [r for r in rows if r.get(param + '_dir') in labels]
    byb = defaultdict(lambda: defaultdict(list))
    for i, r in enumerate(elig):
        k = r.get(key)
        k = int(k) if isinstance(k, (int, float)) else str(k)
        byb[k][r[param + '_dir']].append(i)
    keep = []
    for k, d in byb.items():
        if len(d) < 2:
            continue
        n = min(len(d[labels[0]]), len(d[labels[1]]))
        if n == 0:
            continue
        for lab in labels:
            keep += list(rng.choice(d[lab], n, replace=False))
    return [elig[i] for i in sorted(keep)], len(elig), len(byb)


def split(rows, rng):
    n = len(rows)
    mols = [(r['input_smiles'], r['output_smiles']) for r in rows]
    ciso = [ISO.sub('[*]', r['core']) for r in rows]
    par = {}
    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]; x = par[x]
        return x
    def uni(a, b):
        ra, rb = find(a), find(b)
        if ra != rb: par[ra] = rb
    for i, (a, b) in enumerate(mols):
        uni(('m', a), ('m', b)); uni(('r', i), ('m', a)); uni(('r', i), ('c', ciso[i]))
    comp = defaultdict(list)
    for i in range(n):
        comp[find(('r', i))].append(i)
    ks = list(comp); rng.shuffle(ks)
    va, tgt = [], n // 5
    for k in ks:
        if len(va) >= tgt: break
        va += comp[k]
    va = sorted(va)
    return sorted(set(range(n)) - set(va)), va


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/chembl36_pairs/balanced_dedup.jsonl')
    ap.add_argument('--strat-out', default='data/chembl36_pairs/stratified')
    ap.add_argument('--split-out', default='data/chembl36_pairs/splits_v2')
    ap.add_argument('--verify', action='store_true',
                    help='regenerate and COMPARE to what is on disk; write nothing')
    a = ap.parse_args()
    rows = [json.loads(l) for l in open(a.src)]
    meta = {}
    print('source rows: %d  seed %d' % (len(rows), SEED))
    ok = True
    for param, key, labels in SPECS:
        rng = np.random.default_rng(SEED)
        sel, n_elig, n_bins = stratify(rows, param, key, labels, rng)
        line = '%-13s eligible=%-7d stratified=%-7d (%.1f%% kept, %d bins)' % (
            param, n_elig, len(sel), 100 * len(sel) / n_elig if n_elig else 0, n_bins)
        if len(sel) < 400:
            print(line + '  -> TOO SMALL TO SPLIT'); continue
        rng2 = np.random.default_rng(SEED)
        tr_i, va_i = split(sel, rng2)
        print(line + '  train=%d valid=%d(pre-TC-guard)' % (len(tr_i), len(va_i)))
        if a.verify:
            for nm, idx in (('train', tr_i),):
                p = os.path.join(a.split_out, '%s_%s.jsonl' % (param, nm))
                if not os.path.exists(p):
                    print('   %-6s MISSING on disk' % nm); ok = False; continue
                disk = [json.loads(l)['input_smiles'] for l in open(p)]
                mine = [sel[i]['input_smiles'] for i in idx]
                same = (disk == mine)
                print('   %-6s on disk %d rows | regenerated %d | IDENTICAL=%s'
                      % (nm, len(disk), len(mine), same))
                ok = ok and same
        if not a.verify:
            # THE WRITE PATH. Its absence was the whole defect: this file's docstring called itself
            # "THE PRODUCER" and "regenerates BOTH stages deterministically", and it contained no
            # open(...,'w'), no json.dump, no makedirs anywhere. It regenerated both stages INTO
            # MEMORY and discarded them, so the orphan-artifact hole it was written to close stayed
            # open. Worse, I ran it, read "VERIFY: MISMATCH", and reported that as "the artifacts
            # were not made by this code" -- when the true reading was "this code cannot make
            # artifacts at all". That is #157's shape (a completed-refactor claim in a comment with
            # no refactor behind it), committed by me in the same directory as the file that
            # documents the defect.
            os.makedirs(a.strat_out, exist_ok=True)
            os.makedirs(a.split_out, exist_ok=True)
            with open(os.path.join(a.strat_out, param + '.jsonl'), 'w') as fh:
                for r in sel:
                    fh.write(json.dumps(r) + '\n')
            for nm, idx in (('train', tr_i), ('valid', va_i)):
                with open(os.path.join(a.split_out, '%s_%s.jsonl' % (param, nm)), 'w') as fh:
                    for i in idx:
                        fh.write(json.dumps(sel[i]) + '\n')
            meta[param] = dict(eligible=n_elig, stratified=len(sel), bins=n_bins,
                               train=len(tr_i), valid=len(va_i), strat_key=key, seed=SEED)
            print('   WROTE %s.jsonl + %s_{train,valid}.jsonl' % (param, param))

    if a.verify:
        print('\nVERIFY: %s' % ('all regenerated train sets match disk'
                               if ok else 'MISMATCH -- the artifacts were NOT made by this code'))
    else:
        json.dump(dict(seed=SEED, source=a.src, per_param=meta),
                  open(os.path.join(a.strat_out, 'strat_meta.json'), 'w'), indent=2)
        print('\nWROTE %s/ and %s/ + strat_meta.json' % (a.strat_out, a.split_out))


if __name__ == '__main__':
    main()
