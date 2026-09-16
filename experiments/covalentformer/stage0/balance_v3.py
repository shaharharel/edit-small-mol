"""Balance + cross-target dedup for the v3 corpus.

WRITTEN TO A FILE, NOT PIPED AS A HEREDOC. The first attempt was `nohup python - <<'PY' &`, whose
stdin is the heredoc; when the parent shell hit its timeout the child died with it (exit 143) and
nothing was written. That is also the orphan-artifact pattern one level down -- a producer that
exists only inside a shell invocation cannot be re-run, which is the defect this whole rebuild
exists to escape. Every stage of the v3 chain now lives in the repo.

SOURCE: data/chembl36_pairs_v3/v3.jsonl -- md5 a0b3363d5030580d9a39b0f49519d394, verified
BYTE-IDENTICAL across two independent extractions (extract_v3 and extract_v3b, same code, same
seed). This is the first corpus tonight for which that is true.
"""
import json, random, os
from collections import defaultdict, Counter

SRC = 'data/chembl36_pairs_v3/v3.jsonl'
OUT = 'data/chembl36_pairs_v3/balanced_dedup.jsonl'
PER_CORE, PER_TARGET, SEED = 8, 40000, 20260915

rng = random.Random(SEED)
by_core = defaultdict(list)
n_raw = 0
for line in open(SRC):
    r = json.loads(line); n_raw += 1
    by_core[(r['target'], r['core'])].append(r)
print('raw pairs              : %d' % n_raw, flush=True)
print('distinct (target,core) : %d' % len(by_core), flush=True)

kept = defaultdict(list); dropped_core = 0
for (tgt, core), rows in by_core.items():
    if len(rows) > PER_CORE:
        rows = sorted(rows, key=lambda r: -r.get('tc', 0))
        dropped_core += len(rows) - PER_CORE
        rows = rows[:PER_CORE]
    kept[tgt].extend(rows)
dropped_tgt = 0; final = []
for tgt, rows in kept.items():
    if len(rows) > PER_TARGET:
        rng.shuffle(rows); dropped_tgt += len(rows) - PER_TARGET; rows = rows[:PER_TARGET]
    final.extend(rows)
print('\n=== DROPS, EXPLICIT -- COUNT *AND* COMPOSITION ===', flush=True)
print('per-core cap %d   : %d (%.1f%%)' % (PER_CORE, dropped_core, 100*dropped_core/n_raw), flush=True)
print('per-target cap    : %d (%.1f%%)' % (dropped_tgt, 100*dropped_tgt/n_raw), flush=True)
# THE BANNER USED TO READ "no silent caps" AND PRINT ONLY THE COUNT. The count is not where the
# bias is. This cap keeps the top-8 pairs per (target,core) BY TANIMOTO, and similar pairs change
# the parameter LESS, so the cap SELECTS ON THE LABEL:
#     attach_path SAME  63.61% in the population -> 71.72% in the kept set  (+8.11 points)
#     attach_flex       +10.32 points        wclass  +8.02 points
#     mean tc kept 0.7474 vs dropped 0.6423
# That lands directly on a headline: build_steer_v4.py's rationale opens with "DISCARDED as SAME
# 71.8%", which IS the kept-set figure. The population figure is 64.95%, so the justification for
# keeping SAME rows is overstated by ~6.8 points. A banner that denies silent caps while hiding
# the composition is worse than no banner. Composition is now printed for every param.
print('\n--- COMPOSITION SHIFT INDUCED BY THE CAP (this is where the bias lives) ---', flush=True)
_kept_ids = {id(r) for r in final}
for _col in ('attach_path_dir', 'attach_flex_dir', 'wclass_dir'):
    _pop = Counter(); _kep = Counter()
    for _rows in by_core.values():
        for _r in _rows:
            _v = _r.get(_col)
            if _v:
                _pop[_v] += 1
                if id(_r) in _kept_ids:
                    _kep[_v] += 1
    _pt, _kt = sum(_pop.values()), sum(_kep.values())
    if _pt and _kt:
        _ps, _ks = 100*_pop.get('SAME', 0)/_pt, 100*_kep.get('SAME', 0)/_kt
        print('  %-18s SAME  population %5.2f%%  ->  kept %5.2f%%   (%+.2f points)'
              % (_col, _ps, _ks, _ks - _ps), flush=True)

# cross-target dedup -- the 32.18% defect found in v1
# TIE-BREAK, REWRITTEN. QA-56 proved the original was dead code AND severely biased.
# The old rule was `(sc, target) > (best_sc, best_target)` where sc counts `*_dir` keys. But every
# `*_dir` label is a pure function of (input_smiles, output_smiles) -- which IS the dedup key -- so
# sc is IDENTICAL for every row in a collision group and can never break a tie. Measured over the
# full 7,360,348 rows: 2,161,694 collisions, 0 (0.0%) decided by sc, 2,161,694 (100.0%) decided by
# `max(target)` AS A LEXICOGRAPHIC STRING.
# That is a TOTAL ORDER on target names, so one target loses every contested row to another:
#     CHEMBL2835  111,875 -> 2,823  ( 2.52% survive)   <- 2nd-largest target, gutted
#     CHEMBL5251   93,094 -> 14,079 (15.12% survive)   <- 6.0x better, purely because '2' < '5'
#     range across the top 50 targets: 0.83% .. 17.66% = 21x
# Target coverage and scaffold-diversity claims about this corpus were being decided by a string
# comparison, and `core` -- the union-find key for the leak partition -- was picked the same way.
# THE FIX: break ties on a SEEDED HASH of the pair key. Still fully deterministic (same seed ->
# same corpus, which the byte-identical reproducibility test depends on), but the winner is
# independent of target NAME, so no target is systematically favoured.
# NOT RETROACTIVE: data/chembl36_pairs_v3/ was built under the OLD rule and the arms training right
# now use it. Those numbers carry the bias. This takes effect on the next build.
import hashlib
best = {}; order = []
for r in final:
    a, b = r['input_smiles'], r['output_smiles']
    k = (a, b) if a < b else (b, a)
    tie = hashlib.blake2b(('%d|%s|%s|%s' % (SEED, k[0], k[1], r['target'])).encode(),
                          digest_size=8).hexdigest()
    if k not in best:
        order.append(k)
        best[k] = (tie, r)
    elif tie > best[k][0]:
        best[k] = (tie, r)
rows = [best[k][1] for k in order]
with open(OUT, 'w') as fh:
    for r in rows: fh.write(json.dumps(r) + '\n')
print('kept before dedup : %d' % len(final), flush=True)
print('after dedup       : %d (removed %d, %.2f%%)'
      % (len(rows), len(final)-len(rows), 100*(len(final)-len(rows))/len(final)), flush=True)

print('\n=== STEERABLE YIELD (v3) ===', flush=True)
meta = {}
for col in ('attach_path_dir', 'attach_flex_dir', 'elec_path_dir', 'elec_flex_dir', 'wclass_dir'):
    v = Counter(r.get(col) for r in rows if r.get(col))
    tot = sum(v.values()); st = tot - v.get('SAME', 0)
    if tot:
        print('%-18s labelled=%-8d steerable=%-8d (%.1f%%)' % (col, tot, st, 100*st/tot), flush=True)
        meta[col] = dict(labelled=tot, steerable=st)
json.dump(dict(src=SRC, src_md5='a0b3363d5030580d9a39b0f49519d394', seed=SEED,
               per_core=PER_CORE, per_target=PER_TARGET, n_raw=n_raw,
               dropped_core=dropped_core, dropped_target=dropped_tgt,
               n_kept=len(final), n_dedup=len(rows), yield_=meta),
          open('data/chembl36_pairs_v3/balance_meta.json', 'w'), indent=2)
print('\nwrote %s' % OUT, flush=True)
