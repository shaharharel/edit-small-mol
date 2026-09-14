"""Replicate train_phaseB.py's row build + target split + de-leak filter at all six seeds.

WAS THE FOURTH INDEPENDENT COPY of the de-leak filter, and it retyped build_rows as well -- so it
could have diverged from train_phaseB in either the SPLIT or the FILTER and still printed a
plausible table. It did not diverge (the retyped build_rows consumed the rng in the same order and
defaulted `parent` the same way), but that was luck, not design, and nothing here would have
caught it. Both now come from train_phaseB. The original reason for retyping -- 'No torch' -- was
real but bought nothing: importing torch costs a few seconds and removes a silent-divergence
surface from the one script whose entire job is to certify that the filter behaves as claimed."""
import sys,os,random,collections
CF='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer'
sys.path.insert(0,CF)
from rdkit import RDLogger
RDLogger.DisableLog('rdApp.*')
from train_phaseB import build_rows, de_leak_holdout, _strip_iso as strip_iso, _canon_iso
DATA=CF+'/data/phaseB/phaseB.csv'

print('%-6s %5s %5s | %7s %7s %7s | %8s %8s %8s %8s'%(
  'seed','rows','n0','PRE','POST','delta','iso_extra','par_extra','tgt_only','keepx_only'))
tot=collections.Counter(); per_seed={}
for seed in [404,505,606,707,808,909]:
    rng=random.Random(seed)
    rows,fail=build_rows(DATA,rng)
    by_t=collections.defaultdict(list)
    for r in rows: by_t[r['target_id']].append(r)
    targets=sorted(by_t); rng.shuffle(targets)
    n_val=max(2,int(0.2*len(targets)))
    val_t,tr_t=set(targets[:n_val]),set(targets[n_val:])
    tr=[r for r in rows if r['target_id'] in tr_t]
    va=[r for r in rows if r['target_id'] in val_t]
    tr_tgt={r['target'] for r in tr}; tr_keep={r.get('keep','') for r in tr}
    tr_keep_iso={strip_iso(r.get('keep','')) for r in tr}
    tr_parent={r.get('parent','') for r in tr if r.get('parent')}
    tr_regrow_iso={_canon_iso(r['regrow']) for r in tr if r.get('regrow')}
    n0=len(va)
    def pre(r):  return r['target'] not in tr_tgt and r.get('keep','') not in tr_keep
    def post(r): return pre(r) and strip_iso(r.get('keep','')) not in tr_keep_iso \
                        and (not r.get('parent') or r.get('parent') not in tr_parent)
    n_pre=n0-sum(pre(r) for r in va); n_post=n0-sum(post(r) for r in va)
    # The local pre/post predicates survive ONLY to decompose the drop by cause. The authoritative
    # counts come from the canonical filter, and disagreeing with it is fatal -- this script now
    # CHECKS train_phaseB rather than paraphrasing it.
    # COMPARE ROW IDENTITY, NOT COUNTS. Counts agreeing is a weak check: a filter that dropped a
    # DIFFERENT 67 rows would pass it silently, and 67-vs-67 is exactly the kind of agreement that
    # reads as verification. Compare the surviving rows themselves, in order.
    def _key(r): return (r['target'], r.get('keep',''), r.get('parent',''), r['anchor'])
    _v1=[_key(r) for r in de_leak_holdout(tr,va,version=1)]
    _v2=[_key(r) for r in de_leak_holdout(tr,va,version=2)]
    _lp=[_key(r) for r in va if pre(r)]
    _lq=[_key(r) for r in va if post(r)]
    # v3 IS CERTIFIED TOO. This script's own docstring says its job is "to certify that the filter
    # behaves as claimed", and it was checking v1 and v2 while DELEAK_VERSION was 3 -- the version
    # actually trained and scored with was the one version it did not check. Demonstrated by an
    # auditor, not argued: a corruption that drops a row ONLY when version>=3 left v1/v2 untouched
    # and this script printed OK at every seed. The v3 clause landed six minutes AFTER this
    # certifier was last edited, and the certifier was not extended with it.
    def _post3(r):
        return post(r) and (not r.get('regrow')
                            or _canon_iso(r['regrow']) not in tr_regrow_iso)
    _v3=[_key(r) for r in de_leak_holdout(tr,va,version=3)]
    _lr=[_key(r) for r in va if _post3(r)]
    if _v3!=_lr:
        raise SystemExit('FATAL seed %d: v3 rows kept by train_phaseB.de_leak_holdout (%d) are not '
                         'the rows this script models (%d). The regrow clause has diverged.'
                         %(seed,len(_v3),len(_lr)))
    if _v1!=_lp or _v2!=_lq:
        raise SystemExit('FATAL seed %d: the rows kept by train_phaseB.de_leak_holdout are not the '
                         'rows this script models (v1 %d vs %d, v2 %d vs %d; identity mismatch even '
                         'if the counts agree). The filter being certified is not the filter being '
                         'reimplemented.'%(seed,len(_v1),len(_lp),len(_v2),len(_lq)))
    # decomposition on the RAW held-out set
    hit_t   =[r for r in va if r['target'] in tr_tgt]
    hit_k   =[r for r in va if r.get('keep','') in tr_keep]
    hit_i   =[r for r in va if strip_iso(r.get('keep','')) in tr_keep_iso]
    hit_p   =[r for r in va if r.get('parent') and r.get('parent') in tr_parent]
    iso_extra=sum(1 for r in va if pre(r) and strip_iso(r.get('keep','')) in tr_keep_iso)
    par_extra=sum(1 for r in va if pre(r) and strip_iso(r.get('keep','')) not in tr_keep_iso
                  and r.get('parent') and r.get('parent') in tr_parent)
    print('%-6d %5d %5d | %7d %7d %7d | %8d %8d %8d %8d'%(
        seed,len(rows),n0,n_pre,n_post,n_post-n_pre,iso_extra,par_extra,len(hit_t),len(hit_k)))
    tot['n0']+=n0; tot['pre']+=n_pre; tot['post']+=n_post; tot['iso_extra']+=iso_extra
    tot['par_extra']+=par_extra; tot['iso_raw']+=len(hit_i)
    per_seed[seed]=dict(pre=n_pre,post=n_post,iso_extra=iso_extra,par_extra=par_extra)
print('\nPOOLED: n0=%d  pre-fix drops=%d  post-fix drops=%d  iso_extra=%d  par_extra=%d'%(
    tot['n0'],tot['pre'],tot['post'],tot['iso_extra'],tot['par_extra']))
print('  raw iso-equivalent hits (ignoring whether already dropped) pooled = %d'%tot['iso_raw'])
print('\nARITHMETIC CHECK  pre + iso_extra + par_extra == post ?')
# THE FILE USED TO END ON THAT QUESTION MARK. It printed a check HEADER and never performed the
# check -- and a reader seeing "ARITHMETIC CHECK ... ?" as the last line of output reads it as a
# check that ran and passed. That is the hardcoded-conclusion failure in a new shape: not a stated
# conclusion, but an implied one, with nothing underneath it that could fail.
_bad = [(s, v) for s, v in per_seed.items() if v['pre'] + v['iso_extra'] + v['par_extra'] != v['post']]
for s, v in sorted(per_seed.items()):
    print('  seed %-4d %3d + %3d + %3d == %3d  %s'
          % (s, v['pre'], v['iso_extra'], v['par_extra'], v['post'],
             'OK' if v['pre'] + v['iso_extra'] + v['par_extra'] == v['post'] else 'FAIL'))
_pool_ok = tot['pre'] + tot['iso_extra'] + tot['par_extra'] == tot['post']
print('  POOLED   %3d + %3d + %3d == %3d  %s'
      % (tot['pre'], tot['iso_extra'], tot['par_extra'], tot['post'], 'OK' if _pool_ok else 'FAIL'))
if _bad or not _pool_ok:
    raise SystemExit(
        'FATAL: this script\'s LOCAL decomposition is internally inconsistent at seeds %s. '
        'READ WHAT THIS CAN AND CANNOT CATCH: n_pre, iso_extra and par_extra are ALL computed '
        'from the local pre()/post() predicates over the same rows, and they form an exact '
        'partition by construction -- so this is a check on this script\'s own bookkeeping, NOT '
        'on de_leak_holdout. It CANNOT detect a new clause in the filter, and it did not: it '
        'printed OK at every seed while v3 was added. The ROW-IDENTITY checks above are the ones '
        'with reach. This message previously claimed "the filter gained a clause this script does '
        'not model" as one of its two causes -- it cannot observe that, and a clause did land '
        'while it said OK.' % [s for s, _ in _bad])
# par_extra is 0 at all six seeds: the `parent` clause currently drops nothing anywhere, so it is
# UNTESTED BY LIVE DATA and a bug in it would be invisible here. Say so rather than let six OKs
# imply four clauses were exercised when only three were.
if tot['par_extra'] == 0:
    print('  NOTE: parent clause dropped 0 rows at every seed -- it is UNEXERCISED, not verified.')
print('  NOTE: the v3 regrow clause IS row-identity certified above, but is NOT decomposed in this '
      'table -- the PRE/POST columns are v1/v2 quantities and do not include it.')
