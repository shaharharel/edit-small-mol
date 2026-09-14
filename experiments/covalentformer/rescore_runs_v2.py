"""Put Phase A run 2 and run 3 on ONE de-leak filter so their transfer effects can be compared.

WHY THIS IS A RE-SCORE AND NOT A RETRAIN. de_leak_holdout touches ONLY the held-out rows; tr_rows
is untouched, so an arm trained under v1 has BIT-IDENTICAL weights to the same arm trained under
v2 (verified: max|diff| = 0.000e+00 over 266 tensors). The filter version changes what you SCORE
on, nothing else. I got this wrong once and burned ~45 min x 3 retraining arms that were already
correct; the fix for a version straddle is always a forward pass.

WHY IT IS NEEDED. run 2's arms (ckpt_Bconv_*, init b65ef670) were scored under v1 = 167 held-out
rows at s404. run 3's arms (ckpt_B4c_*, init 701db7aa) are v2 = 141 rows. Comparing +0.045 from
run 3 against the filed +0.027..+0.033 from run 2 would be a cross-boundary comparison of exactly
the kind that has already produced four wrong numbers tonight -- the two are not on the same
held-out set, so the DIFFERENCE between them is partly the filter.

POSITIVE CONTROL, and the reason to trust any of this: the B4c arms were TRAINED and scored under
v2 already, so re-scoring them here must reproduce their recorded history val. If it does not, the
rescorer is wrong and every number below is void. That check runs first and is fatal.
"""
import sys, os, json
CF = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(CF, 'nulls'))
import torch
from replay import replay, load_model, eval_loss

SEEDS = (404, 505, 606)
EPOCHS = (0, 1, 2)
RUNS = {
    'run2 (Bconv, init b65ef670, trained under v1)': ('ckpt_Bconv_from_a_s%d', 'ckpt_Bconv_scratch_s%d'),
    'run3 (B4c,   init 701db7aa, trained under v2)': ('ckpt_B4c_from_a_s%d',   'ckpt_B4c_scratch_s%d'),
}

_cache = {}
def split(seed):
    if seed not in _cache:
        _cache[seed] = replay(seed, deleak_version=2)
    return _cache[seed]

def score(path, seed):
    d = split(seed)
    m, ck, vocab, tok = load_model(os.path.join(CF, path))
    # THIS FILE HARDCODES version=2, WHICH de_leak_holdout's OWN DOCSTRING FORBIDS ("`version` MUST
    # come from the checkpoint's own stamp, never be hardcoded by the caller"). It is deliberate --
    # the whole point is to put run2 and run3 on ONE filter -- but a hardcode that happens to be
    # right is indistinguishable on disk from one that is wrong, and four wrong numbers tonight came
    # from filter straddles. So it now ASSERTS the thing it is assuming. If a v3-stamped checkpoint
    # ever reaches this script, it refuses instead of silently scoring it under v2.
    _v = int(ck.get('deleak_version', 1) or 1)
    if _v != 2:
        raise SystemExit('FATAL %s is stamped deleak_version=%d but this script scores everything '
                         'under v2 by design. Re-score under the stamped version, or use '
                         'rescore_runs_v3.py which compares filters deliberately.' % (path, _v))
    # gstats=None: every arm here has gzscore=False. Passing stats to a raw-trained model is the
    # cond_repeat bug in its fourth costume -- it loads clean and scores wrong.
    if ck.get('gzscore'):
        raise SystemExit('FATAL %s is z-scored; this rescorer assumes raw arms' % path)
    return eval_loss(m, d['va'], vocab, tok, gstats=None), ck

print('POSITIVE CONTROL: re-score the B4c arms, which were ALREADY scored under v2.')
print('Their recorded history val must come back. If not, the rescorer is void.\n')
ok = True
for s in SEEDS:
    for arm in ('ckpt_B4c_scratch_s%d', 'ckpt_B4c_from_a_s%d'):
        p = (arm % s) + '/ep0.ckpt'
        got, ck = score(p, s)
        rec = ck['history'][-1]['val']
        agree = abs(got - rec) < 1e-6
        ok &= agree
        print('  %-34s recorded %.8f  rescored %.8f  match=%s' % (arm % s, rec, got, agree))
if not ok:
    raise SystemExit('\nFATAL: positive control failed. The rescorer does not reproduce values '
                     'that were computed by train_phaseB itself under the same filter. Every '
                     'number this script would print is void.')
print('\nPOSITIVE CONTROL PASSED -- rescorer reproduces train_phaseB exactly.\n')

out = {}
for label, (fa, sc) in RUNS.items():
    print('=== %s -- ALL RE-SCORED UNDER v2 ===' % label)
    print('seed  ep   scratch     from_a      delta')
    per_seed_ep0, per_seed_argmin = [], []
    for s in SEEDS:
        cs, cf = [], []
        for ep in EPOCHS:
            a, _ = score((sc % s) + '/ep%d.ckpt' % ep, s)
            b, _ = score((fa % s) + '/ep%d.ckpt' % ep, s)
            cs.append(a); cf.append(b)
            print('%-5d %d    %.6f    %.6f    %+.6f' % (s, ep, a, b, a - b))
        per_seed_ep0.append(cs[0] - cf[0])
        per_seed_argmin.append(min(cs) - min(cf))
    import statistics as st
    out[label] = dict(ep0=per_seed_ep0, argmin=per_seed_argmin,
                      mean_ep0=st.mean(per_seed_ep0), mean_argmin=st.mean(per_seed_argmin))
    print('  mean ep0 delta    %+.6f   (all positive: %s)' % (st.mean(per_seed_ep0), all(x > 0 for x in per_seed_ep0)))
    print('  mean argmin delta %+.6f   (all positive: %s)\n' % (st.mean(per_seed_argmin), all(x > 0 for x in per_seed_argmin)))

print('=== RUN-LEVEL CONTRAST, one filter, matched seeds ===')
ks = list(RUNS)
a, b = out[ks[0]], out[ks[1]]
print('  %s  ep0 %+.6f  argmin %+.6f' % (ks[0], a['mean_ep0'], a['mean_argmin']))
print('  %s  ep0 %+.6f  argmin %+.6f' % (ks[1], b['mean_ep0'], b['mean_argmin']))
print('  run3 - run2 (ep0):    %+.6f' % (b['mean_ep0'] - a['mean_ep0']))
print('  paired per-seed diffs:', ['%+.6f' % (y - x) for x, y in zip(a['ep0'], b['ep0'])])
print('\n  n=2 Phase A runs. Per #111 a difference-of-differences at this n is not resolvable;')
print('  this is reported as a MAGNITUDE, not a significance claim.')
json.dump(out, open(os.path.join(CF, 'results', 'rescore_runs_v2.json'), 'w'), indent=1)
print('\nwritten -> results/rescore_runs_v2.json')
