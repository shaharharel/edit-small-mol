"""EVALUATE THE MAML ARM THE WAY MAML IS MEANT TO BE EVALUATED: adapt, then score.

THE PROBLEM THIS FIXES. train_phaseB's episodic arm reports `eval_loss(va_rows)` -- the
meta-initialisation's loss with NO inner-loop adaptation. Reptile does not optimise that quantity.
It optimises loss AFTER k gradient steps on a support set. So comparing the logged EPISODE number
(0.5203) against the fine-tuned arms (scratch 0.4730, from_a 0.4389) compares a meta-initialisation
against fully fine-tuned models and will always flatter the latter. That is an apples-to-oranges
comparison of exactly the kind that produced three dead results tonight, and the fix is to run the
adaptation the method assumes.

THE PROTOCOL. For each HELD-OUT protein independently:
    support = k rows from that protein   (adapt on these)
    query   = the remaining rows          (score on these, never adapted on)
Adapt a FRESH COPY of the meta-init on the support set for --inner-steps, then score the query set.
Report per-protein and pooled, and report the SAME split scored WITHOUT adaptation so the gain from
adaptation is visible rather than assumed.

WHY THE BASELINES ARE SCORED ON THE SAME QUERY ROWS. A fine-tuned arm (scratch / from_a) is scored
on the identical query set, so all arms answer "loss on rows of a protein you have k examples of".
Without that, the comparison silently changes which rows are being predicted.

POWER WARNING, STATED UP FRONT because it bounds everything below: the held-out set is 16 proteins
with a brutal size distribution (64, 34, 13, 11, 8, 7, 4, 4, 3, 3, 2, 2, 2, 2, 2, 1) summing to 162.
A protein with n <= k+1 cannot supply both a support set and a query set and is DROPPED -- reported,
not silently skipped. With k=4 that leaves 6 of 16 proteins (37.5%, NOT "roughly half"), and the two
largest supply 79.6% of the query rows, so the pooled number is dominated by two pockets. Query rows
total 113. THAT 113 IS THE DENOMINATOR UNDER #67 AND #69.

THIS PARAGRAPH QUOTED THE CONTAMINATED DISTRIBUTION UNTIL NOW, AND IT IS THE SAME MISS FOUR TIMES
OVER. The old tuple (77, 40, 19, ...) sums to 192 -- the 2-clause de-leak count, i.e. the leaky set.
I fixed the inline table 100 lines below and left this docstring untouched, exactly as I sanitised
`pooled` and left `within_stratum` one field over, guarded role_compliance and skipped
warhead_retention, and moved a script out of /tmp while leaving its output there. The fix and its
unfixed sibling keep shipping together.
NOTE THE DIRECTION: de-leaking made the concentration WORSE, not better -- top-2 share of query rows
went 76.8% -> 79.6%. So the power warning understated the problem while quoting inflated counts.
"""
import os, sys, csv, zlib, copy, json, random, argparse, collections
import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, '/Users/shaharharel/Documents/github/REINVENT4')
HERE_ = os.path.dirname(os.path.abspath(__file__))
import train_phaseB as TB
from train_phaseA import PhaseA, subsequent_mask
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer


def load_arm(path, device='cpu'):
    ck = torch.load(path, map_location='cpu', weights_only=False)
    npm = dict(ck['network_parameter'])
    m = PhaseA(mode=ck.get('mode', 'geom'), **npm)
    m.geom_mlp = nn.Sequential(nn.Linear(3, 64), nn.ReLU(),
                               nn.Linear(64, npm['model_dimension']))
    m.load_state_dict(ck['model_state'])
    m.to(device)
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(_v['tokens'])
    # REPLAY THE GEOMETRY NORMALISATION, or this file silently evaluates a z-scored model on RAW
    # inputs. This is the cond_repeat bug a fourth time: the checkpoint records gzscore/gmu/gsd,
    # load_arm read none of them, and PhaseBData defaults to gstats=None -- so the state_dict loads
    # cleanly and the eval is simply wrong. MEASURED by QA on the real checkpoints: held-out token
    # loss 0.3774 -> 0.8060 (s505) and 0.5075 -> 1.7049 (s808), i.e. inflated by +0.43 to +1.20 nats,
    # which is 43x to 120x the +0.0100 effect being studied. It fails in the direction that makes
    # z-scoring look CATASTROPHIC, so the natural reading of the wrong number is exactly backwards.
    # A loaded gun rather than a fired one: no adapt-eval had been run on a z arm yet.
    gstats = None
    if ck.get('gzscore'):
        if ck.get('gmu') is None or ck.get('gsd') is None:
            raise SystemExit('FATAL: checkpoint says gzscore=True but carries no gmu/gsd. It was '
                             'trained on normalised geometry and cannot be evaluated without the '
                             'statistics. Refusing to score it on raw inputs.')
        gstats = (torch.tensor(ck['gmu'], dtype=torch.float),
                  torch.tensor(ck['gsd'], dtype=torch.float))
        print('  gstats REPLAYED from checkpoint: mu=%s sd=%s'
              % ([round(x, 4) for x in ck['gmu']], [round(x, 4) for x in ck['gsd']]))
    return m, vocab, ck, gstats


def token_loss(model, rows, vocab, tok, ce, bs, device, gstats=None):
    """TOKEN-WEIGHTED, matching the corrected train_phaseB.eval_loss."""
    if not rows:
        return float('nan'), 0
    dl = torch.utils.data.DataLoader(TB.PhaseBData(rows, vocab, tok, gstats=gstats), batch_size=bs,
                                     shuffle=False, collate_fn=TB.collate)
    model.eval()
    num, den = 0.0, 0.0
    with torch.no_grad():
        for b in dl:
            if b is None:
                continue
            src, sm, tgt, g = [x.to(device) for x in b[:4]]
            ti, to = tgt[:, :-1], tgt[:, 1:]
            lg = model.generator(model.forward_cond(src, sm, ti,
                                                    subsequent_mask(ti.size(1), device), g))
            per = ce(lg.reshape(-1, lg.size(-1)), to.reshape(-1)).view(to.shape)
            m = (to != 0).float()
            num += float((per * m).sum())
            den += float(m.sum())
    return num / max(den, 1.0), int(den)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--ckpt', default='experiments/covalentformer/ckpt_B_maml/ep100.ckpt')
    ap.add_argument('--baselines', default='ckpt_B_scratch/ep0.ckpt,ckpt_B_from_a/ep0.ckpt')
    ap.add_argument('--data', default='experiments/covalentformer/data/phaseB/phaseB.csv')
    ap.add_argument('--support', type=int, default=4)
    ap.add_argument('--inner-steps', type=int, default=5)
    ap.add_argument('--inner-lr', type=float, default=1e-5)
    ap.add_argument('--bs', type=int, default=8, help='scoring batch size only')
    # MATCHES train_phaseB --inner-bs (3). Adaptation batch size is now separate from scoring
    # batch size; conflating them is what produced the single-batch inner loop.
    ap.add_argument('--inner-bs', type=int, default=3)
    ap.add_argument('--seed', type=int, default=101)
    ap.add_argument('--device', default='cpu')
    ap.add_argument('--out', default='')
    a = ap.parse_args()
    torch.manual_seed(a.seed)
    rng = random.Random(a.seed)

    # reconstruct the SAME held-out split the arms were trained against
    rows, _fail = TB.build_rows(a.data, rng)
    by_t = collections.defaultdict(list)
    for r in rows:
        by_t[r['target_id']].append(r)
    targets = sorted(by_t)
    rng.shuffle(targets)
    n_val = max(2, int(0.2 * len(targets)))
    val_t, tr_t = set(targets[:n_val]), set(targets[n_val:])
    tr_rows = [r for r in rows if r['target_id'] in tr_t]
    va = [r for r in rows if r['target_id'] in val_t]
    # THIS FILTER WAS TWO CLAUSES WHERE train_phaseB APPLIES FOUR, AND EVERY MAML NUMBER ON THE
    # BOOKS WAS SCORED ON THE LEAKIER SET. train_phaseB:287-290 rejects a held-out row on target,
    # exact keep, ISOTOPE-STRIPPED keep, and parent; this file rejected on only the first two. So
    # the arms were TRAINED against a clean held-out set and EVALUATED against a dirty one.
    # MEASURED side by side on the same build_rows and the same split reconstruction:
    #   seed  maml_va  phaseB_va  EXTRA  inflation
    #   101      192        162     30    +15.62%   <- the seed #54/#67/#69 are quoted on
    #   202      234        225      9     +3.85%
    #   303      106         89     17    +16.04%
    #   404      167        141     26    +15.57%
    #   505      152        131     21    +13.82%
    #   606       73         66      7     +9.59%
    # #67's headline margin is scratch+adapt 0.3560 vs maml+adapt 0.3825 = 0.0265, and #69 is a
    # 5/6 sign test at p=0.22. Neither has room to absorb a 15.6% contaminated denominator.
    # This is the same class as #55/#59 (two estimators straddling a boundary) and it is why the
    # filter now lives in ONE place: train_phaseB.de_leak_holdout, imported rather than retyped.
    # ^ THAT SENTENCE WAS FALSE WHEN WRITTEN. de_leak_holdout did not exist, and the filter was
    # retyped immediately below it. It is true now: the function exists and is imported here.
    #
    # AND THE VERSION IS NO LONGER HARDCODED. This scorer applied v2 unconditionally, including to
    # pre-14:02 checkpoints built under v1 -- scoring 141 rows against a value recorded over 167.
    # The levels that produces are not comparable to ANY recorded val. Severity was narrow (tr_rows
    # is identical across versions, so v2 is more conservative rather than leaky, and #155's paired
    # margin survives) but the numbers were mislabelled. The version now comes off the checkpoint.
    # RESOLVE THE VERSION FROM THE CHECKPOINTS, AND REQUIRE THEM TO AGREE. A mixed-version
    # comparison is the estimator straddle that has now happened five times tonight: each arm
    # would be scored on a DIFFERENT held-out set and the margin between them would be partly the
    # filter. Refuse rather than average over it.
    # RESOLVE EACH PATH ONCE, HERE, AND USE THE SAME RESOLVED STRING EVERYWHERE BELOW.
    # The version guard used to resolve paths with an exists-or-prefix fallback while the scorer
    # unconditionally prefixed the baselines. The guard was therefore MORE permissive than the
    # scorer, which is the wrong polarity: a --ckpt given without the prefix passed the straddle
    # check (fallback found it) and then failed os.path.exists in the scoring loop, where the arm
    # was SKIPPED with a printed MISSING and no error. Since the downstream blocks are all guarded
    # by `if 'maml' in res`, the adaptation-gain section simply would not print and the JSON would
    # be written with baselines only -- an artifact that looks like a completed run which happened
    # to have no maml arm. One resolver, one truth.
    def _resolve(p):
        p = p.strip()
        for cand in (p, os.path.join('experiments/covalentformer', p), os.path.join(HERE_, p)):
            if os.path.exists(cand):
                return cand
        raise SystemExit('FATAL: cannot locate arm checkpoint %r from cwd %s. Refusing to run: a '
                         'missing arm used to be SKIPPED with a printed warning, which produced a '
                         'JSON that looked complete.' % (p, os.getcwd()))
    _arm_paths = [_resolve(a.ckpt)] + [_resolve(p) for p in a.baselines.split(',') if p.strip()]
    _vers = {}
    for _pp in _arm_paths:
        _vers[_pp] = TB.resolve_deleak_version(
            torch.load(_pp, map_location='cpu', weights_only=False))
    if len(set(_vers.values())) > 1:
        raise SystemExit('FATAL: arms straddle de-leak versions %s. They would be scored on '
                         'DIFFERENT held-out sets and the margin would be partly the filter, not '
                         'the model. Re-score under one version instead of comparing across.'
                         % (_vers,))
    deleak_version = next(iter(_vers.values()))
    _n0 = len(va)
    va = TB.de_leak_holdout(tr_rows, va, version=deleak_version)
    print('  de-leak v%d, READ FROM THE CHECKPOINT (not hardcoded): %d -> %d held-out rows '
          '(dropped %d). v1 and v2 produce different held-out SETS, so levels from one are not '
          'comparable to the other.' % (deleak_version, _n0, len(va), _n0 - len(va)), flush=True)
    by_p = collections.defaultdict(list)
    for r in va:
        by_p[r['target_id']].append(r)
    print('held-out: %d rows over %d proteins' % (len(va), len(by_p)), flush=True)

    usable = {p: rs for p, rs in by_p.items() if len(rs) > a.support}
    dropped = {p: len(rs) for p, rs in by_p.items() if len(rs) <= a.support}
    print('  usable (n > support=%d): %d proteins | DROPPED %d: %s'
          % (a.support, len(usable), len(dropped), dict(sorted(dropped.items()))), flush=True)
    if not usable:
        print('  NOTHING USABLE at this support size')
        return 1

    tok = SMILESTokenizer()
    ce = nn.CrossEntropyLoss(ignore_index=0, reduction='none')
    # SAME RESOLVED PATHS the version guard used -- _arm_paths[0] is the maml arm, the rest are
    # the baselines, in order. Re-deriving them here is how the two diverged in the first place.
    _names = ['maml'] + ['base:' + p.strip().split('/')[0].replace('ckpt_B_', '')
                         for p in a.baselines.split(',') if p.strip()]
    arms = list(zip(_names, _arm_paths))
    res = collections.defaultdict(dict)
    for name, path in arms:
        if not os.path.exists(path):
            print('  %-16s MISSING %s' % (name, path), flush=True)
            continue
        model0, vocab, _ck, gstats = load_arm(path, a.device)
        # SEED GUARD. --seed drives BOTH build_rows (which foreign fragment each anchor gets) AND
        # the target shuffle, so it rebuilds a DIFFERENT held-out split: seed 101 -> 192 val rows,
        # 202 -> 234, 303 -> 106. Scoring a _s202 arm under the default seed 101 would make rows
        # that arm TRAINED on into its query rows, and it would look much better. Every checkpoint
        # records its seed; refuse rather than silently mis-score.
        ck_seed = _ck.get('seed')
        if ck_seed is not None and int(ck_seed) != int(a.seed):
            print('  FATAL %s: checkpoint seed=%s but --seed=%d. The held-out split is seed-'
                  'dependent, so this would score the arm on rows it trained on. Refusing.'
                  % (name, ck_seed, a.seed))
            return 2
        # ADAPT EVERY ARM, not just maml. Previously the baselines were scored WITHOUT
        # the support rows while maml got them -- a tilt in maml's favour. The matched
        # question is 'given k examples of a new pocket, which init adapts best', and
        # that requires giving every arm the same k examples.
        for adapt in (False, True):
            tag = name + ('+adapt' if adapt else '')
            num, den = 0.0, 0.0
            per_p = {}
            for p, rs in sorted(usable.items()):
                shuf = rs[:]
                random.Random(a.seed).shuffle(shuf)
                sup, qry = shuf[:a.support], shuf[a.support:]
                m = copy.deepcopy(model0)
                if adapt:
                    # RNG RE-SEEDED PER (ARM, PROTEIN). The model is in train() with dropout=0.1 and
                    # the loader shuffles, both drawing from the global torch RNG. Seeding once
                    # before the arm loop meant arm k adapted under different dropout masks than
                    # arm 1, so ARM ORDER changed the answer -- a real noise floor under a
                    # "which init adapts best" comparison whose effect is ~0.01.
                    # DETERMINISTIC per-protein seed. This line used hash(p), and Python 3
                    # RANDOMISES str hashing per process unless PYTHONHASHSEED is set -- which it
                    # is not here (verified: three interpreters gave 19300 / 48353 / 14323 for the
                    # same protein). So --seed did NOT control the adaptation RNG: within a run all
                    # arms shared a per-protein seed, so the PAIRED contrast was matched, but
                    # RE-RUNNING THE IDENTICAL COMMAND WAS A FRESH DRAW, not a replication. The
                    # dropout/shuffle noise that seeding was added to control is ~0.01, the same
                    # size as the effect. crc32 is stable across processes and versions.
                    torch.manual_seed(a.seed + zlib.crc32(p.encode()) % 100000)
                    # a FRESH copy per protein: adaptation must not leak between pockets
                    opt = torch.optim.Adam(m.parameters(), lr=a.inner_lr)
                    dl = torch.utils.data.DataLoader(
                        TB.PhaseBData(sup, vocab, tok, gstats=gstats),
                        batch_size=min(a.inner_bs, len(sup)), shuffle=True,
                        collate_fn=TB.collate)
                    m.train()
                    # FRESH MINIBATCH PER INNER STEP -- matching train_phaseB's Reptile loop.
                    # The previous form was `for _ in range(inner_steps): for b in dl:`, which with
                    # support=4 and bs=8 yields ONE batch and REUSES IT 5 times. That is precisely
                    # the pathology train_phaseB's own comment says it was fixed to avoid: it
                    # collapses Reptile's cross-minibatch term E[<g_i,g_j>] to the gradient of
                    # ||g||^2 on a single batch, i.e. ordinary joint training. The TRAINER was fixed
                    # and the EVALUATOR was not, so the meta-init was being adapted by the very
                    # procedure it was de-optimised for -- biasing AGAINST maml, the same direction
                    # as the inner-lr mismatch. Third bias in one comparison, all the same way.
                    # It also fixes the step count: the old nesting ran
                    # inner_steps x ceil(support/bs) optimizer steps, so simply matching training's
                    # --support 10 --bs 3 would have QUADRUPLED the budget to 20 and handed maml an
                    # unearned win. This form runs exactly inner_steps steps at any support/bs.
                    it_dl = iter(dl)
                    for _s in range(a.inner_steps):
                        try:
                            b = next(it_dl)
                        except StopIteration:
                            it_dl = iter(dl)
                            b = next(it_dl)
                        if b is None:
                            continue
                        loss = TB.loss_on(m, b, a.device, ce)
                        opt.zero_grad()
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
                        opt.step()
                l, t = token_loss(m, qry, vocab, tok, ce, a.bs, a.device, gstats)
                per_p[p] = (l, len(qry))
                num += l * t
                den += t
            res[tag] = dict(pooled=num / max(den, 1.0), per_protein=per_p)
            print('  %-16s query-set loss %.4f   (%d proteins, %d query rows)'
                  % (tag, num / max(den, 1.0), len(per_p), sum(v[1] for v in per_p.values())),
                  flush=True)

    print()
    if 'maml' in res and 'maml+adapt' in res:
        d = res['maml']['pooled'] - res['maml+adapt']['pooled']
        print('  ADAPTATION GAIN on the meta-init: %+.4f  (%s)'
              % (d, 'adaptation helps' if d > 0 else 'adaptation does NOT help'))
        print('  This is the quantity Reptile optimises. The un-adapted number that train_phaseB')
        print('  logs per episode is NOT what the method is for and should not be compared to a')
        print('  fine-tuned arm.')
    print('  NOTE the power limit: %d of %d held-out proteins were droppable at support=%d, and the'
          % (len(dropped), len(by_p), a.support))
    print('  query rows are dominated by the two largest pockets. Read per-protein, not just pooled.')
    if a.out:
        json.dump({k: {'pooled': v['pooled'],
                       'per_protein': {p: list(x) for p, x in v['per_protein'].items()}}
                   for k, v in res.items()}, open(a.out, 'w'), indent=1)
    return 0


if __name__ == '__main__':
    sys.exit(main())
