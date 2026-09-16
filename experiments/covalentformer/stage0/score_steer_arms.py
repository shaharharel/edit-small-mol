"""The sanctioned readout for the four steering arms, and a launcher that cannot hide a FATAL.

WHY THIS FILE EXISTS. train_phaseA's deterministic all-wrong control is gated on `mode == 'role'`
(train_phaseA.py:409). Every steering arm is mode='geom', so n_w stays 0 and GAP_det is `nan` in
every epoch of every arm -- including on the line the trainer itself annotates "<- quote this one".
That refusal is CORRECT: relabelling to a "wrong role" is meaningless for a continuous channel, and
fabricating a zero would be worse. But it leaves the arms with exactly one non-nan effect size, the
permutation GAP, which train_phaseA's own comments call "DILUTED and BATCH-ORDER-DEPENDENT ... Do
not quote `gap` as an effect size". So as wired, the four-arm comparison has NO sanctioned number
and the arms would be compared on valid loss alone.

THE CONTROL THIS BUILDS, and why negation rather than relabelling. For a continuous steering channel
the meaningful counterfactual is not "a wrong class" but "the opposite instruction": if the model is
using the channel, asking it to make extent SMALLER when the true edit made it LARGER must cost
something. Negation is deterministic (no seed, no batch order), it is defined for every row, and it
is the same magnitude as the true request -- so it cannot be confounded by the conditioning simply
being larger or smaller. That is the property the single-draw controls in this project lacked: a
FIXED seed made every checkpoint inherit the same offset, usable for differences but never levels.

THREE CONTROLS, reported side by side because they fail differently:
  GAP_neg   loss(-c) - loss(c)   DETERMINISTIC. The headline. No seed, no batch-order dependence.
  GAP_perm  loss(c[perm]) - loss(c)  replicated over N seeds, mean and sd reported. Diluted, because
            a permuted value is sometimes close to the true one; strictly a lower bound.
  GAP_zero  loss(0) - loss(c)   ablation: what the channel is worth against having no channel.

WHAT THE `none` ARM MUST REPORT, stated before any number exists so it cannot be narrated later:
its conditioning is constant zero, so -c == c == 0 and GAP_neg is EXACTLY +0.0000 by arithmetic, not
by measurement. Same for GAP_perm and GAP_zero. That is the correct behaviour of a control arm and
it is NOT evidence that "the none arm shows no dependence" -- there is nothing there to depend on.
Any non-zero value in the none row means this script is broken.

INTERPRETATION BOUNDS, IN THE UNITS THIS FILE ACTUALLY COMPUTES.

The previous version of this paragraph quoted "a FLOOR of 40.4% and an ORACLE of 68.2% at tolerance
0.9 A". Both were wrong to put here and one was wrong outright: 68.2 appears nowhere in the repo,
40.4 appears once as a MAJORITY SHARE at deadband 0.25 (not a floor at 0.9), and both are
percentages of a pick-the-variant ACCURACY while this file emits per-token cross-entropy ONLY. So
they could not be compared to a single number produced here. That is the #179 shape -- a bound
quoted in units the producer does not emit.

THE CORRECT CEILING, computed exactly rather than guessed. The readout is teacher-forced per-token
CE over the variant, conditioned on parent + one scalar. Given the parent, the conditioning is a
deterministic function of WHICH variant is being asked for, so the total loss a perfect user of the
channel recovers over a perfect ignorer is exactly the entropy of variant-given-parent: sum log(k)
over rows / total scored tokens. Computed two independent ways (closed form and a prefix-trie
accumulation of -log P(variant|parent)) agreeing to the decimal: 537.0 nats / 36,364 tokens.

  oracle-vs-ignorer ceiling   0.0123 nats/token (smoke3 valid), 0.0148 whole set
  INDEPENDENTLY REPRODUCED: 537.01 nats / 36,364 tokens, 0.014768 whole set, 0.012273 smoke3 valid,
  by a second auditor from a from-scratch tokeniser. score() now RECOMPUTES this per dataset
  (ceiling_oracle_THIS_dataset) because it scales as sum(log k)/tokens and k8 is ~3.3x smoke3.

  THE eps-OBEDIENCE FAMILY IS WITHDRAWN AS A QUOTABLE BOUND -- UNREPRODUCED AND MISLABELLED.
  It read: eps=0.50 -> +0.0116  eps=0.30 -> +0.0182  eps=0.10 -> +0.0307  eps=0.01 -> +0.0550.
  (i) NO PRODUCER exists on disk. An auditor who reproduced 537.01 / 36,364 / 0.0148 / 0.0123 /
      47.89% / 66.67% / 2.0698 all to the digit could not reproduce this family under any of four
      natural closed forms; the nearest gives 0.0119/0.0181/0.0295/0.0513, matching at eps=0.50
      and drifting -6.7% by eps=0.01. The MAGNITUDE ~0.012 is right; the exact values are not.
  (ii) IT WAS LABELLED BACKWARDS. Lower eps = a MORE obedient simulated model = a LARGER value, so
      +0.0116 is the SMALLEST member of its own family. Calling it "THE CORRECT CEILING" and
      printing "% of the +0.0116 ceiling" inverted floor and ceiling -- and the same paragraph two
      lines down already called ~+0.010 a FLOOR. A genuinely steering model would have printed
      >100% "of ceiling" and no reader could have known what that meant.
  QUOTE THE ORACLE CEILING, WHICH REPRODUCES, AND COMPUTE IT ON THE SET BEING SCORED. Never quote
  GAP_neg against the ~0.194 base loss: the channel structurally cannot move most of the loss
  (even at an absurd k_eff=1000 the bound is 0.133), so a reader shown 0.012/0.194 will call a
  near-maximal result noise.

TWO DILUTIONS THAT MUST BE STATED WITH ANY POOLED NUMBER:
  - negation REDIRECTS to a different variant on only 47.89% of VALID rows (66.67% of multi-variant
    valid rows; whole-set the figures are 53.71% / 70.41%). Both reproduce exactly against the mask
    score() actually computes, not merely against this prose. On the rest, the nearest variant to -c
    IS the true variant, so the opposite instruction asks for the same output and the row
    contributes exactly zero.
  - THE "2.07x DILUTION" IS AN ARITHMETIC IDENTITY, NOT A MEASUREMENT, and was twice written here
    as "measured". Because non-redirecting rows contribute EXACTLY ZERO by construction, restricted
    divided by pooled is FORCED to equal total tokens / redirecting tokens = 3678/1777 = 2.0698 on
    smoke3 valid. It carries no information about any model. This is the #50 shape -- an arithmetic
    identity presented as a finding. Worse, the +0.01159 in that ratio was never a measured
    GAP_neg: it is the eps=0.50 SIMULATION value withdrawn above.
  - 23.7% of rows have a SINGLE-VARIANT parent -- that is a WHOLE-SET figure. On the VALID split,
    where the two redirect percentages above live, it is 28.17%, 19% relatively larger. Three
    numbers in one bullet list were being read as one population and were measured on two.

AND A CONFOUND THIS READOUT CANNOT REPORT: a loss difference cannot separate "the model ignores the
channel" from "the model uses it but -c is out of distribution". A generation-based obedience metric
is a COMPLEMENT, not a replacement.
"""
from __future__ import annotations
import math
import os, sys, json, argparse, subprocess
import numpy as np
import torch

ROOT = '/Users/shaharharel/Documents/github/edit-small-mol'
CF = os.path.join(ROOT, 'experiments/covalentformer')
sys.path.insert(0, CF)
ARMS = ('none', 'extent', 'topodiam', 'resid')


def _negate(c):
    """THE negation. ONE definition, imported by both score() and the self-check.

    Previously this expression was typed out twice -- once in score() and once in
    _negation_reaches_model -- so the self-check proved that ITS OWN copy reaches the model, not
    that the scored copy does. Breaking score()'s line alone left the self-check passing cleanly.
    That is precisely the retyped-metric failure steer_params.py's docstring exists to prevent, and
    the one that put a de-leak clause in three of four sites. Now there is one definition and the
    guard tests the code that actually runs.
    """
    import torch as _t
    return _t.stack([-c[:, 0], c[:, 1]], 1)


def _eval(model, loader, dev, transform, detail=False):
    """TOKEN-WEIGHTED (micro) mean loss with `transform` applied to the conditioning.

    NOT THE SAME ESTIMATOR AS train_phaseA's history.json valid, and the difference is 24x the
    effect being reported. The trainer computes sum(loss*mask)/mask.sum() PER BATCH and averages
    those batch means -- a MACRO average, in which a 7-row final batch carries one third of the
    answer. This accumulates loss and tokens across the whole loader and divides once. Measured
    offset on the four probe checkpoints: -0.00483, -0.00486, -0.00488, -0.00482, against GAPs of
    at most +0.00023. Reproducing the trainer's recipe gives its numbers bit-identically in all 17
    digits, so the cause is isolated and it is aggregation alone (the extra pad mask here is inert
    to the last bit; batch size matters only under macro averaging).

    Micro is the better estimator. But loss_true from this function MUST NEVER be quoted beside a
    trainer valid loss -- that is the estimator straddle that produced four wrong numbers earlier in
    this project. The GAPs are differences of two calls to THIS function and are internally
    consistent.
    """
    import torch.nn.functional as F
    model.eval()
    tot, ntok = 0.0, 0
    rl, rt = [], []          # PER-ROW loss sums and token counts, in loader order
    with torch.no_grad():
        for batch in loader:
            if batch is None:
                continue
            src, sm, tgt, cond, _r = batch
            src, sm, tgt, cond = src.to(dev), sm.to(dev), tgt.to(dev), cond.to(dev)
            c = transform(cond)
            ti, to = tgt[:, :-1], tgt[:, 1:]
            import train_phaseA as TA
            tm = TA.subsequent_mask(ti.size(1), dev) & (ti != 0).unsqueeze(-2)
            out = model.forward_cond(src, sm, ti, tm, c)
            lg = model.generator(out)
            m = (to != 0)
            if not m.any():
                continue
            # PER-ROW, then summed -- not reduction='sum' over the flat masked selection.
            # Same arithmetic, but it yields the row-level decomposition that EVERY confidence
            # interval on a GAP needs. Without it the only honest statement about GAP_neg is a
            # point estimate, which is how "all CIs span zero, largest |t|=1.03" stayed unquantified.
            lo = F.cross_entropy(lg.reshape(-1, lg.size(-1)), to.reshape(-1),
                                 reduction='none').view(to.shape) * m
            # WITHIN-ROW SUM IN float32 ON DEVICE, CROSS-ROW ACCUMULATION IN float64 ON CPU.
            # `lo.double()` raises on MPS -- "Cannot convert a MPS Tensor to float64" -- which a dry
            # run caught only because it exercised the scorer end to end; the four arms had already
            # trained to exit 0 before this line was ever reached. The cast therefore has to happen
            # on the CPU side of the boundary, which is also where it actually matters: the within-
            # row sum spans ~100 tokens (float32 is ample) while the accumulation that can drift
            # spans thousands of rows.
            # HONEST SCOPE, correcting what this comment said before: the previous version claimed
            # double accumulation made the path "strictly MORE accurate" than the reduction='sum'
            # it replaced. With the cast moved to CPU the within-row sum is float32, so the correct
            # claim is narrower -- equal precision within a row, better across rows. Measured drift
            # against the old path was 1.5e-5 nats on a 194-nat batch = 2.6e-7 nats/token, ~1000x
            # below this readout's 0.00028 MDE either way.
            rs, rn = lo.sum(1).detach().cpu().double(), m.sum(1)
            rl.append(rs.numpy()); rt.append(rn.detach().cpu().numpy())
            tot += float(rs.sum()); ntok += int(rn.sum())
    pooled = tot / max(ntok, 1)
    if not detail:
        return pooled
    return dict(pooled=pooled,
                row_loss=np.concatenate(rl) if rl else np.zeros(0),
                row_ntok=np.concatenate(rt) if rt else np.zeros(0))


def _negation_reaches_model(ckpt_path, valid_csv):
    """Assert the negated conditioning ACTUALLY reaches the model and changes its output.

    This is the self-check that the `none`-arm test could not perform. `none` is constant-zero
    conditioned, so -c == c by arithmetic and its GAP_neg is 0 whether the transform is applied or
    not -- a silently-unapplied transform passes that check cleanly. The only way to distinguish
    "the channel is inert" from "my transform is a no-op" is to instrument the tensors on a
    CONDITIONED arm, which is what this does. Returns the two magnitudes the caller asserts on.
    """
    import train_phaseA as TA
    from torch.utils.data import DataLoader
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    V = TA.Vocabulary
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, V) else V(_v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    m = TA.PhaseA(mode='geom', **dict(ck['network_parameter']))
    m.load_state_dict(ck['model_state'], strict=True)
    m.eval()
    ds = TA.RoleData(valid_csv, vocab, TA.SMILESTokenizer(), 'geom')
    # INSTRUMENT THE ROWS WITH THE LARGEST |d|, NOT THE FIRST 16. dc = max|c - (-c)| = max|2*c|, so
    # a batch that happens to be all-zero conditioning gives dc = 0 and raises "negation does not
    # reach the model" for a HEALTHY arm. topodiam is an integer channel with 36/71 rows at d=0 on
    # smoke3; one CSV ordering away from a false FATAL that (see below) discarded the whole table.
    # Selecting by |d| makes the probe test the model rather than the row order.
    try:
        import pandas as _pd1
        _dd = _pd1.read_csv(valid_csv)['d'].values.astype(float)
        _ord = np.argsort(-np.abs(_dd))[:16]
        _sub = torch.utils.data.Subset(ds, [int(i) for i in _ord if ds[int(i)] is not None])
        b = next(iter(DataLoader(_sub, batch_size=16, shuffle=False, collate_fn=TA.collate)))
    except Exception:
        b = next(iter(DataLoader(ds, batch_size=16, shuffle=False, collate_fn=TA.collate)))
    src, sm, tgt, cond, _r = b
    neg = _negate(cond)          # ONE definition, see _negate()
    # THE SLOT-1 ASSERT ON REAL DATA CANNOT FIRE, so it is done on a SYNTHETIC vector instead.
    # adapt pins cos_theta = 0.0 for every arm, so slot 1 is identically 0.0 and negating gives
    # -0.0 -- and torch.equal(0.0, -0.0) is True. The assertion therefore passed even if _negate
    # were replaced by a blanket `return -c`, i.e. it proved nothing about WHICH SLOT was negated,
    # which is the one thing it claimed. Sixth guard-that-cannot-fire in this project. A synthetic
    # probe with a NONZERO slot 1 tests the property for real and is independent of the data.
    _p = torch.tensor([[1.5, 2.5], [-3.0, 4.0]])
    _q = _negate(_p)
    assert torch.allclose(_q[:, 0], -_p[:, 0]), 'slot 0 must be negated'
    assert torch.allclose(_q[:, 1], _p[:, 1]), 'slot 1 must be untouched by the negation'
    with torch.no_grad():
        ti = tgt[:, :-1]
        tm = TA.subsequent_mask(ti.size(1), 'cpu') & (ti != 0).unsqueeze(-2)
        o1 = m.generator(m.forward_cond(src, sm, ti, tm, cond.float()))
        o2 = m.generator(m.forward_cond(src, sm, ti, tm, neg.float()))
    return dict(dc=float((cond - neg).abs().max()), dlogit=float((o1 - o2).abs().max()))


def score(ckpt_path, valid_csv, n_perm=20, seed=20260915, expect_epoch=None, expect_total=None):
    # n_perm 5 -> 20. The permutation sd is the yardstick the headline is judged against, and at
    # n=5 its own standard error is ~32% of itself (1/sqrt(2(n-1))), i.e. the ruler was more
    # uncertain than most of the differences being measured against it. At n=20 that falls to ~16%.
    # Cost is 15 extra forward passes over a 71-row valid set -- seconds.
    import train_phaseA as TA
    from torch.utils.data import DataLoader
    dev = 'mps' if torch.backends.mps.is_available() else 'cpu'
    # LOADER WIRED TO train_phaseA's OWN SAVE FORMAT, AND strict=True ON PURPOSE.
    # train_phaseA saves {'model_state', 'network_parameter', 'vocabulary', 'mode', 'seed', ...}.
    # It LOADS the prior with strict=False because the prior has no role_emb/geom_mlp yet -- correct
    # there, wrong here: an arm checkpoint contains every key, so anything missing or unexpected
    # means the architecture I rebuilt is not the architecture that was trained. strict=False is
    # exactly how `cond_repeat` evaluated a k=8 model as k=1 without erroring. A scorer that loads
    # leniently reports a plausible wrong number, which is the failure mode this whole project keeps
    # hitting. So: rebuild from the checkpoint's OWN network_parameter and mode, load strictly, and
    # refuse if the stamped mode is not the one being scored.
    # Take Vocabulary/SMILESTokenizer FROM train_phaseA's namespace rather than re-importing.
    # My first attempt guessed `reinvent_models.molformer.models.vocabulary`; the real path is
    # `reinvent.models.transformer.core.vocabulary`, reachable only after train_phaseA's own
    # sys.path.insert(0, REINVENT). Importing TA already ran that, so TA.Vocabulary is the same
    # class object the checkpoint was written with -- and re-importing by a guessed path risks
    # loading a DIFFERENT class that isinstance() would then silently reject.
    Vocabulary, SMILESTokenizer = TA.Vocabulary, TA.SMILESTokenizer
    ck = torch.load(ckpt_path, map_location='cpu', weights_only=False)
    if ck.get('mode') != 'geom':
        raise SystemExit('FATAL: %s is stamped mode=%r; this scorer negates slot 0 of a geom '
                         'conditioning vector and is meaningless for any other mode.'
                         % (ckpt_path, ck.get('mode')))
    _v = ck['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    tok = SMILESTokenizer()
    model = TA.PhaseA(mode='geom', **dict(ck['network_parameter']))
    model.load_state_dict(ck['model_state'], strict=True)   # strict: see above
    # CHECK THE CHECKPOINT'S OWN PAIRING STAMP. train_phaseA records valid_file precisely so the
    # pairing is checkable afterwards, and this scorer -- built to be paranoid about strict=True --
    # was throwing it away. Demonstrated: scoring the EXTENT checkpoint against none_valid.csv
    # returns GAP_neg +0.000000 with no warning, a flawless "the channel is inert" result
    # manufactured entirely by the wrong data file.
    # C3: the epoch comes from a CLI flag and was never checked against the checkpoint, although
    # every checkpoint stamps it. `--score-only --epochs 3` against a dir trained with --epochs 5
    # silently scores ep2 and reports it as the run -- and per #117/#121 held-out loss moves by up
    # to +0.24 nats between epochs, so that is a plausible-wrong-number path of exactly the
    # cond_repeat class: a switch stored in the checkpoint and not replayed on load.
    # THE `epoch` COMPARISON IS A TAUTOLOGY AND CANNOT CATCH THE CASE IT NAMES. The caller selects
    # the file as ep{epochs-1}.ckpt and train_phaseA writes epN.ckpt stamping epoch=N from the same
    # loop variable, so ck['epoch'] == expect_epoch identically -- verified on all 84 checkpoints
    # under data/steer_k8*/, zero mismatches. Worse, the hazard in the comment above -- `--score-only
    # --epochs 3` against a dir trained with --epochs 5 -- is exactly what still passes: ep2.ckpt of
    # a 5-epoch run stamps epoch=2, expect_epoch=2, guard silent. The discriminating key is in the
    # SAME dict: train_phaseA also stamps `epochs` (the TOTAL). Checked below, and REFUSED when
    # missing, matching the valid_file treatment -- `if _x is not None` degrades a guard to a silent
    # no-op precisely on the hand-made checkpoints where provenance is weakest.
    _ep = ck.get('epoch')
    if _ep is not None and expect_epoch is not None and int(_ep) != int(expect_epoch):
        raise SystemExit('FATAL: %s is stamped epoch=%s but is being scored as epoch %s.'
                         % (ckpt_path, _ep, expect_epoch))
    if expect_total is not None:
        _tot = ck.get('epochs')
        if _tot is None:
            raise SystemExit('FATAL: %s carries no `epochs` (total) stamp, so "is this the LAST '
                             'epoch of the run" cannot be verified and the `epoch` check above is '
                             'a tautology on its own. Refusing.' % ckpt_path)
        if int(_tot) != int(expect_total):
            raise SystemExit('FATAL: %s belongs to a %s-epoch run but is being scored as the final '
                             'checkpoint of a %s-epoch run. Held-out loss moves up to +0.24 nats '
                             'between epochs (#117/#121), so this is a plausible-wrong-number path.'
                             % (ckpt_path, _tot, expect_total))
    _want = ck.get('valid_file')
    # BASENAME BOTH SIDES, AND REFUSE A MISSING STAMP. Two latent holes, neither reachable under
    # launch() but both live for a hand-trained checkpoint: (a) `if _want and ...` degraded the
    # guard to a SILENT NO-OP on any checkpoint without the stamp -- a guard that disappears
    # exactly when provenance is weakest; (b) training with --valid-file sub/extent_valid.csv
    # stored the subpath, so basename-vs-path FALSE-FIRED on the correct file.
    if _want is None:
        raise SystemExit('FATAL: %s carries no valid_file stamp, so the checkpoint/data pairing '
                         'CANNOT be verified. Refusing rather than scoring an unverifiable pair.'
                         % ckpt_path)
    # CONTENT HASH, NOT JUST BASENAME. The pairing guard compares BASENAMES, and four directories
    # on disk hold a file called `extentdisc_valid.csv` with DIFFERENT bin means (the per-split vs
    # pooled refit). Scoring a checkpoint trained on one against another passes the basename check
    # silently and returns a plausible wrong GAP -- the cond_repeat shape, one level out in the
    # filesystem. train_phaseA stamps `valid_file` as a name only, so the hash is compared when the
    # checkpoint carries one and REPORTED (not fatal) when it does not, since every checkpoint
    # trained before this line exists has no hash to compare and must stay scorable.
    _h = ck.get('valid_sha')
    if _h:
        import hashlib as _hl
        _now = _hl.sha256(open(valid_csv, 'rb').read()).hexdigest()[:16]
        if _now != _h:
            raise SystemExit('FATAL: %s was trained on a valid file with sha %s but %s hashes to '
                             '%s. Same basename, different CONTENT -- four dirs hold an '
                             'identically-named extentdisc_valid.csv with different bin means. '
                             'This is the pairing failure the basename check cannot see.'
                             % (ckpt_path, _h, valid_csv, _now))
    else:
        print('    [pairing] checkpoint carries no valid_sha -- basename-only check (pre-dates the '
              'content hash). Four dirs share the name %s.' % os.path.basename(valid_csv))
    if os.path.basename(valid_csv) != os.path.basename(_want):
        raise SystemExit('FATAL: %s was trained against valid_file=%r but is being scored on %r. '
                         'A mismatched pair yields a spectacular fake result.'
                         % (ckpt_path, _want, os.path.basename(valid_csv)))
    model.to(dev)
    ds = TA.RoleData(valid_csv, vocab, tok, 'geom')
    dl = DataLoader(ds, batch_size=32, shuffle=False, collate_fn=TA.collate)
    dbase = _eval(model, dl, dev, lambda c: c, detail=True)
    dneg = _eval(model, dl, dev, _negate, detail=True)
    base, neg = dbase['pooled'], dneg['pooled']
    zero = _eval(model, dl, dev, lambda c: torch.zeros_like(c))
    # Conditioning column read once, up front, and ONLY used if its length matches the rows the
    # scorer actually collected -- same alignment contract as the cluster and redirect blocks.
    _dv_d = None
    try:
        import pandas as _pdp
        _tmp = _pdp.read_csv(valid_csv)['d'].values.astype(float)
        if len(_tmp) == len(dbase['row_ntok']):
            _dv_d = _tmp
        else:
            print('    [perm] full-set permutation REFUSED: csv %d rows, scorer %d'
                  % (len(_tmp), len(dbase['row_ntok'])))
    except Exception as _e:
        print('    [perm] full-set permutation REFUSED: %s' % _e)
    # FULL-SET PERMUTATION, NOT WITHIN-BATCH. `transform` is applied per batch, so
    # torch.randperm(c.size(0)) confined every shuffle to 32 rows (and a 7-row final batch). That is
    # structurally the SAME construction this file's own docstring quotes from train_phaseA and
    # calls "DILUTED and BATCH-ORDER-DEPENDENT ... Do not quote `gap` as an effect size" -- shipped
    # here in the file that condemns it.
    # It is not merely dilution, it is BIASED dilution: adapt writes a parent's variants ADJACENTLY,
    # so a within-batch shuffle disproportionately swaps in a SIBLING's conditioning, and siblings
    # ask for similar values. Measured on the smoke3 extent arm: 38% of sibling pairs share a batch
    # at bs=32 against ~1.0% under a full-set permutation -- a ~38x over-sampling of the swaps that
    # change the request least. The effect understates GAP_perm and, worse, tightens GAP_perm_sd,
    # which is the ruler GAP_neg is measured against.
    # Fixed by drawing ONE permutation over all rows per replicate and indexing it with a running
    # offset. The offset advances in `transform`, which _eval calls once per batch BEFORE any
    # skip, so it stays aligned with the loader. Falls back to the old within-batch shuffle only if
    # the row count cannot be established, and says so rather than silently differing.
    g = torch.Generator().manual_seed(seed)
    _nrows = len(dbase['row_ntok'])
    perms = []
    for _r in range(n_perm):
        _vals = torch.tensor(_dv_d, dtype=torch.float32)[torch.randperm(_nrows, generator=g)] \
            if _dv_d is not None else None
        if _vals is None:
            def t(c, _g=g):
                p = torch.randperm(c.size(0), generator=_g).to(c.device)
                return torch.stack([c[p, 0], c[:, 1]], 1)
        else:
            _st = {'off': 0}
            def t(c, _v=_vals, _s=_st):
                n = c.size(0)
                col = _v[_s['off']:_s['off'] + n].to(c.device).to(c.dtype)
                _s['off'] += n
                if col.numel() != n:                       # alignment lost -> do not fabricate
                    raise RuntimeError('perm offset desync: wanted %d, got %d' % (n, col.numel()))
                return torch.stack([col, c[:, 1]], 1)
        perms.append(_eval(model, dl, dev, t))
    perms = np.array(perms)
    perm_scope = 'full-set' if _dv_d is not None else 'WITHIN-BATCH fallback (row count unknown)'

    # ---- BOOTSTRAP CI ON GAP_neg. Until now GAP_neg was a bare point estimate, so "all CIs span
    # zero, largest |t| = 1.03" was an assertion with nothing behind it. The resample is over ROWS
    # and is PAIRED by construction -- d_i is the same row's negated-minus-true loss, so the
    # within-row correlation that makes the contrast precise is preserved instead of being thrown
    # away by resampling the two arms independently.
    d_row = dneg['row_loss'] - dbase['row_loss']
    n_row = dbase['row_ntok'].astype(float)
    rng = np.random.default_rng(seed)

    # CLUSTER on the parent, DO NOT resample rows. Rows sharing a parent are dependent BY
    # CONSTRUCTION: the conditioning is a deterministic function of which variant is asked for
    # GIVEN the parent, and whether -c redirects at all is a property of that parent's variant set.
    # This mattered little at 1.77 variants/parent, but MAX_REPL=100 makes the cap bind and the
    # post-cap mean is ~6.7 rows/parent, so the design effect grows ~3.5x at exactly the moment a
    # CI starts being quoted. Concretely: ~816 valid rows but only ~121 independent parents -- a
    # row bootstrap would report p<0.001 off 121 real observations. That is the single most likely
    # way this file emits a plausible wrong number, and it errs in the same direction as every
    # prior error in this project. Credit: QA-code-40 flagged this against the row-resampling
    # version I had just written.
    # The true independence unit is the union-find COMPONENT, not the parent -- two parents joined
    # by a shared variant are one cluster. adapt_steer_to_phaseA now carries `_comp` when the
    # source has it; parent is the fallback and is an APPROXIMATION, so which one was used is
    # recorded in the result rather than assumed.
    def _boot(clusters, mask=None, B=2000):
        sel = np.ones(len(d_row), bool) if mask is None else mask
        dd, nn, cc = d_row[sel], n_row[sel], clusters[sel]
        if len(dd) < 2 or nn.sum() <= 0:
            return (float('nan'), float('nan'), float('nan'), 0, 0)
        uniq = np.unique(cc)
        groups = [np.where(cc == u)[0] for u in uniq]
        # Ratio-of-sums inside every replicate -- numerator AND denominator resampled together.
        # Bootstrapping the mean of per-row d_i/n_i would silently switch the estimand to a
        # ROW-weighted one, a different statistic from the token-weighted headline. This file's
        # own docstring forbids straddling two estimators; that applies to its CI too.
        bs = np.empty(B)
        for b in range(B):
            pick = rng.integers(0, len(groups), size=len(groups))
            idx = np.concatenate([groups[j] for j in pick])
            bs[b] = dd[idx].sum() / max(nn[idx].sum(), 1e-9)
        return (float(dd.sum() / nn.sum()), float(np.percentile(bs, 2.5)),
                float(np.percentile(bs, 97.5)), int(len(dd)), int(len(uniq)))

    # NO CLUSTER COLUMN => NO INTERVAL. This used to fall back to clust=arange(n), i.e. a ROW-level
    # bootstrap, carrying an 'ANTI-CONSERVATIVE' caveat string. That is a DEGRADE, and what it
    # degrades to is measured: at this build's geometry the row-level interval excludes zero in
    # 12/40 trials on data with a TRUE effect of exactly zero -- a 30% false-positive rate against
    # a nominal 5% (independently reproduced from this file's own _boot; the cluster version sits
    # at 2/40, i.e. calibrated). A caveat string does not survive into a plot, and every other
    # guard in this file REFUSES rather than degrades (missing valid_file, wrong mode, <4 arms).
    # So: emit NO interval instead of a wrong one. The point estimate is unaffected and still
    # reported. In practice this cannot fire for a normal arm CSV -- `anchor` is always present --
    # so reaching it means something upstream is already broken and NaN is the honest output.
    # _d0 INITIALISED BEFORE THE try, because it is read by the ceiling and yardstick blocks below.
    # Assigned only inside the try, it is a local that stays UNBOUND if read_csv raises -- the exact
    # UnboundLocalError shape just fixed in build_steer_pairs._one_parent, where three early returns
    # referenced a name assigned further down and every one of them raised instead of returning.
    # There the failures were masked as "worker died"; here they would be masked as "[yardstick]
    # REFUSED", which reads like a data problem rather than a code one. One line removes it.
    clust, clust_kind, _d0 = None, None, None
    try:
        import pandas as _pd0
        _d0 = _pd0.read_csv(valid_csv)
        if len(_d0) == len(d_row):
            _col = '_comp' if '_comp' in _d0.columns else 'anchor'
            clust = _pd0.factorize(_d0[_col])[0]
            clust_kind = _col
        else:
            print('    [cluster] REFUSED: csv has %d rows, scorer collected %d' % (len(_d0), len(d_row)))
    except Exception as _e:
        print('    [cluster] REFUSED: %s' % _e)
    if clust is None:
        clust_kind = 'NONE -- CI REFUSED (no verifiable cluster column)'
        gn, lo95, hi95, n_all, n_clust = (float(d_row.sum() / max(n_row.sum(), 1e-9)),
                                          float('nan'), float('nan'), int(len(d_row)), 0)
    else:
        gn, lo95, hi95, n_all, n_clust = _boot(clust)
    # MEASURE THE DESIGN EFFECT, DO NOT ASSUME IT. The row-level CI is computed ONLY to report how
    # much the clustering widens the interval -- it is NOT a result and must never be quoted as one.
    # Two reasons it is worth carrying: (1) it makes the anti-conservatism concrete rather than
    # asserted, and (2) the design effect is a property of THIS BUILD'S GEOMETRY, set directly by
    # per_env_cap. Re-capping the same pairs at a higher cap raises rows/cluster and therefore
    # WIDENS this CI even as the row count grows -- counterintuitive enough that a later reader
    # would otherwise read a wider CI at larger n as a regression. It is the honest direction.
    if clust is None:
        _de = float('nan')          # refused above; do not manufacture a ratio against nothing
    else:
        _, rlo_, rhi_, _, _ = _boot(np.arange(len(d_row)))
        _de = ((hi95 - lo95) / (rhi_ - rlo_)) if (rhi_ - rlo_) > 0 else float('nan')

    # ---- RESTRICTED TO REDIRECTING ROWS. Negation only ASKS for a different variant on some rows;
    # on the rest the nearest achievable variant to -c IS the true target, so the row contributes
    # exactly zero for ANY model and dilutes the pooled number (measured 2.07x). Recomputed here
    # from the valid CSV rather than inherited as a constant.
    # THE GUARD MATTERS: RoleData can drop unparseable rows, and a silently misaligned mask would
    # label the WRONG rows as redirecting and produce a confident wrong number -- the exact failure
    # class this scorer exists to avoid. If the lengths disagree the statistic is REFUSED, not fudged.
    red = None
    try:
        import pandas as _pd
        _dv = _pd.read_csv(valid_csv)
        if len(_dv) == len(d_row):
            _c = _dv['d'].values.astype(float)
            _m = np.zeros(len(_dv), dtype=bool)
            # `.indices` NOT `.groups`: .groups returns LABEL indices while `_c` and `_m` are
            # POSITIONAL. They coincide only because read_csv happens to yield a RangeIndex today.
            # That is an unguarded assumption that fails SILENTLY -- wrong rows marked redirecting,
            # no error -- if the frame ever carries a non-trivial index. .indices is positional by
            # definition, so the coincidence stops being load-bearing.
            for _a, _i in _dv.groupby('anchor').indices.items():
                _i = np.asarray(_i)
                if len(_i) < 2:
                    continue                      # single-variant parent: nothing to redirect to
                for _k in _i:
                    # A REDIRECT REQUIRES A DIFFERENT REQUESTED VALUE, NOT MERELY A DIFFERENT ROW.
                    # np.argmin returns the FIRST minimiser, so when two variants of a parent carry
                    # the SAME d, the later row was marked "redirecting" to a sibling asking for an
                    # IDENTICAL conditioning value -- a row that contributes structurally zero for
                    # any model. Measured on the smoke3 arms: the `none` arm (d identically 0) was
                    # marked 29/71 = 40.8% redirecting, ALL of them ties, printed one line under a
                    # docstring quoting 47.9% -- a control arm appearing to redirect nearly as often
                    # as a real one. topodiam, an INTEGER channel, had 9 of 31 tie artifacts, so its
                    # restricted GAP was diluted ~1.4x where extent and resid (continuous floats, no
                    # ties) had zero. The bias ran AGAINST the arm adapt's docstring calls "THE ARM
                    # THAT MATTERS", and integer channels collide far more at 15k rows than at 700.
                    _j = _i[np.argmin(np.abs(_c[_i] - (-_c[_k])))]
                    _m[_k] = bool(_c[_j] != _c[_k])
            red = _m
        else:
            print('    [redirect] REFUSED: csv has %d rows, scorer collected %d -- mask would be '
                  'misaligned' % (len(_dv), len(d_row)))
    except Exception as e:
        print('    [redirect] REFUSED: %s' % e)
    gnr, rlo, rhi, n_red, n_red_cl = (_boot(clust, red) if (red is not None and clust is not None)
                                      else (float('nan'),) * 3 + (0, 0))

    # ---- ORACLE CEILING COMPUTED ON *THIS* VALID SET, NOT INHERITED.
    # The ceiling is sum_rows log(k_parent) / total_tokens -- the entropy of variant-given-parent,
    # which is what a perfect user of the channel recovers over a perfect ignorer. It therefore
    # MOVES WITH k, and the +0.0116 previously hardcoded here was measured on smoke3 at k=1.77.
    # The k8 rebuild exists precisely to raise k to ~8 (realised 14.3 pre-cap, 8 post-cap), so the
    # inherited constant understates the achievable range by roughly 3-4x on the data this scorer
    # is about to be pointed at. Quoting a GAP as a percentage of the WRONG population's ceiling is
    # the #179 shape -- a bound imported from a different population than the one being measured --
    # and it would make a null look near-ceiling or a real effect look small, depending on sign.
    # Both pieces are already in hand (row token counts from _eval, k from the valid CSV), so this
    # is exact and free rather than estimated.
    # TWO CEILINGS, because the generic one is ARM-BLIND and this arm may not be able to reach it.
    # sum(log k)/tokens assumes the conditioning IDENTIFIES which variant is wanted. A DISCRETE
    # channel cannot: siblings of one parent that share the same d are unresolvable BY CONSTRUCTION,
    # so the achievable ceiling is sum(log k) MINUS sum(log #same-d siblings). Measured on the k8
    # VALID set that is 52.7% of the generic ceiling for topodiam and 100% for every continuous arm.
    # AND THAT IS WHY IT IS A DIAGNOSTIC, NOT A DENOMINATOR. I first shipped it as the per-arm
    # denominator on the reasoning that a single generic bound "understates the discrete arm" --
    # true, but the correction is not arm-symmetric. Ties are broken by NOISE, so every jittered
    # control gets a 1.000x ceiling while the integer it is a control FOR gets 0.527x, and dividing
    # each by its own ceiling multiplies topodiam's share by 1.90x relative to its own noise control
    # and 1.41x relative to extentdisc -- both in the direction of the filed conclusion, and on
    # extentdisc it undoes in the normaliser most of the resolution matching that arm exists to
    # supply. topojit is information-preserving (round(topojit)==topodiam, 8956/8956), so its extra
    # "identifiable" entropy is pure float resolution and not usable information. Both are reported;
    # the PERCENTAGE uses the generic bound, which is common to all arms. Separately the `none` arm
    # was being given a ceiling of 0.0383 when its achievable ceiling is exactly 0 -- now named as
    # zero rather than back-filled from another population.
    # THE MAGNITUDE-ONLY YARDSTICK FOR THE DIRECTIONALITY RATIO. GAP_neg/GAP_perm was being read
    # against an implicit null of 1.0x, and that null is WRONG because the two perturbations are not
    # the same size: negation moves the conditioning by |2d| while permutation moves it by
    # |d_i - d_j| against a random other row. A model that responds ONLY to how far the number moved
    # -- no sign obedience whatever -- therefore scores above 1.0x by construction. Measured on the
    # k8 valid set the ratio mean|2d| / E|d_i-d_j| is 1.33-1.45x depending on the arm, so an arm at
    # 1.2x is BELOW its own magnitude-only null while the old reading called it directional.
    # Computed here per arm, from the same CSV the GAP is measured on, and stored so the verdict is
    # checkable rather than asserted. Its own seeded RNG so it cannot disturb the permutation draw.
    yard = float('nan'); yard_se = float('nan')
    try:
        _dv = _d0['d'].values.astype(float) if _d0 is not None else None
        if _dv is not None and np.abs(_dv).sum() > 0:
            _yr = np.random.default_rng(31337)
            _draws = np.array([np.abs(_dv - _yr.permutation(_dv)).mean() for _ in range(200)])
            _pp = float(_draws.mean())
            yard = float(np.abs(2 * _dv).mean() / _pp) if _pp > 0 else float('nan')
            # SE OF THE YARDSTICK ITSELF. Without this there is no margin on the RATIO axis, and
            # the marginality flag can only ever police the significance axis -- which is exactly
            # the hole an auditor found: the flag fired on a cell whose verdict was already
            # negative and stayed SILENT on extentdisc@s777, the one cell this whole mechanism was
            # built for (ratio 1.4228 vs yardstick 1.3964, +1.89%).
            yard_se = float(_draws.std(ddof=1) / np.sqrt(len(_draws)) / _pp) if _pp > 0 else float('nan')
    except Exception as _e:
        print('    [yardstick] REFUSED: %s' % _e)

    ceil_oracle = float('nan'); ceil_arm = float('nan')
    try:
        if clust is not None:
            _k = _d0.groupby('anchor')['anchor'].transform('size').values.astype(float)
            # DISTINCT conditioning values, not group size. The channel can only separate variants
            # carrying a DIFFERENT d; siblings that tie are unresolvable BY CONSTRUCTION and that
            # entropy is not achievable. This is the MIRROR of the tie bug already fixed in the
            # redirect mask -- ties handled in one place and not the other, the retyped-metric
            # pattern one level up. log(1)=0 so a single-valued parent contributes exactly 0, which
            # makes the `none` arm's ceiling exactly 0.0 and correctly suppresses its percentage.
            _nd = _d0.groupby('anchor')['d'].transform(lambda z: z.round(12).nunique()).values.astype(float)
            _tok = max(n_row.sum(), 1e-9)
            ceil_oracle = float(np.log(_k).sum() / _tok)
            ceil_arm = float(np.log(_nd).sum() / _tok)
    except Exception as _e:
        print('    [ceiling] REFUSED: %s' % _e)

    return dict(loss_true=base, GAP_neg=neg - base, GAP_zero=zero - base,
                GAP_neg_rowsum=gn, GAP_neg_ci95=[lo95, hi95], n_rows_scored=n_all,
                n_clusters=n_clust, cluster_unit=clust_kind,
                rows_per_cluster=(float(n_all) / n_clust) if n_clust else None,
                design_effect_ci_width_ratio=_de,
                design_effect_note=('CI width ratio cluster/row. >1 means the row-level CI was '
                                    'anti-conservative by this factor. It RISES with rows per '
                                    'cluster, which per_env_cap sets -- so a higher cap can widen '
                                    'this CI even while adding rows. More PARENTS buy independent '
                                    'information; more rows per parent mostly buy row count.'),
                ceiling_oracle_THIS_dataset=ceil_oracle,
                ceiling_identifiable_THIS_ARM=ceil_arm,
                ceiling_note=('ceiling_oracle_THIS_dataset is the GENERIC bound, arm-blind and '
                              'IDENTICAL for every arm on this data -- it is the cross-arm '
                              'denominator, and the printed percentage uses it. '
                              'ceiling_identifiable_THIS_ARM subtracts sibling ties this channel '
                              'cannot resolve; it is a RESOLUTION DIAGNOSTIC and MUST NOT be used '
                              'as a cross-arm denominator, because it REWARDS NOISE: jitter breaks '
                              'float ties, so topojit -- provably information-preserving, '
                              'round(topojit)==topodiam on 8956/8956 rows -- scores 1.000x generic '
                              'against topodiam\'s 0.527, inflating topodiam\'s share 1.90x vs its '
                              'own noise control and 1.41x vs the resolution-matched extentdisc. '
                              'A per-arm denominator would need a tie structure SHARED by all arms.'),
                ceiling_INHERITED_smoke3=dict(
                    WITHDRAWN_eps050_DO_NOT_QUOTE=0.0116, oracle=0.0123,
                    scope=('measured on smoke3 at k=1.77. The oracle ceiling scales as '
                           'sum(log k)/tokens, so these are NOT valid for a higher-k dataset -- '
                           'at k=8 the ceiling is ~3.3x larger. Kept only as the reference that '
                           'the eps=0.50 negation ceiling sat at 0.943x the oracle ceiling on '
                           'that set. The eps FAMILY IS WITHDRAWN (no producer, and labelled backwards -- it is the '
                           'SMALLEST member of its own family, not a ceiling); the 0.943x relation built on '
                           'it is withdrawn with it. Quote ceiling_identifiable_THIS_ARM.')),
                GAP_neg_redirecting=gnr, GAP_neg_redirecting_ci95=[rlo, rhi],
                n_redirecting=n_red, n_redirecting_clusters=n_red_cl,
                redirect_frac=(float(n_red) / n_all) if (n_all and n_red) else None,
                GAP_perm_mean=float(perms.mean() - base),
                # GAP_neg MINUS ITS OWN PERMUTATION NULL. Zero is NOT the null: a model that merely
                # responds to the MAGNITUDE of a request moves loss when handed any wrong number, so
                # a CI excluding zero is evidence of DEPENDENCE, not of STEERING (#98). Measured on
                # k8: extent's GAP_neg CI excludes zero on both seeds while DELTA is -0.0004..+0.0001
                # -- i.e. extent sits AT its own null -- and resid sits BELOW it. Both were reported
                # as positive steering results on the strength of the zero-CI. This field exists so
                # that cannot happen again from this file.
                GAP_neg_MINUS_perm=float(neg - base - (perms.mean() - base)),
                magnitude_only_yardstick=yard, magnitude_only_yardstick_se=yard_se,
                yardstick_note=('mean|2d| / E|d_i-d_j| on THIS arm\'s valid column: the '
                                'GAP_neg/GAP_perm ratio a model that responds only to the SIZE of '
                                'the displacement already scores, with no sign obedience at all. '
                                'It is the null for the directionality ratio; 1.0 is NOT.'),
                # ddof=1: np.std defaults to ddof=0, which biases the sd LOW -- and this sd is the
                # yardstick GAP_neg is judged against, so biasing it low makes every arm look more
                # significant than it is. At the old n_perm=5 the understatement was ~11%.
                GAP_perm_sd=float(perms.std(ddof=1)) if n_perm > 1 else float('nan'),
                n_perm=n_perm, perm_seed=seed, perm_scope=perm_scope,
                n_valid_rows=len(ds))


def launch(armdir, outdir, epochs=3, bs=64, seed=20260914, arms=None):
    """Run all four arms, CHECKING EXIT STATUS. Four silent FATALs scrolling past in a long log is
    the guaranteed outcome of a shell loop without `set -e`, and that is exactly what happened to
    the first attempt at this experiment -- every arm exited before step 0 on a canonicality gate
    and nothing in the loop noticed."""
    os.makedirs(outdir, exist_ok=True)
    # PRE-FLIGHT: EVERY REQUESTED ARM'S CSVs MUST EXIST BEFORE ANY ARM TRAINS.
    # make_ctrl_arms writes arms across THREE directories (ctrl / ctrl2 / ctrl3) and every caller
    # has to know which. That has now cost two separate launches: topojitwidescaled once, and
    # topojit+topojitwide again an hour later when I fixed the first and did not apply the lesson
    # to the queue's rep2 entry. Both times the run trained the arms it COULD find -- minutes of
    # MPS -- and only then died on a missing CSV, discarding the completed work because the
    # all-or-nothing FATAL below (correctly) refuses to ship a partial comparison.
    # Checking first turns a multi-minute loss into a one-second refusal, and the message NAMES the
    # directory the arm is actually in rather than leaving the caller to grep for it.
    _missing = []
    for arm in (arms or ARMS):
        for part in ('train', 'valid'):
            _f = os.path.join(armdir, '%s_%s.csv' % (arm, part))
            if not os.path.exists(_f):
                _alt = []
                for _sib in sorted(os.listdir(os.path.dirname(armdir) or '.')):
                    _c = os.path.join(os.path.dirname(armdir), _sib, '%s_%s.csv' % (arm, part))
                    if os.path.exists(_c):
                        _alt.append(os.path.dirname(_c))
                _missing.append((arm, part, _alt))
    if _missing:
        raise SystemExit(
            'FATAL PRE-FLIGHT: %d requested arm file(s) are not in --armdir %s, so this run would '
            'train the arms it CAN find and then die. Refusing before spending any compute.\n%s'
            % (len(_missing), armdir,
               '\n'.join('  %s_%s.csv MISSING -- %s' % (a_, p_,
                          ('found in: ' + ', '.join(alt)) if alt else 'not found in any sibling dir')
                          for a_, p_, alt in _missing)))
    results = {}
    for arm in (arms or ARMS):
        ck = os.path.join(outdir, 'ckpt_%s' % arm)
        cmd = [sys.executable, os.path.join(CF, 'train_phaseA.py'),
               '--data', armdir, '--train-file', '%s_train.csv' % arm,
               '--valid-file', '%s_valid.csv' % arm, '--mode', 'geom',
               '--epochs', str(epochs), '--bs', str(bs), '--seed', str(seed), '--out', ck]
        print('\n=== ARM %s ===' % arm, flush=True)
        rc = subprocess.call(cmd)
        if rc != 0:
            raise SystemExit('FATAL: arm %s exited %d. NOT continuing -- a failed arm makes the '
                             'four-arm comparison incomplete, and continuing would leave that '
                             'discoverable only by noticing a missing checkpoint later.' % (arm, rc))
        results[arm] = dict(ckpt=ck, returncode=rc)
    json.dump(results, open(os.path.join(outdir, 'launch.json'), 'w'), indent=1)
    print('\nall four arms completed with exit status 0')
    return results


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--armdir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--epochs', type=int, default=3)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--score-only', action='store_true')
    # SEED EXPOSED. launch() always took one but __main__ never passed it, so every arm in every
    # run trained at 20260914 and a REPLICATION WAS IMPOSSIBLE WITHOUT EDITING THE FILE. The single
    # most-repeated lesson in this project is SEED AND REPLICATE ANYTHING THAT SAMPLES; the harness
    # that enforces it could not itself be replicated.
    ap.add_argument('--seed', type=int, default=20260914)
    # ARMS EXPOSED. The four-arm tuple was module-level, so testing a MATCHED CONTROL arm required
    # editing the file -- the same shape as --seed being unreachable in the harness whose purpose is
    # replication. The `none` arm must stay in any set: it is the constant-zero control whose GAP
    # must come back EXACTLY 0.0000, and it is the only check that the scorer itself is not broken.
    ap.add_argument('--arms', default=','.join(ARMS),
                    help='comma-separated arm names; must include `none`')
    a = ap.parse_args()
    ARMS = tuple(x.strip() for x in a.arms.split(',') if x.strip())
    if 'none' not in ARMS:
        raise SystemExit('FATAL: the `none` control must be present -- it is the only check that '
                         'this scorer is not silently broken (its GAP must be EXACTLY 0.0000).')
    if not a.score_only:
        launch(a.armdir, a.out, a.epochs, a.bs, a.seed, ARMS)
    tbl = {}
    for arm in ARMS:
        ck = os.path.join(a.out, 'ckpt_%s' % arm, 'ep%d.ckpt' % (a.epochs - 1))
        if not os.path.exists(ck):
            print('  %-9s NO CHECKPOINT at %s' % (arm, ck)); continue
        tbl[arm] = score(ck, os.path.join(a.armdir, '%s_valid.csv' % arm),
                         expect_epoch=a.epochs - 1, expect_total=a.epochs)
        r = tbl[arm]
        # THE CEILING TRAVELS WITH THE NUMBER, AND IT IS *THIS* DATASET'S CEILING.
        # Two separate fixes live here. (1) The "quote against the ceiling, never against the ~0.194
        # base loss" instruction used to exist only in docstring prose, so no printed line or JSON
        # reader carried it. (2) The ceiling it named was +0.0116 -- a CONSTANT measured on smoke3
        # at k=1.77. The ceiling is sum(log k)/tokens and therefore scales with k, so on the k8 data
        # this scorer is about to be pointed at it is ~3.3x larger. Dividing by the smoke3 constant
        # there would overstate every arm's percentage by the same factor: a bound imported from a
        # different population, which is the #179 shape. Now computed per dataset in score().
        # THE CEILING BOUNDS THE *HELP* TERM ONLY, NOT GAP_neg.
        # GAP_neg = loss(-c) - loss(c) decomposes as HELP + HARM, where HELP = GAP_zero =
        # loss(0) - loss(c) is what the true request buys over no request, and HARM is the extra
        # cost of an actively WRONG request. The sum(log #distinct d)/tokens bound is the entropy a
        # perfect USER recovers over a perfect IGNORER -- it bounds HELP. Nothing bounds HARM by
        # that argument: loss(-c) can be arbitrarily worse than loss(no request), and on k8 the harm
        # term is 57-62% of topodiam's GAP_neg. So dividing GAP_neg by the ceiling divides a
        # two-part quantity by a bound on one part, and the answer can exceed 100% without anything
        # being wrong. The percentage now uses GAP_zero, and GAP_neg is reported as what it is: the
        # DIRECTIONAL statistic, judged against its own permutation null, not against a ceiling.
        # THE PER-ARM CEILING IS A DIAGNOSTIC, NOT A CROSS-ARM DENOMINATOR. This block used to
        # divide by ceiling_identifiable_THIS_ARM, and that normaliser REWARDS NOISE. The arm
        # ceiling subtracts sibling ties, and jitter breaks ties: topojit is topodiam + U(-0.5,0.5),
        # provably information-preserving (round(topojit) == topodiam on 8956/8956 rows), yet its
        # arm ceiling is 1.000x generic against topodiam's 0.527 purely because no two floats
        # collide. Dividing each arm by its own ceiling therefore multiplied topodiam's percentage
        # by 1.90x relative to its own noise control and by 1.41x relative to extentdisc -- in the
        # direction of the filed conclusion, and on extentdisc it silently undid 1.41x of exactly
        # the resolution matching that control was built to supply. A normaliser under which adding
        # U(-2,2) to a channel makes it look BETTER USED cannot carry a cross-arm comparison.
        # So: the printed percentage uses the GENERIC ceiling, which is identical for all arms on
        # this data (same parents, same k) and is therefore a common yardstick. The arm ceiling
        # stays in the JSON and is printed BESIDE it as the resolution statement it actually is.
        # A per-arm denominator would need a tie structure SHARED by all arms (e.g. topodiam's tie
        # groups applied to every arm), which is not computable from one arm's CSV in this function.
        _c = r.get('ceiling_oracle_THIS_dataset')
        _ok = _c is not None and _c == _c and _c > 0
        _pct = (100 * r['GAP_zero'] / _c) if _ok else float('nan')
        _ca = r.get('ceiling_identifiable_THIS_ARM')
        if _ca is not None and _ca == _ca and _ok:
            # NAME THE ZERO CASE RATHER THAN SUBSTITUTING ANOTHER POPULATION'S BOUND. The old
            # fallback sent the `none` arm (ceil_arm == 0 exactly) to the generic 0.0383 and then
            # printed it under the label "THIS ARM's achievable ceiling" -- restoring, in a string,
            # the exact mislabelling the comment at :498 says it fixed.
            print('  %-9s   resolution: this channel identifies %.1f%% of the generic ceiling '
                  '(%s); the %% on the NEXT line BELOW is against the GENERIC ceiling so arms '
                  'are comparable'
                  % (arm, 100 * _ca / _c,
                     'ZERO by construction -- constant-zero conditioning' if _ca <= 0
                     else 'arm ceiling %+.4f' % _ca))
        # TWO CONDITIONS, NOT ONE. The old verdict was `GAP_neg - GAP_perm > 3*sd`, which is a
        # SIGNIFICANCE test whose implicit effect-size null is ratio = 1.0. That null is wrong (see
        # the yardstick block in score()): negation is a bigger perturbation than permutation, so a
        # pure magnitude-responder clears 1.0 by construction and clears 3*sd easily once the arm's
        # GAP is large. An arm must now be BOTH significantly above its permutation null AND above
        # its own measured magnitude-only yardstick. Recomputing the ratios from the nine result
        # dirs and the yardsticks from the arm CSVs, every existing verdict survives this --
        # topodiam 1.75-1.84 vs a 1.3305 yardstick, extent 0.88-1.32 vs ~1.377, resid 0.54-0.86 vs
        # ~1.372 -- so no conclusion moves. But extentdisc at seed 777 sits at 1.423 against a
        # yardstick of ~1.396-1.402, a margin of only 1.5-1.9%, where the old test called it a
        # clean pass at z=5.8. That cell is the reason to make the change rather than note it.
        # TWO HONESTY NOTES ON THOSE NUMBERS. (1) The verdicts were RECOMPUTED BY HAND from the
        # GAP fields, not read from disk -- until the `verdict` key added below, nothing on disk
        # carried one, so "re-reading the result dirs" overstated what was available. (2) The
        # yardstick is a MONTE CARLO estimate (200 permutation draws); two independent
        # recomputations with different RNG consumption gave extentdisc 1.3956 and 1.4020, so it is
        # good to ~0.5% and must not be quoted to four digits as if exact. extentdisc's margin sits
        # inside that band, which is precisely why that cell is called marginal rather than passing.
        _dlt = r.get('GAP_neg_MINUS_perm', float('nan'))
        _gp = r.get('GAP_perm_mean', float('nan'))
        _yd = r.get('magnitude_only_yardstick', float('nan'))
        _ratio = (r['GAP_neg'] / _gp) if (_gp == _gp and abs(_gp) > 1e-12) else float('nan')
        _sig = _dlt > 3 * r['GAP_perm_sd']
        _above = (_ratio == _ratio) and (_yd == _yd) and _ratio > _yd
        if _yd != _yd:
            _verdict = ('constant-zero channel -- no yardstick and no displacement to obey'
                        if abs(r['GAP_neg']) < 1e-12 else 'YARDSTICK UNAVAILABLE -- not ruling')
        elif _ratio != _ratio:
            # RATIO UNAVAILABLE IS NOT A VERDICT. An auditor checked the yardstick path and found
            # it withholds correctly, then pointed at the branch next door: with GAP_perm at or
            # near zero the RATIO is nan, so `_above` is False, and a SIGNIFICANT arm fell through
            # to "RESPONDS TO MAGNITUDE, NOT SIGN" -- a confident-sounding verdict manufactured out
            # of a missing number, on the arms where the denominator is weakest. Withhold instead.
            _verdict = ('DIRECTIONALITY RATIO UNAVAILABLE (GAP_perm is at/near zero, so '
                        'GAP_neg/GAP_perm is undefined) -- not ruling on this arm')
        elif _sig and _above:
            _verdict = 'STEERS (above its permutation null AND its magnitude-only yardstick)'
        elif _sig:
            _verdict = ('RESPONDS TO MAGNITUDE, NOT SIGN -- clears the permutation null but sits '
                        'at/below the yardstick a size-only responder already reaches')
        else:
            _verdict = 'AT/BELOW ITS OWN PERMUTATION NULL -- NOT a steering result'
        # PERSIST THE VERDICT. It was only ever printed, so all four branches of tonight's
        # withholding logic -- the entire point of the change -- lived in console scrollback and
        # NOTHING on disk carried them: an auditor loaded all 11 steer_gaps.json and found zero
        # `verdict` keys. That also made my own justifying comment ("re-reading the nine result
        # dirs on disk, every existing verdict survives this") unauditable, because there were no
        # verdicts on disk to re-read -- I had recomputed them by hand and described it as reading.
        # IS THIS VERDICT DECIDED BY THE EFFECT, OR BY A 20-SAMPLE VARIANCE ESTIMATE?
        # GAP_perm_sd is estimated from n_perm draws, so it carries its own relative standard error
        # of ~1/sqrt(2(n_perm-1)) -- 16.2% at n_perm=20. The 3-sigma bar therefore WOBBLES by ~16%
        # run to run independently of the effect.
        # AND HERE IS WHERE I GOT AHEAD OF THE DATA. An auditor showed that giving topodiamscale's
        # seed-777 cell the OTHER seed's sd (0.00010 instead of 0.00005) drops z from 4.3 to 2.3,
        # and I endorsed that as proof the verdict rides on estimation noise. Then I measured the
        # 2-seed sd RATIO across all 13 arms. At n_perm=20 the ratio of two sd estimates has RSE
        # ~23%, so a ratio under ~1.5 is noise. Ten arms sit at 1.03-1.42x, as claimed. THREE DO
        # NOT: extent 2.22x, topodiamscale 1.86x, topodiamshuffle 1.76x -- all ~4 sigma out. For
        # those three the two sds differ because the two TRAINED MODELS genuinely differ in
        # permutation sensitivity, not because 20 draws is few. So the cross-substitution is NOT a
        # clean hold-the-effect-fixed test on topodiamscale: it swaps in a different model's null
        # width. The flip is real; the explanation we both gave for it is not established.
        # Consistent with that, this flag does NOT fire on topodiamscale (2.7 bar-sigmas out). It
        # catches genuine threshold-proximity, which is a DIFFERENT failure mode from that cell's.
        # This does NOT touch the headline arms -- topodiam z=12.6-22.1 and extentdisc z=5.8-11.6
        # survive the same substitution -- so the flag is narrow by design: it fires only where the
        # verdict is inside the bar's own uncertainty. Raising n_perm is the real fix and must be
        # done in ONE pass over every dir, not mid-table, or the z column stops being comparable.
        # TWO AXES, BECAUSE THE VERDICT IS A CONJUNCTION. It requires significance vs the
        # permutation null AND ratio > yardstick, but the first version of this flag measured only
        # the significance axis -- so it was loud on a cell already ruled negative and SILENT on
        # extentdisc@s777, the single cell my own comment names as the reason the flag exists
        # (ratio 1.4228 vs yardstick 1.3964, +1.89%). A guard that cannot protect the case it was
        # built for is the pattern this file has now shipped several times; both margins are
        # computed and OR-ed.
        _rse = (1.0 / math.sqrt(2.0 * (r['n_perm'] - 1))) if r.get('n_perm', 0) > 1 else float('nan')
        _bar = 3.0 * r['GAP_perm_sd']
        _bar_sigmas = ((_dlt - _bar) / (_bar * _rse)) if (_bar > 0 and _rse == _rse) else float('nan')
        # THE DENOMINATOR HERE IS THE RATIO'S OWN SAMPLING sd, NOT THE YARDSTICK'S MC ERROR.
        # My first version of this divided by the se of the 200-draw yardstick mean, which is a
        # number I can shrink to nothing just by taking more draws -- it reported extentdisc@s777
        # as 13.4 sigmas clear of its yardstick. The uncertainty that actually governs `ratio >
        # yardstick` is the cluster bootstrap on GAP_neg, and by THAT measure the same cell is
        # 0.14 sd above its yardstick, i.e. a coin flip. Measuring against the reducible error
        # instead of the irreducible one is how a marginal cell gets an unqualified verdict.
        _yse = r.get('magnitude_only_yardstick_se', float('nan'))   # MC only; recorded, not used here
        _lo, _hi = (r.get('GAP_neg_ci95') or [float('nan')] * 2)[:2]
        _sd_gap = (_hi - _lo) / (2 * 1.96) if (_lo == _lo and _hi == _hi) else float('nan')
        _sd_ratio = (_sd_gap / r['GAP_perm_mean']) if (_sd_gap == _sd_gap
                                                       and r.get('GAP_perm_mean', 0) > 0) else float('nan')
        _y_sigmas = ((_ratio - _yd) / _sd_ratio) if (_yd == _yd and _sd_ratio == _sd_ratio
                                                     and _sd_ratio > 0 and _ratio == _ratio) else float('nan')
        r['directionality_ratio_sd'] = _sd_ratio
        r['verdict_bar_sigmas'] = _bar_sigmas
        r['verdict_yard_sigmas'] = _y_sigmas
        r['verdict_sd_rse'] = _rse
        _near_bar = _bar_sigmas == _bar_sigmas and abs(_bar_sigmas) < 2.0
        _near_yard = _y_sigmas == _y_sigmas and abs(_y_sigmas) < 2.0
        # THE QUALIFIER IS ITS OWN FIELD, NOT CONCATENATED INTO `verdict`.
        # It used to be appended to the same string, so any consumer testing
        # `verdict == 'STEERS (...)'` silently got False on exactly the cells that needed reading.
        r['verdict_qualifier'] = None
        if _near_bar or _near_yard:
            r['verdict_qualifier'] = (
                'MARGINAL: %s. Raise n_perm / draws before quoting this cell.'
                % (' and '.join(
                    ([('%.2f bar-sigmas from the 3-sigma significance threshold (the bar itself '
                       'has %.0f%% uncertainty at n_perm=%d)') % (_bar_sigmas, 100 * _rse,
                                                                  r['n_perm'])] if _near_bar else [])
                    + ([('%.2f bootstrap-sd from the yardstick threshold (ratio %.4f +- %.4f vs '
                         'yardstick %.4f)') % (_y_sigmas, _ratio, _sd_ratio, _yd)] if _near_yard else []))))
        r['verdict'] = _verdict
        r['directionality_ratio'] = _ratio
        r['verdict_inputs'] = dict(delta_vs_perm=_dlt, three_sd=3 * r['GAP_perm_sd'],
                                   yardstick=_yd, significant=bool(_sig), above_yardstick=bool(_above))
        print('  %-9s %s' % (arm, _verdict))
        print('  %-9s   DELTA vs own perm null %+.4f (3sd %.4f) | directionality ratio %.3f vs '
              'magnitude-only yardstick %.3f' % ('', _dlt, 3 * r['GAP_perm_sd'], _ratio, _yd))
        print('  %-9s loss %.4f | GAP_neg %+.4f [%+.4f, %+.4f] = %s | '
              'GAP_zero %+.4f | GAP_perm %+.4f +- %.4f (n=%d)'
              % (arm, r['loss_true'], r['GAP_neg'], r['GAP_neg_ci95'][0], r['GAP_neg_ci95'][1],
                 # THE LABEL FOLLOWS THE DENOMINATOR. I changed _c from the per-arm ceiling to the
                 # generic one and left this string reading "THIS ARM's achievable ceiling" --
                 # reproducing, one line below where I removed it, the exact mislabelling the fix
                 # was for. Caught on the first line of output, not by re-reading the diff.
                 ('GAP_zero is %+.0f%% of the GENERIC ceiling %+.4f (common to all arms)'
                  % (_pct, _c)) if _ok
                 else 'ceiling UNAVAILABLE (not quoting a % of an inherited constant)',
                 r['GAP_zero'], r['GAP_perm_mean'], r['GAP_perm_sd'], r['n_perm']))
        # BRANCH THE WORDING ON THE VALUE. This line used to read "row-level CI would be %.2fx too
        # narrow" unconditionally -- a conclusion hardcoded into a print before the number existed.
        # At 1.77 rows/cluster the design effect legitimately sits at ~1 and the live dry run
        # produced 0.93x and 0.97x, i.e. the CLUSTERED interval was NARROWER, the exact opposite of
        # what the sentence asserted. It should exceed 1 on the k8 geometry (~6.7 rows/cluster) but
        # the sentence must not presume the sign of a quantity it is reporting.
        _de_ = r['design_effect_ci_width_ratio']
        if not (_de_ == _de_):
            _de_s = 'design effect undefined (zero-width row interval)'
        elif _de_ > 1.0:
            _de_s = 'a row-level CI would be %.2fx TOO NARROW here' % _de_
        else:
            _de_s = ('clustering NARROWED the CI by %.2fx -- design effect at or below 1 at this '
                     'rows/cluster, which is the normal reading on a low-k set' % _de_)
        print('  %-9s   CI unit: %s -- %d rows in %d INDEPENDENT clusters (%.2f rows/cluster); %s'
              % ('', r['cluster_unit'], r['n_rows_scored'], r['n_clusters'],
                 r['rows_per_cluster'] or float('nan'), _de_s))
        # The restricted number is the one to read for "does the channel steer", and the pooled one
        # is the one to read for "how much loss does it move on this dataset". Printing both, with
        # the dilution factor, so neither can be quoted as the other. Per the docstring the pooled
        # figure is diluted ~2.07x by rows where the negated request has nowhere else to go.
        if r.get('n_redirecting'):
            # NO RATIO ON A NEAR-ZERO DENOMINATOR. The dry run printed "dilution -5.00x" and
            # "-0.02x" on arms whose pooled GAP is indistinguishable from zero -- which is EXACTLY
            # the error filed as #111 ("the 12.5x is a ratio with a near-zero denominator and the
            # difference-of-differences is t=-1.35"). I reproduced my own filed mistake inside a
            # print statement. The dilution factor is only meaningful when the denominator is
            # separated from zero, so it is gated on the pooled CI excluding zero and otherwise
            # reported as n/a rather than as a number that will be read as a 5x effect.
            _lo, _hi = r['GAP_neg_ci95']
            _sep = (_lo == _lo) and (_hi == _hi) and not (_lo <= 0 <= _hi)
            _dil = ('%.2fx' % (r['GAP_neg_redirecting'] / r['GAP_neg_rowsum'])) if _sep else \
                   'n/a (pooled GAP is not separated from zero)'
            print('  %-9s   redirecting-only %+.4f [%+.4f, %+.4f] on %d/%d rows (%.1f%%), '
                  'dilution %s'
                  % ('', r['GAP_neg_redirecting'], r['GAP_neg_redirecting_ci95'][0],
                     r['GAP_neg_redirecting_ci95'][1], r['n_redirecting'], r['n_rows_scored'],
                     100 * r['redirect_frac'], _dil))
    # THE PREVIOUS SELF-CHECK COULD NOT FIRE. It asserted the `none` arm's GAP_neg is 0 and called a
    # non-zero value proof the scorer is broken. But `none` is constant-zero-conditioned, so -c == c
    # by ARITHMETIC whether the transform is applied or not: simulating a silently-unapplied
    # transform gives 0.000000 for the none arm in BOTH worlds. It tested a property of the DATA and
    # never of the SCORER -- precisely the guard-that-cannot-fire pattern this project has shipped
    # three times (G7 twice, the dropped-row counter once). Replaced with an assertion on a
    # CONDITIONED arm that the tensor entering the model actually differs.
    # PERSIST BEFORE THE SELF-CHECK CAN RAISE. The FATAL below used to sit ABOVE json.dump, so a
    # trip -- including a FALSE trip, see the |d| selection in _negation_reaches_model -- discarded
    # all four arms AFTER they had been computed. Writing first costs nothing and makes the failure
    # recoverable by re-checking rather than re-scoring. The table is stamped with the self-check
    # verdict below, so an unverified table can never be mistaken for a verified one.
    # C5: THE `none` INVARIANT WAS PROSE, NOT A GUARD -- and the row is not exactly zero.
    # The docstring says in capitals "Any non-zero value in the none row means this script is
    # broken", and nothing anywhere checked it. Measured: seed1 shipped GAP_perm_mean 2.776e-17
    # (1 ulp at 0.1428 from averaging 20 bit-identical float64s; seed2 gives exactly 0.0). Benign,
    # but an invariant stated in capitals and never asserted is the guard-that-cannot-fire pattern
    # in its purest form. Tolerance is 1e-12: far above float noise, far below any real effect
    # (the smallest non-null arm measured here is +0.0007).
    json.dump(tbl, open(os.path.join(a.out, 'steer_gaps.json'), 'w'), indent=1)
    print('\nwritten -> %s/steer_gaps.json' % a.out)
    # ...AND THE `none` CHECK NOW SITS *BELOW* THAT DUMP. It was inserted between the comment above
    # and the dump it describes, which restored the exact regression that comment warns about one
    # function down: a trip discarded the whole scored table AFTER every arm had been computed.
    # RELABELLED, because the FATAL text was wrong about what it tests. `none_valid.csv` has d
    # identically 0 in all eight arm CSVs, so -0.0 == 0.0, zeros_like(c) == c, and permuting zeros
    # returns zeros: all three quantities are 0 BY ARITHMETIC ON THE DATA, in a working-scorer world
    # and in a silently-unapplied-transform world alike. That is verbatim the argument at :691 for
    # why the PREVIOUS none-arm check could not fire, so calling a trip here "this scorer is broken"
    # repeats the mislabelling rather than fixing it. It is a DATA-REGRESSION guard: it fires only
    # if none_valid.csv stops being constant-zero. The scorer itself is checked by
    # _negation_reaches_model below, which is the one that can actually distinguish those worlds.
    if 'none' in tbl:
        _n = tbl['none']
        _bad = [(k, _n[k]) for k in ('GAP_neg', 'GAP_zero', 'GAP_perm_mean')
                if abs(_n.get(k, 0.0)) > 1e-12]
        if _bad:
            raise SystemExit('FATAL: none_valid.csv is no longer constant-zero-conditioned -- %s '
                             'must be zero by ARITHMETIC on that data and is not: %s. This is a '
                             'DATA regression, not a scorer bug (the scorer check is the '
                             'negation-reaches-model self-check). The table WAS written first.'
                             % (', '.join(k for k, _ in _bad), _bad))
    _sc_fail = []
    for _arm in [x for x in ARMS if x != 'none']:
        _ck = os.path.join(a.out, 'ckpt_%s' % _arm, 'ep%d.ckpt' % (a.epochs - 1))
        if not os.path.exists(_ck):
            continue
        _d = _negation_reaches_model(_ck, os.path.join(a.armdir, '%s_valid.csv' % _arm))
        print('  SELF-CHECK %s: max|c-(-c)| %.4f | max|dlogit| %.4f' % (_arm, _d['dc'], _d['dlogit']))
        if _arm in tbl:
            tbl[_arm]['self_check'] = dict(dc=_d['dc'], dlogit=_d['dlogit'],
                                           passed=bool(_d['dc'] >= 1e-9 and _d['dlogit'] >= 1e-9))
        if _d['dc'] < 1e-9 or _d['dlogit'] < 1e-9:
            _sc_fail.append((_arm, _d))
    # NO `break` HERE. It used to stop after the first conditioned arm that existed, so `extent`
    # was instrumented and `topodiam`/`resid` never were -- two thirds of the conditioned arms
    # carried an unchecked negation. Three extra forward passes on a 16-row batch.
    json.dump(tbl, open(os.path.join(a.out, 'steer_gaps.json'), 'w'), indent=1)   # re-write with verdicts
    if _sc_fail:
        raise SystemExit('FATAL: negation does not reach the model for %s. Every GAP_neg for those '
                         'arms would be a no-op artefact. The table WAS written (with '
                         'self_check.passed=false) so this is re-checkable without re-scoring.'
                         % ', '.join('%s (dc=%.2e, dlogit=%.2e)' % (n, d['dc'], d['dlogit'])
                                     for n, d in _sc_fail))
    # A SCORING RUN THAT FOUND NOTHING MUST NOT EXIT 0. The previous version printed four
    # "NO CHECKPOINT" lines, wrote `{}`, and returned 0 -- indistinguishable from success to any
    # caller checking $?, in the file whose own title is "a launcher that cannot hide a FATAL".
    if len(tbl) < len(ARMS):
        raise SystemExit('FATAL: scored %d of %d arms. An incomplete table is not a result.'
                         % (len(tbl), len(ARMS)))
