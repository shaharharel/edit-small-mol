"""THE DETERMINISTIC STEERING GAP -- the number the v3 arms were never able to emit.

WHY THIS EXISTS. Every conditioned Phase A arm runs mode='geom', and train_phaseA.py gates its
deterministic control to mode=='role' (line ~440), so `gap_det` is NaN in all 13 epochs on disk.
That leaves `gap` -- the within-batch permutation control -- as the only steering number those arms
produce, and train_phaseA.py:414 says in terms:

    "Do not quote `gap` as an effect size."

So the five completed v3 arms are currently LOSS CURVES, not steering measurements. That is a
scope statement I should have made myself and did not; an auditor had to point it out.

WHY THE PERMUTATION CONTROL IS DILUTED, AND BY EXACTLY HOW MUCH. randperm within a batch returns a
row its OWN d with probability sum_c p_c^2. On a balanced BINARY channel that is 0.5^2 + 0.5^2 =
0.5, so half the rows contribute exactly zero and the reported GAP is ~2x too SMALL. An auditor
measured the deterministic flip at +0.051539 against the permutation's +0.023833 at ep1 -- ratio
2.1625 against a predicted 2.0000. The effect has been UNDERSTATED all night, not overstated.

WHAT THIS COMPUTES INSTEAD. The deterministic wrong-instruction control: feed every row the
*wrong* d and measure how much worse the model does.
  * BINARY d in {-1,+1}      -> the wrong value is unique: flip the sign. GAP_flip = loss(-d) - loss(d).
  * THREE-VALUE d {-1,0,+1}  -> there are TWO wrong values per row, so a sign flip is no longer
                               "the" wrong answer. We average over both, which is the natural
                               generalisation and reduces to the flip when the third class is absent.
No sampling, so nothing to replicate -- the same checkpoint gives the same number every time. That
is the whole point: #113 forbids the single-draw estimator for LEVELS, and this removes the draw.

WHAT IT IS STILL NOT. This is a DEPENDENCE measure -- how much the model leans on the channel --
not a BENEFIT. A model that ignores d entirely scores 0 here, but a model that uses d cannot be
assumed to use it WELL; obedience on generations is a separate measurement against the descriptor
floor. #98 and #106 are the standing warnings: dependence is not benefit, and a two-model contrast
is structurally incapable of returning zero.
"""
import os, sys, json, argparse, hashlib
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from train_phaseA import PhaseA, RoleData, collate, PRIOR  # noqa: E402
from reinvent.models.transformer.core.vocabulary import Vocabulary, SMILESTokenizer  # noqa: E402


def batch_loss(model, bt, dev, cond_override=None):
    src, sm, tgt, cond, _ = bt
    src, sm, tgt, cond = src.to(dev), sm.to(dev), tgt.to(dev), cond.to(dev)
    if cond_override is not None:
        cond = cond_override(cond)
    tin, tout = tgt[:, :-1], tgt[:, 1:]
    K = tin.size(1)
    causal = torch.triu(torch.ones(K, K, dtype=torch.bool, device=dev), 1).logical_not()
    tmask = (tin != 0).unsqueeze(1) & causal
    with torch.no_grad():
        lp = model.generator(model.forward_cond(src, sm, tin, tmask, cond))
        per = torch.nn.functional.nll_loss(lp.reshape(-1, lp.size(-1)), tout.reshape(-1),
                                           reduction='none').view(tout.shape)
        m = (tout != 0).float()
        return float((per * m).sum()), float(m.sum())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', required=True, help='runs/<arm> directory')
    ap.add_argument('--data', required=True, help='arm data dir with valid.csv')
    ap.add_argument('--prior', default=PRIOR)
    ap.add_argument('--bs', type=int, default=64)
    ap.add_argument('--max-batches', type=int, default=0, help='0 = all; else a REPORTED cap')
    a = ap.parse_args()

    dev = torch.device('mps' if torch.backends.mps.is_available() else 'cpu')
    ck0 = torch.load(a.prior, map_location='cpu', weights_only=False)
    _v = ck0['vocabulary']
    vocab = _v if isinstance(_v, Vocabulary) else Vocabulary(
        _v['tokens'] if isinstance(_v, dict) and 'tokens' in _v else _v)
    npm = dict(ck0['network_parameter'])
    _vp = os.path.join(a.data, 'valid.csv')
    va = RoleData(_vp, vocab, SMILESTokenizer(), 'geom')
    va_sha = hashlib.sha256(open(_vp, 'rb').read()).hexdigest()[:16]

    dvals = sorted({float(r.get('d', 0.0)) for r in va.rows})
    # THE "AVERAGE OVER EVERY OTHER VALUE" RULE DEGENERATES ON A CONTINUOUS CHANNEL, AND THE
    # DOCSTRING ONLY EVER CONTEMPLATED 2 OR 3 VALUES. attach_path_continuous/valid.csv carries 33
    # distinct d and attach_flex_continuous 14. Under the substitution rule every row would get 32
    # (resp. 13) "wrong" values, most of them a hair from the row's own d -- so GAP_det collapses
    # toward zero BY CONSTRUCTION. That is the same dilution this file was written to remove,
    # reintroduced one level further down, which is the third time tonight a fix has recreated its
    # own bug at the next level. Cost blows up too: 2 forward passes per (batch, value) is 66 per
    # batch instead of 4.
    # SO: few-valued channels keep the exhaustive per-row rule (it IS the right generalisation of
    # the flip there); a many-valued channel uses the SIGN FLIP d -> -d, which is the natural
    # "wrong instruction" for a signed magnitude and does not shrink with the number of levels.
    # Rows at d == 0 are EXCLUDED from the flip arm rather than silently counted, because -0.0 == 0.0
    # makes them an exact no-op that would dilute the denominator -- the same defect an auditor
    # found in GAP_neg this hour.
    # THE THRESHOLD OF 4 SPLIT THE MATRIX ALONG THE CONTRAST THE MATRIX EXISTS TO MEASURE.
    # Measured d-levels on v4e valid: binary 3, wclass 2, bin 5, flex_continuous 14,
    # path_continuous 33. So `> 4` did not separate "few-valued" from "continuous" -- it separated
    # `binary` from `bin`, and adapt_steer_v2's own docstring says those two "differ only in
    # RESOLUTION, which is the contrast the 18-arm matrix is meant to measure". Scoring them under
    # two different estimators would have made that row of the table meaningless, in two separable
    # ways an auditor laid out: different POPULATIONS (the flip arm drops d==0, 33.33% of every
    # attach_* valid set; the exhaustive arm keeps them) and different CORRUPTION MAGNITUDE (on a
    # 3-level channel half the exhaustive substitutions are "-> 0", i.e. "hold", a strictly milder
    # wrong instruction than a sign flip).
    # FIX: do not choose. The SIGN FLIP is computed for EVERY arm -- it is defined on any signed
    # channel, costs the same everywhere, and is therefore the one statistic that is comparable
    # across the whole matrix. The EXHAUSTIVE rule is computed as well wherever it is affordable
    # (<= 5 levels), because on a 3-level channel it is the more complete corruption. Both are
    # stamped with their own effective row count, and a comparison is only legitimate WITHIN one
    # rule. Quote gap_det_flip for cross-arm tables.
    FEW = len(dvals) <= 5
    n_zero = sum(1 for r in va.rows if abs(float(r.get('d', 0.0))) < 1e-12)
    print('arm: %s' % os.path.basename(a.run))
    print('  valid rows %d | distinct d values %d %s'
          % (len(va.rows), len(dvals), dvals if len(dvals) <= 6 else
             '[%.4f .. %.4f]' % (dvals[0], dvals[-1])))
    print('  rules: SIGN FLIP (all arms, the comparable one)%s'
          % ('  +  EXHAUSTIVE substitution over the other %d value(s)' % (len(dvals) - 1)
             if FEW else '  (exhaustive skipped: %d levels is too many to afford)' % len(dvals)))
    if n_zero:
        print('  d==0 rows: %d (%.2f%%) -- EXCLUDED from the flip arm, they are an exact no-op'
              % (n_zero, 100.0 * n_zero / max(len(va.rows), 1)))

    cks = sorted(f for f in os.listdir(a.run) if f.endswith('.ckpt'))
    if not cks:
        print('  no checkpoints')
        return 2
    print('  %-6s %10s  %-28s %-28s %s' % ('epoch', 'loss(d)', 'GAP_det SIGN-FLIP', 'GAP_det EXHAUSTIVE', 'note'))
    out = {}
    for f in cks:
        ck = torch.load(os.path.join(a.run, f), map_location='cpu', weights_only=False)
        st = ck.get('model_state')
        mode = ck.get('mode')
        if mode != 'geom':
            print('  %-6s SKIP -- mode=%r has no d channel to corrupt' % (f[:-5], mode))
            continue
        # THE ARM'S OWN ARCHITECTURE, NOT THE PRIOR'S. This previously built PhaseA from the
        # PRIOR checkpoint's network_parameter while reading `mode` from the ARM's. num_heads and
        # dropout change NO tensor shape, so a mismatch loads cleanly, `missing` comes back empty,
        # and the FATAL guard below cannot fire -- a model trained at 8 heads would be evaluated
        # at whatever the prior says. That is the cond_repeat failure exactly (#199, where a probe
        # silently measured a RANDOM geom_mlp). Equal on every arm today; wired so it stays true.
        _npm = dict(ck.get('network_parameter') or npm)
        if _npm != npm:
            print('  %-6s NOTE: arm architecture differs from the prior: %s'
                  % (f[:-5], {k: (npm.get(k), v) for k, v in _npm.items() if npm.get(k) != v}))
        # AND THE valid_sha GUARD IS ACTUALLY CHECKED. train_phaseA stamps it so a scorer can
        # catch a checkpoint scored against the wrong directory's valid.csv -- which it says
        # "passes silently and returns a plausible wrong GAP". This file took --run and --data as
        # independent free text and compared nothing.
        _vs = ck.get('valid_sha')
        if _vs and va_sha and _vs != va_sha:
            print('  %-6s FATAL: checkpoint valid_sha %s != --data valid.csv %s.'
                  ' This checkpoint was trained against a DIFFERENT valid set;'
                  ' any GAP_det here would be a plausible wrong number.' % (f[:-5], _vs, va_sha))
            return 2
        m = PhaseA(mode='geom', **_npm)
        missing, unexpected = m.load_state_dict(st, strict=False)
        if missing:
            print('  %-6s FATAL: %d tensors missing -> would be RANDOM: %s'
                  % (f[:-5], len(missing), list(missing)[:4]))
            return 2
        m.to(dev).eval()
            # SEEDED SHUFFLE, so a CAP is a subsample rather than a biased PREFIX.
        # valid.csv is not randomised and the loader was shuffle=False, so --max-batches took the
        # FIRST N batches. Measured on attach_path/valid.csv: first 1,280 rows are 55.86% d=+1,
        # first 7,680 are 52.92%, the full 23,238 are 50.15% -- a 5.7-point class skew at the
        # smallest cap, plus a mean-target-length drift 58.7 -> 57.2. So every capped artifact was
        # a biased prefix ON TOP OF being smaller, and that is a candidate source of the 2.2%
        # disagreement between my capped +0.049372 and an independently-derived +0.051539.
        # Full passes are unaffected (same rows, different order); only capped runs change.
        _g = torch.Generator(); _g.manual_seed(20260915)
        dl = torch.utils.data.DataLoader(va, batch_size=a.bs, shuffle=True, generator=_g,
                                         collate_fn=collate)
        s_ok = n_ok = 0.0
        # PER-ROW WRONG VALUES, AND THE ROW'S OWN VALUE IS GENUINELY EXCLUDED.
        # My first version substituted each possible d GLOBALLY and averaged, so for a row whose
        # true d is +1 the "wrong" set included +1 itself -- a NO-OP. On a binary channel that made
        # bad = (flip + base)/2, i.e. GAP = GAP_flip / 2. Measured +0.024686 against an auditor's
        # independently-derived flip of +0.051539: ratio 2.088, exactly the predicted halving.
        # It is the SAME dilution this file was written to remove, reintroduced one level down --
        # and the old code carried a comment claiming the exclusion while the guard it referred to
        # (`if abs(v - 0.0) >= 0`) is true for every real number and filtered nothing.
        # The fix masks per row: substitute v only where the row's own d differs from v, and
        # accumulate token counts only for those rows, so the denominator matches the numerator.
        s_bad = 0.0; n_bad = 0.0      # exhaustive rule
        s_flip = 0.0; n_flip = 0.0    # sign-flip rule, computed for EVERY arm
        nb = 0
        for bt in dl:
            if bt is None:
                continue
            a_, b_ = batch_loss(m, bt, dev)
            s_ok += a_; n_ok += b_
            cond0 = bt[3]
            # RULE 1, ALWAYS: SIGN FLIP, on the rows the flip actually moves.
            sel = cond0[:, 0].abs() > 1e-12
            if bool(sel.any()):
                sub = [bt[0][sel], bt[1][sel], bt[2][sel], cond0[sel], bt[4][sel]]
                sv, nv = batch_loss(m, sub, dev,
                                    lambda c: torch.cat([-c[:, :1], c[:, 1:]], 1))
                so, _ = batch_loss(m, sub, dev)   # matched baseline on the SAME rows
                s_flip += (sv - so); n_flip += nv
            # RULE 2, ONLY WHERE AFFORDABLE: exhaustive per-row substitution.
            if FEW:
                for v in dvals:
                    sel = ~torch.isclose(cond0[:, 0], torch.tensor(float(v)))
                    if not bool(sel.any()):
                        continue
                    sub = [bt[0][sel], bt[1][sel], bt[2][sel], cond0[sel], bt[4][sel]]
                    sv, nv = batch_loss(m, sub, dev,
                                        lambda c, _v=v: torch.cat(
                                            [torch.full_like(c[:, :1], _v), c[:, 1:]], 1))
                    so, _ = batch_loss(m, sub, dev)   # matched baseline on the SAME rows
                    s_bad += (sv - so); n_bad += nv
            nb += 1
            if a.max_batches and nb >= a.max_batches:
                break
        base = s_ok / max(n_ok, 1)
        # Each rule's delta is a mean over ITS OWN rows. `loss_wrong` used to be
        # base + delta, mixing a token-mean over ALL rows with a delta over a SUBSET -- a column
        # that is not the loss of any population. It is dropped; the deltas ARE the gaps.
        gap_flip = (s_flip / n_flip) if n_flip else float('nan')
        gap_exh = (s_bad / n_bad) if n_bad else float('nan')
        bad = base + gap_exh
        note = 'batches=%d%s' % (nb, '  (CAPPED, not full valid)' if a.max_batches else '')
        print('  %-6s %10.6f  flip %+9.6f (n=%9.0f)  exh %+9.6f (n=%9.0f)  %s'
              % (f[:-5], base, gap_flip, n_flip, gap_exh, n_bad, note))
        out[f[:-5]] = dict(loss=base,
                           gap_det_flip=gap_flip, n_tokens_flip=n_flip,
                           gap_det_exhaustive=gap_exh, n_tokens_exhaustive=n_bad,
                           gap_det=gap_exh if FEW else gap_flip, batches=nb)
    # A CAPPED RUN MUST NOT CLOBBER A FULL ONE. I proved this the expensive way sixty seconds ago:
    # a --max-batches 2 GUARD TEST overwrote v3_wclass's full-valid artifact (+0.003949 /
    # +0.018792 / +0.027155 over 131 batches) with a 2-batch number, and the remote runner SKIPS
    # any run that already has this file -- so the capped result would have been silently adopted
    # as the arm's answer. Capped runs now write their own filename and can never replace the
    # full-valid artifact.
    _name = ('gap_deterministic.json' if not a.max_batches
             else 'gap_deterministic_cap%d.json' % a.max_batches)
    _path = os.path.join(a.run, _name)
    json.dump(dict(run=a.run, data=a.data, d_values=dvals, max_batches=a.max_batches,
                   rules='sign_flip(all) + exhaustive(<=5 levels)',
                   primary='exhaustive' if FEW else 'sign_flip',
                   comparable_across_arms='gap_det_flip',
                   n_zero_d_rows=n_zero, valid_sha=va_sha, epochs=out),
              open(_path, 'w'), indent=2)
    print('  wrote %s' % _path)
    return 0


if __name__ == '__main__':
    sys.exit(main())
