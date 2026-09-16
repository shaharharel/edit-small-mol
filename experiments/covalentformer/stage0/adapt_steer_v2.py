"""THE THREE STEERING ENCODINGS: binary, bin, continuous.

WHY v2 EXISTS. adapt_steer_to_phaseA.py maps every direction to d = +-1.0 and throws the MAGNITUDE
away. The magnitude is on disk -- every row carries `<param>_a` (the input's own value) and
`<param>_b` (the output's) -- so "move path DOWN by 2" was recoverable all along and was being
discarded at the last step. Only the `binary` arm of the 18-arm matrix could be built from the old
adapter; this file builds all three.

THE ENCODINGS, all RELATIVE (the user's call, and the right one to start from -- an absolute target
requires the model to know the input's value, which is a second problem stacked on the first):
  binary      d = +1 UP / -1 DOWN.                      "move it up"
  continuous  d = (b - a) / sd(TRAIN).  SCALE ONLY.   "move it up by 2"
  bin         d = signed ordinal over quantile bins of  "move it up a lot / a little"
              the signed delta, rescaled to ~[-1, 1].

WHY continuous IS Z-SCORED AND bin IS RESCALED. The conditioning vector is fed to an untrained
nn.Linear(2, 64). A raw delta of +7 and a binary +-1 are not on the same scale, so an unnormalised
continuous arm would differ from the binary arm in INPUT SCALE as well as in ENCODING, and the
comparison between them -- which is the entire point of running all three -- would be confounded.
This project has already measured that scale matters: z-scoring the Phase B geometry input helped
6/6 seeds (#134), and the follow-up showed it improved OPTIMISATION rather than informativeness
(#138). So all three encodings are put on a comparable scale ON PURPOSE, and the z-statistics are
computed on TRAIN and APPLIED to valid -- never refit on valid, which would leak.

wclass IS BINARY-ONLY AND THAT IS NOT AN OVERSIGHT. Its values are warhead-class STRINGS
('acrylamide', 'nitrile', ...). There is no ordering on them, so `b - a` is undefined and both
`bin` and `continuous` are meaningless. wclass contributes ONE arm, not three. Fabricating an
integer code for a categorical and calling it continuous would produce a number that looks like the
other arms and means nothing.

THE ARM MATRIX THIS SUPPORTS (18 total, which is where the 18 comes from):
  attach_path  binary + bin + continuous   3
  attach_flex  binary + bin + continuous   3
  elec_path    binary + bin + continuous   3   <- UNDERPOWERED, 1,320 train rows
  elec_flex    binary + bin + continuous   3   <- UNDERPOWERED,   546 train rows
  wclass       binary only                 1
  unconditioned control, one per param     5
The two elec_* params are built and run because the plan asks for them, but 546 and 1,320 rows
cannot resolve an effect of the size attach_path is showing (~1e-4) and their numbers must be read
as such. That is stated here so it is not discovered later as a silent gap.
"""
import os, sys, csv, json, argparse
import numpy as np

UP = ('UP', 'CHANGED')
NUMERIC = ('attach_path', 'attach_flex', 'elec_path', 'elec_flex')
CATEGORICAL = ('wclass',)
ENCODINGS = ('binary', 'bin', 'continuous')


def read(path, param):
    """Rows carrying a usable label, with the signed delta where one is defined."""
    dk, ak, bk = param + '_dir', param + '_a', param + '_b'
    out = []
    for line in open(path):
        r = json.loads(line)
        d = r.get(dk)
        if d is None:
            continue
        delta = None
        if param in NUMERIC:
            a, b = r.get(ak), r.get(bk)
            if isinstance(a, (int, float)) and isinstance(b, (int, float)):
                delta = float(b) - float(a)
        out.append(dict(anchor=r['input_smiles'], target=r['output_smiles'],
                        dir=d, delta=delta))
    return out


def encode(rows, enc, stats):
    """Return the conditioning scalar per row, or None to drop the row."""
    if enc == 'binary':
        # SAME MUST BE 0.0, NOT -1.0. The original `1.0 if dir in UP else -1.0` sent SAME to the
        # same value as DOWN, which (a) destroyed MULTIPLIER 1 -- the entire reason for keeping the
        # 585,342 SAME rows was to add a "hold this value" instruction, and it was being taught as
        # "decrease it"; and (b) falsified my own claim that a three-class corpus escapes QA-56's
        # GAP_neg/GAP_perm = 1/P identity. That escape needs d to actually take three values. With
        # SAME folded onto DOWN the binary arm's d is STILL BINARY and the identity STILL APPLIES.
        # Verified on disk before fixing: attach_path_binary/train.csv had exactly two distinct d
        # values, -1.000000 and 1.000000.
        # TWO-CLASS CATEGORICAL GETS {-1,+1}; A NUMERIC PARAM KEEPS SAME -> 0.
        # I stamped "wclass 2 levels {-1,+1}" one tick ago having printed only the level COUNT and
        # never the VALUES. Measured after an auditor pushed back:
        #     v4e arms/wclass_binary/train.csv   25958 at d=+1.0 (CHANGED) + 25958 at d= 0.0 (SAME)
        #     v3  phaseA_csv/wclass/train.csv    16606 at d=+1.0 (CHANGED) + 16887 at d=-1.0 (SAME)
        # so the stamp was false and the two corpora disagreed on the encoding with nothing
        # recording it. THREE reasons the categorical case takes {-1,+1} rather than {0,+1}:
        #   1. GAP_neg = loss(-d) - loss(d) and -0.0 == 0.0, so EVERY SAME row is an exact
        #      arithmetic no-op: zero numerator, full denominator. On wclass that is 50.00% of
        #      valid (6,859/13,718) -- the headline steering statistic would be understated 2.00x
        #      on the one arm where the instruction is cleanest.
        #   2. d = 0 is the exact constant the mode='none' / A0 control feeds. Encoding a real
        #      instruction as the value that means "no instruction" makes SAME and the control
        #      indistinguishable to the model.
        #   3. v3's wclass arm used {-1,+1}. Keeping it means the v3-vs-v4e wclass contrast is an
        #      experiment about the CORPUS, not about an encoding change no stamp recorded.
        # A NUMERIC param is different and keeps SAME -> 0.0: there "hold this value" is a genuine
        # midpoint between UP and DOWN, and 0 is where it belongs. A 2-class categorical has no
        # midpoint -- SAME is the opposite POLE of CHANGED, not the middle of an axis.
        if set(r['dir'] for r in rows) <= {'CHANGED', 'SAME'}:
            return [1.0 if r['dir'] == 'CHANGED' else -1.0 for r in rows]
        return [1.0 if r['dir'] in UP else (-1.0 if r['dir'] == 'DOWN' else 0.0) for r in rows]
    if enc == 'continuous':
        # SCALE ONLY -- DO NOT SUBTRACT THE MEAN. This is the third instance of the same bug.
        # Full z-scoring, (delta - mu)/sd, moves the NEUTRAL POINT off zero whenever mu != 0, and
        # mu is NOT zero here (attach_path +0.2164, attach_flex +0.2464 -- the corpus has more UP
        # than DOWN). So every delta==0 row, the "hold this value" instruction, was handed
        #     attach_path  -mu/sd = -0.126526   131,057 rows (35.69%)
        #     attach_flex  -mu/sd = -0.189568    94,356 rows (36.06%)
        # a NEGATIVE value -- which in the binary and bin encodings means DOWN. Verified from the
        # CSVs, not inferred: every SAME row in attach_path_continuous/train.csv carries exactly
        # -0.126526. So `binary` said hold=0, `bin` said hold=0, and `continuous` said hold=DOWN,
        # and my docstring claimed all three sign conventions "match exactly". They did not.
        # Dividing by sd alone keeps the three encodings on a comparable scale -- the reason the
        # normalisation exists at all (#134/#138) -- while mapping 0 -> 0 by construction.
        sd = stats['sd']
        return [None if r['delta'] is None else r['delta'] / sd for r in rows]
    # QUANTILE EDGES ARE THE WRONG TOOL FOR THIS DELTA AND PRODUCED A DEGENERATE ENCODING.
    # `delta` is a small signed integer whose MODE is 0 -- 131,057 of 367,161 attach_path rows
    # (35.7%). np.percentile on that mass returns DUPLICATE edges: measured [-1.0, 0.0, 0.0, 1.0].
    # Two identical edges make the bin between them unreachable (bin index 2 had ZERO rows), and
    # the surviving values were {-1.0, -0.5, +0.5, +1.0} -- THERE WAS NO ZERO. So the one
    # instruction the SAME rows exist to express, "hold this value", was unrepresentable, and every
    # delta==0 row was handed +0.5, which is an UP value. The `bin` arm was teaching "hold" as
    # "increase slightly".
    # FIXED ENCODING: explicit signed bins with zero as its own level, which is what the
    # "move a lot / move a little / hold" semantics actually needs and what the integer supports.
    #   delta <= -2  -> -1.0   (down a lot)
    #   delta == -1  -> -0.5   (down a little)
    #   delta ==  0  ->  0.0   (HOLD)
    #   delta == +1  -> +0.5   (up a little)
    #   delta >= +2  -> +1.0   (up a lot)
    # Symmetric by construction, on the same [-1,1] scale as `binary`, and the sign convention
    # matches it exactly so the two arms differ only in RESOLUTION, which is the contrast the
    # 18-arm matrix is meant to measure.
    vals = []
    for r in rows:
        d = r['delta']
        if d is None:
            vals.append(None)
        elif d <= -2:
            vals.append(-1.0)
        elif d == -1:
            vals.append(-0.5)
        elif d == 0:
            vals.append(0.0)
        elif d == 1:
            vals.append(0.5)
        else:
            vals.append(1.0)
    return vals


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--split-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--nbins', type=int, default=5)
    a = ap.parse_args()

    params = sorted({f.rsplit('_', 1)[0] for f in os.listdir(a.split_dir)
                     if f.endswith('_train.jsonl') or f.endswith('_valid.jsonl')})
    meta = {}
    print('%-13s %-11s %8s %8s %8s   %s' % ('param', 'encoding', 'train', 'valid', 'dropped', 'note'))
    for p in params:
        tr = read(os.path.join(a.split_dir, '%s_train.jsonl' % p), p)
        va = read(os.path.join(a.split_dir, '%s_valid.jsonl' % p), p)
        if not tr:
            print('%-13s SKIP -- no rows' % p)
            continue

        encs = ENCODINGS if p in NUMERIC else ('binary',)
        # TRAIN-ONLY statistics. Refitting these on valid would leak the valid distribution into
        # the conditioning signal and inflate every downstream number.
        d_tr = np.array([r['delta'] for r in tr if r['delta'] is not None], dtype=float)
        stats = {}
        if len(d_tr):
            sd = float(d_tr.std())
            stats['mu'] = float(d_tr.mean())
            # a zero sd would make every continuous value inf/nan while still writing a valid-
            # looking CSV, so it is caught here rather than surfacing as a dead arm later
            stats['sd'] = sd if sd > 1e-9 else 1.0
            stats['sd_degenerate'] = sd <= 1e-9
            qs = np.linspace(0, 100, a.nbins + 1)[1:-1]
            stats['edges'] = [float(x) for x in np.percentile(d_tr, qs)]

        for enc in encs:
            if enc != 'binary' and not len(d_tr):
                print('%-13s %-11s   -- no numeric delta available, skipped' % (p, enc))
                continue
            d = os.path.join(a.out, '%s_%s' % (p, enc))
            os.makedirs(d, exist_ok=True)
            counts = {}
            for nm, rows in (('train', tr), ('valid', va)):
                vals = encode(rows, enc, stats)
                keep = [(r, v) for r, v in zip(rows, vals) if v is not None]
                counts[nm] = (len(keep), len(rows) - len(keep))
                with open(os.path.join(d, '%s.csv' % nm), 'w', newline='') as fh:
                    w = csv.writer(fh)
                    w.writerow(['anchor', 'target', 'd', 'cos_theta', 'dir_label'])
                    for r, v in keep:
                        w.writerow([r['anchor'], r['target'], '%.6f' % v, '0.0', r['dir']])
            note = ''
            if enc != 'binary':
                note = 'mu=%.3f sd=%.3f' % (stats['mu'], stats['sd'])
                if stats.get('sd_degenerate'):
                    note += '  WARNING: sd was 0, forced to 1'
            if counts['train'][0] < 5000:
                note += '  UNDERPOWERED'
            print('%-13s %-11s %8d %8d %8d   %s'
                  % (p, enc, counts['train'][0], counts['valid'][0], counts['train'][1], note))
            meta['%s_%s' % (p, enc)] = dict(
                param=p, encoding=enc,
                n_train=counts['train'][0], n_valid=counts['valid'][0],
                dropped_train=counts['train'][1],
                mu=stats.get('mu'), sd=stats.get('sd'), edges=stats.get('edges'))

    os.makedirs(a.out, exist_ok=True)
    # cos_theta IS A DEAD COLUMN AND THAT IS NOW STAMPED. Every arm this file writes emits
    # cos_theta = 0.0 on every row (verified on all five v4e arms: exactly ONE distinct value,
    # 314,061/314,061 on attach_path, 217,245/217,245 on attach_flex, 51,916/51,916 on wclass).
    # train_phaseA feeds the pair [d, cos_theta] to nn.Linear(2, 64), so HALF the conditioning
    # width carries nothing and the whole steering signal is d. That is not a bug -- none of these
    # six params has an angular component, which is exactly why the pose-dependent three (extent,
    # phi, exitang) are a separate build -- but an arm trained here must NOT be described as
    # "geometry-conditioned". It is conditioned on ONE scalar. The number of levels that scalar
    # takes is the honest resolution of the arm and is recorded per arm below:
    #   binary      3 levels  {-1, 0, +1}        (an auditor called this "one bit"; with SAME
    #                                             retained it is three-valued, not two)
    #   bin         5 levels  {-1,-.5,0,+.5,+1}
    #   continuous  38 distinct on attach_path, 16 on attach_flex  (MEASURED, not estimated)
    #   wclass      2 levels  {-1, +1}           -- one bit, CHANGED vs SAME. TRUE ONLY SINCE THE
    #                                             categorical branch in encode(); the arms built
    #                                             before it carry {0,+1} and must be rebuilt.
    # EVERY LEVEL COUNT AND VALUE SET ABOVE WAS PRINTED FROM THE ARM CSVs. The previous version of
    # this block asserted wclass was {-1,+1} on the strength of a level COUNT of 2, and the values
    # were actually {0,+1}. A stamp that reports a count and asserts a set is not a stamp.
    for k, m in meta.items():
        m['cos_theta'] = 'CONSTANT 0.0 on every row -- no angular component in these params'
        m['conditioning_width_used'] = 1
    json.dump(dict(split_dir=a.split_dir, nbins=a.nbins,
                   scope='d only; cos_theta is a constant 0.0 placeholder column',
                   arms=meta),
              open(os.path.join(a.out, 'arms_meta.json'), 'w'), indent=2)
    print('\n%d conditioned arms built in %s' % (len(meta), a.out))
    print('wclass is binary-only by construction: warhead class is categorical, so b-a is undefined.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
