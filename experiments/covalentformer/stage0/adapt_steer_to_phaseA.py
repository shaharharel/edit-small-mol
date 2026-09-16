"""ADAPTER: splits_v2 JSONL -> the CSV schema train_phaseA.py's RoleData expects.

WRITTEN DELIBERATELY RATHER THAN IMPROVISED, because a silently mis-mapped column is how four of
tonight's wrong numbers happened. The contract, read from train_phaseA.py:88-110:
    RoleData needs  'anchor' and 'target' (SMILES strings)
    mode='geom'     additionally reads float 'd' and float 'cos_theta'
Nothing else is consumed. Any other column is inert.

THE MAPPING, and why this one:
    anchor    <- input_smiles
    target    <- output_smiles
    d         <- +1.0 for UP / CHANGED, -1.0 for DOWN / SAME
    cos_theta <- 0.0 constant
`d` carries the SIGNED direction on purpose. score_steer_arms.py measures GAP_neg = loss(-c) -
loss(c), i.e. the cost of being handed the OPPOSITE instruction. With d = +/-1 a negation is
exactly a direction flip, so the existing directionality harness -- GAP_neg, GAP_perm, the
magnitude-only yardstick -- applies unchanged and needs no new estimator. Encoding direction as a
role STRING instead would have made negation undefined and forced a second harness.

cos_theta is a CONSTANT here and that is deliberate, not laziness: these five params have no
angular component. A constant channel is the `none`-arm shape, contributing exactly zero, which is
the behaviour `none` already demonstrates (GAP_zero = 0.00000 across 23 cells). It is stamped
below so nobody later reads a zero and thinks the angle was measured and found null.

NO ROW IS DROPPED SILENTLY. Unparseable rows are counted and printed.
"""
import json, os, csv, argparse

DIR_POS = {'UP', 'CHANGED'}
# SAME IS NO LONGER FOLDED ONTO DOWN. This file is where bug (1) originated; adapt_steer_v2.py was
# fixed and this one was not. It is BENIGN ON EVERY CSV CURRENTLY ON DISK -- verified by cross-
# tabbing dir_label against d across all 18 phaseA_csv arms: no file contains UP, DOWN and SAME
# together, because v3 discarded the SAME rows and wclass is genuinely 2-class where +-1 is right.
# It ARMS the moment --src points at a v4/v4e split, where SAME is exactly one third of every
# attach_* arm (104,687 of 314,061). 33.3% of rows would be taught "decrease it" for "hold it",
# and d would stay BINARY so QA-56's GAP_neg/GAP_perm = 1/P identity would still apply on top.
DIR_NEG = {'DOWN'}
DIR_ZERO = {'SAME'}


def convert(src, dst, param):
    key = param + '_dir'
    n = kept = bad_dir = bad_smi = 0
    with open(dst, 'w', newline='') as fh:
        w = csv.DictWriter(fh, fieldnames=['anchor', 'target', 'd', 'cos_theta', 'dir_label'])
        w.writeheader()
        for line in open(src):
            n += 1
            try:
                r = json.loads(line)
            except Exception:
                bad_smi += 1
                continue
            lab = r.get(key)
            if lab in DIR_POS:
                d = 1.0
            elif lab in DIR_ZERO:
                d = 0.0
            elif lab in DIR_NEG:
                d = -1.0
            else:
                bad_dir += 1
                continue
            a, t = r.get('input_smiles'), r.get('output_smiles')
            if not a or not t:
                bad_smi += 1
                continue
            w.writerow(dict(anchor=a, target=t, d=d, cos_theta=0.0, dir_label=lab))
            kept += 1
    return n, kept, bad_dir, bad_smi


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/chembl36_pairs/splits_v2')
    ap.add_argument('--out', default='data/chembl36_pairs/phaseA_csv')
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    params = sorted({f.rsplit('_', 1)[0] for f in os.listdir(a.src) if f.endswith('.jsonl')})
    print('%-14s %-7s %8s %8s %9s %9s' % ('param', 'split', 'rows', 'kept', 'bad_dir', 'bad_smi'))
    meta = {}
    for p in params:
        d = os.path.join(a.out, p)
        os.makedirs(d, exist_ok=True)
        for sp, fname in (('train', 'train.csv'), ('valid', 'valid.csv')):
            src = os.path.join(a.src, '%s_%s.jsonl' % (p, sp))
            if not os.path.exists(src):
                print('%-14s %-7s MISSING' % (p, sp)); continue
            n, kept, bd, bs = convert(src, os.path.join(d, fname), p)
            print('%-14s %-7s %8d %8d %9d %9d' % (p, sp, n, kept, bd, bs))
            meta['%s_%s' % (p, sp)] = dict(rows=n, kept=kept, bad_dir=bd, bad_smi=bs)
    json.dump(dict(mapping=dict(anchor='input_smiles', target='output_smiles',
                                # THE STAMP WAS ONE REVISION BEHIND THE CODE. It said
                                # '-1 DOWN/SAME' for ~37 minutes after the SAME->0.0 fix
                                # landed, so the adapter wrote correct d in {-1,0,+1}
                                # while its own meta claimed it had binarised -- and a
                                # reviewer would have 'corrected' for a bug that is gone.
                                d='+1 UP/CHANGED, 0 SAME, -1 DOWN',
                                cos_theta='CONSTANT 0.0 -- no angular component in these params'),
                   per_file=meta), open(os.path.join(a.out, 'adapter_meta.json'), 'w'), indent=2)
    print('\nwrote %s/' % a.out)
