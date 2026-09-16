"""v4 STEERING SETS: keep the SAME pairs, and use every pair in BOTH directions.

THE USER FOUND TWO FREE MULTIPLIERS AND BOTH ARE REAL. The v3 funnel for attach_path:
    all pairs                 815,701  100.0%
    labelled                  815,701  100.0%   <- the attachment atom needs only a RING
    DISCARDED as SAME (d==0)  585,342   71.8%   <- a CHOICE, not a constraint
    kept UP/DOWN              230,359   28.2%
    after stratification      116,192   14.2%
    train                      92,954   11.4%
So 71.8% of a fully-labelled corpus was thrown away, and every surviving pair was used once.

MULTIPLIER 1 -- SAME IS A STEERING INSTRUCTION, NOT A MISSING LABEL. "Change the substituent but
HOLD the attachment path" is a thing we want to be able to ask for, and a model trained only on
UP/DOWN has never once been asked to hold a value. Dropping these rows removed both 585,342
training examples AND the entire "no-op" instruction from the vocabulary.

MULTIPLIER 2 -- EVERY PAIR IS TWO EXAMPLES. If A->B is DOWN then B->A is UP. The same chemistry,
read the other way. v3 used each pair once, in whichever order the extractor happened to emit.

THE THIRD CONSEQUENCE, WHICH MATTERS MORE THAN THE DATA VOLUME. QA-56 proved that on the v3 corpus
GAP_neg/GAP_perm is an ARITHMETIC IDENTITY: with d binary in {+1,-1}, permuting d either leaves it
alone or FLIPS it, so E[GAP_perm] = P*E[GAP_neg] and the ratio is 1/P for ANY model. I verified it
myself -- analytic null 2.155 against my reported 2.159. Directionality was UNTESTABLE by that
statistic at any value.
Adding SAME makes d take THREE values {-1, 0, +1}. A permutation can now map +1 -> 0, which is
NOT the negation, so GAP_perm and GAP_neg stop being proportional and the ratio carries real
information again. Keeping the SAME rows is therefore what makes the directionality question
ASKABLE, independently of it tripling the corpus.

SPLIT DISCIPLINE. The split is computed on the ISOTOPE-STRIPPED CORE *BEFORE* augmentation, and
both orientations of a pair inherit their core's side. Augmenting first and splitting after would
put A->B in train and B->A in valid -- a perfect-retrieval leak that would look like a spectacular
result. This is the single most dangerous thing about the reversal idea and it is handled here by
construction rather than checked for afterwards.

STRATIFICATION. Still on the input's own value, now balancing THREE classes per bin instead of two.
The descriptor null (#197) comes from regression to the mean: a low-valued input has more room
above it. Reversal does NOT remove that -- molecule A still sits where it sits -- so stratification
is still required. It is not a substitute, it is orthogonal.

THE v3 NUMBER, SOURCED AND WITH THE RIGHT NAME ON IT. This paragraph used to say "0.6544 measured
on v3" and call it the descriptor null. 0.6544 is one baseline, not the null: it is
INPUT-ONLY ECFP4 alone, liblinear, attach_path, uncapped
(data/chembl36_pairs_v3/baselines_attach_path.json; the lbfgs rerun gives 0.6550). The floor a
steering result has to BEAT is the max over every baseline reachable WITHOUT the instruction, and
on that same file ECFP4+scaffold+bonds reaches 0.6678 -- so quoting 0.6544 as the bar understated
it by 1.35 points and would certify anything in [0.6544, 0.6678] as beating a floor it does not
beat. steering_baselines.py:284 has the same correction in code. Note also that these are UNCAPPED
liblinear figures and are NOT comparable to the matched-cap lbfgs table used for v3-vs-v4e.
"""
import os, sys, json, re, argparse, random
from collections import defaultdict, Counter

from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

_FLAT = {}


def flat_key(smi):
    """Stereo- and isotope-blind canonical SMILES, memoised.

    THE UNION-FIND FIX WAS ABOUT MOLECULE IDENTITY AS A *STRING*, SO IT NEVER SAW THIS CHANNEL.
    Two stereoisomers are different strings and identical flat molecules; an exact-match leak check
    reports 0.00% on them, which is exactly why the defect has survived three corpora. Measured by
    an independent auditor: v3 3.37%, v4c attach_path 2.97%, attach_flex 2.68%, wclass 2.49% of
    VALID rows have an input that is a stereoisomer or isotope-variant of a TRAIN molecule.
    Unioning on the flat key puts every stereoisomer of a molecule in the same component, so the
    channel closes by construction rather than being measured and tolerated.
    """
    v = _FLAT.get(smi)
    if v is None:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            v = smi
        else:
            for at in m.GetAtoms():
                at.SetIsotope(0)
            # RemoveHs IS REQUIRED AND I MISSED IT ON THE FIRST PASS. SetIsotope(0) turns [2H] into
            # [H], which RDKit keeps as an EXPLICIT HYDROGEN ATOM, not as an implicit one -- so a
            # deuterated analogue still canonicalises differently from its parent and the two stay
            # in separate components. Tested directly: '[2H]c1ccccc1C=O' -> '[H]c1ccccc1C=O' vs
            # 'O=Cc1ccccc1'. 384 molecules in this corpus (0.0735%) carry 2H/3H, so the hole was
            # live, not theoretical -- and deuterium-switch series are a standard med-chem pair
            # type, so a corpus that contains them would leak straight through the fix.
            m = Chem.RemoveHs(m)
            Chem.RemoveStereochemistry(m)
            v = Chem.MolToSmiles(m, isomericSmiles=False)
        _FLAT[smi] = v
    return v

ISO = re.compile(r'\[\d+\*\]')
NUMERIC = ('attach_path', 'attach_flex', 'elec_path', 'elec_flex')
SEED = 20260915

# the reverse of a direction label; SAME is its own reverse
REV = {'UP': 'DOWN', 'DOWN': 'UP', 'SAME': 'SAME', 'CHANGED': 'CHANGED'}

# THE SPLIT'S DISJOINTNESS KEYS -- AND THE UNION-FIND GENUINELY ITERATES THIS NOW.
# My previous version declared SPLIT_KEYS and then claimed in a comment that "the union-find loop
# below iterates it". It did not. SPLIT_KEYS was referenced exactly once, in the stamp: I had
# RELOCATED the hardcoded string, not derived it, and then written a comment asserting the
# derivation. That is the same defect as the stamp it replaced, with an extra layer of false
# assurance on top -- and it is why v4e still shipped stamped with the 41.93%-leak description.
# Each entry is (stamp_name, keyfunc); the loop below calls every keyfunc, so adding a channel
# here adds it to BOTH the split and the stamp, and neither can drift from the other.
SPLIT_KEY_FUNCS = (
    ('molecule_identity_string', lambda smi: ('m', smi)),
    ('stereo_and_isotope_blind_molecule', lambda smi: ('f', flat_key(smi))),
)
SPLIT_ROW_KEY = ('isotope_stripped_core', lambda core: ('c', core))
SPLIT_KEYS = tuple(n for n, _ in SPLIT_KEY_FUNCS) + (SPLIT_ROW_KEY[0], 'BEFORE_augmentation')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--src', default='data/chembl36_pairs_v3/balanced_dedup.jsonl')
    ap.add_argument('--out', default='data/chembl36_pairs_v4/splits')
    ap.add_argument('--params', default='attach_path,attach_flex,wclass')
    ap.add_argument('--valid-frac', type=float, default=0.2)
    ap.add_argument('--max-per-bin', type=int, default=0, help='0 = no cap')
    a = ap.parse_args()
    params = a.params.split(',')
    rng = random.Random(SEED)

    rows = [json.loads(l) for l in open(a.src)]
    print('source rows: %d' % len(rows))

    # ---- SPLIT FIRST, AND ON THE UNION OF MOLECULE IDENTITY *AND* CORE.
    #
    # MY FIRST VERSION SPLIT ON THE CORE ALONE AND SILENTLY REGRESSED A LEAK FIX v3 ALREADY HAD.
    # v3's build_steer_splits.py:83 union-finds three ways --
    #     uni(('m',a),('m',b)); uni(('r',i),('m',a)); uni(('r',i),('c',ciso[i]))
    # -- because ONE MOLECULE CARRIES SEVERAL CORES, so a core-disjoint split is NOT
    # molecule-disjoint. Dropping the molecule half measured, on the identical script:
    #     v3 valid inputs verbatim in train:      0 / 16,570 = 0.00%
    #     v4 valid inputs verbatim in train: 26,252 / 62,603 = 41.93%   (42.14% of rows)
    # The old docstring line `split_on: isotope_stripped_core_BEFORE_augmentation` was an ACCURATE
    # description of a WEAKER rule, which is the worst kind of stamp: it reads as provenance and
    # passes review while the guarantee it implies is absent. Every v3-vs-v4 comparison made before
    # this fix was measured across a leak-level change, biased IN v4's FAVOUR.
    #
    # The union-find also makes the augmentation safe for free: unioning ('m',a) with ('m',b) puts
    # A->B and B->A in the SAME component by construction, so no orientation can straddle the split.
    mols = [(r['input_smiles'], r['output_smiles']) for r in rows]
    ciso = [ISO.sub('[*]', r.get('core', '')) for r in rows]
    par = {}

    def find(x):
        while par.setdefault(x, x) != x:
            par[x] = par[par[x]]
            x = par[x]
        return x

    def uni(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            par[rx] = ry

    for i, (ma, mb) in enumerate(mols):
        # EVERY channel in SPLIT_KEY_FUNCS is applied to BOTH endpoints, and the row is joined to
        # each. Driven by the same tuple the stamp prints, so the two cannot disagree.
        for _nm, _kf in SPLIT_KEY_FUNCS:
            ka, kb = _kf(ma), _kf(mb)
            uni(ka, kb)
            uni(('r', i), ka)
            uni(('r', i), kb)
        uni(('r', i), SPLIT_ROW_KEY[1](ciso[i]))
    comp = defaultdict(list)
    for i in range(len(rows)):
        comp[find(('r', i))].append(i)
    ks = list(comp)
    rng.shuffle(ks)
    # --valid-frac IS NOW HONOURED DIRECTLY. `len(rows) // int(round(1 / valid_frac))` inverts,
    # rounds to an integer, and inverts back, so it only lands on reciprocals of integers:
    #     asked 0.20 -> 1/0.2 = 5      -> realised 0.2000   (exact, which is why nothing on disk moved)
    #     asked 0.30 -> round(3.33)=3  -> realised 0.3333   (+11% more valid than requested)
    #     asked 0.15 -> round(6.67)=7  -> realised 0.1429   (-5%)
    # v4e used the 0.2 default and realised 19.97%, so no artifact is affected -- but the header
    # printed a ROW COUNT, never the realised fraction, so any future non-default run would have
    # silently built a different split than the one its own stamp claims.
    target_n = int(len(rows) * a.valid_frac)
    valid_idx = set()
    for k in ks:
        if len(valid_idx) >= target_n:
            break
        valid_idx.update(comp[k])
    side_of = ['valid' if i in valid_idx else 'train' for i in range(len(rows))]
    print('components: %d  -> valid rows %d / train rows %d'
          % (len(comp), len(valid_idx), len(rows) - len(valid_idx)))

    os.makedirs(a.out, exist_ok=True)
    meta = {}
    for p in params:
        dk, ak, bk = p + '_dir', p + '_a', p + '_b'
        numeric = p in NUMERIC

        # ---- AUGMENT: forward + reverse, SAME retained
        recs = defaultdict(list)          # side -> list of records
        for _i, r in enumerate(rows):
            d = r.get(dk)
            if d is None:
                continue
            side = side_of[_i]
            av, bv = r.get(ak), r.get(bk)
            # `ok` gates whether the a/b VALUES are carried through. It used to require int/float,
            # which silently dropped wclass_a/wclass_b -- they are warhead-class STRINGS. With no
            # `wclass_a` on the row, the stratifier's key `rec.get(ak)` is None for EVERY row, so
            # every row lands in ONE bin and stratification does exactly nothing. The wclass arm
            # has been unstratified this whole time and its input-only floor is 74.34%, not ~50%.
            ok = av is not None and bv is not None
            fwd = dict(input_smiles=r['input_smiles'], output_smiles=r['output_smiles'],
                       core=r.get('core', ''), target=r.get('target', ''))
            fwd[dk] = d
            if ok:
                fwd[ak], fwd[bk] = av, bv
            recs[side].append(fwd)
            # the SAME pair, read the other way round
            rev = dict(input_smiles=r['output_smiles'], output_smiles=r['input_smiles'],
                       core=r.get('core', ''), target=r.get('target', ''))
            rev[dk] = REV.get(d, d)
            if ok:
                rev[ak], rev[bk] = bv, av
            recs[side].append(rev)

        # ---- STRATIFY on the input's own value, balancing every class present in the bin
        out = {}
        for side in ('train', 'valid'):
            byb = defaultdict(lambda: defaultdict(list))
            for rec in recs[side]:
                k = rec.get(ak)
                k = int(k) if isinstance(k, (int, float)) else str(rec.get(ak, ''))
                byb[k][rec[dk]].append(rec)
            # DROP BINS THAT CANNOT SUPPORT EVERY CLASS. `min(len(v) for v in d.values())` takes
            # the minimum over the classes PRESENT in the bin, not over the classes POSSIBLE -- so
            # a bin containing only ONE label passed through ENTIRELY, perfectly "balanced" over a
            # set of size one. Those rows are pure label leakage: the bin key determines the answer.
            #   wclass      22,568 rows (30.29%) sat in 1-of-2-class bins. wclass_a=='' means the
            #               input carries NO warhead, which implies CHANGED with probability 1.0.
            #   attach_path 14.17% and attach_flex 16.36% sat in 2-of-3-class bins (a path of 1
            #               cannot go DOWN), so SAME/UP were balanced against an absent DOWN.
            # The measured input-only floor with these in was wclass 0.6514, attach_path 0.3562,
            # attach_flex 0.3626 -- above chance purely from the degenerate blocks. Stratification
            # that balances over "whatever happened to be here" is not stratification.
            n_classes = len({lab for d in byb.values() for lab in d})
            keep, dropped_bins, dropped_rows = [], 0, 0
            for k, d in byb.items():
                if len(d) < n_classes:
                    dropped_bins += 1
                    dropped_rows += sum(len(v) for v in d.values())
                    continue
                m = min(len(v) for v in d.values())
                if a.max_per_bin:
                    m = min(m, a.max_per_bin)
                for lab in sorted(d):
                    pool = d[lab]
                    rng.shuffle(pool)
                    keep.extend(pool[:m])
            rng.shuffle(keep)
            out[side] = keep
            if dropped_bins:
                print('  %-12s %-5s dropped %d degenerate bins (%d rows) that could not carry all '
                      '%d classes' % (p, side, dropped_bins, dropped_rows, n_classes))

        for side in ('train', 'valid'):
            with open(os.path.join(a.out, '%s_%s.jsonl' % (p, side)), 'w') as fh:
                for rec in out[side]:
                    fh.write(json.dumps(rec) + '\n')
        cn = Counter(r[dk] for r in out['train'])
        v3_ref = {'attach_path': 92954, 'attach_flex': 91460, 'wclass': 33493}.get(p)
        # ROWS OVER ROWS OVERSTATES THE GAIN. The augmentation writes A->B and B->A; the second row
        # is the same chemistry read backwards, not a new observation. Measured on v4c, rows/rows
        # gave 3.950x / 2.861x / 2.224x where distinct UNORDERED pairs give 3.063x / 2.384x /
        # 1.397x -- wclass was overstated by 1.59x. The stamp now carries the honest unit, and the
        # row multiplier beside it so the duplication factor is visible rather than hidden.
        unord = len({tuple(sorted((r['input_smiles'], r['output_smiles']))) for r in out['train']})
        mult = (unord / v3_ref) if v3_ref else float('nan')
        mult_rows = (len(out['train']) / v3_ref) if v3_ref else float('nan')
        print('%-13s train=%-8d valid=%-7d  classes=%s\n'
              '              distinct-unordered=%d  HONEST mult %.2fx  (rows mult %.2fx, '
              'duplication %.3f rows/pair)'
              % (p, len(out['train']), len(out['valid']), dict(cn),
                 unord, mult, mult_rows, len(out['train']) / max(unord, 1)))
        meta[p] = dict(train=len(out['train']), valid=len(out['valid']), classes=dict(cn),
                       v3_train=v3_ref, distinct_unordered=unord,
                       multiplier_unordered_HONEST=mult, multiplier_rows=mult_rows)

    # THE STAMP IS NOW DERIVED FROM THE CODE, NOT RETYPED BESIDE IT. It has been a revision behind
    # the builder THREE times: it still read 'isotope_stripped_core_BEFORE_augmentation' after the
    # rule became a core+molecule union-find, and again after the stereo/isotope-blind key was
    # added. A hardcoded provenance string is worse than none -- it reads as verification and
    # passes review while describing a rule the code no longer implements. `SPLIT_KEYS` is the
    # single source of truth: the union-find loop below iterates it, and the stamp prints it.
    # THE REALISED FRACTION IS RECORDED BESIDE THE REQUESTED ONE. They are not the same number and
    # never can be: the split moves whole union-find COMPONENTS, so it overshoots target_n by
    # however large the last component is. v4e asked 0.2 and realised 0.1997. Stamping only the
    # request makes that gap invisible, and invisible is how the reciprocal-rounding bug above
    # survived -- it produced a valid-looking file with a fraction nobody asked for.
    _realised = {p: (m['valid'] / float(m['train'] + m['valid']))
                 for p, m in meta.items() if (m.get('train', 0) + m.get('valid', 0)) > 0}
    json.dump(dict(src=a.src, seed=SEED,
                   valid_frac_requested=a.valid_frac,
                   valid_frac_realised=_realised,
                   split_on=SPLIT_KEYS,
                   augmentation='forward+reverse, SAME retained as a third class',
                   params=meta),
              open(os.path.join(a.out, 'v4_meta.json'), 'w'), indent=2)
    print('\nwrote %s' % a.out)
    print('NOTE: d now takes THREE values, so GAP_neg/GAP_perm is no longer the 1/P identity that')
    print('      QA-56 proved it was on v3. Directionality becomes testable again.')
    return 0


if __name__ == '__main__':
    sys.exit(main())
