#!/usr/bin/env python3
"""Build the one reported table: the 2x2 cells, both zero-shot rows, and the chat-template pilot.

    python summarize_matrix.py                    # discover everything, score it, print the table
    python summarize_matrix.py --no-rescore       # reuse cached .axes.json, print only

Scoring is delegated to eval_axes.py (the canonical scorer) by shelling out with `--json`, once per
generations.jsonl, cached beside it as `<name>.axes.json` and refreshed only when the generations are
newer. This file aggregates; it never defines a metric. That split matters: if a metric is corrected
in eval_axes.py -- as the chance-baseline pairing was -- every row here picks the correction up on
the next run, and there is no second implementation to drift out of agreement with it.

WHAT GETS FLAGGED AS NOISE RATHER THAN RANKED
---------------------------------------------
A target-level split over 860 examples has large variance, and with two seeds the honest summary of
two cells differing by less than their seed spread is "indistinguishable". Every aggregate prints as
mean +/- half-range, and `compare()` refuses to name a winner when the gap is under the larger of
the two spreads. One-seed cells print the gap but are explicitly not ranked, because a spread of
zero from a single seed is not a measurement of zero variance.

THE TWO COMPARISONS THE EXPERIMENT IS ABOUT
    COLUMN  pocket vs blind, same checkpoint -- does the contact table contribute anything? If this
            is inside noise, nothing has been demonstrated, whatever else the table says.
    ROW     SmileyLlama vs plain instruct, same column -- what does the ChEMBL pretraining buy?

WHICH NUMBERS ARE MEANINGLESS ALONE, AND WHAT THEY ARE PRINTED AGAINST
---------------------------------------------------------------------
Three columns here cannot be read without their baseline, so the baseline is printed in the table
rather than in a footnote:

  tanimoto      against the ECHO BASELINE (~0.68). An analogue pair is by construction a small edit,
                so copying the input scores high. `beats_echo` is the honest number; the pilot
                scored 0.52 against 0.68, i.e. WORSE than echoing, while validity was 0.89 -- a
                validity-only report would have called that a success.
  groundedness  against the REFERENCE CEILING (~0.90, not 1.0). It is also gameable by copying
                residue names out of the prompt, which is why both recall lifts sit beside it.
  recall lift   `_all` and `_knowable`, each against its own chance baseline (they agree at 1.77x on
                the pilot once paired correctly). The comparison that means something is against the
                CHAIR's own lift of ~3.5-4x: that is what "half the chemist's selectivity" measures.

Groundedness is chain-blind: a right-residue/wrong-chain citation counts as grounded. Measured on
the pilot, 15.5% of the model's grounded citations and 15.9% of the reference's are chain-ambiguous
on identical tables, so the leniency inflates model and ceiling together and is not a differential
confound for either comparison. It is not corrected, and it is stated beside the number.

Every val figure is a MODEL-SELECTION score, not a held-out test score: the stopping epoch and
checkpoint were chosen on this split. The header says so on every print.
"""
import argparse
import glob
import json
import os
import statistics
import subprocess
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
PY = sys.executable
AXES = os.path.join(HERE, 'eval_axes.py')
DATA = os.path.join(HERE, 'sft_out', 'sft_pocket.jsonl')

# (header, eval_axes key, decimals). Exactly the columns asked for, plus the baselines they need.
COLS = [
    # `format` is the HEADLINE: >=5 prose words AND a parseable SMILES. A model that answers a
    # reasoning task with a bare molecule has failed the task however parseable the molecule is --
    # zero-shot SmileyLlama scores 0.000 here and 1.000 on `smi_recov`, and that gap IS the
    # catastrophic-forgetting result. `smi_recov` is kept beside it because it is what separates a
    # format failure from a chemistry failure. `words` and `names_res` stay so the failure is
    # legible from the table rather than resting on one rate.
    ('format', 'format_compliance', 3),
    ('smi_recov', 'smiles_recoverable', 3),
    ('words', 'mean_words', 1),
    ('names_res', 'names_residue_frac', 3),
    ('valid', 'validity_overall', 3),
    ('distinct', 'distinct_smiles', 0),
    ('tanimoto', 'tanimoto_to_true_analogue', 3),
    ('echo_base', 'tanimoto_echo_baseline', 3),
    ('beats_echo', 'beats_echo_frac', 3),
    ('edit_dir', 'edit_direction_accuracy', 3),
    ('grounded', 'groundedness_micro', 3),
    ('lift_all', 'lift_over_chance_all', 2),
    ('lift_know', 'lift_over_chance_knowable', 2),
]


def score_run(gen_path, rescore=True):
    """Run eval_axes.py on one generations.jsonl, cached. Returns (generated, ceiling) dicts."""
    cache = gen_path.replace('.jsonl', '.axes.json')
    # Invalidate on the SCORER's mtime as well as the generations'. eval_axes.py has been corrected
    # twice mid-run (the chance-baseline pairing, then the format-compliance definition); a cache
    # keyed only on the generations would have gone on serving pre-correction numbers forever, and
    # nothing in the printed table would have hinted at it.
    fresh = (os.path.exists(cache)
             and os.path.getmtime(cache) >= os.path.getmtime(gen_path)
             and os.path.getmtime(cache) >= os.path.getmtime(AXES))
    if rescore and not fresh:
        r = subprocess.run([PY, AXES, gen_path, '--data', DATA, '--json', cache],
                           capture_output=True, text=True)
        if r.returncode != 0 or not os.path.exists(cache):
            return None, None, (r.stderr or r.stdout)[-300:]
    if not os.path.exists(cache):
        return None, None, 'no cached score and --no-rescore given'
    d = json.load(open(cache))
    body = next(iter(d.values()))          # {"<label>": {"generated":..., "reference_ceiling":...}}
    return body.get('generated'), body.get('reference_ceiling'), None


def discover(runs_dir, pilot_dir):
    """-> list of (row_label, group_key, generations_path). group_key aggregates seeds."""
    found = []
    for p in sorted(glob.glob(os.path.join(runs_dir, '*_s[0-9]', 'eval', 'generations.jsonl'))):
        cell = os.path.basename(os.path.dirname(os.path.dirname(p)))   # model_arm_sN
        group = cell.rsplit('_s', 1)[0]
        found.append((cell, group, p))
    for p in sorted(glob.glob(os.path.join(runs_dir, 'zeroshot_*', 'generations.jsonl'))):
        name = os.path.basename(os.path.dirname(p))
        found.append((name, name, p))
    for p in sorted(glob.glob(os.path.join(pilot_dir, '*', 'eval', 'generations.jsonl'))):
        arm = os.path.basename(os.path.dirname(os.path.dirname(p)))
        if arm in ('pocket', 'blind'):
            found.append(('pilot_chattmpl_' + arm, 'pilot_chattmpl_' + arm, p))
    return found


def stopping_epochs(runs_dir, group):
    eps = []
    for p in sorted(glob.glob(os.path.join(runs_dir, group + '_s[0-9]', 'run_state.json'))):
        st = json.load(open(p))
        if st.get('stopping_epoch') is not None:
            eps.append(st['stopping_epoch'])
    return eps


def agg(vals):
    vals = [v for v in vals if v is not None]
    if not vals:
        return None, None, 0
    if len(vals) == 1:
        return vals[0], 0.0, 1
    return statistics.mean(vals), (max(vals) - min(vals)) / 2.0, len(vals)


def cellfmt(mean, spread, n, dp):
    if mean is None:
        return '--'
    if n <= 1:
        return '%.*f' % (dp, mean)
    return '%.*f+-%.*f' % (dp, mean, dp, spread)


def compare(groups, a, b, key, label):
    if a not in groups or b not in groups:
        return '%-34s not comparable yet' % label
    ma, sa, na = agg([g[key] for g in groups[a]])
    mb, sb, nb = agg([g[key] for g in groups[b]])
    if ma is None or mb is None:
        return '%-34s no data' % label
    gap = abs(ma - mb)
    noise = max(sa or 0, sb or 0)
    if na < 2 or nb < 2:
        return ('%-34s %.3f vs %.3f  gap %.3f  -- ONE SEED, not ranked' % (label, ma, mb, gap))
    if gap <= noise:
        return ('%-34s %.3f vs %.3f  gap %.3f  INSIDE seed spread (+-%.3f) -> NOISE, not ranked'
                % (label, ma, mb, gap, noise))
    return ('%-34s %.3f vs %.3f  gap %.3f  > seed spread (+-%.3f)' % (label, ma, mb, gap, noise))


ORDER = ['zeroshot_smiley', 'zeroshot_base',
         'smiley_pocket', 'smiley_blind', 'base_pocket', 'base_blind',
         'pilot_chattmpl_pocket', 'pilot_chattmpl_blind']
PRETTY = {'zeroshot_smiley': 'zero-shot smiley', 'zeroshot_base': 'zero-shot base',
          'smiley_pocket': 'run1 smiley+pocket', 'smiley_blind': 'run2 smiley+blind',
          'base_pocket': 'run3 base+pocket', 'base_blind': 'run4 base+blind',
          'pilot_chattmpl_pocket': 'pilot chat-tmpl pocket',
          'pilot_chattmpl_blind': 'pilot chat-tmpl blind'}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--runs', default=os.path.join(HERE, 'sft_runs'))
    ap.add_argument('--pilot', default=os.path.join(HERE, 'sft_runs_pilot_chattemplate'))
    ap.add_argument('--no-rescore', action='store_true')
    a = ap.parse_args()

    groups, ceil, errs = {}, None, []
    for label, group, path in discover(a.runs, a.pilot):
        gen, c, err = score_run(path, rescore=not a.no_rescore)
        if err:
            errs.append('%s: %s' % (label, err))
            continue
        groups.setdefault(group, []).append(gen)
        if c and ceil is None:
            ceil = c

    head = '%-24s %5s %6s ' % ('row', 'seeds', 'epoch') + ' '.join(
        '%-12s' % h for h, _k, _d in COLS)
    print('=' * len(head))
    print('POCKET-DEPENDENCE MATRIX  --  val = MODEL-SELECTION score, NOT held-out test')
    print('groundedness is chain-blind (15.5%% model vs 15.9%% reference ambiguity on identical')
    print('tables -> inflates model and ceiling together, not a differential confound)')
    print('=' * len(head))
    print(head)
    print('-' * len(head))
    if ceil:
        row = '%-24s %5s %6s ' % ('CEILING (reference)', '-', '-') + ' '.join(
            '%-12s' % ('--' if ceil.get(k) is None else '%.*f' % (d, ceil[k]))
            for _h, k, d in COLS)
        print(row)
        print('-' * len(head))
    for g in ORDER:
        if g not in groups:
            print('%-24s %s' % (PRETTY.get(g, g), '(not run yet)'))
            continue
        seeds = groups[g]
        eps = stopping_epochs(a.runs, g)
        cells = ' '.join('%-12s' % cellfmt(*agg([s.get(k) for s in seeds]), dp=d)
                         for _h, k, d in COLS)
        print('%-24s %5d %6s ' % (PRETTY.get(g, g), len(seeds),
                                  ','.join('%g' % e for e in eps) if eps else '-') + cells)

    print('\nCOLUMN -- does the pocket contribute? (if inside noise, nothing is demonstrated)')
    for m in ('smiley', 'base'):
        for key, nm in (('groundedness_micro', 'grounded'),
                        ('lift_over_chance_knowable', 'recall lift'),
                        ('tanimoto_to_true_analogue', 'tanimoto'),
                        ('beats_echo_frac', 'beats echo'),
                        ('edit_direction_accuracy', 'edit dir')):
            print('  ' + compare(groups, m + '_pocket', m + '_blind', key,
                                 '%s pocket-vs-blind %s' % (m, nm)))
    print('\nROW -- what does the chemistry pretraining buy?')
    for arm in ('pocket', 'blind'):
        for key, nm in (('groundedness_micro', 'grounded'),
                        ('lift_over_chance_knowable', 'recall lift'),
                        ('tanimoto_to_true_analogue', 'tanimoto'),
                        ('format_compliance', 'format')):
            print('  ' + compare(groups, 'smiley_' + arm, 'base_' + arm, key,
                                 '%s smiley-vs-base %s' % (arm, nm)))

    if errs:
        print('\nNOT SCORED:')
        for e in errs:
            print('  ' + e)

    out = {'rows': {g: {'n_seeds': len(v), 'stopping_epochs': stopping_epochs(a.runs, g),
                        **{k: {'mean': agg([s.get(k) for s in v])[0],
                              'half_range': agg([s.get(k) for s in v])[1]}
                           for _h, k, _d in COLS}}
                    for g, v in groups.items()},
            'reference_ceiling': ceil,
            'val_is_model_selection_not_test': True,
            'groundedness_is_chain_blind': True,
            'not_scored': errs}
    path = os.path.join(a.runs, 'matrix_summary.json')
    json.dump(out, open(path, 'w'), indent=2, default=str)
    print('\nwrote %s' % path)


if __name__ == '__main__':
    main()
