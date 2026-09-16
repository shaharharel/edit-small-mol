#!/usr/bin/env python
"""ONE TABLE for every steering param, from the JSONs run_full_eval.sh writes.

This is the deliverable: for each param, did the instruction move the generated COHORT,
and what did the molecules cost.

READING RULES, all of them learned the hard way in this project:

  * COHORT SHIFT, not loss. GAP/BENEFIT are held-out loss and can disagree with generation
    -- linker_atom_count had BENEFIT +0.0103 while an early cohort tied 92% of the time.
  * The null for obedience is 0.5, NOT 0.
  * `UP_vs_none` and `DOWN_vs_none` are the PRODUCT claim. `UP_vs_DOWN` is easier to move
    and is NOT the same statement: a model can separate its own two instructions while
    neither cohort differs from an unsteered baseline.
  * `none_vs_prior` says how much of any effect is the covalent FINE-TUNING rather than the
    instruction. A param whose UP_vs_prior is large but UP_vs_none is ~0 has not steered
    anything -- the fine-tuning did the work.
  * KS p with an unmoved mean is a REAL result (distribution reshaped, centre unchanged).
    Report it; do not bury it.
  * ROLE IS NEVER POOLED -- train is 45x skewed and valid is rebalanced by design.
"""
from __future__ import annotations
import os, sys, json, glob


def fmt(v, w=9, d=4):
    return ('%*.*f' % (w, d, v)) if isinstance(v, (int, float)) else '%*s' % (w, '-')


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else 'results/full_eval'
    files = sorted(glob.glob(os.path.join(d, '*.json')))
    if not files:
        print('no results in %s' % d); return
    print('=' * 104)
    print('%-22s %-14s %10s %10s %10s %10s' %
          ('param', 'contrast', 'mean_shift', 'median', 'KS_D', 'KS_p'))
    print('=' * 104)
    verdicts = []
    for f in files:
        if f.endswith('_panel.json'):
            continue
        j = json.load(open(f))
        p = j.get('param', os.path.basename(f)[:-5])
        cons = j.get('contrasts', {})
        for k, c in cons.items():
            if not c:
                continue
            print('%-22s %-14s %10s %10s %10s %10s'
                  % (p, k, fmt(c.get('mean_shift')), fmt(c.get('median_shift')),
                     fmt(c.get('ks_D')), ('%10.2g' % c['ks_p']) if c.get('ks_p') is not None else '%10s' % '-'))
        # verdict uses the PRODUCT claim, not the easier UP_vs_DOWN
        un, dn = cons.get('UP_vs_none'), cons.get('DOWN_vs_none')
        if un and dn:
            moved = ((un.get('ks_p') or 1) < 0.01) and ((dn.get('ks_p') or 1) < 0.01)
            brack = (un.get('mean_shift', 0) > 0) and (dn.get('mean_shift', 0) < 0)
            verdicts.append((p, 'STEERS COHORT (brackets baseline)' if (moved and brack)
                             else ('shifts one way only' if moved else 'no cohort shift'),
                             un.get('mean_shift'), dn.get('mean_shift')))
        print('-' * 104)

    print('\n=== VERDICTS (product claim: does the cohort move vs an UNSTEERED model?) ===')
    for p, v, u, dd in verdicts:
        print('  %-22s %-36s UP %s  DOWN %s' % (p, v, fmt(u, 8), fmt(dd, 8)))

    print('\n=== GENERATION / MANUSCRIPT PANEL ===')
    print('%-22s %9s %11s %9s %8s %9s' %
          ('param', 'validity', 'uniqueness', 'novelty', 'QED', 'warhead'))
    for f in sorted(glob.glob(os.path.join(d, '*_panel.json'))):
        j = json.load(open(f)).get('summary', {})
        g, m = j.get('generation_panel', {}), j.get('manuscript_panel', {})
        print('%-22s %9s %11s %9s %8s %9s'
              % (j.get('param', '?'), fmt(g.get('validity'), 9, 3),
                 fmt(g.get('uniqueness'), 11, 3), fmt(g.get('novelty_vs_train'), 9, 3),
                 fmt(g.get('qed_mean'), 8, 3), fmt(m.get('warhead_retention'), 9, 3)))
    print('\nNULL for obedience is 0.5. UP_vs_none/DOWN_vs_none is the product claim,')
    print('NOT UP_vs_DOWN. Role results are PER ROLE and must never be pooled.')


if __name__ == '__main__':
    main()
