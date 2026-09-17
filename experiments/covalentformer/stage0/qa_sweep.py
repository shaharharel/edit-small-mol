#!/usr/bin/env python
"""QA sweep over tonight's new code. Runs every pipeline cycle, cheap, fails LOUD.

Each check exists because the corresponding bug actually shipped today.
"""
import sys, os, csv, glob, json, importlib.util
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
fails = []

def chk(name, cond, detail=''):
    print('  %-52s %s %s' % (name, 'PASS' if cond else 'FAIL', detail))
    if not cond: fails.append(name)

print('=== QA SWEEP ===')

# 1. valid CSVs must NOT be MW-ordered (three wrong conclusions came from this)
try:
    import numpy as np
    from rdkit import Chem, RDLogger; RDLogger.DisableLog('rdApp.*')
    from rdkit.Chem import Descriptors
    worst, worstf = 0.0, ''
    for f in glob.glob('data/steer_v*/*_valid.csv'):
        rows = list(csv.DictReader(open(f)))[:300]
        xs, ys = [], []
        for i, r in enumerate(rows):
            m = Chem.MolFromSmiles(r.get('anchor', ''))
            if m is not None: xs.append(i); ys.append(Descriptors.MolWt(m))
        if len(xs) > 50:
            r_ = abs(float(np.corrcoef(xs, ys)[0, 1]))
            if r_ > worst: worst, worstf = r_, f
    chk('valid CSVs unordered (|corr(MW,idx)| < 0.30)', worst < 0.30, '| worst %.3f %s' % (worst, os.path.basename(worstf)))
except Exception as e:
    chk('valid CSV ordering check ran', False, str(e)[:60])

# 2. linker_atom_count must NOT be degenerate per warhead class (the ring bug)
try:
    rows = list(csv.DictReader(open('data/labels/molecule_params_v3.csv')))
    byc = {}
    for r in rows:
        if r['linker_atom_count'] != '':
            byc.setdefault(r['warhead_class'], []).append(int(r['linker_atom_count']))
    bad = [c for c, v in byc.items() if len(v) >= 50 and len(set(v)) == 1]
    chk('no warhead class has a CONSTANT linker_atom_count', not bad, '| %s' % (bad[:3] if bad else 'ok'))
except Exception as e:
    chk('linker degeneracy check ran', False, str(e)[:60])

# 3. cohort_shift must REFUSE a checkpoint whose conditioning did not load
try:
    import cohort_shift
    src = open(cohort_shift.__file__).read()
    chk('cohort_shift asserts the conditioning tensor loaded',
        'CONDITIONING DID NOT LOAD' in src and 'CONDITIONING NOT IN CHECKPOINT' in src)
except Exception as e:
    chk('cohort_shift guard present', False, str(e)[:60])

# 4. trainer must save BEST-ONLY (a full disk reads as file corruption)
try:
    src = open('stage0/train_steer_v5.py').read()
    chk('trainer saves BEST-ONLY, not every epoch', '_best.pt' in src and "_ep%d.pt" not in src)
except Exception as e:
    chk('trainer save-policy check ran', False, str(e)[:60])

# 5. GAP_flip must be interpretable: SAME cannot dominate a valid set
try:
    worst, wf = 0.0, ''
    for f in glob.glob('data/steer_v6*/*_valid.csv'):
        rows = list(csv.DictReader(open(f)))
        if len(rows) < 50: continue
        frac = sum(1 for r in rows if r.get('instr') == 'SAME') / len(rows)
        if frac > worst: worst, wf = frac, f
    chk('SAME < 55% of valid (else GAP_flip == GAP_perm)', worst < 0.55, '| worst %.0f%% %s' % (100*worst, os.path.basename(wf)))
except Exception as e:
    chk('SAME-fraction check ran', False, str(e)[:60])

print('\n%d/%d checks passed' % (5 - len(fails), 5))
if fails: print('FAILED: %s' % ', '.join(fails))
sys.exit(1 if fails else 0)
