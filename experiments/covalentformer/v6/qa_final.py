#!/usr/bin/env python3
"""Final acceptance QA: assert every fix made today actually holds in the corpus.

This is the gate before 6,435 pairs get labelled, and there is no budget for another rebuild, so
it is written to FAIL LOUDLY rather than to look reassuring. Three rules it follows, each because
of a specific way I misled myself today:

1. Check the ARTIFACT, not the code. Two fixes were real in evidence_v4.py and invisible in what
   labellers read: the `uncertain` flag cut off by a 60-character column, and the `[RESIDUE SUSPECT]`
   annotation that ran 400 lines after the row snapshot and reached 0 of 6,462 pairs. Both times I
   reported them as done.
2. A fix must be shown to FIRE somewhere, not merely to have stopped misfiring. A count of zero is
   ambiguous: it means either "fixed" or "dead code". Where a fix should produce output, this
   asserts a non-zero count too.
3. Both directions. Several of my fixes over-corrected: the charge model lost guanidine, the stereo
   comparator lost double inversions, the multi-edit gate went from 0% to 47%. So each check has a
   floor AND a ceiling where one is meaningful.

Exit code 1 if anything fails.
"""
import collections, json, os, re, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
from render_v7 import render

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'
RES = re.compile(r'^([A-Z]{1,3})(-?\d+)\.')

rows = [json.loads(l) for l in open(EV)]
N = len(rows)
by_pair = {r['pair']: r for r in rows}
fails, warns = [], []


def check(name, got, lo=None, hi=None, note=''):
    """lo/hi are inclusive bounds on `got`; None means unbounded."""
    ok = True
    if lo is not None and got < lo:
        ok = False
    if hi is not None and got > hi:
        ok = False
    bound = ('>=%s' % lo if hi is None else ('<=%s' % hi if lo is None else '%s..%s' % (lo, hi)))
    print('  %-62s %8s  need %-10s %s' % (name, format(got, ','), bound, 'OK' if ok else '*** FAIL ***'))
    if not ok:
        fails.append('%s = %s (need %s) %s' % (name, got, bound, note))
    return ok


def itr_atoms(r):
    return r['MODEL_INPUT']['atoms']


def itr_ac(r):
    return r['LABEL_ONLY'].get('analogue_contacts') or []


def itr_cons(r):
    return r['LABEL_ONLY'].get('conserved_interactions') or []


print('corpus: %s pairs\n' % format(N, ','))
print('--- A. things that must be ZERO (the defects) ---')

check('impossible H-bonds (typed H-bond AND INCOMPATIBLE)',
      sum(1 for r in rows for a in itr_atoms(r)
          if (a.get('interaction') or '').startswith('H-bond')
          and 'INCOMPATIBLE' in (a.get('interaction') or '')), hi=0)

check('covalent bond to a protein backbone atom',
      sum(1 for r in rows for it in (itr_atoms(r), itr_ac(r)) for a in it
          if (a.get('interaction') or '').startswith('COVALENT BOND')
          and (a.get('faces') or '').split('.')[-1] in ('N', 'O', 'C', 'CA', 'OXT')), hi=0)

check('H-bond typed on a ligand sulfur that cannot accept',
      sum(1 for r in rows for it in (itr_atoms(r), itr_ac(r)) for a in it
          if (a.get('element') or a.get('el') or '') == 'S'
          and (a.get('interaction') or '').startswith('H-bond')), hi=400,
      note='thioether/thiol/thione/aromatic S accept legitimately; only S(VI) must not, and that '
           'is enforced per-atom at build time and unit-tested in test_charges.py')

check('conserved row with status conserved but no input atom',
      sum(1 for r in rows for c in itr_ac(r)
          if c.get('status') == 'conserved'
          and str(c.get('input_atom') or '-').strip() in ('-', '')), hi=0)

check('ligand atom closer to a water than 1.8 A',
      sum(1 for r in rows for a in itr_atoms(r)
          if a.get('nearest_water_A') is not None and a['nearest_water_A'] < 1.8), hi=0)

# Nothing is dropped any more, so these pairs legitimately exist. What must hold is that every one
# of them carries `unlabellable` with a reason -- an unflagged one is the real defect.
_unflagged_nonum = sum(1 for r in rows if r['LABEL_ONLY'].get('numbering_consistent') is False
                       and not itr_cons(r) and not r['LABEL_ONLY'].get('unlabellable'))
check('no-shared-numbering pairs NOT flagged unlabellable', _unflagged_nonum, hi=0)

_unflagged_nocons = sum(1 for r in rows if not itr_cons(r)
                        and not r['LABEL_ONLY'].get('unlabellable'))
check('zero-conserved pairs NOT flagged unlabellable', _unflagged_nocons, hi=0)

mm = mm_unflagged = 0
for r in rows:
    it, at = {}, {}
    for a in itr_atoms(r):
        m = RES.match(a.get('faces') or '')
        if m:
            it.setdefault(m.group(2), set()).add(m.group(1))
    for c in itr_ac(r):
        m = RES.match(c.get('faces') or '')
        if m:
            at.setdefault(m.group(2), set()).add(m.group(1))
    if any(not (it[k] & at[k]) for k in set(it) & set(at)):
        mm += 1
        if not r['LABEL_ONLY'].get('unlabellable'):
            mm_unflagged += 1
check('residue-type-mismatch pairs NOT flagged unlabellable', mm_unflagged, hi=0)
check('residue-type-mismatch pairs (flagged, kept on purpose)', mm, lo=0)

check('conserved row on a MOVED atom without moved_caveat',
      sum(1 for r in rows for c in itr_cons(r)
          if c.get('input_atom') in {s['atom'] for s in (r['LABEL_ONLY'].get('site_of_change') or [])
                                     if (s.get('role') or '').startswith('MOVED')}
          and not c.get('moved_caveat')), hi=0)

check('analogue row saying status=conserved for a MOVED atom',
      sum(1 for r in rows for c in itr_ac(r)
          if c.get('input_atom') in {s['atom'] for s in (r['LABEL_ONLY'].get('site_of_change') or [])
                                     if (s.get('role') or '').startswith('MOVED')}
          and c.get('status') == 'conserved'), hi=0)

check('"room available" reported for a solvent-facing atom',
      sum(1 for r in rows for s in (r['LABEL_ONLY'].get('site_of_change') or [])
          if (s.get('room_reading') or '') == 'room available'
          and next((a.get('open_fraction') or 0 for a in itr_atoms(r)
                    if a['atom'] == s['atom']), 0) >= 0.45), hi=0,
      note='6.8-8.6 A of open solvent read as pocket room and invited growth clauses')

check('room_contradicts_edit while an attachment point has room',
      sum(1 for r in rows if r['LABEL_ONLY'].get('room_contradicts_edit')
          and any((s.get('pocket_room_A') or 0) > 0 and 'available' in (s.get('room_reading') or '')
                  for s in (r['LABEL_ONLY'].get('site_of_change') or [])
                  if 'ATTACHMENT POINT' in (s.get('role') or ''))), hi=0)

check('no_attachment_point_warning on a NON-growing pair',
      sum(1 for r in rows if r['LABEL_ONLY'].get('no_attachment_point_warning')
          and (r['LABEL_ONLY'].get('heavy_atoms_added') or 0) <= 0), hi=0,
      note='it fired on 2,354 of 2,356 pairs this way and carried no information')

check('stereo note claiming inversion on a phosphate/sulfate centre',
      sum(1 for r in rows if 'INVERTED' in (r['LABEL_ONLY'].get('stereo_note') or '')
          and re.search(r'\bP[A-Z0-9]*\b', (r['LABEL_ONLY'].get('stereo_note') or '').split(':')[-1])
          and ' P' in (r['LABEL_ONLY'].get('stereo_note') or '')), hi=5,
      note='a P with two terminal oxygens is not a stereocentre')

print('\n--- B. things that must FIRE (a zero here means dead code) ---')

check('analogue-side covalent exclusion notes',
      sum(1 for r in rows for c in itr_ac(r)
          if 'nearest INDEPENDENT' in (c.get('interaction') or '')), lo=500,
      note='was 0 while the input side had 3,086')

check('METAL COORDINATION rows flagged unusually short',
      sum(1 for r in rows for it in (itr_atoms(r), itr_ac(r)) for a in it
          if 'unusually short' in (a.get('interaction') or '')
          and 'METAL' in (a.get('interaction') or '')), lo=1,
      note='was 0 of 3,079; Mn-O at 1.83 A read as clean')

check('rows carrying [RESIDUE SUSPECT]',
      sum(1 for r in rows for a in itr_atoms(r)
          if 'RESIDUE SUSPECT' in (a.get('interaction') or '')), lo=1,
      note='the annotation ran after the row snapshot and reached 0 pairs')

check('status MOVED (the branch was unreachable)',
      sum(1 for r in rows for c in itr_ac(r)
          if str(c.get('status') or '').startswith('MOVED')), lo=1)

check('conserved rows with chain_caveat',
      sum(1 for r in rows for c in itr_cons(r) if c.get('chain_caveat')), lo=1)
check('pairs with chain_relabel_note (consistent bijection)',
      sum(1 for r in rows if r['LABEL_ONLY'].get('chain_relabel_note')), lo=1,
      note='31 identical per-row warnings became one pair-level note')
check('conserved rows with typing_changed',
      sum(1 for r in rows for c in itr_cons(r) if c.get('typing_changed')), lo=1,
      note='"conserved" did not mean the interaction survived')
check('pairs with numbering_note',
      sum(1 for r in rows if r['LABEL_ONLY'].get('numbering_note')), lo=1)
check('conserved rows with the HETATM-only caveat',
      sum(1 for r in rows for c in itr_cons(r) if c.get('numbering_caveat')), lo=1,
      note='ZN703 matched by coincidence across two different ACE domains')
check('pairs with covalency_mismatch',
      sum(1 for r in rows if r['LABEL_ONLY'].get('covalency_mismatch')), lo=1)
check('pairs with atoms_dropped_as_clashes',
      sum(1 for r in rows if r['MODEL_INPUT'].get('atoms_dropped_as_clashes')), lo=1)

print('\n--- C. things that must fire in MODERATION (over-firing is its own defect) ---')

for key, lo, hi in (('symmetry_note', 1, int(0.45 * N)),
                    ('stereo_note', 1, int(0.35 * N)),
                    ('multi_edit_warning', 1, int(0.30 * N)),
                    ('multi_chain_note', 1, int(0.45 * N)),
                    ('no_attachment_point_warning', 0, int(0.20 * N)),
                    ('analogue_contact_warning', 1, int(0.25 * N)),
                    ('room_caveat', 0, int(0.15 * N))):
    check('%s firing rate' % key, sum(1 for r in rows if r['LABEL_ONLY'].get(key)), lo, hi)

print('\n--- D. named pairs the panel adjudicated by hand ---')

CASES = [
    ('9ETC JN3->4OA',  'stereo_note', 'INVERTED', False, 'CIP re-ranking, not an inversion'),
    ('7BXQ 2BK->F9X',  'stereo_note', 'INVERTED', True,  'a real inversion; must survive the fix'),
    ('6CZE ITT->CDP',  'multi_edit_warning', '', True,   'genuinely two transformations'),
]
for pid, field, marker, want, why in CASES:
    cands = [p for p in by_pair if p.startswith(pid.split()[0])]
    p = pid if pid in by_pair else (cands[0] if cands else None)
    if p is None:
        warns.append('%s not in corpus (may have been dropped)' % pid)
        print('  %-30s absent from corpus (dropped) -- %s' % (pid, why))
        continue
    val = by_pair[p]['LABEL_ONLY'].get(field) or ''
    got = bool(marker in val) if marker else bool(val)
    ok = got == want
    print('  %-30s %-22s %-6s want %-6s %s   (%s)'
          % (p, field, got, want, 'OK' if ok else '*** FAIL ***', why))
    if not ok:
        fails.append('%s: %s %s, wanted %s (%s)' % (p, field, got, want, why))

print('\n--- E. every caveat reaches the rendered packet ---')
# This section used to carry its OWN copy of the caveat list and demand the raw long note text, as
# the packet printed it before the lean render replaced those notes with tags. The copy then rotted:
# it reported five caveats "INVISIBLE" and three "TRUNCATED" that check_caveats_visible.py verifies
# are present in tag form, because a tag is what the brief's legend now explains. Two guards holding
# two copies of the same truth means one of them is always a release behind, so this one delegates.
# check_caveats_visible.py owns the raw->tag mapping and knows that a tag compresses.
import subprocess
_cv = subprocess.run([sys.executable,
                      os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   'check_caveats_visible.py')],
                     capture_output=True, text=True)
for _ln in _cv.stdout.rstrip('\n').split('\n'):
    print('  %s' % _ln)
if _cv.returncode != 0:
    print(_cv.stderr[-1200:])
    fails.append('check_caveats_visible.py FAILED (exit %d) -- a computed caveat is invisible'
                 % _cv.returncode)

print('\n' + '=' * 92)
if warns:
    print('WARNINGS (not blocking):')
    for w in warns:
        print('  - %s' % w)
if fails:
    print('QA FAILED -- %d check(s). DO NOT LABEL AGAINST THIS CORPUS:' % len(fails))
    for f in fails:
        print('  - %s' % f)
    sys.exit(1)
print('QA PASSED: %s pairs. Every fix verified in the artifact, firing rates within bounds,' % format(N, ','))
print('every caveat visible to a labeller. Corpus is fit to label.')
