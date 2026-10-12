#!/usr/bin/env python3
"""Assert that every caveat the evidence computes actually REACHES the rendered packet.

This exists because the same bug shipped twice. Round 2: the `uncertain` flag was computed on every
row and then cut off by a 60-character column, invisible in 1,116 of 1,116 cases. Round 4: the
`[RESIDUE SUSPECT]` annotation mutated `rows` four hundred lines after `model_rows` had already been
copied, so it reached 0 of 6,462 pairs -- and I reported it as working because the SUMMARY field was
populated while the per-row text was not.

A caveat that is computed and not shown is worse than no caveat: it makes the audit look clean while
the labeller reads unqualified evidence. So the rule is mechanical -- if a field or marker exists in
the corpus, it must appear in the render of a pair that has it, and this script fails loudly when it
does not.
"""
import collections, json, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_v7 import render, _unlab

EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/v7_evidence.jsonl'

# (where it lives, key, a distinctive string that must survive into the rendered text)
CAVEATS = [
    # `unlabellable` was absent from this list, which is the whole reason it reached 634 pairs
    # (9.0% of the corpus) while rendering nowhere. A guard that does not list a field cannot
    # protect it -- any new pair-level field in clean_v7.py belongs here the day it is added.
    ('LABEL_ONLY', 'unlabellable', 'PAIR FLAGGED UNLABELLABLE'),
    ('LABEL_ONLY', 'covalency_mismatch', 'COVALENCY MISMATCH'),
    ('LABEL_ONLY', 'symmetry_note', 'SYMMETRY'),
    ('LABEL_ONLY', 'stereo_note', 'STEREO'),
    ('LABEL_ONLY', 'room_caveat', 'ROOM PROBE CONTRADICTS'),
    ('LABEL_ONLY', 'multi_chain_note', 'MULTI-CHAIN POCKET'),
    ('LABEL_ONLY', 'terminus_note', 'CHAIN TERMINUS CONTACT'),
    ('LABEL_ONLY', 'no_attachment_point_warning', 'NO ATTACHMENT POINT FOUND'),
    ('LABEL_ONLY', 'multi_edit_warning', 'TWO TRANSFORMATIONS'),
    ('LABEL_ONLY', 'analogue_contact_warning', 'INCOMPLETE TABLE'),
    ('LABEL_ONLY', 'numbering_note', 'NO SHARED RESIDUE NUMBERING'),
    ('LABEL_ONLY', 'chain_relabel_note', 'CHAINS RELABELLED CONSISTENTLY'),
    ('MODEL_INPUT', 'atoms_dropped_as_clashes', 'OMITTED AS CLASHES'),
    ('MODEL_INPUT', 'clash_note', None),
    ('MODEL_INPUT', 'input_atom_warning', None),
    ('MODEL_INPUT', 'suspect_residues', 'SUSPECT RESIDUES'),
]
# markers that live inside a row rather than in a top-level field
ROW_MARKERS = [
    ('conserved_interactions', 'chain_caveat', 'CHAINS '),
    ('conserved_interactions', 'typing_changed', 'TYPING CHANGED'),
    ('conserved_interactions', 'moved_caveat', 'MOVED atom'),
    ('conserved_interactions', 'numbering_caveat', 'HETATM-LABEL MATCH ONLY'),
]
# in-string suffixes are now tags; check the TAG appears once per raw occurrence
TEXT_MARKERS = [('treat the crystal model as uncertain', 'uncertain]'),
                ('nearest INDEPENDENT contact', '[INDEP'),
                ('RESIDUE SUSPECT', '[SUSPECT'),
                ('donor/acceptor INCOMPATIBLE', '[INCOMPAT')]

rows = [json.loads(l) for l in open(EV)]
print('corpus: %s pairs' % format(len(rows), ','))

fail = []

def pick(pred, n=3):
    out = []
    for r in rows:
        if pred(r):
            out.append(r)
            if len(out) >= n:
                break
    return out

print('\ntop-level caveat fields')
for blk, key, marker in CAVEATS:
    have = pick(lambda r, b=blk, k=key: r[b].get(k))
    total = sum(1 for r in rows if r[blk].get(key))
    if not have:
        print('  %-30s %6d pairs   (none to test)' % (key, total))
        continue
    shown = 0
    for r in have:
        txt = render(r)
        probe = marker or str(r[blk][key])[:40]
        if probe and probe in txt:
            shown += 1
    ok = shown == len(have)
    print('  %-30s %6d pairs   rendered %d/%d  %s'
          % (key, total, shown, len(have), 'OK' if ok else '*** NOT VISIBLE ***'))
    if not ok:
        fail.append(key)

print('\nper-row caveats')
for field, key, marker in ROW_MARKERS:
    have = pick(lambda r, f=field, k=key: any(c.get(k) for c in (r['LABEL_ONLY'].get(f) or [])))
    total = sum(1 for r in rows for c in (r['LABEL_ONLY'].get(field) or []) if c.get(key))
    if not have:
        print('  %-30s %6d rows    (none to test)' % (key, total))
        continue
    shown = sum(1 for r in have if marker in render(r))
    ok = shown == len(have)
    print('  %-30s %6d rows    rendered %d/%d  %s'
          % (key, total, shown, len(have), 'OK' if ok else '*** NOT VISIBLE ***'))
    if not ok:
        fail.append(key)

print('\nin-string markers -> their tags, checked PER ROW (a tag compresses, so counts legitimately fall)')
for _raw, _tag in TEXT_MARKERS:
    checked = missed = 0
    ex = []
    for r in rows[:400]:
        hits = [a['atom'] for a in r['MODEL_INPUT']['atoms']
                if _raw in (a.get('interaction') or '')]
        if not hits:
            continue
        txt = render(r)
        lines = {ln.split()[0]: ln for ln in txt.split('\n')
                 if ln and not ln.startswith(' ') and ln.split()}
        for at in hits:
            checked += 1
            ln = lines.get(at, '')
            if _tag not in ln:
                missed += 1
                if len(ex) < 2:
                    ex.append('%s %s -> %s' % (r['pair'], at, (ln or '(row absent)')[:90]))
    ok = checked and not missed
    print('  %-30s -> %-12s %d/%d rows  %s'
          % (_raw, _tag, checked - missed, checked, 'OK' if ok else '*** MISSING ***'))
    for e in ex:
        print('        %s' % e)
    if missed:
        fail.append('%s (tag %s, %d rows)' % (_raw, _tag, missed))

# Every DISTINCT unlabellable reason, not just the flag. The flag alone is not actionable -- a
# labeller told only "unlabellable" cannot tell a cofactor swap from a failed atom mapping, and the
# two call for different sentences. The reasons are shortened in lean mode, so this checks that each
# one's rendered form is present on every pair that carries it, which is what the generic probe
# above (it looks for the raw text) cannot do.
print('\nunlabellable reasons, each checked on every pair carrying it')
_reasons = collections.Counter()
for r in rows:
    for x in (r['LABEL_ONLY'].get('unlabellable_reason') or []):
        _reasons[x] += 1
for _raw, _n in _reasons.most_common():
    have = [r for r in rows if _raw in (r['LABEL_ONLY'].get('unlabellable_reason') or [])]
    short = _unlab([_raw], True)[0]
    missed = sum(1 for r in have if short not in render(r))
    print('  %-62s %4d pairs  %s' % (short[:62], _n,
                                     'OK' if not missed else '*** %d NOT VISIBLE ***' % missed))
    if missed:
        fail.append('unlabellable_reason "%s" (%d pairs)' % (short[:40], missed))

print()
if fail:
    print('FAILED: these caveats are computed but not visible to a labeller: %s' % ', '.join(fail))
    sys.exit(1)
print('PASS: every computed caveat reaches the rendered packet.')
