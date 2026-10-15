#!/usr/bin/env python3
"""Refuse to label from a batch file that the current renderer would not reproduce.

This exists because the same failure has now happened SEVEN times, and every time it took the same
shape: fix the source, forget the artifact people read.

  * the interaction column was cut at [:60], hiding `uncertain` in 100% of the rows carrying it
  * `[RESIDUE SUSPECT]` was annotated after the row snapshot and reached 0 of 6,462 pairs
  * the completeness flag was computed but never regenerated into the corpus
  * batch packets were left stale after an evidence rebuild
  * the brief cited fields the lean render had already cut
  * `unlabellable` was computed on 634 pairs and rendered nowhere at all
  * the conserved-row join fix restored 1,396 rows in one 100-pair batch, and the batch that three
    labellers then read was the one built 30 minutes BEFORE the fix -- so the defect all three
    independently reported top-three was still fully present in what they were given

No reviewer can see this class of bug. A labeller reads a plausible packet and has no way to know a
row is missing, and a passing QA run on the CORPUS says nothing about the FILE. So the check is
mechanical: re-render every pair in every batch from the live corpus with the live renderer, and
compare byte for byte. Any difference means the batch is stale and must be rebuilt before anyone
reads it.

Exit 0 = every batch reproduces. Exit 2 = at least one is stale. Nothing is labelled on a 2.
"""
# g5 variant: identical logic, paths repointed at the g5 corpus so the originals stay
# reproducible against g1-g4. Generated from check_packet_fresh.py by hand; keep in sync if that file moves.
import glob
import hashlib
import json
import os
import re
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from render_v7 import render  # noqa: E402

SP = os.path.dirname(os.path.abspath(__file__))

# Batches DELIBERATELY pinned to an earlier renderer, with the reason. A labelling run already
# reading a batch must not have the packet change underneath it: batches 0021-0040 are the g2
# generation, labelled without the pocket envelope, and the envelope landed while that run was in
# flight. They are restored to the pre-envelope render on purpose and will not match the live
# renderer until the run finishes and they are rebuilt.
#
# This list exists so the guard reports PINNED instead of STALE. A guard that goes red for a known
# reason is a guard people learn to ignore, and this one is the only thing standing between us and
# the eighth occurrence of labelling against a packet nobody checked.
PINNED = {
    'batch_%04d.txt' % b: 'g2 generation: labelled without the pocket envelope (run wf_e9c1c64f-d58)'
    for b in range(21, 41)
}
EV = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/g5_evidence.jsonl'

ev = {}
for line in open(EV):
    r = json.loads(line)
    ev[r['pair']] = r
print('corpus         %s pairs' % format(len(ev), ','))

batches = sorted(glob.glob(os.path.join(SP, 'batches_g5', 'batch_[0-9]*.txt')))
if not batches:
    print('\nNO BATCH FILES. Nothing to check -- build them with make_full_batches.py.')
    sys.exit(0)
print('batch files    %d' % len(batches))

# A pair's block runs from its `PAIR <id>` banner to the next one. The id is everything after
# `PAIR ` up to the two-space gap before `target`.
HDR = re.compile(r'^PAIR (.+?)   target ', re.M)

stale, missing, checked, pinned_seen = [], [], 0, []
for path in batches:
    if os.path.basename(path) in PINNED:
        pinned_seen.append(os.path.basename(path))
        continue
    txt = open(path).read()
    hits = list(HDR.finditer(txt))
    if not hits:
        stale.append((os.path.basename(path), 'no PAIR banner found -- not a rendered batch'))
        continue
    for i, m in enumerate(hits):
        pid = m.group(1).strip()
        # the block starts at the '=' rule above the banner
        start = txt.rfind('=' * 100, 0, m.start())
        start = start if start != -1 else m.start()
        end = txt.rfind('=' * 100, 0, hits[i + 1].start()) if i + 1 < len(hits) else len(txt)
        block = txt[start:end].strip('\n')
        checked += 1
        if pid not in ev:
            missing.append((os.path.basename(path), pid))
            continue
        want = render(ev[pid], lean=True).strip('\n')
        if hashlib.sha1(block.encode()).hexdigest() != hashlib.sha1(want.encode()).hexdigest():
            # report the first differing line, which is what tells you WHICH fix went unrendered
            a, b = block.split('\n'), want.split('\n')
            diff = next((j for j in range(max(len(a), len(b)))
                         if (a[j] if j < len(a) else None) != (b[j] if j < len(b) else None)), 0)
            stale.append((os.path.basename(path), pid,
                          'on disk: %s' % (a[diff][:88] if diff < len(a) else '<file ends>'),
                          'should be: %s' % (b[diff][:88] if diff < len(b) else '<block ends>'),
                          '%+d lines' % (len(b) - len(a))))

print('pairs checked  %d' % checked)
if pinned_seen:
    print('\nPINNED to an earlier renderer, not checked (%d files): %s'
          % (len(pinned_seen), ', '.join(sorted(pinned_seen)[:3])
             + (' ...' if len(pinned_seen) > 3 else '')))
    print('  reason: %s' % PINNED[pinned_seen[0]])
    print('  rebuild them once that run has finished, then delete them from PINNED.')

if missing:
    print('\nIN A BATCH BUT NOT IN THE CORPUS (%d): %s' % (len(missing), missing[:4]))
if stale:
    print('\nSTALE BATCHES: %d of %d pairs do not match the live renderer' % (len(stale), checked))
    seen = set()
    for s in stale[:6]:
        print('  %s' % (s[0] if s[0] in seen else s[0]))
        seen.add(s[0])
        for line in s[1:]:
            print('      %s' % line)
    files = sorted({s[0] for s in stale})
    print('\n  affected files (%d): %s%s'
          % (len(files), ', '.join(files[:8]), ' ...' if len(files) > 8 else ''))
    print('\nDO NOT LABEL. Rebuild with make_full_batches.py (or rerender_batches.py) first.')
    sys.exit(2)

print('\nFRESH: every pair in every batch reproduces byte for byte from the live corpus and the '
      'live renderer.')
