#!/usr/bin/env python3
"""Progress and stall check for the 283-batch labelling run.

Written as a script rather than an inline one-liner because it runs unattended every 20 minutes
while the user sleeps, and it has to answer two questions without a human reading it:

  1. how far along are we
  2. is it still moving

For (2) it compares the current file count against the previous check, which it stores in
.label_progress.json next to itself. Two consecutive checks with no new file means the run has
stalled and should be resumed -- a count that merely LOOKS low is not evidence of a stall, because
a 25-pair batch takes 8-12 minutes and the concurrency cap is ~14 agents.
"""
import collections, glob, json, os, sys

SP = os.path.dirname(os.path.abspath(__file__))
STATE = os.path.join(SP, '.label_progress.json')
TOTAL_BATCHES = 283
TOTAL_PAIRS = 7069

raw = sorted(glob.glob(os.path.join(SP, 'label_out', 'raw', 'batch_[0-9]*_lab?.json')))
chair = sorted(glob.glob(os.path.join(SP, 'label_out', 'batch_[0-9][0-9][0-9][0-9].json')))

verd = collections.Counter()
pairs = 0
unparseable = []
empty_final = 0
for p in chair:
    try:
        d = json.load(open(p))
    except Exception as e:
        unparseable.append('%s (%s)' % (os.path.basename(p), str(e)[:40]))
        continue
    for _k, r in d.items():
        pairs += 1
        verd[(r.get('verdict') or '?').split()[0].upper()] += 1
        if not (r.get('final') or '').strip():
            empty_final += 1

import glob as _g
import time
now = time.time()
_all = raw + chair
newest_age_min = min((now - os.path.getmtime(f)) / 60.0 for f in _all) if _all else None
wrote_recently = sum(1 for f in _all if now - os.path.getmtime(f) < 900)

# the workflow journal heartbeats on every agent start/result, cached or live
WF = os.path.expanduser(
    '~/.claude/projects/-Users-shaharharel-Documents-github-edit-small-mol/'
    'a64ce54a-c2cc-42bc-bc54-bfe1ad41d081/subagents/workflows')
journals = _g.glob(os.path.join(WF, '*', 'journal.jsonl'))
journal_age_min = min((now - os.path.getmtime(j)) / 60.0 for j in journals) if journals else None

prev = {}
if os.path.exists(STATE):
    try:
        prev = json.load(open(STATE))
    except Exception:
        prev = {}

n_raw, n_chair = len(raw), len(chair)
# LIVENESS IS A TIMESTAMP QUESTION, NOT A COUNT QUESTION.
files_alive = newest_age_min is not None and newest_age_min < 20
journal_alive = journal_age_min is not None and journal_age_min < 10
alive = files_alive or journal_alive
moved = alive
stalls = 0 if alive else int(prev.get('stalls', 0)) + 1

print('LABELLING PROGRESS')
print('  labeller files   %4d / %d   (%.1f%%)' % (n_raw, TOTAL_BATCHES * 3, 100.0 * n_raw / (TOTAL_BATCHES * 3)))
print('  chair batches    %4d / %d   (%.1f%%)' % (n_chair, TOTAL_BATCHES, 100.0 * n_chair / TOTAL_BATCHES))
print('  pairs labelled   %4d / %d   (%.1f%%)' % (pairs, TOTAL_PAIRS, 100.0 * pairs / TOTAL_PAIRS))
if verd:
    tot = sum(verd.values()) or 1
    print('  verdicts         ADOPT %d (%.0f%%)  REWRITE %d (%.0f%%)  REJECT %d (%.0f%%)'
          % (verd['ADOPT'], 100.0 * verd['ADOPT'] / tot,
             verd['REWRITE'], 100.0 * verd['REWRITE'] / tot,
             verd['REJECT'], 100.0 * verd['REJECT'] / tot))
print('  since last check  raw %+d   chair %+d'
      % (n_raw - int(prev.get('raw', 0)), n_chair - int(prev.get('chair', 0))))
print('  newest write      %s'
      % ('%.1f min ago' % newest_age_min if newest_age_min is not None else 'no files yet'))
print('  files in last 15m %d' % wrote_recently)
print('  workflow journal  %s'
      % ('%.1f min ago' % journal_age_min if journal_age_min is not None else 'none found'))
print('  verdict           %s%s'
      % ('ALIVE' if alive else 'DEAD',
         '  (replaying cache -- journal moving, no file writes yet)'
         if (journal_alive and not files_alive) else ''))
print('  consecutive no-progress checks: %d' % stalls)

if unparseable:
    print('\n  UNPARSEABLE CHAIR FILES (%d): %s' % (len(unparseable), ', '.join(unparseable[:5])))
if empty_final:
    print('  chair rows with an EMPTY final sentence: %d  (CHAIR_FULL.md forbids this)' % empty_final)

# batches whose labellers are all in but which have no chair output -- normal transiently, a
# symptom if it persists
done_raw = collections.Counter()
for p in raw:
    done_raw[os.path.basename(p).split('_lab')[0]] += 1
ready = {b for b, c in done_raw.items() if c == 3}
have_chair = {os.path.basename(p)[:-5] for p in chair}
awaiting = sorted(ready - have_chair)
print('  batches with 3/3 labellers but no chair yet: %d' % len(awaiting))

json.dump({'raw': n_raw, 'chair': n_chair, 'pairs': pairs, 'stalls': stalls},
          open(STATE, 'w'))

if not alive:
    print('\nSTALLED: no file for %.0f min AND no journal activity for %s min. Resume NOW.'
          % (newest_age_min or 0,
             ('%.0f' % journal_age_min) if journal_age_min is not None else 'inf'))
    sys.exit(2)
print('\nmoving')
