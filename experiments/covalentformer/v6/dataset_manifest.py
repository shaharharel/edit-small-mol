#!/usr/bin/env python3
"""Stamp every adjudicated label with the GENERATION it belongs to, and write DATASET.md.

    python dataset_manifest.py              # report what is on disk, change nothing
    python dataset_manifest.py --write       # stamp the label files and regenerate DATASET.md

The corpus is being labelled in generations whose SENTENCES OBEY DIFFERENT RULES. Mixing them
silently would be the most expensive mistake available to us: a model trained on a mixture of
"cites a measured analogue distance" and "never cites one" learns to emit invented precision some
fraction of the time, and no metric computed after the fact can separate the two sources. So the
generation is stamped onto every record the day it is labelled, and the mapping lives here rather
than in anyone's memory.

Generations are assigned by BATCH NUMBER, which is safe because `make_full_batches.py` seeds its
shuffle with a fixed 7: batch N holds the same 100 pairs across every rebuild, so a batch number is
a durable identity. (That seed is load-bearing for this reason and is documented as such.)

  g1  batches 0001-0020   v1_retrospective
      The sentence REPORTS the analogue's crystal: "...to ring carbons at 3.61-3.69 A in the
      analogue's crystal". Names analogue-side residues with attribution, and cites measured
      analogue distances. 2,000 pairs. Measured compliance against the v2 rules: r2 8.2%, r3 23.0%,
      all-six 3.1% -- i.e. this generation is essentially 100% one style and cannot be mixed with
      what follows.

  g2  batches 0021-0040   v2_input_anchored
      Input-anchored: named residues come from the INPUT side only, and no number may be a measured
      analogue distance. The analogue half of the sentence carries direction and kind, never
      magnitude. NO POCKET ENVELOPE -- the input table still lists only each atom's nearest protein
      atom, so on roughly a quarter of pairs there is no legitimate residue to name for where a new
      group lands, and those consequence clauses are deliberately vaguer. 2,000 pairs.

  g3  batches 0041+       v3_input_anchored_envelope
      As g2, plus the pocket envelope: every residue with an atom within 8 A of the input ligand is
      listed, whether or not it is some atom's nearest neighbour. Measured on 70 pairs, an 8 A
      envelope recovers 94% of the residue mentions g2's rule 1 forbids (6 A recovers 75%, 10 A
      96%), which is what makes rule 1 satisfiable rather than merely strict.

The 50-pair re-adjudication that ran after two disclosure flags were added has been MERGED BACK
into batches 0001-0010, so there is exactly one record per pair and no loader can double-count. Each
of those 50 records carries `readjudicated_from`, `readjudicated_reason`, `verdict_changed`, and the
superseded reading under `previous_adjudication` -- nothing was discarded, and the record explains
its own history instead of depending on a second file continuing to exist. The original files are
kept at `label_out/archive_readjudicated/`, outside the glob.

WHAT THIS DOES NOT DO: it does not convert between generations. Rewriting g1 into v2 shape is a
chair pass over 1,715 usable sentences (strip the analogue numbers, re-anchor the ~24% naming an
off-table residue) and belongs in its own script with its own review, not in a stamping tool.
"""
import argparse
import collections
import glob
import json
import os
import re

HERE = os.path.dirname(os.path.abspath(__file__))

# (first batch, last batch, generation id, sentence rule in one line)
GENERATIONS = [
    (1, 20, 'g1_v1_retrospective',
     'reports the analogue crystal; names analogue-side residues; cites measured analogue distances'),
    (21, 40, 'g2_v2_input_anchored',
     'input-anchored; no measured analogue distance; no pocket envelope'),
    (41, 9000, 'g3_v2_input_anchored_envelope',
     'input-anchored; no measured analogue distance; 8 A pocket envelope in the packet'),
]
# Kept so the archived files still resolve to a generation if anyone points this at them.
READJ = (9001, 9002, 'g1_readjudicated_archived',
         'superseded files, merged back into 0001-0010; here only so the archive resolves')


def generation_of(batch_no):
    if READJ[0] <= batch_no <= READJ[1]:
        return READJ[2], READJ[3]
    for lo, hi, gid, rule in GENERATIONS:
        if lo <= batch_no <= hi:
            return gid, rule
    return 'unassigned', 'batch number outside every declared generation range'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--write', action='store_true',
                    help='stamp `label_generation` into each record and write DATASET.md')
    a = ap.parse_args()

    files = sorted(glob.glob(os.path.join(HERE, 'label_out', 'batch_*.json')))
    counts = collections.Counter()
    verdicts = collections.defaultdict(collections.Counter)
    stamped = 0
    for f in files:
        m = re.search(r'batch_(\d+)\.json$', f)
        if not m:
            continue
        gid, _rule = generation_of(int(m.group(1)))
        d = json.load(open(f))
        counts[gid] += len(d)
        for pid, v in d.items():
            verdicts[gid][v.get('verdict', '?')] += 1
            if a.write:
                v['label_generation'] = gid
                # No supersession to track any more: the 50 re-adjudicated pairs were merged back
                # into their own batches, so there is exactly one record per pair. Each carries its
                # own history in `previous_adjudication` rather than relying on a second file.
                v.pop('superseded_by', None)
                v.pop('authoritative', None)
        if a.write:
            json.dump(d, open(f, 'w'), indent=1)
            stamped += len(d)

    # Records and unique pairs must agree. They disagreed once -- 2,050 records over 2,000 pairs,
    # because a 50-pair re-adjudication lived in its own files -- and a loader pinned to a batch
    # range then silently used the superseded reading. The fixes are merged in now, so this is an
    # invariant rather than a statistic, and it is printed where anyone will see it drift.
    n_rec = sum(len(json.load(open(f))) for f in files)
    n_pair = len({p for f in files for p in json.load(open(f))})
    print('label files %d   records %s   unique pairs %s%s'
          % (len(files), format(n_rec, ','), format(n_pair, ','),
             '' if n_rec == n_pair else '   *** %d DUPLICATED ***' % (n_rec - n_pair)))
    for gid in list(dict.fromkeys([g[2] for g in GENERATIONS] + [READJ[2], 'unassigned'])):
        if counts[gid]:
            v = verdicts[gid]
            usable = v['ADOPT'] + v['REWRITE']
            print('  %-32s %5d pairs   ADOPT %4d  REWRITE %4d  REJECT %4d   usable %4d (%.1f%%)'
                  % (gid, counts[gid], v['ADOPT'], v['REWRITE'], v['REJECT'], usable,
                     100.0 * usable / counts[gid]))

    if not a.write:
        print('\n(report only -- pass --write to stamp the records and regenerate DATASET.md)')
        return

    with open(os.path.join(HERE, 'DATASET.md'), 'w') as fh:
        fh.write('# CovalentFormer v7 label corpus — generations\n\n')
        fh.write('Generated by `dataset_manifest.py`. Do not edit by hand.\n\n')
        fh.write('Every label record carries a `label_generation` field. The generations obey\n'
                 '**different sentence rules** and must not be mixed in training without a\n'
                 'deliberate decision: a model trained across g1 and g2 together learns to emit\n'
                 'invented precision some fraction of the time, and no metric computed afterwards\n'
                 'can separate the two sources.\n\n')
        fh.write('Generation is assigned by batch number. `make_full_batches.py` seeds its shuffle\n'
                 'with a fixed 7, so batch N holds the same 100 pairs across every rebuild and a\n'
                 'batch number is a durable identity.\n\n')
        fh.write('| generation | batches | pairs | usable | sentence rule |\n')
        fh.write('|---|---|---|---|---|\n')
        for lo, hi, gid, rule in GENERATIONS + [READJ]:
            if not counts[gid]:
                continue
            v = verdicts[gid]
            fh.write('| `%s` | %04d-%04d | %d | %d | %s |\n'
                     % (gid, lo, hi, counts[gid], v['ADOPT'] + v['REWRITE'], rule))
        fh.write('\n## One record per pair\n\n'
                 'Every label file holds exactly one record per pair, so a plain load of\n'
                 '`label_out/batch_*.json` is correct and cannot double-count.\n\n'
                 'Fifty g1 pairs were re-adjudicated after two identity-disclosure flags were added\n'
                 'to the packet (`COVALENCY MISMATCH IS UNRELIABLE HERE`, `HETERO PARTNER\n'
                 'RENUMBERED`). Those fixes were MERGED BACK into batches 0001-0010 rather than kept\n'
                 'as separate files. Each carries `readjudicated_from`, `readjudicated_reason`,\n'
                 '`verdict_changed`, and the superseded reading under `previous_adjudication`, so\n'
                 'nothing was discarded and the record explains its own history. 6 of the 50\n'
                 'verdicts changed and 0 of the 50 sentences came back identical -- including two\n'
                 'that had asserted a covalent-bond loss the geometry contradicts. The original\n'
                 'files are at `label_out/archive_readjudicated/`, outside the glob.\n\n'
                 '## What changed between g1 and g2, and why\n\n'
                 'g1 sentences report the analogue crystal and cite its measured distances. At\n'
                 'inference the model has no analogue structure and cannot measure anything, so\n'
                 'those targets teach it to invent precision. g2 splits the sentence into an\n'
                 'input-side MEASURED anchor and an analogue-side QUALITATIVE consequence:\n'
                 'direction and kind are predictable from the pocket, magnitude is not.\n\n'
                 'Measured on g1 against the g2 rules: 8.2% cite no analogue distance, 23.0% use no\n'
                 'analogue-provenance phrasing, 3.1% satisfy all six hard rules. So g1 is\n'
                 'effectively 100% one style. Converting it is a chair rewrite of 1,715 usable\n'
                 'sentences, not a re-label.\n\n'
                 '## What g3 adds\n\n'
                 'The 8 A pocket envelope. g2 forbids naming a residue absent from the input table,\n'
                 'but that table lists only each atom NEAREST protein atom, so on ~24% of pairs\n'
                 'there is no legitimate residue to name for where a new group lands. Measured on 70\n'
                 'pairs, an 8 A envelope recovers 94% of those mentions (6 A: 75%, 10 A: 96%).\n'
                 'g2 labels are therefore deliberately vaguer in their consequence clause than g3\n'
                 'labels will be, and that difference is a property of the evidence, not of the\n'
                 'labellers.\n')
    print('\nstamped %s records with `label_generation`; wrote DATASET.md' % format(stamped, ','))


if __name__ == '__main__':
    main()
