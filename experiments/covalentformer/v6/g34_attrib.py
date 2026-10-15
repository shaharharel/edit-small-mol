#!/usr/bin/env python3
"""Attribution quality of the LABELS themselves, g3 vs g4, plus a size-matched lexical check.

Rule compliance says the sentence has the right SHAPE. It does not say the numbers are attached to
the right residues -- the failure three chemists found dominant in model output, and the one thing
a label set must get right if it is to teach it.

  1a  every distance in the sentence exists somewhere in that pair's input contact table
  1b  every (residue, distance) the sentence asserts co-occurs in ONE table row

1b uses a nearest-number-after-the-residue heuristic, so it under-counts when a sentence separates
the two across clauses. That floor applies equally to both generations, which is what makes the
comparison fair even though the absolute number is not 1.0 for perfect labels.

Type/token ratio is also recomputed on EQUAL-SIZED samples: TTR falls mechanically as a corpus
grows, so comparing 2,644 sentences against 6,092 would have manufactured a difference.
"""
import collections
import glob
import json
import os
import random
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import check_label_rules as K  # noqa: E402

G4_DIR = os.path.join(HERE, 'label_out_g4')
NUM = re.compile(r'\d+\.\d+')
RES = re.compile(r'\b([A-Z]{3})(\d+)')
AA = K.AA


def table_rows(rec):
    out = []
    for a in rec['MODEL_INPUT'].get('atoms') or []:
        for dk, fk in (('distance_A', 'faces'), ('polar_distance_A', 'polar_partner')):
            if a.get(dk) is not None and a.get(fk):
                out.append((a['atom'], a[fk], round(a[dk], 2)))
    return out


def attrib(sent, tbl):
    nums = {round(float(x), 2) for x in NUM.findall(sent or '')}
    tnums = {d for _, _, d in tbl}
    a_ok = all(n in tnums for n in nums) if nums else None
    pairs = []
    for m in RES.finditer((sent or '').upper()):
        if m.group(1) not in AA:
            continue
        res = '%s%s' % (m.group(1), m.group(2))
        nm = NUM.search((sent or '')[m.end():m.end() + 80])
        if nm:
            pairs.append((res, round(float(nm.group(0)), 2)))
    ok = sum(1 for r, d in pairs
             if any(f.split('.')[0] == r and abs(dd - d) < 0.005 for _, f, dd in tbl))
    return a_ok, ok, len(pairs)


def load(which):
    out = {}
    if which == 'g3':
        for f in glob.glob(os.path.join(HERE, 'label_out', 'batch_*.json')):
            for pid, v in json.load(open(f)).items():
                if v.get('label_generation') == 'g3_v2_input_anchored_envelope':
                    out[pid] = v
    else:
        for f in sorted(glob.glob(os.path.join(G4_DIR, 'batch_*.json'))):
            out.update(json.load(open(f)))
    return {p: v for p, v in out.items()
            if v.get('verdict') in ('ADOPT', 'REWRITE') and (v.get('final') or '').strip()}


def main():
    g3, g4 = load('g3'), load('g4')
    want = set(g3) | set(g4)
    recs = {}
    for line in open(K.EV):
        r = json.loads(line)
        if r['pair'] in want:
            recs[r['pair']] = r

    res = {}
    for name, labels in (('g3', g3), ('g4', g4)):
        a_ok = a_n = b_ok = b_n = 0
        no_dist = 0
        for pid, v in labels.items():
            rec = recs.get(pid)
            if rec is None:
                continue
            tbl = table_rows(rec)
            ao, bo, bn = attrib(v['final'], tbl)
            if ao is None:
                no_dist += 1
            else:
                a_n += 1
                a_ok += ao
            b_ok += bo
            b_n += bn
        res[name] = (a_ok, a_n, b_ok, b_n, no_dist, len(labels))
        print('%s  n=%s' % (name, format(len(labels), ',')))
        print('   1a every number in the table   %s/%s (%.1f%%)'
              % (format(a_ok, ','), format(a_n, ','), 100 * a_ok / max(a_n, 1)))
        print('   1b (residue,distance) co-occur %s/%s (%.1f%%)'
              % (format(b_ok, ','), format(b_n, ','), 100 * b_ok / max(b_n, 1)))
        print('   sentences citing NO distance   %s (%.1f%%)'
              % (format(no_dist, ','), 100 * no_dist / max(len(labels), 1)))

    # size-matched lexical variety
    rng = random.Random(0)
    n = min(len(g3), len(g4))
    print('\nsize-matched lexical variety (%s sentences each):' % format(n, ','))
    for name, labels in (('g3', g3), ('g4', g4)):
        sample = rng.sample(sorted(labels), n)
        tok = collections.Counter()
        for p in sample:
            for w in re.findall(r"[a-z']+", labels[p]['final'].lower()):
                tok[w] += 1
        print('   %s  TTR %.4f  distinct %s  tokens %s'
              % (name, len(tok) / max(sum(tok.values()), 1), format(len(tok), ','),
                 format(sum(tok.values()), ',')))


if __name__ == '__main__':
    main()
