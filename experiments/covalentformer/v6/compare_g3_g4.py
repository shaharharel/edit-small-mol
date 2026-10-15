#!/usr/bin/env python3
"""Compare label generations on quality, scoring every generation with IDENTICAL code.

g4 is a second 7,069-pair pass over the same corpus, produced by the same pipeline and briefs but
by different labellers and chairs. The question is whether it is as good as g3, better, or worse --
and "better" has to mean something measurable before a chemist is asked to look at anything.

Both generations are scored by importing check_label_rules.check(), so the rule definitions, the
envelope handling and the residue regex are literally the same function. Nothing here re-implements
a rule.

Beyond the six rules, three things that have burned us before are measured directly:

  canned phrasing   the g1 failure signature was a model emitting "3.45 A" as a decorative token.
                    A label set with a few distances repeated across many sentences teaches that.
  duplicate         identical sentences reused across different pairs: the label is then not about
  sentences         the pair, and the model learns a template.
  lexical variety   type/token ratio over the sentence corpus. A set that says the same thing in
                    the same words everywhere supervises style, not chemistry.

Usage:
    python compare_g3_g4.py
"""
import collections
import json
import glob
import os
import re
import statistics
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import check_label_rules as K  # noqa: E402  -- the rule definitions, reused not reimplemented

G4_DIR = os.path.expanduser('~/Downloads/covalentformer_labelling_task/label_out')
G3_GENERATION = 'g3_v2_input_anchored_envelope'
NUM = re.compile(r'\d+\.\d+')


def load_g3():
    out = {}
    for f in glob.glob(os.path.join(HERE, 'label_out', 'batch_*.json')):
        for pid, v in json.load(open(f)).items():
            if v.get('label_generation') == G3_GENERATION:
                out[pid] = v
    return out


def load_g4():
    out = {}
    for f in sorted(glob.glob(os.path.join(G4_DIR, 'batch_*.json'))):
        for pid, v in json.load(open(f)).items():
            out[pid] = v
    return out


def load_evidence(pairs):
    want = set(pairs)
    recs = {}
    for line in open(K.EV):
        r = json.loads(line)
        if r['pair'] in want:
            recs[r['pair']] = r
    return recs


def assess(name, labels, recs):
    usable = {p: v for p, v in labels.items()
              if v.get('verdict') in ('ADOPT', 'REWRITE') and (v.get('final') or '').strip()}
    verdicts = collections.Counter(v.get('verdict') for v in labels.values())

    tally = collections.Counter()
    words, n_res, n_num = [], [], []
    sent_counts = collections.Counter()
    num_counts = collections.Counter()
    tokens = collections.Counter()
    n_scored = 0
    for pid, v in usable.items():
        rec = recs.get(pid)
        if rec is None:
            continue
        s = v['final'].strip()
        res = K.check(s, rec)
        n_scored += 1
        for k in K.RULES:
            tally[k] += bool(res[k])
        tally['ALL'] += all(res[k] for k in K.HARD_RULES)
        words.append(len(s.split()))
        n_res.append(len(K._res_all(s)))
        nums = NUM.findall(s)
        n_num.append(len(nums))
        sent_counts[s] += 1
        for x in nums:
            num_counts[x] += 1
        for w in re.findall(r"[a-z']+", s.lower()):
            tokens[w] += 1

    dup_sent = sum(c for c in sent_counts.values() if c > 1)
    top_num, top_num_c = (num_counts.most_common(1)[0] if num_counts else ('-', 0))
    ttr = len(tokens) / max(sum(tokens.values()), 1)

    print('\n=== %s ===' % name)
    print('  pairs labelled            %s' % format(len(labels), ','))
    print('  ADOPT %s  REWRITE %s  REJECT %s' %
          (format(verdicts['ADOPT'], ','), format(verdicts['REWRITE'], ','),
           format(verdicts['REJECT'], ',')))
    print('  usable                    %s (%.1f%%)'
          % (format(len(usable), ','), 100 * len(usable) / max(len(labels), 1)))
    print('  scored against evidence   %s' % format(n_scored, ','))
    if not n_scored:
        return
    for k in K.RULES:
        print('    %-5s %6s  (%5.1f%%)' % (k, format(tally[k], ','), 100 * tally[k] / n_scored))
    print('    %-5s %6s  (%5.1f%%)' % ('ALL', format(tally['ALL'], ','),
                                       100 * tally['ALL'] / n_scored))
    print('  words        mean %.1f  p50 %.0f  max %d' %
          (statistics.mean(words), statistics.median(words), max(words)))
    print('  residues/sent mean %.2f   distances/sent mean %.2f'
          % (statistics.mean(n_res), statistics.mean(n_num)))
    print('  duplicate sentences       %s of %s (%.1f%%)'
          % (format(dup_sent, ','), format(n_scored, ','), 100 * dup_sent / n_scored))
    print('  most-repeated distance    %s x%d (%.1f%% of sentences)'
          % (top_num, top_num_c, 100 * top_num_c / n_scored))
    print('  lexical type/token ratio  %.4f  (%s distinct words)' % (ttr, format(len(tokens), ',')))
    return {'usable': len(usable), 'n': len(labels), 'scored': n_scored,
            'rules': {k: tally[k] / n_scored for k in list(K.RULES) + ['ALL']},
            'words': statistics.mean(words), 'res': statistics.mean(n_res),
            'nums': statistics.mean(n_num), 'dup': dup_sent / n_scored, 'ttr': ttr}


def main():
    g3, g4 = load_g3(), load_g4()
    shared = set(g3) & set(g4)
    print('g3 %s pairs, g4 %s pairs, overlapping pair ids: %s'
          % (format(len(g3), ','), format(len(g4), ','), format(len(shared), ',')))
    recs = load_evidence(set(g3) | set(g4))
    print('evidence loaded for %s pairs' % format(len(recs), ','))
    a = assess('g3 (3,069-pair pass)', g3, recs)
    b = assess('g4 (7,069-pair pass)', g4, recs)
    if a and b:
        print('\n=== head to head, g4 minus g3 ===')
        for k in list(K.RULES) + ['ALL']:
            d = 100 * (b['rules'][k] - a['rules'][k])
            print('  %-5s %+6.1f pp' % (k, d))
        print('  usable %+.1f pp   words %+.1f   residues/sent %+.2f   dup %+.1f pp   ttr %+.4f'
              % (100 * (b['usable'] / b['n'] - a['usable'] / a['n']),
                 b['words'] - a['words'], b['res'] - a['res'],
                 100 * (b['dup'] - a['dup']), b['ttr'] - a['ttr']))
    json.dump({'g3': a, 'g4': b}, open(os.path.join(HERE, 'g3_vs_g4_quality.json'), 'w'), indent=2)


if __name__ == '__main__':
    main()
