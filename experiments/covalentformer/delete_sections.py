#!/usr/bin/env python3
"""Delete whole <h2> sections from covalentformer_report.html by title, and renumber the rest.

The HTML is the source of truth -- mk_report.py has diverged from it and re-running that would
discard the results table, the walkthroughs and section 4 (see add_report_sections.py). So sections
are removed here, by matching their TITLE rather than their number: the numbers shift as soon as one
is deleted, so a number is not a stable handle for a second deletion.

Asserts every requested title matches exactly one section before writing anything, and reports any
internal #anchor that pointed into a deleted span, because a dead nav link is the usual way this
kind of edit breaks a long document.
"""
import argparse
import html
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.join(HERE, 'covalentformer_report.html')
H2 = re.compile(r'<h2><span class="sn">(\d+)</span>(.*?)</h2>')


def sections(s):
    hs = [(m.start(), m.group(1), html.unescape(re.sub('<[^>]+>', '', m.group(2))).strip())
          for m in H2.finditer(s)]
    out = []
    for k, (pos, n, t) in enumerate(hs):
        end = hs[k + 1][0] if k + 1 < len(hs) else len(s)
        out.append({'start': pos, 'end': end, 'n': n, 'title': t})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('titles', nargs='+', help='leading substring of each section title to delete')
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--dry-run', action='store_true')
    a = ap.parse_args()

    s = open(a.report).read()
    secs = sections(s)
    print('%d sections, %.2f MB' % (len(secs), len(s) / 1e6))

    doomed = []
    for want in a.titles:
        hit = [x for x in secs if x['title'].startswith(want)]
        assert len(hit) == 1, ('%r matched %d sections (%s); need exactly one'
                               % (want, len(hit), [x['title'] for x in hit]))
        doomed.append(hit[0])
    doomed.sort(key=lambda x: x['start'])

    # anchors that live inside a span about to go, and are linked from outside it
    dead = []
    for d in doomed:
        for m in re.finditer(r'id="([^"]+)"', s[d['start']:d['end']]):
            aid = m.group(1)
            outside = (s[:d['start']] + s[d['end']:])
            if ('#%s' % aid) in outside:
                dead.append((d['title'], aid))

    for d in doomed:
        print('  delete %-3s %-52s %6.0f KB' % (d['n'], d['title'][:52],
                                                (d['end'] - d['start']) / 1024))
    if dead:
        print('\n  WARNING: links from outside point into deleted sections:')
        for t, aid in dead:
            print('    #%-24s (in %s)' % (aid, t))
    if a.dry_run:
        print('\ndry run, nothing written')
        return 0

    for d in sorted(doomed, key=lambda x: -x['start']):      # back to front, so offsets hold
        s = s[:d['start']] + s[d['end']:]

    # renumber the sn badges in document order
    seq = iter(range(1, 99))
    s = H2.sub(lambda m: '<h2><span class="sn">%d</span>%s</h2>' % (next(seq), m.group(2)), s)

    open(a.report, 'w').write(s)
    left = sections(s)
    print('\n%d sections remain, %.2f MB' % (len(left), len(s) / 1e6))
    for x in left:
        print('  %-3s %s' % (x['n'], x['title'][:60]))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
