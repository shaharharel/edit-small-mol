#!/usr/bin/env python
"""Build the external work package for v3 episode extraction.

WHAT WAS VALIDATED, AND BY WHAT. The v3 brief was tested on 10 papers with two independent
extractors and then judged by a 3-reviewer chemist panel carrying 30 hidden controls (10 real
v1 fragments, 10 mid-clause truncations, 10 rationales grafted onto another episode's
passage). The panel rejected **30/30 controls**, so its verdicts on the real episodes mean
something, and on 121 real episodes it returned:

    well-formed sentence   96.7%      (v1 purpose: 46%, and 0% named a change)
    states a design intent 84.3%
    briefable              90.1%      (v2: 86.0%)
    chemistry correct      76.0%      (v2: 50.9%)
    chemistry OVERSTATED   14.9%      (v2: 43.9%)
    chemistry WRONG         0.8%      (v2: 3.5%)
    usable (all three)     64.5%      (v2: 43.9%)

The 29-point drop in overstatement came from ONE change: deleting "informed by the quote and
the rest of the paper" from the rationale instruction and replacing it with an explicit ban on
importing groups, properties, measured values, outcomes or residues from anywhere but the
episode's own evidence quote. That is why the brief states it so heavily -- it is the single
load-bearing sentence in the document.

WHY TWO EXTRACTORS. Cross-extractor agreement was 84% on the 10-paper test, and on the v1
corpus agreement was a PERFECT fabrication filter: the agreed stratum carried 0.0% fabricated
rationales against 10-12% for records only one pass found. Agreement is the cheapest quality
signal we have, so every paper goes to two extractors and `agreement` is recorded per episode
rather than used to silently drop rows.

WHY THE PACKETS ARE RE-STRIPPED HERE. 88 of 98 packets in the previous round still carried
publisher furniture welded into body_text -- 4,229 fragments, 2.93% of all text -- because
`lit_paper_prepare.py`'s pattern required a date tail the real ACS stamp does not have. It
splices mid-sentence, and it is how a page number ("270" out of "J. Med. Chem. 2021, 64,
233-278") reached us as a compound label. The fix is applied at PACKET BUILD time here, which
is safe: these are fresh extractions. It must never be applied retroactively to a packet whose
quotes were already extracted -- tested, and it takes absent quotes from 0 to 441, because 836
quotes (4.87%) have the furniture inside them.

ONE REQUIREMENT ADDED SINCE THE TEST: an episode must name at least one compound. The brief
never required it, and the single real extractor disagreement in the 10-paper test was exactly
that -- one extractor emitted 4 contour-map episodes with empty compound fields, which cannot
be joined to a structure or verified against one. It costs 4% of episodes.

NO ANSWER KEYS, NO DOUBLE-LABEL MARKERS inside the handed-out directory, per standing project
policy: the second extractor's existence is in ASSIGNMENTS.tsv, never in a packet.

Usage:
  python scripts/extraction/build_episode_work_package.py \
      --packets ~/Downloads/covalentformer_labelling_task/lit_intent_100/packets \
      --out ~/Downloads/covalentformer_labelling_task/lit_episodes_v3_pkg \
      --extractors ex1 ex2
"""
import argparse
import csv
import json
import os
import re
import sys
from pathlib import Path

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from build_episode_extraction import BRIEF
import build_episode_extraction as _bee
BRIEF_SOURCE_DOC = _bee.__doc__ or ''            # noqa: E402  (the tested brief)
from lit_paper_prepare import strip_inline_furniture  # noqa: E402

START = """# Extracting medicinal-chemistry design episodes

## What this job is

You get one paper at a time, as JSON: `body_text`, `caption_text`, and metadata. You return
the **design episodes** in it. A design episode is one piece of design thinking:

    a HYPOTHESIS the authors held
      -> a STRUCTURAL CHANGE they made because of it
        -> the OUTCOME they measured

`BRIEF.md` is the full specification and is the authority. Read it before you start. The
single most important rule is in it, and it is the one we measure:

> **Every chemical specific in `design_rationale` must be supported by the `evidence_quote`
> you record -- not by the rest of the paper, and not by your own knowledge.**

The rest of the paper tells you *which* passage is a rationale and *which compounds* it
concerns. It must not supply chemistry. We check this, and in the previous round 44% of
rationales failed it by importing a group, a position, a property or a measured value from
somewhere else in the paper. A rationale that is vaguer than you would like but fully
supported by its quote is worth far more to us than a precise one that adds a detail the
quote does not carry.

## How to work

Two passes, in this order:

1. **Find where the authors ARGUE.** Introductions, design paragraphs, structure-based
   arguments, passages opening "In order to...", "We reasoned that...", "To address
   this...", "Based on the crystal structure...". **Do not start from the SAR tables** --
   a sentence comparing two compounds is the sentence least likely to carry a hypothesis.
2. **For each argument, find which compounds it produced**, using the paper's own labels.
   The labels may be a paragraph or a table away. If the text does not let you tie specific
   labels to the argument, say so -- do not attach the nearest number you can see.

## What a normal answer looks like

Most papers yield **a handful** of episodes. A dedicated SAR-optimisation paper may yield
10-16 because it argues in every subsection. A QSAR, modelling, virtual-screening or review
paper usually yields **0-2, and sometimes none** -- its authors made nothing, so there is no
hypothesis -> change -> outcome chain to recover. **Returning zero episodes for such a paper
is a correct answer and we would rather have it than invented ones.** Roughly a quarter to a
third of design-relevant statements in a typical paper carry an authorial rationale; the rest
is SAR bookkeeping and is not an episode.

We train models on this. An invented hypothesis is worse to us than no data.

## Submitting

Write one JSON file per paper into `submissions/` named exactly as the task file, then run:

    python check_my_submission.py submissions/<your file>.json

The checker verifies the schema, the 3-sentence / 80-word cap, that every episode names at
least one compound, that `evidence_quote` really occurs in the paper, and that the rationale's
wording is mostly traceable to that quote. Fix anything it reports before sending.

Your assignment is in `ASSIGNMENTS.tsv`.
"""

CHECKER = r'''#!/usr/bin/env python
"""Self-check a submission before sending it. Run: python check_my_submission.py <file>"""
import json, os, re, sys, unicodedata

REQ = ('reason_verbatim', 'design_rationale', 'change_from', 'change_to',
       'compounds_before', 'compounds_after', 'evidence_quote', 'evidence_section',
       'basis', 'intent_class', 'outcome')
SECT = {'introduction', 'design', 'results', 'discussion', 'conclusion', 'unclear'}
BASIS = {'experimental_structure', 'model', 'sar_only', 'unclear'}
CLASS = {'design_reason', 'post_hoc_explanation'}
STOP = set('the a an of to in on at and or for with that this it its was were is are be by '
           'from as not no than then so such which who whose into onto also both each more '
           'most less least very much many some any all we our they them their there'.split())


def norm(s):
    s = unicodedata.normalize('NFKC', s or '').lower()
    return re.sub(r'[^a-z0-9]+', '', s)


def words(s):
    return [w for w in re.findall(r'[a-z][a-z0-9-]{2,}', (s or '').lower()) if w not in STOP]


def main():
    if len(sys.argv) < 2:
        sys.exit('usage: python check_my_submission.py submissions/<file>.json')
    p = sys.argv[1]
    d = json.loads(open(p).read())
    # The task file sits in tasks/, next to submissions/. Getting this path wrong makes
    # the quote check silently pass everything, which is the most important check here.
    here = os.path.dirname(os.path.abspath(p))
    cand = [os.path.join(os.path.dirname(here), 'tasks', os.path.basename(p)),
            os.path.join(here, '..', 'tasks', os.path.basename(p)),
            os.path.join(os.path.dirname(here), os.path.basename(p))]
    task = next((c for c in cand if os.path.exists(c)), None)
    if task is None:
        print('WARNING: could not find the matching task file next to this submission, so '
              'the evidence_quote check is being SKIPPED. Run this script from inside the '
              'package directory so quotes can be verified.')
    body = ''
    if task:
        t = json.loads(open(task).read())
        body = norm((t.get('body_text') or '') + ' ' + (t.get('caption_text') or ''))
    probs = []
    if 'episodes' not in d:
        probs.append('no "episodes" key (an empty list is a valid answer, but the key '
                     'must be there)')
    eps = d.get('episodes') or []
    for i, e in enumerate(eps):
        tag = f'episode {i + 1}'
        for k in REQ:
            if k not in e:
                probs.append(f'{tag}: missing field "{k}"')
        dr = (e.get('design_rationale') or '').strip()
        if not dr:
            probs.append(f'{tag}: design_rationale is empty')
            continue
        n_sent = len([x for x in re.split(r'[.!?]+\s', dr) if x.strip()])
        n_word = len(dr.split())
        if n_sent > 3:
            probs.append(f'{tag}: design_rationale has {n_sent} sentences, cap is 3')
        if n_word > 80:
            probs.append(f'{tag}: design_rationale has {n_word} words, cap is 80')
        q = (e.get('evidence_quote') or '').strip()
        if not q:
            probs.append(f'{tag}: evidence_quote is empty')
        elif body and norm(q) not in body:
            probs.append(f'{tag}: evidence_quote does not occur in this paper '
                         f'(check for typos, and quote verbatim: "{q[:60]}...")')
        if q:
            cw = words(dr)
            hay = (q or '').lower()
            if cw:
                g = sum(1 for w in cw if w in hay) / len(cw)
                if g < 0.45:
                    probs.append(
                        f'{tag}: only {g:.0%} of the rationale\'s wording appears in its own '
                        f'evidence_quote. Most of it should. If you needed facts from '
                        f'elsewhere in the paper, the rationale is overstating what the '
                        f'quote supports -- cut it back.')
        cb = e.get('compounds_before') or []
        ca = e.get('compounds_after') or []
        if not isinstance(cb, list) or not isinstance(ca, list):
            probs.append(f'{tag}: compounds_before/after must be lists')
        elif not cb and not ca:
            probs.append(f'{tag}: names no compounds at all. An episode must name at least '
                         f'one compound label, or it cannot be linked to a structure.')
        if e.get('evidence_section') not in SECT:
            probs.append(f'{tag}: evidence_section must be one of {sorted(SECT)}')
        if e.get('basis') not in BASIS:
            probs.append(f'{tag}: basis must be one of {sorted(BASIS)}')
        if e.get('intent_class') not in CLASS:
            probs.append(f'{tag}: intent_class must be one of {sorted(CLASS)}')
        rv = (e.get('reason_verbatim') or '').strip()
        if rv and len(rv.split()) < 5:
            probs.append(f'{tag}: reason_verbatim looks like a fragment ("{rv}"). It must be '
                         f'a COMPLETE SENTENCE in the authors\' words.')
    if probs:
        print(f'{len(probs)} problem(s):')
        for x in probs:
            print('  -', x)
        sys.exit(1)
    print(f'OK: {len(eps)} episode(s), schema valid, caps respected, quotes found, '
          f'every episode names a compound.')
    if not eps:
        print('(zero episodes is a valid answer for a review, QSAR or modelling paper)')


if __name__ == '__main__':
    main()
'''


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--packets', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--extractors', nargs='+', default=['ex1', 'ex2'])
    ap.add_argument('--max-body', type=int, default=110000)
    ap.add_argument('--limit', type=int)
    a = ap.parse_args()

    out = Path(os.path.expanduser(a.out))
    if out.exists():
        sys.exit(f'{out} exists -- refusing to overwrite a handed-out package')
    (out / 'tasks').mkdir(parents=True)
    (out / 'submissions').mkdir()

    src = sorted(Path(os.path.expanduser(a.packets)).glob('*.json'))
    if a.limit:
        src = src[:a.limit]
    n_frag = n_chars = 0
    papers = []
    for f in src:
        try:
            d = json.loads(f.read_text())
        except Exception:
            continue
        body, nf, nc = strip_inline_furniture(d.get('body_text') or '')
        caps, nf2, nc2 = strip_inline_furniture(d.get('caption_text') or '')
        n_frag += nf + nf2
        n_chars += nc + nc2
        if len(body) < 4000:
            continue
        doc = f'doc_{d.get("doc_id")}'
        papers.append(doc)
        for ex in a.extractors:
            (out / 'tasks' / f'{doc}__{ex}.json').write_text(json.dumps({
                'instructions': BRIEF, 'extractor': ex, 'paper': doc,
                'title': d.get('title'), 'journal': d.get('journal'), 'year': d.get('year'),
                'doi': d.get('doi'),
                'body_text': body[:a.max_body], 'caption_text': caps[:8000],
            }, indent=1))

    (out / 'BRIEF.md').write_text(
        '# Design-episode extraction brief\n\nThis is the specification the checker '
        'enforces.\n\n```\n' + BRIEF + '\n```\n')
    (out / 'START_HERE.md').write_text(START)
    (out / 'check_my_submission.py').write_text(CHECKER)
    with open(out / 'ASSIGNMENTS.tsv', 'w', newline='') as fh:
        w = csv.writer(fh, delimiter='\t')
        w.writerow(['extractor_id', 'paper', 'task_file'])
        for doc in papers:
            for ex in a.extractors:
                w.writerow([ex, doc, f'tasks/{doc}__{ex}.json'])
    # BUILD.json IS DERIVED, NOT TYPED. These fields were hardcoded to v3's text and v3's
    # panel numbers, and every later package shipped claiming to be v3 "with two independent
    # extractors per paper" -- wrong on the version, wrong on the extractor count, and the
    # second half is exactly the double-labelling disclosure the QA below looks for. The
    # version now comes from the brief itself and the counts from the arguments.
    import re as _re
    _ver = _re.search(r'v(\d+(?:\.\d+)?) extraction', BRIEF_SOURCE_DOC or '')
    (out / 'BUILD.json').write_text(json.dumps({
        'what': 'design-episode extraction, brief %s' % (_ver.group(0) if _ver else 'unknown'),
        'brief_version': _ver.group(1) if _ver else 'unknown',
        'papers': len(papers), 'extractors': a.extractors,
        'extractors_per_paper': len(a.extractors),
        'tasks': len(papers) * len(a.extractors),
        'packets_from': str(Path(os.path.expanduser(a.packets))),
        'furniture_stripped': {'fragments': n_frag, 'chars': n_chars},
    }, indent=2))

    # ---- QA ----
    probs = []
    for f in out.rglob('*'):
        if f.is_file() and 'KEY' in f.name.upper():
            probs.append(f'answer key inside the package: {f.name}')
    for f in (out / 'tasks').glob('*.json'):
        d = json.loads(f.read_text())
        # SCAN ONLY THE FIELDS WE AUTHOR. Four earlier versions grepped the whole task
        # file, i.e. the paper itself, and fired on ordinary chemistry vocabulary:
        # "kind", "Downloaded from", "Annexin V-FITC/PI double-labeled cell cytometry",
        # and "the aqueous phase was extracted twice with CH2Cl2". A paper cannot
        # disclose OUR workflow -- only the instructions and metadata we add can -- so
        # body_text and caption_text are out of scope by construction.
        ours = ' '.join(str(d.get(k) or '') for k in
                        ('instructions', 'extractor', 'paper', 'title', 'journal', 'doi'))
        if re.search(r'(second|another|duplicate|two)\s+extractors?\b'
                     r'|double\s+extraction|extracted\s+by\s+two', ours, re.I):
            probs.append(f'{f.name} discloses the double extraction')
        if '/Users/' in ours:
            probs.append(f'{f.name} contains a home path')
        body = (d.get('body_text') or '')
        if re.search(r'Downloaded\s+(from|via)\s+(https?://)?(pubs|www)\.', body):
            probs.append(f'{f.name} still contains publisher furniture')
    print(f'package: {out}')
    print(f'  {len(papers)} papers x {len(a.extractors)} extractors = '
          f'{len(papers)*len(a.extractors)} tasks')
    print(f'  publisher furniture stripped: {n_frag:,} fragments, {n_chars:,} chars')
    print(f'  BRIEF.md, START_HERE.md, check_my_submission.py, ASSIGNMENTS.tsv, BUILD.json')
    if probs:
        print('\nQA PROBLEMS:')
        for p in probs[:10]:
            print('  ', p)
        sys.exit(1)
    print('  QA: no key, no double-extraction hint, no home paths, no residual furniture.')


if __name__ == '__main__':
    main()
