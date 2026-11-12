#!/usr/bin/env python
"""Turn a directory of medchem PDFs into per-paper extraction packets.

    python scripts/extraction/lit_paper_prepare.py --pdf-dir ~/Downloads/temp2 \
        --out data/lit_extract/round1 --max-pairs 20

WHAT A PACKET IS, AND WHY IT IS PER PAPER RATHER THAN PER PAIR. The first literature round handed
the extractor one pair at a time. That re-reads a paper once per pair -- the median paper here holds
~46 candidate SAR pairs -- and it is also why the extractions came back at series scope: asked
about one pair in isolation, the model answers with the only aim the paper states globally. A packet
carries the paper ONCE with a batch of its own pairs, so the series aim can be stated once and each
edit attributed separately.

TEXT EXTRACTION IS PyMuPDF GEOMETRY, NOT `pdftotext -layout`. That is on record in this project:
-layout interleaves the two columns of a typical J Med Chem page, so a sentence from column 1 is
spliced to one from column 2 and the result reads plausibly while being nobody's sentence. Blocks
are sorted by column then by vertical position here instead.

FIGURE AND TABLE CAPTIONS ARE KEPT SEPARATELY. Pocket interactions are frequently stated only in a
figure caption ("Figure 4. Compound 12 bound to the ATP site, showing the hydrogen bond to Asp86")
while the body text says only "improved potency". If the caption channel is where the pocket lives,
a body-only extractor will keep returning the 72%-pocket-free sentences we measured. Captions are
therefore a named channel, so their contribution can be counted rather than assumed.

WHAT IS NOT DONE HERE. No intent is extracted and no potency is read from the paper. Potency comes
from ChEMBL structured data; the paper is read only for intent. Keeping that boundary is what makes
the pool a million pairs rather than a few thousand.
"""
from __future__ import annotations

import argparse
import collections
import csv
import itertools
import json
import math
import os
import re
import sqlite3
import sys

POT = ('IC50', 'Ki', 'Kd', 'EC50', 'AC50', 'Potency')
CAP = re.compile(r'^\s*(figure|fig\.|scheme|table|chart)\s*\d+', re.I)
RESIDUE = re.compile(
    r'\b(Ala|Arg|Asn|Asp|Cys|Gln|Glu|Gly|His|Ile|Leu|Lys|Met|Phe|Pro|Ser|Thr|Trp|Tyr|Val)'
    r'[\s\-]?(\d{1,4})\b')
# Reviews and QSAR/methods papers carry many compounds and no original design intent, because the
# authors did not make them. They dominate a ranking by compound count, so they are flagged, not
# silently kept: the extractor is told what kind of paper it is reading.
REVIEW_TITLE = re.compile(
    r'\b(recent (advances|developments|progress)|a review|review of|an overview|'
    r'current (status|perspectives)|quantitative structure|QSAR|3D-QSAR|'
    r'molecular similarity matri|docking study of)\b', re.I)


REVIEW_BODY = re.compile(
    r'^\s*(review(\s+article)?|mini[- ]?review|invited review|perspective)\s*$|'
    r'\breview article\b|\bthis review\b|\bin this review\b|\bwe review\b', re.I | re.M)


def _kind_evidence(body):
    """The span that establishes the paper kind, so the call is auditable."""
    m = REVIEW_BODY.search(body[:20000])
    return m.group(0).strip()[:80] if m else ''


def _paper_kind(title, body):
    if REVIEW_BODY.search(body[:20000]):
        return 'review'
    if REVIEW_TITLE.search(title or ''):
        return 'review_or_qsar'
    return 'primary'

# Download furniture that the publisher stamps onto every page. It lands in the text layer
# as its own block, and because the block sits in the left margin it is pulled into the
# reading order IN THE MIDDLE OF A SENTENCE -- one case split "The" from "liable ester group".
#
# Measured on this corpus: 1,211 fragments across 61 of 98 packets, roughly one per page of
# every ACS paper, because these PDFs were fetched through an institutional subscription. It
# is not cosmetic: it breaks verbatim quote verification on the majority of the corpus, and
# every group working on the batch independently had to loosen its quote checker to cope.
# Dropping the block is correct -- it is not part of the article.
FURNITURE = re.compile(
    r'(Downloaded via|by [A-Z][A-Z\- ]+ user on \d|See https?://pubs\.acs\.org/sharingguidelines'
    r'|ACS Publications\b|Downloaded (?:by|from) .{0,60}? on \d{1,2} [A-Z][a-z]+ \d{4}'
    # The date-tailed form above is NOT the one this corpus actually carries. The packets
    # handed to the 100-paper run still held 1,207 watermark fragments across 61 papers
    # because the real stamp is a bare URL with no " on <D Month YYYY>":
    #     'Downloaded from pubs.acs.org/jmcmar/article-pdf/36/25/3993/7174911/jm00077a002...'
    # It lands beside table rows and prose, which is how a page number ('270' out of
    # 'J. Med. Chem. 2021, 64, 233-278 / 270') reached us as a compound label, and why five
    # genuine compound labels looked like furniture to a watermark-adjacency test.
    r'|Downloaded\s+(?:from|via)\s+(?:https?://)?(?:pubs|www)\.'
    r'|article-pdf/\S|/doi/(?:abs|full|pdf)/10\.'
    r'|This article is licensed under|^\s*Published on \d|^\s*Open Access Article'
    r'|is licensed under a Creative Commons)', re.I | re.M)

# Inline furniture, for text that is ALREADY extracted. is_furniture() drops whole PyMuPDF
# blocks, which is right at prepare time, but a packet built before that fix has the stamp
# welded into body_text mid-sentence and there are no blocks left to drop. These patterns
# excise it in place, which is what repairs quote verification on an existing packet.
INLINE_FURNITURE = [
    re.compile(r'Downloaded\s+(?:from|via)\s+(?:https?://)?\S+(?:\s*​)*[^\n]{0,160}', re.I),
    re.compile(r'article-pdf/\S+', re.I),
    re.compile(r'Journal\s+of\s+Medicinal\s+Chemistry[,\s]*(?:\d{4}|Vol)[^\n]{0,60}', re.I),
    re.compile(r'(?:ScienceDirect|www\.elsevier\.com/locate/\S+|Contents\s+lists\s+available'
               r'\s+at\s+\S+)[^\n]{0,80}', re.I),
    re.compile(r'ACS\s+Paragon\s+Plus\s+Environment[^\n]{0,40}', re.I),
]


def strip_inline_furniture(text):
    """-> (cleaned text, fragments removed, chars removed). Idempotent."""
    n = chars = 0
    for rx in INLINE_FURNITURE:
        found = rx.findall(text or '')
        n += len(found)
        chars += sum(len(x) for x in found)
        text = rx.sub(' ', text or '')
    # the stamp leaves a double space or a space before punctuation where it was welded in
    text = re.sub(r'[ \t]{2,}', ' ', text)
    return text, n, chars


def is_furniture(t):
    """A block that is publisher furniture rather than article text."""
    if not t or len(t) > 400:          # a long block is prose that merely cites a URL
        return False
    return bool(FURNITURE.search(t))


def page_text(page):
    """Blocks in reading order for a multi-column page: left column fully, then right."""
    blocks = [b for b in page.get_text('blocks')
              if (b[4] or '').strip() and not is_furniture(b[4])]
    if not blocks:
        return '', []
    mid = page.rect.width / 2.0
    left = sorted([b for b in blocks if (b[0] + b[2]) / 2 < mid], key=lambda b: b[1])
    right = sorted([b for b in blocks if (b[0] + b[2]) / 2 >= mid], key=lambda b: b[1])
    body, caps = [], []
    for b in left + right:
        t = re.sub(r'[ \t]+', ' ', b[4]).strip()
        (caps if CAP.match(t) else body).append(t)
    return '\n'.join(body), caps



_NORM_CACHE = {}


def _norm_doi(d):
    return re.sub(r'[^a-z0-9]', '', (d or '').lower())


def _norm_index(cx):
    """{normalised doi -> (doc_id, pubmed_id, journal, year, title)}, built once."""
    if not _NORM_CACHE:
        for row in cx.execute('SELECT doi, doc_id, pubmed_id, journal, year, title '
                              'FROM docs WHERE doi IS NOT NULL'):
            _NORM_CACHE[_norm_doi(row[0])] = row[1:]
    return _NORM_CACHE


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pdf-dir', required=True)
    ap.add_argument('--out', required=True)
    ap.add_argument('--chembl-db', default='data/chembl_db/chembl/36/chembl_36.db')
    ap.add_argument('--max-pairs', type=int, default=20,
                    help='candidate SAR pairs per packet. Batched so the series aim is stated once '
                         'and each edit attributed separately; one-pair-per-call is what produced '
                         'series-scope answers in round 1.')
    ap.add_argument('--min-tc', type=float, default=0.60)
    ap.add_argument('--min-delta-log', dest='min_delta_log', type=float, default=0.5,
                    help='minimum |log10| potency difference, as in w4 and the stereo set')
    a = ap.parse_args()

    try:
        import fitz
    except ImportError:
        sys.exit('PyMuPDF (fitz) is required: it is what avoids the two-column splice that '
                 '`pdftotext -layout` produces.')
    from rdkit import Chem, RDLogger
    from rdkit.Chem import rdFingerprintGenerator, DataStructs
    RDLogger.DisableLog('rdApp.*')
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=2048)

    cx = sqlite3.connect(a.chembl_db)
    os.makedirs(a.out, exist_ok=True)
    rows_report = []

    pdfs = sorted(f for f in os.listdir(os.path.expanduser(a.pdf_dir)) if f.lower().endswith('.pdf'))
    # ONE CORRUPT PDF MUST NOT ABORT THE RUN. A truncated download raised
    # `RuntimeError: code=7: Invalid number of pages` out of PyMuPDF and killed a 999-paper
    # prep at paper 828 -- all the work lost and no record of which file did it. Each paper is
    # now wrapped, and an unreadable file is reported like any other skip.
    def _one(fn):
        doi = fn.split('__')[0].replace('_', '/', 1)
        r = cx.execute('SELECT doc_id, pubmed_id, journal, year, title FROM docs '
                       'WHERE LOWER(doi)=LOWER(?)', (doi,)).fetchone()
        if r is None:
            # THE FILENAME HAS LOST THE DOI'S PUNCTUATION. Older Elsevier DOIs carry
            # parentheses -- 10.1016/s0960-894x(00)00032-9 -- which the download sanitised
            # away, so an exact match finds nothing and the paper is silently reported as
            # "no chembl doc". In one 999-paper batch that was 995 of them. Fall back to a
            # punctuation-insensitive match built once per run.
            r = _norm_index(cx).get(_norm_doi(doi))
        if not r:
            rows_report.append({'file': fn, 'doi': doi, 'status': 'no chembl doc'})
            return
        doc_id, pmid, journal, year, title = r

        # ---- our pairs for this paper, same assay, with potency, above similarity ----
        # standard_relation = '=' ONLY. A censored inactive -- Table 1's ">5700 nM" -- is stored in
        # ChEMBL as the bare number 5700, and pairing it against a potent analogue yields a delta
        # that is a LOWER BOUND, not a measurement. Because pairs here are ranked by descending
        # delta, censored values are actively selected for: they are 10% of doc 49661's data and
        # landed on 45% of its first batch. ChEMBL-wide, 1.04M of 9.59M potency rows carry '>' and
        # a further 3.26M carry no relation at all, so this filter is not a corner case.
        acts = cx.execute(
            'SELECT a.assay_id, a.molregno, cs.canonical_smiles, a.standard_value, a.standard_type, '
            '       a.standard_units, md.chembl_id '
            '  FROM activities a '
            '  JOIN compound_structures cs ON cs.molregno = a.molregno '
            '  JOIN molecule_dictionary md ON md.molregno = a.molregno '
            ' WHERE a.doc_id = ? AND a.standard_value IS NOT NULL '
            "   AND a.standard_relation = '=' "
            '   AND a.standard_type IN (%s)' % ','.join('?' * len(POT)),
            (doc_id,) + POT).fetchall()
        by_assay = collections.defaultdict(dict)
        info = {}
        for aid, mr, smi, val, typ, unit, chid in acts:
            by_assay[aid][mr] = smi
            info[mr] = {'smiles': smi, 'chembl_id': chid, 'value': val, 'type': typ, 'unit': unit}
        fps = {}
        for mr, d in info.items():
            m = Chem.MolFromSmiles(d['smiles'])
            if m:
                fps[mr] = gen.GetFingerprint(m)

        pairs, seen = [], set()
        for aid, members in by_assay.items():
            ms = [m for m in members if m in fps]
            for x, y in itertools.combinations(sorted(ms), 2):
                if (x, y) in seen:
                    continue
                tc = DataStructs.TanimotoSimilarity(fps[x], fps[y])
                if tc < a.min_tc:
                    continue
                dx, dy = info[x], info[y]
                # A PAIR NEEDS A POTENCY DIFFERENCE OR THERE IS NO INTENT TO FIND. The first run of
                # this script omitted the floor and the top pair of doc 42174 was Ki 0.05 against
                # Ki 0.05 -- two compounds the paper has no reason to contrast. w4 and the stereo
                # pipeline both use 0.5 log; same floor here, and only between like measurements,
                # since an IC50 and a Ki from different assays are not a delta.
                if dx['type'] != dy['type'] or dx['unit'] != dy['unit']:
                    continue
                try:
                    vx, vy = float(dx['value']), float(dy['value'])
                except (TypeError, ValueError):
                    continue
                if vx <= 0 or vy <= 0:
                    continue
                delta = abs(math.log10(vx) - math.log10(vy))
                if delta < a.min_delta_log:
                    continue
                # Identical structures are not a pair, whatever the registry says: salts and
                # unspecified stereocentres collapse to the same canonical SMILES.
                if dx['smiles'] == dy['smiles']:
                    continue
                lo, hi = (x, y) if vx >= vy else (y, x)   # higher IC50/Ki == weaker
                # Dedup ACCEPTED pairs only. Marking a pair seen before the filters meant one
                # rejected in assay A was blocked from assay B, where a different readout might
                # give it a measurable delta -- and because assay order depends on which compounds
                # exist, removing censored rows then CHANGED pair counts upward, which is how this
                # surfaced.
                seen.add((x, y))
                pairs.append({'assay_id': aid, 'tanimoto': round(tc, 3),
                              'delta_log': round(delta, 2),
                              'input': info[lo], 'analogue': info[hi]})
        # Sorting by DESCENDING similarity put Tanimoto 1.00 homologues at the top -- the pairs a
        # paper is least likely to discuss. Sort by the size of the potency change instead: those
        # are the contrasts an author actually explains.
        pairs.sort(key=lambda p: -p['delta_log'])
        batch = pairs[:a.max_pairs]

        # ---- paper text ----
        path = os.path.join(os.path.expanduser(a.pdf_dir), fn)
        doc = fitz.open(path)
        body, caps = [], []
        for pg in doc:
            b, c = page_text(pg)
            body.append(b)
            caps.extend(c)
        doc.close()
        body_txt = '\n'.join(body)
        caps_txt = '\n'.join(caps)

        res_body = {('%s%s' % (m[0], m[1])) for m in RESIDUE.findall(body_txt)}
        res_caps = {('%s%s' % (m[0], m[1])) for m in RESIDUE.findall(caps_txt)}

        packet = {
            'doc_id': doc_id, 'doi': doi, 'pubmed_id': pmid, 'journal': journal, 'year': year,
            'title': title,
            # PAPER KIND FROM THE BODY, NOT THE TITLE. doc 97714 is titled like a primary
            # ("Developing steroidal aromatase inhibitors...") and its first body line is literally
            # "Review article". A review's ChEMBL doc_id pools compounds first reported in many
            # different source papers, so its pairs are not comparisons any author made -- the
            # distinction changes how the pairs may be used, so it cannot rest on a title regex.
            'paper_kind': _paper_kind(title, body_txt),
            'paper_kind_evidence': _kind_evidence(body_txt),
            'n_compounds_with_potency': len(info),
            'n_assays': len(by_assay),
            'n_candidate_sar_pairs': len(pairs),
            'pairs': batch,
            'body_text': body_txt,
            'caption_text': caps_txt,
            'residues_mentioned_body': sorted(res_body),
            'residues_mentioned_captions_only': sorted(res_caps - res_body),
        }
        with open(os.path.join(a.out, 'doc_%s.json' % doc_id), 'w') as fh:
            json.dump(packet, fh, indent=1)

        rows_report.append({
            'file': fn[:48], 'doi': doi, 'doc_id': doc_id, 'kind': packet['paper_kind'],
            'compounds': len(info), 'assays': len(by_assay), 'sar_pairs': len(pairs),
            'batched': len(batch), 'body_chars': len(body_txt), 'caption_chars': len(caps_txt),
            'res_body': len(res_body), 'res_captions_only': len(res_caps - res_body),
            'status': 'ok'})


    for fn in pdfs:
        try:
            _one(fn)
        except Exception as e:
            rows_report.append({'file': fn, 'doi': '', 'kind': '',
                                'status': 'unreadable pdf: %s' % str(e)[:60]})

    print('%-50s %-16s %9s %7s %9s %9s %8s %7s'
          % ('file', 'kind', 'compounds', 'assays', 'sar_pairs', 'body_ch', 'res_body', 'res_cap'))
    for r in rows_report:
        if r.get('status') != 'ok':
            print('%-50s %s' % (r['file'][:48], r.get('status')))
            continue
        print('%-50s %-16s %9d %7d %9d %9d %8d %7d'
              % (r['file'], r['kind'], r['compounds'], r['assays'], r['sar_pairs'],
                 r['body_chars'], r['res_body'], r['res_captions_only']))
    ok = [r for r in rows_report if r.get('status') == 'ok']
    if ok:
        print('\npackets written to %s' % a.out)
        print('  primary papers %d | review/qsar %d'
              % (sum(1 for r in ok if r['kind'] == 'primary'),
                 sum(1 for r in ok if r['kind'] != 'primary')))
        print('  residues named ONLY in captions, summed: %d  (the figure-only channel)'
              % sum(r['res_captions_only'] for r in ok))
    return 0



if __name__ == '__main__':
    raise SystemExit(main())
