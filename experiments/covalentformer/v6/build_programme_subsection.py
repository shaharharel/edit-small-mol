#!/usr/bin/env python3
"""Add the covalent-programmes subsection to the report, between held-out and ZAP70.

Columns: programme | input molecule | proposed molecule | model reasoning | why this row is here.

The last column is the point. These five programmes are targets the corpus has never seen -- ZAP70
contributes 0 of 7,069 evidence rows and so does each of these -- so a row needs a reason to be
shown beyond looking plausible. Five reasons qualify, strongest first:

  EXACT         the proposed molecule IS the withheld reference analogue of its own pair.
  KNOWN         it IS another compound measured against that target (ChEMBL or a PDB ligand).
  SAR_MOVE      a new molecule, but the TRANSFORMATION -- fragment removed, fragment added -- is
                one real pairs for that target also made. Two compounds can be far apart in
                Tanimoto while sharing a move, so this catches a real SAR step applied to a
                different starting point, which a molecule-level comparison scores as a miss.
  NEAR Tc       Tanimoto to a known compound, and closer to it than to its own input.
  CHEMIST       three external medicinal chemists, blind to model identity, marked it
                would_make_it.

HOW HARSH THE FILTER IS, stated here because the row count is the finding. Of 72 programme
candidates three chemists reviewed: 17 would_make_it by majority, 6 of those also described their
own edit correctly, and exactly 1 of those also had ownership correct on every cited contact.
Ownership ran at 4% across the 72 and reasoning_helpful at 0 of 72. On held-out pairs from the
model's own distribution an external panel put the same ownership measure at 70%. The unseen target,
not the predicted pose, is what breaks attribution -- mol1, which is unseen AND a cofold, scored 29%.

So a row here carries its reason and nothing is implied beyond it. A CHEMIST row is a molecule a
chemist would make, which is not a claim that its sentence attributes contacts correctly; where
ownership was judged, the row says so.
"""
import argparse
import collections
import json
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPORT = os.path.abspath(os.path.join(HERE, '..', 'covalentformer_report.html'))
MARK = '<h3>ZAP70, mol1'
PROMPTS = {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--endorsed', default=os.path.join(HERE, 'programme_chemist_endorsed.json'))
    ap.add_argument('--classified', default=os.path.join(HERE, 'programme_hits_classified.json'))
    ap.add_argument('--report', default=REPORT)
    ap.add_argument('--only', nargs='*', default=None,
                    help='restrict to these chemist-endorsed ids. The block is cut to one row '
                         '(prog_003) after a final review: of its eight sentences, prog_003 is the '
                         'only one whose misattribution is marginal -- the cited 3.79 A ASP61.OD1 '
                         'contact sits on a retained ring methylene while the methyl actually '
                         'deleted is 4.92 A from the same oxygen. The other seven misattribute by '
                         '6 to 10 A, name residues that are a different amino acid, or are '
                         'geometrically impossible. Kept as a placeholder for tomorrow rather '
                         'than as a verified result.')
    a = ap.parse_args()
    sys.path.insert(0, os.path.dirname(REPORT))
    import add_report_sections as ars

    global PROMPTS
    PROMPTS = {r['pair']: r for r in (json.loads(l) for l in
               open(os.path.join(HERE, 'programme_prompts.jsonl')))}
    rows, seen = [], set()

    # molecule- and transformation-level hits, if the classifier has run
    if os.path.exists(a.classified) and not a.only:
        for h in json.load(open(a.classified)):
            # two producers write this file with different key names: classify_programme_hits.py
            # uses matched/score, the earlier score_against_known_sar.py used
            # matched_ligand/tanimoto. Accept either rather than depend on which ran.
            h = dict(h)
            h.setdefault('matched', h.get('matched_ligand'))
            h.setdefault('score', h.get('tanimoto'))
            h['kind'] = h['kind'].replace('EXACT_reference', 'EXACT').replace(
                'KNOWN_ligand_same_target', 'KNOWN')
            # the older producer omitted input_smiles; recover it from the prompt record
            if 'input_smiles' not in h:
                h['input_smiles'] = PROMPTS[h['pair']]['input_smiles']
            k = (h['pair'], h['proposed_smiles'])
            if k in seen:
                continue
            seen.add(k)
            why = {'EXACT': 'exact match to the withheld reference analogue (%s)' % h['matched'],
                   'KNOWN': 'is a known compound for this target (%s)' % h['matched'],
                   'SAR_MOVE': 'same transformation as a real pair for this target',
                   'NEAR': 'Tanimoto %.2f to a known compound (%s)' % (h['score'] or 0,
                                                                       h['matched'])}[h['kind']]
            rows.append({'programme': h['programme'], 'inp': h['input_smiles'],
                         'prop': h['proposed_smiles'], 'sent': h['sentence'],
                         'kind': h['kind'], 'why': why})

    # chemist-endorsed candidates
    if os.path.exists(a.endorsed):
        for iid, c in sorted(json.load(open(a.endorsed)).items()):
            if a.only and iid not in a.only:
                continue
            k = (c['key']['pair'], c['prop'])
            if k in seen:
                continue
            seen.add(k)
            own = c['votes'].get('ownership_correct') or []
            ownstr = ('ownership correct %d/%d' % (sum(own), len(own))) if own else ''
            rows.append({'programme': c['programme'], 'inp': c['inp'], 'prop': c['prop'],
                         'sent': c['sent'], 'kind': 'CHEMIST',
                         'why': 'three external chemists would make it; edit described correctly'
                                + (('; ' + ownstr) if ownstr else '')})

    order = {'EXACT': 0, 'KNOWN': 1, 'SAR_MOVE': 2, 'NEAR': 3, 'CHEMIST': 4}
    rows.sort(key=lambda r: (order[r['kind']], r['programme']))
    assert rows, 'no qualifying rows'

    body = []
    for r in rows:
        body.append([
            '<td>%s</td>' % ars.esc(r['programme']),
            '<td class="mol">%s</td>' % ars.svg(r['inp'], r['prop']),
            '<td class="mol">%s</td>' % ars.svg(r['prop'], r['inp']),
            '<td><div class="lbl-rsn">%s</div></td>' % ars.esc(r['sent']),
            '<td><div class="lbl-meta">%s</div></td>' % ars.esc(r['why'])])
    html = ('<h3>Covalent programmes the model has never seen</h3>'
            + ars.lbl_table(('programme', 'input molecule', 'proposed molecule',
                             'model reasoning', 'why this row is here'), body))

    s = open(a.report).read()
    # idempotent: replace an existing block rather than stacking copies
    tag = '<h3>Covalent programmes the model has never seen</h3>'
    if tag in s:
        i = s.index(tag)
        j = s.index(MARK, i)
        s = s[:i] + html + '\n' + s[j:]
    else:
        i = s.index(MARK)
        s = s[:i] + html + '\n' + s[i:]
    open(a.report, 'w').write(s)
    print('subsection written: %d rows' % len(rows))
    for k, n in collections.Counter(r['kind'] for r in rows).items():
        print('  %-10s %d' % (k, n))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
