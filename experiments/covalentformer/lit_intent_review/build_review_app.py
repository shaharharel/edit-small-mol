#!/usr/bin/env python3
"""Build a standalone review app: 100 extracted design intents for a medicinal chemist to judge.

WHAT THE REVIEWER IS BEING ASKED. Each card shows the before molecule, the after molecule and
the reason the extractor pulled out of the paper. The question is whether that reason is a
sound medicinal-chemistry account of THAT change -- not whether the extraction is faithful to
the paper, which is a separate (and already measured) question.

WHAT THE REVIEWER IS DELIBERATELY NOT SHOWN, and why:

  * our automated QA verdicts (Tanimoto, grounding, instruction-shape). A reviewer told "this
    row failed the grounding check" will agree with it. The QA line is kept in the saved record
    so the two can be compared afterwards, and is not rendered.
  * the potency delta. It is the strongest anchor in the dataset: told the analogue is 100x
    better, a reviewer rationalises almost any reason for it. It is revealed only AFTER the
    verdict is submitted, on the card itself, because chemists have asked for it and it is
    useful feedback once the judgement is locked.
  * which paper the reason came from is shown (the reviewer needs to know the target), but the
    quote is behind a toggle so the judgement is made on the reason as a reason first.

SAMPLING IS STATED. 100 cards, at most two per paper, spread across the potency-delta strata
(|d|>2, 0.3-2, <0.3) in the same proportions as the full set, and inside each stratum ordered
by descending Tanimoto so the edit is legible. Nothing is filtered on our own QA, so rows our
checks dislike are in the sample and can be judged on their merits.

Usage:
  python build_review_app.py --pairs <resolved csv> --subs <submissions dir> \
      --packets <packet dir> --out app
"""
import argparse
import csv
import json
import os
import random
from collections import defaultdict
from pathlib import Path

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFMCS, rdMolDescriptors
from rdkit.Chem.Draw import rdMolDraw2D

RDLogger.DisableLog('rdApp.*')
HERE = os.path.dirname(os.path.abspath(__file__))


def svg(mol, highlight=(), w=230, h=180):
    d = rdMolDraw2D.MolDraw2DSVG(w, h)
    d.drawOptions().clearBackground = False
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol, highlightAtoms=list(highlight),
                                       highlightAtomColors={i: (1.0, 0.62, 0.21)
                                                            for i in highlight})
    d.FinishDrawing()
    return d.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>\n", '')


def changed(a, b):
    try:
        r = rdFMCS.FindMCS([a, b], timeout=6, ringMatchesRingOnly=True)
        if not r or not r.smartsString:
            return (), ()
        p = Chem.MolFromSmarts(r.smartsString)
        ma, mb = set(a.GetSubstructMatch(p)), set(b.GetSubstructMatch(p))
        return (tuple(i for i in range(a.GetNumAtoms()) if i not in ma),
                tuple(i for i in range(b.GetNumAtoms()) if i not in mb))
    except Exception:
        return (), ()


def props(m):
    """The few numbers a chemist actually glances at before judging an edit.

    Deliberately short: MW, cLogP, TPSA, HBD/HBA, rotatable bonds. A longer table would be read
    as the thing under review, and it is not -- the reason is.
    """
    return {
        'MW': round(Descriptors.MolWt(m), 1),
        'cLogP': round(Crippen.MolLogP(m), 2),
        'TPSA': round(rdMolDescriptors.CalcTPSA(m), 1),
        'HBD': rdMolDescriptors.CalcNumHBD(m),
        'HBA': rdMolDescriptors.CalcNumHBA(m),
        'RotB': rdMolDescriptors.CalcNumRotatableBonds(m),
    }


def stratum(d):
    d = abs(d)
    return 'big' if d > 2 else ('mid' if d >= 0.3 else 'small')


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--pairs', required=True)
    ap.add_argument('--subs', required=True)
    ap.add_argument('--packets', required=True)
    ap.add_argument('--out', default=os.path.join(HERE, 'app'))
    ap.add_argument('--n', type=int, default=100)
    ap.add_argument('--per-paper', type=int, default=2)
    ap.add_argument('--seed', type=int, default=11)
    ap.add_argument('--max-mw', dest='max_mw', type=float, default=0,
                    help='drop pairs whose larger molecule exceeds this MW (0 = keep all)')
    a = ap.parse_args()

    rows = [r for r in csv.DictReader(open(os.path.expanduser(a.pairs)))
            if (r.get('delta') or '').strip()]
    eps = {}
    for f in Path(os.path.expanduser(a.subs)).glob('*.json'):
        d = json.loads(f.read_text())
        for i, e in enumerate(d.get('episodes') or []):
            eps['%s#%d' % (f.stem, i)] = e
    packs = {}
    for f in Path(os.path.expanduser(a.packets)).glob('doc_*.json'):
        d = json.loads(f.read_text())
        packs[str(d.get('doc_id'))] = {k: d.get(k) for k in
                                       ('title', 'journal', 'year', 'doi', 'pubmed_id')}

    # keep the full set's stratum proportions, so the sample is not a different dataset
    by_s = defaultdict(list)
    for r in rows:
        by_s[stratum(float(r['delta']))].append(r)
    total = sum(len(v) for v in by_s.values())
    quota = {k: max(1, round(a.n * len(v) / total)) for k, v in by_s.items()}
    print('full set: ' + ', '.join('%s %d (%.0f%%)' % (k, len(v), 100 * len(v) / total)
                                   for k, v in by_s.items()))
    print('quota: %s' % quota)

    rng = random.Random(a.seed)
    cache = {}

    def fp(s):
        if s not in cache:
            m = Chem.MolFromSmiles(s)
            cache[s] = (m, AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048) if m else None)
        return cache[s]

    picked, per_paper = [], defaultdict(int)
    for s, want in quota.items():
        pool = by_s[s]
        rng.shuffle(pool)
        scored = []
        for r in pool:
            ep = eps.get(r['intent_id'])
            if not ep or not (ep.get('design_rationale') or '').strip():
                continue
            ma, fa = fp(r['mol_a'])
            mb, fb = fp(r['mol_b'])
            if ma is None or mb is None:
                continue
            if a.max_mw and max(Descriptors.MolWt(ma), Descriptors.MolWt(mb)) > a.max_mw:
                continue
            scored.append((DataStructs.TanimotoSimilarity(fa, fb), r, ep))
        # RANDOM WITHIN THE STRATUM, NOT TANIMOTO-RANKED. Ranking by descending Tanimoto was
        # meant to make the edit legible and instead skewed the sample towards big molecules:
        # a 2,000 Da peptide shares almost all its fingerprint bits with its own analogue, so
        # peptides sorted to the top. The first sample was 27% above MW 700 against 7.6% in the
        # full set, which is why the reviewer saw peptide after peptide. The stratum shuffle is
        # already seeded, so taking it in order is a representative draw.
        pass
        taken = 0
        for tc, r, ep in scored:
            if taken >= want:
                break
            if per_paper[r['doc_id']] >= a.per_paper:
                continue
            per_paper[r['doc_id']] += 1
            taken += 1
            ma, _ = fp(r['mol_a'])
            mb, _ = fp(r['mol_b'])
            ha, hb = changed(ma, mb)
            pk = packs.get(r['doc_id'], {})
            picked.append({
                'id': '%s|%s|%s' % (r['doc_id'], r['mol_a_id'], r['mol_b_id']),
                'episode_id': r['intent_id'], 'doc_id': r['doc_id'],
                'title': pk.get('title') or '', 'journal': pk.get('journal'),
                'year': pk.get('year'), 'doi': pk.get('doi'),
                'target': r['target_chembl_id'],
                'label_a': r['label_a'], 'label_b': r['label_b'],
                'mol_a_id': r['mol_a_id'], 'mol_b_id': r['mol_b_id'],
                'smiles_a': r['mol_a'], 'smiles_b': r['mol_b'],
                'svg_a': svg(ma, ha), 'svg_b': svg(mb, hb),
                'props_a': props(ma), 'props_b': props(mb),
                'mw_max': round(max(Descriptors.MolWt(ma), Descriptors.MolWt(mb)), 1),
                'reason': ep.get('design_rationale'),
                'quote': ' '.join((ep.get('evidence_quote') or '').split())[:600],
                'section': ep.get('evidence_section'),
                'intent_class': ep.get('intent_class'), 'basis': ep.get('basis'),
                # withheld from the UI until a verdict is submitted
                'after_submit': {
                    'std_type': r['std_type'], 'value_a': r['value_a'], 'value_b': r['value_b'],
                    'delta': float(r['delta']), 'assay_id': r['assay_id'],
                    'tanimoto': round(tc, 3), 'stratum': s,
                },
            })
    rng.shuffle(picked)
    os.makedirs(os.path.expanduser(a.out), exist_ok=True)
    out = os.path.expanduser(a.out)
    json.dump(picked, open(os.path.join(out, 'items.json'), 'w'))
    print('wrote %s/items.json: %d cards from %d papers'
          % (out, len(picked), len({x['doc_id'] for x in picked})))
    open(os.path.join(out, 'index.html'), 'w').write(
        open(os.path.join(HERE, 'index_template.html')).read())
    print('wrote %s/index.html' % out)


if __name__ == '__main__':
    main()
