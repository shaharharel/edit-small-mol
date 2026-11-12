#!/usr/bin/env python
"""Resolve the compound labels an intent names to the molecules ChEMBL curated from that paper.

This is identity retrieval, and it replaces pattern retrieval wherever it applies. A SMARTS
says "find me pairs that look like this edit"; `compound_records.compound_key` says "this is
the molecule the author called 12". The difference is everything the chemist review
complained about: pattern retrieval scored 38% clean, with multi-site contamination in half
the pairs and 14% matching a second copy of the motif while the group under discussion never
changed. An identity-resolved pair cannot have those defects -- whatever the author's two
compounds differ by IS what the author was discussing.

ChEMBL's curators read each paper, figures included, and recorded every molecule under the
paper's own label. The column is 100% populated on all 20 papers tested, 200-360 molecules
each, which is why resolution reaches 79% rather than the ~30% that text-visible structures
alone would give.

THE LABEL COLLISION IS REAL AND IS NOT PAPERED OVER HERE. Reviews reuse plain numbers across
sub-sections: in doc_125304, `4` is P22077 in Fig. 8, a 1.5 nM quinazolinone in Fig. 22, and
a Genentech 2-aminopyridine in Fig. 38; two extractors independently flagged it. ChEMBL
encodes the scope in the key itself ('Fig 11, Cpd 12; Fig 12, Cpd 12'), so a bare label that
maps to more than one distinct molecule is reported `ambiguous` and dropped unless the
extractor supplied a figure scope that disambiguates it. Guessing here would silently pair
molecules from different series.

Usage:
  python scripts/extraction/lit_resolve_compounds.py \
      --intents data/lit_extract/intents --packets data/lit_extract/round4 \
      --out data/lit_extract/intent_pairs_resolved.csv
"""
import argparse
import csv
import json
import math
import re
import sqlite3
import sys
from collections import Counter, defaultdict

from rdkit import Chem, RDLogger

RDLogger.DisableLog('rdApp.*')
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from lit_residue_align import paper_target  # noqa: E402

# 'Fig 11, Cpd 12', 'Fig 27, Cpd FT671', 'Table 3, Cpd 5'
SCOPED = re.compile(r'(?:fig(?:ure)?|table|scheme)\.?\s*(\d+[a-z]?)\s*,\s*cpd\s*(\S+)', re.I)
# the figure scope an extractor may append: '12 (Fig. 38)', '3 (Table 2)'
NAMED_SCOPE = re.compile(r'^(\S+?)\s*[\(\[]\s*(?:fig(?:ure)?|table|scheme)\.?\s*(\d+[a-z]?)', re.I)
# strip prose: 'analogue 35', 'compound 14', 'cpd 7'
PROSE = re.compile(r'^(?:analogu?e|compound|cpd|cmpd|derivative)s?\s+', re.I)


def norm_label(s):
    s = PROSE.sub('', str(s or '').strip())
    s = s.strip('.,;:() ')
    return s.lower()


def build_index(cx, doc_id):
    """Two indexes for one paper: bare label -> {chembl_id}, and (scope, label) -> chembl_id."""
    bare = defaultdict(set)
    scoped = {}
    for ck, cid in cx.execute(
            "SELECT cr.compound_key, md.chembl_id FROM compound_records cr "
            "JOIN molecule_dictionary md ON md.molregno = cr.molregno "
            "WHERE cr.doc_id = ? AND cr.compound_key IS NOT NULL", (doc_id,)):
        for part in str(ck).split(';'):
            part = part.strip()
            if not part:
                continue
            m = SCOPED.search(part)
            if m:
                fig, lab = m.group(1).lower(), norm_label(m.group(2))
                scoped[(fig, lab)] = cid
                bare[lab].add(cid)
            else:
                lab = norm_label(part)
                if lab:
                    bare[lab].add(cid)
    return bare, scoped


def resolve(label, bare, scoped):
    """-> (chembl_id, how) or (None, why). Never guesses between colliding labels."""
    raw = str(label or '').strip()
    m = NAMED_SCOPE.match(raw)
    if m:
        lab, fig = norm_label(m.group(1)), m.group(2).lower()
        if (fig, lab) in scoped:
            return scoped[(fig, lab)], 'scoped'
        cand = bare.get(lab, set())
        if len(cand) == 1:
            return next(iter(cand)), 'bare_after_scope_miss'
        return None, 'ambiguous' if cand else 'not_found'
    lab = norm_label(raw)
    cand = bare.get(lab, set())
    if len(cand) == 1:
        return next(iter(cand)), 'bare'
    if len(cand) > 1:
        return None, 'ambiguous'
    return None, 'not_found'


DIRECTION_NOTE = """
DIRECTION IS NOT THE SENTENCE'S WORD ORDER.

The first version of this script paired compounds in the order `compounds_named` listed them
and called that a->b. That is wrong about a third of the time, measured: of 304 pairs where
the intent's own SMARTS can resolve which molecule carries `from` and which carries `to`,
**110 (36%) ran opposite to the naming order**. Authors routinely name the better compound
first -- "compound 12 (IC50 5 nM) was far more potent than 11 (500 nM)" -- which naming order
reads as 12 -> 11, a potency loss.

That single bug produced the 45%-improving figure that looked like a chemistry finding and
was an artefact of English word order.

Direction is therefore resolved in this priority:

  smarts    both SMARTS present and they match the two molecules one way only -- the intent
            itself tells us which side is which. Most reliable.
  potency   no usable SMARTS, but both molecules have a value in one assay. The pair is
            oriented so mol_b is the MORE potent, which makes every such pair an improving
            pair by construction, and `intent_runs_with_potency` records whether the author's
            described change goes that way or against it.
  naming    neither available. Recorded as `naming` so it is never mistaken for measured.

An explained potency LOSS is real data -- the mismatch arm showed the model cannot currently
learn from a rationale at all -- so the loss cases are kept and labelled, not flipped away.
"""

POT_Q = """
SELECT act.assay_id, act.standard_type, act.standard_value
FROM activities act
JOIN assays a ON a.assay_id = act.assay_id
JOIN target_dictionary td ON td.tid = a.tid
JOIN molecule_dictionary md ON md.molregno = act.molregno
WHERE md.chembl_id = ? AND td.chembl_id = ?
  AND act.standard_relation = '=' AND act.standard_value > 0
  AND act.standard_units = 'nM' AND act.standard_type IN ('IC50','Ki','Kd','EC50')
"""


def potency(cx, cid, tgt):
    """{(assay_id, type): value} -- same-assay pairing is what keeps the delta meaningful."""
    out = {}
    for aid, typ, val in cx.execute(POT_Q, (cid, tgt)):
        out.setdefault((aid, typ), float(val))
    return out


def smiles_of(cx, cid):
    r = cx.execute("SELECT cs.canonical_smiles FROM compound_structures cs "
                   "JOIN molecule_dictionary md ON md.molregno = cs.molregno "
                   "WHERE md.chembl_id = ?", (cid,)).fetchone()
    return r[0] if r else None


def resolve_direction(it, sa, sb, va, vb):
    """-> (swap_needed, direction_source, intent_runs_with_potency).

    See DIRECTION_NOTE. `swap_needed` means the a/b in the sentence should be exchanged.
    """
    tr = it.get('transformation') or {}
    pf = Chem.MolFromSmarts(tr['smarts_from']) if tr.get('smarts_from') else None
    pt = Chem.MolFromSmarts(tr['smarts_to']) if tr.get('smarts_to') else None
    ma, mb = Chem.MolFromSmiles(sa), Chem.MolFromSmiles(sb)
    runs_with = None
    if pf is not None and pt is not None and ma is not None and mb is not None:
        fwd = ma.HasSubstructMatch(pf) and mb.HasSubstructMatch(pt)
        rev = ma.HasSubstructMatch(pt) and mb.HasSubstructMatch(pf)
        if fwd != rev:                       # exactly one orientation fits the intent
            swap = rev
            if va is not None and vb is not None:
                lo_a, lo_b = (va, vb) if not swap else (vb, va)
                runs_with = lo_a > lo_b      # nM: larger value = weaker, so from-weaker = gain
            return swap, 'smarts', runs_with
    if va is not None and vb is not None:
        # orient so mol_b is the MORE potent (smaller nM): every such pair is improving
        return (va < vb), 'potency', None
    return False, 'naming', None



LABEL_STEM = re.compile(r'^([A-Za-z]*\d+)')


def pick_reference(refs, analogue_label, analogue_cid, cx, stats):
    """Choose ONE reference for this analogue. Returns (label, chembl_id) or (None, None).

    Order matters, and the first rule is the one that fixes the real cases deterministically:

      1. LABEL STEM. `11a` belongs with `11h`, `20ab` with `20ac`, `F4` with `F8`. Two parallel
         label families are the commonest shape in these papers and the diagonal is the only
         comparison the text licenses; similarity alone gets it right only by luck.
      2. Structural similarity, when the stems do not decide.
      3. List order.

    A tie that survives all three is REFUSED rather than guessed: an analogue attached to the
    wrong reference is a rationale attached to a change nobody described, which is the failure
    this whole pipeline exists to avoid.
    """
    if not refs:
        return None, None
    if len(refs) == 1:
        return refs[0]
    m = LABEL_STEM.match(analogue_label or '')
    if m:
        stem = m.group(1)
        same = [r for r in refs if (LABEL_STEM.match(r[0] or '') or [None]) and
                LABEL_STEM.match(r[0] or '') and LABEL_STEM.match(r[0]).group(1) == stem]
        if len(same) == 1:
            stats['ref_by_label_stem'] += 1
            return same[0]
    best, best_tc = None, -1.0
    sb = smiles_of(cx, analogue_cid)
    if sb:
        from rdkit import Chem, DataStructs
        from rdkit.Chem import AllChem
        mb = Chem.MolFromSmiles(sb)
        fb = AllChem.GetMorganFingerprintAsBitVect(mb, 2, 2048) if mb else None
        ties = []
        for la, ca in refs:
            sa = smiles_of(cx, ca)
            ma = Chem.MolFromSmiles(sa) if sa else None
            if ma is None or fb is None:
                continue
            tc = DataStructs.TanimotoSimilarity(
                AllChem.GetMorganFingerprintAsBitVect(ma, 2, 2048), fb)
            ties.append((tc, la, ca))
        if ties:
            ties.sort(reverse=True)
            if len(ties) == 1 or ties[0][0] - ties[1][0] >= 0.02:
                stats['ref_by_similarity'] += 1
                return ties[0][1], ties[0][2]
            stats['ambiguous_reference'] += 1
            return None, None
    stats['ref_by_list_order'] += 1
    return refs[0]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--intents', required=True)
    ap.add_argument('--packets', required=True)
    ap.add_argument('--chembl-db', default='data/chembl_db/chembl/36/chembl_36.db')
    ap.add_argument('--out', required=True)
    ap.add_argument('--require-potency', action='store_true',
                    help='emit only pairs where both molecules have a value in the SAME assay')
    a = ap.parse_args()

    cx = sqlite3.connect(a.chembl_db)
    w = csv.writer(open(a.out, 'w', newline=''))
    w.writerow(['doc_file', 'doc_id', 'intent_id', 'target_chembl_id', 'label_a', 'label_b',
                'mol_a_id', 'mol_b_id', 'mol_a', 'mol_b', 'assay_id', 'std_type',
                'value_a', 'value_b', 'delta', 'direction_source',
                'intent_runs_with_potency', 'intent_type',
                'pocket_related', 'purpose'])

    how = Counter()
    dirs = Counter()
    stats = Counter()
    print(f"{'paper':24} {'intents':>7} {'named':>6} {'pairs':>6} {'w/potency':>10}")
    print('-' * 60)
    for f in sorted(Path(a.intents).glob('*.json')):
        d = json.loads(f.read_text())
        # `paper` is a dict in the round-4 submissions and a bare doc id string in the
        # episode packages (the v4 task files set it to the doc id directly), so both shapes
        # have to be accepted or the whole 869-paper return crashes on the first file.
        pf = d.get('paper')
        doc = pf.get('doc_id') if isinstance(pf, dict) else (pf or None)
        # ...and the episode packages write it as 'doc_10096' while ChEMBL's doc_id is the bare
        # number. Keyed with the prefix, build_index() returned an empty label index and every
        # one of the 869 papers was skipped before a single compound was looked up -- a silent
        # zero, not an error.
        if isinstance(doc, str) and doc.startswith('doc_'):
            doc = doc[4:]
        if not doc:
            pk = Path(a.packets) / f'{f.stem}.json'
            if pk.exists():
                doc = json.loads(pk.read_text()).get('doc_id')
        if not doc:
            continue
        pk = Path(a.packets) / f'doc_{doc}.json'
        tgt = None
        if pk.exists():
            tgt = paper_target(cx, [p['assay_id'] for p in json.loads(pk.read_text()).get('pairs', [])])
        # The 100-paper packets are text-only ON PURPOSE -- they withhold the candidate
        # pairs so the extractor cannot see them ('withheld (different experiment)'), and
        # that is exactly the key the line above reads. Result: tgt was None for all 98
        # papers, potency() was never queried, and every delta came back null. The target
        # is recoverable without the packet: a ChEMBL document owns its assays, so go
        # doc_id -> assays -> target directly and keep the same SINGLE-PROTEIN preference.
        if tgt is None:
            tgt = paper_target(cx, [r[0] for r in cx.execute(
                'SELECT a.assay_id FROM assays a WHERE a.doc_id = ?', (doc,))])
        bare, scoped = build_index(cx, doc)
        if not bare:
            continue
        # `episodes` is the v4/v3 design-episode schema; `intents`/`records` are the older
        # sentence-level rounds. All three are accepted so one resolver serves every round.
        its = d.get('intents') or d.get('records') or d.get('episodes') or []
        named = npair = npot = 0
        for _ei, it in enumerate(its):
            # EPISODES HAVE NO ID OF THEIR OWN, so give them a stable one: the file plus the
            # episode's index in it. Without this the provenance column is blank for every
            # episode-schema row, and a resolved pair cannot be traced back to the rationale
            # that justified it -- the one audit that protects against training a correct pair
            # against the wrong intent.
            it = dict(it)
            it.setdefault('record_id', '%s#%d' % (f.stem, _ei))

            def _ids(key):
                out = []
                for x in (it.get(key) or []):
                    lb = str(x).strip()
                    if not lb:
                        continue
                    cid, h = resolve(lb, bare, scoped)
                    how[h] += 1
                    if cid:
                        out.append((lb, cid))
                return out

            # AN EPISODE ALREADY KNOWS ITS DIRECTION, so do not re-derive it. The older rounds
            # emitted one undirected list (`compounds_named`) and the direction had to be
            # guessed from potency or SMARTS; an episode states which compounds are BEFORE the
            # change and which are AFTER, which is both correct more often and auditable.
            # Capped at 4 per side: an episode naming a whole series would otherwise emit
            # dozens of comparisons the text never makes.
            # ---------------------------------------------------------------- pairing
            # ONE ROW PER (EPISODE, ANALOGUE). Never a cross-product.
            #
            # The previous rule emitted compounds_before[:4] x compounds_after[:4], which did
            # three bad things at once: it invented comparisons the paper never makes (doc_101787
            # returned two PARALLEL label families, 11a/16a/26a/29a against 11h/16h/26h/29h, of
            # which only the diagonal is licensed, and we emitted all 16 -- including 11a->29h,
            # which changes the pyrimidine methyl AND swaps the whole hydrazone arm while
            # carrying a +1.528 log delta attributed to "replace an aromatic proton with
            # methyl"); the [:4] silently dropped 18,927 labels; and because the slice depends
            # on list order, two runs over identical submissions could differ.
            #
            # The cascade below prefers what the PAPER states over anything we can infer.
            cands = []
            analogues = it.get('analogues') or []
            explicit_pairs = it.get('pairs') or []

            if explicit_pairs:
                # the passage spells its comparisons out ("20y and 20ab with the N-methyl
                # analogues 20aa and 20ac") -- emit exactly those, nothing else
                named += 1
                for pr in explicit_pairs:
                    if not (isinstance(pr, (list, tuple)) and len(pr) == 2):
                        continue
                    (la, lb) = (str(pr[0]).strip(), str(pr[1]).strip())
                    ca, _ = resolve(la, bare, scoped)
                    cb, _ = resolve(lb, bare, scoped)
                    if not ca or not cb or ca == cb:
                        stats['unresolved_or_identical'] += 1
                        continue
                    cands.append(((la, ca), (lb, cb), 'explicit_pairs'))

            elif analogues:
                named += 1
                refs = []
                if it.get('reference'):
                    cid, h = resolve(str(it['reference']).strip(), bare, scoped)
                    how[h] += 1
                    if cid:
                        refs = [(str(it['reference']).strip(), cid)]
                if not refs:
                    # a legacy before-list, or nothing: `reference: null` is a legitimate v5.2
                    # answer and simply means this episode yields no PAIR -- its rationale is
                    # still a record, it just has no edit to attach to
                    refs = _ids('compounds_before')
                for a_entry in analogues:
                    lb = str((a_entry or {}).get('label') or '').strip()
                    if not lb:
                        continue
                    cb, h = resolve(lb, bare, scoped)
                    how[h] += 1
                    if not cb:
                        stats['unresolved_label'] += 1
                        continue
                    if not refs:
                        stats['no_reference_for_analogue'] += 1
                        continue
                    la, ca = pick_reference(refs, lb, cb, cx, stats)
                    if not ca or ca == cb:
                        continue
                    cands.append(((la, ca), (lb, cb), 'schema_reference_analogue'))

            else:
                pre, post = _ids('compounds_before'), _ids('compounds_after')
                if pre and post:
                    # LEGACY submissions (v4 and earlier) have no per-analogue structure, so the
                    # best available rule is one row per after-compound against its OWN best
                    # reference -- still never a cross-product.
                    named += 1
                    seen = set()
                    for lb, cb in post:
                        la, ca = pick_reference(pre, lb, cb, cx, stats)
                        if not ca or ca == cb or (ca, cb) in seen:
                            continue
                        seen.add((ca, cb))
                        cands.append(((la, ca), (lb, cb), 'legacy_before_after'))
                else:
                    labels = [str(x).strip() for x in (it.get('compounds_named') or [])
                              if str(x).strip()]
                    if not labels:
                        continue
                    named += 1
                    got = _ids('compounds_named')
                    seenc, uniq = set(), []
                    for lb, cid in got:
                        if cid not in seenc:
                            seenc.add(cid)
                            uniq.append((lb, cid))
                    if len(uniq) < 2:
                        continue
                    cands = [(x, y, None) for x, y in zip(uniq, uniq[1:])]

            for (la, ca), (lb, cb), forced_dir in cands:
                sa, sb = smiles_of(cx, ca), smiles_of(cx, cb)
                if not sa or not sb:
                    stats['no_structure'] += 1
                    continue
                pa = potency(cx, ca, tgt) if tgt else {}
                pb = potency(cx, cb, tgt) if tgt else {}
                shared = sorted(set(pa) & set(pb))
                if shared:
                    aid, typ = shared[0]
                    va, vb = pa[(aid, typ)], pb[(aid, typ)]
                    delta = round(math.log10(va) - math.log10(vb), 3)
                    npot += 1
                elif a.require_potency:
                    stats['no_shared_assay'] += 1
                    continue
                else:
                    aid = typ = va = vb = delta = None
                if forced_dir:
                    # DIRECTION IS THE EXTRACTOR'S STATEMENT, not an inference. reference ->
                    # analogue is auditable against the quote and is right far more often than
                    # anything derived from potency or SMARTS. Potency may CONFIRM it and is
                    # never allowed to set it: a disagreement is flagged and the row is KEPT,
                    # because under this project's framing the label is reasoning about an edit
                    # and an unexplained potency move does not make the reasoning wrong.
                    swap, dsrc, runs_with = False, forced_dir, None
                    if (it.get('endpoint') == 'potency' and va is not None and vb is not None):
                        measured = 'better' if vb < va else 'worse'
                        stated = None
                        for a_entry in (it.get('analogues') or []):
                            if str((a_entry or {}).get('label') or '').strip() == lb:
                                stated = (a_entry or {}).get('outcome_direction')
                        if stated in ('better', 'worse') and stated != measured:
                            stats['direction_disagrees_with_measurement'] += 1
                            runs_with = 'disagrees'
                else:
                    swap, dsrc, runs_with = resolve_direction(it, sa, sb, va, vb)
                if swap:
                    la, lb = lb, la
                    ca, cb = cb, ca
                    sa, sb = sb, sa
                    va, vb = vb, va
                    if va is not None and vb is not None:
                        delta = round(math.log10(va) - math.log10(vb), 3)
                dirs[dsrc] += 1
                npair += 1
                # PROVENANCE. This read `it.get('intent_id')` while the records carry
                # `record_id` -- a field rename that was never followed through here, so
                # intent_id was empty on all 11,801 rows. The consequence is worse than a
                # blank column: a resolved pair could not be traced back to the record that
                # justified it, so there was no way to audit which rationale had been
                # attached to which molecules. That audit is the only defence against
                # training on a correctly-resolved pair carrying the wrong intent.
                rid = it.get('record_id') or it.get('intent_id')
                # The 120-char truncation silently clipped the intent. It is why the v1
                # intents topped out at exactly 120 characters -- an artefact of this line,
                # not of the extractors. A v2 design_rationale is capped at 80 words (~450
                # chars), so leaving this in place would chop every one of them.
                w.writerow([f.name, doc, rid, tgt, la, lb, ca, cb, sa, sb,
                            aid, typ, va, vb, delta, dsrc, runs_with,
                            it.get('intent_type'), it.get('pocket_related'),
                            (it.get('purpose') or '')])
        stats['intents'] += len(its); stats['named'] += named
        stats['pairs'] += npair; stats['with_potency'] += npot
        print(f'{f.name[:24]:24} {len(its):7d} {named:6d} {npair:6d} {npot:10d}')

    print('-' * 60)
    print(f"{stats['intents']} intents, {stats['named']} name compounds, "
          f"{stats['pairs']} identity-resolved pairs, {stats['with_potency']} with a shared-assay delta")
    print('label resolution: ' + ', '.join(f'{k} {v}' for k, v in how.most_common()))
    print('direction source:  ' + ', '.join(f'{k} {v}' for k, v in dirs.most_common()))
    if stats['no_structure'] or stats['no_shared_assay']:
        print(f"dropped: no_structure {stats['no_structure']}, "
              f"no_shared_assay {stats['no_shared_assay']}")
    print(f'wrote {a.out}')


if __name__ == '__main__':
    main()
