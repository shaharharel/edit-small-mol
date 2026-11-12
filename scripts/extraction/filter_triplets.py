#!/usr/bin/env python
"""Filter the mined triplets down to what is worth a chemist's time, in a stated order.

ORDER MATTERS AND IS NOT ARBITRARY. Normalisation runs before anything structural (a counter-ion
makes every fingerprint see a chlorine disappear); dedup runs early so nothing downstream is
double-counted; the text clean-up runs before any regex reads the reason; and the MCS work runs
LAST, on survivors only, because it is the expensive step.

The rules come from a medicinal chemist's triage of 40 pairs, with their own hit rates. Two of
them invert checks we already had: `group_in_both` is the mirror of "group in neither" (a reason
that names something both molecules already carry is not describing this edit), and
`silent_stereo` is the mirror of our stereo-claim check (a stereochemistry change with no stereo
language in the reason).

TWO RULES DELIBERATELY DO NOT DROP ANYTHING:
  endpoint_mismatch  the edit was made for solubility/clearance/hERG and we attached it to a
                     POTENCY delta. The triplet is good data about the wrong objective, so it is
                     ROUTED to a bucket rather than deleted.
  small_delta        |delta| < 0.3 is at this project's same-assay noise floor. It is excluded
                     from direction-learning objectives at assembly time, not thrown away.

NOT USED AS A FILTER: `intent_class`. It does not separate quality -- in the chemist's sample,
four of the best items were `post_hoc_explanation` and three of the worst were `design_reason`.

Usage:
  python scripts/extraction/filter_triplets.py --resolved <csv> --intents <submissions dir> \
      --out <kept csv> --rejects <rejected csv>
"""
import argparse
import collections
import csv
import json
import os
import re
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS

RDLogger.DisableLog('rdApp.*')

ADMET = re.compile(r'solubilit|aqueous|clearance|microsomal|metabolic stabil|hydroly|permeab|'
                   r'caco|herg|clogp|plasma protein|half-life|CYP|P450|bioavailab|'
                   r'synthetic tractab', re.I)
POTENCY = re.compile(r'potenc|affinit|activit|IC50|EC50|\bKi\b|\bKd\b|inhibit|bind', re.I)
# the extraction brief's own vocabulary, leaking into the label it was supposed to elicit
LEAK = re.compile(r',?\s*(following|retaining|using|treating|consistent with|guided by|based on)'
                  r'\s+(the\s+)?(authors[’\']?|post[- ]hoc|suggested|overall)[^.]{0,90}', re.I)
LEAK2 = re.compile(r'\bpost[- ]hoc\b[^.,]{0,40}', re.I)
TOLERATED = re.compile(r'\btolerated\b|comparable|no significant loss|similar potency', re.I)
STEREO_WORD = re.compile(r'stereo|enantio|diastereo|config|epimer|\bcis\b|\btrans\b|\bE-|\bZ-|'
                         r'\(R\)|\(S\)|chiral', re.I)
LOCANT = re.compile(r'\bC-?\d|\bN-?\d|\b\d-position|\bortho\b|\bmeta\b|\bpara\b|\bR\d\b|'
                    r'\bN\d\b|\b\d\'-', re.I)
GROUPWORD = re.compile(r'methyl|ethyl|propyl|butyl|phenyl|aryl|ring|amide|ester|amine|amino|'
                       r'hydroxyl|methoxy|nitrile|cyano|sulfon|halogen|fluor|chlor|brom|'
                       r'pyridin|piperaz|piperidin|morpholin|carbonyl|ether|acid|nitro|'
                       r'heterocycle|substituent|chain|linker', re.I)


def largest_fragment(smi):
    m = Chem.MolFromSmiles(smi) if smi else None
    if m is None:
        return None, None
    frags = Chem.GetMolFrags(m, asMols=True, sanitizeFrags=False)
    if len(frags) > 1:
        m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
    return m, Chem.MolToSmiles(m)


def stereo_signature(m):
    if m is None:
        return None
    centres = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    bonds = sorted(str(b.GetStereo()) for b in m.GetBonds() if str(b.GetStereo()) != 'STEREONONE')
    return (tuple(sorted(centres)), tuple(bonds))


def edit_profile(ma, mb):
    """-> (n_sites, edit_size) from the MCS symmetric difference. Expensive; run last."""
    try:
        r = rdFMCS.FindMCS([ma, mb], timeout=5, ringMatchesRingOnly=True,
                           completeRingsOnly=False)
        if not r or not r.smartsString:
            return 99, 99
        p = Chem.MolFromSmarts(r.smartsString)
        qa, qb = set(ma.GetSubstructMatch(p)), set(mb.GetSubstructMatch(p))
        out_a = [i for i in range(ma.GetNumHeavyAtoms()) if i not in qa]
        out_b = [i for i in range(mb.GetNumHeavyAtoms()) if i not in qb]
        size = len(out_a) + len(out_b)

        def components(mol, idxs):
            idxs = set(idxs)
            seen, comp = set(), 0
            for i in idxs:
                if i in seen:
                    continue
                comp += 1
                stack = [i]
                while stack:
                    j = stack.pop()
                    if j in seen:
                        continue
                    seen.add(j)
                    for nb in mol.GetAtomWithIdx(j).GetNeighbors():
                        if nb.GetIdx() in idxs:
                            stack.append(nb.GetIdx())
            return comp
        return max(components(ma, out_a), components(mb, out_b)), size
    except Exception:
        return 99, 99


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--resolved', required=True)
    ap.add_argument('--intents', required=True)
    ap.add_argument('--out')
    ap.add_argument('--rejects')
    ap.add_argument('--max-sites', type=int, default=2)
    ap.add_argument('--max-edit-size', type=int, default=8)
    a = ap.parse_args()

    eps = {}
    for f in Path(os.path.expanduser(a.intents)).glob('*.json'):
        d = json.loads(f.read_text())
        for i, e in enumerate(d.get('episodes') or d.get('intents') or []):
            eps['%s#%d' % (f.stem, i)] = e

    rows = [r for r in csv.DictReader(open(os.path.expanduser(a.resolved)))
            if (r.get('delta') or '').strip()]
    n0 = len(rows)
    st = collections.Counter()
    kept, rejected = [], []
    seen_pairs = set()

    for r in rows:
        ep = eps.get(r.get('intent_id')) or {}
        text = ep.get('design_rationale') or ''

        # 1. normalise structures FIRST
        ma, sa = largest_fragment(r['mol_a'])
        mb, sb = largest_fragment(r['mol_b'])
        if ma is None or mb is None:
            st['unparseable'] += 1
            rejected.append((r, 'unparseable'))
            continue
        if sa != r['mol_a'] or sb != r['mol_b']:
            st['normalised_salt'] += 1
        r['mol_a'], r['mol_b'] = sa, sb
        if sa == sb:
            st['no_edit_after_strip'] += 1
            rejected.append((r, 'identical after salt strip'))
            continue

        # 2. dedup on the canonical pair within a target+assay
        key = (sa, sb, r.get('target_chembl_id'), r.get('assay_id'))
        if key in seen_pairs:
            st['duplicate_pair'] += 1
            rejected.append((r, 'duplicate pair+assay'))
            continue
        seen_pairs.add(key)

        # 3. value sanity
        bad_val = False
        for v in (r.get('value_a'), r.get('value_b')):
            try:
                x = float(v)
                if x < 0.01 or x > 100000:
                    bad_val = True
            except (TypeError, ValueError):
                pass
        if bad_val:
            st['value_out_of_range'] += 1
            rejected.append((r, 'potency outside 0.01 nM - 100 uM'))
            continue

        # 4. strip the brief's own vocabulary out of the label
        clean = LEAK2.sub('', LEAK.sub('', text)).strip(' ,;')
        if clean != text:
            st['meta_clause_stripped'] += 1
        r['design_rationale'] = clean

        # 5. a reason with no chemistry at all cannot be evaluated by anything downstream.
        # REQUIRING A LOCANT AS WELL WAS WRONG and dropped 69.5% of the corpus: "Lower the pKa of
        # the dihydropyrrole nitrogen with fluorinated alkyl or nitrile substituents to improve
        # permeability" names its chemistry precisely and carries no locant. Most real rationales
        # do not give one. The locant survives as a RECORDED SIGNAL, not a gate -- it is useful
        # for ranking which rows a chemist should see first.
        if not GROUPWORD.search(clean):
            st['contentless_reason'] += 1
            rejected.append((r, 'no chemistry named in the reason'))
            continue
        r['has_locant'] = 'yes' if LOCANT.search(clean) else 'no'

        # 6. contradiction: "tolerated" against a real move
        if TOLERATED.search(clean) and abs(float(r['delta'])) > 0.5:
            st['tolerated_vs_big_delta'] += 1
            rejected.append((r, 'says tolerated; |delta| > 0.5'))
            continue

        # 7. a stereochemistry change nobody mentioned
        if stereo_signature(ma) != stereo_signature(mb) and not STEREO_WORD.search(clean):
            st['silent_stereo_change'] += 1
            rejected.append((r, 'stereochemistry changes, reason never mentions it'))
            continue

        # 8. MCS last, on survivors only
        sites, size = edit_profile(ma, mb)
        r['n_edit_sites'], r['edit_size'] = sites, size
        if sites > a.max_sites:
            st['multi_site_edit'] += 1
            rejected.append((r, 'edit spans %d disconnected sites' % sites))
            continue
        if size > a.max_edit_size:
            st['edit_too_large'] += 1
            rejected.append((r, 'edit size %d heavy atoms' % size))
            continue

        # 9. labels, not drops
        r['bucket'] = ('admet' if (ADMET.search(clean) and not POTENCY.search(clean))
                       else 'potency')
        st['bucket_' + r['bucket']] += 1
        r['small_delta'] = 'yes' if abs(float(r['delta'])) < 0.3 else 'no'
        st['small_delta'] += (r['small_delta'] == 'yes')
        kept.append(r)

    print('%d triplets in\n' % n0)
    for k in ('normalised_salt', 'meta_clause_stripped'):
        print('  (fixed)  %-24s %6d  %5.1f%%' % (k, st[k], 100 * st[k] / n0))
    print()
    for k in ('unparseable', 'no_edit_after_strip', 'duplicate_pair', 'value_out_of_range',
              'contentless_reason', 'tolerated_vs_big_delta', 'silent_stereo_change',
              'multi_site_edit', 'edit_too_large'):
        if st[k]:
            print('  (dropped) %-24s %6d  %5.1f%%' % (k, st[k], 100 * st[k] / n0))
    print('\n  KEPT %d (%.1f%%)' % (len(kept), 100 * len(kept) / n0))
    print('    of which bucket=potency %d, bucket=admet %d'
          % (st['bucket_potency'], st['bucket_admet']))
    print('    of which |delta| < 0.3 (exclude from direction learning) %d' % st['small_delta'])
    print('    direction-usable potency triplets: %d'
          % sum(1 for r in kept if r['bucket'] == 'potency' and r['small_delta'] == 'no'))
    print('    of which name an explicit locant: %d'
          % sum(1 for r in kept if r.get('has_locant') == 'yes'))
    print('    distinct rationales among kept: %d'
          % len({r.get('design_rationale') for r in kept}))

    if a.out and kept:
        with open(os.path.expanduser(a.out), 'w', newline='') as fh:
            w = csv.DictWriter(fh, fieldnames=list(kept[0].keys()))
            w.writeheader()
            w.writerows(kept)
        print('\nwrote %s' % a.out)
    if a.rejects and rejected:
        with open(os.path.expanduser(a.rejects), 'w', newline='') as fh:
            w = csv.writer(fh)
            w.writerow(['doc_id', 'intent_id', 'mol_a', 'mol_b', 'why'])
            for r, why in rejected:
                w.writerow([r.get('doc_id'), r.get('intent_id'), r.get('mol_a'),
                            r.get('mol_b'), why])
        print('wrote %s (%d rows)' % (a.rejects, len(rejected)))


if __name__ == '__main__':
    main()
