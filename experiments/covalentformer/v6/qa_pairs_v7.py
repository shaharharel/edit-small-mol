#!/usr/bin/env python3
"""Independent QA of v7_pairs.jsonl. Re-derives every invariant from meta.json and the SMILES
rather than trusting the builder, because the v6 defects were all things the builder believed.
"""
import collections, json, sys
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')

D = '/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
NULLS = {'', '?', '.', 'None', 'null'}
M = json.load(open(D + '/meta.json'))
rows = [json.loads(l) for l in open(D + '/' + (sys.argv[1] if len(sys.argv) > 1 else 'v7_pairs.jsonl'))]
bad, warn = collections.Counter(), collections.Counter()

seen = set()
for r in rows:
    hp, lp, tgt = r['pdb'], r['analogue_pdb'], r['target']
    hv, lv = M.get(hp, {}), M.get(lp, {})

    # --- the three defects that produced the v6 corpus
    if hp == lp:
        bad['hit and lead are the same crystal'] += 1
    if tgt not in (hv.get('unp') or ()) or tgt not in (lv.get('unp') or ()):
        bad['target is not an accession of BOTH entries'] += 1
    def cite(v):
        for k in ('doi', 'pm'):
            x = v.get(k)
            if x is not None and str(x).strip() not in NULLS:
                return str(x).strip()
        return None
    ch, cl = cite(hv), cite(lv)
    if ch is None or cl is None:
        bad['an entry has no citation'] += 1
    elif ch != cl:
        bad['hit and lead are from DIFFERENT papers'] += 1
    elif ch != r['paper']:
        bad['stated paper disagrees with meta.json'] += 1
    if str(tgt).strip() in NULLS:
        bad['target is a null placeholder'] += 1

    # --- uniqueness at the PAIR level
    k = (r['input'], r['analogue'])
    if k in seen:
        bad['duplicate (hit SMILES, lead SMILES)'] += 1
    seen.add(k)

    # --- chemistry
    h, l = Chem.MolFromSmiles(r['input']), Chem.MolFromSmiles(r['analogue'])
    if h is None or l is None:
        bad['SMILES does not parse'] += 1
        continue
    t = DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(h, 2, 2048),
                                       AllChem.GetMorganFingerprintAsBitVect(l, 2, 2048))
    if t < 0.50:
        bad['Tanimoto below the 0.50 floor'] += 1
    if abs(t - r['tanimoto']) > 0.02:
        bad['stated Tanimoto disagrees with the molecules'] += 1
    if Chem.MolToSmiles(h, isomericSmiles=False) == Chem.MolToSmiles(l, isomericSmiles=False):
        bad['hit and lead identical ignoring stereochemistry'] += 1
    if [h.GetNumHeavyAtoms(), l.GetNumHeavyAtoms()] != r['heavy']:
        bad['heavy-atom counts disagree'] += 1
    if r['input_code'] == r['analogue_code']:
        bad['hit and lead are the same ligand code'] += 1

n = len(rows)
print('v7 pairs                %6s' % format(n, ','))
print('distinct targets        %6d' % len({r['target'] for r in rows}))
print('distinct papers         %6d' % len({r['paper'] for r in rows}))
print('distinct hit crystals   %6d' % len({r['pdb'] for r in rows}))
tc = sorted(r['tanimoto'] for r in rows)
hd = sorted(r['heavy'][1] - r['heavy'][0] for r in rows)
print('tanimoto   p5 %.2f median %.2f p95 %.2f' % (tc[n // 20], tc[n // 2], tc[19 * n // 20]))
print('heavy delta p5 %d median %d p95 %d  max %d' % (hd[n // 20], hd[n // 2], hd[19 * n // 20], hd[-1]))
tg = collections.Counter(r['target'] for r in rows).most_common(1)[0]
print('largest target series   %6d rows (%.1f%%)  %s' % (tg[1], 100 * tg[1] / n, tg[0]))

dd = [r['heavy'][1] - r['heavy'][0] for r in rows]
neg, pos = sum(1 for x in dd if x < 0), sum(1 for x in dd if x > 0)
print('direction balance        answer smaller %.1f%%  larger %.1f%%  equal %.1f%%'
      % (100.0 * neg / n, 100.0 * pos / n, 100.0 * (n - neg - pos) / n))
if neg == 0 or pos == 0:
    bad['direction is one-sided: the answer is never smaller/never larger'] += 1
elif abs(neg - pos) > 0.25 * (neg + pos):
    warn['direction is skewed by more than 25%'] += 1

print('\nBLOCKERS' if bad else '\nBLOCKERS: none')
for k, v in bad.most_common():
    print('  %-52s %6s' % (k, format(v, ',')))
print('\nWARNINGS' if warn else 'WARNINGS: none')
for k, v in warn.most_common():
    print('  %-52s %6s' % (k, format(v, ',')))
sys.exit(1 if bad else 0)
