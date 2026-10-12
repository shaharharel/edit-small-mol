#!/usr/bin/env python3
"""QA gate on the v6 pair corpus. Exits non-zero if anything that would poison labelling is found."""
import json, os, sys, collections
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
RDLogger.DisableLog('rdApp.*')
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
f=sys.argv[1] if len(sys.argv)>1 else D+'/v6_pairs.jsonl'
rows=[json.loads(l) for l in open(f)]
print('rows: %s' % format(len(rows),','))
bad=collections.Counter(); seen=set(); dup=0
COF=set('NAD NAP NDP FAD FMN ATP ADP AMP GTP GDP GMP ANP ACP SAM SAH COA HEM UDP UTP CTP PLP'.split())
tcs=[]; ev_mismatch=0
for r in rows:
    h=Chem.MolFromSmiles(r['hit']); a=Chem.MolFromSmiles(r['analogue'])
    if h is None or a is None: bad['unparseable SMILES']+=1; continue
    if Chem.MolToSmiles(h)==Chem.MolToSmiles(a): bad['hit == analogue']+=1
    if r['hit_code'] in COF or r['ana_code'] in COF: bad['cofactor survived']+=1
    t=DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(h,2,2048),
                                     AllChem.GetMorganFingerprintAsBitVect(a,2,2048))
    tcs.append(t)
    if not (0.29<=t<=1.001): bad['Tanimoto out of band']+=1
    if not (10<=r['heavy'][0]<=70 and 10<=r['heavy'][1]<=70): bad['heavy atoms out of range']+=1
    if not (10<=len(r['pocket_residues'])<=200): bad['pocket size implausible']+=1
    if not r['ligand_evidence']: bad['no ligand evidence']+=1
    else:
        ds=[e['d'] for e in r['ligand_evidence']]
        if min(ds)<0.8 or max(ds)>25: bad['evidence distance implausible']+=1
        if abs(len(r['ligand_evidence'])-r['heavy'][0])>6: ev_mismatch+=1
    k=(r['target'],r['hit_code'],r['ana_code'])
    if k in seen: dup+=1
    seen.add(k)
    if r['heavy'][0]>r['heavy'][1]: bad['orientation not smaller->larger']+=1
tcs.sort()
print('\nTanimoto: p10 %.2f  median %.2f  p90 %.2f' % (tcs[len(tcs)//10],tcs[len(tcs)//2],tcs[9*len(tcs)//10]))
print('duplicate (target,hitcode,anacode): %d' % dup)
print('evidence-atom count differs from hit heavy count by >6: %d (%.1f%%)'
      % (ev_mismatch, 100.0*ev_mismatch/max(1,len(rows))))
tc=collections.Counter(r['target'] for r in rows)
print('\ntargets %d | series %d' % (len(tc), len({(r['target'],r['paper']) for r in rows})))
print('top target share: %.1f%% (%s)' % (100.0*tc.most_common(1)[0][1]/len(rows), tc.most_common(1)[0][0]))
print('top-10 targets hold %.1f%% of rows' % (100.0*sum(n for _,n in tc.most_common(10))/len(rows)))
print('\nPROBLEMS:' if bad else '\nPROBLEMS: none')
for k,v in bad.most_common(): print('   %-34s %d' % (k,v))
sys.exit(1 if bad else 0)