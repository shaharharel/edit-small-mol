"""Which 'delta_bond_count' definition did the lead use? Sweep candidates on the OLD pool
(the 64,308 set their +0.7781 / 6.1% / 82.3% came from)."""
import sys,json,glob,csv,statistics as st
sys.path.insert(0,'/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
import numpy as np
from scipy.stats import pearsonr,spearmanr
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reachability import electrophile_index
D='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/data'
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
POOL={}
for f in ['envelopes.jsonl','envelopes_20k.jsonl']:
    for r in load(D+'/'+f): POOL.setdefault(r['frag'],r)
POOL_NEW=dict(POOL)
for p in sorted(glob.glob(D+'/envcov/shard*.jsonl')):
    for r in load(p): POOL_NEW[r['frag']]=r
def descr(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    star=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]
    if len(star)!=1: return None
    el,_=electrophile_index(m)
    if el is None: return None
    mh=Chem.AddHs(m)
    return dict(bond_path=len(Chem.GetShortestPath(m,star[0],el))-1,
                n_heavy=m.GetNumHeavyAtoms()-1,
                nbonds=m.GetNumBonds(),
                nbonds_heavy=sum(1 for b in m.GetBonds() if b.GetBeginAtom().GetAtomicNum()>0 and b.GetEndAtom().GetAtomicNum()>0),
                rot=Chem.rdMolDescriptors.CalcNumRotatableBonds(m),
                nbonds_H=mh.GetNumBonds())
DES={f:d for f in POOL_NEW if (d:=descr(f)) is not None}
MEDD={f:st.median(r['d']) for f,r in POOL_NEW.items()}
rows=list(csv.DictReader(open(D+'/roles/train_strat.csv')))
KEYS=['bond_path','n_heavy','nbonds','nbonds_heavy','rot','nbonds_H']
for label,pool in [('OLD POOL n=64,308',POOL),('NEW POOLED n=88,891',POOL_NEW)]:
    cov=[r for r in rows if r['frag_from'] in pool and r['frag_to'] in pool
         and r['frag_from'] in DES and r['frag_to'] in DES]
    dr=np.array([MEDD[r['frag_to']]-MEDD[r['frag_from']] for r in cov])
    print('\n===== %s  (usable after descriptors: %d) ====='%(label,len(cov)))
    print('%-14s %8s %8s %8s | %10s %12s %12s'%('delta of','pearson','r2','spearman','pure-shape%','|d|>0.5A%','med|d| A'))
    for k in KEYS:
        db=np.array([DES[r['frag_to']][k]-DES[r['frag_from']][k] for r in cov])
        pr,_=pearsonr(dr,db); sp,_=spearmanr(dr,db); m=(db==0)
        print('%-14s %+8.4f %8.4f %+8.4f | %9.2f%% %11.2f%% %11.3f'%(
            k,pr,pr*pr,sp,100*m.mean(),100*(np.abs(dr[m])>0.5).mean() if m.sum() else float('nan'),
            float(np.median(np.abs(dr[m]))) if m.sum() else float('nan')))
