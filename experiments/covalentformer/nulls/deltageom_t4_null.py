"""TARGET 4: can ECFP4 of (frag_from, frag_to) predict delta_reach WITHOUT conformers?
FRAGMENT-DISJOINT splits, 3 replicate seeds. If a fingerprint nails it, the label is a
topological restatement and the channel teaches nothing a tokeniser cannot already see."""
import sys,json,glob,csv,random,statistics as st
sys.path.insert(0,'/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
import numpy as np
from scipy import sparse
from sklearn.linear_model import Ridge, LogisticRegression
from sklearn.metrics import roc_auc_score, r2_score
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import rdFingerprintGenerator
RDLogger.DisableLog('rdApp.*')
from reachability import electrophile_index
D='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/data'
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
POOL={}
for f in ['envelopes.jsonl','envelopes_20k.jsonl']: 
    for r in load(D+'/'+f): POOL.setdefault(r['frag'],r)
for p in sorted(glob.glob(D+'/envcov/shard*.jsonl')):
    for r in load(p): POOL[r['frag']]=r
MEDD={f:st.median(r['d']) for f,r in POOL.items()}
NB=2048
gen=rdFingerprintGenerator.GetMorganGenerator(radius=2,fpSize=NB)
FP={}; BP={}
for f in POOL:
    m=Chem.MolFromSmiles(f)
    if m is None: continue
    star=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]
    el,_=electrophile_index(m)
    if len(star)!=1 or el is None: continue
    bv=gen.GetFingerprint(m); arr=np.zeros(NB,dtype=np.int8); DataStructs.ConvertToNumpyArray(bv,arr)
    FP[f]=arr; BP[f]=len(Chem.GetShortestPath(m,star[0],el))-1
rows=[r for r in csv.DictReader(open(D+'/roles/train_strat.csv'))
      if r['frag_from'] in FP and r['frag_to'] in FP]
print('rows usable = %d ; distinct fragments = %d'%(len(rows),len({r['frag_from'] for r in rows}|{r['frag_to'] for r in rows})))
y=np.array([MEDD[r['frag_to']]-MEDD[r['frag_from']] for r in rows])
dbp=np.array([BP[r['frag_to']]-BP[r['frag_from']] for r in rows],dtype=float)
frags=sorted({r['frag_from'] for r in rows}|{r['frag_to'] for r in rows})
X=sparse.vstack([sparse.csr_matrix(np.concatenate([FP[r['frag_from']],FP[r['frag_to']]]).astype(np.float32)) for r in rows]).tocsr()
print('X %s  y mean %.3f sd %.3f'%(X.shape,y.mean(),y.std()))
print('\n%-6s %8s %10s %10s %10s %10s %10s'%('seed','n_test','R2_ecfp','AUROC_sign','R2_bondpath','AUROC_bp','R2_shuffled'))
res=[]
for seed in [11,22,33]:
    rng=random.Random(seed); fs=list(frags); rng.shuffle(fs)
    te=set(fs[:int(0.2*len(fs))])
    ti=np.array([i for i,r in enumerate(rows) if r['frag_from'] not in te and r['frag_to'] not in te])
    vi=np.array([i for i,r in enumerate(rows) if r['frag_from'] in te and r['frag_to'] in te])
    if len(vi)<50: print('seed %d: too few test rows (%d)'%(seed,len(vi))); continue
    m=Ridge(alpha=1.0,solver='sparse_cg'); m.fit(X[ti],y[ti]); p=m.predict(X[vi])
    r2=r2_score(y[vi],p)
    sgn=(y[vi]>0).astype(int)
    au=roc_auc_score(sgn,p) if 0<sgn.mean()<1 else float('nan')
    # topology-only single-feature baseline, fit on the same rows
    a=np.polyfit(dbp[ti],y[ti],1); pb=np.polyval(a,dbp[vi])
    r2b=r2_score(y[vi],pb); aub=roc_auc_score(sgn,pb) if 0<sgn.mean()<1 else float('nan')
    # label-shuffled control
    ys=y[ti].copy(); np.random.RandomState(seed).shuffle(ys)
    ms=Ridge(alpha=1.0,solver='sparse_cg'); ms.fit(X[ti],ys); r2s=r2_score(y[vi],ms.predict(X[vi]))
    print('%-6d %8d %10.4f %10.4f %10.4f %10.4f %10.4f'%(seed,len(vi),r2,au,r2b,aub,r2s))
    res.append((r2,au,r2b,aub))
if res:
    print('\nMEAN over %d fragment-disjoint splits: R2_ecfp=%.4f  AUROC_sign=%.4f  |  R2_bondpath=%.4f  AUROC_bp=%.4f'%(
        len(res),np.mean([x[0] for x in res]),np.mean([x[1] for x in res]),
        np.mean([x[2] for x in res]),np.mean([x[3] for x in res])))
