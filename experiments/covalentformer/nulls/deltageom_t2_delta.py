"""Delta-geometry coverage + descriptor null, recomputed from disk on ALL rows."""
import os as _os
_AUD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'aud_intermediates')
# REPOINTED. This script was copied out of /tmp and still read and wrote /tmp/aud/*.json, so
# preserving it preserved nothing: t1c writes eldiff.json, which t1f and t2_delta both CONSUME.
# The chain broke the moment /tmp cleared. Intermediates are now committed beside the scripts.
import sys,json,glob,csv,collections,statistics as st
sys.path.insert(0,'/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reachability import electrophile_index
D='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/data'

def load(p):
    return [json.loads(l) for l in open(p) if l.strip()]
old_files=[('envelopes.jsonl',load(D+'/envelopes.jsonl')),('envelopes_20k.jsonl',load(D+'/envelopes_20k.jsonl'))]
new=[r for p in sorted(glob.glob(D+'/envcov/shard*.jsonl')) for r in load(p)]
POOL_OLD={}; SRC={}
for nm,rs in old_files:
    for r in rs:
        if r['frag'] not in POOL_OLD: POOL_OLD[r['frag']]=r; SRC[r['frag']]=nm
POOL_NEW=dict(POOL_OLD)
for r in new: POOL_NEW[r['frag']]=r; SRC.setdefault(r['frag'],'envcov')
for r in new: SRC[r['frag']]='envcov'
print('pool_old=%d  pool_new=%d'%(len(POOL_OLD),len(POOL_NEW)))

tainted={x[0] for x in json.load(open(_os.path.join(_AUD,'eldiff.json')))['now_rejected']}
print('tainted (old rows the new acceptance rule rejects): %d'%len(tainted))

# per-fragment descriptors
def descr(smi):
    m=Chem.MolFromSmiles(smi)
    if m is None: return None
    star=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]
    if len(star)!=1: return None
    el,cls=electrophile_index(m)
    if el is None: return None
    try:
        path=Chem.GetShortestPath(m,star[0],el)
        nb=len(path)-1
    except Exception: return None
    return dict(bond_path=nb,n_heavy=m.GetNumHeavyAtoms()-1,nbonds=m.GetNumBonds())
DES={}
for f in POOL_NEW:
    d=descr(f)
    if d is not None: DES[f]=d
print('descriptors computed for %d/%d pool fragments'%(len(DES),len(POOL_NEW)))

MED={f:(st.median(r['d']),st.median(r['theta']),len(r['d'])) for f,r in POOL_NEW.items()}

rows=list(csv.DictReader(open(D+'/roles/train_strat.csv')))
print('train_strat rows=%d'%len(rows))

def cover(pool):
    return [r for r in rows if r['frag_from'] in pool and r['frag_to'] in pool]
cov_old=cover(POOL_OLD); cov_new=cover(POOL_NEW)
print('COVERAGE old-pool  = %d  (%.2f%% of train rows)'%(len(cov_old),100*len(cov_old)/len(rows)))
print('COVERAGE new-pool  = %d  (%.2f%% of train rows)'%(len(cov_new),100*len(cov_new)/len(rows)))
tain_rows=[r for r in cov_new if r['frag_from'] in tainted or r['frag_to'] in tainted]
print('  rows touching a TAINTED fragment: %d (%.2f%% of new coverage)'%(len(tain_rows),100*len(tain_rows)/len(cov_new)))
# rows touching the n_conf=150 file
n150=[r for r in cov_new if SRC.get(r['frag_from'])=='envelopes_20k.jsonl' or SRC.get(r['frag_to'])=='envelopes_20k.jsonl']
mixed=[r for r in cov_new if (SRC.get(r['frag_from'])=='envelopes_20k.jsonl')!=(SRC.get(r['frag_to'])=='envelopes_20k.jsonl')]
print('  rows touching envelopes_20k (n_conf=150): %d (%.2f%%)'%(len(n150),100*len(n150)/len(cov_new)))
print('  rows MIXING a 150-conf and a 300-conf fragment (delta is part sampling artifact): %d (%.2f%%)'%(len(mixed),100*len(mixed)/len(cov_new)))

def stats(cov,label):
    dr=[];db=[];dh=[];keep=[]
    for r in cov:
        a,b=r['frag_from'],r['frag_to']
        if a not in DES or b not in DES: continue
        dr.append(MED[b][0]-MED[a][0]); db.append(DES[b]['bond_path']-DES[a]['bond_path'])
        dh.append(DES[b]['n_heavy']-DES[a]['n_heavy']); keep.append(r)
    dr=np.array(dr);db=np.array(db);dh=np.array(dh)
    from scipy.stats import pearsonr,spearmanr
    pr,_=pearsonr(dr,db); sp,_=spearmanr(dr,db)
    prh,_=pearsonr(dr,dh)
    print('\n[%s] n=%d'%(label,len(dr)))
    print('  rho(delta_reach, delta_BOND_PATH)  pearson %+0.4f  (r2=%.4f)  spearman %+0.4f'%(pr,pr*pr,sp))
    print('  rho(delta_reach, delta_n_heavy)    pearson %+0.4f  (r2=%.4f)'%(prh,prh*prh))
    m=(db==0)
    print('  PURE-SHAPE subset (delta bond_path == 0): %d rows = %.2f%%'%(m.sum(),100*m.mean()))
    if m.sum():
        print('     of those |delta_reach| > 0.5 A : %.2f%%   > 1.0 A : %.2f%%   median |delta| %.3f A'%(
            100*(np.abs(dr[m])>0.5).mean(),100*(np.abs(dr[m])>1.0).mean(),float(np.median(np.abs(dr[m])))))
    print('  |delta_reach| overall: median %.3f A  p90 %.3f A'%(float(np.median(np.abs(dr))),float(np.percentile(np.abs(dr),90))))
    return dr,db,keep
dr_o,db_o,_=stats(cov_old,'OLD POOL (reproduces the 64,308 claim)')
dr_n,db_n,keep_n=stats(cov_new,'NEW POOLED (reproduces the 88,891 claim)')
# clean version: drop tainted
cov_clean=[r for r in cov_new if r['frag_from'] not in tainted and r['frag_to'] not in tainted]
stats(cov_clean,'NEW POOLED, TAINTED FRAGMENTS DROPPED')
# and drop the n_conf=150 file entirely
cov_h=[r for r in cov_new if SRC.get(r['frag_from'])!='envelopes_20k.jsonl' and SRC.get(r['frag_to'])!='envelopes_20k.jsonl']
stats(cov_h,'HOMOGENEOUS n_conf=300 ONLY (envelopes.jsonl + envcov)')
json.dump({'cov_new':[[r['frag_from'],r['frag_to'],r['role']] for r in keep_n]},open(_os.path.join(_AUD,'cov_new.json'),'w'))
