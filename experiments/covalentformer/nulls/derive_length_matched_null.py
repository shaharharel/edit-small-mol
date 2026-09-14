"""DERIVE LOOKUP_NULL_BINNED from artifacts on disk.

A "shape-blind pure-length lookup policy" = it sees d_req, picks a BOND COUNT by a length lookup,
then emits a fragment drawn at random from the real fragment population at that bond count. Its
reach therefore depends on d_req ONLY through length. Scored with sweep_test.py's exact binned
metric (same bins, same spearman, same MIN_STRATUM=12, same n-weighted pooling).

One free parameter: the lookup TOLERANCE tau. tau->0 is a hard lookup, tau->inf ignores d.
We sweep tau and read the null off at the tau that reproduces the model's OWN length response
rho(d_req, bonds) = +0.4992 -- the only calibration that makes the null comparable to the model.
"""
import json, collections, numpy as np, sys, os
D_SWEEP=[4.,5.,6.,7.,8.,9.]; MIN_STRATUM=12
def spearman(x,y):
    x,y=np.asarray(x,float),np.asarray(y,float)
    if len(x)<4 or len(set(x.tolist()))<2 or len(set(y.tolist()))<2: return float('nan')
    rx=np.argsort(np.argsort(x)); ry=np.argsort(np.argsort(y))
    return float(np.corrcoef(rx,ry)[0,1])
def bin_of(nb):
    return ('3-4' if nb<=4 else '5-6' if nb<=6 else '7-9' if nb<=9 else '10-13' if nb<=13 else '14+')
def score(recs):
    by=collections.defaultdict(list)
    for r in recs: by[bin_of(r[1])].append(r)
    rhos=[]; drop=[]
    for nb in sorted(by):
        g=by[nb]
        if len(g)<MIN_STRATUM: drop.append((nb,len(g))); continue
        rhos.append((nb,len(g),spearman([x[0] for x in g],[x[2] for x in g])))
    if not rhos: return float('nan'),rhos,drop
    return float(np.average([r for _,_,r in rhos],weights=[c for _,c,_ in rhos])),rhos,drop

# DEFAULT TO THE REPO COPY. Defaulting to /tmp meant the documented invocation silently depended
# on a file no checkout contains, so the constant sweep_test.py hardcodes twice could not be
# re-derived on a fresh machine. /tmp is still reachable by passing the path explicitly.
POOL=sys.argv[1] if len(sys.argv)>1 else os.path.join(
    os.path.dirname(os.path.abspath(__file__)),'_pool_envelopes.json')
NTOT=int(sys.argv[2]) if len(sys.argv)>2 else 2265
pool=json.load(open(POOL))
by_b=collections.defaultdict(list)
for b,m,_ in pool: by_b[b].append(m)
bonds=sorted(by_b); med={b:float(np.median(by_b[b])) for b in bonds}
arr={b:np.array(by_b[b]) for b in bonds}
print('pool %s  n=%d  bond counts %d..%d   n per d level = %d'%(POOL,len(pool),bonds[0],bonds[-1],NTOT//6))

def simulate(tau,rng,nper):
    recs=[]
    for d in D_SWEEP:
        w=np.array([np.exp(-((med[b]-d)**2)/(2*tau*tau)) for b in bonds]); w/=w.sum()
        bs=rng.choice(len(bonds),nper,p=w)
        for i in bs:
            b=bonds[i]; recs.append((d,b,float(rng.choice(arr[b]))))
    return recs

print('\n%6s %8s %10s %12s %10s'%('tau','rho(d,b)','rho(d,reach)','BINNED null','nan-reps'))
rowsout=[]
for tau in [0.15,0.25,0.4,0.6,0.8,1.0,1.25,1.5,2.0,2.5,3.0,4.0,6.0,10.0]:
    pb=[];pr=[];bn=[];nan=0
    for rep in range(200):
        rng=np.random.default_rng(100000+rep)
        recs=simulate(tau,rng,NTOT//6)
        pb.append(spearman([r[0] for r in recs],[r[1] for r in recs]))
        pr.append(spearman([r[0] for r in recs],[r[2] for r in recs]))
        v,_,_=score(recs)
        if np.isnan(v): nan+=1
        else: bn.append(v)
    print('%6.2f %8.4f %12.4f %7.4f+-%.4f %6d/200'%(tau,np.mean(pb),np.mean(pr),np.mean(bn) if bn else float('nan'),np.std(bn) if bn else 0,nan))
    rowsout.append((tau,np.mean(pb),np.mean(pr),np.mean(bn) if bn else float('nan'),np.std(bn) if bn else 0))
# calibrate: tau where rho(d,bonds) == model's +0.4992
xs=[r[1] for r in rowsout]; ys=[r[3] for r in rowsout]; ts=[r[0] for r in rowsout]
tgt=0.4992
o=np.argsort(xs)
xs2=np.array(xs)[o]; ys2=np.array(ys)[o]; ts2=np.array(ts)[o]
print('\nCALIBRATED NULL at the model\'s own length response rho(d,bonds)=%.4f:'%tgt)
print('  interpolated tau        = %.3f'%np.interp(tgt,xs2,ts2))
print('  interpolated BINNED null= %+.4f'%np.interp(tgt,xs2,ys2))
print('  model binned (3 seeds)  = +0.1389 (sd 0.0175)')
