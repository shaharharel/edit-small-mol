
import os as _os
_AUD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'aud_intermediates')
# REPOINTED. This script was copied out of /tmp and still read and wrote /tmp/aud/*.json, so
# preserving it preserved nothing: t1c writes eldiff.json, which t1f and t2_delta both CONSUME.
# The chain broke the moment /tmp cleared. Intermediates are now committed beside the scripts.
import sys,json,random,statistics as st,collections
sys.path.insert(0,'/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
from reach_envelope import envelope
D='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/data'
diff={x[0] for x in json.load(open(_os.path.join(_AUD,'eldiff.json')))['now_rejected']}
byfile={}
for f in ['envelopes.jsonl','envelopes_20k.jsonl']:
    byfile[f]={json.loads(l)['frag']:json.loads(l) for l in open(D+'/'+f)}
rng=random.Random(20260914)
out={}
for f,NC_GUESS in [('envelopes_20k.jsonl',150),('envelopes.jsonl',300)]:
    cand=sorted(x for x in byfile[f] if x not in diff)
    samp=rng.sample(cand,20)
    res=collections.Counter(); rows=[]
    for s in samp:
        r=byfile[f][s]
        e150=envelope(s,n_conf=150); e300=envelope(s,n_conf=300)
        if 'err' in e150 or 'err' in e300: res['err']+=1; continue
        for tag,e in (('150',e150),('300',e300)):
            if [round(x,2) for x in e['d']]==r['d'] and [round(x,1) for x in e['theta']]==r['theta']:
                res['exact_at_'+tag]+=1
        d1,d3=e150['d'],e300['d']
        rows.append(dict(frag=s,n150=len(d1),n300=len(d3),
            med150=st.median(d1),med300=st.median(d3),
            max150=max(d1),max300=max(d3),min150=min(d1),min300=min(d3),
            sd150=st.pstdev(d1),sd300=st.pstdev(d3),
            th_med150=st.median(e150['theta']),th_med300=st.median(e300['theta'])))
    print('%s n=%d %s'%(f,len(samp),dict(res)),flush=True)
    if rows:
        for k in ['med','max','min','sd','th_med']:
            dl=[r[k+'300']-r[k+'150'] for r in rows]
            print('   d(%s) 300-150: mean %+0.4f  median %+0.4f  max|.| %.4f'%(k,sum(dl)/len(dl),st.median(dl),max(abs(x) for x in dl)),flush=True)
        rng2=[r['max300']-r['max150'] for r in rows]
        print('   REACH EXTENT (max d) grew in %d/%d fragments, never shrank: %s'%(sum(1 for x in rng2 if x>0),len(rng2),all(x>=-1e-9 for x in rng2)),flush=True)
    out[f]=rows
json.dump(out,open(_os.path.join(_AUD,'nconf_effect.json'),'w'))
print('DONE')
