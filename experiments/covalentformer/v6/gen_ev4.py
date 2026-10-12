#!/usr/bin/env python3
"""Regenerate evidence with every defect the two reviewers found, then hard-QA it."""
import json, os, sys
from multiprocessing import Pool, cpu_count
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_v3 import build
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
def one(r):
    # a +15 heavy-atom change is not one design decision; reviewer 2: "compressing a 21-heavy-atom
    # dimerisation into one pocket sentence is a fiction regardless of the geometry fields"
    if abs(r['heavy'][1]-r['heavy'][0]) > 15: return {'_err':'edit too large'}
    try:
        e=build(r)
        return e if e else {'_err':'no usable complex'}
    except Exception as ex:
        return {'_err':type(ex).__name__}
if __name__=='__main__':
    rows=[json.loads(l) for l in open(D+'/v6_pairs_tc50.jsonl')]
    print('candidates: %s'%format(len(rows),','), flush=True)
    ok=0; drop={}
    out=open(D+'/v6_evidence_v4.jsonl','w')
    with Pool(max(1,cpu_count()-1)) as p:
        for n,e in enumerate(p.imap_unordered(one, rows, chunksize=8),1):
            if '_err' in e: drop[e['_err']]=drop.get(e['_err'],0)+1
            else: out.write(json.dumps(e)+'\n'); ok+=1
            if n%2500==0: print('  %d/%d ok %d'%(n,len(rows),ok), flush=True)
    out.close()
    print('kept %s'%format(ok,','))
    for k,v in sorted(drop.items(), key=lambda x:-x[1]): print('  dropped %-22s %s'%(k,format(v,',')))