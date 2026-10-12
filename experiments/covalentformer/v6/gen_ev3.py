#!/usr/bin/env python3
import json, os, sys
from multiprocessing import Pool, cpu_count
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from evidence_v3 import build
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
def one(r):
    try:
        e=build(r)
        return e if e else {'_err':'no pocket'}
    except Exception as ex:
        return {'_err':type(ex).__name__}
if __name__=='__main__':
    rows=[json.loads(l) for l in open(D+'/v6_pairs_tc50.jsonl')]
    print('pairs: %s' % format(len(rows),','), flush=True)
    ok=err=0; out=open(D+'/v6_evidence_v3.jsonl','w')
    with Pool(max(1,cpu_count()-1)) as p:
        for n,e in enumerate(p.imap_unordered(one, rows, chunksize=8),1):
            if '_err' in e: err+=1
            else: out.write(json.dumps(e)+'\n'); ok+=1
            if n%1500==0: print('  %d/%d ok %d err %d'%(n,len(rows),ok,err), flush=True)
    out.close(); print('DONE %s rows (err %d)'%(format(ok,','),err))