"""OLD vs NEW electrophile_index over every fragment in the old envelope files.
OLD logic reconstructed VERBATIM from the fix docstring in reachability.py:
  'the loop returned on the FIRST class with exactly one match' and
  'the multiple-warhead guard only ever fired WITHIN one class'.
"""
import os as _os
_AUD = _os.path.join(_os.path.dirname(_os.path.abspath(__file__)), 'aud_intermediates')
# REPOINTED. This script was copied out of /tmp and still read and wrote /tmp/aud/*.json, so
# preserving it preserved nothing: t1c writes eldiff.json, which t1f and t2_delta both CONSUME.
# The chain broke the moment /tmp cleared. Intermediates are now committed beside the scripts.
import sys,json,collections
sys.path.insert(0,'/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reachability import electrophile_index as NEW, PATS

def OLD(mol):
    for name,p,idx in PATS:
        if p is None: continue
        mt=mol.GetSubstructMatches(p)
        if mt:
            if len(mt)>1: return None,'multiple_warheads'
            return mt[0][idx],name
    return None,None

# DEAD LOAD REMOVED. This read sets.json into `S`, which was never referenced again -- the script
# only needed the file to EXIST. It was the root of the /tmp chain and the reason t1c could not run
# standalone. Deleting one line makes t1c self-sufficient, which unblocks t1f and t2_delta, and
# drops sets.json (783 KB) from the preserved set entirely.
D='/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer/data'
stored={}
for f in ['envelopes.jsonl','envelopes_20k.jsonl']:
    for ln in open(D+'/'+f):
        r=json.loads(ln); stored.setdefault(r['frag'],r['cls'])
frags=sorted(stored)
print('old-file fragments: %d'%len(frags))
c=collections.Counter(); diff_atom=[]; now_rejected=[]; cls_changed=[]
for s in frags:
    m=Chem.MolFromSmiles(s)
    if m is None: c['unparseable']+=1; continue
    ao,co=OLD(m); an,cn=NEW(m)
    if ao==an and co==cn: c['identical']+=1
    else:
        c['DIVERGENT']+=1
        if an is None and ao is not None:
            now_rejected.append((s,co,cn)); c['now_rejected']+=1
        elif ao!=an:
            diff_atom.append((s,ao,co,an,cn)); c['diff_atom_index']+=1
        else:
            cls_changed.append((s,co,cn)); c['class_only_changed']+=1
    # does stored cls match OLD or NEW?
    if stored[s]==co: c['stored_matches_OLD']+=1
    if stored[s]==cn: c['stored_matches_NEW']+=1
print(dict(c))
print('\n--- now_rejected (would produce NO envelope under new code) sample ---')
for x in now_rejected[:10]: print('  ',x)
print('\n--- different ELECTROPHILE ATOM (geometry itself changes) sample ---')
for x in diff_atom[:10]: print('  ',x)
print('\n--- class label changed only (atom same) sample ---')
for x in cls_changed[:10]: print('  ',x)
json.dump({'now_rejected':now_rejected,'diff_atom':diff_atom,'cls_changed':cls_changed},
          open(_os.path.join(_AUD,'eldiff.json'),'w'))
