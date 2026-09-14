"""Build the (bond count, reach_med) fragment population from the envelope artifacts."""
import json, sys, os, collections
import numpy as np
sys.path.insert(0,'/Users/shaharharel/Documents/github/edit-small-mol/experiments/covalentformer')
sys.path.insert(0,'/Users/shaharharel/Documents/github/REINVENT4')
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
from reachability import electrophile_index, announce_classifier_version
# This pool is built from envelopes*.jsonl, which were LABELLED v1, using v2 bond
# counts -- so the model statistic AND the null it is judged against are both
# recomputed across the boundary. Neither said so until now.
announce_classifier_version('build_length_null_pool')
def bond_count(frag_smi):
    m = Chem.MolFromSmiles(frag_smi)
    if m is None: return None
    star=[a.GetIdx() for a in m.GetAtoms() if a.GetAtomicNum()==0]
    el,_=electrophile_index(m)
    if not star or el is None: return None
    try: return len(Chem.GetShortestPath(m, star[0], el))-1
    except Exception: return None
for fn in ['data/envelopes.jsonl','data/envelopes_20k.jsonl']:
    p='experiments/covalentformer/'+fn
    pool=[]
    for line in open(p):
        r=json.loads(line)
        if 'd' not in r or not r['d']: continue
        nb=bond_count(r['frag'])
        if nb is None: continue
        pool.append((nb, float(np.median(r['d'])), len(r['d']), r['frag']))
    print('%s: %d fragments with bond count + reach' % (fn, len(pool)))
    c=collections.Counter(b for b,_,_,_ in pool)
    for b in sorted(c): 
        rs=[m for bb,m,_,_ in pool if bb==b]
        print('   bonds %2d  n=%5d  reach_med median %.2f  [p10 %.2f p90 %.2f]'%(b,c[b],np.median(rs),np.percentile(rs,10),np.percentile(rs,90)))
    # WRITE BESIDE THIS SCRIPT, NOT INTO /tmp. This pool is the first link in the chain that
    # produces the (0.0971, 0.1495) length-matched null, and sweep_test.py hardcodes that constant
    # in TWO places. A hardcoded number whose derivation lives in /tmp is an unsourced number the
    # moment the machine reboots -- #139's failure mode, and the reason #129/#154 went three
    # revisions before anyone could check them.
    _out = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                        '_pool_' + os.path.basename(fn).replace('.jsonl', '') + '.json')
    json.dump([[b, m, n] for b, m, n, _ in pool], open(_out, 'w'))
    print('   pool -> %s' % _out)
