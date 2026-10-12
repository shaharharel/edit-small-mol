#!/usr/bin/env python3
"""Recover the author-numbering offset between the hit's and the lead's crystal.

The two members of a pair are different PDB entries, deposited by different groups, so auth_seq_id
is whatever each depositor chose. One observed pair had the hit at MET357/PHE372 and the lead at
MET861/PHE877 -- the same residues, +504. That is a constant shift, not a different pocket, so
discarding those pairs (22% of the corpus) throws away good data.

Method: for every lead residue of type X and every hit residue of the same type, the candidate
offset is their numbering difference. The true offset is the mode, and it is accepted only if it
explains a majority of the lead's contacts -- otherwise the two really are different sites.
"""
import collections, json, re, sys
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
RES=re.compile(r'^([A-Z]{3})(\d+)')

def parse(tok):
    m=RES.match(tok)
    return (m.group(1), int(m.group(2))) if m else None

def hit_residues(e):
    out=set()
    for a in e['MODEL_INPUT']['atoms']:
        p=parse(a['faces'])
        if p: out.add(p)
    for a in e.get('LABEL_ONLY',{}).get('site_of_change',[]):
        for r in a.get('pocket_room_lining',[]):
            p=parse(r)
            if p: out.add(p)
    return out

def recover(e):
    lead=e.get('LABEL_ONLY',{}).get('lead_contacts') or []
    if not lead: return None,0.0
    H=hit_residues(e)
    L=[parse(c['faces']) for c in lead]
    L=[x for x in L if x]
    if not H or not L: return None,0.0
    byname=collections.defaultdict(list)
    for nm,num in H: byname[nm].append(num)
    votes=collections.Counter()
    for nm,num in L:
        for hn in byname.get(nm,()): votes[num-hn]+=1
    if not votes: return None,0.0
    off,cnt=votes.most_common(1)[0]
    return off, cnt/len(L)

if __name__=='__main__':
    src=sys.argv[1] if len(sys.argv)>1 else D+'/v6_evidence_final.jsonl'
    ev=[json.loads(l) for l in open(src)]
    stats=collections.Counter(); fixed=0; shifts=collections.Counter()
    out=open(D+'/v6_evidence_aligned.jsonl','w')
    for e in ev:
        off,conf=recover(e)
        lo=e.setdefault('LABEL_ONLY',{})
        if off is None:
            stats['no lead contacts']+=1
        elif off==0 and conf>=0.3:
            stats['already aligned']+=1
        elif conf>=0.5:
            for c in lo.get('lead_contacts',[]):
                p=parse(c['faces'])
                if p:
                    c['faces']=re.sub(r'^[A-Z]{3}\d+', '%s%d'%(p[0],p[1]-off), c['faces'])
            lo['numbering_offset_applied']=off
            stats['offset recovered']+=1; fixed+=1; shifts[off]+=1
        else:
            stats['no consistent offset']+=1
        lo['numbering_confidence']=round(conf,2)
        out.write(json.dumps(e)+'\n')
    out.close()
    print('rows: %s' % format(len(ev),','))
    for k,v in stats.most_common(): print('  %-24s %6s (%.0f%%)'%(k,format(v,','),100.0*v/len(ev)))
    print('\nnon-zero offsets recovered: %s' % format(fixed,','))
    for k,v in shifts.most_common(6): print('   shift %+6d : %s pairs'%(k,format(v,',')))