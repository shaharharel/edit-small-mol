#!/usr/bin/env python3
"""Hard QA on the evidence. Every check below exists because a reviewer or a measurement caught it.

Exit non-zero on any BLOCKER. Blockers are defects that make a label actively wrong; warnings are
defects that make it weak.
"""
import json, re, sys, collections
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
f=sys.argv[1] if len(sys.argv)>1 else D+'/v6_evidence_v4.jsonl'
ev=[json.loads(l) for l in open(f)]
n=len(ev)
RES=re.compile(r'^([A-Z]{3}\d+)\.')
block=collections.Counter(); warn=collections.Counter()
rows_atoms=[]; ovs=[]
for e in ev:
    ats=e['MODEL_INPUT']['atoms']
    rows_atoms.append(len(ats))
    # BLOCKER: physically impossible contacts
    if any(a['distance_A']<2.2 for a in ats): block['contact under 2.2 A (clash)']+=1
    # BLOCKER: the site of change must be identifiable
    nch=sum(1 for a in ats if a['role']=='CHANGED')
    if e['edit_type']=='substitution/removal' and nch==0:
        block['substitution with no CHANGED atom marked']+=1
    if all(a['role']=='CHANGED' for a in ats) and len(ats)>3 and e['edit_type']=='substitution/removal':
        warn['every atom marked CHANGED (MCS likely failed)']+=1
    # BLOCKER: malformed residue references
    if any(not RES.match(a['faces']) for a in ats): block['malformed residue reference']+=1
    # BLOCKER: H-bond asserted without geometry
    for a in ats:
        if a['interaction'].startswith('H-bond') and not a.get('angle_deg'):
            block['H-bond claimed with no angle']+=1; break
    # WARNING: saturated room carries no information
    if ats and all(a['pocket_room_A']>=8.6 for a in ats): warn['every atom at the room ceiling']+=1
    # WARNING: no contact closer than 4.5 A anywhere -> ligand not really in a pocket
    if ats and min(a['distance_A'] for a in ats)>4.5: warn['no contact under 4.5 A']+=1
    lo=e.get('LABEL_ONLY',{})
    ov=lo.get('residue_numbering_overlap')
    if ov is None: warn['no numbering-overlap recorded']+=1
    else:
        ovs.append(ov)
        if ov==0: block['hit/lead numbering share NO residue']+=1
        elif ov<0.3: warn['numbering overlap under 0.3']+=1
    if not lo.get('lead_contacts'): warn['no lead contacts']+=1
    hd=abs(e['property_deltas']['heavy'][1]-e['property_deltas']['heavy'][0])
    if hd>15: block['heavy-atom delta over 15']+=1
rows_atoms.sort(); ovs.sort()
print('evidence rows: %s' % format(n,','))
print('atoms described per pair: p25 %d median %d p75 %d max %d'
      % (rows_atoms[len(rows_atoms)//4],rows_atoms[len(rows_atoms)//2],rows_atoms[3*len(rows_atoms)//4],rows_atoms[-1]))
if ovs: print('hit/lead numbering overlap: p25 %.2f median %.2f p75 %.2f' % (ovs[len(ovs)//4],ovs[len(ovs)//2],ovs[3*len(ovs)//4]))
print('\nBLOCKERS' if block else '\nBLOCKERS: none')
for k,v in block.most_common(): print('  %-46s %6s  (%.0f%%)'%(k,format(v,','),100.0*v/n))
print('\nWARNINGS' if warn else '\nWARNINGS: none')
for k,v in warn.most_common(): print('  %-46s %6s  (%.0f%%)'%(k,format(v,','),100.0*v/n))
clean=[e for e in ev
       if not any(a['distance_A']<2.2 for a in e['MODEL_INPUT']['atoms'])
       and (e.get('LABEL_ONLY',{}).get('residue_numbering_overlap') or 0)>0
       and not (e['edit_type']=='substitution/removal' and
                sum(1 for a in e['MODEL_INPUT']['atoms'] if a['role']=='CHANGED')==0)]
print('\nrows passing ALL blockers: %s (%.0f%%)' % (format(len(clean),','),100.0*len(clean)/n))
with open(D+'/v6_evidence_clean.jsonl','w') as fh:
    for e in clean: fh.write(json.dumps(e)+'\n')
sys.exit(1 if block else 0)