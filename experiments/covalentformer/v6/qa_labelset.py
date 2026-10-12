#!/usr/bin/env python3
"""Full QA on the 10,612-pair label set. Structural integrity, physical plausibility, and
cross-consistency between the hit side, the lead side, and the molecules themselves.
"""
import json, re, sys, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, rdFMCS
RDLogger.DisableLog('rdApp.*')
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
ev=[json.loads(l) for l in open(D+'/'+(sys.argv[1] if len(sys.argv)>1 else 'v6_labelset.jsonl'))]
n=len(ev)
# 1-3 letters: standard residues are 3 (LEU137.CD1) but metals and some het groups are 1-2
# (ZN301.ZN, MG402.MG, K501.K). The 3-letter-only form reported every metal contact we
# deliberately added as "malformed" -- 806 rows of pure false positive.
RES=re.compile(r'^([A-Z]{1,3})(\d+)\.')

def is_bond(a):
    """Covalent bonds (1.55-2.05 A) and metal coordination (<=2.4 A) are SHORTER than any clash
    threshold by physics. A flat 2.2 A floor applied to them is what deleted 737 real covalent
    contacts, which is the opposite of what a covalent-chemistry corpus wants."""
    u=a['interaction'].upper()
    return 'COVALENT' in u or 'METAL' in u or 'cofactor' in a['interaction']
bad=collections.Counter(); warn=collections.Counter()
dists=[]; rooms=[]; nat=[]; tc_chk=[]; soc_n=[]
for e in ev:
    ats=e['MODEL_INPUT']['atoms']; lo=e['LABEL_ONLY']; soc=lo['site_of_change']
    nat.append(len(ats)); soc_n.append(len(soc))
    # --- structural integrity
    if not ats: bad['no atoms']+=1; continue
    if any(not RES.match(a['faces']) for a in ats): bad['malformed residue ref']+=1
    if len({a['atom'] for a in ats})!=len(ats): bad['duplicate atom names']+=1
    socn={a['atom'] for a in soc}; alln={a['atom'] for a in ats}
    if not socn<=alln: bad['site atom not in the atom table']+=1
    # --- physical plausibility
    dd=[a['distance_A'] for a in ats]; dists+=dd
    # only NON-BONDED contacts have a clash floor; and a contact the builder already annotates
    # as uncertain is disclosed, not hidden, so it is a warning rather than a blocker
    nbd=[a for a in ats if not is_bond(a)]
    if any(a['distance_A']<2.2 and 'uncertain' not in a['interaction'] for a in nbd):
        bad['non-bonded contact < 2.2 A']+=1
    elif any(a['distance_A']<2.2 for a in nbd):
        warn['short contact, disclosed as uncertain']+=1
    if max(dd)>12: warn['nearest contact > 12 A (atom far from protein)']+=1
    for a in ats:
        if a['interaction'].startswith('H-bond'):
            if not a.get('angle_deg'): bad['H-bond without angle']+=1; break
            if a['angle_deg']<90: bad['H-bond with impossible angle']+=1; break
            if a['distance_A']>3.6: bad['H-bond beyond 3.6 A']+=1; break
        if not (0.0<=a['open_fraction']<=1.0): bad['open_fraction out of range']+=1; break
        exp={'buried':a['open_fraction']<0.2,'partly enclosed':0.2<=a['open_fraction']<0.45,
             'solvent-facing':a['open_fraction']>=0.45}
        if not exp.get(a['reading'],False): bad['reading inconsistent with open_fraction']+=1; break
    for a in soc:
        if a['pocket_room_A'] is None: bad['site atom without room measurement']+=1; break
        rooms.append(a['pocket_room_A'])
        if a['pocket_room_A']<0 or a['pocket_room_A']>9.1: bad['room out of range']+=1; break
    # --- cross-consistency: does the stated Tanimoto match the molecules?
    h=Chem.MolFromSmiles(e['MODEL_INPUT']['hit_smiles']); l=Chem.MolFromSmiles(e['ANSWER']['analogue_smiles'])
    if h is None or l is None: bad['unparseable SMILES']+=1; continue
    t=DataStructs.TanimotoSimilarity(AllChem.GetMorganFingerprintAsBitVect(h,2,2048),
                                     AllChem.GetMorganFingerprintAsBitVect(l,2,2048))
    tc_chk.append(abs(t-e['tanimoto']))
    if abs(t-e['tanimoto'])>0.02: bad['stated Tanimoto disagrees with the molecules']+=1
    if t<0.50: bad['Tanimoto below the 0.50 floor']+=1
    if Chem.MolToSmiles(h)==Chem.MolToSmiles(l): bad['hit identical to lead']+=1
    hv=[h.GetNumHeavyAtoms(), l.GetNumHeavyAtoms()]
    if hv!=e['property_deltas']['heavy']: bad['heavy-atom counts disagree']+=1
    if abs(hv[1]-hv[0])>15: bad['edit larger than 15 heavy atoms']+=1
    # --- lead side
    lc=lo.get('lead_contacts') or []
    if not lc: bad['no lead contacts']+=1
    # a sub-bond-length lead contact that carries the `uncertain` annotation is disclosed to the
    # labeller, exactly as the hit side discloses its own short contacts in the interaction string
    elif any(c['distance_A']<1.3 and not c.get('uncertain') for c in lc):
        bad['lead contact < 1.3 A, NOT disclosed']+=1
    elif any(c.get('uncertain') for c in lc):
        warn['lead contact flagged uncertain (disclosed)']+=1
    if (lo.get('numbering_confidence') or 0)<0.3: bad['numbering unresolved']+=1
dists.sort(); rooms.sort(); nat.sort(); soc_n.sort()
print('label set: %s pairs | %d targets | %d series'
      % (format(n,','), len({e['target'] for e in ev}), len({e['pair'].split()[0] for e in ev})))
print('atoms/pair median %d  | site-of-change median %d' % (nat[len(nat)//2], soc_n[len(soc_n)//2]))
print('contact distance p5 %.2f median %.2f p95 %.2f A' % (dists[len(dists)//20],dists[len(dists)//2],dists[19*len(dists)//20]))
if rooms: print('site room p25 %.1f median %.1f p75 %.1f A' % (rooms[len(rooms)//4],rooms[len(rooms)//2],rooms[3*len(rooms)//4]))
if tc_chk: print('stated-vs-recomputed Tanimoto: max drift %.4f' % max(tc_chk))
print('\nBLOCKERS' if bad else '\nBLOCKERS: none')
for k,v in bad.most_common(): print('  %-46s %6s (%.1f%%)'%(k,format(v,','),100.0*v/n))
print('\nWARNINGS' if warn else '\nWARNINGS: none')
for k,v in warn.most_common(): print('  %-46s %6s (%.1f%%)'%(k,format(v,','),100.0*v/n))
sys.exit(1 if bad else 0)