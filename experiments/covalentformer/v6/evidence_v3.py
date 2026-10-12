#!/usr/bin/env python3
"""Evidence v3. Two halves, deliberately separated.

MODEL_INPUT   computable from hit + pocket ALONE, because that is all inference has.
LABEL_ONLY    adds the lead's own crystal contacts. The labeller needs them to explain how the LEAD
              sits in the pocket differently from the hit; the model must never see them or it
              learns to depend on information that will not exist at inference.

New in v3, both from the hit structure:
  * free volume along the exit vector from every atom -- what makes a pure ADDITION describable
  * H-bond geometry via the heavy-atom angle, so a 2.9 A N...O can be told from a clash
"""
import gzip, json, math, os, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
RDLogger.DisableLog('rdApp.*')
D='/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror'
STD=set('ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL'.split())
CCD=json.load(open(D+'/ccd_atoms.json'))
BO={'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}
POLAR={'N','O','S'}
METALS={'ZN','MG','MN','FE','CU','NI','CO','CA','CD','K','NA','HG','PT','RU','V','W','MO'}
# Only these elements coordinate a metal. Carbon does not: in 108 rows a ligand CARBON sitting
# 2.5-2.7 A from a manganese was reported as "METAL COORDINATION", which is geometrically
# inevitable for a carbon bonded to a coordinating oxygen and chemically impossible as a claim.
COORDINATING={'N','O','S','P','SE','CL','BR','F','AS'}
MIN_FRAG=3; MAX_MCS_ROUNDS=6


def conserved_multi(A, B, min_frag=MIN_FRAG, timeout=5):
    """Conserved hit atoms, lead atoms, and the hit->lead atom map, by multi-fragment MCS.

    rdFMCS.FindMCS returns a single CONNECTED substructure, so a molecule whose conserved parts are
    separated by the edit gets all but the largest conserved region reported as changed. Iterating
    the MCS over the unmatched remainder recovers the other regions. Fragments under min_frag atoms
    are rejected as coincidental element matches rather than conserved groups.
    """
    keepA, keepB, amap = set(), set(), {}
    for _ in range(MAX_MCS_ROUNDS):
        a_rem = [i for i in range(A.GetNumAtoms()) if i not in keepA]
        b_rem = [i for i in range(B.GetNumAtoms()) if i not in keepB]
        if len(a_rem) < min_frag or len(b_rem) < min_frag:
            break
        try:
            sa = Chem.RWMol(A)
            for i in sorted(set(range(A.GetNumAtoms())) - set(a_rem), reverse=True):
                sa.RemoveAtom(i)
            sb = Chem.RWMol(B)
            for i in sorted(set(range(B.GetNumAtoms())) - set(b_rem), reverse=True):
                sb.RemoveAtom(i)
            ma, mb = sa.GetMol(), sb.GetMol()
            Chem.SanitizeMol(ma, catchErrors=True)
            Chem.SanitizeMol(mb, catchErrors=True)
            mc = rdFMCS.FindMCS([ma, mb], timeout=timeout, completeRingsOnly=False,
                                ringMatchesRingOnly=True)
            if mc.canceled or not mc.smartsString or mc.numAtoms < min_frag:
                break
            q = Chem.MolFromSmarts(mc.smartsString)
            ha, hb = ma.GetSubstructMatch(q), mb.GetSubstructMatch(q)
            if not ha or not hb or len(ha) != len(hb):
                break
            for x, y in zip(ha, hb):
                if x < len(a_rem) and y < len(b_rem):
                    amap[a_rem[x]] = b_rem[y]
            keepA |= {a_rem[i] for i in ha if i < len(a_rem)}
            keepB |= {b_rem[i] for i in hb if i < len(b_rem)}
        except Exception:
            break
    return keepA, keepB, amap
# cofactors that sit IN the site and are contacted directly. Excluded as ANALOGUE candidates
# earlier -- a different decision from excluding them as part of the pocket.
COFACTOR_HET={'NAD','NAP','NDP','NAI','FAD','FMN','FDA','ATP','ADP','AMP','GTP','GDP','GMP','ANP',
              'ACP','AGS','GNP','SAM','SAH','COA','ACO','HEM','HEC','HEA','PLP','TPP','TDP','BTN',
              'FES','SF4','MGD','PQQ','B12','CLA','BCL','UDP','UTP','CTP','UMP','CMP','F43','COM'}
META=json.load(open(D+'/meta.json'))
LIG2PDB=collections.defaultdict(set)
for _p,_v in META.items():
    for _l in _v['lig']: LIG2PDB[_l].add(_p)

def fib(n=64):
    out,ga=[],math.pi*(3-math.sqrt(5))
    for i in range(n):
        y=1-2*i/float(n-1); r=math.sqrt(max(0.,1-y*y)); t=ga*i
        out.append((math.cos(t)*r,y,math.sin(t)*r))
    return np.array(out)
DIRS=fib(64)
# 64, not 14. A self-test measured the 14-direction probe disagreeing with 64 by a median
# of 3.8 A - it misses channels entirely. Full resolution is affordable because the room
# measurement now runs only at the handful of site atoms, not all ~33.
PROBE=fib(64)

def ccd_mol(code):
    d=CCD.get(code)
    if not d: return None,None
    names=[a[0] for a in d['atoms']]; idx={n:i for i,n in enumerate(names)}
    m=Chem.RWMol()
    for n,el in d['atoms']:
        try: m.AddAtom(Chem.Atom(el.capitalize() if len(el)>1 else el.upper()))
        except Exception: return None,None
    for a1,a2,vo in d['bonds']:
        if a1 in idx and a2 in idx:
            try: m.AddBond(idx[a1],idx[a2],BO.get(vo,Chem.BondType.SINGLE))
            except Exception: pass
    mm=m.GetMol()
    for alt in (False,True):
        try:
            Chem.SanitizeMol(mm) if not alt else Chem.SanitizeMol(mm, Chem.SANITIZE_ALL ^ Chem.SANITIZE_PROPERTIES ^ Chem.SANITIZE_KEKULIZE)
            return mm,names
        except Exception: continue
    return None,None

def read_cx(pid, lig3):
    f=os.path.join(D,'cif','%s.cif.gz'%pid)
    if not os.path.exists(f): return None,None
    P,L,hdr,idx=[],[],[],None
    try:
        with gzip.open(f,'rt',errors='ignore') as fh:
            for ln in fh:
                if ln.startswith('_atom_site.'): hdr.append(ln.strip().split('.',1)[1]); idx=None; continue
                if hdr and idx is None:
                    try: idx={k:hdr.index(k) for k in ('type_symbol','label_atom_id','label_comp_id','auth_seq_id','auth_asym_id','Cartn_x','Cartn_y','Cartn_z')}
                    except ValueError: return None,None
                if not ln.startswith(('ATOM','HETATM')):
                    if P or L: break
                    continue
                p=ln.split()
                if idx is None or len(p)<=max(idx.values()): continue
                if p[idx['type_symbol']] in ('H','D'): continue
                comp=p[idx['label_comp_id']]
                try: xyz=[float(p[idx['Cartn_x']]),float(p[idx['Cartn_y']]),float(p[idx['Cartn_z']])]
                except ValueError: continue
                rec={'el':p[idx['type_symbol']],'name':p[idx['label_atom_id']].strip('"'),'res':comp,
                     'rn':p[idx['auth_seq_id']],'ch':p[idx['auth_asym_id']],'xyz':xyz}
                if comp in METALS or comp in COFACTOR_HET:
                    P.append({'el':p[idx['type_symbol']],'xyz':xyz,'res':comp,
                              'rn':p[idx['auth_seq_id']],'ch':p[idx['auth_asym_id']],
                              'name':p[idx['label_atom_id']].strip('"'),
                              'hetero':True})
                elif comp==lig3: L.append(rec)
                elif comp in STD: P.append(rec)
    except Exception: return None,None
    if len(L)<5 or len(P)<20: return None,None
    by=collections.defaultdict(list)
    for a in L: by[(a['ch'],a['rn'])].append(a)
    return P, max(by.values(),key=len)

def vector_space(p, Px, Lx):
    """Room to grow INSIDE the pocket, and the SHAPE of that room.

    Depth alone cannot tell a flat slot (wants an aryl) from a round one (wants a tBu), which is one
    of the commonest real decisions. The rays we already cast give that for free: each ray's enclosed
    depth is a point, and the principal extents of the resulting point cloud ARE the slot shape. A
    grid version of this timed out; this costs the same as the depth measurement alone.

    A ray counts only while its tip stays as enclosed as the ligand itself (>=35 protein heavy atoms
    within 8 A; ligand atoms sit at a median of 36, a tip at 8 A at a median of 9).
    """
    far = Lx[np.linalg.norm(Lx - p, axis=1) > 1.9]
    depths=np.zeros(len(PROBE)); escape=0.0
    for i,v in enumerate(PROBE):
        d=0.0; esc=0.0
        for t in np.arange(2.6,9.01,0.6):
            q=p+v*t
            if (np.linalg.norm(Px-q,axis=1)<2.9).any(): break
            if len(far) and (np.linalg.norm(far-q,axis=1)<2.4).any(): break
            esc=t
            if (np.linalg.norm(Px-q,axis=1)<=8.0).sum() >= 35: d=t
            else: break
        depths[i]=d; escape=max(escape,esc)
    best=int(np.argmax(depths))
    shape=None
    # Only the CONE around the best direction. A substituent grows one way; including rays pointing
    # the opposite way measured a disc across the atom and reported extents of 13-17 A, longer than
    # the 9 A ray ceiling, which is how nearly every site came out a "flat slot".
    if depths[best] > 0:
        cone = (PROBE @ PROBE[best] >= 0.34) & (depths > 0)      # within ~70 deg
        pts = PROBE[cone] * depths[cone][:, None]
    else:
        pts = np.zeros((0, 3))
    if len(pts)>=4:
        C=pts-pts.mean(0)   # extents are now the span of the growth cone, not of a disc
        try:
            _,_,vt=np.linalg.svd(C, full_matrices=False)
            ext=sorted((round(float(2*np.abs(C@vt[k]).max()),1) for k in range(3)), reverse=True)
            a,b,c=ext
            if a>=6.0 and c<=4.0:   lab='flat slot - suits a planar group (aryl/heteroaryl)'
            elif a>=6.0 and b<=4.5: lab='narrow channel - suits a linear chain'
            elif min(ext)>=5.0:     lab='roomy and round - suits a branched/bulky group (tBu, cyclopropyl)'
            else:                   lab='small niche - suits one small substituent (methyl, halogen, OH)'
            shape={'extents_A':ext,'reading':lab,'open_directions':int((depths>0).sum())}
        except Exception:
            shape=None
    return float(depths[best]), PROBE[best] if depths[best]>0 else None, escape, shape
def hbond_angle(don, acc, don_nb):
    """Heavy-atom approximation: angle at the donor between its neighbour and the acceptor.
    Near 180 deg = well-oriented; below ~90 deg the contact cannot be a hydrogen bond."""
    if don_nb is None: return None
    v1=don_nb-don; v2=acc-don
    c=v1.dot(v2)/(np.linalg.norm(v1)*np.linalg.norm(v2)+1e-9)
    return round(float(np.degrees(np.arccos(np.clip(c,-1,1)))),0)

def build(r, with_label_side=True):
    P,L=read_cx(r['pdb'], r['hit_code'])
    if not P: return None
    Px=np.array([a['xyz'] for a in P]); Lx=np.array([a['xyz'] for a in L])
    # Keep only protein within 15 A of the ligand. Rays reach at most 9 A from a ligand atom, so
    # nothing further can block them -- but ray-casting against the WHOLE protein (10,000+ atoms)
    # was costing 41M distance computations per pair and put the full run at 6 hours.
    _near = np.linalg.norm(Px[:, None, :] - Lx[None, :, :], axis=2).min(1) <= 15.0
    P = [P[i] for i in np.where(_near)[0]]
    Px = Px[_near]
    A=Chem.MolFromSmiles(r['hit']); B=Chem.MolFromSmiles(r['analogue'])
    # A single connected MCS mis-assigns conserved atoms. FindMCS returns ONE connected common
    # substructure, so when a hit and lead share two conserved regions separated by the part that
    # changed, only the larger region is recognised and the rest is reported as "changed".
    # 8ARY NQU->NG9 (a covalent pair) marked 8 atoms including a 4-chlorophenyl present in BOTH
    # molecules, while the ring that actually changed went unmarked -- so a labeller anchoring the
    # rationale on those atoms would describe replacing a group the chemists kept.
    # Multi-fragment matching takes the union of iteratively found common fragments: 8 -> 2 atoms
    # on that pair (the ether O and the cyclohexane carbon), which is the real edit.
    ka,_kb,_amap=conserved_multi(A,B)
    ca=[i for i in range(A.GetNumAtoms()) if i not in ka]
    cm,names=ccd_mol(r['hit_code']); mt=cm.GetSubstructMatch(A) if cm else None
    nm_of={i:names[mt[i]] for i in range(A.GetNumAtoms())} if mt and len(mt)==A.GetNumAtoms() else {}
    pos={a['name']:np.array(a['xyz']) for a in L}
    # EVERY ligand atom, not only the changed ones. Describing just the site of change threw away
    # the whole-molecule picture a chemist actually reasons from -- what each part touches, what is
    # buried, what is wasted on solvent -- and for a one-atom substitution left a single row of
    # evidence. The prompt is ~256 tokens, so there was no reason to truncate it.
    cand = list(range(A.GetNumAtoms()))
    changed_set = set(ca)
    # PURE ADDITIONS have no changed atom in the hit, so nothing was ever marked and the labeller
    # had to guess where the group goes. The attachment point is recoverable: find the analogue's
    # new atoms, see which core atom they bond to, and map that back through the MCS to the hit.
    # This used to run ONLY for pure additions (`if not ca`), but 53% of pairs both change atoms
    # AND add them. On those, the added group had no attachment point marked at all, so the
    # labeller could see that something was removed but not where the new group hangs.
    attach_set = set()
    try:
        b2a = {b: a for a, b in _amap.items()}
        for bi in set(range(B.GetNumAtoms())) - set(_kb):
            for nb in B.GetAtomWithIdx(bi).GetNeighbors():
                if nb.GetIdx() in b2a:
                    attach_set.add(b2a[nb.GetIdx()])
    except Exception:
        pass
    attach_set -= changed_set          # an atom that changed is reported as CHANGED, not as anchor
    rows=[]
    for i in cand:
        nm=nm_of.get(i)
        if nm is None or nm not in pos: continue
        p=pos[nm]; d=np.linalg.norm(Px-p,axis=1); j=int(np.argmin(d))
        site = (i in changed_set) or (i in attach_set)
        if site:
            depth,vec,escape,shape=vector_space(p,Px,Lx)
        else:
            depth,vec,escape,shape=0.0,None,0.0,None
        lining=[]
        if vec is not None:
            for t in np.arange(2.0,max(depth,2.0)+0.01,1.0):
                q=p+vec*t
                dd=np.linalg.norm(Px-q,axis=1); k=int(np.argmin(dd))
                if dd[k]<=6.0: lining.append('%s%s'%(P[k]['res'],P[k]['rn']))
        lining=list(dict.fromkeys(lining))[:4]
        el=A.GetAtomWithIdx(i).GetSymbol()
        ang=None
        # What makes a contact covalent here is the ELEMENT PAIR, not a tight distance window.
        # A covalent bond needs an electrophilic ligand atom -- C, or S/B/P for a disulfide,
        # boronate or phosphonate -- attacking Cys S / Ser-Thr O / Lys-His N. The old rule allowed
        # any of C/N/O/S on the ligand side, so a 2.15 A ligand-O to Glu-OE1 carboxylate contact
        # came out "covalent" (O-O covalent is 1.48 A: impossible). Excluding ligand O/N fixes that
        # without touching the distance range.
        #
        # The ceiling stays generous at 2.35 A on purpose. Covalent hit-to-lead is the point of this
        # corpus, and crystallographic refinement of an adduct is often loose: 16% of real bonds here
        # sit at 2.05-2.35 A, including the class-A beta-lactamase acyl-enzymes (C-O to SER70.OG at
        # 2.29 A) and 21 disulfides. A tighter window silently demoted all of them to "polar contact",
        # which is the same class of mistake as the filter that once deleted 737 covalent contacts.
        # Bonds longer than the ideal are kept and FLAGGED, never dropped.
        cov_max = 2.4 if (el=='S' and P[j]['el']=='S') else 2.35
        covalent = (1.1 <= d[j] <= cov_max
                    and el in ('C','S','B','P') and P[j]['el'] in ('S','O','N'))
        # covalent carries its own range (above); the metal gate must not cap it, which is what the
        # old shared `d[j] <= 2.3` prefix did.
        coordinates = (P[j]['res'] in METALS and el.upper() in COORDINATING)
        if covalent or (coordinates and d[j] <= 2.3):
            if P[j]['res'] in METALS:
                typ='METAL COORDINATION to %s' % P[j]['res']
            else:
                ideal={'C-S':1.81,'C-O':1.43,'C-N':1.47,'S-S':2.05,'B-O':1.40,
                       'P-O':1.60,'S-O':1.60,'S-N':1.70}.get(el+'-'+P[j]['el'])
                typ='COVALENT BOND to %s' % ('%s%s.%s'%(P[j]['res'],P[j]['rn'],P[j]['name']))
                # A reviewer citing a covalent bond deserves to know when the crystal models it
                # stretched well past the ideal length -- the bond is real, the geometry is soft.
                if ideal and d[j] > ideal + 0.35:
                    typ += ' (modelled long at %.2f A vs %.2f A ideal -- bond is real, geometry soft)' % (d[j], ideal)
        elif P[j]['res'] in METALS:
            # a carbon near a metal is NOT coordinating it -- it is bonded to whatever is
            typ=('METAL COORDINATION to %s' % P[j]['res']
                 if (coordinates and d[j]<=2.8) else 'near %s metal' % P[j]['res'])
        elif P[j].get('hetero'):
            typ='contact with cofactor %s' % P[j]['res']
        else:
            typ='hydrophobic contact' if el not in POLAR and P[j]['el'] not in POLAR else 'polar/apolar packing'
        # This block used to be able to overwrite a COVALENT/METAL label, because it is a separate
        # `if` rather than a branch of the chain above: a polar pair at 1.8 A came out as
        # "H-bond, good geometry". It now refuses to touch a bonded row, and refuses to call
        # anything below 2.4 A an H-bond -- heavy-atom donor/acceptor pairs sit at 2.6-3.2 A, so
        # a shorter one is a crystal-model question, not a design feature a reviewer should cite.
        if el in POLAR and P[j]['el'] in POLAR and d[j]<=3.6 and P[j]['res'] not in METALS \
                and not P[j].get('hetero') and not covalent:
            if d[j] < 2.4:
                typ='unusually short polar contact (%.2f A) -- treat the crystal model as uncertain' % d[j]
        if el in POLAR and P[j]['el'] in POLAR and 2.4 <= d[j] <= 3.6 and P[j]['res'] not in METALS \
                and not P[j].get('hetero') and not covalent:
            nb=[pos[nm_of[n.GetIdx()]] for n in A.GetAtomWithIdx(i).GetNeighbors()
                if n.GetIdx() in nm_of and nm_of[n.GetIdx()] in pos]
            ang=hbond_angle(p, np.array(P[j]['xyz']), nb[0] if nb else None)
            typ=('H-bond, good geometry' if ang and ang>=100 else
                 'polar contact, geometry poor' if ang else 'polar contact, geometry unknown')
        blocked=sum(1 for v in DIRS[::2] if (np.linalg.norm(Px-(p+v*3.5),axis=1)<2.6).any())
        openf=round(1-blocked/len(DIRS[::2]),2)
        # per ATOM, not per pair. The old expression tested `ca` (the list, truthy when non-empty)
        # rather than membership, so every atom of a substitution was labelled CHANGED and the site
        # of change was meaningless in 67% of rows.
        # The anchor role must say what actually happened. Computing attachment points for every
        # pair (not just pure additions) exposed this: on a SAME-SIZE substitution the anchors were
        # still labelled "new group added here", so a labeller would describe an added group that
        # does not exist. The wording now follows the heavy-atom delta.
        _grew = (r['heavy'][1]-r['heavy'][0]) > 0
        _anchor = ('ATTACHMENT POINT (new group added here)' if _grew else
                   'SUBSTITUTION BOUNDARY (this atom is kept; its neighbour differs in the lead)')
        rows.append({'atom':nm,'element':el,
                     'role':('CHANGED' if i in changed_set else
                             _anchor if i in attach_set
                             else 'unchanged'),
            'faces':'%s%s.%s'%(P[j]['res'],P[j]['rn'],P[j]['name']),'distance_A':round(float(d[j]),2),
            'interaction':typ,'angle_deg':ang,
            'open_fraction':openf,
            'reading':'solvent-facing' if openf>=0.45 else ('partly enclosed' if openf>=0.2 else 'buried'),
            'pocket_room_A':(round(depth,1) if site else None),
            'room_reading':(None if not site else
                            ('blocked by protein' if depth==0 and openf<0.45 else
                             'not in a pocket - open to solvent' if depth==0 else
                             'room available')),
            'pocket_room_lining':lining,
            '_exit':round(escape,1),
            'slot_shape':shape})
    rows.sort(key=lambda x: x['atom'])      # hit-only key: no site-of-change signal in order
    def _is_bond(r):
        return (1.55 <= r['distance_A'] <= 2.25 and
                ('METAL' in r['interaction'] or 'COVALENT' in r['interaction'] or
                 'cofactor' in r['interaction'] or r['element'] in ('N','O','S','C')))
    nclash=sum(1 for r in rows if r['distance_A'] < 2.2 and not _is_bond(r))
    if nclash > max(1, 0.15*len(rows)):
        return None                      # pervasive clashes: the complex is unusable
    rows=[r for r in rows if r['distance_A'] >= 2.2 or _is_bond(r)]
    if not rows:
        return None
    # The site of change must NOT be in MODEL_INPUT. At inference there is no lead, so nothing can
    # be marked; a model trained on marked rows would learn "edit where marked" and then find no
    # mark. The marking is label-side only, where it tells the reviewer which rows of the atom table
    # correspond to the edit they can already see in the two SMILES.
    site_atoms=[{'atom':a['atom'],'role':a['role'],'pocket_room_A':a['pocket_room_A'],
                 'room_reading':a.get('room_reading'),
                 'pocket_room_lining':a['pocket_room_lining']}
                for a in rows if a['role']!='unchanged']
    # WHITELIST, not a blacklist. This was a blacklist of four site-only fields, and the leak came
    # back a fourth time the moment new fields were added: `room_reading` and `_exit` are computed
    # only when `site` is true, so they were None/0.0 on every unchanged atom and non-null on the
    # edited ones -- `_exit` alone identified the edit site in 93% of rows. With a whitelist, any
    # field added later is excluded until someone deliberately decides it is inference-computable.
    MODEL_FIELDS=('atom','element','faces','distance_A','interaction','angle_deg',
                  'open_fraction','reading')
    model_rows=[{k:a[k] for k in MODEL_FIELDS if k in a} for a in rows]
    out={'pair':'%s %s->%s'%(r['pdb'],r['hit_code'],r['ana_code']),'target':r['target'],
         'MODEL_INPUT':{'hit_smiles':r['hit'],
            'note':'computable from the hit and its pocket alone - no lead information',
            'atoms':model_rows},
         'ANSWER':{'analogue_smiles':r['analogue']},
         'tanimoto':r['tanimoto'],'property_deltas':{'heavy':r['heavy'],'clogp':r['clogp'],
            'tpsa':r['tpsa'],'rotb':r['rotb'],'hbd':r['hbd'],'hba':r['hba']},
         'edit_type':'pure addition' if not ca else 'substitution/removal',
         'QA':{'clashes_removed':nclash,'atoms_reported':len(rows),
               'heavy_delta':r['heavy'][1]-r['heavy'][0]}}
    if with_label_side:
        lab=None
        for pid in sorted(LIG2PDB.get(r['ana_code'],()))[:1]:
            P2,L2=read_cx(pid, r['ana_code'])
            if P2:
                Px2=np.array([a['xyz'] for a in P2])
                cs=[]
                for a in L2:
                    dd=np.linalg.norm(Px2-np.array(a['xyz']),axis=1); k=int(np.argmin(dd))
                    if dd[k]<=4.2:
                        cs.append({'atom':a['name'],'el':a['el'],
                                   'faces':'%s%s.%s'%(P2[k]['res'],P2[k]['rn'],P2[k]['name']),
                                   'distance_A':round(float(dd[k]),2)})
                lab={'lead_pdb':pid,'lead_contacts':cs}
                break
        if lab:
            import re as _re
            _L={_re.match(r'([A-Z]{3}\d+)',c['faces']).group(1) for c in lab['lead_contacts']
                if _re.match(r'([A-Z]{3}\d+)',c['faces'])}
            _H={_re.match(r'([A-Z]{3}\d+)',a['faces']).group(1) for a in rows
                if _re.match(r'([A-Z]{3}\d+)',a['faces'])}
            _H|={r for a in rows for r in a['pocket_room_lining']}
            ov=len(_L&_H)/len(_L) if _L else 0.0
            lab['residue_numbering_overlap']=round(ov,2)
            lab['numbering_consistent']=bool(ov>=0.3)
        if lab is None: lab={}
        lab['site_of_change']=site_atoms
        lab['site_note']=('which rows of the atom table the edit touches, and the room available '
                          'there. Label-side only: the model is not told where to edit.')
        out['LABEL_ONLY']={'note':'the LEAD\'s own crystal. For writing the label only - NEVER shown '
                                  'to the model, which at inference has no lead structure.',
                           **(lab or {'lead_pdb':None,'lead_contacts':[]})}
    return out

if __name__=='__main__':
    rows=[json.loads(l) for l in open(D+'/v6_pairs_tc50.jsonl')]
    pure=[r for r in rows if r['changed_hit']==0][0]
    e=build(pure)
    print('=== PURE ADDITION example (the case that had no evidence before) ===')
    print(json.dumps({k:e[k] for k in ('pair','edit_type','MODEL_INPUT')}, indent=1)[:1500])
    print('\n=== LABEL_ONLY side (lead crystal, %s) ===' % e['LABEL_ONLY']['lead_pdb'])
    for c in e['LABEL_ONLY']['lead_contacts'][:6]: print('   %s' % c)
