#!/usr/bin/env python3
"""Evidence v4. Two halves, deliberately separated.

MODEL_INPUT   computable from hit + pocket ALONE, because that is all inference has.
LABEL_ONLY    adds the lead's own crystal contacts. The labeller needs them to explain how the LEAD
              sits in the pocket differently from the hit; the model must never see them or it
              learns to depend on information that will not exist at inference.

v3 is kept unmodified as the provenance anchor: it reproduces the shipped v6 corpus byte for byte,
and that check is the only evidence the recovered pipeline is the one that built the data.

Fixed in v4, each from a measurement recorded in README.md:

 1. CONSERVED-ATOM ASSIGNMENT.  rdFMCS sets `canceled` when it stops searching early but still
    returns its best match, and v3 discarded the result on that flag -- so a complete 53-of-55-atom
    MCS was thrown away and all 55 atoms came out CHANGED. 501 rows had EVERY atom marked changed
    (corpus median 0.11); another 141 pure additions got no attachment point, because the anchor is
    derived from the same atom map. One clause; ~650 rows.
 2. REPORTING ASYMMETRY.  The hit side reported the nearest protein atom at any distance (to 11.1 A)
    while the lead side truncated at 4.2 A and dropped the atom entirely -- 76% of pairs silently
    omitted at least one lead atom, so "the lead lost contact X" could be a cutoff artifact. Both
    sides now use one cutoff and emit an explicit no-contact row instead of dropping an atom.
 3. NEAREST-ATOM-ONLY HID HYDROGEN BONDS.  A polar ligand atom whose nearest neighbour is a carbon
    had its polar partner reported nowhere: 9IXL showed zero H-bond rows for a buried carboxylate.
    Each polar atom now also carries its nearest polar partner. Two fields on polar rows only --
    cheaper than listing every neighbour within 4 A, which is the other way to fix it.
 4. FALSE COVALENT CALLS.  Typing keyed on the element pair with a flat 2.35 A ceiling, so 57 C...N
    and 17 C...O contacts past 2.20 A were called covalent bonds. The ceiling is now per element
    pair, and the stretched-but-real class (beta-lactamase acyl-enzymes at ~2.29 A) gets its own
    `probable covalent` type rather than being promoted or deleted.
 5. BONDED NEIGHBOURS SOLD AS CONTACTS.  In a covalent pair the atoms 1-3 and 1-4 from the new bond
    have the warhead's own residue as nearest neighbour, so `C1 C CYS38.SG 2.70 polar/apolar
    packing` described the covalent link a second time. Those rows now report their nearest
    INDEPENDENT contact and say why.
 6. SHORT-CONTACT DISCLOSURE.  v3 compared against ideal BOND length, so it could only catch near
    superposition: 3,521 sub-vdW contacts, 314 disclosed. Now tested against the vdW sum.
 7. CHAIN.  `faces` drops it, so two chains of a heterodimer are indistinguishable. Added as its own
    field rather than mangling the residue token every downstream regex depends on.
 8. ORDERED WATERS and LINING COMPOSITION, for every atom (not only site atoms -- that is how the
    `_exit` leak happened). Both come from the hit's own crystal, so both are legitimately
    MODEL_INPUT and available at inference.
 9. LEAD ATOM NAMES.  The lead's contacts used the lead's own CCD namespace, so `C19` on the two
    sides need not be the same atom and no before/after ledger was possible. Lead contacts are now
    partitioned: conserved atoms are reported under the HIT's atom name, new atoms are marked NEW.
"""
import gzip, json, math, os, re, collections
import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import rdFMCS
try:
    from rdkit.Chem import rdRascalMCES as _RASCAL
except ImportError:
    _RASCAL = None
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
CONTACT_CUTOFF=4.0      # one cutoff, both sides. See fix 2 in the module docstring.
# Ideal single-bond lengths. v4 uses these as the covalent CEILING per element pair, not only as
# annotation: a flat 2.35 A ceiling let C...N at 2.20 A (ideal 1.47) be called a covalent bond.
IDEAL_BOND={'C-S':1.81,'C-O':1.43,'C-N':1.47,'S-S':2.05,'B-O':1.40,'P-O':1.60,'S-O':1.60,
            'S-N':1.70,'C-SE':1.98,'SE-S':2.19,'P-S':2.10,'B-N':1.58,'P-N':1.65}
COV_TOL=0.35            # within this of ideal -> a bond a reviewer may call covalent
COV_SOFT=0.90           # beyond COV_TOL but within this -> "probable", disclosed, never cited bare
# vdW radii, for the short-contact test. v3 compared against ideal BOND length, which is ~1 A
# shorter than the vdW sum, so only near-superposition could ever trip it.
VDW={'H':1.10,'C':1.70,'N':1.55,'O':1.52,'F':1.47,'P':1.80,'S':1.80,'CL':1.75,'BR':1.85,'I':1.98,
     'SE':1.90,'B':1.92,'SI':2.10,'AS':1.85,
     'FE':2.04,'CO':2.00,'ZN':2.01,'MG':1.73,'MN':2.05,'CU':1.96,'NI':1.97,'CA':2.31,
     'NA':2.27,'K':2.75,'CD':2.18,'HG':2.23,'PT':2.13,'RU':2.13,'V':2.07,'W':2.10,'MO':2.10}
VDW_FRAC=0.80           # d < 0.80 * (r1+r2) and not a bond -> disclose as uncertain
# A FRACTION scales with the radii, so a big atom pair buys a bigger free pass: a Br/O contact
# at 2.72 A is 0.65 A inside its 3.37 A sum and escaped disclosure by 0.02 A, while a C/C pair
# with the same 0.65 A overlap is flagged. Disclose on EITHER test -- a fractional cut for
# small atoms, an absolute overlap for everything -- so the threshold stops behaving as a cliff.
VDW_OVERLAP=0.45        # d < (r1+r2) - 0.45 A and not a bond -> disclose, whatever the radii
# Protein BACKBONE atoms. Not nucleophiles for a ligand adduct, so a short contact to one is a
# model error, never a covalent bond. Side-chain nucleophiles (SG, NZ, OG, OH, ND/NE, SD) are the
# ones real warheads react with and are deliberately absent from this set.
BACKBONE={'N','O','C','CA','OXT'}
# A coordination bond that is too SHORT to exist is a refinement artifact, but the vdW test above
# never reached the METAL COORDINATION branch (0 of 3,079 rows), so Mn-O at 1.83 A read as clean
# while a carbon 2.66 A away was disowned. Shortest credible metal-ligand distances, by metal.
MIN_COORD={'ZN':1.80,'MG':1.95,'MN':2.05,'FE':1.90,'CU':1.85,'NI':1.90,'CO':1.90,'CA':2.20,
           'CD':2.15,'K':2.60,'NA':2.20,'HG':2.10,'PT':1.95,'RU':1.95,'V':1.85,'W':1.90,'MO':1.90}
CHARGED={'ARG':1,'LYS':1,'HIS':1,'ASP':-1,'GLU':-1}
# A hydrogen bond needs a DONOR and an ACCEPTOR. v4 checked element and angle only, so a ligand
# carbonyl against a carboxylate (acceptor vs acceptor) and a ligand N-H against a backbone amide
# N-H or a lysine ammonium (donor vs donor) both came out "H-bond, good geometry": 1,586 of 17,135
# rows, 9.3% of the corpus, chemically impossible at a perfectly plausible distance and angle. The
# 6GQO label repeated one verbatim -- an amide N-H reported as hydrogen bonding LYS868.NZ.
PROT_DONOR={('LYS','NZ'),('ARG','NE'),('ARG','NH1'),('ARG','NH2'),('TRP','NE1'),('ASN','ND2'),
            ('GLN','NE2'),('HIS','ND1'),('HIS','NE2'),('SER','OG'),('THR','OG1'),('TYR','OH'),
            ('CYS','SG')}
PROT_ACCEPTOR={('ASP','OD1'),('ASP','OD2'),('GLU','OE1'),('GLU','OE2'),('ASN','OD1'),
               ('GLN','OE1'),('SER','OG'),('THR','OG1'),('TYR','OH'),('HIS','ND1'),
               ('HIS','NE2'),('MET','SD')}


def prot_hb_role(res, name):
    """'donor' / 'acceptor' / 'both' / None for a protein atom's hydrogen-bonding capability."""
    if name == 'N':
        return 'donor'                       # backbone amide N-H
    if name in ('O', 'OXT'):
        return 'acceptor'                    # backbone carbonyl / terminal carboxylate
    d = (res, name) in PROT_DONOR
    a = (res, name) in PROT_ACCEPTOR
    return 'both' if d and a else 'donor' if d else 'acceptor' if a else None
# Salt bridges, on the medicinal-chemistry convention: CHARGE IDENTITY + DISTANCE, no angle test.
# Unlike a hydrogen bond, where geometry decides, a chemist calls a carboxylate against an
# arginine within ~4 A a salt bridge. Our heavy-atom angle is computed without placed hydrogens
# and so systematically understates the real geometry, which is why these were landing as
# `polar contact, geometry poor` and were undescribable under rule 2.
PROT_IONIC={('ASP','OD1'):-1,('ASP','OD2'):-1,('GLU','OE1'):-1,('GLU','OE2'):-1,
            ('ARG','NE'):1,('ARG','NH1'):1,('ARG','NH2'):1,('LYS','NZ'):1,
            ('HIS','ND1'):1,('HIS','NE2'):1}
SALT_MAX=4.0
# Anionic ligand oxygens (carboxylate, phosphate/phosphonate, sulfonate/sulfate) and cationic
# ligand nitrogens (protonated amine, guanidine/amidine). Charged at physiological pH even when
# the deposited SMILES is drawn neutral, which is the common case.
ANIONIC_SMARTS='[OX1,OX2H1]-[CX3,PX4,SX4](=O)'
# A thioamide N was given +1 (N-C=S was not excluded) and then reported as a SALT BRIDGE to a
# carboxylate -- an ion pair that cannot exist. Worse, the salt-bridge branch returns BEFORE the
# H-bond branch, so the row also lost its angle and its correct `H-bond, good geometry` typing:
# the one true statement about that pair was reachable only through a false one.
# NOT excluded: amidine and guanidine. Those really are protonated at physiological pH -- an
# arginine mimetic's guanidinium is one of the commonest cationic groups in this corpus -- and a
# first pass at this fix excluded them, which would have deleted the salt bridges that matter most.
# `!$(N-a)` excludes the ANILINE class -- an amine bonded straight onto an aromatic ring. Its lone
# pair is conjugated into the ring, so aniline's pKa is ~4.6 and it is NOT protonated at
# physiological pH; neither is an adenine or 2-aminopyrimidine exocyclic NH2, which are everywhere
# in kinase inhibitors. Without this, every one of them was given +1 and reported as a SALT BRIDGE
# to any nearby carboxylate -- 6TTT's adenine NH2 against ASP377.OD1 was the pair's most
# attractive and most wrong sentence. A BENZYLamine (N-C-c) is still basic and still scores +1.
CATIONIC_SMARTS=('[NX3;H0,H1,H2;!$(N-[CX3]=[OX1]);!$(N-[CX3]=[SX1]);!$(N-S(=O)=O);'
                 '!$(N(=O)=O);!$(N-a);!$(n)]')


def _rascal_opts():
    o = _RASCAL.RascalOptions()
    o.similarityThreshold = 0.10   # we have already imposed Tc>=0.50; do not reject on top of it
    o.allBestMCESs = False
    o.singleLargestFrag = False     # a disconnected match is the whole point -- see conserved_multi
    # NO wall-clock timeout. A timeout is what made v6's site_of_change depend on machine load, and
    # setting one here would have reintroduced exactly that. The search is bounded instead by
    # maxBondMatchPairs, which is deterministic. The default 1000 silently returns ZERO atoms for
    # 60-70 heavy-atom pairs (peptides, macrocycles) -- those then fell through to the rdFMCS
    # fallback and took minutes each. At 20000 the same four pairs resolve to 60-69 matched atoms
    # in under 0.01s.
    o.timeout = -1
    o.maxBondMatchPairs = 20000
    return o


def conserved_map(A, B, min_frag=MIN_FRAG):
    """Conserved hit atoms, lead atoms, and the hit->lead atom map. Deterministic.

    RASCAL maximum common EDGE subgraph replaces the iterated rdFMCS loop for three reasons:

      * determinism. rdFMCS is bounded by wall clock, so the same pair got a different
        site_of_change on different runs -- 9CCY is stored as "all 31 atoms changed" and recomputes
        today as all 31 conserved. A corpus whose labels depend on machine load cannot be QA'd.
      * it returns a FRAGMENTED match natively, which is exactly what the iterated loop was
        hand-rolling: a hit and lead sharing two regions either side of the edit.
      * it is ~100x faster (30 pairs: 0.0s vs 2.4s) and finds MORE conserved atoms than the
        iterated version on a third of pairs (36 vs 28, 29 vs 20).

    rdFMCS stays as the fallback for the pairs RASCAL declines, with the `canceled` flag ignored --
    see fix 1 in the module docstring.
    """
    if _RASCAL is not None:
        try:
            res = _RASCAL.FindMCES(A, B, _rascal_opts())
            if res:
                am = res[0].atomMatches()
                if am and len(am) >= min_frag:
                    amap = {int(a): int(b) for a, b in am}
                    # `moved`: atoms that ARE mapped but whose bonds are not. Defining "changed"
                    # purely as "absent from the atom map" is blind to every edit that keeps the
                    # atom multiset and moves a connection -- a sulfonamide reversal
                    # (S(=O)(=O)NCc1ccccc1 -> CNS(=O)(=O)c1ccccc1) or a meta->para halogen shift
                    # maps 100% of atoms, so 207 real SAR pairs reported no site of change at all.
                    # 6ZUC QPZ->S5N matches 24/24 atoms but only 24/26 bonds, and the two unmatched
                    # bonds are exactly the edit.
                    #
                    # These are reported SEPARATELY rather than folded into CHANGED. An atom at the
                    # edge of an ordinary substitution also has an unmatched bond, so folding them
                    # in relabelled 77% of rows and turned kept boundary atoms into deletions --
                    # the roles ATTACHMENT POINT and SUBSTITUTION BOUNDARY already describe those
                    # correctly. Only a moved atom that is neither changed nor a boundary gets the
                    # MOVED role in build().
                    mb = {int(x) for x, _y in res[0].bondMatches()}
                    # BOTH ends of the broken bond must be conserved atoms. "Any unmatched bond"
                    # also fires on the atom that merely lost a substituent, which is the boundary
                    # of an ordinary substitution and is already described by SUBSTITUTION
                    # BOUNDARY / ATTACHMENT POINT. Measured on 500 MOVED pairs, that looser test
                    # was wrong 48% of the time (7IJ3, 7U2I are plain substitutions), and it
                    # overstated relocation edits as 40% of the corpus instead of ~21%.
                    # drop implausible pairs before anything is derived from the map
                    _drop={i for i,j in amap.items()
                           if (A.GetAtomWithIdx(i).GetIsAromatic()
                               != B.GetAtomWithIdx(j).GetIsAromatic())
                           or (A.GetAtomWithIdx(i).IsInRing()
                               != B.GetAtomWithIdx(j).IsInRing())}
                    if _drop and len(amap)-len(_drop) >= min_frag:
                        amap={i:j for i,j in amap.items() if i not in _drop}
                    moved = set()
                    for b in A.GetBonds():
                        if b.GetIdx() in mb:
                            continue
                        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                        if i in amap and j in amap:
                            moved.add(i)
                            moved.add(j)
                    # SECOND OPINION. Measured on 200 pairs, RASCAL beats rdFMCS by >=3 atoms 24% of
                    # the time and loses by >=3 atoms 8.5% of the time -- neither method dominates.
                    # In that losing tail the cost is severe and silent: 6POB is one thiazole->oxazole
                    # S->O swap that rdFMCS maps 26/27, while RASCAL's 23/27 reports 4 CHANGED and 4
                    # NEW atoms, three of which exist in both molecules. Since every role, every
                    # conserved row and the whole site of change derive from this map, take whichever
                    # method maps MORE atoms. Deterministic: strictly larger wins, RASCAL keeps ties.
                    try:
                        ka2, kb2, am2 = conserved_multi(A, B, min_frag=min_frag, timeout=10)
                        if len(am2) > len(amap):
                            mv2 = set()
                            _pairs = set(am2.items())
                            for b in A.GetBonds():
                                i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
                                if i in am2 and j in am2:
                                    bb = B.GetBondBetweenAtoms(am2[i], am2[j])
                                    if bb is None:
                                        mv2.add(i)
                                        mv2.add(j)
                            return set(am2), set(am2.values()), am2, mv2
                    except Exception:
                        pass
                    return set(amap), set(amap.values()), amap, moved
        except Exception:
            pass
    # Short timeout on purpose: this path is now reached for almost nothing, and a long one here
    # costs minutes per pair while reintroducing wall-clock dependence. Rows that do take it are
    # caught by the two-build reproducibility check in gen_ev5.py.
    ka, kb, am = conserved_multi(A, B, min_frag=min_frag, timeout=5)
    return ka, kb, am, set()


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
            # `canceled` means FindMCS stopped searching early -- NOT that it failed. It still
            # returns the best common substructure found so far, and on these molecules that is
            # already 53 of 55 atoms. v3 treated the flag as failure and broke out with nothing
            # conserved, which is why 501 rows reported every atom as CHANGED; raising the timeout
            # changed nothing because the flag is set regardless. Judge the RESULT, not the flag.
            if not mc.smartsString or mc.numAtoms < min_frag:
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
    """-> protein/hetero atoms, the ligand copy, and ordered waters.

    v3 discarded HOH. Ordered waters are the single most-used explanatory handle in real analogue
    design ("the nitrile displaces the water bridging Thr211"), they are in every one of these
    files (8ARQ 306, 9IXL 356, 7G2Q 451), and because they come from the HIT's own crystal they are
    legitimately MODEL_INPUT rather than label-side.
    """
    f=os.path.join(D,'cif','%s.cif.gz'%pid)
    if not os.path.exists(f): return None,None,None
    P,L,W,hdr,idx=[],[],[],[],None
    try:
        with gzip.open(f,'rt',errors='ignore') as fh:
            for ln in fh:
                if ln.startswith('_atom_site.'): hdr.append(ln.strip().split('.',1)[1]); idx=None; continue
                if hdr and idx is None:
                    try: idx={k:hdr.index(k) for k in ('type_symbol','label_atom_id','label_comp_id','auth_seq_id','auth_asym_id','Cartn_x','Cartn_y','Cartn_z')}
                    except ValueError: return None,None,None
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
                if comp=='HOH':
                    W.append(xyz)
                elif comp in METALS or comp in COFACTOR_HET:
                    P.append({'el':p[idx['type_symbol']],'xyz':xyz,'res':comp,
                              'rn':p[idx['auth_seq_id']],'ch':p[idx['auth_asym_id']],
                              'name':p[idx['label_atom_id']].strip('"'),
                              'hetero':True})
                elif comp==lig3: L.append(rec)
                elif comp in STD: P.append(rec)
    except Exception: return None,None,None
    if len(L)<5 or len(P)<20: return None,None,None
    by=collections.defaultdict(list)
    for a in L: by[(a['ch'],a['rn'])].append(a)
    copy=max(by.values(),key=len)
    # One row per atom NAME. An atom refined in two alternate conformations appears twice with the
    # same label_atom_id, which double-counted 345 pairs' contacts and inflated
    # conserved_interactions ("13 here" where 7 were distinct). Keep the first conformer.
    seen=set(); uniq=[]
    for a in copy:
        if a['name'] in seen: continue
        seen.add(a['name']); uniq.append(a)
    return P, uniq, (np.array(W) if W else np.zeros((0,3)))

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

def parity_inverted(A, B, i, j, amap):
    """Is centre i of A physically inverted relative to centre j of B? -> True / False / None.

    None means "not comparable": the neighbour sets do not correspond, so the question is about an
    edit at the centre rather than an inversion of it.

    This compares CONFIGURATION, not CIP codes. A CIP label encodes substituent priorities, so
    deleting a hydroxyl elsewhere in the molecule can flip R<->S while the centre has not moved --
    9ETC's steroid reads CCW on both sides and was reported as inverted.
    """
    ai, bj = A.GetAtomWithIdx(i), B.GetAtomWithIdx(j)
    na = [n.GetIdx() for n in ai.GetNeighbors()]
    nb = [n.GetIdx() for n in bj.GetNeighbors()]
    if len(na) != len(nb) or len(na) < 3:
        return None
    mapped = [amap.get(x) for x in na]
    if any(m is None for m in mapped) or set(mapped) != set(nb):
        return None
    perm = [mapped.index(x) for x in nb]
    seen = [False] * len(perm)
    par = 0
    for k in range(len(perm)):
        if seen[k]:
            continue
        l, sz = k, 0
        while not seen[l]:
            seen[l] = True
            l = perm[l]
            sz += 1
        par ^= (sz - 1) & 1
    return bool((ai.GetChiralTag() != bj.GetChiralTag()) ^ bool(par))


def sulfur_hb_ok(mol, i):
    """May this ligand sulfur accept a hydrogen bond?

    Only with a lone pair to offer: a thiol, thioether, thione or aromatic sulfur. A fully
    substituted S(VI) -- sulfone, sulfonamide, sulfonate -- cannot, yet 178 rows typed `H-bond,
    good geometry` on one. 8IW6's was the geometric shadow of an oxygen placement the pipeline
    already distrusted: the companion O row at 2.27 A was flagged uncertain while the S "H-bond"
    at 3.46 A passed clean, and rule 2 then licensed writing it.

    A SULFOXIDE is refused as well. It does have a lone pair, but in structural practice the
    hydrogen bond goes to its oxygen; being conservative is right where the output feeds labels.

    Module level, not nested, so test_charges.py exercises THIS function rather than a copy of it.
    """
    at = mol.GetAtomWithIdx(i)
    if at.GetSymbol() != 'S':
        return True
    heavy = sum(1 for nb in at.GetNeighbors() if nb.GetAtomicNum() > 1)
    oxo = sum(1 for nb in at.GetNeighbors()
              if nb.GetSymbol() == 'O'
              and mol.GetBondBetweenAtoms(i, nb.GetIdx()).GetBondTypeAsDouble() >= 2)
    return not (oxo >= 1 and heavy >= 3)


def cov_check(el, pel, dist, pname=None):
    """Is this a covalent bond? -> (is_bond, ideal_length, confidence).

    v3 keyed on the element pair with a flat 2.35 A ceiling for every pair, so a C...N at 2.20 A
    (ideal 1.47) and a C...O at 2.29 A (ideal 1.43) were reported as covalent bonds -- 57 and 17
    rows that no chemist would accept. The ceiling is now relative to the ideal length for THAT
    pair, and the stretched class gets its own `probable` label: the beta-lactamase acyl-enzymes
    really are covalent adducts whose crystals model the bond long, so deleting them is as wrong as
    promoting them. Only `confirmed` may be called a covalent bond in a label (REVIEWER.md rule 3).
    """
    if el not in ('C', 'S', 'B', 'P') or pel not in ('S', 'O', 'N', 'SE'):
        return False, None, None
    # The element pair alone cannot tell a cysteine thiol from a backbone carbonyl oxygen, so a
    # ligand aromatic carbon 1.48 A from ALA129.O was typed `COVALENT BOND` -- on a ligand with no
    # electrophile anywhere -- and rule 3 then positively licensed writing it. A protein BACKBONE
    # atom is not a nucleophile that forms a ligand adduct; a contact that short to one is a
    # mismodelled ligand. 56 rows across 46 pairs. Side-chain nucleophiles are unaffected.
    if pname is not None and pname.upper() in BACKBONE:
        return False, None, None
    ideal = IDEAL_BOND.get('%s-%s' % (el, pel)) or IDEAL_BOND.get('%s-%s' % (pel, el))
    if ideal is None or dist < ideal - 0.15:
        # below any real bond length for this element pair: a refinement artifact, not chemistry
        return False, ideal, None
    if dist <= ideal + COV_TOL:
        return True, ideal, 'confirmed'
    if dist <= ideal + COV_SOFT:
        return True, ideal, 'probable'
    return False, ideal, None


def ligand_charges(M):
    """-> {atom_idx: +1/-1} for the ligand's ionic atoms at physiological pH."""
    out = {}
    for a in M.GetAtoms():
        if a.GetFormalCharge():
            out[a.GetIdx()] = 1 if a.GetFormalCharge() > 0 else -1
    for smarts, q in ((ANIONIC_SMARTS, -1), (CATIONIC_SMARTS, 1)):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for m in M.GetSubstructMatches(patt):
            for idx in m:
                if M.GetAtomWithIdx(idx).GetSymbol() in ('O', 'N'):
                    out.setdefault(idx, q)
    # CHARGE-SEPARATED BUT NET-NEUTRAL groups. A nitro drawn as [N+](=O)[O-] carries formal charges
    # that the loop above reads literally, so its oxygen was treated as an anion and could pair with
    # an arginine -- a salt bridge that does not exist, since a nitro group is net neutral and not
    # basic or acidic. Same for an N-oxide and an azide. Strip the whole group rather than one atom.
    for smarts in ('[NX3+](=[OX1])[OX1-]', '[NX2-]=[NX2+]=[NX1-]', '[nX3+][OX1-]',
                   '[NX4+]([OX1-])', '[SX3+]([OX1-])'):
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        for m in M.GetSubstructMatches(patt):
            for idx in m:
                out.pop(idx, None)
    return out


def type_contact(el, pa, dist, p=None, nb_xyz=None, lig_q=0, lig_h=None, lig_hb_ok=True):
    """The typed interaction for ONE ligand atom against ONE protein atom.

    Factored out because v3 typed the hit side and left the lead side as bare distances, so the
    labeller was asked to compare a typed before-picture with an untyped after-picture. Both sides
    now call this.
    """
    pel = pa['el']
    ang = None
    if pa['res'] in METALS or pel.upper() in METALS:
        metal = pa['res'] if pa['res'] in METALS else pel.upper()
        where = ('' if pa['res'] in METALS else ' in %s%s' % (pa['res'], pa['rn']))
        if el.upper() in COORDINATING and dist <= 2.8:
            # The closeness check used to be unreachable from here: 0 of 3,079 METAL COORDINATION
            # rows ever carried it, so a 1.83 A Mn-O -- shorter than any real Mn-O (~2.1) -- read as
            # clean coordination while a carbon 2.66 A away was disowned. A coordination bond that
            # is too SHORT to exist is a refinement artifact like any other.
            lo = MIN_COORD.get(metal, 1.8)
            t = 'METAL COORDINATION to %s%s' % (metal, where)
            if dist < lo:
                t += (' -- unusually short (%.2f A vs %.2f A minimum for %s); treat the crystal '
                      'model as uncertain' % (dist, lo, metal))
            return t, None
        t = 'near %s metal%s' % (metal, where)
        sm, qm = VDW.get(el.upper()), VDW.get(metal)
        if sm and qm and dist < VDW_FRAC * (sm + qm):
            # The old text asserted this atom was "probably forced there by its neighbour that does
            # coordinate the metal" -- a causal claim this function cannot check, and which was
            # false where no atom in the molecule coordinated anything. State the geometry only.
            t += (' -- unusually close (%.2f A vs %.1f A vdW sum); treat the crystal model as '
                  'uncertain unless a neighbouring atom coordinates the metal' % (dist, sm + qm))
        return t, None
    is_cov, ideal, conf = cov_check(el, pel, dist, pa.get('name'))
    if is_cov:
        tok = '%s%s.%s' % (pa['res'], pa['rn'], pa['name'])
        if conf == 'confirmed':
            return 'COVALENT BOND to %s' % tok, None
        return ('probable covalent bond to %s (modelled at %.2f A vs %.2f A ideal -- disclosed as '
                'uncertain; NOT citable as a covalent bond)' % (tok, dist, ideal)), None
    # Salt bridge BEFORE the H-bond branch: charge identity and distance decide it, and the angle
    # test the H-bond branch would apply is the thing that was wrongly demoting these.
    pq = PROT_IONIC.get((pa['res'], pa['name']), 0)
    if lig_q and pq and lig_q * pq < 0 and dist <= SALT_MAX:
        return ('SALT BRIDGE to %s%s.%s (%s ligand atom against %s residue; charge pairing, so no '
                'angle criterion applies)'
                % (pa['res'], pa['rn'], pa['name'],
                   'anionic' if lig_q < 0 else 'cationic',
                   'cationic' if pq > 0 else 'anionic')), None
    if pa.get('hetero'):
        typ = 'contact with cofactor %s' % pa['res']
    elif el in POLAR and pel in POLAR and dist <= 3.6 and lig_hb_ok:
        if dist < 2.4:
            typ = ('unusually short polar contact (%.2f A) -- treat the crystal model as uncertain'
                   % dist)
        else:
            ang = hbond_angle(p, np.array(pa['xyz']), nb_xyz)
            # donor/acceptor compatibility, not just distance and angle
            prole = prot_hb_role(pa['res'], pa['name'])
            lrole = None if lig_h is None else ('both' if lig_h else 'acceptor')
            incompatible = (prole and lrole and
                            ((prole == 'acceptor' and lrole == 'acceptor') or
                             (prole == 'donor' and lrole == 'both' and el == 'N' and lig_h)))
            if incompatible:
                typ = ('polar contact, donor/acceptor INCOMPATIBLE (%s vs %s): not a hydrogen bond'
                       % (('ligand ' + lrole), (pa['res'] + ' ' + prole)))
            else:
                typ = ('H-bond, good geometry' if ang and ang >= 100 else
                       'polar contact, geometry poor' if ang else
                       'polar contact, geometry unknown')
    else:
        typ = ('hydrophobic contact' if el not in POLAR and pel not in POLAR
               else 'polar/apolar packing')
    # fix 6: disclosure against the vdW sum. v3 compared against ideal BOND length, ~1 A shorter,
    # so it could only catch near-superposition: 3,521 sub-vdW contacts and 314 disclosed.
    # NOT for a polar donor/acceptor pair: a hydrogen bond is by definition shorter than the vdW
    # sum (2.6-3.2 A against a 3.04 A O...O sum, and a strong one reaches 2.43), so applying this
    # test there flagged eight perfectly real short H-bonds as crystal-model errors. The polar
    # branch above carries its own, correct threshold at 2.4 A.
    polar_pair = (el in POLAR and pel in POLAR
                  and 'INCOMPATIBLE' not in typ)
    s, q = VDW.get(el.upper()), VDW.get(pel.upper())
    if (s and q and not polar_pair and 'uncertain' not in typ
            and (dist < VDW_FRAC * (s + q) or dist < (s + q) - VDW_OVERLAP)):
        typ += (' -- unusually close (%.2f A vs %.1f A vdW sum); treat the crystal model as '
                'uncertain' % (dist, s + q))
    return typ, ang


def contact_row(d, el, polar_mask, exclude=frozenset()):
    """Nearest protein atom, and -- separately -- nearest POLAR partner.

    v3 reported only the single nearest atom. When a polar ligand atom's nearest neighbour is a
    carbon, its hydrogen-bond partner appeared nowhere: 9IXL showed no H-bond rows at all for a
    buried carboxylate, and a labeller cannot invent what is not in the table. Listing every
    neighbour within 4 A is the other way to fix this; it multiplies the table by ~8, and the
    partner that matters for an H-bond is the polar one, so we add that instead of all of them.

    `d` is the precomputed distance vector and `polar_mask` a precomputed boolean over the same
    protein atoms -- both vectorised, because the first cut did this as a Python loop over every
    protein atom for every ligand atom and cost 4s/pair.

    `exclude` drops protein atom indices that are not independent contacts -- see the covalent
    through-bond filter in build().
    """
    if exclude:
        d = d.copy()
        d[list(exclude)] = np.inf
    j = int(np.argmin(d))
    if not np.isfinite(d[j]):
        return None, None, None, None
    pj = None
    if el in POLAR:
        cand = polar_mask & (d <= CONTACT_CUTOFF)
        if cand.any():
            masked = np.where(cand, d, np.inf)
            k = int(np.argmin(masked))
            if k != j:
                pj = k
    return j, float(d[j]), pj, (float(d[pj]) if pj is not None else None)


def build(r, with_label_side=True):
    P,L,WAT=read_cx(r['pdb'], r['input_code'])
    if not P: return None
    Px=np.array([a['xyz'] for a in P]); Lx=np.array([a['xyz'] for a in L])
    # Keep only protein within 15 A of the ligand. Rays reach at most 9 A from a ligand atom, so
    # nothing further can block them -- but ray-casting against the WHOLE protein (10,000+ atoms)
    # was costing 41M distance computations per pair and put the full run at 6 hours.
    _near = np.linalg.norm(Px[:, None, :] - Lx[None, :, :], axis=2).min(1) <= 15.0
    P = [P[i] for i in np.where(_near)[0]]
    Px = Px[_near]
    if len(Px) < 10:
        return None          # no protein within 15 A of the ligand: not a complex we can describe
    A=Chem.MolFromSmiles(r['input']); B=Chem.MolFromSmiles(r['analogue'])
    # A single connected MCS mis-assigns conserved atoms. FindMCS returns ONE connected common
    # substructure, so when a hit and lead share two conserved regions separated by the part that
    # changed, only the larger region is recognised and the rest is reported as "changed".
    # 8ARY NQU->NG9 (a covalent pair) marked 8 atoms including a 4-chlorophenyl present in BOTH
    # molecules, while the ring that actually changed went unmarked -- so a labeller anchoring the
    # rationale on those atoms would describe replacing a group the chemists kept.
    # Multi-fragment matching takes the union of iteratively found common fragments: 8 -> 2 atoms
    # on that pair (the ether O and the cyclohexane carbon), which is the real edit.
    ka,_kb,_amap,_moved=conserved_map(A,B)
    ca=[i for i in range(A.GetNumAtoms()) if i not in ka]
    cm,names=ccd_mol(r['input_code']); mt=cm.GetSubstructMatch(A) if cm else None
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
    # a mapped atom whose connectivity moved, and which is neither a deletion nor a boundary: this
    # is the whole of a positional-isomer edit (Cl meta->para), which was previously invisible.
    moved_set = (_moved or set()) - changed_set - attach_set
    # ---- fix 5 pre-pass: find the covalent links, then the ligand atoms 1-2/1-3 bonds away from
    # one. Those atoms are geometrically forced to have the warhead's own residue as their nearest
    # protein atom, so v3 reported the covalent link a second and third time as an independent
    # contact (`C1 C CYS38.SG 2.70 polar/apolar packing`). They now report their nearest contact
    # OUTSIDE that residue, which is the only contact of theirs that carries information.
    cov_res_of = {}
    for i in cand:
        nm = nm_of.get(i)
        if nm is None or nm not in pos:
            continue
        dd = np.linalg.norm(Px - pos[nm], axis=1)
        k = int(np.argmin(dd))
        if cov_check(A.GetAtomWithIdx(i).GetSymbol(), P[k]['el'], float(dd[k]),
                     P[k].get('name'))[2] == 'confirmed':
            cov_res_of[i] = (P[k]['res'], P[k]['rn'], P[k]['ch'])
    through_bond = {}
    for ci, key in cov_res_of.items():
        for nb in A.GetAtomWithIdx(ci).GetNeighbors():
            if nb.GetIdx() not in cov_res_of:
                through_bond.setdefault(nb.GetIdx(), key)
            for nb2 in nb.GetNeighbors():
                if nb2.GetIdx() != ci and nb2.GetIdx() not in cov_res_of:
                    through_bond.setdefault(nb2.GetIdx(), key)
    res_atoms = collections.defaultdict(set)
    for k, pa in enumerate(P):
        res_atoms[(pa['res'], pa['rn'], pa['ch'])].add(k)
    LIGQ = ligand_charges(A)
    LIGH = {i: A.GetAtomWithIdx(i).GetTotalNumHs() for i in range(A.GetNumAtoms())}
    LIGHB = {i: sulfur_hb_ok(A, i) for i in range(A.GetNumAtoms())}
    POLAR_MASK = np.array([pa['el'] in POLAR for pa in P])
    IS_CARBON = np.array([pa['el'] == 'C' for pa in P])
    RES_KEY = [(pa['res'], pa['rn'], pa['ch']) for pa in P]

    rows=[]
    for i in cand:
        nm=nm_of.get(i)
        if nm is None or nm not in pos: continue
        p=pos[nm]; d=np.linalg.norm(Px-p,axis=1); j=int(np.argmin(d))
        excl = res_atoms[through_bond[i]] if i in through_bond else frozenset()
        site = (i in changed_set) or (i in attach_set) or (i in moved_set)
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
        nb=[pos[nm_of[n.GetIdx()]] for n in A.GetAtomWithIdx(i).GetNeighbors()
            if n.GetIdx() in nm_of and nm_of[n.GetIdx()] in pos]
        nb0 = nb[0] if nb else None
        j, dj, pj, dpj = contact_row(d, el, POLAR_MASK, excl)
        if j is None:
            continue
        dj = round(dj, 2)      # round first, then branch, so stored value and branch agree
        if dj > CONTACT_CUTOFF:
            # Symmetry with the lead side, which this used to lack: v3 reported the nearest protein
            # atom at ANY distance with a contact type attached, so `polar/apolar packing` appeared
            # at 5.58 A and read as a real interaction. Both sides now state plainly when there is
            # nothing within the cutoff.
            typ, ang = ('no contact within %.1f A (nearest protein atom is %.2f A away)'
                        % (CONTACT_CUTOFF, dj)), None
        else:
            typ, ang = type_contact(el, P[j], dj, p, nb0, LIGQ.get(i, 0), LIGH.get(i),
                                    LIGHB.get(i, True))
        if i in through_bond:
            typ += (' [nearest INDEPENDENT contact; %s%s is excluded as the residue its covalent '
                    'link already describes]' % (through_bond[i][0], through_bond[i][1]))
        # fix 3: the nearest POLAR partner, reported alongside the nearest atom. Without it a polar
        # atom whose nearest neighbour is a carbon had its hydrogen-bond partner in no field at all.
        ptyp = pang = None
        if pj is not None:
            ptyp, pang = type_contact(el, P[pj], dpj, p, nb0, LIGQ.get(i, 0), LIGH.get(i))
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
                   'SUBSTITUTION BOUNDARY (this atom is kept; its neighbour differs in the analogue)')
        # fix 8: both for EVERY atom, never only for site atoms -- that asymmetry is exactly how
        # `_exit` leaked the edit location in 93% of rows. Both come from the hit's own crystal, so
        # both are legitimately available at inference.
        wn, wd = 0, None
        if len(WAT):
            dw = np.linalg.norm(WAT - p, axis=1)
            wn = int((dw <= 3.5).sum())
            if wn:
                wd = round(float(dw.min()), 2)
        m45 = d <= 4.5
        apol = int((m45 & IS_CARBON).sum())
        polr = int((m45 & POLAR_MASK).sum())
        chg = sum(CHARGED.get(k[0], 0) for k in {RES_KEY[x] for x in np.where(m45)[0]})
        rows.append({'atom':nm,'element':el,
                     'role':('CHANGED' if i in changed_set else
                             _anchor if i in attach_set else
                             'MOVED (this atom is kept but its attachment point differs in the '
                             'analogue)' if i in moved_set
                             else 'unchanged'),
            'faces':'%s%s.%s'%(P[j]['res'],P[j]['rn'],P[j]['name']),'distance_A':round(dj,2),
            'chain':P[j]['ch'],
            'interaction':typ,'angle_deg':ang,
            'polar_partner':('%s%s.%s'%(P[pj]['res'],P[pj]['rn'],P[pj]['name'])
                             if pj is not None else None),
            'polar_distance_A':(round(dpj,2) if dpj is not None else None),
            'polar_interaction':ptyp,'polar_angle_deg':pang,
            'waters_within_3.5A':wn,'nearest_water_A':wd,
            'lining':'%d apolar, %d polar, net charge %+d' % (apol, polr, chg),
            'open_fraction':openf,
            'reading':'solvent-facing' if openf>=0.45 else ('partly enclosed' if openf>=0.2 else 'buried'),
            'pocket_room_A':(round(depth,1) if site else None),
            'room_reading':(None if not site else
                            ('blocked by protein' if depth==0 and openf<0.45 else
                             'not in a pocket - open to solvent' if depth==0 else
                             '%.1f A of OPEN SOLVENT, not pocket room - this atom is '
                             'solvent-facing (open fraction %.2f), so a group grown here need not '
                             'touch the protein at all'%(depth,openf) if openf>=0.45 else
                             'room available')),
            'pocket_room_lining':lining,
            '_exit':round(escape,1),
            'slot_shape':shape})
    rows.sort(key=lambda x: x['atom'])      # hit-only key: no site-of-change signal in order
    def _is_bond(r):
        u=r['interaction'].upper()
        return ('COVALENT BOND' in u or 'PROBABLE COVALENT' in u
                or 'METAL COORDINATION' in u or 'cofactor' in r['interaction'])
    nclash=sum(1 for r in rows if r['distance_A'] < 2.2 and not _is_bond(r))
    if nclash > max(1, 0.15*len(rows)):
        return None                      # pervasive clashes: the complex is unusable
    # The dropped atoms were disclosed NOWHERE: `atoms_not_modelled` only covers atoms absent from
    # the coordinates, so a clash-dropped atom simply vanished from a table the brief promises
    # covers every atom of the input molecule (210 atoms across 3.1% of pairs). Worse, the row that
    # vanished is the one that would have warned the labeller the residue is modelled inside the
    # ligand -- 7I9I drops C6 for clashing with LYS12.NZ and then presents a clean, quotable salt
    # bridge to that same NZ. Keep the names and say why they are gone.
    _clashed=sorted({r['atom'] for r in rows if r['distance_A'] < 2.2 and not _is_bond(r)})
    _clash_res=sorted({(r.get('faces') or '?').split('.')[0]
                       for r in rows if r['distance_A'] < 2.2 and not _is_bond(r)})
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
    MODEL_FIELDS=('atom','element','faces','chain','distance_A','interaction','angle_deg',
                  'polar_partner','polar_distance_A','polar_interaction','polar_angle_deg',
                  'waters_within_3.5A','nearest_water_A','lining',
                  'open_fraction','reading')
    # Contamination is a property of the RESIDUE in this pair, not of one atom. A salt bridge and an
    # H-bond are exempt from the vdW test by design (a real H-bond IS sub-vdW), so a residue could be
    # disowned on five atoms and still serve a clean, quotable salt bridge on a sixth -- 9T23 models
    # LYS12.NZ inside the ligand and then offers `O1 SALT BRIDGE to LYS12.NZ 2.73 A` as sound.
    # THIS MUST RUN BEFORE model_rows IS COPIED: the first version ran 400 lines later and reached
    # 0 of 6,462 pairs, which is the same way the truncated `uncertain` column stayed invisible.
    _susp=collections.Counter()
    for _a in rows:
        if 'uncertain' in (_a['interaction'] or '').lower() and _a.get('faces'):
            _susp[_a['faces'].split('.')[0]] += 1
    _bad_res={k for k, v in _susp.items() if v >= 2}
    for _a in rows:
        _r=(_a.get('faces') or '').split('.')[0]
        if _r in _bad_res and 'uncertain' not in (_a['interaction'] or '').lower():
            _a['interaction'] += (
                ' [RESIDUE SUSPECT: %d other atoms of this ligand are modelled implausibly close to '
                '%s, so its position in this crystal is unreliable even though THIS contact is a '
                'normal distance. Treat any claim about %s with the same caution as an uncertain '
                'row.]' % (_susp[_r], _r, _r))
    model_rows=[{k:a[k] for k in MODEL_FIELDS if k in a} for a in rows]
    out={'pair':'%s %s->%s'%(r['pdb'],r['input_code'],r['analogue_code']),'target':r['target'],
         'MODEL_INPUT':{'input_smiles':r['input'],
            'note':'computable from the input molecule and its pocket alone - no analogue information',
            # Disclosed, not left to be inferred. ~15% of pairs have ligand atoms that the crystal
            # does not model (disordered side chains, a loose halogen), on BOTH sides -- median 1
            # atom. Without saying so, a labeller reading 23 rows for a 31-heavy-atom molecule
            # would conclude 8 atoms lost all their contacts, when they were never located at all.
            # That is the same false inference the 4.2 A cutoff used to invite on the lead side.
            'atoms_not_modelled':sorted({nm_of[i] for i in range(A.GetNumAtoms())
                                         if i in nm_of and nm_of[i] not in pos}),
            # separate from atoms_not_modelled: these atoms ARE in the coordinates, but sit closer
            # than 2.2 A to a protein atom without being bonded to it, so the model is locally
            # wrong. Named rather than silently dropped.
            'atoms_dropped_as_clashes':_clashed,
            'suspect_residues':sorted(_bad_res),
            'clash_note':('these ligand atoms are present in the crystal but overlap a protein '
                          'atom (<2.2 A, no bond), so their rows are omitted as unreliable. The '
                          'residue(s) they overlap are %s -- a residue that clashes with one ligand '
                          'atom is suspect for its OTHER contacts in this pair too, so treat every '
                          'row naming %s with caution.'
                          %(', '.join(_clash_res), ', '.join(_clash_res))) if _clashed else None,
            'atoms':model_rows},
         'ANSWER':{'analogue_smiles':r['analogue']},
         'tanimoto':r['tanimoto'],'property_deltas':{'heavy':r['heavy'],'clogp':r['clogp'],
            'tpsa':r['tpsa'],'rotb':r['rotb'],'hbd':r['hbd'],'hba':r['hba']},
         'edit_type':'pure addition' if not ca else 'substitution/removal',
         'QA':{'clashes_removed':nclash,'atoms_reported':len(rows),
               'heavy_delta':r['heavy'][1]-r['heavy'][0],
               # The commonest reason a pair has no site of change is NOT an MCS failure -- the MCS
               # identifies the edit correctly and the hit crystal simply does not model the edited
               # atom (a disordered halogen, say). 7EX8 I27->IA5 is a clean Cl->Br whose CL2 is
               # absent from the coordinates. Naming the reason lets the driver drop it as a data
               # limit rather than carry it as an empty site a labeller cannot use.
               'unmodelled_site_atoms':sorted({nm_of[i] for i in (changed_set | attach_set | moved_set)
                                               if i in nm_of and nm_of[i] not in pos})}}
    if with_label_side:
        lab=None
        # fix 2: the lead structure comes from the PAIR RECORD, chosen in build_pairs_v7 from the
        # same paper and verified to be the same protein. v3 took sorted(LIG2PDB[analogue_code])[0] --
        # the alphabetically first entry in the whole 43k-file mirror containing that ligand code,
        # with no requirement that it be the same protein, which paired two unrelated proteins in
        # 12% of rows while a same-series candidate existed.
        for pid in ([r['analogue_pdb']] if r.get('analogue_pdb')
                    else sorted(LIG2PDB.get(r['analogue_code'],()))[:1]):
            P2,L2,_W2=read_cx(pid, r['analogue_code'])
            if P2:
                Px2=np.array([a['xyz'] for a in P2])
                # the hit's name for each conserved lead atom, so the two sides can be compared
                bm,bnames=ccd_mol(r['analogue_code'])
                bmt=bm.GetSubstructMatch(B) if bm else None
                bnm={i:bnames[bmt[i]] for i in range(B.GetNumAtoms())} \
                    if bmt and len(bmt)==B.GetNumAtoms() else {}
                _moved_names={bnm[b] for ai,b in (_amap or {}).items()
                              if ai in moved_set and b in bnm}
                input_name_of={}
                for ai,bi in (_amap or {}).items():
                    if bi in bnm and ai in nm_of:
                        input_name_of[bnm[bi]]=nm_of[ai]
                cs=[]
                reported={x['atom'] for x in rows}
                _pos2={a['name']:np.array(a['xyz']) for a in L2}
                _b_of_name={v:k for k,v in bnm.items()}
                def _nb2(nm2):
                    bi=_b_of_name.get(nm2)
                    if bi is None: return None
                    for nb in B.GetAtomWithIdx(bi).GetNeighbors():
                        q2=_pos2.get(bnm.get(nb.GetIdx()))
                        if q2 is not None: return q2
                    return None
                _bq=ligand_charges(B)
                ANALOGUEH={bnm[bi]:B.GetAtomWithIdx(bi).GetTotalNumHs()
                           for bi in range(B.GetNumAtoms()) if bi in bnm}
                ANALOGUEQ={bnm[bi]:q for bi,q in _bq.items() if bi in bnm}
                ANALOGUEHB={bnm[bi]:sulfur_hb_ok(B,bi) for bi in range(B.GetNumAtoms()) if bi in bnm}
                # fix 5, FINALLY applied to this side too. The input table suppresses the atoms
                # 1-2/1-3 bonds from a covalent warhead -- 3,086 rows carry the `[nearest
                # INDEPENDENT contact]` note -- but the analogue table carried it ZERO times, so the
                # warhead's own residue was reported two and three times over as if independent.
                # 8ALV reads `C02 probable covalent bond to CYS38.SG (2.65 A)` beside C01's real
                # 1.82 A bond to the same SG, which invites "gains a second covalent contact".
                _res_atoms2=collections.defaultdict(set)
                for _k2,_pa2 in enumerate(P2):
                    _res_atoms2[(_pa2['res'],_pa2['rn'],_pa2['ch'])].add(_k2)
                _cov_res2={}
                for _a2 in L2:
                    _q2=np.array(_a2['xyz'])
                    _dd2=np.linalg.norm(Px2-_q2,axis=1); _k2=int(np.argmin(_dd2))
                    if cov_check(_a2['el'],P2[_k2]['el'],float(_dd2[_k2]),P2[_k2].get('name'))[2]=='confirmed':
                        _cov_res2[_a2['name']]=(P2[_k2]['res'],P2[_k2]['rn'],P2[_k2]['ch'])
                _through2={}
                for _cn,_key in _cov_res2.items():
                    _bi=_b_of_name.get(_cn)
                    if _bi is None: continue
                    for _nb in B.GetAtomWithIdx(_bi).GetNeighbors():
                        _n1=bnm.get(_nb.GetIdx())
                        if _n1 and _n1 not in _cov_res2:
                            _through2.setdefault(_n1,_key)
                        for _nb2a in _nb.GetNeighbors():
                            _n2=bnm.get(_nb2a.GetIdx())
                            if _n2 and _nb2a.GetIdx()!=_bi and _n2 not in _cov_res2:
                                _through2.setdefault(_n2,_key)
                for a in L2:
                    q=np.array(a['xyz'])
                    dd=np.linalg.norm(Px2-q,axis=1)
                    # exclude the warhead's own residue for the atoms bonded next to it, exactly as
                    # the input side does, so the nearest reported contact is an INDEPENDENT one
                    _ex2=_res_atoms2[_through2[a['name']]] if a['name'] in _through2 else frozenset()
                    if _ex2:
                        _dm=dd.copy()
                        for _ei in _ex2: _dm[_ei]=np.inf
                        k=int(np.argmin(_dm)) if np.isfinite(_dm).any() else int(np.argmin(dd))
                    else:
                        k=int(np.argmin(dd))
                    # round FIRST, then compare, so the stored distance and the branch taken cannot
                    # disagree: 4.0049 rounds to 4.00 and then read as inside a 4.0 cutoff it was
                    # excluded from (526 rows).
                    dk=round(float(dd[k]),2)
                    hn=input_name_of.get(a['name'])
                    # the hit's atom table omits atoms with no CCD name, atoms absent from the hit
                    # crystal, and rows the clash filter removed. Pointing at one of those is worse
                    # than pointing at nothing, so the bridge is only claimed when it resolves.
                    if hn is not None and hn not in reported:
                        hn=None
                    row={'atom':a['name'],'el':a['el'],
                         # fix 9: the lead's CCD namespace is not the hit's, so `C19` on the two
                         # sides need not be the same atom and no before/after ledger was possible.
                         'input_atom':hn,
                         # MOVED is its own status: the atom exists in both molecules but its
                         # bonds differ, so it is neither conserved (no valid before/after
                         # comparison) nor new. Collapsing it into either one is what let 6GQO
                         # compare two nitrogens across a reversed amide.
                         'status':('MOVED (present in both molecules but its bonds differ, so '
                                    'there is no valid before/after comparison)'
                                   if a['name'] in _moved_names
                                   else 'conserved' if a['name'] in input_name_of
                                   else 'NEW')}
                    # fix 2: one cutoff on both sides, and an explicit no-contact row instead of
                    # dropping the atom. v3 truncated the lead at 4.2 A while the hit side ran to
                    # 11.1 A, and 76% of pairs silently omitted at least one lead atom -- so "the
                    # lead lost contact X" could be an artifact of the cutoff.
                    if dk<=CONTACT_CUTOFF:
                        t2,g2=type_contact(a['el'],P2[k],dk,q,_nb2(a['name']),
                                           ANALOGUEQ.get(a['name'],0),
                                           ANALOGUEH.get(a['name']),
                                           ANALOGUEHB.get(a['name'],True))
                        if a['name'] in _through2:
                            t2+=(' [nearest INDEPENDENT contact; %s%s is excluded as the residue '
                                 'its covalent link already describes]'
                                 %(_through2[a['name']][0],_through2[a['name']][1]))
                        row.update({'faces':'%s%s.%s'%(P2[k]['res'],P2[k]['rn'],P2[k]['name']),
                                    'chain':P2[k]['ch'],'distance_A':dk,
                                    'interaction':t2,'angle_deg':g2})
                    else:
                        row.update({'faces':None,'chain':None,'distance_A':dk,
                                    'interaction':'no contact within %.1f A (nearest protein atom '
                                                  'is %.2f A away)'%(CONTACT_CUTOFF,dk),
                                    'angle_deg':None})
                    cs.append(row)
                # CONSERVED INTERACTIONS. An interaction present in BOTH crystals on the same
                # atom against the same residue -- the covalent anchor that is held while the edit
                # decorates the far end, which is the first thing a chemist says about an
                # optimisation. Rule 5 used to forbid saying it, because before the input_atom bridge
                # existed conservation could only be guessed at. It is now MEASURED on both sides,
                # so it is no longer an inference.
                #
                # LABEL_ONLY by construction: it requires the lead crystal. It is a labelling aid,
                # not an evidence field -- the model never sees it and must never be trained to
                # expect it.
                _hit_by={x['atom']:x for x in rows}
                cons=[]
                for c in cs:
                    hn=c.get('input_atom')
                    if not hn or not c.get('faces'):
                        continue
                    hrow=_hit_by.get(hn)
                    if not hrow or not hrow.get('faces'):
                        continue
                    # A MOVED atom is the same atom with different bonds, so "the same atom faces
                    # the same residue in both crystals" is true of the atom and misleading about
                    # the interaction -- 400 rows invited "retained" for an atom whose connectivity
                    # changed. Keep the row, but say what it is.
                    _is_moved = hn in {x['atom'] for x in site_atoms
                                       if (x.get('role') or '').startswith('MOVED')}
                    _imet=collections.Counter(); _amet=collections.Counter()
                    for _x in rows:
                        _m=re.match(r'^([A-Z]{1,3})(\d+)\.', _x.get('faces') or '')
                        if _m and _m.group(1) in METALS: _imet[_m.group(1)]+=1
                    for _x in cs:
                        _m=re.match(r'^([A-Z]{1,3})(\d+)\.', _x.get('faces') or '')
                        if _m and _m.group(1) in METALS: _amet[_m.group(1)]+=1
                    def _nf(tok, resname):
                        # a metal/cofactor's residue NUMBER is arbitrary and differs between
                        # depositions -- but ONLY when there is one of them. With two magnesiums in
                        # the site, ignoring the number matches an arbitrary one of the two.
                        _unique=(resname in COFACTOR_HET or
                                 (resname in METALS and _imet[resname]<=1 and _amet[resname]<=1))
                        return tok.split('.',1)[-1] if _unique else tok
                    _hr=hrow['faces'].split('.')[0]
                    _cr=c['faces'].split('.')[0]
                    _hres=''.join(ch for ch in _hr if ch.isalpha())
                    _cres=''.join(ch for ch in _cr if ch.isalpha())
                    if (hrow['interaction'].startswith('no contact within')
                            or c['interaction'].startswith('no contact within')):
                        continue
                    if _nf(hrow['faces'],_hres)==_nf(c['faces'],_cres):
                        # The CHAIN was ignored, so residue name+number+atom alone decided it --
                        # and 6,664 rows across 9.6% of pairs matched two DIFFERENT chains while
                        # the field asserted "the same protein atom in both crystals". In a dimer
                        # the two chains share numbering, so chain B's TYR113 was being compared
                        # with chain A's. Disclose it rather than assert equality we have not
                        # checked: the contact may be the same interaction in the other copy, or a
                        # lattice contact, and the labeller must decide which.
                        _row={'input_atom':hn,'analogue_atom':c['atom'],'faces':c['faces'],
                              # both labels, because they are not always the same token: a metal or
                              # cofactor's residue NUMBER differs between depositions and the match
                              # deliberately ignores it, so printing only the analogue's label made
                              # the input side name something absent from its own table (311 pairs).
                              'input_faces':hrow['faces'],'analogue_faces':c['faces'],
                              'input_distance_A':hrow['distance_A'],
                              'analogue_distance_A':c['distance_A'],
                              'input_interaction':hrow['interaction'],
                              'analogue_interaction':c['interaction'],
                              'input_chain':hrow.get('chain'),'analogue_chain':c.get('chain')}
                        if _is_moved:
                            _row['moved_caveat']=(
                                'THIS ATOM IS MOVED: it exists in both molecules but its bonds '
                                'differ, so the contact is not a before/after comparison of the '
                                'same chemical group. Do not call it retained.')
                        _ia=(hrow['interaction'] or ''); _ib=(c['interaction'] or '')
                        def _cls(t):
                            for k in ('COVALENT BOND','METAL COORDINATION','SALT BRIDGE','H-bond'):
                                if t.startswith(k) or k in t[:24]: return k
                            return 'contact'
                        if _cls(_ia)!=_cls(_ib):
                            _row['typing_changed']=(
                                'NOT THE SAME INTERACTION: the input side is %s and the analogue '
                                'side is %s. The two atoms are in contact in both crystals, which '
                                'is what "conserved" means -- it does NOT mean the interaction '
                                'survived. Do not write that this %s is retained.'
                                %(_cls(_ia),_cls(_ib),_cls(_ia)))
                        if hrow.get('chain') and c.get('chain') and hrow['chain']!=c['chain']:
                            _row['chain_caveat']=(
                                'DIFFERENT CHAINS: chain %s in the input, chain %s in the analogue. '
                                'Same residue number, but not verified to be the same copy of the '
                                'protein. Do not call this retained without saying so.'
                                %(hrow['chain'],c['chain']))
                        cons.append(_row)
                _cmap=collections.defaultdict(set)
                for _c in cons:
                    if _c.get('input_chain') and _c.get('analogue_chain'):
                        _cmap[_c['input_chain']].add(_c['analogue_chain'])
                _bijection=bool(_cmap) and all(len(v)==1 for v in _cmap.values()) and \
                    len({next(iter(v)) for v in _cmap.values()})==len(_cmap)
                _crossed=[_c for _c in cons if _c.get('chain_caveat')]
                if _bijection and _crossed:
                    for _c in _crossed:
                        _c.pop('chain_caveat',None)
                lab={'analogue_pdb':pid,'analogue_contacts':cs,
                     '_chain_bijection':(', '.join('%s->%s'%(k,next(iter(v)))
                                                   for k,v in sorted(_cmap.items()))
                                         if (_bijection and _crossed) else None),
                     'conserved_interactions':cons,
                     'conserved_caveat':None,
                     'conserved_note':('contacts the edit did NOT disturb: the same input atom faces '
                         'the same protein atom in both crystals. Measured on both sides, not '
                         'inferred. You MAY describe these as retained/held (rule 5 permits it); '
                         'they are a labelling aid and are never shown to the model.'),
                     'analogue_contact_note':('every analogue heavy atom is listed. `input_atom` gives the '
                         'INPUT molecule\'s name for atoms conserved across the edit, so gains and losses can '
                         'be read off directly; `status: NEW` atoms exist only in the analogue. Both '
                         'sides use the same %.1f A cutoff.'%CONTACT_CUTOFF)}
                break
        if lab:
            import re as _re
            # `faces` is now None on an explicit no-contact row (fix 2), so the overlap is computed
            # over the atoms that actually have a contact.
            _L={_re.match(r'([A-Z]{3}\d+)',c['faces']).group(1) for c in lab['analogue_contacts']
                if c.get('faces') and _re.match(r'([A-Z]{3}\d+)',c['faces'])}
            _H={_re.match(r'([A-Z]{3}\d+)',a['faces']).group(1) for a in rows
                if _re.match(r'([A-Z]{3}\d+)',a['faces'])}
            _H|={r for a in rows for r in a['pocket_room_lining']}
            ov=len(_L&_H)/len(_L) if _L else 0.0
            lab['residue_numbering_overlap']=round(ov,2)
            lab['numbering_consistent']=bool(ov>=0.3)
        if lab is None: lab={}
        # Stated here rather than in a later pass, which is how v7's first render came out saying
        # "EDIT: (not annotated)". 84% of the corpus adds atoms and the categorical edit_type alone
        # gives no magnitude -- a labeller cannot reason about whether a group fits a pocket without
        # knowing how big the group is. Lead-derived, so LABEL_ONLY.
        _d=r['heavy'][1]-r['heavy'][0]
        lab['heavy_atoms_added']=_d
        lab['edit_size_note']=(
            'the analogue adds %d heavy atoms to the input molecule (%d -> %d)'%(_d,r['heavy'][0],r['heavy'][1])
            if _d>0 else
            'the analogue removes %d heavy atoms from the input molecule (%d -> %d)'%(-_d,r['heavy'][0],r['heavy'][1])
            if _d<0 else
            'same heavy-atom count as the input molecule (%d): a substitution or a relocation, not '
            'a change in size'%r['heavy'][0])
        # The flag was `max(room over ALL site atoms) == 0.0`, so one unrelated site atom with room
        # masked a genuinely blocked attachment point -- 202 false negatives, exactly the rule-7
        # case the flag exists to catch -- while ignoring `room_reading` produced 47 false positives
        # where every site atom read "open to solvent" rather than blocked, and the caveat then
        # described the opposite situation. Judge each GROWTH POINT on its own, and require the
        # reading to actually say blocked.
        _ap=[x for x in site_atoms
             if 'ATTACHMENT POINT' in (x.get('role') or '') or 'CHANGED' in (x.get('role') or '')]
        _blocked=[x for x in _ap if x.get('pocket_room_A')==0.0
                  and 'blocked' in (x.get('room_reading') or '')]
        _ap_open=[x for x in _ap if (x.get('pocket_room_A') or 0) > 0
                  and 'available' in (x.get('room_reading') or '')]
        lab['room_contradicts_edit']=bool(_d>0 and _blocked and not _ap_open)
        if lab['room_contradicts_edit']:
            lab['room_caveat']=(
                'The room probe reports 0.0 A and "blocked by protein" at %s, yet the analogue adds '
                '%d heavy atoms at this site. The probe measures enclosure around an atom that is '
                'already bonded into the molecule, which is not the same question as whether a '
                'substituent can grow from it -- and the analogue proves one can. Do NOT write that '
                'there is no room here.'%(', '.join(x['atom'] for x in _blocked), _d))
        # STEREOCHEMISTRY. 709 pairs (10.6%) differ in their chiral tags with nothing in the
        # evidence saying so: 6YV0 inverts [C@@H] to [C@H] with identical neighbours and the only
        # annotation was "adds 4 heavy atoms". An inversion is a real edit with real pocket
        # consequences, and atom-by-atom "conserved" claims across an inverted centre are not
        # comparisons of the same thing.
        try:
            from rdkit.Chem import FindMolChiralCenters as _fmc
            # Compare each centre to ITS COUNTERPART through the atom map. The first version
            # compared the tag lists positionally, which is meaningless across molecules with
            # different atom ordering; the "fix" compared sorted multisets, which silently withdrew
            # every DOUBLE inversion because {R,S} == {S,R}. And a centre whose substituents the
            # edit changed has no comparable descriptor at all -- that is an edit at the centre,
            # not an inversion of it.
            _ca=dict(_fmc(A,useLegacyImplementation=False,includeUnassigned=False))
            _cb=dict(_fmc(B,useLegacyImplementation=False,includeUnassigned=False)) if B is not None else {}
            _inv=[]; _incomp=[]; _unmapped=[]; _lost=[]
            def _pseudo_centre(mol,idx):
                at=mol.GetAtomWithIdx(idx)
                if at.GetSymbol() not in ('P','S'):
                    return False
                return sum(1 for nb in at.GetNeighbors()
                           if nb.GetSymbol() in ('O','S') and nb.GetDegree()==1) >= 2
            for _i,_ta in _ca.items():
                if _pseudo_centre(A,_i):
                    continue
                _j=(_amap or {}).get(_i)
                if _j is None:
                    _unmapped.append(nm_of.get(_i) or ('atom%d'%_i))
                    continue
                if _j not in _cb:
                    _lost.append(nm_of.get(_i) or ('atom%d'%_i))
                    continue
                _na={(_amap or {}).get(n.GetIdx()) for n in A.GetAtomWithIdx(_i).GetNeighbors()}
                _nb={n.GetIdx() for n in B.GetAtomWithIdx(_j).GetNeighbors()}
                _flip=parity_inverted(A,B,_i,_j,_amap or {})
                if None in _na or _na!=_nb or _flip is None:
                    _incomp.append(nm_of.get(_i) or ('atom%d'%_i))
                elif _flip:
                    _inv.append((nm_of.get(_i) or ('atom%d'%_i),_ta,_cb[_j]))
            if _inv:
                lab['stereo_note']=(
                    'A STEREOCENTRE IS INVERTED: %s. The substituents at that centre are unchanged, '
                    'so this is a genuine inversion and may be the whole point of the pair. Atom '
                    'rows marked conserved across it are the same ATOM, not the same geometry.'
                    %', '.join('%s %s->%s'%(n,x,y) for n,x,y in _inv))
            elif len(_ca)!=len(_cb):
                lab['stereo_note']=(
                    'the edit changes the NUMBER of stereocentres (%d -> %d).'%(len(_ca),len(_cb)))
            elif _unmapped or _lost:
                lab['stereo_note']=(
                    'A STEREOCENTRE COULD NOT BE COMPARED ACROSS THE EDIT: %s. The atom map does '
                    'not connect it to a stereocentre in the analogue, so whether its configuration '
                    'is retained, inverted or destroyed is UNKNOWN from this evidence. Do not state '
                    'or imply either.'%', '.join(_unmapped+_lost))
            elif _incomp:
                lab['stereo_note']=(
                    'stereochemistry at %s cannot be compared: the edit changes the substituents on '
                    'that centre, so its R/S label is not the same question on the two sides. This '
                    'is NOT an inversion.'%', '.join(_incomp))
        except Exception as _e:
            lab['stereo_note_failed']=repr(_e)[:200]
        # COVALENCY MISMATCH. 8ANF's input is a reacted chloroacetamide bonded to CYS38.SG with its
        # chloride gone; the analogue's warhead is intact and unreacted 3.71 A away. That is the
        # pair's entire story, and it appeared nowhere: the bond is simply ABSENT from
        # conserved_interactions, so rule 5's own worked template ("the warhead's Cys38 bond is
        # held") produces a sentence the analogue table contradicts. State the mismatch explicitly.
        _icov={a['atom'] for a in rows if (a['interaction'] or '').startswith('COVALENT BOND')}
        _acov={c['atom'] for c in (lab.get('analogue_contacts') or [])
               if (c.get('interaction') or '').startswith('COVALENT BOND')}
        if _icov and not _acov:
            lab['covalency_mismatch']=(
                'THE COVALENT BOND IS NOT PRESENT IN THE ANALOGUE. The input molecule is covalently '
                'bonded to the protein (%s), and no atom of the analogue is. Either the analogue is '
                'an unreacted warhead or it binds non-covalently. Do NOT write that the covalent '
                'bond is retained -- that this pair LOSES it may be the most important thing to say.'
                %', '.join(sorted(_icov)))
        elif _acov and not _icov:
            lab['covalency_mismatch']=(
                'THE ANALOGUE IS COVALENT AND THE INPUT IS NOT. No input atom is covalently bonded '
                'to the protein; analogue atom(s) %s are. The edit appears to install a covalent '
                'bond, which is a major change and is attributable to the analogue crystal.'
                %', '.join(sorted(_acov)))
        # SYMMETRY-EQUIVALENT ATOMS. The MCS maps graph-equivalent atoms arbitrarily, so a pair of
        # chemically identical phenyls or the three oxygens of a phosphonate can come back mapped
        # crosswise. Read literally, `input_atom` then says the two rings EXCHANGED sub-pockets
        # (7Q1P) or hides a retained salt bridge because the oxygen was relabelled (8C25). Neither
        # is a chemical event. Detect it from canonical ranks -- atoms sharing a rank are
        # indistinguishable, so any claim keyed on their individual identity is unsafe.
        try:
            from rdkit.Chem import CanonicalRankAtoms as _cra
            _ra=list(_cra(A,breakTies=False))
            _dup=collections.Counter(_ra)
            _sym={nm_of[i] for i in range(A.GetNumAtoms())
                  if i in nm_of and _dup[_ra[i]]>1}
            # resonance-equivalent terminal oxygens/sulfurs: a carboxylate, phosphate, sulfonate,
            # sulfonamide or nitro group's oxygens are interchangeable however the SMILES draws
            # them, and canonical ranks do not see that.
            for _at in A.GetAtoms():
                if _at.GetSymbol() not in ('C','P','S','N'):
                    continue
                _term=[nb.GetIdx() for nb in _at.GetNeighbors()
                       if nb.GetSymbol() in ('O','S') and nb.GetDegree()==1]
                if len(_term)>1:
                    _sym |= {nm_of[i] for i in _term if i in nm_of}
            _symsite=_sym & {x['atom'] for x in site_atoms}
            # also rank the ANALOGUE: two isopropyl methyls that only the analogue has are assigned
            # arbitrarily by the mapping, and ranking A alone could never see that.
            try:
                _rb=list(_cra(B,breakTies=False)); _db=collections.Counter(_rb)
                _symb={bnm[i] for i in range(B.GetNumAtoms())
                       if i in bnm and _db[_rb[i]]>1}
                for _at in B.GetAtoms():
                    if _at.GetSymbol() in ('C','P','S','N'):
                        _t=[nb.GetIdx() for nb in _at.GetNeighbors()
                            if nb.GetSymbol() in ('O','S') and nb.GetDegree()==1]
                        if len(_t)>1:
                            _symb |= {bnm[i] for i in _t if i in bnm}
            except Exception:
                _symb=set()
            _symcons={c['input_atom'] for c in (lab.get('conserved_interactions') or [])
                      if c.get('input_atom') in _sym}
            # an analogue_contacts row is the table labellers read gains and losses off, so an
            # ambiguity visible only there is exactly the one that misleads
            _symcons |= {c.get('input_atom') for c in (lab.get('analogue_contacts') or [])
                         if c.get('input_atom') in _sym}
            _symcons |= {c['atom'] for c in (lab.get('analogue_contacts') or [])
                         if c.get('atom') in _symb}
            _symcons={x for x in _symcons if x}
            # per-atom partner residue, input side and analogue side
            _iface={a2['atom']:(a2.get('faces') or '').split('.')[0] for a2 in rows}
            _aface={}
            for _c in (lab.get('analogue_contacts') or []):
                if _c.get('input_atom'):
                    _aface[_c['input_atom']]=(_c.get('faces') or '').split('.')[0]
            _classes=collections.defaultdict(set)
            for _i in range(A.GetNumAtoms()):
                if _i in nm_of and nm_of[_i] in _sym:
                    _classes[_ra[_i]].add(nm_of[_i])
            _exchanged=set()
            for _cls in _classes.values():
                _mem=[m for m in _cls if m in _iface and m in _aface]
                if len(_mem)<2:
                    continue
                _in={m:_iface[m] for m in _mem}
                _an={m:_aface[m] for m in _mem}
                # same residues involved on both sides, different assignment -> a permutation
                if set(_in.values())==set(_an.values()) and any(_in[m]!=_an[m] for m in _mem):
                    _exchanged |= set(_mem)
            # Fire on the DAMAGE, not on the presence of symmetry. Triggering whenever any
            # conserved or analogue row touched a symmetric atom reached 96.9% of pairs -- nearly
            # every drug-like molecule has a phenyl -- which is noise that trains labellers to skip
            # caveats. The two cases that actually mislead: the mapping IS a permutation within a
            # class (_exchanged), or the EDIT itself lands on a symmetric atom (_symsite).
            # A symmetric atom the edit DELETED carries no cross-molecule identity claim: there
            # is nothing on the analogue side it could have been confused with. 6CZE's O2G/O3G and
            # 9T23's F1/F2/F3 are all CHANGED rows with no counterpart, so the warning there was
            # inert and it chilled the only citable facts about those deletions. Require the
            # symmetric edited atom to actually have a counterpart.
            _counterpart={c.get('input_atom') for c in (lab.get('analogue_contacts') or [])
                          if c.get('input_atom')}
            _counterpart |= {c.get('input_atom') for c in (lab.get('conserved_interactions') or [])
                             if c.get('input_atom')}
            _symsite = _symsite & _counterpart
            if _symsite or _exchanged:
                lab['symmetry_equivalent_atoms']=sorted(_sym)
                if _exchanged:
                    lab['symmetry_exchanged_atoms']=sorted(_exchanged)
                # Built by concatenation, not by chained %-formatting. The previous version applied
                # two `%` operations to one string whose four placeholders did not line up, so it
                # raised TypeError on every pair -- and the bare `except` below swallowed it. The
                # two atom lists above were written first, which is why the corpus showed 1,471
                # pairs with exchanged atoms and ZERO with the note that explains them.
                _names=', '.join(sorted(_symsite | _exchanged))
                _note=('SYMMETRY-EQUIVALENT ATOMS CARRY AN IDENTITY CLAIM HERE: %s %s '
                       'indistinguishable from other atoms of the same molecule by graph, so the '
                       'mapping between the two molecules is arbitrary among them. A row saying one '
                       'of these moved to a different residue may be a relabelling rather than a '
                       'change in pose -- two chemically identical rings can come back mapped '
                       'crosswise, which reads as the rings swapping sub-pockets when nothing '
                       'moved. Do not build a claim on which individual one of these atoms went '
                       'where.'
                       % (_names, 'is' if len(_symsite | _exchanged) == 1 else 'are'))
                if _exchanged:
                    _note += (' THE MAPPING FOR %s IS A PERMUTATION WITHIN ITS SYMMETRY CLASS: '
                              'those atoms face the same set of residues in both crystals but are '
                              'assigned to them differently, which is relabelling, not motion. A '
                              'conserved row on one of them is NOT a measurement of the same '
                              'contact on both sides.' % ', '.join(sorted(_exchanged)))
                lab['symmetry_note']=_note
        except Exception as _e:
            # Never silent again. A bare pass here hid a TypeError that killed this caveat on every
            # pair in the corpus while leaving its inputs in place, so the data looked computed.
            lab['symmetry_note_failed']=repr(_e)[:200]
        _nnew=sum(1 for c in (lab.get('analogue_contacts') or []) if c.get('status')=='NEW')
        if (_nnew and _d > 0
                and not any('ATTACHMENT POINT' in (x.get('role') or '') for x in site_atoms)):
            lab['no_attachment_point_warning']=(
                'THE ANALOGUE HAS %d NEW ATOM(S) BUT NO ATTACHMENT POINT ROW. The mapper could not '
                'identify which kept atom the new group hangs off -- this happens when a group is '
                'INSERTED between two kept atoms, or when a ring is rotated so the mapping shifts '
                'around it. Treat every SUBSTITUTION BOUNDARY role text saying "the molecule did '
                'NOT grow" as unreliable for this pair, and do not assert where the new group sits '
                'relative to the old one.'%_nnew)
        if lab.get('numbering_consistent') is False:
            lab['numbering_note']=(
                'THE TWO CRYSTALS DO NOT SHARE A RESIDUE NUMBERING (overlap %.2f). A residue number '
                'on one side need not mean the same residue on the other -- the two depositions may '
                'number differently, or be different domains or paralogues of the same family. Any '
                'conserved row in this pair that survived on a HETATM label alone (a metal or '
                'cofactor, whose residue number is arbitrary) is especially unsafe: the label can '
                'match by coincidence across two different sites. Do not call anything retained '
                'here without a residue-level reason.'%(lab.get('residue_numbering_overlap') or 0.0))
            # and a metal/cofactor-only match cannot carry a conserved row when the numbering is
            # already untrustworthy -- that combination is how two different active sites got
            # reported as sharing a zinc.
            _keep=[]
            for _c in (lab.get('conserved_interactions') or []):
                _rn=''.join(ch for ch in (_c.get('faces') or '').split('.')[0] if ch.isalpha())
                if _rn in METALS or _rn in COFACTOR_HET:
                    _c['numbering_caveat']=(
                        'MATCHED ON A HETATM LABEL ONLY, AND THIS PAIR HAS NO SHARED NUMBERING: '
                        '%s may not be the same site in the two crystals. Not citable as retained.'
                        %(_c.get('faces') or '?'))
                _keep.append(_c)
            lab['conserved_interactions']=_keep
        if lab.get('_chain_bijection'):
            lab['chain_relabel_note']=(
                'THE TWO DEPOSITIONS LABEL THE CHAINS DIFFERENTLY, CONSISTENTLY: %s. Every conserved '
                'row crosses chains in the same direction, which is the signature of a relabelled '
                'deposition rather than a different protein copy -- so these contacts may be '
                'described as retained. (Where the mapping is NOT consistent, individual rows carry '
                'their own !! DIFFERENT CHAINS warning instead.)'
                %lab['_chain_bijection'])
            lab.pop('_chain_bijection',None)
        _ch=collections.Counter(a['chain'] for a in rows if a.get('chain'))
        if len(_ch)>1:
            _mj,_mjn=_ch.most_common(1)[0]
            _minor=sorted(set(_ch)-{_mj})
            _mres=sorted({a['faces'].split('.')[0] for a in rows
                          if a.get('chain') in _minor and a.get('faces')})
            lab['multi_chain_note']=(
                'THE LIGAND CONTACTS MORE THAN ONE PROTEIN CHAIN. %d of %d contacts are chain %s; '
                'the rest come from chain(s) %s (%s). The target annotation names ONE protein, so '
                'a minority chain may be a second copy in the lattice, the other half of a '
                'biological dimer, or a bound peptide from a different protein entirely -- the '
                'evidence cannot tell you which. Do not describe a residue from a minority chain '
                'as pocket wall or as a growth direction without saying it is on another chain.'
                %(_mjn,sum(_ch.values()),_mj,','.join(_minor),', '.join(_mres[:6])))
        _term=sorted({a['faces'].split('.')[0] for a in rows
                      if (a.get('faces') or '').endswith('.OXT')})
        if _term:
            lab['terminus_note']=(
                'contacts a protein CHAIN TERMINUS (%s, via its terminal carboxylate OXT). A chain '
                'end is usually a neighbouring copy in the crystal rather than part of the binding '
                'site. Treat it as a lattice contact unless the rest of the evidence says otherwise.'
                %', '.join(_term))
        try:
            _tn={x['atom'] for x in site_atoms
                 if not (x.get('role') or '').startswith('SUBSTITUTION BOUNDARY')}
            _ti=[i for i in range(A.GetNumAtoms()) if nm_of.get(i) in _tn]
            if len(_ti)>=2:
                _dm=Chem.GetDistanceMatrix(A)
                # cluster touched atoms that are within 2 bonds of each other
                _cl=[]
                for i in _ti:
                    _hit=[c for c in _cl if any(_dm[i][k]<=4 for k in c)]
                    if _hit:
                        _hit[0].add(i)
                        for extra in _hit[1:]:
                            _hit[0] |= extra
                            _cl.remove(extra)
                    else:
                        _cl.append({i})
                if len(_cl)>=2:
                    _sep=max(min(_dm[a2][b2] for a2 in _cl[x] for b2 in _cl[y])
                             for x in range(len(_cl)) for y in range(x+1,len(_cl)))
                    if _sep>=5:
                        lab['multi_edit_warning']=(
                            'THIS PAIR LOOKS LIKE TWO INDEPENDENT TRANSFORMATIONS. The edited atoms '
                            'form %d separate groups, and the nearest atoms of two of them are %d '
                            'bonds apart in the molecule -- too far to be one contiguous change. One '
                            'sentence cannot cover both: describe the change with the larger pocket '
                            'consequence and say nothing about the other, or reject.'
                            %(len(_cl),int(_sep)))
        except Exception as _e:
            lab['multi_edit_failed']=repr(_e)[:200]
        lab['_covalent_present']=bool(_acov or _icov)
        lab['site_of_change']=site_atoms
        lab['site_note']=('which rows of the atom table the edit touches, and the room available '
                          'there. Label-side only: the model is not told where to edit.')
        out['LABEL_ONLY']={'note':'the ANALOGUE\'s own crystal. For writing the label only - NEVER shown '
                                  'to the model, which at inference has no analogue structure.',
                           **(lab or {'analogue_pdb':None,'analogue_contacts':[]})}
    return out

if __name__=='__main__':
    rows=[json.loads(l) for l in open(D+'/v6_pairs_tc50.jsonl')]
    pure=[r for r in rows if r['changed_hit']==0][0]
    e=build(pure)
    print('=== PURE ADDITION example (the case that had no evidence before) ===')
    print(json.dumps({k:e[k] for k in ('pair','edit_type','MODEL_INPUT')}, indent=1)[:1500])
    print('\n=== LABEL_ONLY side (analogue crystal, %s) ===' % e['LABEL_ONLY']['analogue_pdb'])
    for c in e['LABEL_ONLY']['analogue_contacts'][:6]: print('   %s' % c)
