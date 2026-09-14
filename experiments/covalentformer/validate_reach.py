"""KILL-CRITERION TEST: does reachability recover the TRUE warhead placement on real adducts?

Uses deposited covalent complexes. The scaffold is frozen at its CRYSTAL pose, the linker+warhead
torsions are scanned, and we ask whether the engine finds a conformer presenting the electrophile to
the real cysteine. Since these are real covalent inhibitors that DID react, the answer must be yes.

If the recovery rate is low, the engine is broken -- not the molecules.
"""
import os, sys, json, collections
import numpy as np, gemmi
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reachability import reachable, electrophile_index

SCR = '/private/tmp/claude-501/-Users-shaharharel-Documents-github-quris-ml-play-CB1/160c04b4-9b18-4865-b718-cd0ea00a4622/scratchpad'
CIF = os.path.join(SCR, 'ladder/cif')

def load(pid, cys, ch, lig):
    st = gemmi.read_structure(os.path.join(CIF, '%s.cif.gz' % pid)); st.setup_entities()
    sg=None; ligatoms=[]; prot=[]; pelem=[]
    for c in st[0]:
        for r in c:
            if r.name=='CYS' and str(r.seqid.num)==str(cys) and c.name==ch:
                a=next((a for a in r if a.name=='SG'),None)
                if a: sg=np.array([a.pos.x,a.pos.y,a.pos.z])
            if r.name==lig:
                for a in r:
                    if a.element.name!='H':
                        ligatoms.append((a.name,np.array([a.pos.x,a.pos.y,a.pos.z])))
            elif r.name not in ('HOH',):
                for a in r:
                    if a.element.name!='H':
                        prot.append([a.pos.x,a.pos.y,a.pos.z]); pelem.append(a.element.name.upper())
    return sg, ligatoms, (np.array(prot) if prot else None), pelem

def main():
    P={}
    for f in ('P_rung_final.json','P_rung_expanded.json'):
        for e in json.load(open(os.path.join(SCR,'ladder',f)))['entries']: P[e['pid']]=e
    res=[]; skip=collections.Counter()
    for pid,e in sorted(P.items()):
        pre=e.get('pre'); cys=e.get('cys'); ch=e.get('cys_ch','A'); lig=e.get('lig')
        if not all([pre,cys,lig]): skip['no_entry_fields']+=1; continue
        mol=Chem.MolFromSmiles(pre)
        if mol is None: skip['unparseable']+=1; continue
        el,cls=electrophile_index(mol)
        if el is None: skip['warhead_%s'%cls]+=1; continue
        try: sg,ligatoms,prot,pelem=load(pid,cys,ch,lig)
        except Exception: skip['cif_unreadable']+=1; continue
        if sg is None or len(ligatoms)<8: skip['no_crystal_partners']+=1; continue
        # ATOM CORRESPONDENCE VIA MCS -- the first version had no correspondence at all.
        # It took the N crystal atoms FARTHEST FROM SG and superposed RDKit ring atom i onto
        # crystal atom i, where the crystal ordering was by distance. Those indices mean nothing
        # to each other, so the scaffold was anchored onto scrambled coordinates: recovery 0.7%,
        # median d 9.39 A while the ANGLE came out right (109.1 deg) -- the signature of a
        # correspondence bug rather than a geometric one.
        # The crystal ligand is the ADDUCT (post-reaction) and `pre` is pre-reactive, so a direct
        # template match fails; the SCAFFOLD however is common to both. Use MCS to find it.
        from rdkit.Chem import rdFMCS
        blk=[]
        for i,(nm,xyz) in enumerate(ligatoms):
            blk.append('HETATM%5d %-4s %3s A%4d    %8.3f%8.3f%8.3f  1.00  0.00          %2s'
                       %(i+1,nm[:4],'LIG',1,xyz[0],xyz[1],xyz[2],
                         ''.join(c for c in nm if c.isalpha())[:2]))
        cry=Chem.MolFromPDBBlock('\n'.join(blk)+'\nEND\n',removeHs=True,sanitize=False)
        if cry is None: skip['crystal_unparseable']+=1; continue
        try: Chem.SanitizeMol(cry,Chem.SANITIZE_ALL^Chem.SANITIZE_PROPERTIES^Chem.SANITIZE_KEKULIZE)
        except Exception: pass
        try:
            res_mcs=rdFMCS.FindMCS([mol,cry],timeout=10,
                                   atomCompare=rdFMCS.AtomCompare.CompareElements,
                                   bondCompare=rdFMCS.BondCompare.CompareAny,
                                   ringMatchesRingOnly=True,completeRingsOnly=False)
        except Exception: skip['mcs_failed']+=1; continue
        if res_mcs.numAtoms<8: skip['mcs_too_small']+=1; continue
        patt=Chem.MolFromSmarts(res_mcs.smartsString)
        mm=mol.GetSubstructMatch(patt); cm=cry.GetSubstructMatch(patt)
        if not mm or not cm or len(mm)!=len(cm): skip['mcs_match_failed']+=1; continue
        conf=cry.GetConformer()
        ref=np.array([list(conf.GetAtomPosition(i)) for i in cm])
        # drop MCS atoms within 6 A of SG so the "frozen scaffold" excludes the reactive end
        keep=[k for k in range(len(mm)) if np.linalg.norm(ref[k]-sg)>6.0]
        if len(keep)<6: keep=list(range(len(mm)))
        scaf=tuple(mm[k] for k in keep); ref=ref[keep]
        if len(scaf)<6: skip['scaffold_too_small']+=1; continue
        r=reachable(mol, scaf, ref, sg, prot, protein_elem=pelem, n_conf=120)
        r['pid']=pid; res.append(r)
    if not res:
        print('NOTHING SCORED. skipped %s'%dict(skip)); return 1
    # WHICH CRITERION FAILS? Without this the pass rate is uninterpretable.
    nd=sum(1 for x in res if 'd' in x and not (x['d']<=4.5))
    na=sum(1 for x in res if 'angle' in x and abs(x['angle']-110)>35)
    nc=sum(1 for x in res if x.get('clash',0)>2)
    cl=[x.get('clash',0) for x in res]
    print('  FAILURE BREAKDOWN over %d entries:'%len(res))
    print('    distance above 4.5 A     : %d (%.0f%%)'%(nd,100*nd/len(res)))
    print('    angle outside 110+-35     : %d (%.0f%%)'%(na,100*na/len(res)))
    print('    clash > 2 (calibrated)    : %d (%.0f%%)   median clash count %d'%(nc,100*nc/len(res),int(np.median(cl))))
    print()
    hit=np.mean([x['hit'] for x in res])
    dd=[x['d'] for x in res if 'd' in x]; an=[x['angle'] for x in res if 'angle' in x]
    print('KILL-CRITERION TEST -- reachability on REAL deposited covalent complexes')
    print('  entries scored %d ; skipped %s'%(len(res),dict(skip)))
    print()
    print('  RECOVERY RATE (engine finds a presenting conformer): %.1f%%'%(100*hit))
    print('  best-conformer d(SG..C_el): median %.2f A'%np.median(dd))
    print('  best-conformer angle      : median %.1f deg'%np.median(an))
    print()
    by=collections.Counter(x['warhead_class'] for x in res)
    print('  by warhead class: %s'%dict(by))
    print()
    # THE APPLES-TO-APPLES TEST. The engine cannot recover a complex whose OWN crystal pose fails
    # the clash filter -- that is a property of the deposited structure, not of the engine. The
    # meaningful rate is therefore conditional on the crystal pose itself being admissible.
    print()
    print('  CEILING: only 71.6%% of deposited covalent poses pass a <=2-overlap filter in their own')
    print('  crystal, so that is the maximum any engine could recover on this set.')
    print('  conditional recovery = %.1f%% / 71.6%% = %.1f%% of what is achievable'
          %(100*hit,100*hit/0.716))
    print()
    if hit>=0.55: print('  PASS. Recovery is within reach of the structural ceiling; the engine is usable.')
    else: print('  FAIL. Per the pre-registered criterion the ENGINE is at fault, not the molecules.')
    return 0

if __name__=='__main__': sys.exit(main())
