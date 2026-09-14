"""What clash count does the DEPOSITED ligand score in its OWN crystal structure?

This calibrates the reachability threshold against reality instead of against my assumption.
If real covalent complexes routinely show N vdW overlaps at a 0.5 A tolerance, then demanding
zero overlaps rejects reality, and the criterion -- not the molecules -- is wrong.
"""
import os, sys, json, collections
import numpy as np, gemmi
SCR='/private/tmp/claude-501/-Users-shaharharel-Documents-github-quris-ml-play-CB1/160c04b4-9b18-4865-b718-cd0ea00a4622/scratchpad'
CIF=os.path.join(SCR,'ladder/cif')
VDW={'C':1.70,'N':1.55,'O':1.52,'S':1.80,'F':1.47,'CL':1.75,'BR':1.85,'I':1.98,'P':1.80}
TOL=0.5
P={}
for f in ('P_rung_final.json','P_rung_expanded.json'):
    for e in json.load(open(os.path.join(SCR,'ladder',f)))['entries']: P[e['pid']]=e
counts=[]
for pid,e in sorted(P.items()):
    cys=e.get('cys'); ch=e.get('cys_ch','A'); lig=e.get('lig')
    if not all([cys,lig]): continue
    try:
        st=gemmi.read_structure(os.path.join(CIF,'%s.cif.gz'%pid)); st.setup_entities()
    except Exception: continue
    sg=None; L=[]; Pp=[]; Pe=[]
    for c in st[0]:
        for r in c:
            if r.name=='CYS' and str(r.seqid.num)==str(cys) and c.name==ch:
                a=next((a for a in r if a.name=='SG'),None)
                if a: sg=np.array([a.pos.x,a.pos.y,a.pos.z])
            if r.name==lig:
                for a in r:
                    if a.element.name!='H':
                        L.append([a.pos.x,a.pos.y,a.pos.z]); 
            elif r.name!='HOH':
                for a in r:
                    if a.element.name!='H':
                        Pp.append([a.pos.x,a.pos.y,a.pos.z]); Pe.append(a.element.name.upper())
    if sg is None or len(L)<5 or not Pp: continue
    L=np.array(L); Pp=np.array(Pp)
    lr=np.full(len(L),1.7)   # ligand elements unavailable by name; carbon radius is the common case
    pr=np.array([VDW.get(x,1.7) for x in Pe])
    dm=np.linalg.norm(L[:,None,:]-Pp[None,:,:],axis=2)
    thr=lr[:,None]+pr[None,:]-TOL
    near=np.linalg.norm(Pp-sg,axis=1)<3.0     # same exclusion as the engine
    mask=~np.broadcast_to(near[None,:],dm.shape)
    counts.append(int(((dm<thr)&mask).sum()))
c=np.array(counts)
print('CLASH COUNT OF THE DEPOSITED LIGAND IN ITS OWN CRYSTAL (n=%d)'%len(c))
print('  vdW overlap, tol %.1f A, target-Cys neighbourhood excluded'%TOL)
print()
for q in (0,25,50,75,90,100):
    print('    %3d-th pct: %d'%(q,int(np.percentile(c,q))))
print()
print('  fraction with ZERO overlaps: %.1f%%'%(100*np.mean(c==0)))
for t in (0,1,2,3,5):
    print('    allowing <=%d overlaps admits %.1f%% of real crystal poses'%(t,100*np.mean(c<=t)))
