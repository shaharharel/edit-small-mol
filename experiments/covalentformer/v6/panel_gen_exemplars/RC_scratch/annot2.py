import json,re,sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
d=json.load(open('/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/ccd_atoms.json'))
BO={'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}
for code in sys.argv[1:]:
    if code not in d: print(code,'MISSING'); continue
    e=d[code]; rw=Chem.RWMol(); idx={}; leg=[]
    for i,(n,el) in enumerate([x for x in e['atoms'] if x[1]!='H']):
        a=Chem.Atom(el.capitalize() if len(el)>1 else el); a.SetAtomMapNum(i+1)
        idx[n]=rw.AddAtom(a); leg.append(f"{i+1}={n}")
    for a,b,t in e['bonds']:
        if a in idx and b in idx: rw.AddBond(idx[a],idx[b],BO.get(t,Chem.BondType.SINGLE))
    m=rw.GetMol()
    try: Chem.SanitizeMol(m)
    except Exception: pass
    print(code, Chem.MolToSmiles(m))
    print('   LEGEND', ' '.join(leg))
