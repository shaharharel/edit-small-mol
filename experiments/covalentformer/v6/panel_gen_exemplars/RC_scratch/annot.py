import json,re,sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
d=json.load(open('/Users/shaharharel/Documents/github/edit-small-mol/data/pdb_mirror/ccd_atoms.json'))
BO={'SING':Chem.BondType.SINGLE,'DOUB':Chem.BondType.DOUBLE,'TRIP':Chem.BondType.TRIPLE,'AROM':Chem.BondType.AROMATIC}
def annot(code):
    e=d[code]; rw=Chem.RWMol(); idx={}
    for n,el in e['atoms']:
        if el=='H': continue
        a=Chem.Atom(el.capitalize() if len(el)>1 else el)
        num=int(re.sub(r'[^0-9]','',n) or 0)
        a.SetAtomMapNum(num%1000)
        idx[n]=rw.AddAtom(a)
    for a,b,t in e['bonds']:
        if a in idx and b in idx: rw.AddBond(idx[a],idx[b],BO.get(t,Chem.BondType.SINGLE))
    m=rw.GetMol()
    try: Chem.SanitizeMol(m)
    except Exception as ex: pass
    return Chem.MolToSmiles(m)
for c in sys.argv[1:]:
    print(c, annot(c) if c in d else 'MISSING')
