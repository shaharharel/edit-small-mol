from rdkit import Chem
from rdkit.Chem import rdFMCS, rdMolDescriptors, Descriptors

INP = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
m = Chem.MolFromSmiles(INP)

# table naming: element + running index over heavy atoms in molecule (SMILES) order
counts = {}
names = {}
for a in m.GetAtoms():
    el = a.GetSymbol()
    counts[el] = counts.get(el, 0) + 1
    names[a.GetIdx()] = f"{el}{counts[el]}"

TABLE = {
 "C1":("CYS346.SG",1.40),"C2":("CYS346.SG",2.74),"C3":("PRO421.CG",3.60),
 "O1":("LYS424.NZ",3.02),"N1":("PRO421.CD",3.40),"C4":("PRO421.CD",3.71),
 "C5":("GLY420.CA",3.71),"C6":("GLY420.CA",3.67),"C7":("GLY418.O",3.36),
 "C8":("ALA417.O",3.27),"C9":("LEU344.CD1",3.35),"C10":("LEU344.CD1",3.33),
 "O2":("LEU344.CD1",3.52),"N2":("ALA417.O",3.13),"C11":("LEU468.CD1",3.56),
 "C12":("LEU468.CD1",3.44),"N3":("ALA367.CB",3.33),"C13":("MET414.SD",3.63),
 "C14":("LYS369.CG",3.67),"C15":("MET414.SD",3.53),"C16":("GLU415.O",3.12),
 "N4":("ALA417.N",2.86),"C17":("GLY420.CA",3.82),"C18":("PRO421.CD",3.68),
}

# group assignment by substructure
groups = {}
def tag(patt, label):
    q = Chem.MolFromSmarts(patt)
    for match in m.GetSubstructMatches(q):
        for i in match:
            groups.setdefault(i, []).append(label)

tag("C=CC(=O)N", "ACRYLAMIDE")
tag("[CH2]n1cnc(c1)", "IMIDAZOLE")          # ring
tag("c1cnc(n1)", "IMIDAZOLE")
tag("C(=O)Nc1cnc[nH0]1", "CARBOXAMIDE")
tag("n1cncc1", "IMIDAZOLE")
tag("[CH](C)(C)n", "N-ISOPROPYL")
tag("C1Cc2ccccc2C1", "ISOINDOLINE")
tag("c1ccccc1", "ARYL(isoindoline)")

print(f"{'idx':>3} {'name':>4} {'el':>2} arom nbrs            groups                        table")
for a in m.GetAtoms():
    i = a.GetIdx()
    nm = names[i]
    nbr = ",".join(names[x.GetIdx()] for x in a.GetNeighbors())
    g = "|".join(sorted(set(groups.get(i, ["?"]))))
    t = TABLE.get(nm, ("-", 0))
    print(f"{i:>3} {nm:>4} {a.GetSymbol():>2} {str(a.GetIsAromatic()):>5} {nbr:<18} {g:<28} {t[0]} {t[1]}")

print("\nINPUT formula:", rdMolDescriptors.CalcMolFormula(m), "MW %.1f" % Descriptors.MolWt(m))
