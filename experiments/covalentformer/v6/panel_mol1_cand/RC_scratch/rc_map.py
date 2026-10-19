from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFMCS, Descriptors

INP = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

def names(smi):
    m = Chem.MolFromSmiles(smi)
    cnt = {}
    out = []
    for a in m.GetAtoms():
        el = a.GetSymbol()
        cnt[el] = cnt.get(el, 0) + 1
        out.append((f"{el}{cnt[el]}", a.GetIdx(), el, a.GetIsAromatic(),
                    a.GetTotalNumHs(), [n.GetIdx() for n in a.GetNeighbors()]))
    return m, out

m, tbl = names(INP)
print("INPUT atom naming (element+running index, molecule order):")
for nm, idx, el, ar, nh, nbr in tbl:
    print(f"  {nm:5s} idx={idx:2d} {el} arom={int(ar)} H={nh} nbrs={nbr}")

print("\nRing info:")
for r in m.GetRingInfo().AtomRings():
    print("  ring size", len(r), "atoms", [tbl[i][0] for i in r])

# group assignment by substructure
groups = {
 "acrylamide": "C=CC(=O)N",
 "isoindoline_N_CH2x2": "C1c2ccccc2CN1",
 "carboxamide": "cC(=O)N",
 "imidazole": "c1cncn1",
 "N_isopropyl": "nC(C)C",
}
for g, sma in groups.items():
    q = Chem.MolFromSmarts(sma)
    ms = m.GetSubstructMatches(q)
    print(f"\n{g}: {[[tbl[i][0] for i in mm] for mm in ms]}")
