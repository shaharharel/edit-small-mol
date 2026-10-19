from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFMCS
from collections import Counter

INP = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
m = Chem.MolFromSmiles(INP)

# table names: element + running index over heavy atoms in molecule (SMILES) order
cnt = Counter(); names = {}
for a in m.GetAtoms():
    el = a.GetSymbol(); cnt[el] += 1
    names[a.GetIdx()] = f"{el}{cnt[el]}"

# group assignment via SMARTS on the input
groups = {
 "acrylamide_warhead": "[CH2]=[CH]C(=O)N",
 "isoindoline_CH2":    "[CH2;R]N",
 "benzo_ring":         "c1ccccc1",
 "carboxamide":        "c-C(=O)[NH]c",
 "imidazole_ring":     "c1cncn1",
 "N_isopropyl":        "n-[CH]([CH3])[CH3]",
}
assign = {i: [] for i in names}
for g, sma in groups.items():
    q = Chem.MolFromSmarts(sma)
    for mt in m.GetSubstructMatches(q):
        for i in mt:
            assign[i].append(g)

print("=== INPUT atom map ===")
for a in m.GetAtoms():
    i = a.GetIdx()
    print(f"{names[i]:<4} idx{i:<3} {a.GetSymbol():<2} arom={int(a.GetIsAromatic())} "
          f"nbrs={[names[n.GetIdx()] for n in a.GetNeighbors()]:} groups={assign[i]}")

print("\n=== isopropyl-owned atoms ===")
q = Chem.MolFromSmarts("[nX3]-[CH]([CH3])[CH3]")
mt = m.GetSubstructMatches(q)[0]
print("n attach:", names[mt[0]], " iPr carbons:", [names[i] for i in mt[1:]])
