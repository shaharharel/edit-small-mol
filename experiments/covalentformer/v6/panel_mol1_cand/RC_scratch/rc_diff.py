from rdkit import Chem
from rdkit.Chem import rdMolDescriptors as rdmd, rdFMCS, AllChem

INP = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
C = {
1:"C=CC(=O)N1Cc2cccc(C(=O)Nc3c[nH]cn3)c2C1",
2:"C=CC(=O)N1Cc2cccc(C(=O)Nc3c[nH]cn3)c2C1",
3:"C=CC(=O)N1Cc2c(C(=O)Nc3cn(C)cn3)cccc2C1CC(=O)OC",
4:"C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C[C@H](C)C(F)(F)F)cn3)c2C1",
5:"C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(CCC(C)(C)O)cn3)c2C1",
6:"C=CC(=O)N1Cc2cccc(C(=O)Nc3ccnc(C#N)n3)c2C1",
7:"C=CC(=O)N1Cc2cccc(C(=O)Nc3c[nH]cn3)c2C1",
8:"C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(Cc4ccc(Cl)cc4)cn3)c2C1",
9:"C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C4CCC4)cn3)c2C1",
10:"C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)(C)CO)cn3)c2C1",
}
acryl = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")      # intact, unsubstituted alpha
methacryl = Chem.MolFromSmarts("[CH2]=[C]([CH3])C(=O)N")
imid = Chem.MolFromSmarts("c1cncn1")
ipr  = Chem.MolFromSmarts("[n]C([CH3])[CH3]")

mi = Chem.MolFromSmiles(INP)
print("INPUT", rdmd.CalcMolFormula(mi), "| canon:", Chem.MolToSmiles(mi))
print()
for k in sorted(C):
    m = Chem.MolFromSmiles(C[k])
    rings = sorted(len(r) for r in m.GetRingInfo().AtomRings())
    mcs = rdFMCS.FindMCS([mi, m], timeout=20, ringMatchesRingOnly=True, completeRingsOnly=False)
    stereo = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    print(f"cand_{k:02d} {rdmd.CalcMolFormula(m):18s} rings={rings} MCS={mcs.numAtoms}atoms/{mcs.numBonds}b")
    print(f"   acrylamide_intact={bool(m.GetSubstructMatches(acryl))} methacrylamide={bool(m.GetSubstructMatches(methacryl))}"
          f" imidazole={len(m.GetSubstructMatches(imid))} NiPr={len(m.GetSubstructMatches(ipr))}")
    print(f"   stereocentres={stereo}  canon={Chem.MolToSmiles(m)}")
    print(f"   same_as_input={Chem.MolToSmiles(m)==Chem.MolToSmiles(mi)}  MCSsmarts={mcs.smartsString[:90]}")
    print()
# duplicate detection among candidates
from collections import defaultdict
d=defaultdict(list)
for k in C: d[Chem.MolToSmiles(Chem.MolFromSmiles(C[k]))].append(k)
print("duplicate groups:", [v for v in d.values() if len(v)>1])
