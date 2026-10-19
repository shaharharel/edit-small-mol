from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

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
mi = Chem.MolFromSmiles(INP)
acryl = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
methacryl = Chem.MolFromSmarts("[CH2]=[C]([CH3])C(=O)N")
imid = Chem.MolFromSmarts("c1cncn1")

def info(m):
    ri = m.GetRingInfo()
    return dict(f=rdMolDescriptors.CalcMolFormula(m),
                rings=sorted(len(r) for r in ri.AtomRings()),
                stereo=Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False),
                acryl=len(m.GetSubstructMatches(acryl)),
                methacryl=len(m.GetSubstructMatches(methacryl)),
                imid5=len(m.GetSubstructMatches(imid)))
print("INPUT", info(mi), Chem.MolToSmiles(mi))
for k in sorted(C):
    m = Chem.MolFromSmiles(C[k])
    mcs = rdFMCS.FindMCS([mi,m], ringMatchesRingOnly=True, completeRingsOnly=True, timeout=20)
    lost = mi.GetNumHeavyAtoms()-mcs.numAtoms; gained = m.GetNumHeavyAtoms()-mcs.numAtoms
    print(f"\ncand_{k:02d} {Chem.MolToSmiles(m)}")
    print("   ", info(m))
    print(f"    MCS={mcs.numAtoms}/{mi.GetNumHeavyAtoms()} lostFromInput={lost} newInCand={gained} same_as_input={Chem.MolToSmiles(m)==Chem.MolToSmiles(mi)}")
    print("    scaffold_same=", Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))==Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(mi)))
