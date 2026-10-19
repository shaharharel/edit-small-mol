from rdkit import Chem
from rdkit.Chem import rdMolDescriptors, Descriptors, rdFMCS

INP = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
CANDS = {
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
acryl = Chem.MolFromSmarts("[CH2]=[CH]C(=O)[NX3]")
methacryl = Chem.MolFromSmarts("[CH2]=[C]([CH3])C(=O)[NX3]")
imid = Chem.MolFromSmarts("c1cnc[nX3]1")

print("INPUT", rdMolDescriptors.CalcMolFormula(mi), "| canon:", Chem.MolToSmiles(mi))
for k in sorted(CANDS):
    m = Chem.MolFromSmiles(CANDS[k])
    mcs = rdFMCS.FindMCS([mi, m], timeout=10, ringMatchesRingOnly=True, completeRingsOnly=True)
    rings = sorted(len(r) for r in m.GetRingInfo().AtomRings())
    st = Chem.FindMolChiralCenters(m, includeUnassigned=True, useLegacyImplementation=False)
    print(f"\ncand_{k:02d} {Chem.MolToSmiles(m)}")
    print(f"   formula={rdMolDescriptors.CalcMolFormula(m)} MW={Descriptors.MolWt(m):.1f} "
          f"cLogP={Descriptors.MolLogP(m):.2f} rings={rings} stereo={st}")
    print(f"   acrylamide={bool(m.GetSubstructMatches(acryl))} alpha_methyl={bool(m.GetSubstructMatches(methacryl))} "
          f"Nsubst_imidazole={bool(m.GetSubstructMatches(imid))}")
    print(f"   MCS={mcs.numAtoms}/{mi.GetNumAtoms()} atoms  smarts={mcs.smartsString[:90]}")
    print(f"   same_as_input={Chem.MolToSmiles(m)==Chem.MolToSmiles(mi)}")
