from rdkit import Chem
def c(s): return Chem.MolToSmiles(Chem.MolFromSmiles(s))
# N-methyl analogue of input (what cand_03 claims to be + a "methylene insertion")
ref  = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C)cn3)c2C1"
c3_destripped = "C=CC(=O)N1Cc2c(C(=O)Nc3cn(C)cn3)cccc2C1"   # cand_03 minus the ester arm
print("ref (input, iPr->Me)      :", c(ref))
print("cand_03 core (ester removed):", c(c3_destripped))
print("carboxamide regiochem preserved:", c(ref)==c(c3_destripped))
# what is the added arm?
m=Chem.MolFromSmiles("C=CC(=O)N1Cc2c(C(=O)Nc3cn(C)cn3)cccc2C1CC(=O)OC")
print("methyl ester present:", bool(m.GetSubstructMatches(Chem.MolFromSmarts("[CX4]C(=O)O[CH3]"))))
print("new stereocentre unassigned:", Chem.FindMolChiralCenters(m,includeUnassigned=True,useLegacyImplementation=False))
# cand_06 ring identity
m6=Chem.MolFromSmiles("C=CC(=O)N1Cc2cccc(C(=O)Nc3ccnc(C#N)n3)c2C1")
print("cand_06 2-cyanopyrimidine (nitrile C flanked by 2 ring N):",
      bool(m6.GetSubstructMatches(Chem.MolFromSmarts("n:c(:n)C#N"))))
# cand_10 attachment: tertiary carbon on imidazole N?
m10=Chem.MolFromSmiles("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)(C)CO)cn3)c2C1")
print("cand_10 N-quaternary-C attach:", bool(m10.GetSubstructMatches(Chem.MolFromSmarts("[nX3]-[CX4]([CH3])([CH3])[CH2][OX2H]"))))
