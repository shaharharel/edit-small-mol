from rdkit import Chem
s="CCC[C@H](CC)Oc1ccc(C(C)(C)C)cc1NC(=O)c1nnn(-c2cc(OC)ccc2OC)c1C"
m=Chem.MolFromSmiles(s)
can=Chem.MolToSmiles(m)
order=list(m.GetPropsAsDict(includePrivate=True,includeComputed=True)['_smilesAtomOutputOrder'])
print(can)
print(''.join(m.GetAtomWithIdx(i).GetSymbol() for i in order))
target="OCNCCO"+"C"*17+"NNN"+"CCOCCCCOCC"
print(target)
