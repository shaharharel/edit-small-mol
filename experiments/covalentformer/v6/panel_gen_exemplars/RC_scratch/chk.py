import re, sys
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')
txt=open('../panel.txt').read()
blocks=re.split(r'^PAIR \d+ of 12', txt, flags=re.M)[1:]
ids=re.findall(r'pair_id (pair_\d+)\s+(\S+)\s+(\S+)', txt)
for i,b in enumerate(blocks):
    pid=ids[i][0]
    inp=re.search(r'INPUT SMILES: (\S+)', b).group(1)
    mi=Chem.MolFromSmiles(inp)
    ci=Chem.MolToSmiles(mi) if mi else None
    print(f"=== {pid} {ids[i][1]} {ids[i][2]}  input atoms={mi.GetNumHeavyAtoms() if mi else 'BAD'} canon_ok={ci is not None}")
    for L,s in re.findall(r'PROPOSAL ([ABC])\n\s*molecule: (\S+)', b):
        m=Chem.MolFromSmiles(s)
        if m is None:
            print(f"  {L}: PARSE_FAIL {s}"); continue
        c=Chem.MolToSmiles(m)
        print(f"  {L}: na={m.GetNumHeavyAtoms():3d} echo={c==ci} canon={c}")
