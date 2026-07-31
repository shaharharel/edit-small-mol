"""Generate BMX YAMLs with STRATIFIED decoy sampling across all parent actives.

Fixes the `decoys.head(100)` bug — instead, draws ~100 decoys spanning ALL
parent actives in si_004's BMX_decoys sheet (~50/parent → take 4 per parent
randomly).

Output:
  /tmp/bmx_stratified_yamls/BMX/BMX_dec_*.yaml   (100 NEW stratified decoys)
  /tmp/bmx_stratified_yamls/manifest.csv         (these decoys only)
"""
import pandas as pd
import requests, time, sys
from pathlib import Path
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np

SI4 = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/covalid/ja5c22222_si_004.xlsx")
SI2 = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/covalid/ja5c22222_si_002.xlsx")
OUT_ROOT = Path("/tmp/bmx_stratified_yamls/BMX")
MANIFEST = Path("/tmp/bmx_stratified_yamls/manifest.csv")
N_DECOYS = 100
SEED = 42


from rdkit.Chem import AllChem
ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=[O])[N]")
ACRYLATE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=[O])[O]")

def boltz_atom_name_for_warhead(smi: str) -> str | None:
    """Return Boltz atom name for the warhead β-CH2 (Michael target).
    Matches the canonical generator in gen_covalid_boltz_yamls.py."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None: return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    for smarts in (ACRYLAMIDE_SMARTS, ACRYLATE_SMARTS):
        matches = mol_h.GetSubstructMatches(smarts)
        if matches:
            term_ch2_idx = matches[0][0]
            return f"C{can[term_ch2_idx] + 1}"
    return None


def fetch_uniprot_sequence(uniprot_id: str) -> str:
    url = f"https://rest.uniprot.org/uniprotkb/{uniprot_id}.fasta"
    r = requests.get(url, timeout=15)
    r.raise_for_status()
    lines = r.text.strip().split("\n")
    return "".join(lines[1:])


def make_yaml(seq, smi, cys_resi, warhead_atom):
    return f"""version: 1
sequences:
  - protein:
      id: A
      sequence: {seq}
  - ligand:
      id: B
      smiles: '{smi}'
constraints:
  - bond:
      atom1: [A, {cys_resi}, SG]
      atom2: [B, 1, {warhead_atom}]
"""


def main():
    OUT_ROOT.mkdir(parents=True, exist_ok=True)
    # 1. BMX metadata
    struct = pd.read_excel(SI2, sheet_name="Covalent_PDB_structures")
    bmx = struct[struct["protein_target"] == "BMX"].iloc[0]
    cys_resi = int(bmx["Cys_index"])
    seq = fetch_uniprot_sequence(bmx["uniprot_ID"])
    if seq[cys_resi - 1] != "C":
        print(f"WARN: residue {cys_resi} is '{seq[cys_resi-1]}', not C")
    print(f"BMX: uniprot={bmx['uniprot_ID']}, Cys{cys_resi}, len={len(seq)}")

    # 2. Load full BMX decoys
    decoys = pd.read_excel(SI4, sheet_name="BMX_decoys")
    print(f"Loaded {len(decoys)} BMX decoys from si_004")
    print(f"Unique parent actives: {decoys['active_compound_name'].nunique()}")
    print(f"Decoys per parent (sample):")
    print(decoys.groupby("active_compound_name").size().head())

    # 3. Stratified sample: ceil(100 / n_parents) per parent, then trim to N_DECOYS
    rng = np.random.default_rng(SEED)
    parents = decoys["active_compound_name"].unique()
    per_parent = int(np.ceil(N_DECOYS / len(parents)))
    print(f"\nSampling {per_parent} decoys per parent (~{per_parent * len(parents)} total)…")
    sampled = (
        decoys.groupby("active_compound_name", group_keys=False)
              .apply(lambda g: g.sample(min(per_parent, len(g)), random_state=SEED))
              .sample(N_DECOYS, random_state=SEED)
              .reset_index(drop=True)
    )
    print(f"Final stratified sample: {len(sampled)} decoys spanning {sampled['active_compound_name'].nunique()} parents")
    print(f"Decoys per parent in sample:")
    print(sampled.groupby("active_compound_name").size().describe())

    # 4. Write YAMLs
    manifest = []
    n_ok = 0; n_skip = 0
    for _, row in sampled.iterrows():
        smi = row["protomer_smiles"]
        atom_name = boltz_atom_name_for_warhead(smi)
        if atom_name is None: n_skip += 1; continue
        protomer_ind = int(row["protomer_ind"])
        name = f"BMX_dec_strat_{protomer_ind:05d}"
        yaml_text = make_yaml(seq, smi, cys_resi, atom_name)
        (OUT_ROOT / f"{name}.yaml").write_text(yaml_text)
        manifest.append({
            "name": name, "target": "BMX",
            "label": "dec", "is_active": 0,
            "smiles": smi,
            "uniprot": bmx["uniprot_ID"], "cys_resi": cys_resi,
            "warhead_atom_name": atom_name,
            "protomer_ind": protomer_ind,
            "parent_active": row["active_compound_name"],
        })
        n_ok += 1

    df = pd.DataFrame(manifest)
    df.to_csv(MANIFEST, index=False)
    print(f"\nWrote {n_ok} YAMLs to {OUT_ROOT}, skipped {n_skip} (no warhead match)")
    print(f"Manifest: {MANIFEST}")
    print(f"\nParent diversity: {df['parent_active'].nunique()} unique parents across {len(df)} decoys")


if __name__ == "__main__":
    main()
