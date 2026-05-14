#!/usr/bin/env python3
"""Generate Boltz-2 input YAML files for the top-10 Medchem ZAP70 candidates.

Each YAML contains:
  - ZAP70 full-length sequence (UniProt P43403)
  - Candidate ligand SMILES
  - Covalent bond constraint: ZAP70 Cys560 SG  <->  ligand acrylamide β-CH2

The atom name for the β-CH2 is computed per-ligand because Boltz-2 names atoms
as <element><canonical_rank+1> over the H-added molecule, and the rank depends
on the whole molecule.

Output: experiments/boltz_inputs/medchem_top10/<idx>_<name>.yaml
"""

from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[1]

from rdkit import Chem
from rdkit.Chem import AllChem

ZAP70_SEQ = (
    "MPDPAAHLPFFYGSISRAEAEEHLKLAGMADGLFLLRQCLRSLGGYVLSLVHDVRFHHFPIERQLNGTYAI"
    "AGGKAHCGPAELCEFYSRDPDGLPCNLRKPCNRPSGLEPQPGVFDCLRDAMVRDYVRQTWKLEGEALEQAI"
    "ISQAPQVEKLIATTAHERMPWYHSSLTREEAERKLYSGAQTDGKFLLRPRKEQGTYALSLIYGKTVYHYLI"
    "SQDKAGKYCIPEGTKFDTLWQLVEYLKLKADGLIYCLKEACPNSSASNASGAAAPTLPAHPSTLTHPQRRI"
    "DTLNSDGYTPEPARITSPDKPRPMPMDTSVYESPYSDPEELKDKKLFLKRDNLLIADIELGCGNFGSVRQG"
    "VYRMRKKQIDVAIKVLKQGTEKADTEEMMREAQIMHQLDNPYIVRLIGVCQAEALMLVMEMAGGGPLHKFL"
    "VGKREEIPVSNVAELLHQVSMGMKYLEEKNFVHRDLAARNVLLVNRHYAKISDFGLSKALGADDSYYTARS"
    "AGKWPLKWYAPECINFRKFSSRSDVWSYGVTMWEALSYGQKPYKKMKGPEVMAFIEQGKRMECPPECPPEL"
    "YALMSDCWIYKWEDRPDFLTVEQRMRACYYSLASKVEGPPGSTQKAEAACA"
)
TARGET_CYS = 560

TOP10 = [
    ("thiq_ring_expand",     "C=CC(=O)N1CCc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"),
    ("sulfonamide_linker",   "C=CC(=O)N1Cc2cccc(S(=O)(=O)Nc3cn(C(C)C)cn3)c2C1"),
    ("imidazole_C2_NH2",     "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)c(N)n3)c2C1"),
    ("aza_isoindoline",      "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2N1"),
    ("phenyl_CN",            "C=CC(=O)N1Cc2cc(C#N)cc(C(=O)Nc3cn(C(C)C)cn3)c2C1"),
    ("ethyl_spacer_amide",   "C=CC(=O)N1Cc2cccc(CCC(=O)Nc3cn(C(C)C)cn3)c2C1"),
    ("pyrazole_swap",        "C=CC(=O)N1Cc2cccc(C(=O)Nc3cnn(C(C)C)c3)c2C1"),
    ("imidazole_C2_Me",      "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)c(C)n3)c2C1"),
    ("methyl_spacer_amide",  "C=CC(=O)N1Cc2cccc(CC(=O)Nc3cn(C(C)C)cn3)c2C1"),
    ("imidazole_C2_OH",      "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)c(O)n3)c2C1"),
]

ACRYLAMIDE_PATTERN = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def boltz_atom_name_for_terminal_ch2(smi: str) -> str:
    """Compute Boltz-2's auto-assigned atom name for the acrylamide β-CH2.

    Boltz-2 names ligand atoms as <element_uppercase><canonical_rank+1> over
    the H-added molecule (see boltz/data/parse/schema.py:1249).
    """
    mol = AllChem.MolFromSmiles(smi)
    mol_h = AllChem.AddHs(mol)
    can_ranks = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACRYLAMIDE_PATTERN)
    if not matches:
        raise RuntimeError(f"No acrylamide warhead in SMILES: {smi}")
    term_ch2_idx = matches[0][0]   # first atom of pattern = terminal [CH2]
    rank = can_ranks[term_ch2_idx]
    return f"C{rank + 1}"


OUT_DIR = PROJECT_ROOT / "experiments" / "boltz_inputs" / "medchem_top10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"{'#':<3}{'name':<22}{'atom_name':<12}{'SMILES (truncated)':<60}")
print("-" * 100)
for idx, (name, smi) in enumerate(TOP10, 1):
    atom_name = boltz_atom_name_for_terminal_ch2(smi)
    yaml_path = OUT_DIR / f"{idx:02d}_{name}.yaml"
    yaml_text = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {ZAP70_SEQ}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{smi}'\n"
        "constraints:\n"
        "  - bond:\n"
        f"      atom1: [A, {TARGET_CYS}, SG]\n"
        f"      atom2: [B, 1, {atom_name}]\n"
    )
    yaml_path.write_text(yaml_text)
    print(f"{idx:<3}{name:<22}{atom_name:<12}{smi[:58]}")

print(f"\nWrote {len(TOP10)} YAML files to {OUT_DIR}")
