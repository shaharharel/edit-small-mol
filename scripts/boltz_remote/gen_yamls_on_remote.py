#!/usr/bin/env python3
"""Generate Boltz-2 YAMLs on the REMOTE machine using its RDKit version.

Reads CSV from $REMOTE_CHUNK_CSV (e.g. ~/chunk_1_of_3.csv), writes YAMLs
to ~/boltz3597/yamls/<row_id>.yaml. CSV columns: row_id, smiles, method.
"""
from __future__ import annotations
import csv
import os
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

HOME = Path.home()
RUN_DIR = HOME / "boltz3597"
YAML_DIR = RUN_DIR / "yamls"
YAML_DIR.mkdir(parents=True, exist_ok=True)
SKIPPED_CSV = RUN_DIR / "skipped.csv"

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
TARGET_CYS = 346
assert ZAP70_SEQ[TARGET_CYS - 1] == "C"

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def boltz_atom_name(smi: str):
    """Use IDENTICAL logic to Boltz's schema.py atom naming."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol))
    matches = mol.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not matches:
        return None
    term_ch2_idx = matches[0][0]  # First CH2 in the SMARTS match
    elem = mol.GetAtomWithIdx(term_ch2_idx).GetSymbol().upper()
    return f"{elem}{can[term_ch2_idx] + 1}"


def main():
    csv_path = Path(sys.argv[1]) if len(sys.argv) > 1 else HOME / "cohort.csv"
    skipped = []
    wrote = 0
    seen = 0
    with csv_path.open() as fh:
        for r in csv.DictReader(fh):
            seen += 1
            uid = str(r["row_id"]).strip()
            smi = r["smiles"].strip()
            atom_name = boltz_atom_name(smi)
            if atom_name is None:
                skipped.append({"row_id": uid, "smi": smi, "reason": "no_acrylamide"})
                continue
            yaml = (
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
            (YAML_DIR / f"{uid}.yaml").write_text(yaml)
            wrote += 1
    with SKIPPED_CSV.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=["row_id", "smi", "reason"])
        w.writeheader()
        w.writerows(skipped)
    print(f"seen={seen} wrote={wrote} skipped={len(skipped)}  YAML_DIR={YAML_DIR}")


if __name__ == "__main__":
    main()
