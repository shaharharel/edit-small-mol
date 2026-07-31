#!/usr/bin/env python3
"""Generate Boltz-2 YAMLs for the 401 unique Lingo3DMol mols (ZAP70 Cys346).

Reads ~/boltz_run/lingo3dmol_all_unique.csv and writes one YAML per uid into
~/boltz_run/yamls/. Each YAML constrains the acrylamide beta-CH2 to the Cys346 SG.

Records skips (mols without acrylamide) to ~/boltz_run/skipped.csv.
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

HOME = Path.home()
BOLTZ_RUN = HOME / "boltz_run"
INPUT_CSV = BOLTZ_RUN / "lingo3dmol_all_unique.csv"
OUT_DIR = BOLTZ_RUN / "yamls"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SKIPPED = BOLTZ_RUN / "skipped.csv"

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
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not matches:
        return None
    term_ch2_idx = matches[0][0]
    return f"C{can[term_ch2_idx] + 1}"


def main():
    rows = list(csv.DictReader(INPUT_CSV.open()))
    print(f"Loaded {len(rows)} mols")
    skipped = []
    wrote = 0
    for r in rows:
        uid = r["uid"]
        smi = r["canonical_smi"]
        atom_name = boltz_atom_name(smi)
        if atom_name is None:
            skipped.append({"uid": uid, "smi": smi, "reason": "no_acrylamide"})
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
        (OUT_DIR / f"{uid}.yaml").write_text(yaml)
        wrote += 1

    with SKIPPED.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=["uid", "smi", "reason"])
        w.writeheader()
        w.writerows(skipped)

    print(f"Wrote {wrote} YAMLs to {OUT_DIR}")
    print(f"Skipped {len(skipped)} (no acrylamide) -> {SKIPPED}")


if __name__ == "__main__":
    main()
