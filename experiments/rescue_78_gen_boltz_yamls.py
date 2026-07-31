"""Generate Boltz-2 YAMLs for the rescue-78 cohort.

Mirrors scripts/boltz_remote/gen_yamls.py and experiments/boltz_inputs_survivors_1887.py:
  - ZAP70 sequence (full)
  - Cys346 SG → acrylamide β-CH2 covalent constraint
  - rescue_row_id used as the YAML basename for traceability

Output:
  experiments/boltz_inputs/rescue_78__zap70_cys346/<rescue_row_id>.yaml
  experiments/boltz_inputs/rescue_78__zap70_cys346/manifest.csv
"""
from __future__ import annotations
import csv
import re
import sys
from pathlib import Path

from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
INPUT = ROOT / "data/tier4_scored/rescue_78_input.csv"
OUT_DIR = ROOT / "experiments/boltz_inputs/rescue_78__zap70_cys346"
OUT_DIR.mkdir(parents=True, exist_ok=True)
SKIPPED = OUT_DIR / "skipped.csv"

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
    import pandas as pd
    df = pd.read_csv(INPUT)
    print(f"Loaded {len(df)} rescue mols")
    manifest, skipped, wrote = [], [], 0
    for _, r in df.iterrows():
        rid = str(r["rescue_row_id"])
        smi = str(r["smiles"]).strip()
        if not rid or not smi or smi == "nan":
            skipped.append({"rescue_row_id": rid, "smi": smi, "reason": "empty"})
            continue
        atom_name = boltz_atom_name(smi)
        if atom_name is None:
            skipped.append({"rescue_row_id": rid, "smi": smi, "reason": "no_acrylamide"})
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
        (OUT_DIR / f"{rid}.yaml").write_text(yaml)
        manifest.append({"rescue_row_id": rid, "smiles": smi,
                         "warhead_atom_name": atom_name,
                         "method": r.get("method", ""), "_source_cohort": r.get("_source_cohort", "")})
        wrote += 1

    pd.DataFrame(manifest).to_csv(OUT_DIR / "manifest.csv", index=False)
    if skipped:
        with SKIPPED.open("w") as fh:
            w = csv.DictWriter(fh, fieldnames=["rescue_row_id", "smi", "reason"])
            w.writeheader(); w.writerows(skipped)
    print(f"Wrote {wrote} YAMLs to {OUT_DIR}")
    if skipped:
        print(f"Skipped {len(skipped)} → {SKIPPED}")


if __name__ == "__main__":
    main()
