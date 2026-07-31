#!/usr/bin/env python3
"""Generate Boltz-2 YAMLs for the 3,597 cohort A relaxed mols (ZAP70 Cys346).

Adapted from gen_yamls.py to read row_id/smiles columns (vs uid/canonical_smi).
Writes YAMLs per chunk to data/boltz_results/cohort_3597_full/yamls/{chunk_N}/{row_id}.yaml.
"""
from __future__ import annotations
import csv
import sys
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import AllChem

REPO = Path(__file__).resolve().parents[2]
CHUNKS_DIR = REPO / "data" / "tier4_scored" / "boltz_cohort_chunks"
OUT_BASE = REPO / "data" / "boltz_results" / "cohort_3597_full" / "yamls"
SKIPPED_CSV = REPO / "data" / "boltz_results" / "cohort_3597_full" / "skipped.csv"

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
assert ZAP70_SEQ[TARGET_CYS - 1] == "C", "Cys346 mismatch"

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
    OUT_BASE.mkdir(parents=True, exist_ok=True)
    skipped = []
    total_wrote = 0
    total_seen = 0
    by_chunk = {}
    for chunk_idx in (1, 2, 3):
        chunk_csv = CHUNKS_DIR / f"chunk_{chunk_idx}_of_3.csv"
        chunk_out = OUT_BASE / f"chunk_{chunk_idx}"
        chunk_out.mkdir(parents=True, exist_ok=True)
        wrote = 0
        with chunk_csv.open() as fh:
            for r in csv.DictReader(fh):
                total_seen += 1
                uid = str(r["row_id"]).strip()
                smi = r["smiles"].strip()
                atom_name = boltz_atom_name(smi)
                if atom_name is None:
                    skipped.append({"row_id": uid, "smi": smi, "reason": "no_acrylamide", "chunk": chunk_idx})
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
                (chunk_out / f"{uid}.yaml").write_text(yaml)
                wrote += 1
        by_chunk[chunk_idx] = wrote
        total_wrote += wrote
        print(f"chunk_{chunk_idx}: wrote {wrote} YAMLs -> {chunk_out}")

    with SKIPPED_CSV.open("w") as fh:
        w = csv.DictWriter(fh, fieldnames=["row_id", "smi", "reason", "chunk"])
        w.writeheader()
        w.writerows(skipped)

    print(f"\nTOTAL seen={total_seen} wrote={total_wrote} skipped={len(skipped)}")
    print(f"Per-chunk: {by_chunk}")
    print(f"Skipped CSV: {SKIPPED_CSV}")


if __name__ == "__main__":
    main()
