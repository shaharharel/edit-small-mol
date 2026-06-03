#!/usr/bin/env python3
"""Build Boltz-2 input YAMLs for the 1887 cascade-filter survivors.

Reads data/boltz_poses/survivors_1887_for_boltz.csv (exported from the
dashboard /api/filter endpoint with all default filters) and writes one
YAML per mol with the acrylamide β-CH2 → Cys346 SG covalent bond.

Outputs:
  experiments/boltz_inputs/survivors_1887__zap70_cys346/<row_id>_<short>.yaml
  experiments/boltz_inputs/survivors_1887__zap70_cys346/manifest.csv
"""
import sys, re
from pathlib import Path
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

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
    """Compute Boltz-2 atom name for the acrylamide β-CH2 (None if no warhead)."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    m = mol_h.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not m:
        return None
    return f"C{can[m[0][0]] + 1}"


def slugify(s):
    return re.sub(r"[^A-Za-z0-9]+", "_", str(s))[:24].strip("_")


SRC = PROJECT_ROOT / "data/boltz_poses/survivors_1887_for_boltz.csv"
OUT = PROJECT_ROOT / "experiments/boltz_inputs/survivors_1887__zap70_cys346"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(SRC)
print(f"Loaded {len(df)} survivors from {SRC.name}")

manifest, skipped, written = [], 0, 0
for _, r in df.iterrows():
    rid = int(r["row_id"]) if pd.notna(r["row_id"]) else None
    smi = str(r["SMILES"]).strip()
    if rid is None or not smi or smi == "nan":
        skipped += 1; continue
    atom_name = boltz_atom_name(smi)
    if atom_name is None:
        skipped += 1; continue
    slug = slugify(r.get("method", ""))
    name = f"{rid:06d}_{slug}_{rid}"
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
    (OUT / f"{name}.yaml").write_text(yaml)
    manifest.append({"row_id": rid, "yaml_name": name, "method": r.get("method", ""),
                     "smiles": smi, "pIC50_film": r.get("pIC50_film", ""),
                     "warhead_atom_name": atom_name})
    written += 1

pd.DataFrame(manifest).to_csv(OUT / "manifest.csv", index=False)
print(f"Wrote {written} YAMLs (skipped {skipped} without acrylamide / parse failures)")
print(f"Output: {OUT}")
