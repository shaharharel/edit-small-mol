#!/usr/bin/env python3
"""Generate Boltz-2 input YAMLs for the 280 ZAP70 ChEMBL anchor molecules.

These are the SAME 280 mols FiLMDelta uses as anchors. They have experimental
pIC50 (4-9 range), so cofolding them gives us:
  1. The Boltz mPAE distribution for KNOWN ZAP70 binders → calibration target
  2. Reference poses we can compare against our 997 candidate cofolds

NOT applying any covalent bond constraint (most aren't covalent). Pure cofold
of sequence + ligand. Same cached MSA as the top1000 batch.
"""
from pathlib import Path
import re
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]

import pandas as pd

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

DOCK_CSV = PROJECT_ROOT / "data" / "docking_chembl_zap70" / "docking_results.csv"
OUT = PROJECT_ROOT / "experiments" / "boltz_inputs" / "zap70_anchors"
OUT.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(DOCK_CSV)
print(f"Loaded {len(df)} ChEMBL anchors")
print(df["pIC50_exp"].describe().to_string())

# Path to cached MSA on the A100 box (relative — we'll set absolute on remote)
SHARED_MSA = "/home/shaharh_quris_ai/zap70_msa.csv"

manifest = []
for i, r in df.iterrows():
    chembl_id = r["chembl_id"]
    smi = r["smiles"]
    pic = r["pIC50_exp"]
    name = f"{i:03d}_{chembl_id}"
    yaml = (
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {ZAP70_SEQ}\n"
        f"      msa: {SHARED_MSA}\n"
        "  - ligand:\n"
        "      id: B\n"
        f"      smiles: '{smi}'\n"
    )
    (OUT / f"{name}.yaml").write_text(yaml)
    manifest.append({
        "yaml_name": name,
        "chembl_id": chembl_id,
        "smiles": smi,
        "pIC50_exp": pic,
        "vina_score": r.get("vina_score"),
    })

mdf = pd.DataFrame(manifest)
mdf.to_csv(OUT.parent / "zap70_anchors_manifest.csv", index=False)
print(f"\nWrote {len(manifest)} YAMLs to {OUT}")
print(f"Manifest: {OUT.parent / 'zap70_anchors_manifest.csv'}")
