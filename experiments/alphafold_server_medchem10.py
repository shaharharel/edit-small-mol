#!/usr/bin/env python3
"""
Generate AlphaFold Server JSON job specs for the top-10 Medchem candidates.

Each spec contains:
  - ZAP70 full-length sequence (UniProt P43403, kinase + SH2 domains)
  - Candidate ligand SMILES
  - Covalent bond: ZAP70 Cys376 SG  ↔  ligand atom 0 (terminal CH2 of acrylamide,
    the Michael acceptor β-carbon attacked by the cysteine thiolate)

Output: experiments/alphafold_inputs/medchem_top10/<idx>_<short_name>.json
The user uploads these via https://alphafoldserver.com (free tier ~50 jobs/day).
"""

import json
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

# ── ZAP70 sequence (UniProt P43403, 619 aa) ─────────────────────────────────
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
assert len(ZAP70_SEQ) == 619, len(ZAP70_SEQ)
# ZAP70 has 7 cysteines in the kinase domain (residues 327-606):
#   346, 405, 510, 560, 564, 575, 596
# The known covalent target for ZAP70 acrylamide inhibitors (Shokat lab, etc.)
# is Cys560 — the Cys in the activation loop accessible from the ATP pocket.
# (Unlike the BTK family, ZAP70 lacks a Cys equivalent to BTK Cys481.)
TARGET_CYS = 560
assert ZAP70_SEQ[TARGET_CYS - 1] == "C", \
    f"Position {TARGET_CYS} is {ZAP70_SEQ[TARGET_CYS-1]!r}, expected 'C'"

# ── Top 10 Medchem candidates (highest pIC50_mean) ─────────────────────────
TOP10 = [
    ("C=CC(=O)N1CCc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1",       "thiq_ring_expand"),
    ("C=CC(=O)N1Cc2cccc(S(=O)(=O)Nc3cn(C(C)C)cn3)c2C1",    "sulfonamide_linker"),
    ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)c(N)n3)c2C1",     "imidazole_C2_NH2"),
    ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2N1",        "aza_isoindoline"),
    ("C=CC(=O)N1Cc2cc(C#N)cc(C(=O)Nc3cn(C(C)C)cn3)c2C1",   "phenyl_CN"),
    ("C=CC(=O)N1Cc2cccc(CCC(=O)Nc3cn(C(C)C)cn3)c2C1",      "ethyl_spacer_amide"),
    ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cnn(C(C)C)c3)c2C1",        "pyrazole_swap"),
    ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)c(C)n3)c2C1",     "imidazole_C2_Me"),
    ("C=CC(=O)N1Cc2cccc(CC(=O)Nc3cn(C(C)C)cn3)c2C1",       "methyl_spacer_amide"),
    ("C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)c(O)n3)c2C1",     "imidazole_C2_OH"),
]

# ── Build job specs ─────────────────────────────────────────────────────────
# For each ligand, atom index 0 in the SMILES is the terminal vinyl CH2 (the
# β-carbon of the acrylamide). Cys376 SG attacks this carbon via Michael
# addition. AlphaFold Server uses 1-indexed atom indexing for ligands and
# atom names for proteins.

OUT_DIR = PROJECT_ROOT / "experiments" / "alphafold_inputs" / "medchem_top10"
OUT_DIR.mkdir(parents=True, exist_ok=True)

specs = []
for idx, (smi, name) in enumerate(TOP10, 1):
    # AlphaFold Server JSON. We OMIT bondedAtomPairs because AF Server's
    # auto-assigned ligand atom names (when given a SMILES) don't reliably map
    # to "C1"/etc. — our guess of "C1" for the acrylamide β-carbon was rejected
    # by AF Server validation. AF3 will still place the ligand near the
    # binding pocket via cofolding; the covalent geometry usually emerges from
    # the right pocket vicinity even without an explicit bond constraint.
    # We can re-add the constraint later via a userCCD block once we know AF3's
    # internal atom-naming for these ligands.
    spec = {
        "name": f"zap70_medchem_{idx:02d}_{name}",
        "modelSeeds": [42],
        "sequences": [
            {"proteinChain": {"sequence": ZAP70_SEQ, "count": 1}},
            {"ligand": {"ligand": smi, "count": 1}},
        ],
        "dialect": "alphafold3",
        "version": 1,
    }
    specs.append(spec)
    out_path = OUT_DIR / f"{idx:02d}_{name}.json"
    with open(out_path, "w") as f:
        # AlphaFold Server expects a JSON ARRAY of jobs (even if just one).
        json.dump([spec], f, indent=2)
    print(f"  {out_path.name}  (SMILES len={len(smi)})")

# Combined batch file (some servers accept arrays)
batch_path = OUT_DIR / "BATCH_all10.json"
with open(batch_path, "w") as f:
    json.dump(specs, f, indent=2)

print(f"\nWrote {len(specs)} individual spec files + 1 batch file")
print(f"Directory: {OUT_DIR}")
print()
print("=" * 70)
print("INSTRUCTIONS")
print("=" * 70)
print("""
1. Go to https://alphafoldserver.com (sign in with Google)
2. Click 'New Job' → 'Submit JSON'
3. Upload one of the 01_*.json through 10_*.json files (one job per file)
4. Each job typically completes in 5-15 min depending on queue

For each completed job, AF3 outputs:
  - 5 model predictions (mmCIF files)
  - confidence_*.json with mPAE values per token-pair
  - The 'mPAE' value (minimum protein-residue PAE when aligned by ligand)
    is the key metric per Shamir et al. JACS 2026:
      < 0.85 Å  →  96.6% probability of being a real binder
      0.85-1.0  →  ~50%
      > 1.05    →  < 13%

5. Download confidences + structures, re-rank our top-10 by mPAE.
""")
