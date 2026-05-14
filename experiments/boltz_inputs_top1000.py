#!/usr/bin/env python3
"""Build Boltz-2 input YAMLs for the top 1000 MW<400 candidates.

Selection:
  - All tiers, all methods (after backend filtering: warhead intact, no
    disconnected SMILES, drop Tier 4 / Methods A-B).
  - MW < 400.
  - Sort by pIC50_mean (3-seed ensemble) if present, else pIC50_method.
  - Take top 1000.

Each ligand gets its correct Boltz atom name for the acrylamide β-CH2 — Boltz
names atoms as <element><canonical_rank+1> over the H-added molecule.

Outputs:
  experiments/boltz_inputs/top1000/<row_id>_<short>.yaml  (per-mol YAML)
  experiments/boltz_inputs/top1000/manifest.csv           (row_id -> name -> SMILES)
"""
import sys
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

from anchordiff.config import CURRENT_TARGET, ZAP70_CYS346, assert_target

# Hard fail if anyone ever tries to run this with the wrong cysteine target.
assert_target(ZAP70_CYS346)

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
# Sanity check: the residue at TARGET_CYS must actually be Cys
TARGET_CYS = ZAP70_CYS346.cys_residue
assert ZAP70_SEQ[TARGET_CYS - 1] == "C", \
    f"ZAP70 position {TARGET_CYS} is {ZAP70_SEQ[TARGET_CYS-1]!r}, expected 'C'"

ACRYLAMIDE_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def boltz_atom_name(smi: str):
    """Compute Boltz-2 atom name for the acrylamide β-CH2 (returns None if no warhead)."""
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACRYLAMIDE_SMARTS)
    if not matches:
        return None, None
    term_ch2_idx = matches[0][0]
    return f"C{can[term_ch2_idx] + 1}", term_ch2_idx


def slugify(name: str) -> str:
    """Filesystem-safe short name."""
    return re.sub(r"[^A-Za-z0-9]+", "_", name)[:24].strip("_")


# ── Load + apply backend filters (mirror backend.py exactly) ──────────────
DF = pd.read_csv(PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv")
DF["method"] = DF["method"].replace({
    "Tier 1 — Med-Chem Playbook (rule-based)": "Medchem Rules",
    "Tier 1.5 — Warhead Controls + Med-Chem Tricks": "Medchem Rules",
    "Tier 2 — Fragment Replacement (curated 204)": "Amine Replacements",
    "Tier 2 SCALED — Fragment Replacement (498K)": "Amine Replacements",
    "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)": "Amine Replacements",
})
DROPPED = [
    "Tier 4 — De Novo unconstrained",
    "Tier 4 — Mol2Mol unconstrained",
    "Method A — De Novo FiLMDelta-driven",
    "Method B — Mol2Mol FiLMDelta-driven",
]
DF = DF[~DF["method"].isin(DROPPED)].reset_index(drop=True)
DF = DF[DF["warhead_intact"] == True].reset_index(drop=True)
DF = DF[~DF["smiles"].astype(str).str.contains(".", regex=False, na=False)].reset_index(drop=True)
DF["row_id"] = DF.index

# ── Apply MW filter + ranking ─────────────────────────────────────────────
# Selection rule (updated 2026-05-08):
#   - ALL Medchem Rules (MW<400) included automatically — full medchem coverage
#   - Remaining slots filled by top non-Medchem candidates, ranked by
#     pIC50_mean (3-seed ensemble) if present, else pIC50_method.
# This guarantees Medchem representation since their pIC50 caps at ~7.14
# while other tiers reach ~7.97 — pure ranking would drop all Medchem.
TARGET_TOTAL = 1000

mw_sub = DF[DF["MW"] < 400].copy()
mw_sub["rank_score"] = mw_sub["pIC50_mean"].fillna(mw_sub["pIC50_method"])

medchem = mw_sub[mw_sub["method"] == "Medchem Rules"].copy()
others  = mw_sub[mw_sub["method"] != "Medchem Rules"].copy()
n_medchem = len(medchem)
n_others_quota = TARGET_TOTAL - n_medchem
print(f"Medchem (auto-included): {n_medchem}")
print(f"Other-tier quota: {n_others_quota}")
top_others = others.nlargest(n_others_quota, "rank_score")
top1000 = pd.concat([medchem, top_others], ignore_index=True)
top1000 = top1000.sort_values("rank_score", ascending=False).reset_index(drop=True)
print(f"Selected {len(top1000)} candidates total")
print("Method breakdown:")
print(top1000.groupby("method").agg(
    n=("smiles", "count"),
    used_ensemble=("pIC50_mean", lambda s: s.notna().sum()),
).to_string())
print(f"rank_score range: {top1000['rank_score'].min():.3f} – {top1000['rank_score'].max():.3f}")

# ── Generate YAMLs + manifest ─────────────────────────────────────────────
OUT = PROJECT_ROOT / "experiments" / "boltz_inputs" / "top1000__zap70_cys346"
OUT.mkdir(parents=True, exist_ok=True)

manifest_rows = []
skipped = 0
for _, r in top1000.iterrows():
    rid = int(r["row_id"])
    smi = r["smiles"]
    atom_name, _ = boltz_atom_name(smi)
    if atom_name is None:
        skipped += 1
        continue
    slug = slugify(r["method"]) + "_" + str(rid)
    name = f"{rid:06d}_{slug}"
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
    manifest_rows.append({
        "row_id": rid,
        "yaml_name": name,
        "method": r["method"],
        "smiles": smi,
        "MW": r["MW"],
        "pIC50_method": r["pIC50_method"],
        "pIC50_mean": r["pIC50_mean"],
        "rank_score": r["rank_score"],
        "Tc_to_Mol1": r["Tc_to_Mol1"],
        "max_Tc_train": r["max_Tc_train"],
        "warhead_atom_name": atom_name,
    })

mdf = pd.DataFrame(manifest_rows)
mdf.to_csv(OUT / "manifest.csv", index=False)
print(f"\nWrote {len(mdf)} YAMLs (+ skipped {skipped} without acrylamide)")
print(f"Output: {OUT}")
