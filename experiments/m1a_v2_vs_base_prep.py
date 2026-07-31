#!/usr/bin/env python3
"""Phase 1 prep for M1a v2 vs base mol2mol Boltz head-to-head.

Subsamples 500 Mol1-anchored mols from each cohort, flags acrylamide retention,
and emits per-mol Boltz cofold YAMLs (with Cys346 Sγ — ligand-acrylamide Cβ
covalent bond at 1.85 Å) for the acrylamide-bearing subset.

Inputs:
  data/m1a_v2_ablation/cohort_A.csv           (v2 cohort, 5000 mols)
  experiments/exp_covft_value/samples_base.csv (base prior, 10000 mols)

Outputs:
  data/m1a_v2_vs_base/manifest.csv             (1000-mol manifest w/ cohort, has_acr, cofold_yaml_name)
  data/m1a_v2_vs_base/yamls/*.yaml             (per-mol Boltz YAMLs, only for acrylamide-bearing)
  data/m1a_v2_vs_base/summary_phase1.json      (counts + acrylamide retention)
"""
from __future__ import annotations
import json
import re
import sys
import time
from pathlib import Path

from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import AllChem

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
V2_CSV = PROJECT_ROOT / "data/m1a_v2_ablation/cohort_A.csv"
BASE_CSV = PROJECT_ROOT / "experiments/exp_covft_value/samples_base.csv"
OUT_DIR = PROJECT_ROOT / "data/m1a_v2_vs_base"
YAML_DIR = OUT_DIR / "yamls"
YAML_DIR.mkdir(parents=True, exist_ok=True)

N_PER_COHORT = 500
SEED = 42

# Full ZAP70 sequence (UniProt P43403; cached MSA on A100 matches this exactly).
# Cys346 is at absolute position 346 (the M1a v2 covalent target).
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
assert ZAP70_SEQ[TARGET_CYS - 1] == "C", f"pos {TARGET_CYS} is {ZAP70_SEQ[TARGET_CYS-1]!r}"
# MSA path on the A100 — referenced inside each YAML
SHARED_MSA = "/home/shaharh_quris_ai/zap70_msa.csv"

ACR = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def boltz_atom_name(smi: str):
    """Compute Boltz-2 atom name for the acrylamide β-CH2.

    Returns (atom_name, atom_idx) or (None, None) if no warhead.
    Boltz names atoms as <element><canonical_rank+1> over the H-added molecule.
    """
    mol = AllChem.MolFromSmiles(smi)
    if mol is None:
        return None, None
    mol_h = AllChem.AddHs(mol)
    can = list(AllChem.CanonicalRankAtoms(mol_h))
    matches = mol_h.GetSubstructMatches(ACR)
    if not matches:
        return None, None
    term_ch2_idx = matches[0][0]
    return f"C{can[term_ch2_idx] + 1}", term_ch2_idx


def has_acrylamide(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    return m is not None and bool(m.GetSubstructMatches(ACR))


def disconnected(smi: str) -> bool:
    return "." in str(smi)


def sample_cohort(df: pd.DataFrame, smiles_col: str, n: int,
                   cohort_label: str, rng: np.random.Generator) -> pd.DataFrame:
    """Dedup SMILES, drop disconnected/invalid, then random-sample n. If <n
    unique mols available, take all."""
    df = df.copy()
    df["smiles"] = df[smiles_col].astype(str)
    df = df[~df["smiles"].apply(disconnected)]
    # Drop unparseable mols
    df["valid"] = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[df["valid"]].drop(columns="valid")
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    print(f"  [{cohort_label}] unique valid mols: {len(df)}")
    if len(df) > n:
        df = df.sample(n=n, random_state=rng.integers(0, 2**31)).reset_index(drop=True)
    df["cohort"] = cohort_label
    return df[["smiles", "cohort"]]


def main():
    rng = np.random.default_rng(SEED)

    # ---- Load + sample ----
    print(f"[1/3] Sampling {N_PER_COHORT} mols per cohort ...")
    v2_raw = pd.read_csv(V2_CSV)
    base_raw = pd.read_csv(BASE_CSV)
    print(f"  v2 cohort A raw rows: {len(v2_raw)}")
    print(f"  base prior raw rows: {len(base_raw)}")

    v2 = sample_cohort(v2_raw, "SMILES", N_PER_COHORT, "v2", rng)
    base = sample_cohort(base_raw, "SMILES", N_PER_COHORT, "base", rng)
    # Augment base with every unique acrylamide-bearing mol from the full 10K
    # base prior cohort (improves statistical power; base prior's natural acr
    # rate is so low — 1.4% — that 500 random mols only yield ~7 cofolds).
    base_all = base_raw.copy()
    base_all["smiles"] = base_all["SMILES"].astype(str)
    base_all = base_all[~base_all["smiles"].apply(disconnected)]
    base_all = base_all[base_all["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)]
    base_all = base_all.drop_duplicates(subset="smiles")
    base_all["acr"] = base_all["smiles"].apply(has_acrylamide)
    base_acr_extra = base_all[base_all.acr][["smiles"]].copy()
    base_acr_extra["cohort"] = "base"
    # Drop any already in the random base 500
    existing = set(base["smiles"])
    base_acr_extra = base_acr_extra[~base_acr_extra["smiles"].isin(existing)].reset_index(drop=True)
    print(f"  [base] extra acrylamide mols added beyond random 500: {len(base_acr_extra)}")
    manifest = pd.concat([v2, base, base_acr_extra], ignore_index=True)
    print(f"  combined manifest: {len(manifest)} mols "
          f"(v2={len(v2)}, base_random={len(base)}, base_acr_extra={len(base_acr_extra)})")

    # ---- Annotate acrylamide ----
    print(f"[2/3] Annotating acrylamide retention + Boltz atom names ...")
    rows = []
    for i, r in manifest.iterrows():
        smi = r["smiles"]
        atom_name, atom_idx = boltz_atom_name(smi)
        rows.append({
            "mol_id": f"{r['cohort']}_{i:04d}",
            "cohort": r["cohort"],
            "smiles": smi,
            "has_acrylamide": atom_name is not None,
            "warhead_atom_name": atom_name,
            "warhead_atom_idx": atom_idx,
        })
    mdf = pd.DataFrame(rows)
    n_v2_acr = mdf[(mdf.cohort == "v2") & mdf.has_acrylamide].shape[0]
    n_v2_total = mdf[mdf.cohort == "v2"].shape[0]
    n_base_total = mdf[mdf.cohort == "base"].shape[0]
    n_base_acr = mdf[(mdf.cohort == "base") & mdf.has_acrylamide].shape[0]
    print(f"  v2:   {n_v2_acr}/{n_v2_total} ({100*n_v2_acr/n_v2_total:.1f}%) acrylamide")
    print(f"  base: {n_base_acr}/{n_base_total} ({100*n_base_acr/n_base_total:.1f}%) acrylamide  "
          f"[NOTE: base augmented w/ all unique base-acr mols → cofold-statistical-power]")

    # ---- Emit YAMLs (acrylamide-only) ----
    print(f"[3/3] Writing per-mol Boltz YAMLs to {YAML_DIR} ...")
    n_written = 0
    for _, r in mdf.iterrows():
        if not r["has_acrylamide"]:
            continue
        name = r["mol_id"]
        yaml = (
            "version: 1\n"
            "sequences:\n"
            "  - protein:\n"
            "      id: A\n"
            f"      sequence: {ZAP70_SEQ}\n"
            f"      msa: {SHARED_MSA}\n"
            "  - ligand:\n"
            "      id: B\n"
            f"      smiles: '{r['smiles']}'\n"
            "constraints:\n"
            "  - bond:\n"
            f"      atom1: [A, {TARGET_CYS}, SG]\n"
            f"      atom2: [B, 1, {r['warhead_atom_name']}]\n"
        )
        (YAML_DIR / f"{name}.yaml").write_text(yaml)
        n_written += 1
    mdf["yaml_path"] = mdf.apply(
        lambda r: f"yamls/{r['mol_id']}.yaml" if r.has_acrylamide else None, axis=1)
    mdf.to_csv(OUT_DIR / "manifest.csv", index=False)

    summary = {
        "phase": "phase1_prep_complete",
        "timestamp": int(time.time()),
        "n_per_cohort_target": N_PER_COHORT,
        "n_v2_total": int(n_v2_total),
        "n_base_total": int(n_base_total),
        "n_v2_acrylamide": int(n_v2_acr),
        "n_base_acrylamide": int(n_base_acr),
        "v2_acrylamide_rate_pct": float(100 * n_v2_acr / n_v2_total),
        "base_acrylamide_rate_pct_random_500": float(
            100 * mdf[(mdf.cohort == "base") & (mdf.index < N_PER_COHORT + n_v2_total)].has_acrylamide.mean()
        ),
        "base_acrylamide_rate_pct_section31": 1.4,  # ground truth from §3.1
        "base_n_extra_acrylamide_pulled_for_power": int(n_base_total - N_PER_COHORT),
        "n_yamls_written": int(n_written),
        "yaml_dir": str(YAML_DIR),
        "manifest_path": str(OUT_DIR / "manifest.csv"),
    }
    (OUT_DIR / "summary_phase1.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPhase 1 done: wrote {n_written} YAMLs, manifest at {OUT_DIR/'manifest.csv'}")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
