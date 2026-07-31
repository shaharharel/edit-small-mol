#!/usr/bin/env python3
"""Phase 1 prep for M1a v2 vs covFT mol2mol Boltz head-to-head.

Both priors are warhead-rich (~94% acrylamide retention), so this is a fair
500-vs-500 cofold comparison of architectural pose conditioning (M1a v2) vs
SMILES-only covalent FT (covFT) on Boltz-validated geometry.

Inputs:
  data/m1a_v2_ablation/cohort_A.csv           (v2 cohort,  5000 mols)
  experiments/exp_covft_value/samples_covft.csv (covFT cohort, 10000 mols)

Outputs:
  data/m1a_v2_vs_covft/manifest.csv             (1000-mol manifest)
  data/m1a_v2_vs_covft/yamls/*.yaml             (per-mol Boltz YAMLs, acr only)
  data/m1a_v2_vs_covft/summary_phase1.json
"""
from __future__ import annotations
import json
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
V2_CSV    = PROJECT_ROOT / "data/m1a_v2_ablation/cohort_A.csv"
COVFT_CSV = PROJECT_ROOT / "experiments/exp_covft_value/samples_covft.csv"
OUT_DIR   = PROJECT_ROOT / "data/m1a_v2_vs_covft"
YAML_DIR  = OUT_DIR / "yamls"
YAML_DIR.mkdir(parents=True, exist_ok=True)

N_PER_COHORT = 500
SEED = 42

# Full ZAP70 sequence (UniProt P43403, cached MSA on A100 matches this exactly)
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
SHARED_MSA = "/home/shaharh_quris_ai/zap70_msa.csv"

ACR = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def boltz_atom_name(smi: str):
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
    m = Chem.MolFromSmiles(str(smi))
    return m is not None and bool(m.GetSubstructMatches(ACR))


def disconnected(smi) -> bool:
    return "." in str(smi)


def sample_cohort(df: pd.DataFrame, smiles_col: str, n: int,
                   cohort_label: str, rng: np.random.Generator) -> pd.DataFrame:
    """Dedup SMILES, drop disconnected/invalid, then random-sample n.
    NOTE: we sample from VALID UNIQUE mols (not raw) so each cohort delivers
    n distinct mols, matching how the §3.1 quality metrics are computed."""
    df = df.copy()
    df["smiles"] = df[smiles_col].astype(str)
    df = df[~df["smiles"].apply(disconnected)]
    df["valid"] = df["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)
    df = df[df["valid"]].drop(columns="valid")
    df = df.drop_duplicates(subset="smiles").reset_index(drop=True)
    print(f"  [{cohort_label}] unique valid mols available: {len(df)}")
    if len(df) > n:
        df = df.sample(n=n, random_state=int(rng.integers(0, 2**31))).reset_index(drop=True)
    else:
        print(f"  WARN: {cohort_label} only has {len(df)} unique mols (< {n} target)")
    df["cohort"] = cohort_label
    return df[["smiles", "cohort"]]


def main():
    rng = np.random.default_rng(SEED)

    print(f"[1/3] Sampling {N_PER_COHORT} mols per cohort ...")
    v2_raw    = pd.read_csv(V2_CSV)
    covft_raw = pd.read_csv(COVFT_CSV)
    print(f"  v2 cohort A raw rows: {len(v2_raw)}")
    print(f"  covFT raw rows: {len(covft_raw)}")

    v2    = sample_cohort(v2_raw,    "SMILES", N_PER_COHORT, "v2",    rng)
    covft = sample_cohort(covft_raw, "SMILES", N_PER_COHORT, "covft", rng)
    manifest = pd.concat([v2, covft], ignore_index=True)
    print(f"  combined manifest: {len(manifest)} mols "
          f"(v2={len(v2)}, covft={len(covft)})")

    print(f"[2/3] Annotating acrylamide retention + Boltz atom names ...")
    rows = []
    # Track per-cohort indices for stable IDs
    counters = {"v2": 0, "covft": 0}
    for _, r in manifest.iterrows():
        smi = r["smiles"]; cohort = r["cohort"]
        idx = counters[cohort]; counters[cohort] += 1
        atom_name, atom_idx = boltz_atom_name(smi)
        rows.append({
            "mol_id": f"{cohort}_{idx:04d}",
            "cohort": cohort,
            "smiles": smi,
            "has_acrylamide": atom_name is not None,
            "warhead_atom_name": atom_name,
            "warhead_atom_idx": atom_idx,
        })
    mdf = pd.DataFrame(rows)
    n_v2_acr    = int(((mdf.cohort == "v2")    & mdf.has_acrylamide).sum())
    n_covft_acr = int(((mdf.cohort == "covft") & mdf.has_acrylamide).sum())
    n_v2    = int((mdf.cohort == "v2").sum())
    n_covft = int((mdf.cohort == "covft").sum())
    print(f"  v2:    {n_v2_acr}/{n_v2} ({100*n_v2_acr/max(n_v2,1):.1f}%) acrylamide")
    print(f"  covft: {n_covft_acr}/{n_covft} ({100*n_covft_acr/max(n_covft,1):.1f}%) acrylamide")

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
        "n_v2_total": n_v2,
        "n_covft_total": n_covft,
        "n_v2_acrylamide": n_v2_acr,
        "n_covft_acrylamide": n_covft_acr,
        "v2_acrylamide_rate_pct":    float(100 * n_v2_acr    / max(n_v2,    1)),
        "covft_acrylamide_rate_pct": float(100 * n_covft_acr / max(n_covft, 1)),
        "n_yamls_written": int(n_written),
        "yaml_dir": str(YAML_DIR),
        "manifest_path": str(OUT_DIR / "manifest.csv"),
    }
    (OUT_DIR / "summary_phase1.json").write_text(json.dumps(summary, indent=2))
    print(f"\nPhase 1 done: wrote {n_written} YAMLs")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
