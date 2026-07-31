#!/usr/bin/env python3
"""Prepare cohort SDFs for ZAP70 covalent-pocket docking comparison.

Inputs:
- Lingo3DMol cohort SDFs (Pocket+covalent-aware, 3D poses already present)
- REINVENT4 baseline CSVs (SMILES only — need ETKDG embedding)
- aichem Amine-Replacement top-50K CSV (SMILES only — need ETKDG embedding)

Outputs:
- One SDF per cohort under data/cohort_comparison/cohorts/<cohort>/input.sdf
- A manifest at results/paper_evaluation/cohort_comparison/cohort_manifest.json
  with the cohort name, sdf path, N mols, source type, etc.

Downsample to 500 per cohort. For baselines, prefer rows with acrylamide warhead
("warhead gate"). For Amine Replacements: top-K by FiLM pIC50.

Skip cohorts whose SDF/CSV is missing or empty.
"""
import json
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import cpu_count
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors

RDLogger.logger().setLevel(RDLogger.ERROR)
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).parent.parent
OUT_DIR = PROJECT_ROOT / "data" / "cohort_comparison" / "cohorts"
OUT_DIR.mkdir(parents=True, exist_ok=True)
MANIFEST_DIR = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison"
MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
ACRYLAMIDE_PATT = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
TARGET_N = 500
RANDOM_SEED = 42

# Cohort source paths
LINGO_COHORTS = {
    # name: (sdf_path, group)
    "L0_vanilla": (PROJECT_ROOT / "data/lingo3dmol_L2_inpaint/samples.sdf", "lingo_pcov"),
    "L_locked": (PROJECT_ROOT / "data/lingo3dmol_BD_corrected_t4/L2_ext_L_BD_N500/samples.sdf", "lingo_pcov"),
    "H1": (PROJECT_ROOT / "data/lingo3dmol_BD_corrected_H1_N500_v3/samples.sdf", "lingo_pcov"),
    "H2": (PROJECT_ROOT / "data/lingo3dmol_L2_extended_H2_N500/samples.sdf", "lingo_pcov"),
    "H3": (PROJECT_ROOT / "data/lingo3dmol_BD_corrected_t4/L2_ext_H3_BD_N500/samples.sdf", "lingo_pcov"),
    "C1": (PROJECT_ROOT / "data/lingo3dmol_BD_corrected_t4/L2_scaffold_C1_BD_N500/samples.sdf", "lingo_pcov"),
    "C5": (PROJECT_ROOT / "data/lingo3dmol_L2_scaffold_C5_N500/samples.sdf", "lingo_pcov"),
    "L1_FT_H2": (PROJECT_ROOT / "data/lingo3dmol_L1_FT_H2_N500/samples.sdf", "lingo_pcov"),
}

# Baseline SMILES CSVs
BASELINE_CSVS = {
    "DeNovo_warhead_gate": (
        PROJECT_ROOT / "results/paper_evaluation/reinvent4/reinvent/reinvent_denovo_1.csv",
        "SMILES",
        "FiLMDelta pIC50 (raw)",
        "seqonly",
    ),
    "Mol2Mol_warhead_gate": (
        PROJECT_ROOT / "results/paper_evaluation/reinvent4/mol2mol/mol2mol_optimize_1.csv",
        "SMILES",
        "FiLMDelta pIC50 (raw)",
        "seqonly",
    ),
    "LibInvent_locked": (
        # 2026-06-01: corrected source path. The previous path
        # results/paper_evaluation/reinvent4/libinvent/libinvent_rgroup_1.csv was
        # an older LibInvent run with scaffold "C1(O)CN([*])CC1" (pyrrolidinol, no
        # warhead). The real Tier 3 v3 "LibInvent_locked" run uses the acrylamide
        # scaffold "C=CC(=O)N1Cc2cccc(C(=O)N[*:1])c2C1" — sourced from
        # aigpu_overnight. Future prep runs will embed acrylamide-bearing mols
        # so warhead_largest_frag should be ~100% (matches dashboard 99.96%).
        PROJECT_ROOT / "results/paper_evaluation/aigpu_overnight/libinvent_locked/libinvent_locked_1.csv",
        "SMILES",
        "FiLMDelta pIC50 (raw)",
        "seqonly",
    ),
    "Amine_Replacements": (
        PROJECT_ROOT / "results/paper_evaluation/aichem_tier2_scaled/products_top50k.csv",
        "smiles",
        "pIC50_film",
        "seqonly",
    ),
}


def has_acrylamide(smi: str) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        return mol.HasSubstructMatch(ACRYLAMIDE_PATT)
    except Exception:
        return False


def mw_in_range(smi: str, lo=200, hi=700) -> bool:
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        mw = Descriptors.MolWt(mol)
        return lo <= mw <= hi
    except Exception:
        return False


def acrylamide_on_largest_frag(smi: str) -> bool:
    """Per the inpaint_v2_warhead_retention_caveat: only count warhead if on the
    LARGEST connected fragment."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return False
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return False
        largest = max(frags, key=lambda m: m.GetNumHeavyAtoms())
        return largest.HasSubstructMatch(ACRYLAMIDE_PATT)
    except Exception:
        return False


def embed_smiles_to_mol(smi: str, seed: int = 42):
    """SMILES -> 3D embed -> RDKit mol with coords."""
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            return None
        mol = Chem.AddHs(mol)
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        res = AllChem.EmbedMolecule(mol, params)
        if res == -1:
            params.useRandomCoords = True
            res = AllChem.EmbedMolecule(mol, params)
            if res == -1:
                return None
        try:
            AllChem.MMFFOptimizeMolecule(mol, maxIters=500)
        except Exception:
            try:
                AllChem.UFFOptimizeMolecule(mol, maxIters=500)
            except Exception:
                pass
        # Remove Hs for cleaner SDF (we'll re-add at meeko step)
        mol = Chem.RemoveHs(mol)
        return mol
    except Exception:
        return None


def _embed_one(args):
    idx, smi, seed = args
    mol = embed_smiles_to_mol(smi, seed=seed)
    if mol is None:
        return (idx, None)
    mb = Chem.MolToMolBlock(mol)
    return (idx, mb)


def prep_lingo_cohort(name: str, src: Path, target_n: int = TARGET_N):
    """Just copy the SDF (3D coords already present), downsample to target_n."""
    if not src.exists():
        return None
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = out_dir / "input.sdf"

    supp = Chem.SDMolSupplier(str(src), removeHs=False, sanitize=True)
    mols = []
    for i, mol in enumerate(supp):
        if mol is None:
            continue
        smi = Chem.MolToSmiles(mol)
        if not smi or not mw_in_range(smi):
            continue
        # Name = cohort_<idx>
        mol.SetProp("_Name", f"{name}_{i}")
        mol.SetProp("smi", smi)
        mol.SetProp("source", "lingo")
        mols.append(mol)

    rng = np.random.default_rng(RANDOM_SEED)
    if len(mols) > target_n:
        idx = rng.choice(len(mols), size=target_n, replace=False)
        mols = [mols[i] for i in sorted(idx.tolist())]

    writer = Chem.SDWriter(str(out_sdf))
    for m in mols:
        writer.write(m)
    writer.close()

    return {
        "name": name,
        "group": "lingo_pcov",
        "source_path": str(src),
        "input_sdf": str(out_sdf),
        "n_mols": len(mols),
    }


def prep_baseline_cohort(name: str, csv: Path, smi_col: str, score_col: str | None,
                         target_n: int = TARGET_N, n_workers: int = 6):
    """For SMILES-only baselines.

    Strategy:
    - Read CSV, drop dups, filter MW range
    - For "warhead_gate" cohorts: require acrylamide on largest fragment
    - For LibInvent_locked / Amine_Replacements: no warhead gate (use natural distribution)
    - If we have a score column, prefer top-scoring; otherwise random sample
    - ETKDG-embed in parallel, write SDF
    """
    if not csv.exists():
        return None
    out_dir = OUT_DIR / name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_sdf = out_dir / "input.sdf"

    df = pd.read_csv(csv)
    if smi_col not in df.columns:
        return None
    df = df[df[smi_col].notna() & (df[smi_col].astype(str).str.len() > 4)]
    df["smi_canon"] = df[smi_col].apply(
        lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if Chem.MolFromSmiles(s) else None
    )
    df = df[df["smi_canon"].notna()].drop_duplicates("smi_canon")
    df = df[df["smi_canon"].apply(mw_in_range)]

    if "warhead_gate" in name:
        df = df[df["smi_canon"].apply(acrylamide_on_largest_frag)]

    if score_col and score_col in df.columns:
        df = df[df[score_col].notna()]
        df["score_num"] = pd.to_numeric(df[score_col], errors="coerce")
        df = df[df["score_num"].notna()]
        # higher pIC50 is better
        df = df.sort_values("score_num", ascending=False)

    if len(df) == 0:
        return None
    if len(df) > target_n:
        # If we have score, take top-N (which is already sorted)
        if score_col and score_col in df.columns:
            df = df.head(target_n)
        else:
            df = df.sample(target_n, random_state=RANDOM_SEED)

    smiles_list = df["smi_canon"].tolist()
    print(f"  {name}: filtered to {len(smiles_list)} mols, embedding...")

    # Parallel ETKDG embed
    rng = np.random.default_rng(RANDOM_SEED)
    seeds = rng.integers(0, 1_000_000, size=len(smiles_list)).tolist()
    args = [(i, s, seeds[i]) for i, s in enumerate(smiles_list)]
    mb_results = {}
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        for r in ex.map(_embed_one, args, chunksize=4):
            idx, mb = r
            mb_results[idx] = mb

    writer = Chem.SDWriter(str(out_sdf))
    n_written = 0
    for i, smi in enumerate(smiles_list):
        mb = mb_results.get(i)
        if mb is None:
            continue
        mol = Chem.MolFromMolBlock(mb)
        if mol is None:
            continue
        mol.SetProp("_Name", f"{name}_{i}")
        mol.SetProp("smi", smi)
        mol.SetProp("source", "reinvent" if "Replacements" not in name else "amine")
        writer.write(mol)
        n_written += 1
    writer.close()

    return {
        "name": name,
        "group": "seqonly",
        "source_path": str(csv),
        "input_sdf": str(out_sdf),
        "n_mols": n_written,
    }


def main():
    print("=" * 72)
    print("  Preparing cohort SDFs for ZAP70 docking comparison")
    print(f"  Target N per cohort: {TARGET_N}")
    print(f"  Output: {OUT_DIR}")
    print("=" * 72)

    manifest = []
    # Lingo cohorts: just downsample existing SDFs (sequential, fast)
    print("\n[1/2] Lingo3DMol pocket+covalent-aware cohorts...")
    for name, (src, group) in LINGO_COHORTS.items():
        info = prep_lingo_cohort(name, src)
        if info is None:
            print(f"  SKIP {name}: source missing or empty ({src})")
            continue
        print(f"  OK   {name}: {info['n_mols']} mols -> {info['input_sdf']}")
        manifest.append(info)

    # Baseline cohorts: SMILES embed (parallel)
    print("\n[2/2] Sequence-only baselines (ETKDG embed)...")
    n_workers = max(2, cpu_count() // 2)
    for name, (csv, smi_col, score_col, group) in BASELINE_CSVS.items():
        info = prep_baseline_cohort(name, csv, smi_col, score_col, n_workers=n_workers)
        if info is None:
            print(f"  SKIP {name}: source missing or empty ({csv})")
            continue
        print(f"  OK   {name}: {info['n_mols']} mols -> {info['input_sdf']}")
        manifest.append(info)

    manifest_path = MANIFEST_DIR / "cohort_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2))
    print(f"\nManifest -> {manifest_path}")
    print(f"Total cohorts: {len(manifest)}")
    total_mols = sum(c["n_mols"] for c in manifest)
    print(f"Total molecules to dock: {total_mols}")


if __name__ == "__main__":
    main()
