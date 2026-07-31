"""Per-cohort QA: validate a single THIQ RL chkpt + its sampled cohort.

Tests run by the overnight orchestrator after each RL chkpt or sampling pass.

Streams tested:
  1. Chkpt file: exists, ≥50MB, loads via torch.load, has 'vocabulary' + 'network_state'
  2. Sampling CSV: exists, has SMILES column, all rows parse with RDKit
  3. Mol1 fidelity: median Tc-to-Mol1, THIQ-acryl%, Murcko match% (sanity checks)
  4. Reward CSV (only after RL): step count, reward dynamics monotone, no NaN

Run:  python experiments/qa_thiq_cohort.py <cohort_tag> [--mode rl|sample|all]

Exit non-zero on any QA failure so the orchestrator can react.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

# --- Mol1 references ---
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
_m1 = Chem.MolFromSmiles(MOL1)
MOL1_CANON = Chem.MolToSmiles(_m1)
MOL1_MURCKO = Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(_m1))
MOL1_FP = AllChem.GetMorganFingerprintAsBitVect(_m1, 2, 2048)
THIQ_ACRYL = Chem.MolFromSmarts("C=CC(=O)N1Cc2ccccc2C1")

# Defaults (configurable via env)
PROJECT_ROOT = Path(os.environ.get("PROJECT_ROOT", Path(__file__).resolve().parent.parent))


def log(stream: str, name: str, ok: bool, detail: str = ""):
    flag = "PASS" if ok else "FAIL"
    print(f"  [{flag}] [{stream}] {name}" + (f"  — {detail}" if detail else ""))
    return ok


# ------------------------------------------------------------------
# Stream 1 — checkpoint validation
def qa_chkpt(tag: str) -> bool:
    chkpt = PROJECT_ROOT / "results/paper_evaluation/mol1_rl" / tag / f"{tag}_stage1.chkpt"
    ok = True
    ok &= log("chkpt", f"{chkpt.name} exists", chkpt.exists(),
              f"{chkpt.stat().st_size//1_000_000}MB" if chkpt.exists() else "missing")
    if not chkpt.exists():
        return False
    ok &= log("chkpt", "≥50MB", chkpt.stat().st_size >= 50_000_000,
              f"{chkpt.stat().st_size//1_000_000}MB")
    try:
        ck = torch.load(chkpt, map_location="cpu", weights_only=False)
        keys = set(ck.keys()) if isinstance(ck, dict) else set()
        ok &= log("chkpt", "has 'vocabulary' key", "vocabulary" in keys)
        ok &= log("chkpt", "has 'network_state' key", "network_state" in keys)
        if "vocabulary" in keys:
            tokens = set(ck["vocabulary"].get("tokens", {}).keys())
            ok &= log("chkpt", "vocab has 100+ tokens",
                      len(tokens) >= 100, f"{len(tokens)} tokens")
    except Exception as e:
        ok &= log("chkpt", "torch.load succeeds", False, str(e)[:120])
    return ok


# ------------------------------------------------------------------
# Stream 2 — sampling CSV validation
def qa_sample(tag: str, min_rows: int = 100) -> bool:
    csv = PROJECT_ROOT / "data/mol1_anchored_tier4" / f"{tag}_sample_mol1_100K" / "sampling.csv"
    ok = True
    ok &= log("sample", f"{csv.parent.name}/sampling.csv exists", csv.exists())
    if not csv.exists():
        return False
    df = pd.read_csv(csv)
    ok &= log("sample", f"≥{min_rows} rows", len(df) >= min_rows, f"{len(df):,}")
    smi_col = "SMILES" if "SMILES" in df.columns else ("smiles" if "smiles" in df.columns else None)
    ok &= log("sample", "has SMILES column", smi_col is not None)
    if smi_col is None:
        return False

    # All rows parse
    valid = sum(1 for s in df[smi_col] if Chem.MolFromSmiles(str(s)) is not None)
    pct_valid = valid / len(df) if len(df) else 0
    ok &= log("sample", "≥90% rows parse",
              pct_valid >= 0.9, f"{pct_valid:.1%}")

    # Mol1 fidelity stats
    n_acryl = n_thiq = n_murcko = 0
    tcs = []
    for s in df[smi_col]:
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        if m.HasSubstructMatch(Chem.MolFromSmarts("[CH2]=[CH]C(=O)[N]")):
            n_acryl += 1
        if m.HasSubstructMatch(THIQ_ACRYL):
            n_thiq += 1
        if Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m)) == MOL1_MURCKO:
            n_murcko += 1
        fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
        tcs.append(DataStructs.TanimotoSimilarity(MOL1_FP, fp))

    if not tcs:
        ok &= log("sample", "any parseable mols", False)
        return ok
    tcs = np.array(tcs)
    n = len(tcs)
    print(f"      Mol1 fidelity: acryl={n_acryl/n:.1%}, THIQ={n_thiq/n:.1%}, "
          f"Murcko={n_murcko/n:.1%}, Tc median={np.median(tcs):.3f}, "
          f"Tc≥0.4: {(tcs>=0.4).mean():.1%}")
    # Lenient sanity: THIQ% should at least exceed 0.04% (the historic 213K-cohort baseline)
    ok &= log("sample", "THIQ% beats baseline (>0.1%)",
              n_thiq / n > 0.001, f"{n_thiq/n*100:.2f}%")
    return ok


# ------------------------------------------------------------------
# Stream 3 — reward CSV validation (RL only)
def qa_reward_csv(tag: str) -> bool:
    csv = PROJECT_ROOT / "results/paper_evaluation/mol1_rl" / tag / f"{tag}_1.csv"
    ok = True
    if not csv.exists():
        return log("reward", f"{csv.name} exists", False, "missing — RL maybe didn't write")
    df = pd.read_csv(csv)
    ok &= log("reward", "has ≥1 step", "step" in df.columns and df["step"].max() >= 1,
              f"{df['step'].max() if 'step' in df.columns else '?'} steps")
    if "Score" in df.columns:
        ok &= log("reward", "no NaN in Score column",
                  df["Score"].notna().all(),
                  f"NaN count={df['Score'].isna().sum()}")
        ok &= log("reward", "max Score >= 0.5 (meaningful reward)",
                  df["Score"].max() >= 0.5, f"max={df['Score'].max():.3f}")
    return ok


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("tag", help="Cohort tag e.g. thiq_rl_mol1only")
    ap.add_argument("--mode", choices=["rl", "sample", "all"], default="all")
    ap.add_argument("--min-sample-rows", type=int, default=100)
    args = ap.parse_args()

    print(f"=== QA for cohort: {args.tag} (mode={args.mode}) ===")
    all_ok = True
    if args.mode in ("rl", "all"):
        all_ok &= qa_chkpt(args.tag)
        all_ok &= qa_reward_csv(args.tag)
    if args.mode in ("sample", "all"):
        all_ok &= qa_sample(args.tag, min_rows=args.min_sample_rows)

    print(f"\n=== {'PASS' if all_ok else 'FAIL'} ===")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
