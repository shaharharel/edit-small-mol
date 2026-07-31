"""Score arbitrary cohort with the 10_XGB_MultiFP checkpoint from the ZAP70 challenge.

Loads:
  results/zap70_challenge/checkpoints/10_XGB_MultiFP_full.pkl

Computes per-SMILES the same 6311-d feature stack used during training:
  Morgan r=2 (2048) | RDKit fp (2048) | MACCS (167) | AtomPair (2048)

Writes pIC50_xgb_multifp into the cohort CSV (idempotent merge on SMILES).

Usage:
    python scripts/score_xgb_multifp_cohort.py \
        --cohort data/tier4_scored/vina_40k_for_triage.csv \
        --smiles-col smiles \
        --out-col pIC50_xgb_multifp \
        --in-place
"""
from __future__ import annotations

import argparse
import pickle
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs
from rdkit.Chem import AllChem, MACCSkeys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
CKPT = PROJECT_ROOT / "results/zap70_challenge/checkpoints/10_XGB_MultiFP_full.pkl"


def morgan(smi: str, n_bits: int = 2048, radius: int = 2) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def rdkit_fp(smi: str, n_bits: int = 2048) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = Chem.RDKFingerprint(m, fpSize=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def maccs(smi: str) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = MACCSkeys.GenMACCSKeys(m)
    arr = np.zeros(167, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def atompair(smi: str, n_bits: int = 2048) -> np.ndarray | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    fp = AllChem.GetHashedAtomPairFingerprintAsBitVect(m, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.uint8)
    DataStructs.ConvertToNumpyArray(fp, arr)
    return arr


def featurize(smiles: list[str]) -> tuple[np.ndarray, np.ndarray]:
    """Return (X 6311-d, valid_mask)."""
    n = len(smiles)
    X = np.zeros((n, 6311), dtype=np.float32)
    valid = np.zeros(n, dtype=bool)
    t0 = time.perf_counter()
    for i, s in enumerate(smiles):
        m = morgan(s)
        if m is None:
            continue
        r = rdkit_fp(s)
        k = maccs(s)
        a = atompair(s)
        if r is None or k is None or a is None:
            continue
        X[i, :2048] = m
        X[i, 2048:4096] = r
        X[i, 4096:4263] = k
        X[i, 4263:6311] = a
        valid[i] = True
        if (i + 1) % 1000 == 0:
            rate = (i + 1) / (time.perf_counter() - t0)
            print(f"  featurized {i+1}/{n}  ({rate:.0f}/s)", flush=True)
    return X, valid


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort", required=True, type=Path)
    ap.add_argument("--smiles-col", default="smiles")
    ap.add_argument("--out-col", default="pIC50_xgb_multifp")
    ap.add_argument("--in-place", action="store_true",
                    help="Overwrite cohort CSV with the new column. Otherwise writes a parallel _xgb.csv.")
    ap.add_argument("--limit", type=int, default=None, help="Score first N rows only (smoke test).")
    args = ap.parse_args()

    df = pd.read_csv(args.cohort, low_memory=False)
    n_total = len(df)
    print(f"Cohort: {args.cohort.name} ({n_total:,} rows, columns: {len(df.columns)})", flush=True)

    if args.limit:
        df = df.head(args.limit).copy()
        print(f"  --limit applied: {len(df):,} rows", flush=True)

    # Dedupe SMILES to avoid recomputing identical features
    unique_smi = df[args.smiles_col].astype(str).fillna("").tolist()
    uniq = pd.Series(unique_smi).drop_duplicates().tolist()
    print(f"Unique SMILES: {len(uniq):,}", flush=True)

    print("Featurizing (Morgan + RDKit + MACCS + AtomPair, 6311-d)...", flush=True)
    X, valid = featurize(uniq)
    print(f"  valid: {valid.sum():,} / {len(uniq):,} ({100*valid.sum()/len(uniq):.1f}%)", flush=True)

    print(f"Loading XGB checkpoint: {CKPT}", flush=True)
    with open(CKPT, "rb") as f:
        ckpt = pickle.load(f)
    model = ckpt["obj"]
    n_feats = model.n_features_in_
    assert n_feats == 6311, f"Checkpoint expects {n_feats} features, got 6311 — feature spec mismatch."

    print(f"Predicting (XGBRegressor, n_features={n_feats})...", flush=True)
    preds = np.full(len(uniq), np.nan, dtype=np.float32)
    if valid.any():
        preds[valid] = model.predict(X[valid])
    smi_to_pred = dict(zip(uniq, preds))

    print(f"Merging predictions back to {len(df):,} rows...", flush=True)
    df[args.out_col] = df[args.smiles_col].astype(str).map(smi_to_pred)
    n_filled = df[args.out_col].notna().sum()
    print(f"  filled {n_filled:,} / {len(df):,} ({100*n_filled/len(df):.1f}%)", flush=True)

    # Quick summary stats
    s = df[args.out_col].dropna()
    print(f"\nXGB pIC50 stats:")
    print(f"  count={len(s):,}  mean={s.mean():.3f}  median={s.median():.3f}  std={s.std():.3f}")
    print(f"  p10={s.quantile(0.10):.3f}  p50={s.quantile(0.50):.3f}  p90={s.quantile(0.90):.3f}")
    print(f"  min={s.min():.3f}  max={s.max():.3f}")
    print(f"  ≥7.0: {(s >= 7.0).sum():,} ({100*(s>=7.0).mean():.1f}%)")
    print(f"  ≥7.5: {(s >= 7.5).sum():,} ({100*(s>=7.5).mean():.1f}%)")
    print(f"  ≥8.0: {(s >= 8.0).sum():,} ({100*(s>=8.0).mean():.1f}%)")

    if args.in_place:
        df.to_csv(args.cohort, index=False)
        print(f"\nWrote in-place to {args.cohort}")
    else:
        out = args.cohort.with_suffix(".xgb.csv")
        df.to_csv(out, index=False)
        print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
