#!/usr/bin/env python3
"""Stream C: Build DPO preference pairs from GEOMETRY-ONLY pose quality.

Per coordinator audit (2026-06-30):
- d_SG is degenerate (Boltz cofold restraint forces ~1.85 Å)
- Quality collapses to BD-angle alone:
      q = exp(-bd_dev_deg^2 / (2 * 8^2))

Data source: data/tier4_scored/boltz2_cohort_A_relaxed.csv (3472 rows with bd dev)

Pair generation strategy (mol2mol seq2seq DPO):
  source = Mol1 anchor SMILES (universal prompt for the policy)
  chosen = sample with high q
  rejected = sample with low q
  filters:
    - both must be valid Morgan-FP molecules
    - Tc(chosen, rejected) > 0.4 (related transformations, sharper preference signal)
    - |q_chosen - q_rejected| > 0.15

Sub-strategies:
  A. Within-method: pair high-q vs low-q WITHIN each method (EXP6_v5, EXP2_V2_RL_v2, ...)
  B. Cross-method: pair high-q from method-X vs low-q from method-Y (same anchor)

Target: ~25-40K pairs.
"""
from __future__ import annotations
import argparse
import gc
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
ANCHOR_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"

COHORT_A = PROJECT_ROOT / "data" / "tier4_scored" / "boltz2_cohort_A_relaxed.csv"
OUT_PARQUET = PROJECT_ROOT / "data" / "dpo_pairs" / "geometry_only.parquet"
OUT_PARQUET.parent.mkdir(parents=True, exist_ok=True)


def quality_from_bd(bd_dev_deg: float) -> float:
    """q = exp(-bd_dev^2 / (2*8^2)). bd_dev in degrees."""
    return float(np.exp(-(bd_dev_deg ** 2) / (2.0 * 8.0 ** 2)))


def morgan_fp(smi: str, radius: int = 2, nbits: int = 1024):
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nbits)


def tanimoto(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def load_geometry_table() -> pd.DataFrame:
    print(f"[load] reading {COHORT_A}")
    df = pd.read_csv(
        COHORT_A,
        usecols=["row_id", "smiles", "method", "burgi_dunitz_dev_deg", "d_SG"],
    )
    df = df.dropna(subset=["burgi_dunitz_dev_deg", "smiles"]).copy()
    df["q"] = df["burgi_dunitz_dev_deg"].apply(quality_from_bd)
    df["smiles_canon"] = df["smiles"].apply(canonical)
    df = df.dropna(subset=["smiles_canon"]).copy()
    df = df.drop_duplicates(subset=["smiles_canon"]).copy()
    df["method_short"] = (
        df["method"]
        .str.extract(r"(EXP[\w_]+)")[0]
        .fillna("UNK")
    )
    print(
        f"[load] {len(df)} unique canonical SMILES with bd-dev; "
        f"methods: {df['method_short'].value_counts().to_dict()}"
    )
    print(f"[load] q distribution:")
    print(df["q"].describe())
    return df


def precompute_fps(smiles_list: list[str]) -> dict[str, object]:
    out = {}
    for s in smiles_list:
        fp = morgan_fp(s)
        if fp is not None:
            out[s] = fp
    return out


def generate_pairs(
    df: pd.DataFrame,
    fps: dict,
    high_q_cut: float = 0.5,
    low_q_cut: float = 0.05,
    margin: float = 0.15,
    tc_min: float = 0.4,
    tc_max: float = 0.95,
    max_pairs: int = 80000,
    seed: int = 7,
) -> pd.DataFrame:
    """Pair every high-q sample against several low-q samples.

    Tc filter window (tc_min, tc_max):
      - tc>=tc_min: chosen/rejected must share warhead+scaffold context (meaningful pref)
      - tc<=tc_max: avoid trivial pairs (same mol with minor difference)
    """
    rng = np.random.default_rng(seed)
    high = df[df["q"] >= high_q_cut].copy()
    low = df[df["q"] <= low_q_cut].copy()
    print(f"[pair] high-q (q>={high_q_cut}): {len(high)}, low-q (q<={low_q_cut}): {len(low)}")

    high_recs = high.to_records(index=False)
    low_recs = low.to_records(index=False)
    rng.shuffle(low_recs)  # randomize order so we don't always pick the same low

    n_low_per_high = max(1, max_pairs // max(1, len(high)))
    print(f"[pair] target ≤{n_low_per_high} rejected per chosen → cap ~{max_pairs}")

    pairs = []
    seen_smiles_pairs = set()
    for hi in high_recs:
        if len(pairs) >= max_pairs:
            break
        hi_smi = str(hi.smiles_canon)
        hi_q = float(hi.q)
        hi_fp = fps.get(hi_smi)
        if hi_fp is None:
            continue
        kept = 0
        for lo in low_recs:
            if kept >= n_low_per_high:
                break
            lo_smi = str(lo.smiles_canon)
            lo_q = float(lo.q)
            if lo_smi == hi_smi:
                continue
            if hi_q - lo_q < margin:
                continue
            key = (hi_smi, lo_smi)
            if key in seen_smiles_pairs:
                continue
            lo_fp = fps.get(lo_smi)
            tc = tanimoto(hi_fp, lo_fp)
            if tc < tc_min or tc > tc_max:
                continue
            pairs.append({
                "source_smiles": ANCHOR_SMI,
                "chosen_smiles": hi_smi,
                "rejected_smiles": lo_smi,
                "q_chosen": hi_q,
                "q_rejected": lo_q,
                "delta_q": hi_q - lo_q,
                "tc": tc,
                "method_chosen": str(hi.method_short),
                "method_rejected": str(lo.method_short),
            })
            seen_smiles_pairs.add(key)
            kept += 1
    print(f"[pair] kept {len(pairs)} preference pairs")
    return pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--high-q", type=float, default=0.5,
                        help="quality threshold for chosen samples")
    parser.add_argument("--low-q", type=float, default=0.05,
                        help="quality threshold for rejected samples")
    parser.add_argument("--margin", type=float, default=0.15,
                        help="min |q_chosen - q_rejected|")
    parser.add_argument("--tc-min", type=float, default=0.4)
    parser.add_argument("--tc-max", type=float, default=0.95)
    parser.add_argument("--max-pairs", type=int, default=80000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", type=str, default=str(OUT_PARQUET))
    args = parser.parse_args()

    df = load_geometry_table()

    # Precompute Morgan FPs once for all unique SMILES touched
    candidate_smiles = set(df.loc[df["q"] >= args.low_q, "smiles_canon"]).union(
        set(df.loc[df["q"] <= max(args.low_q, 0.5), "smiles_canon"])
    )
    # Easier: precompute for ALL SMILES (max 3472)
    print(f"[fp] computing Morgan FPs for {df['smiles_canon'].nunique()} SMILES")
    fps = precompute_fps(df["smiles_canon"].unique().tolist())
    print(f"[fp] valid: {len(fps)}/{df['smiles_canon'].nunique()}")
    gc.collect()

    pairs_df = generate_pairs(
        df, fps,
        high_q_cut=args.high_q,
        low_q_cut=args.low_q,
        margin=args.margin,
        tc_min=args.tc_min,
        tc_max=args.tc_max,
        max_pairs=args.max_pairs,
        seed=args.seed,
    )

    # 90/10 train/val split — group-safe (a chosen smiles can't be in both)
    rng = np.random.default_rng(args.seed)
    unique_chosen = pairs_df["chosen_smiles"].unique()
    rng.shuffle(unique_chosen)
    val_chosen = set(unique_chosen[: max(1, len(unique_chosen) // 10)])
    pairs_df["split"] = np.where(
        pairs_df["chosen_smiles"].isin(val_chosen), "val", "train"
    )
    n_train = (pairs_df["split"] == "train").sum()
    n_val = (pairs_df["split"] == "val").sum()
    print(f"[split] train={n_train}, val={n_val}")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs_df.to_parquet(out, index=False)
    print(f"[save] wrote {out} ({len(pairs_df)} pairs)")

    # Also save lightweight JSON summary
    summary = {
        "n_pairs_total": int(len(pairs_df)),
        "n_train": int(n_train),
        "n_val": int(n_val),
        "high_q_cut": args.high_q,
        "low_q_cut": args.low_q,
        "margin": args.margin,
        "tc_window": [args.tc_min, args.tc_max],
        "delta_q_mean": float(pairs_df["delta_q"].mean()),
        "delta_q_min": float(pairs_df["delta_q"].min()),
        "delta_q_max": float(pairs_df["delta_q"].max()),
        "tc_mean": float(pairs_df["tc"].mean()),
        "n_unique_chosen": int(pairs_df["chosen_smiles"].nunique()),
        "n_unique_rejected": int(pairs_df["rejected_smiles"].nunique()),
        "method_chosen_counts": pairs_df["method_chosen"].value_counts().to_dict(),
        "method_rejected_counts": pairs_df["method_rejected"].value_counts().to_dict(),
        "scoring_formula": "q = exp(-bd_dev_deg^2 / (2 * 8^2))",
        "scoring_notes": "d_SG dropped (Boltz cofold restraint=1.85 makes it degenerate).",
        "anchor_smiles": ANCHOR_SMI,
    }
    (out.parent / (out.stem + ".summary.json")).write_text(json.dumps(summary, indent=2))
    print(f"[save] summary → {out.parent / (out.stem + '.summary.json')}")


if __name__ == "__main__":
    main()
