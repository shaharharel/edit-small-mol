"""Build DPO preference pairs from Boltz cofold poses using a composite quality score.

Composite quality q (per-mol, CORRECTED 2026-06-30):
  q = 0.40·sigmoid((iptm - 0.78) / 0.05)
    + 0.30·sigmoid((2.0 - mPAE) / 1.0)
    + 0.30·exp(-(burgi_dunitz_dev_deg - 0)^2 / (2·8°^2))   # BD angle score

The original spec also weighted the warhead Cβ→Sγ distance (d_SG) at 0.25, but
audit found d_SG ~ constant (median 1.94 Å, std 0.20) because Boltz cofold runs
were conducted with a covalent bond restraint at 1.85 Å. That term carried no
useful preference signal and was dropped per coordinator correction; the freed
weight was redistributed to iptm (+0.10), mPAE (+0.10), angle (+0.05).

Pairs:
  - Within-cohort: top-K vs bot-K of cohort A (shared anchor = Mol1)
  - Filter:
      * Tanimoto similarity > 0.4 between chosen/rejected (Morgan r=2, 2048b)
      * |q_chosen - q_rejected| > 0.15
  - Target: 20K-40K pairs

Source: data/tier4_scored/boltz2_cohort_A_relaxed.csv (3,472 valid rows). Cohort B
strict lacks Boltz iptm/mPAE/angle columns, so excluded. CovBinderInPDB augmentation
skipped (no cache available; fallback per task spec).

Output: data/dpo_pairs/composite_quality.parquet
        columns: prompt, chosen, rejected, q_chosen, q_rejected, tanimoto, source
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).resolve().parents[1]
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def composite_q(df: pd.DataFrame) -> pd.Series:
    """Compute corrected composite quality column-wise.

    Uses columns: boltz_iptm, mPAE_paper (== mPAE_full), burgi_dunitz_dev_deg.
    Note: burgi_dunitz_dev_deg is the DEVIATION (degrees) from the ideal 107°,
    so the angle term is centered at 0.

    d_SG dropped: nearly constant (~1.85 Å) due to Boltz cofold covalent restraint.
    """
    q_iptm = 0.40 * sigmoid((df["boltz_iptm"].values - 0.78) / 0.05)
    q_mpae = 0.30 * sigmoid((2.0 - df["mPAE_paper"].values) / 1.0)
    q_angle = 0.30 * np.exp(-((df["burgi_dunitz_dev_deg"].values - 0) ** 2) / (2 * 8.0 ** 2))
    return pd.Series(q_iptm + q_mpae + q_angle, index=df.index)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--cohort_csv", default=str(ROOT / "data/tier4_scored/boltz2_cohort_A_relaxed.csv"))
    p.add_argument("--out_parquet", default=str(ROOT / "data/dpo_pairs/composite_quality.parquet"))
    p.add_argument("--top_pct", type=float, default=0.20, help="top-K percentile (chosen)")
    p.add_argument("--bot_pct", type=float, default=0.20, help="bottom-K percentile (rejected)")
    p.add_argument("--min_dq", type=float, default=0.15)
    p.add_argument("--min_tanimoto", type=float, default=0.40)
    p.add_argument("--max_pairs", type=int, default=40000)
    p.add_argument("--anchor_smiles", default=MOL1)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    rng = np.random.default_rng(args.seed)

    print(f"[{time.strftime('%H:%M:%S')}] Loading cohort: {args.cohort_csv}")
    A = pd.read_csv(args.cohort_csv)
    print(f"  cohort rows: {len(A)}")

    needed = ["boltz_iptm", "mPAE_paper", "burgi_dunitz_dev_deg", "smiles", "warhead_intact"]
    for c in needed:
        if c not in A.columns:
            sys.exit(f"missing required column: {c}")
    mask = (
        A["boltz_iptm"].notna()
        & A["mPAE_paper"].notna()
        & A["burgi_dunitz_dev_deg"].notna()
        & A["smiles"].notna()
        & A["warhead_intact"].fillna(False).astype(bool)
    )
    A = A[mask].reset_index(drop=True)
    print(f"  rows with all metrics + warhead_intact: {len(A)}")

    # rdkit parseable
    A = A[A["smiles"].apply(lambda s: Chem.MolFromSmiles(s) is not None)].reset_index(drop=True)
    print(f"  rdkit-parseable: {len(A)}")

    # Deduplicate by canonical SMILES, keep best q
    A["q"] = composite_q(A)
    A["smiles_canon"] = A["smiles"].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)))
    A = A.sort_values("q", ascending=False).drop_duplicates("smiles_canon", keep="first").reset_index(drop=True)
    print(f"  after canonical dedup: {len(A)}")

    print(f"  q stats: mean={A['q'].mean():.4f} std={A['q'].std():.4f} min={A['q'].min():.4f} max={A['q'].max():.4f}")
    print(f"  top-{int(args.top_pct*100)}% threshold q>={A['q'].quantile(1-args.top_pct):.4f}")
    print(f"  bot-{int(args.bot_pct*100)}% threshold q<={A['q'].quantile(args.bot_pct):.4f}")

    # Build Morgan fingerprints
    print(f"[{time.strftime('%H:%M:%S')}] Computing Morgan fingerprints...")
    fps = [AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, 2048) for s in A["smiles"]]
    print(f"  computed {len(fps)} fingerprints")

    # Threshold-based selection
    top_thr = A["q"].quantile(1 - args.top_pct)
    bot_thr = A["q"].quantile(args.bot_pct)
    top_idx = np.where(A["q"].values >= top_thr)[0]
    bot_idx = np.where(A["q"].values <= bot_thr)[0]
    print(f"  top pool: {len(top_idx)}, bot pool: {len(bot_idx)}")

    # Pair generation: random sample of pairs satisfying both constraints
    print(f"[{time.strftime('%H:%M:%S')}] Generating preference pairs...")
    pairs = []
    q_vals = A["q"].values
    smiles = A["smiles_canon"].values
    # cap candidate-pair budget
    max_candidates = args.max_pairs * 50
    n_tried = 0
    seen = set()
    while len(pairs) < args.max_pairs and n_tried < max_candidates:
        ti = int(rng.choice(top_idx))
        bi = int(rng.choice(bot_idx))
        n_tried += 1
        if ti == bi:
            continue
        key = (ti, bi)
        if key in seen:
            continue
        seen.add(key)
        dq = q_vals[ti] - q_vals[bi]
        if abs(dq) <= args.min_dq:
            continue
        sim = DataStructs.TanimotoSimilarity(fps[ti], fps[bi])
        if sim < args.min_tanimoto:
            continue
        pairs.append(dict(
            prompt=args.anchor_smiles,
            chosen=smiles[ti] if dq > 0 else smiles[bi],
            rejected=smiles[bi] if dq > 0 else smiles[ti],
            q_chosen=float(max(q_vals[ti], q_vals[bi])),
            q_rejected=float(min(q_vals[ti], q_vals[bi])),
            tanimoto=float(sim),
            source="within_cohortA",
        ))
        if len(pairs) % 5000 == 0:
            print(f"  pairs={len(pairs)} tried={n_tried}")

    print(f"[{time.strftime('%H:%M:%S')}] Generated {len(pairs)} pairs (tried {n_tried})")
    out = pd.DataFrame(pairs)
    print(f"  mean |dq|: {(out['q_chosen']-out['q_rejected']).mean():.4f}")
    print(f"  mean tanimoto: {out['tanimoto'].mean():.4f}")
    print(f"  unique chosen: {out['chosen'].nunique()}; unique rejected: {out['rejected'].nunique()}")

    out_path = Path(args.out_parquet)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(out_path, index=False)
    print(f"[{time.strftime('%H:%M:%S')}] Wrote {out_path} ({out_path.stat().st_size/1e6:.2f} MB)")


if __name__ == "__main__":
    main()
