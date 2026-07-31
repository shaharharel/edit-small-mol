"""Build R1 regularized DPO pairs by filtering composite_quality.parquet.

Rationale:
  Stream B (composite_quality, β=0.1, 5 epochs) showed catastrophic mode collapse:
  warhead retention dropped from 100% (in pair data) to 2.7% (in samples).
  Diagnosis of pair set:
    - chosen acrylamide rate = 100%, rejected acrylamide rate = 100%
      (so collapse is NOT due to warhead-absent training signal)
    - chosen Tc(Mol1) median = 0.20, only 27.9% in [0.3, 0.7]
      (collapse pushed policy way off-anchor)
    - Stream B final loss ~3e-4, val_acc=1.0, val_gap=191 (severe overfit)

Fixes embedded in R1:
  1. Hard guardrail on `chosen`: SMARTS must match (already true in source).
  2. Hard Tc band [0.3, 0.7] for `chosen` vs Mol1 (Tanimoto guardrail).
  3. Also restrict `rejected` to Tc <= 0.7 to keep contrasts well-defined.
  4. Pair with β=0.05 (was 0.1) and 3 epochs (was 5) at training stage.

Output: data/dpo_pairs/regularized_r1.parquet
"""
from __future__ import annotations
import argparse
import warnings
from pathlib import Path

import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parents[1]
MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
SMARTS = "[CH2]=[CH]C(=O)N"


def has_acryl(s: str, patt) -> bool:
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    return m is not None and m.HasSubstructMatch(patt)


def tc_to(s: str, fp_ref) -> float:
    m = Chem.MolFromSmiles(s) if isinstance(s, str) else None
    if m is None:
        return 0.0
    fp = AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)
    return float(DataStructs.TanimotoSimilarity(fp, fp_ref))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--src", default=str(ROOT / "data/dpo_pairs/composite_quality.parquet"))
    ap.add_argument("--out", default=str(ROOT / "data/dpo_pairs/regularized_r1.parquet"))
    ap.add_argument("--tc_lo", type=float, default=0.30)
    ap.add_argument("--tc_hi", type=float, default=0.70)
    ap.add_argument("--tc_rej_hi", type=float, default=0.70,
                    help="cap rejected Tc(Mol1) to avoid noise; -1 to disable")
    args = ap.parse_args()

    df = pd.read_parquet(args.src)
    print(f"Loaded {len(df)} pairs from {args.src}")

    mol1 = Chem.MolFromSmiles(MOL1)
    fp1 = AllChem.GetMorganFingerprintAsBitVect(mol1, 2, 2048)
    patt = Chem.MolFromSmarts(SMARTS)

    print("Computing chosen Tc to Mol1...")
    df["chosen_tc_mol1"] = df["chosen"].apply(lambda s: tc_to(s, fp1))
    df["rejected_tc_mol1"] = df["rejected"].apply(lambda s: tc_to(s, fp1))
    df["chosen_acryl"] = df["chosen"].apply(lambda s: has_acryl(s, patt))
    df["rejected_acryl"] = df["rejected"].apply(lambda s: has_acryl(s, patt))

    print(f"chosen acrylamide retention: {df.chosen_acryl.mean():.3f}")
    print(f"rejected acrylamide retention: {df.rejected_acryl.mean():.3f}")
    print(f"chosen Tc(Mol1) median = {df.chosen_tc_mol1.median():.3f}")

    mask = (
        df["chosen_acryl"]
        & (df["chosen_tc_mol1"] >= args.tc_lo)
        & (df["chosen_tc_mol1"] <= args.tc_hi)
    )
    if args.tc_rej_hi > 0:
        mask &= df["rejected_tc_mol1"] <= args.tc_rej_hi
    out = df.loc[mask, ["prompt", "chosen", "rejected", "q_chosen", "q_rejected",
                         "tanimoto", "source"]].reset_index(drop=True)
    print(f"After R1 filter: {len(out)} pairs ({len(out)/len(df)*100:.1f}% retained)")

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.out, index=False)
    print(f"Wrote {args.out}")


if __name__ == "__main__":
    main()
