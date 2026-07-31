#!/usr/bin/env python3
"""Build BOLTZ-GEOM DPO preference pairs for the C2/C3/C4 campaign.

Source: data/paper_pair_training/zap70_cofold_harvest/zap70_acryl_mpae_extended.csv

Guardrails (must ALL hold for both winner + loser):
  1. Contains acrylamide warhead on LARGEST FRAGMENT:
     SMARTS = "[CH2]=[CH][C](=O)[N]"
  2. Tc(mol, Mol1_anchor) >= 0.35  (Morgan r=2, nBits=2048)
  3. (Winner, loser) must share Bemis-Murcko scaffold cluster
     OR Tc(winner, loser) >= 0.5
  4. Winner has BOTH tighter |bd_angle-105|-difference >= 10 deg
     AND tighter |d_b_nuc-3.5|-difference >= 0.5 A vs loser.

Target: 3000-5000 pairs.
Output: data/paper_pair_training/boltz_dpo_campaign/pairs_geom.parquet
Also writes pairs_geom.summary.json.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path
from collections import defaultdict

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CSV_PATH = PROJECT_ROOT / "data" / "paper_pair_training" / "zap70_cofold_harvest" / "zap70_acryl_mpae_extended.csv"
OUT_DIR = PROJECT_ROOT / "data" / "paper_pair_training" / "boltz_dpo_campaign"
OUT_PARQUET = OUT_DIR / "pairs_geom.parquet"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ANCHOR_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
ACRYL_SMARTS = "[CH2]=[CH][C](=O)[N]"


def largest_fragment(smi: str) -> str | None:
    """Return SMILES of the largest fragment (by atom count) of a multi-frag SMILES."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    frags = Chem.GetMolFrags(mol, asMols=True, sanitizeFrags=False)
    if not frags:
        return None
    frags = sorted(frags, key=lambda m: m.GetNumHeavyAtoms(), reverse=True)
    return Chem.MolToSmiles(frags[0], canonical=True)


def has_acrylamide_lf(smi: str, patt) -> bool:
    lf = largest_fragment(smi)
    if lf is None:
        return False
    m = Chem.MolFromSmiles(lf)
    if m is None:
        return False
    return m.HasSubstructMatch(patt)


def morgan_fp(smi: str, radius: int = 2, nbits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


def tanimoto(fp_a, fp_b) -> float:
    if fp_a is None or fp_b is None:
        return 0.0
    return DataStructs.TanimotoSimilarity(fp_a, fp_b)


def canonical(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m, canonical=True)


def murcko_scaffold(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc, canonical=True)
    except Exception:
        return None


def geom_quality(bd_deg: float, d_a: float) -> float:
    """Composite quality:
      q = exp(-|bd-105|^2 / (2*15^2)) * exp(-|d-3.5|^2 / (2*0.6^2))
    Higher = better.
    """
    bd_dev = abs(bd_deg - 105.0)
    d_dev = abs(d_a - 3.5)
    return float(
        np.exp(-(bd_dev ** 2) / (2.0 * 15.0 ** 2))
        * np.exp(-(d_dev ** 2) / (2.0 * 0.6 ** 2))
    )


def build_pairs(
    df: pd.DataFrame,
    tc_min_to_mol1: float = 0.35,
    tc_pair_min: float = 0.5,
    bd_gap_deg: float = 10.0,
    d_gap_a: float = 0.5,
    max_pairs_per_scaffold: int = 40,
    max_pairs_total: int = 5000,
    seed: int = 7,
) -> pd.DataFrame:
    """Build preference pairs per guardrail spec."""
    rng = np.random.default_rng(seed)
    patt = Chem.MolFromSmarts(ACRYL_SMARTS)

    print(f"[filter] input rows: {len(df)}")
    df = df.dropna(subset=["smiles", "bd_angle_deg", "d_b_nuc_angstrom"]).copy()
    print(f"[filter] with bd+d: {len(df)}")

    # canonicalize
    df["smi_canon"] = df["smiles"].apply(canonical)
    df = df.dropna(subset=["smi_canon"]).copy()

    # acrylamide-on-largest-frag filter
    df["has_acryl_lf"] = df["smi_canon"].apply(lambda s: has_acrylamide_lf(s, patt))
    df = df[df["has_acryl_lf"]].copy()
    print(f"[filter] with acrylamide-on-largest-frag: {len(df)}")

    # Tc to Mol1
    mol1_fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(ANCHOR_SMI), 2, 2048)
    df["fp"] = df["smi_canon"].apply(morgan_fp)
    df["tc_to_mol1"] = df["fp"].apply(lambda fp: tanimoto(fp, mol1_fp))
    df = df[df["tc_to_mol1"] >= tc_min_to_mol1].copy()
    print(f"[filter] Tc>={tc_min_to_mol1} to Mol1: {len(df)}")

    # per-molecule best quality (dedupe by canonical SMILES; take average)
    df["q"] = df.apply(lambda r: geom_quality(r["bd_angle_deg"], r["d_b_nuc_angstrom"]), axis=1)
    agg = (
        df.groupby("smi_canon", sort=False)
        .agg({
            "bd_angle_deg": "mean",
            "d_b_nuc_angstrom": "mean",
            "q": "mean",
            "tc_to_mol1": "first",
            "fp": "first",
        })
        .reset_index()
    )
    print(f"[unique] unique canonical SMILES after dedupe: {len(agg)}")

    # murcko scaffold clusters
    agg["scaffold"] = agg["smi_canon"].apply(murcko_scaffold)
    agg = agg.dropna(subset=["scaffold"]).copy()
    scaffold_counts = agg["scaffold"].value_counts()
    print(f"[scaffold] n_scaffolds: {agg['scaffold'].nunique()}; top5: {scaffold_counts.head(5).to_dict()}")

    # Build pairs
    pairs = []
    scaffold_pair_count = defaultdict(int)

    # Group by scaffold
    scaffold_groups = defaultdict(list)
    for idx, row in agg.iterrows():
        scaffold_groups[row["scaffold"]].append(idx)

    # Within-scaffold pairs: iterate scaffolds, form high-vs-low pairs
    for sc, idxs in scaffold_groups.items():
        if len(idxs) < 2:
            continue
        sub = agg.loc[idxs].sort_values("q", ascending=False).reset_index(drop=True)
        # take top-quartile vs bottom-quartile within this scaffold
        n = len(sub)
        top = sub.iloc[: max(1, n // 3)]
        bot = sub.iloc[-max(1, n // 3):]
        for _, hi in top.iterrows():
            if scaffold_pair_count[sc] >= max_pairs_per_scaffold:
                break
            for _, lo in bot.iterrows():
                if scaffold_pair_count[sc] >= max_pairs_per_scaffold:
                    break
                if hi["smi_canon"] == lo["smi_canon"]:
                    continue
                # geometry gap requirements
                hi_bd_dev = abs(hi["bd_angle_deg"] - 105.0)
                lo_bd_dev = abs(lo["bd_angle_deg"] - 105.0)
                hi_d_dev = abs(hi["d_b_nuc_angstrom"] - 3.5)
                lo_d_dev = abs(lo["d_b_nuc_angstrom"] - 3.5)
                if not (lo_bd_dev - hi_bd_dev >= bd_gap_deg):
                    continue
                if not (lo_d_dev - hi_d_dev >= d_gap_a):
                    continue
                # both mols in-scaffold; guardrails already applied.
                pairs.append({
                    "source_smiles": ANCHOR_SMI,
                    "chosen_smiles": hi["smi_canon"],
                    "rejected_smiles": lo["smi_canon"],
                    "q_chosen": hi["q"],
                    "q_rejected": lo["q"],
                    "delta_q": hi["q"] - lo["q"],
                    "bd_chosen": hi["bd_angle_deg"],
                    "bd_rejected": lo["bd_angle_deg"],
                    "d_chosen": hi["d_b_nuc_angstrom"],
                    "d_rejected": lo["d_b_nuc_angstrom"],
                    "tc_chosen_mol1": hi["tc_to_mol1"],
                    "tc_rejected_mol1": lo["tc_to_mol1"],
                    "tc_pair": tanimoto(hi["fp"], lo["fp"]),
                    "scaffold_match": True,
                })
                scaffold_pair_count[sc] += 1
        if len(pairs) >= max_pairs_total:
            break

    print(f"[within-scaffold] pairs so far: {len(pairs)}")

    # Cross-scaffold high-Tc pairs
    if len(pairs) < max_pairs_total:
        # Take top-quartile-quality molecules globally
        agg_sorted = agg.sort_values("q", ascending=False).reset_index(drop=True)
        n = len(agg_sorted)
        top_pool = agg_sorted.iloc[: n // 4]
        bot_pool = agg_sorted.iloc[-n // 4:]
        top_recs = top_pool.to_dict("records")
        bot_recs = bot_pool.to_dict("records")
        rng.shuffle(bot_recs)
        seen = {(p["chosen_smiles"], p["rejected_smiles"]) for p in pairs}
        for hi in top_recs:
            if len(pairs) >= max_pairs_total:
                break
            per_hi = 0
            for lo in bot_recs:
                if per_hi >= 20:
                    break
                if hi["smi_canon"] == lo["smi_canon"]:
                    continue
                if (hi["smi_canon"], lo["smi_canon"]) in seen:
                    continue
                if hi["scaffold"] == lo["scaffold"]:
                    continue  # already covered
                tc_pair = tanimoto(hi["fp"], lo["fp"])
                if tc_pair < 0.5:
                    continue
                hi_bd_dev = abs(hi["bd_angle_deg"] - 105.0)
                lo_bd_dev = abs(lo["bd_angle_deg"] - 105.0)
                hi_d_dev = abs(hi["d_b_nuc_angstrom"] - 3.5)
                lo_d_dev = abs(lo["d_b_nuc_angstrom"] - 3.5)
                if not (lo_bd_dev - hi_bd_dev >= bd_gap_deg):
                    continue
                if not (lo_d_dev - hi_d_dev >= d_gap_a):
                    continue
                pairs.append({
                    "source_smiles": ANCHOR_SMI,
                    "chosen_smiles": hi["smi_canon"],
                    "rejected_smiles": lo["smi_canon"],
                    "q_chosen": hi["q"],
                    "q_rejected": lo["q"],
                    "delta_q": hi["q"] - lo["q"],
                    "bd_chosen": hi["bd_angle_deg"],
                    "bd_rejected": lo["bd_angle_deg"],
                    "d_chosen": hi["d_b_nuc_angstrom"],
                    "d_rejected": lo["d_b_nuc_angstrom"],
                    "tc_chosen_mol1": hi["tc_to_mol1"],
                    "tc_rejected_mol1": lo["tc_to_mol1"],
                    "tc_pair": tc_pair,
                    "scaffold_match": False,
                })
                seen.add((hi["smi_canon"], lo["smi_canon"]))
                per_hi += 1

    print(f"[final] total preference pairs: {len(pairs)}")
    return pd.DataFrame(pairs)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", default=str(CSV_PATH))
    parser.add_argument("--tc-min-mol1", type=float, default=0.35)
    parser.add_argument("--tc-pair-min", type=float, default=0.5)
    parser.add_argument("--bd-gap", type=float, default=10.0)
    parser.add_argument("--d-gap", type=float, default=0.5)
    parser.add_argument("--max-per-scaffold", type=int, default=40)
    parser.add_argument("--max-pairs", type=int, default=5000)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--out", default=str(OUT_PARQUET))
    args = parser.parse_args()

    df = pd.read_csv(args.csv)
    pairs = build_pairs(
        df,
        tc_min_to_mol1=args.tc_min_mol1,
        tc_pair_min=args.tc_pair_min,
        bd_gap_deg=args.bd_gap,
        d_gap_a=args.d_gap,
        max_pairs_per_scaffold=args.max_per_scaffold,
        max_pairs_total=args.max_pairs,
        seed=args.seed,
    )

    if len(pairs) == 0:
        raise SystemExit("no pairs built; check filters")

    # train/val split by chosen SMILES (group-safe)
    rng = np.random.default_rng(args.seed)
    unique_chosen = pairs["chosen_smiles"].unique()
    rng.shuffle(unique_chosen)
    val_chosen = set(unique_chosen[: max(1, len(unique_chosen) // 10)])
    pairs["split"] = np.where(pairs["chosen_smiles"].isin(val_chosen), "val", "train")

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    pairs.to_parquet(out, index=False)
    print(f"[save] wrote {out} ({len(pairs)} pairs)")

    summary = {
        "n_pairs_total": int(len(pairs)),
        "n_train": int((pairs["split"] == "train").sum()),
        "n_val": int((pairs["split"] == "val").sum()),
        "n_within_scaffold": int(pairs["scaffold_match"].sum()),
        "n_cross_scaffold_hitc": int((~pairs["scaffold_match"]).sum()),
        "n_unique_chosen": int(pairs["chosen_smiles"].nunique()),
        "n_unique_rejected": int(pairs["rejected_smiles"].nunique()),
        "tc_chosen_mol1_mean": float(pairs["tc_chosen_mol1"].mean()),
        "tc_rejected_mol1_mean": float(pairs["tc_rejected_mol1"].mean()),
        "tc_pair_mean": float(pairs["tc_pair"].mean()),
        "delta_q_mean": float(pairs["delta_q"].mean()),
        "bd_gap_min_deg": args.bd_gap,
        "d_gap_min_a": args.d_gap,
        "tc_min_mol1": args.tc_min_mol1,
        "acryl_smarts": ACRYL_SMARTS,
        "anchor_smiles": ANCHOR_SMI,
        "quality_formula": "q = exp(-(bd-105)^2/(2*15^2)) * exp(-(d-3.5)^2/(2*0.6^2))",
    }
    summary_path = out.parent / (out.stem + ".summary.json")
    summary_path.write_text(json.dumps(summary, indent=2))
    print(f"[save] summary → {summary_path}")


if __name__ == "__main__":
    main()
