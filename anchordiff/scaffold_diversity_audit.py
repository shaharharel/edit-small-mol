"""Scaffold-diversity audit on the 997-mol ZAP70 cofold leaderboard.

The scientific-analyst flagged this as a potentially fatal blow to the
"de novo discovery" framing — if the top 50 collapse to a single
isoindolinone-acrylamide scaffold (paraphrases of Mol 1, the parent), then
the pipeline is a *refiner*, not a *generator*. This script answers that
quantitatively.

Per molecule we compute:
  - Bemis-Murcko generic scaffold SMILES
  - ECFP4 fingerprint (radius=2, 2048 bits)

We then cluster by:
  - Exact BM scaffold identity (cluster id = scaffold SMILES)
  - Single-linkage cluster on Tanimoto distance ≥ 0.5

For top-K cuts (K ∈ {20, 50, 100, 250}) we report:
  - # unique BM scaffolds
  - # diversity clusters at Tc 0.5
  - Tc to Mol 1 (the parent) distribution
  - Tc to the nearest anchor (any of the 280 known ZAP70 actives)

A high count of unique scaffolds = de novo discovery story holds.
A low count, especially clustered around Mol 1 = refiner story; we need
to reframe.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold
RDLogger.DisableLog("rdApp.*")

MOL1 = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


def bm_scaffold(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(m)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaf)
    except Exception:
        return None


def fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048)


def tc(fp1, fp2):
    return DataStructs.TanimotoSimilarity(fp1, fp2)


def cluster_single_linkage(fps, threshold=0.5):
    """Single-linkage Tanimoto clustering. Return list of cluster IDs per mol."""
    n = len(fps)
    cid = list(range(n))
    for i in range(n):
        for j in range(i + 1, n):
            if tc(fps[i], fps[j]) >= threshold:
                # merge: set both to min(cid[i], cid[j])
                a, b = cid[i], cid[j]
                m = min(a, b); M = max(a, b)
                for k in range(n):
                    if cid[k] == M:
                        cid[k] = m
    # remap to dense ids
    unique = sorted(set(cid))
    remap = {c: i for i, c in enumerate(unique)}
    return [remap[c] for c in cid]


def main():
    csv = PROJECT_ROOT / "results" / "anchordiff" / "cys346_cofold_leaderboard.csv"
    df = pd.read_csv(csv)
    print(f"loaded {len(df)} mols from {csv}")
    df = df.dropna(subset=["boltz_iptm"]).copy()
    print(f"  {len(df)} have boltz_iptm")

    # Combined score (same as in score_cys346_cofold.py)
    df["combined"] = df["rank_score"].rank(pct=True) * 0.6 + \
                     df["boltz_ligand_iptm"].rank(pct=True) * 0.4
    df = df.sort_values("combined", ascending=False).reset_index(drop=True)

    print("computing BM scaffolds + FPs (this takes a minute)...")
    df["bm_scaffold"] = df["smiles"].apply(bm_scaffold)
    df["fp"] = df["smiles"].apply(fp)

    parent_fp = fp(MOL1)
    df["tc_to_mol1"] = df["fp"].apply(lambda f: tc(f, parent_fp) if f else float("nan"))

    # Top-K audit
    print("\n=== Scaffold diversity at top-K cuts ===")
    print(f"{'K':>6}  {'#bm_scaffolds':>14}  {'#tc0.5_clusters':>18}  "
          f"{'med_tc_mol1':>13}  {'max_tc_mol1':>13}  "
          f"{'frac_with_mol1_scaf':>20}")
    mol1_bm = bm_scaffold(MOL1)
    print(f"        (parent BM scaffold: {mol1_bm})")
    rows_out = []
    for K in [20, 50, 100, 250, len(df)]:
        sub = df.head(K)
        n_scaf = sub["bm_scaffold"].nunique()
        sub_fps = sub["fp"].tolist()
        # cluster only first 250 (single-linkage is O(N^2))
        if K <= 250:
            cids = cluster_single_linkage(sub_fps, threshold=0.5)
            n_cluster = len(set(cids))
        else:
            n_cluster = float("nan")
        med_tc = sub["tc_to_mol1"].median()
        max_tc = sub["tc_to_mol1"].max()
        frac_mol1_scaf = (sub["bm_scaffold"] == mol1_bm).mean()
        print(f"{K:>6}  {n_scaf:>14}  {str(n_cluster):>18}  "
              f"{med_tc:>13.3f}  {max_tc:>13.3f}  "
              f"{frac_mol1_scaf:>20.2%}")
        rows_out.append({
            "top_K": K, "n_bm_scaffolds": n_scaf,
            "n_tc05_clusters": n_cluster, "median_tc_to_mol1": med_tc,
            "max_tc_to_mol1": max_tc, "frac_with_mol1_scaffold": frac_mol1_scaf,
        })

    # Show the top 5 BM scaffolds (most-frequent) in top-50
    top50 = df.head(50)
    print("\n=== Most-common BM scaffolds in top-50 ===")
    for scaf, count in top50["bm_scaffold"].value_counts().head(5).items():
        print(f"  {count:>3}× : {scaf}")

    # Save
    out_dir = PROJECT_ROOT / "results" / "anchordiff"
    pd.DataFrame(rows_out).to_csv(out_dir / "cys346_scaffold_diversity_audit.csv", index=False)
    keep_cols = ["yaml_name", "method", "smiles", "rank_score", "boltz_ligand_iptm",
                 "combined", "bm_scaffold", "tc_to_mol1"]
    df[keep_cols].to_csv(out_dir / "cys346_cofold_leaderboard_scaffolds.csv", index=False)
    print(f"\nwrote {out_dir/'cys346_scaffold_diversity_audit.csv'}")
    print(f"wrote {out_dir/'cys346_cofold_leaderboard_scaffolds.csv'}")


if __name__ == "__main__":
    main()
