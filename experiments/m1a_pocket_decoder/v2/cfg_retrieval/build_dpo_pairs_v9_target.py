"""v9 per-target DPO pair builder from a target's baseline sample panel.

Input:
  covalent_metric_panel_{target}_baseline.csv
  covalent_vina_cov_panel_{target}_baseline.csv

Winner = LOWER vina_cov_score.
Global quantile split (top 40% vs bottom 40%), min_gap default 10 kcal/mol,
cap 300 pairs per target.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", required=True)
    ap.add_argument("--panel_2d", required=True)
    ap.add_argument("--panel_vina", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--winner_quantile", type=float, default=0.40)
    ap.add_argument("--loser_quantile", type=float, default=0.60)
    ap.add_argument("--n_pairs", type=int, default=300)
    ap.add_argument("--min_gap", type=float, default=10.0)
    args = ap.parse_args()

    p2 = pd.read_csv(args.panel_2d)
    pv = pd.read_csv(args.panel_vina)
    m = p2.merge(pv[["cell", "sample_idx", "vina_cov_score", "vina_cov_ok"]],
                  on=["cell", "sample_idx"], how="inner")
    m = m[(m["valid"] == True) & (m["acryl_largest"] == True) &
            (m["vina_cov_ok"] == True)].copy()
    print(f"[{args.target}] {len(m)} valid+acryl+vina_ok rows", flush=True)

    if len(m) < 20:
        print(f"[{args.target}] too few mols for DPO; skipping.", flush=True)
        pd.DataFrame().to_csv(args.out_csv, index=False)
        return

    vs = m["vina_cov_score"].astype(float)
    w_thresh = vs.quantile(args.winner_quantile)
    l_thresh = vs.quantile(args.loser_quantile)
    winners = m[vs <= w_thresh].sort_values("vina_cov_score",
                                                ascending=True).reset_index(drop=True)
    losers = m[vs >= l_thresh].sort_values("vina_cov_score",
                                               ascending=False).reset_index(drop=True)
    print(f"  {len(winners)} winners, {len(losers)} losers", flush=True)

    pairs = []
    rng = np.random.default_rng(0)
    w_idx = rng.permutation(len(winners))
    l_idx = rng.permutation(len(losers))
    i = 0
    max_iters = max(len(w_idx), len(l_idx)) * 6
    while len(pairs) < args.n_pairs and i < max_iters:
        wi = w_idx[i % len(w_idx)]; li = l_idx[i % len(l_idx)]
        w = winners.iloc[wi]; l = losers.iloc[li]
        i += 1
        if str(w["largest_frag_SMILES"]) == str(l["largest_frag_SMILES"]):
            continue
        gap = float(l["vina_cov_score"] - w["vina_cov_score"])
        if gap < args.min_gap: continue
        pairs.append({
            "target": args.target,
            "cell_source": f"{w['cell']}|{l['cell']}",
            "winner_smi": str(w["largest_frag_SMILES"]),
            "winner_score": float(w["vina_cov_score"]),
            "loser_smi": str(l["largest_frag_SMILES"]),
            "loser_score": float(l["vina_cov_score"]),
            "score_gap": gap,
        })
    pdf = pd.DataFrame(pairs)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(args.out_csv, index=False)
    print(f"Wrote {args.out_csv} ({len(pdf)} pairs)", flush=True)


if __name__ == "__main__":
    main()
