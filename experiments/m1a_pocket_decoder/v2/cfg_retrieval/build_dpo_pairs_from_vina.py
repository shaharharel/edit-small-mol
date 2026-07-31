"""Build DPO preference pairs from all v3+v4 samples with cov-Vina scores.

Sources (all in data/paper_pair_training/*):
  cfg_retrieval/covalent_vina_cov_panel.csv           (v3 p0.10, 570 mols)
  cfg_retrieval_p05/covalent_vina_cov_panel_p05.csv   (v3 p0.05, 583 mols)
  cfg_retrieval_v4_film/covalent_vina_cov_panel_v4.csv (v4, 800 mols)

Merge with each panel's 2D file to get the SMILES per (cell, sample_idx).
For each SOURCE CELL, split by tethered vina_cov_score:
  - winner = top-quartile score (better binder ⇒ LOWER kcal/mol)
  - loser  = bottom-quartile score (worse binder)
Then take (winner, loser) pairs: n_pairs = min(n_winners, n_losers).
Per coordinator directive: cap ~25 winners × 25 losers per cell.

Output: data/paper_pair_training/cfg_retrieval_v6_dpo/dpo_pairs.csv
    columns: cell_source, winner_smi, loser_smi, winner_score, loser_score
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def _load_source(panel_2d_csv, vina_csv, tag):
    if not Path(panel_2d_csv).exists() or not Path(vina_csv).exists():
        print(f"[skip {tag}] missing files", flush=True)
        return None
    p2 = pd.read_csv(panel_2d_csv)
    pv = pd.read_csv(vina_csv)
    # Merge (cell, sample_idx).
    m = p2.merge(pv[["cell", "sample_idx", "vina_cov_score", "vina_cov_ok"]],
                  on=["cell", "sample_idx"], how="inner")
    m = m[(m["valid"] == True) & (m["acryl_largest"] == True) &
            (m["vina_cov_ok"] == True)].copy()
    m["source"] = tag
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v6_dpo/dpo_pairs.csv"))
    ap.add_argument("--strategy", default="global",
                    choices=["per_cell", "global"],
                    help="per_cell: original buggy v6 behavior (~338 pairs). "
                          "global: pool across all sources, pair top-N vs "
                          "bottom-N of GLOBAL vina_cov_score distribution "
                          "-> ~2000+ pairs.")
    ap.add_argument("--per_cell_cap", type=int, default=25)
    ap.add_argument("--winner_quantile", type=float, default=0.40,
                    help="Loosened from 0.25 per QA #8.")
    ap.add_argument("--loser_quantile", type=float, default=0.60,
                    help="Loosened from 0.75 per QA #8.")
    ap.add_argument("--global_n_pairs", type=int, default=2000,
                    help="Cap on total pairs when strategy=global.")
    ap.add_argument("--min_gap", type=float, default=10.0,
                    help="Minimum |loser_score - winner_score| in kcal/mol.")
    args = ap.parse_args()

    sources = []
    for tag, p2, pv in [
        ("v3_p0.10", "cfg_retrieval/covalent_metric_panel_v2.csv",
                        "cfg_retrieval/covalent_vina_cov_panel.csv"),
        ("v3_p0.05", "cfg_retrieval_p05/covalent_metric_panel_p05.csv",
                        "cfg_retrieval_p05/covalent_vina_cov_panel_p05.csv"),
        ("v4_film",  "cfg_retrieval_v4_film/covalent_metric_panel_v4.csv",
                        "cfg_retrieval_v4_film/covalent_vina_cov_panel_v4.csv"),
    ]:
        m = _load_source(PROJECT_ROOT / "data/paper_pair_training" / p2,
                            PROJECT_ROOT / "data/paper_pair_training" / pv, tag)
        if m is not None:
            sources.append(m)
            print(f"[loaded {tag}] {len(m)} valid+acryl+vina_ok rows", flush=True)

    if not sources:
        raise SystemExit("No sources available.")
    all_df = pd.concat(sources, ignore_index=True)
    print(f"Total pool: {len(all_df)} rows across {all_df['cell'].nunique()} source cells",
           flush=True)

    pairs = []
    if args.strategy == "per_cell":
        for cell, cd in all_df.groupby("cell"):
            vs = cd["vina_cov_score"].astype(float)
            w_thresh = vs.quantile(args.winner_quantile)
            l_thresh = vs.quantile(args.loser_quantile)
            winners = cd[vs <= w_thresh].sort_values("vina_cov_score",
                                                        ascending=True)
            losers = cd[vs >= l_thresh].sort_values("vina_cov_score",
                                                       ascending=False)
            n_pairs = min(len(winners), len(losers), args.per_cell_cap)
            for i in range(n_pairs):
                w = winners.iloc[i]; l = losers.iloc[i]
                if str(w["largest_frag_SMILES"]) == str(l["largest_frag_SMILES"]):
                    continue
                pairs.append({
                    "cell_source": cell,
                    "source_family": str(cd.iloc[0]["source"]),
                    "winner_smi": str(w["largest_frag_SMILES"]),
                    "winner_score": float(w["vina_cov_score"]),
                    "loser_smi": str(l["largest_frag_SMILES"]),
                    "loser_score": float(l["vina_cov_score"]),
                    "score_gap": float(l["vina_cov_score"] - w["vina_cov_score"]),
                })
    else:
        # GLOBAL strategy: pool across all sources, split by GLOBAL quantile.
        vs = all_df["vina_cov_score"].astype(float)
        w_thresh = vs.quantile(args.winner_quantile)
        l_thresh = vs.quantile(args.loser_quantile)
        winners = all_df[vs <= w_thresh].sort_values("vina_cov_score",
                                                        ascending=True).reset_index(drop=True)
        losers = all_df[vs >= l_thresh].sort_values("vina_cov_score",
                                                       ascending=False).reset_index(drop=True)
        print(f"[global] {len(winners)} winners  {len(losers)} losers  "
               f"(thresh w<={w_thresh:.2f} l>={l_thresh:.2f})", flush=True)
        rng = np.random.default_rng(0)
        w_idx = rng.permutation(len(winners))
        l_idx = rng.permutation(len(losers))
        i = 0
        max_iters = max(len(w_idx), len(l_idx)) * 3
        while len(pairs) < args.global_n_pairs and i < max_iters:
            wi = w_idx[i % len(w_idx)]
            li = l_idx[i % len(l_idx)]
            w = winners.iloc[wi]; l = losers.iloc[li]
            i += 1
            if str(w["largest_frag_SMILES"]) == str(l["largest_frag_SMILES"]):
                continue
            gap = float(l["vina_cov_score"] - w["vina_cov_score"])
            if gap < args.min_gap:
                continue
            pairs.append({
                "cell_source": f"{w['cell']}|{l['cell']}",
                "source_family": str(w["source"]),
                "winner_smi": str(w["largest_frag_SMILES"]),
                "winner_score": float(w["vina_cov_score"]),
                "loser_smi": str(l["largest_frag_SMILES"]),
                "loser_score": float(l["vina_cov_score"]),
                "score_gap": gap,
            })
    pdf = pd.DataFrame(pairs)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv} ({len(pdf)} pairs)", flush=True)
    print(f"Per source_family: {pdf['source_family'].value_counts().to_dict()}",
           flush=True)
    print(f"Score gap median: {pdf['score_gap'].median():.2f} kcal/mol",
           flush=True)
    print(f"Score gap min:    {pdf['score_gap'].min():.2f}", flush=True)


if __name__ == "__main__":
    main()
