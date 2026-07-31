"""v7 DPO pair builder: from --local_only Vina cov scores.

Sources (all real, existing on ai-gpu):
  v3_p10: cfg_retrieval/covalent_vina_local_panel_p10.csv (154 rows)
  v4_film: cfg_retrieval_v4_film/covalent_vina_local_panel_v4.csv (100)
  v5_K:   cfg_retrieval_v5_retrievalK/covalent_vina_local_panel_v5.csv (125)

Total ~380 --local_only-rescored mols with absolute affinities on the usual
−N kcal/mol scale (typical range −8 .. +120; medians ~-0.9 kcal/mol).

For each source's SMILES, we also need the source's 2D panel (to confirm
valid+acryl and get the SMILES field — the local_only CSV already carries
`smi`).

Pairing strategy (GLOBAL): pool all local_only-rescored mols, sort by
vina_cov_score, split by quantile, then pair top-N winners with bottom-N
losers.

Winner = LOWER Vina cov score (better binder).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v7_dpo_local/"
                     "dpo_pairs_local.csv"))
    ap.add_argument("--winner_quantile", type=float, default=0.25,
                    help="Fraction below (lower Vina cov = better).")
    ap.add_argument("--loser_quantile", type=float, default=0.75)
    ap.add_argument("--global_n_pairs", type=int, default=800)
    ap.add_argument("--min_gap", type=float, default=3.0,
                    help="Minimum |loser - winner| in kcal/mol on --local_only "
                          "scale (looser than v6's 10 because local_only scores "
                          "have tighter dynamic range).")
    args = ap.parse_args()

    sources = [
        ("v3_p10", PROJECT_ROOT /
            "data/paper_pair_training/cfg_retrieval/covalent_vina_local_panel_p10.csv"),
        ("v4_film", PROJECT_ROOT /
            "data/paper_pair_training/cfg_retrieval_v4_film/covalent_vina_local_panel_v4.csv"),
        ("v5_K",   PROJECT_ROOT /
            "data/paper_pair_training/cfg_retrieval_v5_retrievalK/covalent_vina_local_panel_v5.csv"),
    ]
    dfs = []
    for tag, p in sources:
        if not p.exists():
            print(f"[skip {tag}] no {p}", flush=True)
            continue
        d = pd.read_csv(p)
        d = d[d["vina_cov_ok"] == True].copy()
        d["source"] = tag
        # `smi` column carries the SMILES for these local_only panels.
        if "smi" not in d.columns and "SMILES" in d.columns:
            d["smi"] = d["SMILES"]
        dfs.append(d[["cell", "sample_idx", "smi", "vina_cov_score", "source"]])
        print(f"[loaded {tag}] {len(d)} rows", flush=True)
    if not dfs:
        raise SystemExit("No local_only panels found.")
    all_df = pd.concat(dfs, ignore_index=True)
    print(f"Total pool: {len(all_df)} rows across {all_df['cell'].nunique()} cells "
           f"from {all_df['source'].nunique()} sources", flush=True)

    vs = all_df["vina_cov_score"].astype(float)
    w_thresh = vs.quantile(args.winner_quantile)
    l_thresh = vs.quantile(args.loser_quantile)
    winners = all_df[vs <= w_thresh].sort_values("vina_cov_score",
                                                     ascending=True).reset_index(drop=True)
    losers = all_df[vs >= l_thresh].sort_values("vina_cov_score",
                                                    ascending=False).reset_index(drop=True)
    print(f"[global] {len(winners)} winners (score<={w_thresh:.2f}) "
           f"{len(losers)} losers (score>={l_thresh:.2f})", flush=True)

    pairs = []
    rng = np.random.default_rng(0)
    w_idx = rng.permutation(len(winners))
    l_idx = rng.permutation(len(losers))
    i = 0
    max_iters = max(len(w_idx), len(l_idx)) * 4
    while len(pairs) < args.global_n_pairs and i < max_iters:
        wi = w_idx[i % len(w_idx)]
        li = l_idx[i % len(l_idx)]
        w = winners.iloc[wi]; l = losers.iloc[li]
        i += 1
        if str(w["smi"]) == str(l["smi"]):
            continue
        gap = float(l["vina_cov_score"] - w["vina_cov_score"])
        if gap < args.min_gap:
            continue
        pairs.append({
            "cell_source": f"{w['cell']}|{l['cell']}",
            "source_family": str(w["source"]),
            "winner_smi": str(w["smi"]),
            "winner_score": float(w["vina_cov_score"]),
            "loser_smi": str(l["smi"]),
            "loser_score": float(l["vina_cov_score"]),
            "score_gap": gap,
        })
    pdf = pd.DataFrame(pairs)
    Path(args.out_csv).parent.mkdir(parents=True, exist_ok=True)
    pdf.to_csv(args.out_csv, index=False)
    print(f"\nWrote {args.out_csv} ({len(pdf)} pairs)", flush=True)
    if len(pdf):
        print(f"Score gap: median={pdf['score_gap'].median():.2f} "
               f"min={pdf['score_gap'].min():.2f} "
               f"max={pdf['score_gap'].max():.2f} kcal/mol", flush=True)
        print(f"Winner score: median={pdf['winner_score'].median():.2f} "
               f"min={pdf['winner_score'].min():.2f}", flush=True)
        print(f"Loser score: median={pdf['loser_score'].median():.2f} "
               f"max={pdf['loser_score'].max():.2f}", flush=True)


if __name__ == "__main__":
    main()
