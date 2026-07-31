"""v8 DPO scale-up pair builder — 10K pairs from --score_only sources.

Extension of v6-FIXED strategy=global. Widens quantile thresholds (0.30/0.70
vs 0.40/0.60) AND expands source pool with v5 K=10, K=20 samples that
weren't in v6's build.

Winner = LOWER --score_only vina_cov_score (better binder, subject to
tether-clash inflation).
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")


def _load_source(panel_2d, vina, tag):
    if not Path(panel_2d).exists() or not Path(vina).exists():
        print(f"[skip {tag}] missing", flush=True)
        return None
    p2 = pd.read_csv(panel_2d)
    pv = pd.read_csv(vina)
    m = p2.merge(pv[["cell", "sample_idx", "vina_cov_score", "vina_cov_ok"]],
                  on=["cell", "sample_idx"], how="inner")
    m = m[(m["valid"] == True) & (m["acryl_largest"] == True)
            & (m["vina_cov_ok"] == True)].copy()
    m["source"] = tag
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_csv", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/cfg_retrieval_v8_scaleup/"
                     "dpo_pairs_scaleup.csv"))
    ap.add_argument("--winner_quantile", type=float, default=0.30)
    ap.add_argument("--loser_quantile", type=float, default=0.70)
    ap.add_argument("--global_n_pairs", type=int, default=10000)
    ap.add_argument("--min_gap", type=float, default=5.0)
    args = ap.parse_args()

    sources = [
        ("v3_p0.10", "cfg_retrieval/covalent_metric_panel_v2.csv",
                        "cfg_retrieval/covalent_vina_cov_panel.csv"),
        ("v3_p0.05", "cfg_retrieval_p05/covalent_metric_panel_p05.csv",
                        "cfg_retrieval_p05/covalent_vina_cov_panel_p05.csv"),
        ("v4_film",  "cfg_retrieval_v4_film/covalent_metric_panel_v4.csv",
                        "cfg_retrieval_v4_film/covalent_vina_cov_panel_v4.csv"),
        ("v5_K",     "cfg_retrieval_v5_retrievalK/covalent_metric_panel_v5.csv",
                        "cfg_retrieval_v5_retrievalK/covalent_vina_cov_panel_v5.csv"),
    ]
    dfs = []
    for tag, p2, pv in sources:
        m = _load_source(PROJECT_ROOT / "data/paper_pair_training" / p2,
                            PROJECT_ROOT / "data/paper_pair_training" / pv, tag)
        if m is not None:
            dfs.append(m)
            print(f"[loaded {tag}] {len(m)} valid+acryl+vina_ok rows",
                   flush=True)
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
    max_iters = max(len(w_idx), len(l_idx)) * 6
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
    if len(pdf):
        print(f"Per source_family: {pdf['source_family'].value_counts().to_dict()}",
               flush=True)
        print(f"Score gap: median={pdf['score_gap'].median():.2f} "
               f"min={pdf['score_gap'].min():.2f} "
               f"max={pdf['score_gap'].max():.2f} kcal/mol", flush=True)


if __name__ == "__main__":
    main()
