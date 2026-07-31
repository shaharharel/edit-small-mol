"""D3 — Combined FiLMDelta + mPAE ranking on OUR minimum-viable cohort (1479 Boltz cofolds).

Inputs:
  - data/covalid_mv_cofolds/metrics.csv      ← mPAE/iPTM/pLDDT per cofold (from A100 batch)
  - experiments/boltz_inputs/covalid_minimum_viable/manifest.csv  ← target/name → SMILES, label
  - results/covalid/covalid_d1_per_mol_scores.json  ← per-mol FiLMDelta scores

Combiners evaluated per target (matching London-style adj_LogAUC):
  baseline 1: FiLM alone
  baseline 2: -mPAE alone (lower mPAE → more confident)
  (a) Rank-sum: rank FiLM↓ + rank mPAE↑ → adj_LogAUC
  (b) Linear blend: α·z(FiLM) + (1-α)·(-z(mPAE)), sweep α
  (c) Logistic regression: full-fit on (z_film, -z_mpae) → adj_LogAUC

Output: results/covalid/covalid_d3_mv_cofolds.json
"""
from __future__ import annotations
import sys, json, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent
METRICS_CSV = PROJECT_ROOT / "data" / "covalid_mv_cofolds" / "metrics.csv"
MANIFEST_CSV = PROJECT_ROOT / "experiments" / "boltz_inputs" / "covalid_minimum_viable" / "manifest.csv"
D1_PERMOL = PROJECT_ROOT / "results" / "covalid" / "covalid_d1_per_mol_scores.json"
OUT = PROJECT_ROOT / "results" / "covalid" / "covalid_d3_mv_cofolds.json"


def canon(smi: str) -> str | None:
    if not smi or not isinstance(smi, str): return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m)


def adj_log_auc(labels, scores, lam=10):
    """Mysinger/Shoichet adj_LogAUC — emphasizes early enrichment.
    Higher score = better. labels: 1=active, 0=decoy.

    Implementation matches Mysinger & Shoichet (2010): integrate TPR over log(FPR)
    from FPR=1/lam to FPR=1, then normalize so random=0%, perfect=100%.
    """
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order].astype(float)
    n_act = float(y.sum()); n_dec = float(len(y) - n_act)
    if n_act == 0 or n_dec == 0: return 0.0
    # ROC curve points (TPR vs FPR), prepend (0,0)
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_act])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / n_dec])
    # Restrict to FPR in [1/lam, 1]
    lo = 1.0 / lam
    mask = fpr >= lo
    if not mask.any(): return 0.0
    fpr_w = np.concatenate([[lo], fpr[mask]])
    # TPR at exactly fpr=lo: linear interp from prev point
    idx = np.searchsorted(fpr, lo, side="left")
    if idx == 0:
        tpr_lo = 0.0
    elif idx >= len(fpr):
        tpr_lo = tpr[-1]
    else:
        x0, x1 = fpr[idx - 1], fpr[idx]
        y0, y1 = tpr[idx - 1], tpr[idx]
        tpr_lo = y0 + (y1 - y0) * (lo - x0) / (x1 - x0) if x1 > x0 else y0
    tpr_w = np.concatenate([[tpr_lo], tpr[mask]])
    log_fpr = np.log10(fpr_w.clip(min=1e-12))
    auc_lambda = float(np.trapz(tpr_w, log_fpr))
    # Normalization: random ranker integrates uniformly, perfect = 100%
    auc_random = (1 - lo) / np.log(10)  # ∫_{1/lam}^1 x d(log10 x) = (1 - 1/lam)/ln(10)
    auc_perfect = -np.log10(lo)  # ∫_{1/lam}^1 1 d(log10 x) = log10(1/(1/lam)) = log10(lam)
    return 100.0 * (auc_lambda - auc_random) / (auc_perfect - auc_random)


def main():
    print("=" * 80)
    print("D3 — Combined FiLMDelta + mPAE ranking on COValid minimum-viable cohort")
    print("=" * 80)

    # 1. Load manifest (target/name → SMILES, is_active)
    manifest = pd.read_csv(MANIFEST_CSV)
    print(f"\nManifest: {len(manifest)} entries across {manifest['target'].nunique()} targets")

    # 2. Load metrics (Boltz cofold outputs)
    metrics = pd.read_csv(METRICS_CSV)
    print(f"Metrics: {len(metrics)} cofolds")

    # 3. Join manifest ↔ metrics on (target, name)
    df = manifest.merge(metrics, on=["target", "name"], how="inner")
    print(f"Joined: {len(df)} cofolds with both manifest+metrics")
    print("Per-target join counts:")
    print(df.groupby(["target", "is_active"]).size().unstack(fill_value=0))

    # Canonical SMILES
    df["smi_canon"] = df["smiles"].apply(canon)
    df = df.dropna(subset=["smi_canon"])

    # 4. Load D1 per-mol FiLMDelta scores
    print(f"\nLoading D1 FiLM scores from {D1_PERMOL.name}")
    with open(D1_PERMOL) as f:
        d1 = json.load(f)
    film_dict = {}  # (target, canon_smi) → film_score
    for tgt_label, payload in d1.items():
        if "per_mol" not in payload: continue
        for r in payload["per_mol"]:
            c = canon(r["smiles"])
            if c is None: continue
            film_dict[(tgt_label, c)] = float(r["score"])
    print(f"D1 dict: {len(film_dict)} (target, canon_smi) → film_score entries")
    print(f"D1 targets: {sorted({k[0] for k in film_dict.keys()})}")

    # 5. Match on (target, canon_smi)
    df["film"] = df.apply(lambda r: film_dict.get((r["target"], r["smi_canon"])), axis=1)

    # Pick mPAE column — prefer the interface-block estimate, fall back to mean of full PAE
    df["mpae"] = df["mpae_interface"].fillna(df["mpae_meanfull"])
    # ipde = complex_ipde is Boltz's predicted distance error at the interface
    df["ipde"] = df["complex_ipde"]

    # Drop rows missing either film or mpae
    matched = df.dropna(subset=["film", "mpae", "ipde"]).copy()
    print(f"\nMatched (have FiLM + mPAE + ipde): {len(matched)} / {len(df)}")
    print(matched.groupby(["target", "is_active"]).size().unstack(fill_value=0))

    # 6. Per-target combiners
    results = {"per_target": {}}
    for target, g in matched.groupby("target"):
        n_act = int(g["is_active"].sum())
        n_dec = int(len(g) - n_act)
        if n_act < 1 or n_dec < 1:
            print(f"\n=== {target}: skipped (need ≥1 act + ≥1 dec, got {n_act}+{n_dec}) ===")
            continue
        print(f"\n=== {target}: {n_act} act + {n_dec} dec, total {len(g)} ===")

        # Baselines (also report ROC AUC for sanity)
        from sklearn.metrics import roc_auc_score
        auc_film = roc_auc_score(g["is_active"], g["film"])
        auc_mpae = roc_auc_score(g["is_active"], -g["mpae"])
        auc_ipde = roc_auc_score(g["is_active"], -g["ipde"])
        adj_film = adj_log_auc(g["is_active"], g["film"])
        adj_mpae = adj_log_auc(g["is_active"], -g["mpae"])
        adj_ipde = adj_log_auc(g["is_active"], -g["ipde"])

        # (a) Rank-sum of film + (-mpae)
        rank_film = g["film"].rank(method="average")
        rank_mpae = (-g["mpae"]).rank(method="average")
        adj_ranksum = adj_log_auc(g["is_active"], rank_film + rank_mpae)
        # (a') Rank-sum of film + (-ipde) — uses ipde instead of mpae
        rank_ipde = (-g["ipde"]).rank(method="average")
        adj_ranksum_ipde = adj_log_auc(g["is_active"], rank_film + rank_ipde)

        # (b) Linear blend — sweep α
        z_film = StandardScaler().fit_transform(g[["film"]].values).flatten()
        z_mpae = -StandardScaler().fit_transform(g[["mpae"]].values).flatten()
        best_alpha, best_blend = 0.5, 0.0
        for alpha in np.arange(0.0, 1.05, 0.1):
            blend = alpha * z_film + (1 - alpha) * z_mpae
            a = adj_log_auc(g["is_active"], blend)
            if a > best_blend:
                best_blend, best_alpha = a, float(alpha)

        # (c) Logistic regression
        X = np.column_stack([z_film, z_mpae])
        try:
            lr = LogisticRegression(max_iter=1000).fit(X, g["is_active"].values)
            proba = lr.predict_proba(X)[:, 1]
            adj_lr = adj_log_auc(g["is_active"], proba)
            coefs = lr.coef_[0]
        except Exception:
            adj_lr = None
            coefs = [None, None]

        results["per_target"][target] = {
            "n_actives": n_act, "n_decoys": n_dec,
            "auc_film": float(auc_film), "adj_film": float(adj_film),
            "auc_mpae": float(auc_mpae), "adj_mpae": float(adj_mpae),
            "auc_ipde": float(auc_ipde), "adj_ipde": float(adj_ipde),
            "a_rank_sum_film_mpae": float(adj_ranksum),
            "a_rank_sum_film_ipde": float(adj_ranksum_ipde),
            "b_linear_blend_film_mpae": {"adj": float(best_blend), "best_alpha": best_alpha},
            "c_logistic_film_mpae": {
                "adj": float(adj_lr) if adj_lr is not None else None,
                "coef_film": float(coefs[0]) if coefs[0] is not None else None,
                "coef_neg_mpae": float(coefs[1]) if coefs[1] is not None else None,
            },
        }
        print(f"  FiLM alone:    AUC={auc_film:.3f}  adj_LogAUC={adj_film:5.1f}%")
        print(f"  -mPAE alone:   AUC={auc_mpae:.3f}  adj_LogAUC={adj_mpae:5.1f}%")
        print(f"  -ipde alone:   AUC={auc_ipde:.3f}  adj_LogAUC={adj_ipde:5.1f}%")
        print(f"  (a1) rank-sum FiLM+(-mPAE): {adj_ranksum:5.1f}%")
        print(f"  (a2) rank-sum FiLM+(-ipde): {adj_ranksum_ipde:5.1f}%")
        print(f"  (b)  blend FiLM/-mPAE:      {best_blend:5.1f}% (α={best_alpha:.1f})")
        if adj_lr is not None:
            print(f"  (c)  logistic FiLM,-mPAE:   {adj_lr:5.1f}%  (coef_film={coefs[0]:+.2f}, neg_mpae={coefs[1]:+.2f})")

    # Aggregate
    pt = results["per_target"]
    if pt:
        results["averages"] = {
            "auc_film":      float(np.mean([v["auc_film"] for v in pt.values()])),
            "auc_mpae":      float(np.mean([v["auc_mpae"] for v in pt.values()])),
            "auc_ipde":      float(np.mean([v["auc_ipde"] for v in pt.values()])),
            "adj_film":      float(np.mean([v["adj_film"] for v in pt.values()])),
            "adj_mpae":      float(np.mean([v["adj_mpae"] for v in pt.values()])),
            "adj_ipde":      float(np.mean([v["adj_ipde"] for v in pt.values()])),
            "a1_rank_film_mpae":     float(np.mean([v["a_rank_sum_film_mpae"] for v in pt.values()])),
            "a2_rank_film_ipde":     float(np.mean([v["a_rank_sum_film_ipde"] for v in pt.values()])),
            "b_blend_film_mpae":     float(np.mean([v["b_linear_blend_film_mpae"]["adj"] for v in pt.values()])),
            "c_logistic_film_mpae":  float(np.mean([v["c_logistic_film_mpae"]["adj"] for v in pt.values() if v["c_logistic_film_mpae"]["adj"] is not None])),
        }
        print("\n=== AVERAGE across matched targets ===")
        print("  Metric                       adj_LogAUC  (also ROC AUC where applicable)")
        for k, v in results["averages"].items():
            print(f"  {k:30s} {v:5.1f}{'%' if 'auc' not in k else ''}")
        print(f"  (London JACS 2025 reported avg: 71.8%)")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
