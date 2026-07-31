"""D3 with LONDON's mPAE definition: min(PAE[protein_rows, ligand_cols]).

Reproduces COValid-PDB Table S2 (per-target adj_LogAUC with mPAE worst-case),
using our Boltz-2 cofolds instead of London's AF3.

London Table S2 (AF3-mPAE worst-case):
  BMX 56.4 | FGFR1 79.4 | FGFR4_477 83.8 | FGFR4_552 82.5 | JAK3 71.6 |
  EGFR 68.1 | MAP3K7 78.4 | KRAS 74.7 | BTK 72.3 | ITK 65.8

Ours (Boltz-2 mPAE_min) — to be compared:
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).parent.parent
WARHEAD_CSV = PROJECT_ROOT / "data" / "covalid_mv_cofolds" / "warhead_metrics.csv"
MANIFEST_CSV = PROJECT_ROOT / "experiments" / "boltz_inputs" / "covalid_minimum_viable" / "manifest.csv"
D1_PERMOL = PROJECT_ROOT / "results" / "covalid" / "covalid_d1_per_mol_scores.json"
OUT = PROJECT_ROOT / "results" / "covalid" / "covalid_d3_london_mpae.json"

# London Table S2 (per-target adj_LogAUC % for AF3-mPAE, worst-case-protomer)
LONDON_TABLE_S2 = {
    "BMX": 56.4, "FGFR1": 79.4, "FGFR4_477": 83.8, "FGFR4_552": 82.5,
    "JAK3": 71.6, "EGFR": 68.1, "MAP3K7": 78.4, "KRAS": 74.7,
    "BTK": 72.3, "ITK": 65.8,
}


def canon(smi):
    try: return Chem.MolToSmiles(Chem.MolFromSmiles(smi))
    except: return None


def adj_log_auc(y, s, lam=1000):
    order = np.argsort(-np.asarray(s))
    y = np.asarray(y)[order].astype(float)
    n_act, n_dec = float(y.sum()), float(len(y) - y.sum())
    if n_act == 0 or n_dec == 0: return 0.0
    tpr = np.concatenate([[0.0], np.cumsum(y) / n_act])
    fpr = np.concatenate([[0.0], np.cumsum(1 - y) / n_dec])
    lo = 1.0 / lam
    mask = fpr >= lo
    if not mask.any(): return 0.0
    idx = np.searchsorted(fpr, lo, side="left")
    if idx == 0: tpr_lo = 0.0
    elif idx >= len(fpr): tpr_lo = tpr[-1]
    else:
        x0, x1 = fpr[idx-1], fpr[idx]; y0, y1 = tpr[idx-1], tpr[idx]
        tpr_lo = y0 + (y1-y0)*(lo-x0)/(x1-x0) if x1 > x0 else y0
    fpr_w = np.concatenate([[lo], fpr[mask]])
    tpr_w = np.concatenate([[tpr_lo], tpr[mask]])
    auc_lam = float(np.trapz(tpr_w, np.log10(fpr_w.clip(min=1e-12))))
    auc_rand = (1 - lo) / np.log(10)
    auc_perf = -np.log10(lo)
    return 100.0 * (auc_lam - auc_rand) / (auc_perf - auc_rand)


def main():
    print("=" * 90)
    print("D3 with London's mPAE definition (min over protein × ligand PAE block)")
    print("Comparing our Boltz-2 to London's AF3 (Table S2)")
    print("=" * 90)

    manifest = pd.read_csv(MANIFEST_CSV)
    metrics = pd.read_csv(WARHEAD_CSV)
    df = manifest.merge(metrics, on=["target", "name"], how="inner")
    df["smi_canon"] = df["smiles"].apply(canon)
    df = df.dropna(subset=["smi_canon", "mpae_london_min"])
    print(f"\nCofolds with London-style mPAE: {len(df)}")

    # FiLM
    with open(D1_PERMOL) as f: d1 = json.load(f)
    film = {(t, canon(r["smiles"])): float(r["score"])
            for t, p in d1.items() if "per_mol" in p
            for r in p["per_mol"] if canon(r["smiles"])}
    df["film"] = df.apply(lambda r: film.get((r["target"], r["smi_canon"])), axis=1)

    # Worst-case-protomer reduction: for each (target, active_compound), pick HIGHEST mpae
    # We don't have a "compound_id" linking protomers — manifest treats each YAML as
    # a single protomer. Without protomer grouping we evaluate at the protomer level
    # (best-case ≈ worst-case here).
    df["mpae"] = df["mpae_london_min"].astype(float)

    results = {"per_target": {}, "london_table_S2": LONDON_TABLE_S2}
    print(f"\n{'target':12s}  {'n_act':>5s}  {'n_dec':>5s}    {'AUC_mpae':>9s}  "
          f"{'adj_us':>7s}  {'adj_L':>7s}  {'gap':>6s}    {'AUC_film':>9s}  {'adj_film':>9s}")
    print("-" * 110)

    rows_summary = []
    for tgt, g in df.groupby("target"):
        if g["is_active"].nunique() < 2: continue
        y = g["is_active"]
        auc_m = roc_auc_score(y, -g["mpae"])
        adj_m = adj_log_auc(y, -g["mpae"])
        adj_L = LONDON_TABLE_S2.get(tgt)
        gap = adj_m - adj_L if adj_L is not None else None

        film_subset = g.dropna(subset=["film"])
        if len(film_subset) > 0 and film_subset["is_active"].nunique() >= 2:
            auc_f = roc_auc_score(film_subset["is_active"], film_subset["film"])
            adj_f = adj_log_auc(film_subset["is_active"], film_subset["film"])
        else:
            auc_f, adj_f = None, None

        # Combiner: linear blend of z(film) + z(-mpae), sweep alpha
        if len(film_subset) > 0:
            zf = StandardScaler().fit_transform(film_subset[["film"]].values).flatten()
            zm = -StandardScaler().fit_transform(film_subset[["mpae"]].values).flatten()
            best_blend, best_alpha = -float("inf"), 0.5
            for alpha in np.arange(0.0, 1.05, 0.1):
                b = alpha * zf + (1 - alpha) * zm
                a = adj_log_auc(film_subset["is_active"], b)
                if a > best_blend:
                    best_blend, best_alpha = a, float(alpha)
            adj_lr = None
            try:
                X = np.column_stack([zf, zm])
                lr = LogisticRegression(max_iter=1000).fit(X, film_subset["is_active"].values)
                adj_lr = adj_log_auc(film_subset["is_active"], lr.predict_proba(X)[:, 1])
            except Exception:
                pass
        else:
            best_blend, best_alpha, adj_lr = None, None, None

        results["per_target"][tgt] = {
            "n_act": int(y.sum()), "n_dec": int((1 - y).sum()),
            "auc_mpae_ours": float(auc_m), "adj_mpae_ours": float(adj_m),
            "adj_mpae_london_S2": adj_L, "gap_us_minus_london": float(gap) if gap is not None else None,
            "auc_film": float(auc_f) if auc_f is not None else None,
            "adj_film": float(adj_f) if adj_f is not None else None,
            "adj_blend_best": float(best_blend) if best_blend is not None else None,
            "blend_best_alpha": best_alpha,
            "adj_logistic": float(adj_lr) if adj_lr is not None else None,
        }
        rows_summary.append((tgt, int(y.sum()), int((1-y).sum()), auc_m, adj_m, adj_L, gap, auc_f, adj_f, best_blend, adj_lr))
        adj_l_str = f"{adj_L:>6.1f}%" if adj_L is not None else "  N/A"
        gap_str = f"{gap:+5.1f}" if gap is not None else "  N/A"
        auc_f_str = f"{auc_f:>9.3f}" if auc_f is not None else "      N/A"
        adj_f_str = f"{adj_f:>7.1f}%" if adj_f is not None else "    N/A"
        print(f"{tgt:12s}  {int(y.sum()):>5d}  {int((1-y).sum()):>5d}    "
              f"{auc_m:>9.3f}  {adj_m:>6.1f}%  {adj_l_str}  {gap_str}    "
              f"{auc_f_str}  {adj_f_str}")

    # Aggregates
    if rows_summary:
        adj_us = np.array([r[4] for r in rows_summary])
        adj_L_arr = np.array([r[5] for r in rows_summary if r[5] is not None])
        results["averages"] = {
            "ours_avg_adj_mpae": float(adj_us.mean()),
            "london_avg_adj_mpae_S2": float(adj_L_arr.mean()),
            "avg_gap_us_minus_london": float(adj_us.mean() - adj_L_arr.mean()),
            "ours_avg_auc_mpae": float(np.mean([r[3] for r in rows_summary])),
        }
        print("-" * 110)
        print(f"AVG (matched targets):                             "
              f"{np.mean([r[3] for r in rows_summary]):>9.3f}  "
              f"{adj_us.mean():>6.1f}%  {adj_L_arr.mean():>6.1f}%  "
              f"{adj_us.mean() - adj_L_arr.mean():+5.1f}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
