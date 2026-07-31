#!/usr/bin/env python3
"""D3 — Combined FiLMDelta + mPAE ranking on London COValid (JACS 2025).

Three combiner approaches (the user requested a/b/c):
  a) Rank-sum: rank both signals, sum the ranks. Robust, no tuning.
  b) Linear blend: α·zscore(FiLM) + (1-α)·(-zscore(mPAE)). Sweep α∈[0,1] per target.
  c) Logistic regression: fit on (FiLM_z, -mPAE_z) → active label. Full-data fit.

Per-target mPAE values come from London's si_005.xlsx (899K rows, full benchmark).
FiLMDelta scores come from re-running D1 (kinase-pretrain + per-target fine-tune).

Output: results/covalid/covalid_d3_combined_ranking.json
"""
from __future__ import annotations
import sys, json, gc, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from rdkit import Chem, RDLogger
RDLogger.DisableLog('rdApp.*')

PROJECT_ROOT = Path(__file__).parent.parent
COVALID_SI4 = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_004.xlsx"
COVALID_SI5 = PROJECT_ROOT / "data" / "covalid" / "ja5c22222_si_005.xlsx"
D1_OUTPUT = PROJECT_ROOT / "results" / "covalid" / "covalid_d1_per_mol_scores.json"
OUT_DIR = PROJECT_ROOT / "results" / "covalid"

TARGETS = list({
    "BMX":       "CHEMBL2581",
    "BTK":       "CHEMBL5251",
    "FGFR1":     "CHEMBL3650",
    "JAK3":      "CHEMBL2148",
    "KRAS":      "CHEMBL2189121",
    "EGFR":      "CHEMBL203",
    "FGFR4_477": "CHEMBL3973",
    "FGFR4_552": "CHEMBL3973",
}.keys())


def canon(smi: str) -> str | None:
    """Canonical SMILES for cross-source matching."""
    if not smi or not isinstance(smi, str): return None
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m)


def adj_log_auc(labels, scores, lam=10):
    """Mysinger/Shoichet adj_LogAUC — emphasizes early enrichment.
    Higher score = better. labels: 1=active, 0=decoy."""
    order = np.argsort(-np.asarray(scores))
    y = np.asarray(labels)[order]
    n = len(y); n_act = int(y.sum()); n_dec = n - n_act
    if n_act == 0 or n_dec == 0: return 0.0, 0.0
    fpr_lam = lam / n_dec
    cum_act = np.cumsum(y) / n_act
    cum_dec = np.cumsum(1 - y) / n_dec
    mask = (cum_dec >= fpr_lam / lam) & (cum_dec <= 1)
    if not mask.any(): return 0.0, float(cum_act.max())
    x = cum_dec[mask]; y2 = cum_act[mask]
    log_x = np.log10(x.clip(min=1e-6))
    raw_auc = float(np.trapezoid(y2, x))
    # adj logAUC normalization (per Mysinger)
    lim = np.log10(1 / lam)
    auc_lambda = float(np.trapezoid(y2, log_x))
    auc_random = (1 - 1 / lam) / np.log(10) * np.log10(lam)
    adj = 100 * (auc_lambda - auc_random) / (lim * (1 - 1 / lam) - auc_random)
    return adj, raw_auc


def load_si005_mpae_dict(verbose=True):
    """Load si_005.xlsx (899K rows) into a dict {canonical_smiles: median_mpae}.
    Multiple rows per SMILES (different protomers/conformers) → take median."""
    import openpyxl
    if verbose: print(f"Loading si_005 mPAE dict from {COVALID_SI5}…")
    wb = openpyxl.load_workbook(COVALID_SI5, read_only=True)
    sh = wb['test']
    smi_to_mpaes: dict[str, list[float]] = {}
    n = 0
    t0 = time.time()
    for i, row in enumerate(sh.iter_rows(values_only=True)):
        if i == 0: continue  # header
        if row is None or len(row) < 3: continue
        mol_name, mpae, smi = row[0], row[1], row[2]
        if smi is None or mpae is None: continue
        c = canon(smi)
        if c is None: continue
        smi_to_mpaes.setdefault(c, []).append(float(mpae))
        n += 1
        if verbose and i % 100000 == 0:
            print(f"  loaded {i:,} rows, {len(smi_to_mpaes):,} unique canonical SMILES, {time.time()-t0:.0f}s")
    out = {k: float(np.median(v)) for k, v in smi_to_mpaes.items()}
    if verbose:
        print(f"  total rows={n:,}  unique canonical SMILES={len(out):,}  in {time.time()-t0:.0f}s")
    return out


def main():
    print("=" * 80); print("COValid D3 — Combined FiLMDelta + mPAE ranking"); print("=" * 80)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Load per-mol FiLMDelta scores from D1 (regenerate if missing)
    if not D1_OUTPUT.exists():
        sys.exit(f"Missing per-mol D1 output at {D1_OUTPUT}. "
                 f"Run experiments/run_covalid_d1_filmdelta.py first with --save-per-mol")
    print(f"Loading D1 per-mol scores from {D1_OUTPUT}")
    with open(D1_OUTPUT) as f:
        d1 = json.load(f)

    # 2. Build SMILES → mPAE dict from si_005
    mpae_dict = load_si005_mpae_dict()

    # 3. Per target: match D1 per-mol scores to mPAE
    results = {'per_target': {}, 'combiners': ['rank_sum', 'linear_blend', 'logistic_regression']}
    for target, payload in d1.items():
        if 'per_mol' not in payload: continue
        rows = payload['per_mol']  # list of {smiles, label, score}
        # Match mPAE by canonical SMILES
        matched = []
        for r in rows:
            c = canon(r['smiles'])
            if c is None: continue
            mp = mpae_dict.get(c)
            if mp is None: continue
            matched.append({'smiles': c, 'label': int(r['label']),
                            'film': float(r['score']), 'mpae': float(mp)})
        if len(matched) < 20:
            print(f"=== {target}: only {len(matched)} matched mols — SKIP")
            continue
        df = pd.DataFrame(matched)
        n_act = int(df.label.sum()); n_dec = len(df) - n_act
        print(f"\n=== {target}: matched {len(df)} mols ({n_act} actives, {n_dec} decoys) ===")

        # Compute baselines + 3 combiners
        # baseline 1: FiLM alone
        adj_film, _ = adj_log_auc(df.label, df.film)
        # baseline 2: -mPAE alone (lower mPAE → more confident active)
        adj_mpae, _ = adj_log_auc(df.label, -df.mpae)

        # a) Rank-sum: rank each, sum ranks
        rank_film = df.film.rank(method='average')
        rank_mpae = (-df.mpae).rank(method='average')
        rank_sum = rank_film + rank_mpae
        adj_ranksum, _ = adj_log_auc(df.label, rank_sum)

        # b) Linear blend: sweep α
        z_film = StandardScaler().fit_transform(df[['film']].values).flatten()
        z_mpae = StandardScaler().fit_transform(df[['mpae']].values).flatten() * -1  # higher = better
        best_alpha = 0.5; best_adj_blend = 0.0
        for alpha in np.arange(0.0, 1.05, 0.1):
            blend = alpha * z_film + (1 - alpha) * z_mpae
            adj_, _ = adj_log_auc(df.label, blend)
            if adj_ > best_adj_blend:
                best_adj_blend = adj_; best_alpha = float(alpha)

        # c) Logistic regression (full-fit; not ideal but shows upper bound)
        X = np.column_stack([z_film, z_mpae])
        lr = LogisticRegression(max_iter=1000).fit(X, df.label.values)
        proba = lr.predict_proba(X)[:, 1]
        adj_lr, _ = adj_log_auc(df.label, proba)
        coefs = lr.coef_[0]

        results['per_target'][target] = {
            'n_actives': n_act, 'n_decoys': n_dec, 'n_matched': len(df),
            'adj_film_alone': float(adj_film),
            'adj_mpae_alone': float(adj_mpae),
            'a_rank_sum':     float(adj_ranksum),
            'b_linear_blend': {'adj': float(best_adj_blend), 'best_alpha': best_alpha},
            'c_logistic':     {'adj': float(adj_lr),
                                'coef_film': float(coefs[0]),
                                'coef_neg_mpae': float(coefs[1])},
        }
        print(f"  FiLM alone:   {adj_film:.1f}%")
        print(f"  mPAE alone:   {adj_mpae:.1f}%")
        print(f"  (a) rank-sum: {adj_ranksum:.1f}%")
        print(f"  (b) blend:    {best_adj_blend:.1f}% (α={best_alpha:.1f})")
        print(f"  (c) logistic: {adj_lr:.1f}%  (coef film={coefs[0]:+.2f}, neg_mpae={coefs[1]:+.2f})")

    # Aggregate
    pt = results['per_target']
    if pt:
        results['averages'] = {
            'film_alone':     np.mean([v['adj_film_alone']     for v in pt.values()]),
            'mpae_alone':     np.mean([v['adj_mpae_alone']     for v in pt.values()]),
            'a_rank_sum':     np.mean([v['a_rank_sum']         for v in pt.values()]),
            'b_linear_blend': np.mean([v['b_linear_blend']['adj']  for v in pt.values()]),
            'c_logistic':     np.mean([v['c_logistic']['adj'] for v in pt.values()]),
        }
        print("\n=== AVERAGE adj_LogAUC across matched targets ===")
        for k, v in results['averages'].items():
            print(f"  {k:18s} {v:.1f}%")
        print(f"  (London reported avg: 71.8%)")
    json.dump(results, open(OUT_DIR / "covalid_d3_combined_ranking.json", 'w'), indent=2, default=str)
    print(f"\nWrote {OUT_DIR / 'covalid_d3_combined_ranking.json'}")


if __name__ == "__main__":
    main()
