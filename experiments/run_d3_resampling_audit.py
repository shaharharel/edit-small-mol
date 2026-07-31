"""Use existing 1479 Boltz-2 cofolds to audit decoy bias by 3 reanalyses.

(A) Matched-active-only: restrict to actives whose parent_id matches our decoys'
    `matched_parent` annotation (from si_004 `active_compound_name`). This gives
    a small but property-matched per-target benchmark.

(B) Bootstrap-CI: bootstrap adj_LogAUC over (active, decoy) draws to get 95% CI.

(C) Cross-target decoys: re-evaluate each target with OTHER kinases' actives
    used as "decoys". Tests pocket specificity. If Boltz-2 ranks the BMX active
    against an EGFR active correctly (BMX_act < EGFR_act in mPAE for BMX pocket),
    it knows the pocket. If it can't, it's just "kinase-active-vs-random".

(D) Bonus: per-target ROC for ligand_iptm, complex_ipde — also report (they
    are nearly co-linear with mpae_london_min per Agent C).
"""
from __future__ import annotations
import sys, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

ROOT = Path(__file__).parent.parent
WARHEAD_CSV = ROOT / "data/covalid_mv_cofolds/warhead_metrics.csv"
MANIFEST = ROOT / "experiments/boltz_inputs/covalid_minimum_viable/manifest.csv"
SI4 = ROOT / "data/covalid/ja5c22222_si_004.xlsx"
OUT = ROOT / "results/covalid/covalid_d3_resampling_audit.json"


def canon(s):
    try: return Chem.MolToSmiles(Chem.MolFromSmiles(s))
    except: return None


def adj_log_auc(y, s, lam=10):
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
    return 100.0 * (auc_lam - (1 - lo) / np.log(10)) / (-np.log10(lo) - (1 - lo) / np.log(10))


def load_decoy_parents():
    """For each target, build (canon_smi → parent_active_id) for decoys, from si_004."""
    import openpyxl
    wb = openpyxl.load_workbook(SI4, read_only=True)
    targets = {s.replace("_decoys", "") for s in wb.sheetnames if s.endswith("_decoys")}
    out = {}
    for tgt in targets:
        sh = wb[f"{tgt}_decoys"]
        rows = list(sh.iter_rows(values_only=True))
        header = rows[0]
        idx_smi = header.index("protomer_smiles")
        idx_par = header.index("active_compound_name")
        m = {}
        for r in rows[1:]:
            if r is None or len(r) < max(idx_smi, idx_par) + 1: continue
            smi, par = r[idx_smi], r[idx_par]
            if smi is None: continue
            c = canon(smi)
            if c is None: continue
            m[c] = par
        out[tgt] = m
    return out


def main():
    print("=" * 90)
    print("D3 RESAMPLING AUDIT — using existing 1479 cofolds")
    print("=" * 90)

    manifest = pd.read_csv(MANIFEST)
    metrics = pd.read_csv(WARHEAD_CSV)
    df = manifest.merge(metrics, on=["target", "name"], how="inner")
    df["smi_canon"] = df["smiles"].apply(canon)
    df = df.dropna(subset=["smi_canon", "mpae_london_min"]).copy()
    df["mpae"] = df["mpae_london_min"].astype(float)
    print(f"Loaded {len(df)} cofolds, {df['target'].nunique()} targets")

    # Map our decoys to their parent active
    print("\nLoading parent-active map from si_004 (this takes ~30s)…")
    parent_map = load_decoy_parents()

    df["parent_active"] = None
    for tgt in df["target"].unique():
        m = parent_map.get(tgt, {})
        mask = (df["target"] == tgt) & (df["is_active"] == 0)
        df.loc[mask, "parent_active"] = df.loc[mask, "smi_canon"].map(m)

    results = {"per_target": {}, "global": {}}

    # ============================================
    # (A) Matched-actives-only: restrict to parent-actives + their decoys
    # ============================================
    print("\n" + "=" * 90)
    print("(A) MATCHED-ACTIVES-ONLY: restrict to parent actives whose decoys we have")
    print("=" * 90)
    print(f"{'target':12s}  {'n_par':>5s}  {'n_dec':>5s}  {'n_act_full':>10s}  "
          f"{'AUC_match':>10s}  {'adj_match':>10s}  {'AUC_full':>9s}  {'adj_full':>9s}")
    print("-" * 95)

    rows_a = []
    for tgt, g in df.groupby("target"):
        decoys = g[g["is_active"] == 0]
        parents = set(decoys["parent_active"].dropna())
        if not parents: continue
        # Load full active list from si_004 to find which actives are parents
        import openpyxl
        wb = openpyxl.load_workbook(SI4, read_only=True)
        if f"{tgt}_actives" not in wb.sheetnames: continue
        sh = wb[f"{tgt}_actives"]
        rows = list(sh.iter_rows(values_only=True))
        header = rows[0]
        i_smi = header.index("protomer_smiles")
        i_name = header.index("active_compound_name")
        parent_smiles = set()
        for r in rows[1:]:
            if r is None or len(r) < max(i_smi, i_name) + 1: continue
            if r[i_name] in parents:
                c = canon(r[i_smi])
                if c: parent_smiles.add(c)
        actives_in_cohort = g[(g["is_active"] == 1) & (g["smi_canon"].isin(parent_smiles))]
        full_actives = g[g["is_active"] == 1]
        # Matched-only evaluation
        sub = pd.concat([actives_in_cohort, decoys])
        if sub["is_active"].nunique() < 2 or len(actives_in_cohort) < 2:
            print(f"{tgt:12s}  too few parent actives ({len(actives_in_cohort)}) — skipping")
            continue
        auc_m = roc_auc_score(sub["is_active"], -sub["mpae"])
        adj_m = adj_log_auc(sub["is_active"], -sub["mpae"])
        # Full evaluation for comparison
        auc_f = roc_auc_score(g["is_active"], -g["mpae"])
        adj_f = adj_log_auc(g["is_active"], -g["mpae"])
        rows_a.append((tgt, len(actives_in_cohort), len(decoys), len(full_actives),
                       auc_m, adj_m, auc_f, adj_f))
        print(f"{tgt:12s}  {len(actives_in_cohort):>5d}  {len(decoys):>5d}  {len(full_actives):>10d}  "
              f"{auc_m:>10.3f}  {adj_m:>9.1f}%  {auc_f:>9.3f}  {adj_f:>8.1f}%")
        results["per_target"][tgt] = {
            "n_parent_actives": len(actives_in_cohort),
            "n_full_actives": len(full_actives),
            "n_decoys": len(decoys),
            "matched_auc_mpae": float(auc_m), "matched_adj_mpae": float(adj_m),
            "full_auc_mpae": float(auc_f), "full_adj_mpae": float(adj_f),
        }
    if rows_a:
        am = float(np.mean([r[4] for r in rows_a])); af = float(np.mean([r[6] for r in rows_a]))
        bm = float(np.mean([r[5] for r in rows_a])); bf = float(np.mean([r[7] for r in rows_a]))
        print("-" * 95)
        print(f"{'AVG':12s}                                       "
              f"{am:>10.3f}  {bm:>9.1f}%  {af:>9.3f}  {bf:>8.1f}%")
        results["global"]["matched_avg_auc"] = am
        results["global"]["matched_avg_adj"] = bm
        results["global"]["full_avg_auc"] = af
        results["global"]["full_avg_adj"] = bf

    # ============================================
    # (B) Bootstrap CI on full cohort
    # ============================================
    print("\n" + "=" * 90)
    print("(B) BOOTSTRAP 95% CI on full-cohort adj_LogAUC")
    print("=" * 90)
    rng = np.random.default_rng(0)
    boot = {}
    for tgt, g in df.groupby("target"):
        if g["is_active"].nunique() < 2: continue
        y = g["is_active"].values; s = -g["mpae"].values
        boots = []
        for _ in range(500):
            idx = rng.integers(0, len(y), len(y))
            try:
                if len(set(y[idx])) > 1:
                    boots.append(adj_log_auc(y[idx], s[idx]))
            except Exception: pass
        if boots:
            boots = np.array(boots)
            ci_lo, ci_hi = float(np.percentile(boots, 2.5)), float(np.percentile(boots, 97.5))
            print(f"{tgt:12s}  adj_LogAUC = {boots.mean():.1f}% [95% CI: {ci_lo:.1f}, {ci_hi:.1f}]")
            boot[tgt] = {"mean": float(boots.mean()), "ci_lo": ci_lo, "ci_hi": ci_hi}
    results["bootstrap_ci"] = boot

    # ============================================
    # (C) Cross-target: use OTHER targets' actives as decoys
    # ============================================
    print("\n" + "=" * 90)
    print("(C) CROSS-TARGET — actives of OTHER kinases used as 'decoys'")
    print("=" * 90)
    print(f"{'target':12s}  {'n_act':>5s}  {'n_xdec':>6s}    {'AUC_x':>6s}  {'adj_x':>6s}")
    print("-" * 60)

    xt = {}
    for tgt in df["target"].unique():
        own = df[(df["target"] == tgt) & (df["is_active"] == 1)]
        other = df[(df["target"] != tgt) & (df["is_active"] == 1)]
        if len(own) < 2 or len(other) < 10: continue
        merged = pd.concat([
            own.assign(label=1),
            other.assign(label=0),
        ])
        # Use THIS target's mPAE (own evaluation); for other-target actives the
        # mpae is from their OWN target cofold — that's the wrong frame. Instead
        # we need other actives' mpae computed against THIS pocket — we don't have
        # that without re-running. As proxy, compare absolute mPAE values from
        # cross-target cofolds (each at its own target). This isn't a strict
        # control but gives a rough sense.
        auc_x = roc_auc_score(merged["label"], -merged["mpae"])
        adj_x = adj_log_auc(merged["label"], -merged["mpae"])
        xt[tgt] = {"n_own": len(own), "n_other": len(other),
                    "auc_cross": float(auc_x), "adj_cross": float(adj_x)}
        print(f"{tgt:12s}  {len(own):>5d}  {len(other):>6d}    {auc_x:>6.3f}  {adj_x:>5.1f}%")
    results["cross_target"] = xt
    print("\nNOTE on (C): this is a *rough* control. Each 'cross-target active' was")
    print("Boltz-cofolded against its OWN target's pocket, not the target we are evaluating.")
    print("Strict control would require recofolding all actives × all targets (8x extra cost).")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
