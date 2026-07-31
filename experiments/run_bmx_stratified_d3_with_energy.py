"""BMX-stratified D3 analysis WITH energy + Boltz signals.

Combines:
  - Energy scoring (dG_bind, ligand_strain) from local OpenMM run
  - Boltz mPAE_min, complex_ipde, ligand_iptm, complex_plddt from earlier extraction
  - Manifest (is_active, smiles) for labels

Compares the broken non-stratified BMX D3 (98% adj_LogAUC) to the corrected
stratified one. If the stratified gives a lower number on the same metric,
we confirm the decoy-sampling bug was driving the inflated headline.
"""
from __future__ import annotations
import sys, json
from pathlib import Path
import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")


def roc_auc_score(y_true, y_score):
    """Numpy-only ROC AUC (Mann-Whitney U formulation)."""
    y = np.asarray(y_true).astype(float)
    s = np.asarray(y_score).astype(float)
    pos = s[y == 1]
    neg = s[y == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    n_correct = 0.0
    for p in pos:
        n_correct += (neg < p).sum() + 0.5 * (neg == p).sum()
    return n_correct / (len(pos) * len(neg))

ROOT = Path(__file__).parent.parent
ENERGY_MAIN = ROOT / "data/covalid_mv_cofolds/energy_scores.csv"
ENERGY_PRIO = ROOT / "data/covalid_mv_cofolds/energy_scores_bmx_act.csv"
WARHEAD = ROOT / "data/covalid_mv_cofolds/warhead_metrics.csv"
WARHEAD_BMX_STRAT = ROOT / "data/covalid_mv_cofolds/warhead_metrics_bmx_strat.csv"
MANIFEST_MV = ROOT / "experiments/boltz_inputs/covalid_minimum_viable/manifest.csv"
MANIFEST_BMX_STRAT = Path("/tmp/bmx_stratified_yamls/manifest.csv")
OUT = ROOT / "results/covalid/covalid_d3_bmx_stratified_with_energy.json"


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
    x = np.log10(fpr_w.clip(min=1e-12)); yv = tpr_w
    auc_lam = float(np.sum((yv[1:] + yv[:-1]) * 0.5 * (x[1:] - x[:-1])))
    return 100.0 * (auc_lam - (1 - lo) / np.log(10)) / (-np.log10(lo) - (1 - lo) / np.log(10))


def report_signal(name, y, score, results, dropna=True):
    y = np.asarray(y); s = np.asarray(score)
    mask = ~np.isnan(s.astype(float))
    if mask.sum() < 4 or len(np.unique(y[mask])) < 2:
        print(f"  {name:25s} insufficient data ({mask.sum()} rows)")
        return
    auc = roc_auc_score(y[mask], s[mask])
    adj = adj_log_auc(y[mask], s[mask])
    results[name] = {"auc": float(auc), "adj_logauc": float(adj), "n_used": int(mask.sum())}
    print(f"  {name:25s} AUC={auc:.3f}  adj_LogAUC={adj:5.1f}%  (n={mask.sum()})")


def main():
    print("=" * 100)
    print("BMX D3 — stratified decoys + energy + Boltz signals")
    print("=" * 100)

    # 1. Load energy scores
    main_e = pd.read_csv(ENERGY_MAIN)
    prio_e = pd.read_csv(ENERGY_PRIO)
    energy = pd.concat([main_e, prio_e], ignore_index=True)
    energy = energy[energy["success_flag"] == 1].copy()
    print(f"\nEnergy rows (ok): {len(energy)} from main + priority")

    # 2. Manifests
    mv_man = pd.read_csv(MANIFEST_MV)
    strat_man = pd.read_csv(MANIFEST_BMX_STRAT)
    mv_man["dataset"] = "mv"; strat_man["dataset"] = "strat"
    manifest = pd.concat([
        mv_man[["target", "name", "is_active", "smiles", "dataset"]],
        strat_man[["target", "name", "is_active", "smiles", "dataset"]],
    ], ignore_index=True)

    # 3. Boltz mPAE (combine MV + BMX strat if available)
    warhead_parts = [pd.read_csv(WARHEAD)]
    if WARHEAD_BMX_STRAT.exists():
        warhead_parts.append(pd.read_csv(WARHEAD_BMX_STRAT))
        print(f"Loaded BMX strat warhead metrics: {len(warhead_parts[-1])} rows")
    warhead = pd.concat(warhead_parts, ignore_index=True, sort=False)

    # Join
    df = (manifest
          .merge(energy, on=["target", "name"], how="inner")
          .merge(warhead, on=["target", "name"], how="left"))
    df["smi_canon"] = df["smiles"].apply(canon)

    bmx = df[df["target"] == "BMX"].copy()
    bmx["mpae"] = bmx["mpae_london_min"]
    bmx["dG"] = bmx["dG_bind_kcalmol"]
    bmx["strain"] = bmx["ligand_strain_kcalmol"]
    bmx["ipde"] = bmx["complex_ipde"]
    bmx["lig_iptm"] = bmx["ligand_iptm"]

    print(f"\nBMX rows after join: {len(bmx)}")
    print(bmx.groupby(["dataset", "is_active"]).size().unstack(fill_value=0))

    results = {}

    # Compare 3 setups for BMX:
    #   (a) OLD non-stratified: 35 actives + 100 original decoys (the inflated 98% case)
    #   (b) NEW stratified:     35 actives + 99 stratified decoys (the fair case)
    #   (c) Both decoy sets:    35 actives + 199 decoys
    setups = {
        "(a) non_stratified": bmx[(bmx["is_active"] == 1) | (bmx["name"].str.startswith("BMX_dec_") & ~bmx["name"].str.contains("strat"))],
        "(b) stratified":     bmx[(bmx["is_active"] == 1) | (bmx["name"].str.contains("dec_strat"))],
        "(c) both_decoys":    bmx,
    }

    for setup_name, g in setups.items():
        print(f"\n{'='*100}")
        print(f"BMX setup {setup_name}: {len(g)} rows ({int(g['is_active'].sum())} act + {int(len(g) - g['is_active'].sum())} dec)")
        print("=" * 100)
        if g["is_active"].nunique() < 2 or len(g) < 4:
            print("  insufficient data — skipping")
            continue
        y = g["is_active"].values
        sub_results = {}
        # Boltz signals (lower = better → use negative)
        report_signal("mpae_london_min",   y, -g["mpae"],     sub_results)
        report_signal("complex_ipde",      y, -g["ipde"],     sub_results)
        report_signal("ligand_iptm",       y,  g["lig_iptm"], sub_results)
        # Energy signals (lower dG = better; lower strain = better)
        report_signal("dG_bind",           y, -g["dG"],       sub_results)
        report_signal("ligand_strain",     y, -g["strain"],   sub_results)
        # Combinations (rank sum)
        m_rank = (-g["mpae"]).rank(method="average")
        d_rank = (-g["dG"]).rank(method="average")
        i_rank = (-g["ipde"]).rank(method="average")
        s_rank = (-g["strain"]).rank(method="average")
        if m_rank.notna().all() and d_rank.notna().all():
            report_signal("rank_mpae+dG",      y, m_rank+d_rank,             sub_results)
        if m_rank.notna().all() and s_rank.notna().all():
            report_signal("rank_mpae+strain",  y, m_rank+s_rank,             sub_results)
        if d_rank.notna().all() and s_rank.notna().all():
            report_signal("rank_dG+strain",    y, d_rank+s_rank,             sub_results)
        if d_rank.notna().all() and s_rank.notna().all() and m_rank.notna().all():
            report_signal("rank_mpae+dG+strain", y, m_rank+d_rank+s_rank,    sub_results)
        if i_rank.notna().all() and d_rank.notna().all():
            report_signal("rank_ipde+dG",      y, i_rank+d_rank,             sub_results)
        results[setup_name] = {"n": int(len(g)), "n_act": int(y.sum()), "n_dec": int(len(g)-y.sum()), "signals": sub_results}

    OUT.parent.mkdir(parents=True, exist_ok=True)
    json.dump(results, open(OUT, "w"), indent=2, default=str)
    print(f"\nWrote {OUT}")


if __name__ == "__main__":
    main()
