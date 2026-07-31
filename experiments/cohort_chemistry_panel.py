#!/usr/bin:env python3
"""Experiment C: Per-cohort SMILES-topology chemistry panel.

Purely SMILES-based per cohort:
  - MW, TPSA, RotB, HBA, HBD, QED distributions
  - Murcko scaffold overlap between cohorts (Jaccard)
  - Nearest-neighbor Tanimoto within-cohort vs across-cohort
  - Basic distributional differences (KS on physchem)

Usage:
    python experiments/cohort_chemistry_panel.py \\
        --samples_dir data/paper_pair_training/v2_curriculum_clean/steering_samples_merged \\
        --n_per_cohort 100 \\
        --out_csv data/paper_pair_training/v2_curriculum_clean/cohort_chemistry_panel.csv \\
        --out_summary_json data/paper_pair_training/v2_curriculum_clean/cohort_chemistry_summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Descriptors, QED
from rdkit.Chem.Scaffolds import MurckoScaffold
from scipy.stats import ks_2samp

RDLogger.DisableLog("rdApp.*")


def _dedup(smis):
    seen = set()
    out = []
    for s in smis:
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        frags = Chem.GetMolFrags(m, asMols=True)
        if len(frags) > 1:
            m = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        c = Chem.MolToSmiles(m)
        if c in seen:
            continue
        seen.add(c)
        out.append(c)
    return out


def _scaffold(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        return MurckoScaffold.MurckoScaffoldSmiles(mol=m, includeChirality=False)
    except Exception:
        return None


def _fp(smi, radius=2, nbits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nbits)


def _tanimoto_nn(fps_query, fps_ref, exclude_self=False):
    """For each fp in fps_query, find max Tanimoto to any fp in fps_ref.

    If exclude_self, assume fps_query == fps_ref and skip identical index.
    """
    out = []
    for i, q in enumerate(fps_query):
        if q is None:
            out.append(np.nan)
            continue
        best = -1.0
        for j, r in enumerate(fps_ref):
            if r is None:
                continue
            if exclude_self and i == j:
                continue
            t = DataStructs.TanimotoSimilarity(q, r)
            if t > best:
                best = t
        out.append(best if best >= 0 else np.nan)
    return np.array(out)


def _physchem(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return {}
    try:
        return {
            "mw": Descriptors.MolWt(m),
            "logp": Descriptors.MolLogP(m),
            "tpsa": Descriptors.TPSA(m),
            "rotb": Descriptors.NumRotatableBonds(m),
            "hba": Descriptors.NumHAcceptors(m),
            "hbd": Descriptors.NumHDonors(m),
            "qed": QED.qed(m),
        }
    except Exception:
        return {}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--n_per_cohort", type=int, default=100)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_summary_json", required=True)
    args = ap.parse_args()

    sdir = Path(args.samples_dir)
    cohorts = ["theta_90", "theta_105", "theta_130", "null_pose"]
    all_rows = []
    smis_by_cohort = {}
    scaff_by_cohort = {}
    fps_by_cohort = {}
    for c in cohorts:
        raw = pd.read_csv(sdir / f"samples_{c}.csv")["SMILES"].astype(str).tolist()
        uniq = _dedup(raw)[: args.n_per_cohort]
        smis_by_cohort[c] = uniq
        scaff_by_cohort[c] = [_scaffold(s) for s in uniq]
        fps_by_cohort[c] = [_fp(s) for s in uniq]
        for i, s in enumerate(uniq):
            p = _physchem(s)
            p.update({"cohort": c, "idx": i, "smiles": s,
                      "scaffold": scaff_by_cohort[c][i] or ""})
            all_rows.append(p)
        print(f"[cohort {c}] {len(uniq)} unique SMILES")

    df = pd.DataFrame(all_rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[chem] wrote {len(df)} rows → {out_csv}")

    # Cohort summary + KS
    metrics = ["mw", "logp", "tpsa", "rotb", "hba", "hbd", "qed"]
    summary = {"per_cohort": {}, "ks_vs_theta_90": {},
               "scaffold_jaccard": {}, "nn_tanimoto": {}}
    for c in cohorts:
        sub = df[df["cohort"] == c]
        summary["per_cohort"][c] = {}
        for m in metrics:
            summary["per_cohort"][c][m] = {
                "median": float(sub[m].median()),
                "mean": float(sub[m].mean()),
                "std": float(sub[m].std()),
            }
    # KS against theta_90 (arbitrary reference; also do vs null_pose)
    ref_pairs = [("theta_90", "theta_105"),
                 ("theta_90", "theta_130"),
                 ("theta_90", "null_pose"),
                 ("null_pose", "theta_105"),
                 ("null_pose", "theta_130"),
                 ("theta_105", "theta_130")]
    for m in metrics:
        summary["ks_vs_theta_90"][m] = {}
        for (a, b) in ref_pairs:
            va = df[df["cohort"] == a][m].values
            vb = df[df["cohort"] == b][m].values
            va = va[~np.isnan(va)]
            vb = vb[~np.isnan(vb)]
            if len(va) < 3 or len(vb) < 3:
                summary["ks_vs_theta_90"][m][f"{a}_vs_{b}"] = {"stat": None, "p": None}
                continue
            r = ks_2samp(va, vb)
            summary["ks_vs_theta_90"][m][f"{a}_vs_{b}"] = {
                "stat": float(r.statistic), "p": float(r.pvalue),
            }

    # Scaffold Jaccard
    for a in cohorts:
        for b in cohorts:
            if a == b:
                continue
            sa = set([x for x in scaff_by_cohort[a] if x])
            sb = set([x for x in scaff_by_cohort[b] if x])
            if not sa or not sb:
                summary["scaffold_jaccard"][f"{a}_vs_{b}"] = None
                continue
            j = len(sa & sb) / len(sa | sb)
            summary["scaffold_jaccard"][f"{a}_vs_{b}"] = float(j)

    # Nearest-neighbor Tanimoto within vs across
    for c in cohorts:
        fps_c = fps_by_cohort[c]
        # Within: exclude self
        within = _tanimoto_nn(fps_c, fps_c, exclude_self=True)
        summary["nn_tanimoto"][c] = {
            "within_median": float(np.nanmedian(within)),
            "within_mean": float(np.nanmean(within)),
            "across": {},
        }
        for other in cohorts:
            if other == c:
                continue
            across = _tanimoto_nn(fps_c, fps_by_cohort[other], exclude_self=False)
            summary["nn_tanimoto"][c]["across"][other] = {
                "median": float(np.nanmedian(across)),
                "mean": float(np.nanmean(across)),
            }

    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2))
    print(f"[chem] summary → {args.out_summary_json}")

    # Terse printout
    print("\n=== KS on physchem (any p<0.05 = distributionally different) ===")
    for m in metrics:
        for k, v in summary["ks_vs_theta_90"][m].items():
            if v["p"] is not None and v["p"] < 0.05:
                print(f"  {m:<8} {k:<32} stat={v['stat']:.3f} p={v['p']:.4g} *")

    print("\n=== Scaffold Jaccard (0 = disjoint, 1 = identical) ===")
    for k, v in summary["scaffold_jaccard"].items():
        if v is not None:
            print(f"  {k:<32} {v:.3f}")

    print("\n=== NN Tanimoto (within vs closest across) ===")
    for c in cohorts:
        w = summary["nn_tanimoto"][c]["within_median"]
        print(f"  {c:<12} within_median={w:.3f}", end="  across_median={")
        parts = [f"{o}:{summary['nn_tanimoto'][c]['across'][o]['median']:.3f}"
                 for o in cohorts if o != c]
        print(", ".join(parts) + "}")


if __name__ == "__main__":
    main()
