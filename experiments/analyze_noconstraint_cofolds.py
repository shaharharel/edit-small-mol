#!/usr/bin/env python3
"""Analyze Experiment A: Boltz-no-constraint cofolds for v1_clean cohorts.

Consumes the per-cohort track_A_v2curr_clean_NOCONSTR_*.csv files
(harvest_cohort output from boltz_verdict_driver.py) and computes:

  - Distribution stats for d_b_nuc_angstrom, bd_angle_deg, phi_planar_deg,
    complex_iptm, ligand_iptm, complex_plddt.
  - Headline: f_BD_ready = fraction of cofolds with
        d ∈ [2.5, 5.5] Å  AND  bd ∈ [80, 130] °
    with bootstrap 95 % CI (n_boot=1000).
  - KS tests for pose-conditioned metrics between cohorts.

Usage:
    python experiments/analyze_noconstraint_cofolds.py \\
        --cofold_dir data/paper_pair_training/v2_curriculum_clean/noconstraint_cofolds \\
        --out_summary_json data/paper_pair_training/v2_curriculum_clean/noconstraint_summary.json
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import ks_2samp

COHORTS = ["theta_90", "theta_105", "theta_130", "null_pose"]

# Bürgi-Dunitz productive window (strict)
D_LO, D_HI = 2.5, 5.5
BD_LO, BD_HI = 80.0, 130.0
# Loose docking window (β-C within ligand-scale of Sγ)
D_LOOSE = 7.0


def _bd_ready(sub: pd.DataFrame, d_lo=D_LO, d_hi=D_HI, bd_lo=BD_LO, bd_hi=BD_HI):
    """Boolean array: cofold is BD-ready under (d, bd) window."""
    ok = sub[sub["cofold_ok"] == True].copy()
    d = ok["d_b_nuc_angstrom"].values
    bd = ok["bd_angle_deg"].values
    return (d >= d_lo) & (d <= d_hi) & (bd >= bd_lo) & (bd <= bd_hi)


def _d_within(sub: pd.DataFrame, d_max=D_LOOSE):
    """Boolean array: β-C within `d_max` Å of Sγ (any angle)."""
    ok = sub[sub["cofold_ok"] == True].copy()
    d = ok["d_b_nuc_angstrom"].values
    return d <= d_max


def _bootstrap_ci(x: np.ndarray, n_boot: int = 1000, alpha: float = 0.05, seed: int = 42):
    """Bootstrap CI on the fraction of True."""
    if len(x) == 0:
        return (float("nan"), float("nan"), float("nan"))
    rng = np.random.default_rng(seed)
    boots = np.empty(n_boot)
    n = len(x)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = float(x[idx].mean())
    lo = float(np.percentile(boots, 100 * alpha / 2))
    hi = float(np.percentile(boots, 100 * (1 - alpha / 2)))
    return (float(x.mean()), lo, hi)


def _ks(a, b):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return {"stat": None, "p": None, "n_a": len(a), "n_b": len(b)}
    r = ks_2samp(a, b)
    return {"stat": float(r.statistic), "p": float(r.pvalue),
            "n_a": len(a), "n_b": len(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cofold_dir", required=True)
    ap.add_argument("--out_summary_json", required=True)
    ap.add_argument("--out_merged_csv", default=None)
    ap.add_argument("--constrained_baseline_dir", default=None,
                    help="Optional dir of previously-computed constrained "
                    "cohort tracks (track_A_v2curr_clean_<cohort>.csv). "
                    "If provided, include side-by-side d/bd/iptm comparison.")
    args = ap.parse_args()

    root = Path(args.cofold_dir)
    frames = []
    for c in COHORTS:
        csv = root / f"track_A_v2curr_clean_NOCONSTR_{c}.csv"
        if not csv.exists():
            print(f"[warn] missing {csv}")
            continue
        try:
            df = pd.read_csv(csv)
        except pd.errors.EmptyDataError:
            print(f"[warn] empty {csv} — skipping")
            continue
        if len(df) == 0:
            print(f"[warn] no rows in {csv} — skipping")
            continue
        df["cohort_short"] = c
        frames.append(df)
    if not frames:
        raise SystemExit("no cohort CSVs found")
    merged = pd.concat(frames, ignore_index=True)
    if args.out_merged_csv:
        merged.to_csv(args.out_merged_csv, index=False)
        print(f"[merge] wrote {len(merged)} rows to {args.out_merged_csv}")

    summary = {"per_cohort": {}, "ks": {}, "f_BD_ready": {},
               "f_d_within_7A": {}, "config": {
        "d_window": [D_LO, D_HI], "bd_window": [BD_LO, BD_HI],
        "d_loose_max": D_LOOSE,
    }}
    metrics = ["d_b_nuc_angstrom", "bd_angle_deg", "phi_planar_deg",
               "complex_iptm", "ligand_iptm", "complex_plddt",
               "mpae_prot_lig_mean", "mpae_prot_lig_min"]
    for c in COHORTS:
        sub = merged[merged["cohort_short"] == c]
        ok = sub[sub["cofold_ok"] == True]
        stats = {"n_total": int(len(sub)), "n_ok": int(len(ok))}
        for m in metrics:
            if m not in ok.columns:
                continue
            v = ok[m].astype(float).values
            v = v[~np.isnan(v)]
            if len(v) == 0:
                stats[m] = {"median": None, "mean": None, "std": None, "n": 0}
                continue
            stats[m] = {"median": float(np.median(v)), "mean": float(np.mean(v)),
                        "std": float(np.std(v)), "n": int(len(v)),
                        "q25": float(np.percentile(v, 25)),
                        "q75": float(np.percentile(v, 75)),
                        "min": float(v.min()), "max": float(v.max())}
        # f_BD_ready with bootstrap CI (strict window)
        br = _bd_ready(sub).astype(int) if len(sub) > 0 else np.array([])
        mean, lo, hi = _bootstrap_ci(br) if len(br) > 0 else (float("nan"),) * 3
        summary["f_BD_ready"][c] = {
            "value": mean, "ci95_lo": lo, "ci95_hi": hi, "n": int(len(br)),
        }
        # Loose window: d<=7 Å (any angle)
        dw = _d_within(sub).astype(int) if len(sub) > 0 else np.array([])
        dmean, dlo, dhi = _bootstrap_ci(dw) if len(dw) > 0 else (float("nan"),) * 3
        summary.setdefault("f_d_within_7A", {})[c] = {
            "value": dmean, "ci95_lo": dlo, "ci95_hi": dhi, "n": int(len(dw)),
        }
        summary["per_cohort"][c] = stats

    # KS between cohorts
    pairs = [("theta_90", "theta_130"),
             ("theta_90", "null_pose"),
             ("theta_105", "theta_130"),
             ("theta_90", "theta_105"),
             ("null_pose", "theta_130")]
    ks_metrics = metrics + []
    for m in ks_metrics:
        summary["ks"][m] = {}
        for (a, b) in pairs:
            va = merged[(merged["cohort_short"] == a) &
                        (merged["cofold_ok"] == True)][m].values \
                if m in merged.columns else np.array([])
            vb = merged[(merged["cohort_short"] == b) &
                        (merged["cofold_ok"] == True)][m].values \
                if m in merged.columns else np.array([])
            summary["ks"][m][f"{a}_vs_{b}"] = _ks(va, vb)
    # Also KS on f_BD_ready — treat as per-cofold Bernoulli
    summary["ks"]["f_BD_ready_bernoulli"] = {}
    summary["ks"]["f_d_within_7A_bernoulli"] = {}
    for (a, b) in pairs:
        va = _bd_ready(merged[merged["cohort_short"] == a]).astype(int)
        vb = _bd_ready(merged[merged["cohort_short"] == b]).astype(int)
        summary["ks"]["f_BD_ready_bernoulli"][f"{a}_vs_{b}"] = _ks(va, vb)
        va2 = _d_within(merged[merged["cohort_short"] == a]).astype(int)
        vb2 = _d_within(merged[merged["cohort_short"] == b]).astype(int)
        summary["ks"]["f_d_within_7A_bernoulli"][f"{a}_vs_{b}"] = _ks(va2, vb2)

    # Optional constrained-baseline comparison
    if args.constrained_baseline_dir:
        base_root = Path(args.constrained_baseline_dir)
        summary["constrained_baseline"] = {}
        for c in COHORTS:
            base_csv = base_root / f"track_A_v2curr_clean_{c}.csv"
            if not base_csv.exists():
                continue
            base = pd.read_csv(base_csv)
            base_ok = base[base["cofold_ok"] == True] if "cofold_ok" in base.columns else base
            entry = {"n": int(len(base)), "n_ok": int(len(base_ok))}
            for m in ["d_b_nuc_angstrom", "bd_angle_deg", "complex_iptm"]:
                if m in base_ok.columns:
                    v = base_ok[m].astype(float).dropna().values
                    entry[m] = {"median": float(np.median(v)) if len(v) else None,
                                "mean": float(np.mean(v)) if len(v) else None,
                                "n": int(len(v))}
            summary["constrained_baseline"][c] = entry

    Path(args.out_summary_json).write_text(json.dumps(summary, indent=2))
    print(f"[analysis] summary → {args.out_summary_json}")

    # Print concise report
    print("\n=== Per-cohort headline metrics ===")
    print(f"{'cohort':<12} {'n':<4} {'d_med':<7} {'bd_med':<7} "
          f"{'iptm_med':<9} {'f_BD_ready':<20} {'f_d<7A':<20}")
    for c in COHORTS:
        s = summary["per_cohort"].get(c, {})
        d = s.get("d_b_nuc_angstrom", {}).get("median", float("nan"))
        bd = s.get("bd_angle_deg", {}).get("median", float("nan"))
        ip = s.get("complex_iptm", {}).get("median", float("nan"))
        fr = summary["f_BD_ready"].get(c, {})
        val = fr.get("value", float("nan"))
        lo = fr.get("ci95_lo", float("nan"))
        hi = fr.get("ci95_hi", float("nan"))
        dw = summary["f_d_within_7A"].get(c, {})
        dw_val = dw.get("value", float("nan"))
        dw_lo = dw.get("ci95_lo", float("nan"))
        dw_hi = dw.get("ci95_hi", float("nan"))
        d = d if d is not None else float("nan")
        bd = bd if bd is not None else float("nan")
        ip = ip if ip is not None else float("nan")
        print(f"{c:<12} {s.get('n_ok',0):<4} "
              f"{d:<7.2f} {bd:<7.1f} {ip:<9.3f} "
              f"{val:.3f}[{lo:.2f},{hi:.2f}]  "
              f"{dw_val:.3f}[{dw_lo:.2f},{dw_hi:.2f}]")

    print("\n=== KS tests (p < 0.05 marked *) ===")
    for m, results in summary["ks"].items():
        for k, v in results.items():
            if v["p"] is None:
                continue
            star = " *" if v["p"] < 0.05 else ""
            print(f"  {m:<28} {k:<28} stat={v['stat']:.3f} p={v['p']:.4g}{star}")


if __name__ == "__main__":
    main()
