#!/usr/bin/env python3
"""Generate campaign report comparing all cohorts on covalent geometry + confidence.

Inputs: data/paper_pair_training/boltz_dpo_campaign/track_A_<cohort>.csv (per cohort)
Output: data/paper_pair_training/boltz_dpo_campaign/report.md

Adds:
  - Bootstrap-median panel (2.5/50/97.5 percentiles) with N=1000 resamples per cohort.
  - Burgi-Dunitz window fraction (100 deg <= bd_angle_deg <= 120 deg) per cohort.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = [
    "c1_composite_v1",
    "c2_boltz_geom_dpo_only",
    "c3_boltz_geom_dpo_plus_dap",
    "c4_boltz_geom_dpo_regularized",
]

METRICS = [
    ("d_b_nuc_angstrom", "Warhead-Cys distance (A)", "lower_is_better"),
    ("bd_angle_deg", "Burgi-Dunitz angle (deg)", "target=100-120"),
    ("phi_planar_deg", "Planarity phi (deg)", "target=0"),
    ("mpae_warhead_cys", "Warhead-Cys mPAE (proxy)", "lower_is_better"),
    ("complex_iptm", "Complex ipTM", "higher_is_better"),
    ("ligand_iptm", "Ligand ipTM", "higher_is_better"),
    ("complex_plddt", "Complex pLDDT", "higher_is_better"),
]

# Burgi-Dunitz preferred range for nucleophilic addition to C=C (or C=O).
BD_LOW = 100.0
BD_HIGH = 120.0


def bootstrap_median(v: np.ndarray, n_boot: int = 1000, seed: int = 42):
    """Return (median, 2.5%, 97.5%) via nonparametric bootstrap."""
    if len(v) == 0:
        return float("nan"), float("nan"), float("nan")
    rng = np.random.default_rng(seed)
    n = len(v)
    reps = np.empty(n_boot, dtype=float)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        reps[i] = float(np.median(v[idx]))
    lo, hi = np.percentile(reps, [2.5, 97.5])
    return float(np.median(v)), float(lo), float(hi)


def summarize(df: pd.DataFrame):
    stats = {"n_total": int(len(df))}
    if "cofold_ok" in df.columns:
        stats["n_ok"] = int(df["cofold_ok"].sum())
    for key, _label, _pol in METRICS:
        if key in df.columns:
            v = df[key].dropna().astype(float).values
            if len(v):
                med, lo, hi = bootstrap_median(v)
                stats[f"{key}_median"] = med
                stats[f"{key}_median_ci_lo"] = lo
                stats[f"{key}_median_ci_hi"] = hi
                stats[f"{key}_mean"] = float(v.mean())
                stats[f"{key}_std"] = float(v.std())
                stats[f"{key}_n"] = int(len(v))
    # BD window fraction
    if "bd_angle_deg" in df.columns:
        v = df["bd_angle_deg"].dropna().astype(float).values
        if len(v):
            in_win = int(((v >= BD_LOW) & (v <= BD_HIGH)).sum())
            stats["bd_window_frac"] = float(in_win) / float(len(v))
            stats["bd_window_n"] = in_win
            stats["bd_window_denom"] = int(len(v))
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root",
                    default="/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/boltz_dpo_campaign")
    ap.add_argument("--out_md")
    ap.add_argument("--out_json")
    ap.add_argument("--note", default="")
    args = ap.parse_args()

    root = Path(args.root)
    out_md = Path(args.out_md) if args.out_md else root / "report.md"
    out_json = Path(args.out_json) if args.out_json else root / "report.json"

    per_cohort = {}
    for c in COHORTS:
        csv = root / f"track_A_{c}.csv"
        if not csv.exists():
            per_cohort[c] = {"status": "MISSING"}
            continue
        try:
            df = pd.read_csv(csv)
        except Exception as e:
            per_cohort[c] = {"status": f"READ_FAIL: {e}"}
            continue
        per_cohort[c] = {"status": "OK", **summarize(df)}

    out_json.write_text(json.dumps(per_cohort, indent=2))

    md = ["# Boltz DPO Campaign Report", "",
          f"Generated: {pd.Timestamp.utcnow()}Z"]
    if args.note:
        md.append("")
        md.append(f"**Note**: {args.note}")
    md.append("")
    md.append("## Cohort overview")
    md.append("")
    md.append("| Cohort | Status | n_total | n_ok |")
    md.append("|---|---|---|---|")
    for c, s in per_cohort.items():
        md.append(f"| {c} | {s.get('status')} | {s.get('n_total', '-')} | {s.get('n_ok', '-')} |")
    md.append("")

    # Burgi-Dunitz window fraction
    md.append(f"## Burgi-Dunitz window ({BD_LOW:.0f} deg <= bd_angle_deg <= {BD_HIGH:.0f} deg)")
    md.append("")
    md.append("| Cohort | in-window / total | fraction |")
    md.append("|---|---|---|")
    for c, s in per_cohort.items():
        if "bd_window_frac" in s:
            md.append(f"| {c} | {s['bd_window_n']}/{s['bd_window_denom']} | {s['bd_window_frac']:.3f} |")
        else:
            md.append(f"| {c} | - | - |")
    md.append("")

    # Bootstrap median panel per metric
    for key, label, pol in METRICS:
        md.append(f"## {label} ({pol})")
        md.append("")
        md.append("| Cohort | median [95% CI] | mean +/- std | n |")
        md.append("|---|---|---|---|")
        for c, s in per_cohort.items():
            if f"{key}_median" in s:
                md.append(
                    f"| {c} | {s[f'{key}_median']:.3f} [{s[f'{key}_median_ci_lo']:.3f}, {s[f'{key}_median_ci_hi']:.3f}] "
                    f"| {s[f'{key}_mean']:.3f} +/- {s[f'{key}_std']:.3f} | {s[f'{key}_n']} |"
                )
            else:
                md.append(f"| {c} | - | - | 0 |")
        md.append("")

    out_md.write_text("\n".join(md))
    print(f"[report] wrote {out_md} and {out_json}")


if __name__ == "__main__":
    main()
