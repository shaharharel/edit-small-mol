#!/usr/bin/env python3
"""Analyze Boltz verdict campaign: compute cohort medians, Wasserstein, verdict.

Reads track_A_*.csv and track_B_*.csv from boltz_verdict/ and produces
boltz_verdict_report.md with per-cohort statistics and pairwise Wasserstein
distances.
"""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd

try:
    from scipy.stats import wasserstein_distance
    HAVE_SCIPY = True
except ImportError:
    HAVE_SCIPY = False

BOLTZ_ROOT = Path("data/paper_pair_training/boltz_verdict")
COHORTS = {
    "ablation3": "track_A_ablation3.csv",
    "v2cond": "track_A_v2cond.csv",
    "v4composite": "track_A_v4composite_clampA.csv",
    "v4d": "track_A_v4d_clampA.csv",
    "v4theta": "track_A_v4theta_clampA.csv",
}
TRACK_B_CSV = "track_B_v4comp_clampC.csv"
METRICS = [
    "d_b_nuc_angstrom", "bd_angle_deg", "phi_planar_deg",
    "mpae_warhead_cys", "mpae_prot_lig_min", "mpae_prot_lig_mean",
    "complex_iptm", "ligand_iptm", "complex_plddt",
]


def summarize_cohort(df: pd.DataFrame, cohort: str) -> dict:
    ok = df[df["cofold_ok"].fillna(False).astype(bool)]
    out = {
        "cohort": cohort,
        "n_yamls": len(df),
        "n_ok": len(ok),
        "cofold_fail_pct": round(100 * (1 - len(ok) / max(len(df), 1)), 1),
    }
    for m in METRICS:
        if m not in ok.columns:
            continue
        v = pd.to_numeric(ok[m], errors="coerce").dropna().values
        if len(v) == 0:
            continue
        out[f"{m}_median"] = float(np.median(v))
        out[f"{m}_q25"] = float(np.percentile(v, 25))
        out[f"{m}_q75"] = float(np.percentile(v, 75))
        out[f"{m}_min"] = float(np.min(v))
        out[f"{m}_max"] = float(np.max(v))
        out[f"{m}_n"] = len(v)
        if m == "bd_angle_deg":
            out[f"{m}_dev105_median"] = float(np.median(np.abs(v - 105.0)))
    return out


def bootstrap_baseline_wasserstein(v: np.ndarray, n_iter: int = 200, seed: int = 42) -> float:
    """W(A, A') noise floor via random split bootstrap."""
    if len(v) < 20 or not HAVE_SCIPY:
        return float("nan")
    rng = np.random.default_rng(seed)
    dists = []
    for _ in range(n_iter):
        idx = rng.permutation(len(v))
        h = len(v) // 2
        a1, a2 = v[idx[:h]], v[idx[h:2*h]]
        dists.append(wasserstein_distance(a1, a2))
    return float(np.median(dists))


def load_cohort_metric(cohort_csv: Path, metric: str) -> np.ndarray:
    if not cohort_csv.exists():
        return np.array([])
    df = pd.read_csv(cohort_csv)
    if metric not in df.columns:
        return np.array([])
    v = pd.to_numeric(df[metric], errors="coerce").dropna().values
    return v


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=BOLTZ_ROOT)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    root = args.root
    out_path = args.out or (root / "boltz_verdict_report.md")

    lines = []
    lines.append("# Boltz Verdict Report\n")
    lines.append(f"Generated: {pd.Timestamp.utcnow().isoformat()}\n")
    lines.append("\nBoltz-2 cofold results for 5 Track-A cohorts + Track-B v4composite Clamp A/C spot-check.\n")
    lines.append("Constraint: covalent bond between warhead β-C and Cys346 SG.\n\n")
    lines.append("## Per-cohort summary\n\n")

    summaries = {}
    for cohort, fname in COHORTS.items():
        p = root / fname
        if not p.exists():
            lines.append(f"### {cohort}\n**MISSING** ({p})\n\n")
            continue
        df = pd.read_csv(p)
        s = summarize_cohort(df, cohort)
        summaries[cohort] = s
        lines.append(f"### {cohort} (n_ok={s['n_ok']} / n_yamls={s['n_yamls']}, "
                     f"fail_pct={s['cofold_fail_pct']}%)\n\n")
        lines.append("| Metric | median | IQR (q25–q75) | range | n |\n")
        lines.append("|---|---|---|---|---|\n")
        for m in METRICS:
            k = f"{m}_median"
            if k not in s:
                continue
            lines.append(f"| {m} | {s[k]:.3f} | "
                          f"{s[f'{m}_q25']:.3f} – {s[f'{m}_q75']:.3f} | "
                          f"{s[f'{m}_min']:.3f} – {s[f'{m}_max']:.3f} | "
                          f"{s[f'{m}_n']} |\n")
        if "bd_angle_deg_dev105_median" in s:
            lines.append(f"| bd_angle_dev105_median | {s['bd_angle_deg_dev105_median']:.3f} | – | – | – |\n")
        lines.append("\n")

    # Track B
    tb = root / TRACK_B_CSV
    if tb.exists():
        df = pd.read_csv(tb)
        s = summarize_cohort(df, "v4comp_clampC")
        summaries["v4comp_clampC"] = s
        lines.append(f"### Track B — v4comp Clamp C (n_ok={s['n_ok']})\n\n")
        lines.append("| Metric | median | IQR | range | n |\n")
        lines.append("|---|---|---|---|---|\n")
        for m in METRICS:
            k = f"{m}_median"
            if k not in s:
                continue
            lines.append(f"| {m} | {s[k]:.3f} | "
                          f"{s[f'{m}_q25']:.3f} – {s[f'{m}_q75']:.3f} | "
                          f"{s[f'{m}_min']:.3f} – {s[f'{m}_max']:.3f} | "
                          f"{s[f'{m}_n']} |\n")
        lines.append("\n")

    # Pairwise Wasserstein vs Ablation #3
    lines.append("## Wasserstein distances — trained vs Ablation #3 baseline\n\n")
    if HAVE_SCIPY and "ablation3" in summaries:
        for metric in ["d_b_nuc_angstrom", "bd_angle_deg", "mpae_warhead_cys",
                        "complex_iptm", "phi_planar_deg"]:
            base = load_cohort_metric(root / COHORTS["ablation3"], metric)
            if len(base) < 20:
                continue
            wa_bootstrap = bootstrap_baseline_wasserstein(base)
            lines.append(f"### {metric}\n\n")
            lines.append(f"Baseline (Ablation #3) noise floor W(A,A') = **{wa_bootstrap:.4f}**\n\n")
            lines.append("| Cohort | median | ΔMedian vs Ablation | W(cohort, Ablation) | W ratio | winner? |\n")
            lines.append("|---|---|---|---|---|---|\n")
            base_med = np.median(base)
            for cohort, fname in COHORTS.items():
                if cohort == "ablation3":
                    continue
                v = load_cohort_metric(root / fname, metric)
                if len(v) < 20:
                    continue
                w = wasserstein_distance(base, v)
                med = np.median(v)
                ratio = w / wa_bootstrap if wa_bootstrap > 0 else float("inf")
                # Winner criterion for d_b_nuc / mpae / bd_angle_dev / phi:
                # lower is better; for iptm, higher is better.
                if metric == "complex_iptm":
                    good = med > base_med
                elif metric == "bd_angle_deg":
                    good = abs(med - 105) < abs(base_med - 105)
                else:  # d_b_nuc, mpae, phi
                    good = med < base_med
                winner = "YES" if (ratio > 2.0 and good) else ("dir-good" if good else "no")
                lines.append(f"| {cohort} | {med:.3f} | {med - base_med:+.3f} | "
                              f"{w:.4f} | {ratio:.2f}× | {winner} |\n")
            lines.append("\n")
    else:
        lines.append("(scipy not available or ablation3 missing — skipping W analysis)\n\n")

    # Track B: v4composite Clamp A vs Clamp C (h1 vs h2)
    lines.append("## Track B — Clamp A vs Clamp C (v4composite)\n\n")
    if HAVE_SCIPY and (root / COHORTS["v4composite"]).exists() and tb.exists():
        for metric in ["d_b_nuc_angstrom", "bd_angle_deg", "mpae_warhead_cys",
                        "complex_iptm", "phi_planar_deg"]:
            a = load_cohort_metric(root / COHORTS["v4composite"], metric)
            c = load_cohort_metric(tb, metric)
            if len(a) < 5 or len(c) < 5:
                continue
            w_ac = wasserstein_distance(a, c)
            w_bootstrap = bootstrap_baseline_wasserstein(a)
            ratio = w_ac / w_bootstrap if w_bootstrap > 0 else float("inf")
            verdict = "H2 (steering)" if ratio > 2.0 else "H1 (no steering)"
            lines.append(f"- **{metric}**: W(A,C)={w_ac:.4f}, W(A,A')={w_bootstrap:.4f}, "
                          f"ratio={ratio:.2f}× → {verdict}\n")
        lines.append("\n")

    # Overall verdict
    lines.append("## Overall Verdict\n\n")
    if "ablation3" in summaries:
        base_d = summaries["ablation3"].get("d_b_nuc_angstrom_median")
        for cohort in ["v4d", "v4theta", "v4composite"]:
            if cohort not in summaries or base_d is None:
                continue
            m = summaries[cohort].get("d_b_nuc_angstrom_median")
            if m is None:
                continue
            delta = m - base_d
            lines.append(f"- {cohort}: d_b_nuc median = {m:.3f} vs Ablation baseline "
                          f"{base_d:.3f} ({delta:+.3f})\n")

    out_path.write_text("".join(lines))
    print(f"Wrote {out_path}")


if __name__ == "__main__":
    main()
