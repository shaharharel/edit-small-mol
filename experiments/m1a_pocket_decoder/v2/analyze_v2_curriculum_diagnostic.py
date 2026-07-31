"""Analyze the Phase 1e steering diagnostic: does the retrained v2 respond
to [POSE] conditioning?

Inputs (all under data/paper_pair_training/v2_curriculum/):
  steering_samples/samples_{theta_90,theta_105,theta_130,null_pose}.csv
  cofold_track/track_A_{theta_90,theta_105,theta_130,null_pose}.csv
    (parsed geometry produced by boltz_dpo_campaign_driver.py)

Output:
  steering_diagnostic_post.csv  — combined table (cohort × sample × output theta)
  steering_comparison.md        — pre vs post retrain comparison + KS-test verdict

Pre-retrain baseline: routing_diagnostic/T05_pose_shift.json.
The Phase 0 diagnostic measured planar-dihedral shift at T=0.5; delta_AC ~ 0.08°
(essentially no response). Our post-retrain metric is the OUTPUT bd_angle_deg
distribution across cohorts (theta_90 vs theta_105 vs theta_130), which is a
STRONGER signal — we're conditioning on the same axis (bd_angle) rather than
inferring shift on a downstream axis.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

COHORTS = ["theta_90", "theta_105", "theta_130", "null_pose"]
BASE_DIR = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum"


def load_cofold_tracks(track_dir: Path) -> pd.DataFrame:
    frames = []
    for c in COHORTS:
        p = track_dir / f"track_A_{c}.csv"
        if not p.exists():
            print(f"[analyze] WARNING: missing {p}")
            continue
        df = pd.read_csv(p)
        df["cohort"] = c
        frames.append(df)
    if not frames:
        raise SystemExit(f"No cofold tracks under {track_dir}")
    return pd.concat(frames, ignore_index=True)


def summarize(df: pd.DataFrame) -> dict:
    out = {}
    for c in COHORTS:
        sub = df[df["cohort"] == c]
        theta = sub["bd_angle_deg"].dropna().values if "bd_angle_deg" in sub else np.array([])
        d = sub["d_b_nuc_angstrom"].dropna().values if "d_b_nuc_angstrom" in sub else np.array([])
        iptm = sub["complex_iptm"].dropna().values if "complex_iptm" in sub else np.array([])
        out[c] = {
            "n_cofolded": int(len(sub)),
            "n_valid_theta": int(len(theta)),
            "theta_mean": float(theta.mean()) if len(theta) else None,
            "theta_median": float(np.median(theta)) if len(theta) else None,
            "theta_std": float(theta.std()) if len(theta) else None,
            "theta_q25": float(np.quantile(theta, 0.25)) if len(theta) else None,
            "theta_q75": float(np.quantile(theta, 0.75)) if len(theta) else None,
            "d_median": float(np.median(d)) if len(d) else None,
            "iptm_median": float(np.median(iptm)) if len(iptm) else None,
        }
    return out


def ks_pairs(df: pd.DataFrame) -> dict:
    """KS-test on bd_angle_deg between every pair of cohorts."""
    out = {}
    for i, a in enumerate(COHORTS):
        for b in COHORTS[i+1:]:
            xa = df[df["cohort"] == a]["bd_angle_deg"].dropna().values
            xb = df[df["cohort"] == b]["bd_angle_deg"].dropna().values
            if len(xa) < 5 or len(xb) < 5:
                out[f"{a}_vs_{b}"] = {"na": int(len(xa)), "nb": int(len(xb)),
                                       "ks_stat": None, "p_value": None}
                continue
            s, p = stats.ks_2samp(xa, xb)
            out[f"{a}_vs_{b}"] = {"na": int(len(xa)), "nb": int(len(xb)),
                                   "ks_stat": float(s), "p_value": float(p),
                                   "median_a": float(np.median(xa)),
                                   "median_b": float(np.median(xb)),
                                   "delta_median": float(np.median(xa) - np.median(xb))}
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--track_dir", default=str(BASE_DIR / "cofold_track"))
    ap.add_argument("--out_csv", default=str(BASE_DIR / "steering_diagnostic_post.csv"))
    ap.add_argument("--out_md", default=str(BASE_DIR / "steering_comparison.md"))
    ap.add_argument("--prior_diag_json", default=str(PROJECT_ROOT /
                    "data/paper_pair_training/mpae_zap70/routing_diagnostic/T05_pose_shift.json"))
    args = ap.parse_args()

    df = load_cofold_tracks(Path(args.track_dir))
    df.to_csv(args.out_csv, index=False)
    print(f"[analyze] merged N={len(df)} rows -> {args.out_csv}")

    summ = summarize(df)
    ks = ks_pairs(df)

    # ---- Load prior diagnostic ----
    prior = {}
    prior_path = Path(args.prior_diag_json)
    if prior_path.exists():
        prior = json.loads(prior_path.read_text())

    # ---- Build markdown report ----
    lines = ["# Phase 1e — v2 Curriculum Steering Diagnostic"]
    lines.append("")
    lines.append(f"Total cofolded mols: N={len(df)}, cohorts={sorted(df['cohort'].unique().tolist())}")
    lines.append("")
    lines.append("## Per-cohort output bd_angle statistics")
    lines.append("")
    lines.append("| Cohort | Target θ (deg) | N cofolded | N valid θ | median θ | mean θ | std θ | IQR |")
    lines.append("|---|---|---|---|---|---|---|---|")
    targets = {"theta_90": 90, "theta_105": 105, "theta_130": 130, "null_pose": None}
    for c in COHORTS:
        s = summ.get(c, {})
        tt = targets[c]
        med = f"{s['theta_median']:.1f}" if s.get('theta_median') is not None else "n/a"
        mn = f"{s['theta_mean']:.1f}" if s.get('theta_mean') is not None else "n/a"
        std = f"{s['theta_std']:.1f}" if s.get('theta_std') is not None else "n/a"
        iqr = (f"[{s['theta_q25']:.1f},{s['theta_q75']:.1f}]"
               if s.get('theta_q25') is not None else "n/a")
        lines.append(f"| {c} | {tt if tt else 'null'} | {s.get('n_cofolded', 0)} | "
                     f"{s.get('n_valid_theta', 0)} | {med} | {mn} | {std} | {iqr} |")
    lines.append("")
    lines.append("## KS-tests on output θ distributions")
    lines.append("")
    lines.append("| Pair | n_a | n_b | KS stat | p-value | Δ median (deg) |")
    lines.append("|---|---|---|---|---|---|")
    any_sig = False
    for k, v in ks.items():
        if v.get("p_value") is None:
            lines.append(f"| {k} | {v['na']} | {v['nb']} | n/a | n/a | n/a |")
        else:
            ma = v.get("median_a")
            mb = v.get("median_b")
            dm = v.get("delta_median", 0.0)
            sig = "**SIG**" if v["p_value"] < 0.01 else ""
            if v["p_value"] < 0.01: any_sig = True
            lines.append(f"| {k} | {v['na']} | {v['nb']} | {v['ks_stat']:.3f} | "
                         f"{v['p_value']:.2e} {sig} | {dm:+.2f} |")
    lines.append("")

    # ---- Verdict ----
    lines.append("## Verdict")
    lines.append("")
    if prior:
        prior_shift = prior.get("T05_delta_AC_deg", 0.0)
        lines.append(f"**Pre-retrain baseline** (Phase 0, `T05_pose_shift.json`):")
        lines.append(f"- Delta AC at T=0.5 (planar dihedral, indirect axis): {prior_shift:.2f}°")
        lines.append(f"- Delta AC at T=1.0: {prior.get('T10_delta_AC_deg', 0.0):.2f}°")
        lines.append(f"- N valid per clamp: A={prior.get('T05_n_valid_A', 0)}, "
                     f"C={prior.get('T05_n_valid_C', 0)}")
        lines.append("")
    # Post-retrain response magnitude: theta_90 vs theta_130 median difference
    post_shift = 0.0
    if summ.get("theta_90", {}).get("theta_median") is not None and \
       summ.get("theta_130", {}).get("theta_median") is not None:
        post_shift = abs(summ["theta_130"]["theta_median"] -
                          summ["theta_90"]["theta_median"])
    lines.append(f"**Post-retrain response** (bd_angle_deg, direct axis):")
    lines.append(f"- Δ median θ(cohort=theta_130) - θ(cohort=theta_90) = {post_shift:.2f}°")
    lines.append(f"- Any pairwise KS p<0.01 = **{'YES' if any_sig else 'NO'}**")
    lines.append("")
    if any_sig and post_shift > 5.0:
        lines.append("## WIN — architectural claim DEFENDED")
        lines.append("")
        lines.append("Post-retrain v2 shows POSE-conditioned steering: output θ distributions "
                     "differ significantly (KS p<0.01) between cohorts, and the direct "
                     "θ response is > 5° (well above pre-retrain <0.1°).")
    elif any_sig:
        lines.append("## PARTIAL — significant but weak")
        lines.append("")
        lines.append("KS-test is significant but the median θ response is < 5°. The model "
                     "responds to pose conditioning but not strongly; further curriculum "
                     "iterations or larger LR may improve steering strength.")
    else:
        lines.append("## RED FLAG — steering did not emerge")
        lines.append("")
        lines.append("Post-retrain v2 still ignores the [POSE] token: no pairwise "
                     "KS-test reaches p<0.01. The curriculum-data hypothesis is FALSIFIED. "
                     "Next steps: check pose-encoder gradient flow, add explicit pose "
                     "distillation loss (KL against a pose-oracle head), or reconsider the "
                     "architectural placement of the [POSE] token (currently prepended to "
                     "encoder memory — try injecting into every decoder layer).")

    Path(args.out_md).write_text("\n".join(lines))
    print(f"[analyze] wrote {args.out_md}")

    # Also dump the raw summary JSON
    summary_json = {
        "per_cohort": summ,
        "ks_tests": ks,
        "post_shift_theta130_minus_theta90": post_shift,
        "prior_diagnostic": prior,
        "any_pairwise_ks_sig_p001": any_sig,
    }
    Path(str(args.out_md).replace(".md", ".json")).write_text(
        json.dumps(summary_json, indent=2, default=str))


if __name__ == "__main__":
    main()
