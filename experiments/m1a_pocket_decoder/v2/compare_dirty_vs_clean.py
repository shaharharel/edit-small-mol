"""Head-to-head comparison: v2_curriculum (v1_dirty) vs v2_curriculum_clean.

Loads steering_samples cohorts + Boltz cofold track_A CSVs for BOTH runs.
Computes per-arm mean/median d_b_nuc + bd_angle_deg, KS-test p-values,
Wasserstein distances, and bootstrap 95% CIs. Also loads both training
curves to compare final pose-response KL (kl_A_vs_B_theta+1.5 and
kl_A_vs_C_theta-1.5).

Writes:
  data/paper_pair_training/v2_curriculum_clean/comparison_report.md
  data/paper_pair_training/v2_curriculum_clean/comparison_report.json
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
TARGETS = {"theta_90": 90, "theta_105": 105, "theta_130": 130, "null_pose": None}

DIRTY_TRACK = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum/cofold_track"
DIRTY_CURVE = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum/training_curve.json"
DIRTY_STEER = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum/steering_samples"
CLEAN_TRACK = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum_clean/cofold_track"
CLEAN_CURVE = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum_clean/training_curve.json"
CLEAN_STEER = PROJECT_ROOT / "data/paper_pair_training/v2_curriculum_clean/steering_samples"

ACRYL_SMARTS = "C=CC(=O)N"


def _smiles_stats(steer_dir: Path) -> dict:
    """For each cohort, compute: (n_sampled, n_valid, n_acryl_any, n_acryl_largest)."""
    try:
        from rdkit import Chem, RDLogger
        from rdkit.Chem import AllChem
        RDLogger.DisableLog("rdApp.*")
    except Exception:
        return {}
    acryl_patt = Chem.MolFromSmarts(ACRYL_SMARTS)
    out = {}
    for c in COHORTS:
        p = steer_dir / f"samples_{c}.csv"
        if not p.exists():
            out[c] = {"n_sampled": 0, "n_valid": 0, "n_acryl_any": 0, "n_acryl_largest": 0}
            continue
        df = pd.read_csv(p)
        n_sampled = len(df)
        n_valid = 0
        n_acryl_any = 0
        n_acryl_largest = 0
        for s in df["SMILES"].astype(str).values:
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            n_valid += 1
            if m.HasSubstructMatch(acryl_patt):
                n_acryl_any += 1
                # Largest fragment must contain acrylamide
                frags = Chem.GetMolFrags(m, asMols=True)
                if frags:
                    lg = max(frags, key=lambda x: x.GetNumHeavyAtoms())
                    if lg.HasSubstructMatch(acryl_patt):
                        n_acryl_largest += 1
        out[c] = {
            "n_sampled": n_sampled,
            "n_valid": n_valid,
            "n_acryl_any": n_acryl_any,
            "n_acryl_largest": n_acryl_largest,
            "validity_pct": n_valid / n_sampled * 100 if n_sampled else 0.0,
            "acryl_any_pct_of_sampled": n_acryl_any / n_sampled * 100 if n_sampled else 0.0,
            "acryl_largest_pct_of_sampled": n_acryl_largest / n_sampled * 100 if n_sampled else 0.0,
            "acryl_largest_pct_of_valid": n_acryl_largest / n_valid * 100 if n_valid else 0.0,
        }
    return out


def load_tracks(track_dir: Path) -> pd.DataFrame:
    frames = []
    for c in COHORTS:
        p = track_dir / f"track_A_{c}.csv"
        if not p.exists():
            continue
        df = pd.read_csv(p)
        df["cohort"] = c
        frames.append(df)
    if not frames:
        return pd.DataFrame(columns=["cohort", "bd_angle_deg", "d_b_nuc_angstrom"])
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(vals: np.ndarray, stat_fn=np.median, n_boot: int = 1000,
                  alpha: float = 0.05, rng=None) -> tuple[float, float]:
    if rng is None:
        rng = np.random.default_rng(0)
    if len(vals) < 3:
        return (float("nan"), float("nan"))
    boots = np.zeros(n_boot)
    n = len(vals)
    for i in range(n_boot):
        idx = rng.integers(0, n, size=n)
        boots[i] = stat_fn(vals[idx])
    lo, hi = np.quantile(boots, [alpha / 2, 1 - alpha / 2])
    return float(lo), float(hi)


def wasserstein(a: np.ndarray, b: np.ndarray) -> float:
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(stats.wasserstein_distance(a, b))


def per_cohort_stats(df: pd.DataFrame, metric: str) -> dict:
    out = {}
    for c in COHORTS:
        vals = df[df["cohort"] == c][metric].dropna().values
        if len(vals) == 0:
            out[c] = {"n": 0}
            continue
        med_lo, med_hi = bootstrap_ci(vals, np.median)
        mean_lo, mean_hi = bootstrap_ci(vals, np.mean)
        out[c] = {
            "n": int(len(vals)),
            "median": float(np.median(vals)),
            "mean": float(np.mean(vals)),
            "std": float(np.std(vals)),
            "q25": float(np.quantile(vals, 0.25)),
            "q75": float(np.quantile(vals, 0.75)),
            "median_ci95": [med_lo, med_hi],
            "mean_ci95": [mean_lo, mean_hi],
        }
    return out


def pairwise_ks(df: pd.DataFrame, metric: str) -> dict:
    out = {}
    for i, a in enumerate(COHORTS):
        for b in COHORTS[i + 1:]:
            xa = df[df["cohort"] == a][metric].dropna().values
            xb = df[df["cohort"] == b][metric].dropna().values
            if len(xa) < 5 or len(xb) < 5:
                out[f"{a}_vs_{b}"] = {"na": int(len(xa)), "nb": int(len(xb)),
                                       "ks_stat": None, "p_value": None,
                                       "wasserstein": None,
                                       "delta_median": None}
                continue
            s, p = stats.ks_2samp(xa, xb)
            out[f"{a}_vs_{b}"] = {
                "na": int(len(xa)), "nb": int(len(xb)),
                "ks_stat": float(s),
                "p_value": float(p),
                "wasserstein": wasserstein(xa, xb),
                "delta_median": float(np.median(xa) - np.median(xb)),
                "delta_mean": float(np.mean(xa) - np.mean(xb)),
            }
    return out


def load_training_kl(curve_path: Path) -> dict:
    if not curve_path.exists():
        return {"available": False}
    d = json.loads(curve_path.read_text())
    pr = d.get("pose_response", [])
    if not pr:
        return {"available": True, "n_probes": 0}
    kl_ab = [x.get("kl_A_vs_B_theta+1.5", 0.0) for x in pr]
    kl_ac = [x.get("kl_A_vs_C_theta-1.5", 0.0) for x in pr]
    kl_aa = [x.get("kl_A_vs_A_sanity", 0.0) for x in pr]
    return {
        "available": True,
        "n_probes": len(pr),
        "final_kl_ab": kl_ab[-1] if kl_ab else None,
        "final_kl_ac": kl_ac[-1] if kl_ac else None,
        "final_kl_aa_sanity": kl_aa[-1] if kl_aa else None,
        "max_kl_ab": max(kl_ab) if kl_ab else None,
        "max_kl_ac": max(kl_ac) if kl_ac else None,
        "mean_last3_kl_ab": float(np.mean(kl_ab[-3:])) if len(kl_ab) >= 3 else None,
        "mean_last3_kl_ac": float(np.mean(kl_ac[-3:])) if len(kl_ac) >= 3 else None,
        "final_step": d.get("steps", [None])[-1] if d.get("steps") else None,
        "best_val": d.get("best_val"),
        "best_step": d.get("best_step"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(
        PROJECT_ROOT / "data/paper_pair_training/v2_curriculum_clean/comparison_report.md"))
    ap.add_argument("--out_json", default=str(
        PROJECT_ROOT / "data/paper_pair_training/v2_curriculum_clean/comparison_report.json"))
    args = ap.parse_args()

    dirty_df = load_tracks(DIRTY_TRACK)
    clean_df = load_tracks(CLEAN_TRACK)
    dirty_kl = load_training_kl(DIRTY_CURVE)
    clean_kl = load_training_kl(CLEAN_CURVE)
    dirty_smi = _smiles_stats(DIRTY_STEER)
    clean_smi = _smiles_stats(CLEAN_STEER)

    print(f"[compare] dirty rows: {len(dirty_df)}; clean rows: {len(clean_df)}")
    print(f"[compare] dirty cohorts present: {sorted(dirty_df['cohort'].unique()) if len(dirty_df) else []}")
    print(f"[compare] clean cohorts present: {sorted(clean_df['cohort'].unique()) if len(clean_df) else []}")

    metrics = ["bd_angle_deg", "d_b_nuc_angstrom"]
    summary = {
        "dirty": {},
        "clean": {},
        "kl": {"dirty": dirty_kl, "clean": clean_kl},
        "smiles_stats": {"dirty": dirty_smi, "clean": clean_smi},
    }
    for m in metrics:
        summary["dirty"][m] = {
            "per_cohort": per_cohort_stats(dirty_df, m),
            "pairwise_ks": pairwise_ks(dirty_df, m),
        }
        summary["clean"][m] = {
            "per_cohort": per_cohort_stats(clean_df, m),
            "pairwise_ks": pairwise_ks(clean_df, m),
        }

    # ------------ Markdown report ------------
    L = []
    L.append("# v2 Curriculum — DIRTY vs CLEAN head-to-head comparison")
    L.append("")
    L.append(f"- Dirty cofold rows: N={len(dirty_df)} across cohorts "
             f"{sorted(dirty_df['cohort'].unique().tolist()) if len(dirty_df) else []}")
    L.append(f"- Clean cofold rows: N={len(clean_df)} across cohorts "
             f"{sorted(clean_df['cohort'].unique().tolist()) if len(clean_df) else []}")
    L.append("")

    # -------- Training-curve KL --------
    L.append("## 1) Training-curve pose-response KL (val-time probe)")
    L.append("")
    L.append("Measures how much the decoder's next-token logits change when the "
             "[POSE] input is shifted by ±1.5 std on theta_z. KL≈0 means the "
             "decoder ignores pose; KL>0 means it listens.")
    L.append("")
    L.append("| Arm | final_step | final KL_A_vs_B (+1.5σ) | final KL_A_vs_C (−1.5σ) | mean last-3 KL_ab | mean last-3 KL_ac | KL_A_vs_A (sanity) |")
    L.append("|-----|-----------:|------------------------:|------------------------:|-------------------:|-------------------:|--------------------:|")

    def _fmt(v, dp=3):
        return f"{v:.{dp}f}" if isinstance(v, (int, float)) and v is not None else "n/a"

    for tag, kl in [("v1_dirty", dirty_kl), ("v1_clean", clean_kl)]:
        if not kl.get("available"):
            L.append(f"| {tag} | (missing training_curve.json) |||||")
            continue
        L.append(f"| {tag} | {kl.get('final_step')} | "
                 f"{_fmt(kl.get('final_kl_ab'))} | "
                 f"{_fmt(kl.get('final_kl_ac'))} | "
                 f"{_fmt(kl.get('mean_last3_kl_ab'))} | "
                 f"{_fmt(kl.get('mean_last3_kl_ac'))} | "
                 f"{_fmt(kl.get('final_kl_aa_sanity'))} |")
    L.append("")

    # -------- SMILES validity + acrylamide retention (steering samples pre-cofold) --------
    L.append("## 1b) Decoder sample quality (pre-cofold)")
    L.append("")
    L.append("Steering samples validity and acrylamide retention. Low validity in v1_clean is a "
             "known caveat: the strict AND filter cut training pairs from 6340 -> 2248 (-64.5%), and "
             "the same 10k-step budget on the smaller dataset leads to earlier overfitting.")
    L.append("")
    L.append("| Cohort | Arm | n sampled | valid % | acryl any % of sampled | acryl-largest % of sampled | acryl-largest % of valid |")
    L.append("|--------|-----|----------:|--------:|------------------------:|-----------------------------:|--------------------------:|")
    for c in COHORTS:
        for tag, smi in [("v1_dirty", dirty_smi), ("v1_clean", clean_smi)]:
            s = smi.get(c, {})
            if not s or s.get("n_sampled", 0) == 0:
                L.append(f"| {c} | {tag} | 0 | n/a | n/a | n/a | n/a |")
                continue
            L.append(f"| {c} | {tag} | {s['n_sampled']} | "
                     f"{s['validity_pct']:.1f}% | "
                     f"{s['acryl_any_pct_of_sampled']:.1f}% | "
                     f"{s['acryl_largest_pct_of_sampled']:.1f}% | "
                     f"{s['acryl_largest_pct_of_valid']:.1f}% |")
    L.append("")

    # -------- Per-cohort bd_angle_deg (theta) --------
    for m, label in [("bd_angle_deg", "output bd_angle_deg (theta)"),
                      ("d_b_nuc_angstrom", "output d_b_nuc (Å)")]:
        L.append(f"## 2) Boltz-observed {label} — per cohort")
        L.append("")
        L.append("| Cohort | Target | Arm | N | median [95% CI] | mean [95% CI] | std | IQR |")
        L.append("|--------|-------:|-----|--:|-----------------|---------------|-----|-----|")
        for c in COHORTS:
            tt = TARGETS[c]
            for tag, arm in [("v1_dirty", summary["dirty"][m]["per_cohort"]),
                              ("v1_clean", summary["clean"][m]["per_cohort"])]:
                s = arm.get(c, {"n": 0})
                if s.get("n", 0) == 0:
                    L.append(f"| {c} | {tt if tt else 'null'} | {tag} | 0 | n/a | n/a | n/a | n/a |")
                    continue
                med = f"{s['median']:.2f} [{s['median_ci95'][0]:.2f},{s['median_ci95'][1]:.2f}]"
                mn = f"{s['mean']:.2f} [{s['mean_ci95'][0]:.2f},{s['mean_ci95'][1]:.2f}]"
                L.append(f"| {c} | {tt if tt else 'null'} | {tag} | {s['n']} | "
                         f"{med} | {mn} | {s['std']:.2f} | "
                         f"[{s['q25']:.2f},{s['q75']:.2f}] |")
        L.append("")

    # -------- Pairwise KS tests on bd_angle_deg --------
    L.append("## 3) Pairwise cohort KS-tests (bd_angle_deg)")
    L.append("")
    L.append("Small p-values (<0.01 **SIG**) mean the two cohorts produce "
             "significantly different output theta distributions — i.e., the "
             "model IS steering. Wasserstein is distribution shift magnitude.")
    L.append("")
    L.append("| Comparison | Arm | n_a | n_b | KS stat | p-value | Wasserstein | Δmedian | Δmean |")
    L.append("|------------|-----|----:|----:|--------:|--------:|-------------:|--------:|-------:|")
    for k in summary["dirty"]["bd_angle_deg"]["pairwise_ks"].keys():
        for tag, arm in [("v1_dirty", summary["dirty"]["bd_angle_deg"]["pairwise_ks"]),
                          ("v1_clean", summary["clean"]["bd_angle_deg"]["pairwise_ks"])]:
            v = arm.get(k, {})
            if v.get("p_value") is None:
                L.append(f"| {k} | {tag} | {v.get('na', 0)} | {v.get('nb', 0)} | n/a | n/a | n/a | n/a | n/a |")
                continue
            sig = "**SIG**" if v["p_value"] < 0.01 else ""
            L.append(f"| {k} | {tag} | {v['na']} | {v['nb']} | "
                     f"{v['ks_stat']:.3f} | {v['p_value']:.2e} {sig} | "
                     f"{v['wasserstein']:.2f} | "
                     f"{v['delta_median']:+.2f} | {v['delta_mean']:+.2f} |")
    L.append("")

    # -------- Verdict --------
    L.append("## 4) Verdict")
    L.append("")
    # Compute headline steering strength: median_theta_130 - median_theta_90
    def _steering(arm_stats):
        p = arm_stats["bd_angle_deg"]["per_cohort"]
        s90 = p.get("theta_90", {}).get("median")
        s130 = p.get("theta_130", {}).get("median")
        if s90 is None or s130 is None:
            return None
        return s130 - s90

    dirty_steer = _steering(summary["dirty"])
    clean_steer = _steering(summary["clean"])
    L.append(f"- Median θ(theta_130) − θ(theta_90) response:")
    L.append(f"  - v1_dirty: {dirty_steer:.2f}°" if dirty_steer is not None else "  - v1_dirty: n/a")
    L.append(f"  - v1_clean: {clean_steer:.2f}°" if clean_steer is not None else "  - v1_clean: n/a")
    if dirty_steer is not None and clean_steer is not None:
        winner = "v1_clean" if abs(clean_steer) > abs(dirty_steer) else "v1_dirty"
        L.append(f"  - Larger absolute response: **{winner}**")

    # Training-curve KL winner
    if dirty_kl.get("available") and clean_kl.get("available"):
        d_kl_avg = np.mean([dirty_kl.get("mean_last3_kl_ab") or 0.0,
                             dirty_kl.get("mean_last3_kl_ac") or 0.0])
        c_kl_avg = np.mean([clean_kl.get("mean_last3_kl_ab") or 0.0,
                             clean_kl.get("mean_last3_kl_ac") or 0.0])
        kl_winner = "v1_clean" if c_kl_avg > d_kl_avg else "v1_dirty"
        L.append(f"- Training-curve pose-response KL (avg |ab|+|ac|, last 3 probes):")
        L.append(f"  - v1_dirty: {d_kl_avg:.3f}")
        L.append(f"  - v1_clean: {c_kl_avg:.3f}")
        L.append(f"  - Stronger [POSE] response: **{kl_winner}**")
    L.append("")

    # Sample-quality caveat: report validity gap explicitly
    def _mean_validity(smi_stats):
        vals = [s["validity_pct"] for s in smi_stats.values() if s.get("n_sampled", 0)]
        return float(np.mean(vals)) if vals else None

    d_val = _mean_validity(dirty_smi)
    c_val = _mean_validity(clean_smi)
    if d_val is not None and c_val is not None:
        L.append(f"- Sample validity (avg across cohorts):")
        L.append(f"  - v1_dirty: {d_val:.1f}%")
        L.append(f"  - v1_clean: {c_val:.1f}%")
        if c_val < 70.0:
            L.append(f"  - **CAVEAT**: v1_clean validity is well below 70%. The 64.5% training-pair "
                     f"cut from strict filtering caused earlier decoder overfitting. Cofold-based "
                     f"steering conclusions apply only to the VALID subset ({c_val:.0f}% of samples).")
    L.append("")
    L.append("## 5) Recommended next-steps")
    L.append("")
    L.append("- If v1_clean beats v1_dirty on BOTH training KL and Boltz-observed steering:")
    L.append("  the 'clean filter' hypothesis is defended — enforce strict AND filter for all future curriculum runs.")
    L.append("- If v1_clean beats v1_dirty on training KL but NOT on Boltz-observed steering:")
    L.append("  the model listens more but downstream cofold does not respond — pose axis or ")
    L.append("  encoder-decoder integration may be the bottleneck; consider explicit pose distillation loss.")
    L.append("- If v1_dirty ≥ v1_clean:")
    L.append("  data volume beats data purity at this scale. Consider curriculum-only mode (drop crystal),")
    L.append("  larger dtheta_gap / dd_gap thresholds, or moving to Scheme-D (multi-pocket contrastive).")

    Path(args.out_md).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out_md).write_text("\n".join(L))
    Path(args.out_json).write_text(json.dumps(summary, indent=2, default=str))
    print(f"[compare] wrote {args.out_md}")
    print(f"[compare] wrote {args.out_json}")


if __name__ == "__main__":
    main()
