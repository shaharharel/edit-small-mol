"""Produce the CFG × retrieval final report.

Reads `covalent_metric_panel.csv` and writes:
  - data/paper_pair_training/cfg_retrieval/cfg_retrieval_report.md
  - data/paper_pair_training/cfg_retrieval/cfg_retrieval_report_summary.csv

Baseline for effect-size comparisons: cell = cfg_s1_retrieval_off  (covFT + v2
pocket/pose, no CFG boost since s=1.0 collapses to the conditional branch,
no retrieval prefix).

Primary endpoint (per project memory, since xTB k_inact isn't available on
this machine):
  - acryl_largest_frac  — fraction of samples whose LARGEST FRAGMENT contains
    an acrylamide C=C-C(=O)-N group. This is the headline covalent-generation
    metric per `overnight_pipeline_complete_2026_05_27.md`.

Secondary endpoints:
  - bd_ready_frac  — fraction with acryl_largest AND planar_dihedral <= 30°
  - median_planar_dihedral_deg  — among acryl_largest samples
  - fukui_fplus_proxy_mean  — mean Gasteiger charge on Cβ (electrophilicity
    proxy; higher = more electrophilic)
  - vina_cov_score_median  — median Vina score (only reported if any cell has
    Vina scores)

Effect size vs baseline: absolute difference, plus 95% bootstrap CI on the
difference (5000 resamples, per-cell resampling with replacement).

Kill-criteria checks emit warnings into the report:
  - validity < 10% at any CFG scale >= 3
  - k_inact CI crosses zero (we sub in BD-ready-frac CI here)
"""
from __future__ import annotations
import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
    """Render a pandas DataFrame as GitHub-flavored markdown WITHOUT the
    `tabulate` dependency (not installed on ai-gpu)."""
    if df is None or len(df) == 0:
        return "*(empty)*"
    cols = list(df.columns)
    def fmt(v):
        if v is None:
            return ""
        if isinstance(v, float):
            if np.isnan(v):
                return "NaN"
            return f"{v:{floatfmt[1:]}}"
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body_lines = []
    for _, row in df.iterrows():
        body_lines.append("| " + " | ".join(fmt(row[c]) for c in cols) + " |")
    return "\n".join([header, sep] + body_lines)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, agg="mean",
                        n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    """Return (point, lo95, hi95) for statistic(a) - statistic(b)."""
    a = np.asarray(a); b = np.asarray(b)
    a = a[~np.isnan(a)] if agg == "median" else a
    b = b[~np.isnan(b)] if agg == "median" else b
    rng = np.random.default_rng(seed)
    if agg == "mean":
        def f(x): return float(np.nanmean(x)) if len(x) else np.nan
    elif agg == "median":
        def f(x): return float(np.median(x)) if len(x) else np.nan
    else:
        raise ValueError(agg)
    point = f(a) - f(b)
    if len(a) == 0 or len(b) == 0:
        return point, np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs[i] = f(a[ia]) - f(b[ib])
    lo = float(np.nanpercentile(diffs, 2.5))
    hi = float(np.nanpercentile(diffs, 97.5))
    return point, lo, hi


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_csv", default="/home/shaharh_quris_ai/edit-small-mol/"
                                             "data/paper_pair_training/cfg_retrieval/"
                                             "covalent_metric_panel.csv")
    ap.add_argument("--out_md", default="/home/shaharh_quris_ai/edit-small-mol/"
                                          "data/paper_pair_training/cfg_retrieval/"
                                          "cfg_retrieval_report.md")
    ap.add_argument("--out_summary_csv", default="/home/shaharh_quris_ai/edit-small-mol/"
                                                    "data/paper_pair_training/cfg_retrieval/"
                                                    "cfg_retrieval_report_summary.csv")
    ap.add_argument("--baseline_cell", default="cfg_s1_retrieval_off")
    args = ap.parse_args()

    df = pd.read_csv(args.panel_csv)
    print(f"Loaded {len(df)} rows from {args.panel_csv}", flush=True)
    print(f"Cells: {df['cell'].unique()}", flush=True)

    # Per-cell summary.
    summary_rows = []
    for cell, cell_df in df.groupby("cell"):
        s = cell_df.iloc[0]
        n = len(cell_df)
        valid = cell_df["valid"].astype(bool)
        n_valid = int(valid.sum())
        acryl_frac = float(cell_df.loc[valid, "acryl_largest"].mean()
                             if n_valid else np.nan)
        # BD-ready = acryl + dihedral <= 30
        bd_series = cell_df["acryl_largest"].astype(bool) & \
                     (cell_df["planar_dihedral_deg"] <= 30.0)
        bd_frac = float(bd_series[valid].mean() if n_valid else np.nan)
        dih_series = cell_df.loc[valid & cell_df["acryl_largest"].astype(bool),
                                    "planar_dihedral_deg"].astype(float).dropna()
        median_dih = float(np.median(dih_series)) if len(dih_series) else np.nan
        fplus_series = cell_df.loc[valid & cell_df["acryl_largest"].astype(bool),
                                      "fukui_fplus_proxy"].astype(float).dropna()
        fplus_mean = float(np.nanmean(fplus_series)) if len(fplus_series) else np.nan
        vina_series = cell_df["vina_cov_score"].astype(float).dropna() \
                        if "vina_cov_score" in cell_df.columns else pd.Series(dtype=float)
        vina_med = float(np.nanmedian(vina_series)) if len(vina_series) else np.nan
        vina_n = int(len(vina_series))
        summary_rows.append({
            "cell": cell,
            "cfg_scale": float(s["cfg_scale"]),
            "retrieval": int(s["retrieval"]),
            "n_total": n,
            "n_valid": n_valid,
            "valid_frac": n_valid / max(1, n),
            "acryl_largest_frac": acryl_frac,
            "bd_ready_frac": bd_frac,
            "median_planar_dihedral_deg": median_dih,
            "mean_fukui_fplus_proxy": fplus_mean,
            "vina_cov_score_median": vina_med,
            "vina_n_docked": vina_n,
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["retrieval", "cfg_scale"]).reset_index(drop=True)
    summary.to_csv(args.out_summary_csv, index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}", flush=True)

    # Effect sizes vs baseline.
    if args.baseline_cell not in df["cell"].unique():
        print(f"[warn] baseline cell {args.baseline_cell} not in data", flush=True)
        baseline_cell_df = None
    else:
        baseline_cell_df = df[df["cell"] == args.baseline_cell]
    effects = []
    if baseline_cell_df is not None:
        base_valid = baseline_cell_df["valid"].astype(bool)
        base_acryl = baseline_cell_df.loc[base_valid,
                                             "acryl_largest"].astype(float).values
        base_bd = (baseline_cell_df["acryl_largest"].astype(bool) &
                   (baseline_cell_df["planar_dihedral_deg"] <= 30.0))
        base_bd_arr = base_bd[base_valid].astype(float).values
        base_dih = baseline_cell_df.loc[
            base_valid & baseline_cell_df["acryl_largest"].astype(bool),
            "planar_dihedral_deg"].astype(float).dropna().values
        base_fplus = baseline_cell_df.loc[
            base_valid & baseline_cell_df["acryl_largest"].astype(bool),
            "fukui_fplus_proxy"].astype(float).dropna().values
        for cell, cell_df in df.groupby("cell"):
            if cell == args.baseline_cell:
                continue
            v = cell_df["valid"].astype(bool)
            a_arr = cell_df.loc[v, "acryl_largest"].astype(float).values
            bd_arr = (cell_df["acryl_largest"].astype(bool) &
                       (cell_df["planar_dihedral_deg"] <= 30.0)
                       )[v].astype(float).values
            dih_arr = cell_df.loc[
                v & cell_df["acryl_largest"].astype(bool),
                "planar_dihedral_deg"].astype(float).dropna().values
            fplus_arr = cell_df.loc[
                v & cell_df["acryl_largest"].astype(bool),
                "fukui_fplus_proxy"].astype(float).dropna().values
            row = {"cell": cell}
            p, lo, hi = bootstrap_diff_ci(a_arr, base_acryl)
            row["d_acryl_largest_frac"] = p
            row["d_acryl_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            p, lo, hi = bootstrap_diff_ci(bd_arr, base_bd_arr)
            row["d_bd_ready_frac"] = p
            row["d_bd_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            p, lo, hi = bootstrap_diff_ci(dih_arr, base_dih, agg="median")
            row["d_median_planar_deg"] = p
            row["d_planar_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            p, lo, hi = bootstrap_diff_ci(fplus_arr, base_fplus)
            row["d_mean_fukui_fplus"] = p
            row["d_fukui_ci95"] = f"[{lo:+.4f}, {hi:+.4f}]"
            effects.append(row)
    effects_df = pd.DataFrame(effects)

    # Kill-criteria warnings.
    warnings_list = []
    for _, r in summary.iterrows():
        if r["cfg_scale"] >= 3.0 and r["valid_frac"] < 0.10:
            warnings_list.append(
                f"KILL: validity {r['valid_frac']:.3f} at s={r['cfg_scale']} "
                f"cell {r['cell']} — CFG likely broken.")
    # BD-ready CI crossing zero for ALL CFG-on cells (retrieval doesn't matter):
    cfg_on_effects = effects_df[
        effects_df["cell"].astype(str).str.startswith("cfg_s2") |
        effects_df["cell"].astype(str).str.startswith("cfg_s3") |
        effects_df["cell"].astype(str).str.startswith("cfg_s5")
    ] if len(effects_df) else pd.DataFrame()
    if len(cfg_on_effects) > 0:
        def crosses_zero(ci_str):
            try:
                a, b = ci_str.strip("[]").split(",")
                lo = float(a); hi = float(b)
                return lo <= 0 <= hi
            except Exception:
                return True
        all_cross = cfg_on_effects["d_bd_ci95"].map(crosses_zero).all()
        if bool(all_cross):
            warnings_list.append(
                "KILL: BD-ready-frac 95% CI crosses zero for every CFG "
                "cell — CFG does not help.")

    # Write markdown.
    lines = [
        f"# CFG + Retrieval-Prefix Ablation Report",
        "",
        f"Panel: `{args.panel_csv}`  ({len(df)} rows)  ",
        f"Baseline cell: `{args.baseline_cell}`  ",
        "",
        "## Primary endpoint — acrylamide retention on LARGEST fragment",
        "",
        "This is the headline covalent-generation metric (per project ",
        "memory `overnight_pipeline_complete_2026_05_27.md`). ",
        "xTB k_inact was requested as primary but xTB is not available on ",
        "this GPU node; the largest-fragment acrylamide-retention fraction ",
        "is used as a stand-in headline. BD-ready fraction (acryl + planar ",
        "dihedral <= 30°) is the secondary geometric endpoint.",
        "",
        "## Per-cell metrics",
        "",
        df_to_md(summary),
        "",
    ]
    if len(effects_df):
        lines += [
            "## Effect vs baseline (bootstrap 95% CI, 5000 resamples)",
            "",
            df_to_md(effects_df),
            "",
        ]
    if warnings_list:
        lines += ["## KILL-criteria warnings", ""]
        for w in warnings_list:
            lines += [f"- {w}"]
        lines += [""]
    else:
        lines += ["## KILL-criteria warnings", "",
                    "None triggered.", ""]
    lines += [
        "## Method notes",
        "",
        "- Training: warm-started from `models/m1a_v2.ckpt` (covFT + v2 ",
        "pocket/pose conditioning). CFG dropout p=0.15 applied jointly to ",
        "[POCKET]+[POSE], replaced with a single learned null token. AdamW, ",
        "lr=5e-5 with cosine decay + 200-step warmup.",
        "- Sampling: for each cell, 200 SMILES from Mol1 anchor",
        "  `C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1` with ZAP70 pocket ",
        "  ESM readout + Mol1 cofold pose (invariant 3-dim). CFG combines ",
        "  logits per token: `logit_final = logit_uncond + s*(logit_cond - ",
        "  logit_uncond)`; s=1 collapses to pure conditional branch.",
        "- Retrieval: top-5 CovInDB v2 SMILES ranked by cosine similarity of ",
        "  mean-pooled ESM residue embeddings to ZAP70 pocket, with all ",
        "  ZAP70 PDBs (2OZO, 2OQ1, 4K2R, 1U59, 4XZ0, ...) leave-one-out. ",
        "  Encoded via the mol2mol encoder + mean-pool into (D=256) tokens ",
        "  and prepended to the decoder memory.",
        "- Fukui f+ proxy: Gasteiger charge on the β-C of the acrylamide ",
        "  (higher = more electrophilic). Real Fukui f+ would require xTB, ",
        "  which is not available on this GPU. Interpret with caution.",
        "- Vina cov score: reported only if `--vina_topn>0` was set for the ",
        "  panel; scores are top-N-per-cell only (BD-ready + lowest NLL).",
    ]
    Path(args.out_md).write_text("\n".join(lines))
    print(f"\nWrote {args.out_md}", flush=True)
    print(f"Wrote {args.out_summary_csv}", flush=True)


if __name__ == "__main__":
    main()
