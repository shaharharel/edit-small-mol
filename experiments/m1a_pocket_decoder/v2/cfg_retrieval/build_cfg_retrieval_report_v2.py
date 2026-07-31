"""Produce the v2 CFG × retrieval final report.

Reads:
  - `covalent_metric_panel_v2.csv` (RDKit-2D metrics, all 1600 samples)
  - `covalent_cofold_panel.csv`   (real BD geometry + xTB Fukui + Vina;
                                      subset per cell)

Writes:
  - `cfg_retrieval_report_v2.md`
  - `cfg_retrieval_report_v2_summary.csv`

Baseline for effect-size comparisons: cell = cfg_s1_retrieval_off (covFT+v2
CFG-trained checkpoint, no CFG boost, no retrieval).

PRIMARY endpoint: mean xTB Fukui f+ at the acrylamide β-C among BD-ready
cofolded samples (bootstrap 95% CI vs baseline).
SECONDARY:
  - bd_ready_frac (cofold-derived: d_SG ∈ [2.5,5.5]Å AND theta ∈ [80,130]°)
  - vina_cov_score_median among BD-ready
  - valid_frac (KILL if <30% at any cell)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_md(df: pd.DataFrame, floatfmt: str = ".4f") -> str:
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
    body = ["| " + " | ".join(fmt(row[c]) for c in cols) + " |"
             for _, row in df.iterrows()]
    return "\n".join([header, sep] + body)


def bootstrap_diff_ci(a: np.ndarray, b: np.ndarray, agg="mean",
                        n_boot: int = 5000, seed: int = 0) -> tuple[float, float, float]:
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    if agg == "mean":
        f = lambda x: float(np.mean(x)) if len(x) else np.nan
    elif agg == "median":
        f = lambda x: float(np.median(x)) if len(x) else np.nan
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
    return point, float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--panel_2d_csv", default="/home/shaharh_quris_ai/edit-small-mol/"
                                                  "data/paper_pair_training/cfg_retrieval/"
                                                  "covalent_metric_panel_v2.csv")
    ap.add_argument("--panel_cofold_csv", default="/home/shaharh_quris_ai/edit-small-mol/"
                                                       "data/paper_pair_training/cfg_retrieval/"
                                                       "covalent_cofold_panel.csv")
    ap.add_argument("--out_md", default="/home/shaharh_quris_ai/edit-small-mol/"
                                           "data/paper_pair_training/cfg_retrieval/"
                                           "cfg_retrieval_report_v2.md")
    ap.add_argument("--out_summary_csv", default="/home/shaharh_quris_ai/edit-small-mol/"
                                                     "data/paper_pair_training/cfg_retrieval/"
                                                     "cfg_retrieval_report_v2_summary.csv")
    ap.add_argument("--baseline_cell", default="cfg_s1_retrieval_off")
    args = ap.parse_args()

    p2d = pd.read_csv(args.panel_2d_csv)
    cofold_available = Path(args.panel_cofold_csv).exists()
    pco = pd.read_csv(args.panel_cofold_csv) if cofold_available else None
    print(f"2D panel: {len(p2d)} rows over {p2d['cell'].nunique()} cells", flush=True)
    if pco is not None:
        print(f"Cofold panel: {len(pco)} rows", flush=True)

    # Per-cell 2D summary.
    summary_rows = []
    for cell, cd in p2d.groupby("cell"):
        n = len(cd); valid = cd["valid"].astype(bool)
        n_v = int(valid.sum())
        acryl_frac = float(cd.loc[valid, "acryl_largest"].mean()
                             if n_v else np.nan)
        # 2D pre-gate (cheap): acryl + planar_2d <= 30
        p2d_gate = cd["acryl_largest"].astype(bool) & \
                     (cd["planar_dihedral_deg"] <= 30.0)
        p2d_gate_arr = p2d_gate[valid]
        rec = {
            "cell": cell,
            "cfg_scale": float(cd.iloc[0]["cfg_scale"]),
            "retrieval": int(cd.iloc[0]["retrieval"]),
            "n_total": n, "n_valid": n_v,
            "valid_frac": n_v / max(1, n),
            "acryl_largest_frac": acryl_frac,
            "planar2d_pass_frac": float(p2d_gate_arr.mean()) if n_v else np.nan,
        }
        # Cofold subset.
        cof_ready = np.nan; cof_theta = np.nan; fukui_mean = np.nan
        vina_med = np.nan; n_cofold = 0; n_bd_ready = 0
        if pco is not None:
            ccof = pco[pco["cell"] == cell]
            n_cofold = int(len(ccof))
            if n_cofold > 0:
                bd = ccof["bd_ready"].astype(bool)
                n_bd_ready = int(bd.sum())
                cof_ready = float(bd.mean())
                cof_theta = float(ccof["bd_theta"].astype(float).median()
                                     if bd.any() else np.nan)
                fukui_vals = ccof.loc[bd,
                    "fukui_fplus_real"].astype(float).dropna()
                fukui_mean = float(fukui_vals.mean()) if len(fukui_vals) else np.nan
                vina_vals = ccof.loc[bd,
                    "vina_cov_score"].astype(float).dropna()
                vina_med = float(vina_vals.median()) if len(vina_vals) else np.nan
        rec.update({
            "n_cofolded": n_cofold,
            "cofold_bd_ready_frac": cof_ready,
            "n_bd_ready_cofold": n_bd_ready,
            "cofold_bd_theta_median": cof_theta,
            "cofold_fukui_fplus_mean": fukui_mean,
            "cofold_vina_cov_score_median": vina_med,
        })
        summary_rows.append(rec)

    summary = pd.DataFrame(summary_rows).sort_values(
        ["retrieval", "cfg_scale"]).reset_index(drop=True)
    summary.to_csv(args.out_summary_csv, index=False)
    print(f"\nSummary:\n{summary.to_string(index=False)}", flush=True)

    # Effect sizes vs baseline (cofold-derived).
    effects = []
    if pco is not None and (pco["cell"] == args.baseline_cell).any():
        base_cd = pco[pco["cell"] == args.baseline_cell]
        base_bd = base_cd["bd_ready"].astype(bool)
        base_bd_arr = base_bd.astype(float).values
        base_fukui = base_cd.loc[base_bd,
            "fukui_fplus_real"].astype(float).dropna().values
        base_vina = base_cd.loc[base_bd,
            "vina_cov_score"].astype(float).dropna().values
        base_theta = base_cd["bd_theta"].astype(float).dropna().values
        for cell, cd in pco.groupby("cell"):
            if cell == args.baseline_cell:
                continue
            bd = cd["bd_ready"].astype(bool)
            bd_arr = bd.astype(float).values
            fukui = cd.loc[bd,
                "fukui_fplus_real"].astype(float).dropna().values
            vina = cd.loc[bd,
                "vina_cov_score"].astype(float).dropna().values
            theta = cd["bd_theta"].astype(float).dropna().values
            row = {"cell": cell,
                   "n_cofold_a": int(len(cd)),
                   "n_cofold_b": int(len(base_cd))}
            p, lo, hi = bootstrap_diff_ci(bd_arr, base_bd_arr)
            row["d_bd_ready_frac"] = p
            row["d_bd_ready_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            p, lo, hi = bootstrap_diff_ci(fukui, base_fukui)
            row["d_fukui_fplus"] = p
            row["d_fukui_ci95"] = f"[{lo:+.4f}, {hi:+.4f}]"
            p, lo, hi = bootstrap_diff_ci(vina, base_vina, agg="median")
            row["d_vina_median"] = p
            row["d_vina_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            p, lo, hi = bootstrap_diff_ci(theta, base_theta, agg="median")
            row["d_theta_median"] = p
            row["d_theta_ci95"] = f"[{lo:+.2f}, {hi:+.2f}]"
            effects.append(row)
    effects_df = pd.DataFrame(effects)

    # Kill criteria.
    warnings_list = []
    for _, r in summary.iterrows():
        if r["valid_frac"] < 0.30:
            warnings_list.append(
                f"KILL: validity {r['valid_frac']:.3f} < 0.30 for cell "
                f"{r['cell']}")

    # Write markdown.
    lines = [
        "# CFG + Retrieval-Prefix Ablation Report (v2 — real cofold panel)",
        "",
        f"2D panel: `{args.panel_2d_csv}`  ({len(p2d)} rows)  ",
        f"Cofold panel: `{args.panel_cofold_csv}`  "
        f"({len(pco) if pco is not None else 0} rows)  ",
        f"Baseline cell: `{args.baseline_cell}`  ",
        "",
        "## Primary endpoint",
        "",
        "Mean **xTB Fukui f+** at the acrylamide β-C, computed on each "
        "sample's ETKDG-embedded conformer with `xtb --gfn 2 --vfukui`.  "
        "Bootstrap 95% CI on the difference vs baseline (5000 resamples).",
        "",
        "## Secondary endpoints",
        "",
        "- **Cofold BD-ready fraction**: Boltz-2 cofold + real geometry gate: "
        "`d(SG–Cβ) ∈ [2.5, 5.5] Å` AND `θ(SG→Cβ→Cα) ∈ [80°, 130°]` measured "
        "from the covalent-restrained cofold pose.",
        "- **Vina covalent-tethered rescore median** (kcal/mol; lower = better)",
        "- **Validity fraction** (kill criterion: <30% at any cell)",
        "",
        "## Per-cell summary",
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
                    "None triggered (all cells >= 30% valid).", ""]
    lines += [
        "## Method notes",
        "",
        "- Training: warm-started from `models/m1a_v2.ckpt` (covFT + v2 "
        "pocket/pose conditioning).  CFG dropout `p=0.10` applied jointly "
        "to [POCKET]+[POSE], replaced with a single learned null token.  "
        "Base mol2mol decoder unfrozen with LR `1e-5` (5× lower than the "
        "new modules' `5e-5`) so covFT knowledge is preserved while the "
        "decoder learns to accept the null token.  15 epochs.",
        "- Sampling: CFG combines logits per token: "
        "`logit_final = logit_uncond + s * (logit_cond - logit_uncond)`.  "
        "8 cells: `s ∈ {1.0, 1.5, 2.0, 3.0}` × `retrieval ∈ {off, on}`.",
        "- Retrieval prefix: top-5 CovInDB SMILES ranked by cosine of "
        "mean-pooled ESM residue embeddings to ZAP70, with all "
        "`source == 'boltz_zap70'` rows (the ZAP70 curriculum) and canonical "
        "ZAP70 PDB prefixes excluded.",
        "- Boltz-2 cofold: 3 recycling steps, 200 sampling steps, MSA reused "
        "from the Mol1 cofold cache (skips MSA search for every sample).",
        "- xTB Fukui f+: GFN2-xTB single-point with `--vfukui` on the "
        "ETKDG-embedded lowest-energy conformer of the ligand.  We report "
        "the f+ value at the acrylamide β-C (SMARTS `[CH2;X3]=[CH;X3][C;X3]"
        "(=O)[N]` match 0).",
        "- Vina cov score: `experiments/run_covalent_docking.py` "
        "AD-CovDock: meeko CovalentBuilder tethers β-C to Cys346 Cβ, then "
        "Vina scores against Cys346-stripped ZAP70 (PDB 4K2R).",
    ]
    Path(args.out_md).write_text("\n".join(lines))
    print(f"\nWrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
