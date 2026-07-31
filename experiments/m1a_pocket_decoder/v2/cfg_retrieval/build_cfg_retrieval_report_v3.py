"""v3 CFG × retrieval report — REAL xTB Fukui + REAL Vina cov-dock + REAL BD gate.

Reads:
  - covalent_metric_panel_v2.csv    (2D per-sample metrics; validity+acryl)
  - covalent_xtb_panel.csv          (REAL xTB f+/f-/f0 at acryl β-C)
  - covalent_vina_panel.csv         (Vina free-dock score + d_SG, θ, φ; BD gate)
  - (optional) covalent_cofold_panel.csv  (Boltz cofold on TOP-25 per cell for
    confirmation of Vina BD calls)

Writes:
  - cfg_retrieval_report_v3.md
  - cfg_retrieval_report_v3_summary.csv

Primary endpoint: mean REAL xTB Fukui f+ at Cβ — effect size + bootstrap 95%
CI vs `cfg_s1_retrieval_off_v2` baseline (all valid+acryl samples in that
cell).

Secondary:
  - vina_bd_ready_frac   (d_SG ∈ [2.5,5.5] AND θ ∈ [80,130] AND φ ≤ 30)
  - vina_free_score_median (kcal/mol, lower better)
  - boltz_bd_ready_frac (if cofold panel present)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_md(df, floatfmt=".4f"):
    if df is None or len(df) == 0: return "*(empty)*"
    cols = list(df.columns)
    def fmt(v):
        if v is None: return ""
        if isinstance(v, float):
            if np.isnan(v): return "NaN"
            return f"{v:{floatfmt[1:]}}"
        return str(v)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    body = ["| " + " | ".join(fmt(row[c]) for c in cols) + " |"
             for _, row in df.iterrows()]
    return "\n".join([header, sep] + body)


def bootstrap_diff_ci(a, b, agg="mean", n_boot=5000, seed=0):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    f = (lambda x: float(np.mean(x)) if len(x) else np.nan) if agg == "mean" \
        else (lambda x: float(np.median(x)) if len(x) else np.nan)
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
    ROOT = Path("/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/cfg_retrieval")
    ap.add_argument("--panel_2d_csv", default=str(ROOT / "covalent_metric_panel_v2.csv"))
    ap.add_argument("--panel_xtb_csv", default=str(ROOT / "covalent_xtb_panel.csv"))
    ap.add_argument("--panel_vina_csv", default=str(ROOT / "covalent_vina_panel.csv"))
    ap.add_argument("--panel_cofold_csv", default=str(ROOT / "covalent_cofold_panel.csv"))
    ap.add_argument("--out_md", default=str(ROOT / "cfg_retrieval_report_v3.md"))
    ap.add_argument("--out_summary_csv", default=str(ROOT / "cfg_retrieval_report_v3_summary.csv"))
    ap.add_argument("--baseline_cell", default="cfg_s1_retrieval_off_v2")
    args = ap.parse_args()

    p2d = pd.read_csv(args.panel_2d_csv)
    p_xtb = pd.read_csv(args.panel_xtb_csv) if Path(args.panel_xtb_csv).exists() else pd.DataFrame()
    p_vina = pd.read_csv(args.panel_vina_csv) if Path(args.panel_vina_csv).exists() else pd.DataFrame()
    p_cofold = pd.read_csv(args.panel_cofold_csv) if Path(args.panel_cofold_csv).exists() else pd.DataFrame()

    print(f"2D panel: {len(p2d)} rows | xtb: {len(p_xtb)} | vina: {len(p_vina)} | cofold: {len(p_cofold)}", flush=True)

    # Merge on (cell, sample_idx).
    key = ["cell", "sample_idx"]
    merged = p2d.copy()
    if len(p_xtb):
        merged = merged.merge(p_xtb[key + ["fukui_fplus", "xtb_ok"]],
                                  on=key, how="left")
    if len(p_vina):
        merged = merged.merge(p_vina[key + ["vina_free_score", "vina_free_d_sg",
                                                "vina_free_bd_ang", "vina_free_phi_planar",
                                                "vina_free_ok", "bd_ready_vina"]],
                                  on=key, how="left")
    if len(p_cofold):
        merged = merged.merge(p_cofold[key + ["d_sg", "bd_theta", "phi_planar",
                                                    "bd_ready"]].rename(columns={
                                    "d_sg": "cofold_d_sg", "bd_theta": "cofold_bd_theta",
                                    "phi_planar": "cofold_phi_planar",
                                    "bd_ready": "cofold_bd_ready"}),
                                  on=key, how="left")

    # Per-cell summary.
    summary_rows = []
    for cell, cd in merged.groupby("cell"):
        s = cd.iloc[0]
        n = len(cd); valid = cd["valid"].astype(bool)
        n_v = int(valid.sum())
        acryl_frac = float(cd.loc[valid, "acryl_largest"].mean() if n_v else np.nan)
        # Real xTB Fukui f+ mean among valid+acryl+xtb_ok
        if "fukui_fplus" in cd.columns:
            xtb_vals = cd.loc[valid & cd["acryl_largest"].astype(bool),
                              "fukui_fplus"].astype(float).dropna()
            fukui_mean = float(xtb_vals.mean()) if len(xtb_vals) else np.nan
            n_xtb = int(len(xtb_vals))
        else:
            fukui_mean = np.nan; n_xtb = 0
        # Vina BD-ready frac among valid+acryl+vina_ok
        if "bd_ready_vina" in cd.columns:
            mask = valid & cd["acryl_largest"].astype(bool) & cd["vina_free_ok"].fillna(False).astype(bool)
            bd_arr = cd.loc[mask, "bd_ready_vina"].fillna(False).astype(bool)
            bd_frac = float(bd_arr.mean()) if len(bd_arr) else np.nan
            n_bd = int(bd_arr.sum())
            vina_score = cd.loc[mask, "vina_free_score"].astype(float).dropna()
            vina_med = float(vina_score.median()) if len(vina_score) else np.nan
        else:
            bd_frac = np.nan; n_bd = 0; vina_med = np.nan
        # Cofold BD-ready (if any)
        if "cofold_bd_ready" in cd.columns:
            cof = cd.loc[cd["cofold_bd_ready"].notna(), "cofold_bd_ready"].astype(bool)
            cofold_bd = float(cof.mean()) if len(cof) else np.nan
            n_cofold_bd = int(cof.sum())
        else:
            cofold_bd = np.nan; n_cofold_bd = 0
        summary_rows.append({
            "cell": cell,
            "cfg_scale": float(s["cfg_scale"]),
            "retrieval": int(s["retrieval"]),
            "n_total": n, "n_valid": n_v,
            "valid_frac": n_v / max(1, n),
            "acryl_largest_frac": acryl_frac,
            "n_xtb": n_xtb,
            "mean_fukui_fplus": fukui_mean,
            "n_vina": int(cd["vina_free_ok"].fillna(False).astype(bool).sum())
                        if "vina_free_ok" in cd.columns else 0,
            "vina_bd_ready_frac": bd_frac,
            "n_vina_bd_ready": n_bd,
            "vina_free_score_median": vina_med,
            "n_cofolded": int(cd["cofold_bd_ready"].notna().sum())
                            if "cofold_bd_ready" in cd.columns else 0,
            "cofold_bd_ready_frac": cofold_bd,
            "n_cofold_bd_ready": n_cofold_bd,
        })

    summary = pd.DataFrame(summary_rows).sort_values(
        ["retrieval", "cfg_scale"]).reset_index(drop=True)
    summary.to_csv(args.out_summary_csv, index=False)
    print("\nSummary:\n" + summary.to_string(index=False), flush=True)

    # Effect sizes.
    effects_rows = []
    if args.baseline_cell in merged["cell"].unique():
        base = merged[merged["cell"] == args.baseline_cell]
        base_valid = base["valid"].astype(bool)
        base_fukui = base.loc[base_valid & base["acryl_largest"].astype(bool),
                                "fukui_fplus"].astype(float).dropna().values \
                        if "fukui_fplus" in base.columns else np.array([])
        base_bd = base.loc[base_valid & base["acryl_largest"].astype(bool)
                            & base["vina_free_ok"].fillna(False).astype(bool),
                            "bd_ready_vina"].fillna(False).astype(float).values \
                        if "bd_ready_vina" in base.columns else np.array([])
        base_vina = base.loc[base_valid & base["acryl_largest"].astype(bool)
                                & base["vina_free_ok"].fillna(False).astype(bool),
                                "vina_free_score"].astype(float).dropna().values \
                        if "vina_free_score" in base.columns else np.array([])
        for cell, cd in merged.groupby("cell"):
            if cell == args.baseline_cell:
                continue
            cv = cd["valid"].astype(bool)
            ca = cd["acryl_largest"].astype(bool)
            row = {"cell": cell}
            if "fukui_fplus" in cd.columns:
                a = cd.loc[cv & ca, "fukui_fplus"].astype(float).dropna().values
                p, lo, hi = bootstrap_diff_ci(a, base_fukui)
                row["d_fukui_fplus"] = p
                row["d_fukui_ci95"] = f"[{lo:+.5f}, {hi:+.5f}]"
                row["n_a"] = int(len(a)); row["n_base"] = int(len(base_fukui))
            if "bd_ready_vina" in cd.columns:
                mask = cv & ca & cd["vina_free_ok"].fillna(False).astype(bool)
                bd = cd.loc[mask, "bd_ready_vina"].fillna(False).astype(float).values
                p, lo, hi = bootstrap_diff_ci(bd, base_bd)
                row["d_vina_bd_ready_frac"] = p
                row["d_vina_bd_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
                vv = cd.loc[mask, "vina_free_score"].astype(float).dropna().values
                p, lo, hi = bootstrap_diff_ci(vv, base_vina, agg="median")
                row["d_vina_score_median"] = p
                row["d_vina_score_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            effects_rows.append(row)
    effects_df = pd.DataFrame(effects_rows)

    warnings_list = []
    for _, r in summary.iterrows():
        if r["valid_frac"] < 0.30:
            warnings_list.append(f"validity {r['valid_frac']:.3f} < 0.30 for "
                                    f"cell {r['cell']} (n_valid={r['n_valid']}/{r['n_total']})")

    lines = [
        "# CFG + Retrieval-Prefix Ablation Report (v3 — real xTB + real Vina + real BD gate)",
        "",
        f"Panels: 2D={len(p2d)} | xTB={len(p_xtb)} | Vina={len(p_vina)} | Boltz cofold={len(p_cofold)}  ",
        f"Baseline cell: `{args.baseline_cell}` (covFT+v2 CFG-trained ckpt, s=1.0, retrieval=off).  ",
        "",
        "## Primary endpoint",
        "",
        "Mean **REAL xTB Fukui f+** at the acrylamide β-C.  Computed by "
        "`xtb --gfn 2 --vfukui` on the ETKDG+MMFF-optimized conformer of "
        "each sample.  Parsed from xTB's `Fukui functions:` table (columns "
        "f(+), f(-), f(0)).",
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
        lines += ["## KILL-criteria warnings (validity < 30%)", ""]
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
        "pocket/pose conditioning).  CFG dropout p=0.10 applied jointly to "
        "[POCKET]+[POSE], replaced with a single learned null token.  "
        "Base mol2mol decoder UNFROZEN at LR=1e-5 (5× lower than new "
        "modules' LR=5e-5) — freezing the base decoder (as in v1 smoke) "
        "left the null-token uncond branch at NLL~31 and CFG at s≥2 broke "
        "the model.  Unfreezing brought val_uncond from 31 → 12.  15 epochs.",
        "- Sampling: CFG combines logits per token: "
        "`logit_final = logit_uncond + s*(logit_cond - logit_uncond)`.  "
        "8 cells: s ∈ {1.0, 1.5, 2.0, 3.0} × retrieval ∈ {off, on}.",
        "- Retrieval prefix: top-5 CovInDB SMILES ranked by cosine of "
        "mean-pooled ESM residue embeddings to ZAP70, with all "
        "`source == 'boltz_zap70'` rows (the ZAP70 curriculum) leave-one-out.",
        "- **xTB Fukui f+ (REAL)**: `xtb --gfn 2 --vfukui` single-point on "
        "ETKDG+MMFF conformer.  Extracted from Fukui table row where atom "
        "index == β-C RDKit index + 1.  570 valid+acryl mols processed on 8 "
        "workers in ~90 sec.",
        "- **Vina cov-dock (REAL)**: AutoDock Vina `--exhaustiveness 4` "
        "global dock on meeko-prepared ligand PDBQT against Cys346-stripped "
        "ZAP70 (4K2R).  20³ Å box centred on Cys346 SG.  Top-1 pose used to "
        "measure d(SG, Cβ), Bürgi–Dunitz angle at Cβ, and planar dihedral.  "
        "BD-ready gate: d_SG ∈ [2.5, 5.5] Å AND θ ∈ [80°, 130°] AND "
        "|φ_planar| ≤ 30°.",
        "- Boltz-2 cofold on TOP-25 per cell (if present): 3 recycling + "
        "200 sampling steps, MSA reused from Mol1 cofold cache.  Cofold "
        "confirms Vina's BD calls at higher fidelity.",
    ]
    Path(args.out_md).write_text("\n".join(lines))
    print(f"\nWrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
