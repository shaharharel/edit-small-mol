"""v4 per-layer FiLM report — reuses v3 panel scripts (xtb, vina_cov).

2x2 cell matrix: {film ∈ {off, on}} x {retrieval ∈ {off, on}}
Baseline: `film_off_retrieval_off` (FiLM disabled at sampling, no retrieval).

Reads:
  data/paper_pair_training/cfg_retrieval_v4_film/
    - covalent_metric_panel_v4.csv (2D, from covalent_metric_panel.py)
    - covalent_xtb_panel_v4.csv    (REAL xTB Fukui)
    - covalent_vina_cov_panel_v4.csv (REAL AD-CovDock score_only)
    - covalent_vina_local_panel_v4.csv (top-25/cell --local_only)

Writes:
  data/paper_pair_training/cfg_retrieval_v4_film/cfg_retrieval_report_v4_film.md
  data/paper_pair_training/cfg_retrieval_v4_film/cfg_retrieval_report_v4_film_summary.csv
  data/paper_pair_training/cfg_retrieval_v4_film/cfg_retrieval_report_v4_film_effects.csv

Primary comparisons vs baseline (film_off_retrieval_off):
  - film_on_retrieval_off - film_off_retrieval_off      (FiLM effect, no retrieval)
  - film_off_retrieval_on - film_off_retrieval_off      (retrieval alone)
  - film_on_retrieval_on  - film_off_retrieval_off      (both, joint effect)

Bootstrap 95% CI on:
  - valid_frac
  - Fukui f+ mean
  - vina_cov_score median (score_only, tethered pose)

Verdict guide (from coordinator):
  - "FIRST REAL ARCHITECTURAL WIN" iff FiLM shifts Fukui f+ OR vina_cov_score
    with CI excluding zero AND magnitude significant (Fukui>=0.0015, Vina>=5.0).
  - Else CFG conditioning approach is dead for mol2mol.
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


def df_to_md(df, floatfmt=".4f"):
    if df is None or len(df) == 0:
        return "*(empty)*"
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
    if not len(a) or not len(b): return point, np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs[i] = f(a[ia]) - f(b[ib])
    return point, float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def crosses_zero(ci_str):
    if not isinstance(ci_str, str): return True
    try:
        a, b = ci_str.strip("[]").split(",")
        return float(a) <= 0 <= float(b)
    except Exception:
        return True


def main():
    ROOT = Path("/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/cfg_retrieval_v4_film")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(ROOT / "cfg_retrieval_report_v4_film.md"))
    ap.add_argument("--out_summary_csv", default=str(ROOT / "cfg_retrieval_report_v4_film_summary.csv"))
    ap.add_argument("--out_effects_csv", default=str(ROOT / "cfg_retrieval_report_v4_film_effects.csv"))
    ap.add_argument("--baseline_cell", default="film_off_retrieval_off")
    args = ap.parse_args()

    p2d = pd.read_csv(ROOT / "covalent_metric_panel_v4.csv")
    p_xtb = pd.read_csv(ROOT / "covalent_xtb_panel_v4.csv") if (ROOT / "covalent_xtb_panel_v4.csv").exists() else pd.DataFrame()
    p_cov = pd.read_csv(ROOT / "covalent_vina_cov_panel_v4.csv") if (ROOT / "covalent_vina_cov_panel_v4.csv").exists() else pd.DataFrame()
    p_loc = pd.read_csv(ROOT / "covalent_vina_local_panel_v4.csv") if (ROOT / "covalent_vina_local_panel_v4.csv").exists() else pd.DataFrame()

    print(f"2D:{len(p2d)} xtb:{len(p_xtb)} cov:{len(p_cov)} local:{len(p_loc)}",
           flush=True)

    m = p2d.copy()
    if len(p_xtb):
        m = m.merge(p_xtb[["cell", "sample_idx", "fukui_fplus", "xtb_ok"]],
                    on=["cell", "sample_idx"], how="left")
    if len(p_cov):
        m = m.merge(p_cov[["cell", "sample_idx", "vina_cov_score",
                              "vina_cov_d_sg", "vina_cov_bd_angle",
                              "vina_cov_ok"]],
                    on=["cell", "sample_idx"], how="left")

    # Per-cell summary.
    rows = []
    for cell, cd in m.groupby("cell"):
        s0 = cd.iloc[0]
        n = len(cd); valid = cd["valid"].astype(bool); n_v = int(valid.sum())
        acryl_frac = float(cd.loc[valid, "acryl_largest"].mean() if n_v else np.nan)
        n_uniq = int(cd.loc[valid, "canonical_SMILES"].nunique()) if "canonical_SMILES" in cd.columns else 0
        novelty = n_uniq / max(1, n_v)
        fukui_mean = np.nan; n_xtb = 0
        if "fukui_fplus" in cd.columns:
            xtb_vals = cd.loc[valid & cd["acryl_largest"].astype(bool),
                              "fukui_fplus"].astype(float).dropna()
            fukui_mean = float(xtb_vals.mean()) if len(xtb_vals) else np.nan
            n_xtb = int(len(xtb_vals))
        cov_score_med = np.nan; n_cov = 0
        if "vina_cov_ok" in cd.columns:
            mask_c = valid & cd["acryl_largest"].astype(bool) & \
                       cd["vina_cov_ok"].fillna(False).astype(bool)
            n_cov = int(mask_c.sum())
            if n_cov:
                cov = cd.loc[mask_c, "vina_cov_score"].astype(float)
                cov_score_med = float(cov.median())
        # FiLM enabled / retrieval flag decoded from cell name.
        rows.append({
            "cell": cell,
            "film_enabled": int("film_on" in cell),
            "retrieval": int("retrieval_on" in cell),
            "n_total": n, "n_valid": n_v, "valid_frac": n_v / max(1, n),
            "novelty_frac": novelty,
            "acryl_largest_frac": acryl_frac,
            "n_xtb": n_xtb, "mean_fukui_fplus": fukui_mean,
            "n_vina_cov": n_cov,
            "vina_cov_score_median": cov_score_med,
        })
    summary = pd.DataFrame(rows).sort_values(
        ["film_enabled", "retrieval"]).reset_index(drop=True)
    summary.to_csv(args.out_summary_csv, index=False)

    # Effects vs baseline.
    if args.baseline_cell not in m["cell"].unique():
        print(f"[warn] baseline cell {args.baseline_cell} not present", flush=True)
        effects_df = pd.DataFrame()
    else:
        base = m[m["cell"] == args.baseline_cell]
        b_valid = base["valid"].astype(bool)
        b_v_arr = b_valid.astype(float).values
        b_acryl = base["acryl_largest"].astype(bool)
        b_fukui = base.loc[b_valid & b_acryl,
            "fukui_fplus"].astype(float).dropna().values \
                if "fukui_fplus" in base.columns else np.array([])
        b_cov_mask = b_valid & b_acryl & \
                        base.get("vina_cov_ok",
                                    pd.Series(False, index=base.index)
                                ).fillna(False).astype(bool)
        b_cov = base.loc[b_cov_mask,
            "vina_cov_score"].astype(float).dropna().values \
                if "vina_cov_score" in base.columns else np.array([])

        effects_rows = []
        for cell, cd in m.groupby("cell"):
            if cell == args.baseline_cell:
                continue
            v = cd["valid"].astype(bool)
            a = cd["acryl_largest"].astype(bool)
            row = {"cell": cell}
            # validity
            p, lo, hi = bootstrap_diff_ci(v.astype(float).values, b_v_arr)
            row["d_valid_frac"] = p
            row["d_valid_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            # Fukui
            if "fukui_fplus" in cd.columns:
                a_f = cd.loc[v & a, "fukui_fplus"].astype(float).dropna().values
                p, lo, hi = bootstrap_diff_ci(a_f, b_fukui)
                row["d_fukui_fplus"] = p
                row["d_fukui_ci95"] = f"[{lo:+.5f}, {hi:+.5f}]"
                row["n_a_fukui"] = int(len(a_f))
                row["n_b_fukui"] = int(len(b_fukui))
            # Vina cov score
            if "vina_cov_ok" in cd.columns:
                cov_mask = v & a & cd["vina_cov_ok"].fillna(False).astype(bool)
                a_cov = cd.loc[cov_mask, "vina_cov_score"].astype(float).dropna().values
                p, lo, hi = bootstrap_diff_ci(a_cov, b_cov, agg="median")
                row["d_vina_cov_score_median"] = p
                row["d_vina_cov_score_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
                row["n_a_cov"] = int(len(a_cov))
                row["n_b_cov"] = int(len(b_cov))
            effects_rows.append(row)
        effects_df = pd.DataFrame(effects_rows)
    effects_df.to_csv(args.out_effects_csv, index=False)

    # Verdict: architectural win if FiLM shifts Fukui or Vina at 95% CI.
    hits = []; harmful = []
    for _, r in effects_df.iterrows():
        v = r.get("d_fukui_fplus", 0)
        if not crosses_zero(r.get("d_fukui_ci95", "")) and abs(v) >= 0.0015:
            msg = (f"{r['cell']}: Fukui f+ shift = {v:+.5f} {r['d_fukui_ci95']}")
            (hits if v > 0 else harmful).append(msg)
        v = r.get("d_vina_cov_score_median", 0)
        if not crosses_zero(r.get("d_vina_cov_score_ci95", "")) and abs(v) >= 5.0:
            msg = (f"{r['cell']}: Vina cov score shift = "
                    f"{v:+.3f} {r['d_vina_cov_score_ci95']}")
            (hits if v < 0 else harmful).append(msg)
        v = r.get("d_valid_frac", 0)
        if not crosses_zero(r.get("d_valid_ci95", "")):
            msg = (f"{r['cell']}: valid_frac shift = "
                    f"{v:+.3f} {r['d_valid_ci95']}")
            (hits if v >= 0.05 else harmful).append(msg)

    if hits:
        verdict = "FIRST REAL ARCHITECTURAL WIN — FiLM steers at least one primary metric."
    elif harmful:
        verdict = ("FiLM ACTIVE BUT HARMFUL — CI-significant shifts are in the "
                    "WRONG direction (validity down / Vina worse).")
    else:
        verdict = ("FiLM DEAD FOR THIS TARGET — no metric moves beyond bootstrap "
                    "CI at practical magnitudes.  Conditioning-based mol2mol "
                    "approach appears fundamentally dead.")

    # Top-25 local_only table.
    loc_section = []
    if len(p_loc):
        p_loc_ok = p_loc[p_loc["vina_cov_ok"] == True].copy()
        loc_rows = []
        for cell, cd in p_loc_ok.groupby("cell"):
            vs = cd["vina_cov_score"].astype(float)
            loc_rows.append({
                "cell": cell, "n_top_local": int(len(cd)),
                "median_kcalmol": float(vs.median()),
                "best_kcalmol": float(vs.min()),
                "frac_le_neg5": float((vs <= -5.0).mean()),
                "frac_negative": float((vs < 0).mean()),
            })
        loc_df = pd.DataFrame(loc_rows).sort_values("cell")
        best_i = p_loc_ok["vina_cov_score"].astype(float).idxmin()
        loc_section = [
            "",
            "## Top-25/cell absolute-usable Vina scores (--local_only)",
            "",
            df_to_md(loc_df),
            "",
            f"**Best absolute affinity across the 4-cell top-25 cohort**: "
            f"{p_loc_ok.loc[best_i, 'vina_cov_score']:.2f} kcal/mol "
            f"(cell `{p_loc_ok.loc[best_i, 'cell']}`).  Overall median: "
            f"{p_loc_ok['vina_cov_score'].astype(float).median():.2f} kcal/mol.  "
            f"{int((p_loc_ok['vina_cov_score'].astype(float) < 0).sum())}/"
            f"{len(p_loc_ok)} of the top-25 cohort have negative "
            f"vina_cov_score after local minimization.",
            "",
        ]

    lines = [
        "# v4 Per-Layer FiLM Ablation Report",
        "",
        "**Architecture**: v3 CFG-token approach relied only on encoder-memory "
        "cross-attention, which QA #4 confirmed is too weak a coupling for CFG "
        "guidance to steer generation.  v4 keeps that pathway (belt-and-"
        "suspenders) AND adds per-layer FiLM at the output of every decoder "
        "block, forcing conditioning to modulate every layer.  FiLM gamma/beta "
        "are zero-init so training starts from the exact v2 checkpoint.",
        "",
        "**2×2 cell matrix**: {FiLM ∈ {off, on}} × {retrieval ∈ {off, on}} × N=200",
        "",
        "**Baseline**: `film_off_retrieval_off` (pure v2/covFT via zero-FiLM path).",
        "",
        "**Verdict**:",
        "",
        f"> **{verdict}**",
        "",
    ]
    if hits:
        lines += ["Beneficial hits:", ""]
        for h in hits:
            lines += [f"- {h}"]
        lines += [""]
    if harmful:
        lines += ["Harmful CI-significant shifts (transparency):", ""]
        for h in harmful:
            lines += [f"- {h}"]
        lines += [""]
    lines += [
        "## Per-cell summary",
        "",
        df_to_md(summary),
        "",
        "## Effect vs baseline (bootstrap 95% CI, 5000 resamples)",
        "",
        df_to_md(effects_df),
        "",
    ]
    lines += loc_section
    lines += [
        "## Method notes",
        "",
        "- **Backbone**: covFT prior + v2 pocket/pose (warm-start `m1a_v2.ckpt`).  "
        "Per-layer FiLM added at every decoder block (6 layers × [gamma, beta] "
        "MLPs, each 256→256→d_model).  Gamma/beta output projections "
        "**zero-initialised** so at step 0 the model is identical to v2.",
        "- **Two-LR schedule**: new modules (pocket_enc, pose_enc, condition_pooler, "
        "film_layers, null_emb) at LR=5e-5; base mol2mol encoder+decoder at "
        "LR=1e-5 (5× lower to preserve covFT).  15 epochs, cosine warmup 500 "
        "steps.  No CFG dropout (FiLM doesn't need it).",
        "- **Sampling**: greedy multinomial with per-token softmax.  FiLM=off "
        "cell uses the same trained checkpoint but toggles `film_enabled=False` "
        "at inference — this recovers the pure encoder-memory-token pathway "
        "(matches v3 baseline mathematically).",
        "- **Retrieval prefix**: top-5 CovInDB SMILES ranked by cosine of "
        "mean-pooled ESM residue embeddings to ZAP70 (LOO 'boltz_zap70' + "
        "canonical ZAP70 PDBs).  Encoded through the mol2mol encoder + "
        "mean-pooled to D=256 tokens, prepended to decoder memory.",
        "- **xTB Fukui f+**: `xtb --gfn 2 --vfukui` single-point on ETKDG+MMFF "
        "conformer.  Value at acrylamide β-C.",
        "- **Vina cov-dock (--score_only)**: meeko CovalentBuilder tethers β-C "
        "to Cys346 CB; Vina scores tethered pose.  Reused from v3 pipeline.  "
        "AD-CovDock geometry constant BD angle ~119° (structural artefact of "
        "meeko tether); BD gate not discriminating per QA #6, so only "
        "vina_cov_score effect is reported.",
        "- **Vina --local_only** on top-25/cell (Newton min around tether) "
        "for absolute-usable affinities.",
    ]
    Path(args.out_md).write_text("\n".join(lines))
    print(f"Wrote {args.out_md}", flush=True)
    print(f"Wrote {args.out_summary_csv}", flush=True)
    print(f"Wrote {args.out_effects_csv}", flush=True)


if __name__ == "__main__":
    main()
