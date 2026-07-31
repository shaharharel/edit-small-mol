"""v6 DPO report.  Reuses v3/v4 report structure; adapted for 2x2 DPO grid."""
from __future__ import annotations
import argparse
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
    except Exception: return True


def main():
    ROOT = Path("/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/cfg_retrieval_v6_dpo")
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_md", default=str(ROOT / "cfg_retrieval_report_v6_dpo.md"))
    ap.add_argument("--out_summary_csv", default=str(ROOT / "cfg_retrieval_report_v6_dpo_summary.csv"))
    ap.add_argument("--out_effects_csv", default=str(ROOT / "cfg_retrieval_report_v6_dpo_effects.csv"))
    ap.add_argument("--baseline_cell", default="dpo_off_retrieval_off")
    ap.add_argument("--panel_2d", default=str(ROOT / "covalent_metric_panel_v6.csv"))
    ap.add_argument("--panel_xtb", default=str(ROOT / "covalent_xtb_panel_v6.csv"))
    ap.add_argument("--panel_cov", default=str(ROOT / "covalent_vina_cov_panel_v6.csv"))
    ap.add_argument("--panel_local", default=str(ROOT / "covalent_vina_local_panel_v6.csv"))
    args = ap.parse_args()

    p2d = pd.read_csv(args.panel_2d)
    p_xtb = pd.read_csv(args.panel_xtb) if Path(args.panel_xtb).exists() else pd.DataFrame()
    p_cov = pd.read_csv(args.panel_cov) if Path(args.panel_cov).exists() else pd.DataFrame()
    p_loc = pd.read_csv(args.panel_local) if Path(args.panel_local).exists() else pd.DataFrame()
    print(f"2D:{len(p2d)} xtb:{len(p_xtb)} cov:{len(p_cov)} local:{len(p_loc)}", flush=True)

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
        n = len(cd); valid = cd["valid"].astype(bool); n_v = int(valid.sum())
        acryl_frac = float(cd.loc[valid, "acryl_largest"].mean() if n_v else np.nan)
        n_uniq = int(cd.loc[valid, "canonical_SMILES"].nunique()) if "canonical_SMILES" in cd.columns else 0
        novelty = n_uniq / max(1, n_v)
        fukui_mean = np.nan; n_xtb = 0
        if "fukui_fplus" in cd.columns:
            xt = cd.loc[valid & cd["acryl_largest"].astype(bool),
                          "fukui_fplus"].astype(float).dropna()
            fukui_mean = float(xt.mean()) if len(xt) else np.nan
            n_xtb = int(len(xt))
        cov_med = np.nan; n_cov = 0
        if "vina_cov_ok" in cd.columns:
            cm = valid & cd["acryl_largest"].astype(bool) & \
                   cd["vina_cov_ok"].fillna(False).astype(bool)
            n_cov = int(cm.sum())
            if n_cov:
                cov_med = float(cd.loc[cm, "vina_cov_score"].astype(float).median())
        rows.append({
            "cell": cell,
            "dpo": int("dpo_on" in cell),
            "retrieval": int("retrieval_on" in cell),
            "n_total": n, "n_valid": n_v, "valid_frac": n_v / max(1, n),
            "novelty_frac": novelty, "acryl_largest_frac": acryl_frac,
            "n_xtb": n_xtb, "mean_fukui_fplus": fukui_mean,
            "n_vina_cov": n_cov, "vina_cov_score_median": cov_med,
        })
    summary = pd.DataFrame(rows).sort_values(
        ["dpo", "retrieval"]).reset_index(drop=True)
    summary.to_csv(args.out_summary_csv, index=False)

    # Effects vs baseline.
    baseline_cell = args.baseline_cell
    if baseline_cell not in m["cell"].unique():
        effects_df = pd.DataFrame()
    else:
        base = m[m["cell"] == baseline_cell]
        b_valid = base["valid"].astype(bool)
        b_v_arr = b_valid.astype(float).values
        b_acryl = base["acryl_largest"].astype(bool)
        b_fukui = base.loc[b_valid & b_acryl,
            "fukui_fplus"].astype(float).dropna().values \
                if "fukui_fplus" in base.columns else np.array([])
        b_cov_mask = b_valid & b_acryl & \
            base.get("vina_cov_ok",
                       pd.Series(False, index=base.index)
                     ).fillna(False).astype(bool) \
                if "vina_cov_ok" in base.columns else \
                pd.Series(False, index=base.index)
        b_cov = base.loc[b_cov_mask,
            "vina_cov_score"].astype(float).dropna().values \
                if "vina_cov_score" in base.columns else np.array([])
        rows_e = []
        for cell, cd in m.groupby("cell"):
            if cell == baseline_cell:
                continue
            v = cd["valid"].astype(bool); a = cd["acryl_largest"].astype(bool)
            row = {"cell": cell}
            p, lo, hi = bootstrap_diff_ci(v.astype(float).values, b_v_arr)
            row["d_valid_frac"] = p
            row["d_valid_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
            if "fukui_fplus" in cd.columns:
                a_f = cd.loc[v & a, "fukui_fplus"].astype(float).dropna().values
                p, lo, hi = bootstrap_diff_ci(a_f, b_fukui)
                row["d_fukui_fplus"] = p
                row["d_fukui_ci95"] = f"[{lo:+.5f}, {hi:+.5f}]"
                row["n_a_fukui"] = int(len(a_f))
                row["n_b_fukui"] = int(len(b_fukui))
            if "vina_cov_ok" in cd.columns:
                cov_mask = v & a & cd["vina_cov_ok"].fillna(False).astype(bool)
                a_cov = cd.loc[cov_mask,
                    "vina_cov_score"].astype(float).dropna().values
                p, lo, hi = bootstrap_diff_ci(a_cov, b_cov, agg="median")
                row["d_vina_cov_score_median"] = p
                row["d_vina_cov_score_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
                row["n_a_cov"] = int(len(a_cov))
                row["n_b_cov"] = int(len(b_cov))
            rows_e.append(row)
        effects_df = pd.DataFrame(rows_e)
    effects_df.to_csv(args.out_effects_csv, index=False)

    # Verdict.
    hits = []; harmful = []
    for _, r in effects_df.iterrows():
        v = r.get("d_fukui_fplus", 0)
        if not crosses_zero(r.get("d_fukui_ci95", "")) and abs(v) >= 0.0015:
            (hits if v > 0 else harmful).append(
                f"{r['cell']}: Fukui f+ shift = {v:+.5f} {r['d_fukui_ci95']}")
        v = r.get("d_vina_cov_score_median", 0)
        # v6 headline: 3 kcal/mol threshold (looser than v3-v5's 5).
        if not crosses_zero(r.get("d_vina_cov_score_ci95", "")) and abs(v) >= 3.0:
            (hits if v < 0 else harmful).append(
                f"{r['cell']}: Vina cov score shift = {v:+.3f} "
                f"{r['d_vina_cov_score_ci95']}")
        v = r.get("d_valid_frac", 0)
        if not crosses_zero(r.get("d_valid_ci95", "")):
            (hits if v >= 0.05 else harmful).append(
                f"{r['cell']}: valid_frac shift = {v:+.3f} {r['d_valid_ci95']}")

    if hits:
        verdict = "DPO STEERS SAMPLING — at least one primary metric moves."
    elif harmful:
        verdict = "DPO ACTIVE BUT HARMFUL — significant shifts wrong direction."
    else:
        verdict = "DPO NULL — no metric CI-excludes zero."

    # Top-25 local_only.
    loc_section = []
    if len(p_loc):
        ok = p_loc[p_loc["vina_cov_ok"] == True].copy()
        loc_rows = []
        for cell, cd in ok.groupby("cell"):
            vs = cd["vina_cov_score"].astype(float)
            loc_rows.append({
                "cell": cell, "n_top_local": int(len(cd)),
                "median_kcalmol": float(vs.median()),
                "best_kcalmol": float(vs.min()),
                "frac_le_neg5": float((vs <= -5.0).mean()),
                "frac_negative": float((vs < 0).mean()),
            })
        loc_df = pd.DataFrame(loc_rows).sort_values("cell")
        best_i = ok["vina_cov_score"].astype(float).idxmin()
        loc_section = ["",
                          "## Top-25/cell absolute-usable Vina scores (--local_only)",
                          "",
                          df_to_md(loc_df),
                          "",
                          f"**Best absolute affinity**: "
                          f"{ok.loc[best_i, 'vina_cov_score']:.2f} kcal/mol "
                          f"(cell `{ok.loc[best_i, 'cell']}`).  "
                          f"Overall median: "
                          f"{ok['vina_cov_score'].astype(float).median():.2f} "
                          f"kcal/mol.  "
                          f"{int((ok['vina_cov_score'].astype(float) < 0).sum())}"
                          f"/{len(ok)} negative.", ""]

    lines = [
        "# v6 DPO on Vina Cov Score Report",
        "",
        "**Motivation**: v3 (CFG-token) and v4 (per-layer FiLM) both encoded "
        "pocket/pose signal in training loss but the decoder ignored it at "
        "sampling.  DPO trains DIRECTLY on sampled behavior via a preference "
        "loss, bypassing the scoring-vs-sampling gap.",
        "",
        f"**Baseline**: `{baseline_cell}` (covFT+v2 with FiLM disabled, no retrieval).",
        "",
        "**Verdict**:",
        "",
        f"> **{verdict}**",
        "",
    ]
    if hits:
        lines += ["Beneficial hits:", ""]
        for h in hits: lines += [f"- {h}"]
        lines += [""]
    if harmful:
        lines += ["Harmful CI-significant shifts:", ""]
        for h in harmful: lines += [f"- {h}"]
        lines += [""]
    lines += ["## Per-cell summary", "", df_to_md(summary), "",
                "## Effect vs baseline (bootstrap 95% CI, 5000 resamples)", "",
                df_to_md(effects_df), ""]
    lines += loc_section
    lines += [
        "## Method notes",
        "",
        "- **DPO training**: β=0.1, LR (base=1e-6, new=5e-6), 3 epochs.  "
        "Reference = frozen covFT+v2, policy = trainable copy.  Preference "
        "pairs built from v3+v4 samples with real cov-Vina scores:",
        "  * winner = top-quartile Vina cov score (lower kcal/mol) per source cell",
        "  * loser  = bottom-quartile",
        "  * cap 25 winners × 25 losers per source cell",
        "  * ~1500-2000 pairs total across v3 p0.10/p0.05 + v4",
        "- **Sampling**: 2×2 grid {DPO ∈ {off, on}} × {retrieval ∈ {off, on}} × N=200.  "
        "DPO=off uses `m1a_v2.ckpt` (baseline covFT+v2).  DPO=on uses "
        "`dpo_final.ckpt` (weights shifted by DPO).",
        "- **Panels**: reuse v3-v5 pipelines (xTB Fukui + cov-Vina --score_only + top-25 --local_only).",
    ]
    Path(args.out_md).write_text("\n".join(lines))
    print(f"Wrote {args.out_md}", flush=True)
    print(f"Wrote {args.out_summary_csv}", flush=True)
    print(f"Wrote {args.out_effects_csv}", flush=True)


if __name__ == "__main__":
    main()
