"""v9 multi-target summary: aggregate per-target reports into one file."""
from __future__ import annotations
import argparse
from pathlib import Path

import numpy as np
import pandas as pd


def bootstrap_diff_ci(a, b, agg="median", n_boot=5000, seed=0):
    a = np.asarray(a, dtype=float); a = a[np.isfinite(a)]
    b = np.asarray(b, dtype=float); b = b[np.isfinite(b)]
    rng = np.random.default_rng(seed)
    f = (lambda x: float(np.median(x)) if len(x) else np.nan) if agg == "median" \
        else (lambda x: float(np.mean(x)) if len(x) else np.nan)
    point = f(a) - f(b)
    if not len(a) or not len(b): return point, np.nan, np.nan
    diffs = np.empty(n_boot)
    for i in range(n_boot):
        ia = rng.integers(0, len(a), size=len(a))
        ib = rng.integers(0, len(b), size=len(b))
        diffs[i] = f(a[ia]) - f(b[ib])
    return point, float(np.nanpercentile(diffs, 2.5)), float(np.nanpercentile(diffs, 97.5))


def main():
    ROOT = Path("/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/cfg_retrieval_v9_multitarget")
    ap = argparse.ArgumentParser()
    ap.add_argument("--targets", default="BTK,EGFR,JAK3,KRAS_G12C")
    ap.add_argument("--out_md", default=str(ROOT / "multi_target_summary.md"))
    ap.add_argument("--out_csv", default=str(ROOT / "multi_target_summary.csv"))
    args = ap.parse_args()

    targets = args.targets.split(",")
    rows = []
    for t in targets:
        tdir = ROOT / t
        base_2d = tdir / f"covalent_metric_panel_{t}_baseline.csv"
        base_cov = tdir / f"covalent_vina_cov_panel_{t}_baseline.csv"
        dpo_2d = tdir / f"covalent_metric_panel_{t}_dpo.csv"
        dpo_cov = tdir / f"covalent_vina_cov_panel_{t}_dpo.csv"
        if not (base_2d.exists() and base_cov.exists()):
            print(f"[skip {t}] baseline panels missing", flush=True)
            continue
        b2 = pd.read_csv(base_2d); bc = pd.read_csv(base_cov)
        bm = b2.merge(bc[["cell", "sample_idx", "vina_cov_score", "vina_cov_ok"]],
                        on=["cell", "sample_idx"], how="inner")
        b_ok = bm[(bm["valid"] == True) & (bm["acryl_largest"] == True)
                    & (bm["vina_cov_ok"] == True)]
        base_scores = b_ok["vina_cov_score"].astype(float).values
        row = {"target": t,
                 "baseline_n": int(len(b_ok)),
                 "baseline_valid_frac": float(b2["valid"].mean()),
                 "baseline_median_vina_cov": float(np.median(base_scores)) if len(base_scores) else np.nan,
                 "baseline_min_vina_cov": float(np.min(base_scores)) if len(base_scores) else np.nan,
                 }
        if dpo_2d.exists() and dpo_cov.exists():
            d2 = pd.read_csv(dpo_2d); dc = pd.read_csv(dpo_cov)
            dm = d2.merge(dc[["cell", "sample_idx", "vina_cov_score", "vina_cov_ok"]],
                            on=["cell", "sample_idx"], how="inner")
            d_ok = dm[(dm["valid"] == True) & (dm["acryl_largest"] == True)
                        & (dm["vina_cov_ok"] == True)]
            dpo_scores = d_ok["vina_cov_score"].astype(float).values
            row["dpo_n"] = int(len(d_ok))
            row["dpo_valid_frac"] = float(d2["valid"].mean())
            row["dpo_median_vina_cov"] = float(np.median(dpo_scores)) if len(dpo_scores) else np.nan
            row["dpo_min_vina_cov"] = float(np.min(dpo_scores)) if len(dpo_scores) else np.nan
            d, lo, hi = bootstrap_diff_ci(dpo_scores, base_scores, agg="median")
            row["d_vina_cov_median"] = d
            row["d_vina_cov_ci95"] = f"[{lo:+.3f}, {hi:+.3f}]"
        else:
            row["dpo_n"] = 0; row["dpo_valid_frac"] = np.nan
            row["dpo_median_vina_cov"] = np.nan
            row["d_vina_cov_median"] = np.nan; row["d_vina_cov_ci95"] = "NA"
        rows.append(row)

    df = pd.DataFrame(rows)
    df.to_csv(args.out_csv, index=False)
    print(df.to_string(index=False), flush=True)

    def crosses_zero(ci_str):
        if not isinstance(ci_str, str) or ci_str == "NA": return True
        try:
            a, b = ci_str.strip("[]").split(",")
            return float(a) <= 0 <= float(b)
        except Exception:
            return True

    win_targets = []
    for _, r in df.iterrows():
        v = r.get("d_vina_cov_median", 0)
        if isinstance(v, float) and not np.isnan(v) and v <= -3.0 and not crosses_zero(r.get("d_vina_cov_ci95", "")):
            win_targets.append(f"{r['target']}: {v:+.2f} kcal/mol {r['d_vina_cov_ci95']}")

    if win_targets:
        verdict = f"DPO GENERALIZES to {len(win_targets)}/{len(df)} kinases."
    else:
        verdict = "DPO DOES NOT GENERALIZE beyond ZAP70 at present hyperparameters/data."

    lines = [
        "# v9 Multi-Target DPO Summary",
        "",
        "**Motivation**: v6-FIXED showed DPO shifts ZAP70 Vina cov score by "
        "-14 kcal/mol.  v9 asks: does this generalize to other kinases?",
        "",
        "**Verdict**:",
        "",
        f"> **{verdict}**",
        "",
    ]
    if win_targets:
        lines += ["Targets where DPO CI-significantly beats baseline:", ""]
        for h in win_targets:
            lines += [f"- {h}"]
        lines += [""]
    lines += ["## Per-target metrics",
                "",
                df.to_string(index=False),
                "",
                "## Method notes",
                "",
                "- Per-target pipeline: (a) sample 200 mols from covFT+v2 baseline "
                "with target-specific ESM pocket + anchor SMILES, zero pose norm; "
                "(b) cov-Vina --score_only rescore; (c) build ~300 preference "
                "pairs (top-40% vs bottom-40% Vina cov, min gap 10); "
                "(d) DPO train 3 epochs (β=0.1, LR base=1e-6/new=5e-6); "
                "(e) resample 200 with DPO ckpt; (f) rescore; (g) bootstrap CI.",
                "- Targets: BTK (5P9J anchor ibrutinib), EGFR (5GTY anchor afatinib), "
                "JAK3 (5TOZ anchor PF-06651600-like), KRAS-G12C (6OIM anchor sotorasib).",
                "- All targets share same reproducibility print / save-fix ckpt pattern.",
                ""]
    Path(args.out_md).write_text("\n".join(lines))
    print(f"Wrote {args.out_md}", flush=True)


if __name__ == "__main__":
    main()
