"""QA #7 (2026-06-24): Aggregate exp5b deep-comparison results into a summary.

Reads results/paper_evaluation/exp5b_filmdelta_zap70_deep.json (incremental;
re-readable as more runs land), produces summary table per (split, N).

Compares:
  - FiLMDelta delta MAE
  - Direct (reconstructed delta) MAE
  - Direct (absolute pIC50) MAE
  - Per metric mean and std across seeds
"""
from __future__ import annotations
import json
import time
from pathlib import Path
import pandas as pd
import numpy as np

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "results" / "paper_evaluation"

INP = OUT_DIR / "exp5b_filmdelta_zap70_deep.json"


def main():
    if not INP.exists():
        print("exp5b JSON not found")
        return
    d = json.loads(INP.read_text())
    runs = d.get("runs", [])
    if not runs:
        print("no runs in exp5b JSON")
        return
    print(f"[{time.strftime('%H:%M:%S')}] reading {len(runs)} runs")
    rows = []
    for r in runs:
        rows.append({
            "split": r["split"],
            "seed": r["seed"],
            "N": r["N_requested"],
            "film_mae": r["filmdelta"]["mae"],
            "film_spearman": r["filmdelta"]["spearman"],
            "film_pearson": r["filmdelta"]["pearson"],
            "film_r2": r["filmdelta"]["r2"],
            "direct_delta_mae": r["direct_reconstructed_delta"]["mae"],
            "direct_delta_spearman": r["direct_reconstructed_delta"]["spearman"],
            "direct_abs_mae": r["direct_abs_pIC50"]["mae"],
            "direct_abs_spearman": r["direct_abs_pIC50"]["spearman"],
            "n_test_pairs": r["n_test_pairs"],
        })
    df = pd.DataFrame(rows)
    summary = (df.groupby(["split", "N"])
                 .agg(n_seeds=("seed", "nunique"),
                      film_mae_mean=("film_mae", "mean"),
                      film_mae_std=("film_mae", "std"),
                      film_spr_mean=("film_spearman", "mean"),
                      direct_delta_mae_mean=("direct_delta_mae", "mean"),
                      direct_delta_spr_mean=("direct_delta_spearman", "mean"),
                      direct_abs_mae_mean=("direct_abs_mae", "mean"),
                      direct_abs_spr_mean=("direct_abs_spearman", "mean"))
                 .round(3).reset_index())
    summary["mae_adv_film_vs_direct_delta"] = (summary["direct_delta_mae_mean"] - summary["film_mae_mean"]).round(3)
    summary["mae_adv_pct"] = (100 * summary["mae_adv_film_vs_direct_delta"] / summary["direct_delta_mae_mean"]).round(2)
    summary["mae_adv_film_vs_direct_abs"] = (summary["direct_abs_mae_mean"] - summary["film_mae_mean"]).round(3)
    summary["mae_adv_abs_pct"] = (100 * summary["mae_adv_film_vs_direct_abs"] / summary["direct_abs_mae_mean"]).round(2)
    out_csv = OUT_DIR / "exp5b_summary.csv"
    summary.to_csv(out_csv, index=False)
    print(f"Wrote {out_csv}")
    print(summary.to_string(index=False))

    # MD
    lines = ["# Exp 5b — FiLMDelta vs direct (delta + absolute) on ZAP70\n",
             f"Reads `{INP.name}` snapshot.  Re-run as more runs land on disk.\n",
             "## Per (split, N) means (across seeds)\n"]
    headers = list(summary.columns)
    lines.append("| " + " | ".join(headers) + " |")
    lines.append("|" + "---|" * len(headers))
    for _, r in summary.iterrows():
        row_vals = []
        for h in headers:
            v = r[h]
            if isinstance(v, float):
                if "pct" in h:
                    row_vals.append(f"{v:.1f}%")
                else:
                    row_vals.append(f"{v:.3f}")
            else:
                row_vals.append(str(v))
        lines.append("| " + " | ".join(row_vals) + " |")
    lines.append("")
    lines.append("## Interpretation\n")
    lines.append("- **`mae_adv_pct`**: % MAE reduction by FiLMDelta vs Direct (reconstructed delta). Positive = FiLM wins on Δ-prediction.")
    lines.append("- **`mae_adv_abs_pct`**: % MAE reduction by FiLMDelta vs Direct (absolute pIC50 — direct predictor for B's pIC50).")
    lines.append("- FiLMDelta is structurally designed for Δ; direct-abs is the strong baseline a paper reviewer will ask about.\n")
    md = OUT_DIR / "exp5b_summary.md"
    md.write_text("\n".join(lines))
    print(f"Wrote {md}")


if __name__ == "__main__":
    main()
