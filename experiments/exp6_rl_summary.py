"""EXP6 retrospective LO — Phase 2 RL summary builder.

Aggregates per-cell metrics from rl_results.json + the per-target cohort CSVs
(after they have been pulled back from a100-b) into a single markdown report.

Run locally after `gcloud compute scp` of:
  data/exp6_retrospective/_rl/rl_results.json
  data/exp6_retrospective/<target>/iter*_<strategy>_<scorer>_cohort.csv

Writes:
  results/paper_evaluation/exp6_phase2_rl_summary.md
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

EXP6_ROOT = Path("data/exp6_retrospective")
OUT_MD = Path("results/paper_evaluation/exp6_phase2_rl_summary.md")


def main():
    results_path = EXP6_ROOT / "_rl" / "rl_results.json"
    if not results_path.exists():
        print(f"ERROR: {results_path} missing — did you pull it back from a100-b?")
        return
    results = json.loads(results_path.read_text())
    df = pd.DataFrame(results)
    # parse cell_id
    cells = []
    for cid in df["cell_id"]:
        parts = cid.split("_")
        # target may be 2 or 3 tokens (egfr_t790m, btk, kras_g12c)
        # iter_n is always 'iterN'
        try:
            iter_idx = next(i for i, p in enumerate(parts) if p.startswith("iter"))
            target = "_".join(parts[: iter_idx - 1])  # before strategy
            strategy = parts[iter_idx - 1]
            iter_n = int(parts[iter_idx].replace("iter", ""))
            scorer = parts[iter_idx + 1]
            cells.append((target, strategy, iter_n, scorer))
        except Exception:
            cells.append(("?", "?", -1, "?"))
    df[["target", "strategy", "iter", "scorer"]] = pd.DataFrame(cells, index=df.index)
    df = df.sort_values(["target", "scorer", "strategy", "iter"])

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    lines.append("# EXP6 Phase 2 RL — Per-Cell Summary\n")
    lines.append(f"\nTotal cells: {len(df)} | OK: {(df['status']=='ok').sum()} | "
                 f"resumed: {(df['status']=='resumed').sum()} | failed: {(~df['status'].isin(['ok','resumed'])).sum()}\n")

    for target in df["target"].unique():
        if target == "?":
            continue
        lines.append(f"\n## {target}\n")
        sub = df[df["target"] == target].copy()
        cols = ["cell_id", "status", "n_total", "valid_frac",
                "warhead_strict_frac", "warhead_generic_frac",
                "predicted_pIC50_mean", "predicted_pIC50_max",
                "tc_drug_max", "tc_drug_top10_mean", "tc_drug_ge_0.5_count",
                "n_unique_scaffolds", "dt"]
        present = [c for c in cols if c in sub.columns]
        # format numeric
        sub_disp = sub[present].copy()
        for c in present:
            if c in ("valid_frac", "warhead_strict_frac", "warhead_generic_frac",
                     "tc_drug_max", "tc_drug_top10_mean"):
                sub_disp[c] = sub_disp[c].apply(lambda x: f"{float(x):.3f}" if pd.notna(x) else "-")
            elif c in ("predicted_pIC50_mean", "predicted_pIC50_max"):
                sub_disp[c] = sub_disp[c].apply(lambda x: f"{float(x):.2f}" if pd.notna(x) else "-")
            elif c == "dt":
                sub_disp[c] = sub_disp[c].apply(lambda x: f"{float(x):.0f}s" if pd.notna(x) else "-")
        lines.append(sub_disp.to_markdown(index=False))
        lines.append("\n")

    # Headline: did any cell reach Tc>=0.5 to drug?
    lines.append("\n## Headline metrics\n")
    if "tc_drug_max" in df.columns:
        hits = df[df["tc_drug_max"] >= 0.5]
        lines.append(f"\n- Cells with at least one Tc>=0.5 to the drug: **{len(hits)} / {len(df)}**\n")
        if len(hits) > 0:
            for _, r in hits.iterrows():
                lines.append(f"  - {r['cell_id']}: max Tc-to-drug = {r['tc_drug_max']:.3f}\n")
    if "warhead_strict_frac" in df.columns:
        mean_war = df.groupby(["target", "scorer", "strategy"])["warhead_strict_frac"].mean()
        lines.append("\n### Mean warhead retention by (target, scorer, strategy)\n")
        lines.append("\n```\n" + mean_war.to_string() + "\n```\n")

    # Phase 3 recommendation
    lines.append("\n## Phase 3 deep-dive recommendation\n")
    if len(df) > 0 and "tc_drug_max" in df.columns:
        df["composite_quality"] = (
            df["warhead_strict_frac"].fillna(0).clip(0, 1) * 0.4
            + df["tc_drug_max"].fillna(0).clip(0, 1) * 0.4
            + (df["predicted_pIC50_mean"].fillna(0).clip(0, 10) / 10) * 0.2
        )
        top = df.sort_values("composite_quality", ascending=False).head(5)
        lines.append("\nTop 5 cells by composite quality (warhead_strict 0.4 + tc_drug_max 0.4 + pIC50/10 0.2):\n\n")
        lines.append(top[["cell_id", "warhead_strict_frac", "tc_drug_max",
                          "predicted_pIC50_mean", "composite_quality"]].to_markdown(index=False))
        lines.append("\n")

    OUT_MD.write_text("\n".join(lines))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
