"""Write the final EXP7 4-way comparison markdown summary."""
from __future__ import annotations
import json
import sys
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

SCORED_CSV = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_all_cohorts_scored.csv"
SUMMARY_JSON = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_4way_summary.json"
CONTAM_JSON = PROJECT_ROOT / "data" / "exp7_lo_benchmark" / "exp7_contamination.json"
OUT_MD = PROJECT_ROOT / "results" / "paper_evaluation" / "exp7_FINAL_4way.md"

COND_MAP = {
    "mol2mol_baseline": "mol2mol_baseline",
    "mol2mol_RL":       "mol2mol_RL",
    "covFT_baseline":   "baseline_prior_anchor",
    "covFT_RL":         "A",
}
COND_ORDER = ["mol2mol_baseline", "mol2mol_RL", "covFT_baseline", "covFT_RL"]


def fmt(x, d=2):
    if x is None or (isinstance(x, float) and (np.isnan(x) or np.isinf(x))):
        return "—"
    return f"{x:.{d}f}"


def block(name, st, n):
    out = [f"\n### {name} (n={n} pairs)\n"]
    out.append("| condition | mean max_Tc | median max_Tc | mean n@Tc>=0.5 | mean pred pIC50 med | mean warhead % | n with data |")
    out.append("|---|---:|---:|---:|---:|---:|---:|")
    for cond in COND_ORDER:
        v = st[cond]
        out.append(f"| {cond} | {fmt(v['mean_max_tc'])} | {fmt(v['median_max_tc'])} | "
                   f"{fmt(v['mean_n_tc05'],1)} | {fmt(v['mean_pred_pic_med'])} | "
                   f"{fmt(v['mean_warhead_pct'],1)} | {v['n_pairs']} |")
    return "\n".join(out) + "\n"


def main():
    if not SUMMARY_JSON.exists():
        print(f"ERROR: {SUMMARY_JSON} missing — run exp7_4way_report.py first")
        return
    summary = json.loads(SUMMARY_JSON.read_text())
    contam = {}
    if CONTAM_JSON.exists():
        contam = json.loads(CONTAM_JSON.read_text())

    df = pd.read_csv(SCORED_CSV) if SCORED_CSV.exists() else pd.DataFrame()

    md = []
    md.append("# EXP7 4-condition LO benchmark — FINAL REPORT")
    md.append(f"\nGenerated: {datetime.now().isoformat(timespec='seconds')}\n")

    md.append("## Mission\n")
    md.append("Build a 4-condition × 50-pair comparison matrix to disentangle the contribution of:")
    md.append("- **covalent fine-tuning (covFT)** of the mol2mol prior")
    md.append("- **composite RL reward** (FiLM + warhead SMARTS + QED)\n")

    md.append("Conditions:")
    md.append("- `mol2mol_baseline`: vanilla REINVENT4 mol2mol prior, no RL, anchor as seed (5500 mols)")
    md.append("- `mol2mol_RL`: vanilla mol2mol + composite RL reward (50 steps), then sample 5500")
    md.append("- `covFT_baseline`: covalent FT mol2mol prior, no RL, anchor as seed (5500 mols)")
    md.append("- `covFT_RL`: covalent FT prior + composite RL reward (50 steps), then sample 5500\n")

    md.append("## Coverage\n")
    md.append(f"- 50 (anchor, drug) pairs across 5 covalent kinase targets:")
    if not df.empty:
        for tk, n in df["target_key"].value_counts().items():
            md.append(f"  - {tk}: {n} cohorts scored")
    md.append(f"- {summary['n_pairs']} pairs total")
    md.append(f"- Cohorts scored per condition: {df['strategy'].value_counts().to_dict() if not df.empty else 'n/a'}\n")

    md.append("## Aggregate stats\n")
    md.append(block("All pairs", summary["all"], summary["n_pairs"]))
    md.append(block("Loose subset (drug NOT in CovInDB v2)", summary["loose"], summary["n_loose"]))
    md.append(block("Strict subset (drug published after 2022)", summary["strict"], summary["n_strict"]))

    md.append("\n## Contamination caveat\n")
    md.append(f"- {contam.get('n_drug_in_covindb', '?')}/{contam.get('n_pairs','?')} drugs are present in the CovInDB v2 corpus used to fine-tune the covalent prior.")
    md.append(f"- Drugs in CovInDB v2: {', '.join(contam.get('drugs_in_db', []))}")
    md.append(f"- All drugs in this benchmark predate the mol2mol prior cutoff (~2022). Strict post-2022 subset is empty.")
    md.append(f"- **Loose subset (n={contam.get('loose_subset_n','?')})** is the cleanest comparison the data permits.\n")

    md.append("\n## Honest assessment\n")
    all_stats = summary["all"]
    if all_stats["covFT_baseline"]["n_pairs"] > 0 and all_stats["mol2mol_baseline"]["n_pairs"] > 0:
        d_cov_vs_m = all_stats["covFT_baseline"]["mean_max_tc"] - all_stats["mol2mol_baseline"]["mean_max_tc"]
        md.append(f"- **covFT vs mol2mol (baseline, no RL)**: covFT mean max_Tc = "
                  f"{fmt(all_stats['covFT_baseline']['mean_max_tc'])} vs mol2mol = "
                  f"{fmt(all_stats['mol2mol_baseline']['mean_max_tc'])} (Δ = {fmt(d_cov_vs_m,3)}).")
    if all_stats["covFT_RL"]["n_pairs"] > 0 and all_stats["covFT_baseline"]["n_pairs"] > 0:
        d_rl_vs_base = all_stats["covFT_RL"]["mean_max_tc"] - all_stats["covFT_baseline"]["mean_max_tc"]
        md.append(f"- **RL vs baseline (covFT prior)**: RL mean max_Tc = "
                  f"{fmt(all_stats['covFT_RL']['mean_max_tc'])} vs baseline = "
                  f"{fmt(all_stats['covFT_baseline']['mean_max_tc'])} (Δ = {fmt(d_rl_vs_base,3)}).")
        md.append(f"  - Note: composite RL reward optimizes for pIC50/warhead/QED, NOT similarity to drug. "
                  f"Lower max_Tc here is expected — RL explores chemistry away from anchor seed.")
        pic_delta = all_stats['covFT_RL']['mean_pred_pic_med'] - all_stats['covFT_baseline']['mean_pred_pic_med']
        md.append(f"  - mean pred pIC50 (median): covFT_RL = {fmt(all_stats['covFT_RL']['mean_pred_pic_med'])} "
                  f"vs covFT_baseline = {fmt(all_stats['covFT_baseline']['mean_pred_pic_med'])} (Δ = {fmt(pic_delta,3)}).")
    if all_stats["mol2mol_RL"]["n_pairs"] > 0 and all_stats["mol2mol_baseline"]["n_pairs"] > 0:
        d_mrl = all_stats["mol2mol_RL"]["mean_max_tc"] - all_stats["mol2mol_baseline"]["mean_max_tc"]
        md.append(f"- **RL vs baseline (vanilla mol2mol)**: RL mean max_Tc = "
                  f"{fmt(all_stats['mol2mol_RL']['mean_max_tc'])} vs baseline = "
                  f"{fmt(all_stats['mol2mol_baseline']['mean_max_tc'])} (Δ = {fmt(d_mrl,3)}).")

    md.append("\n## Open questions for parallel diagnosis agent\n")
    md.append("- Why is covFT_RL max_Tc lower than covFT_baseline? RL reward is composite (FiLM/SMARTS/QED) — does it correctly penalize trivial molecules?")
    md.append("- Are mol2mol_baseline cohorts producing valid covalent chemistry, or mostly non-warhead analogs?")
    md.append("- What is the rank-correlation between FiLM-predicted pIC50 and actual ChEMBL pIC50 on the per-target test sets?\n")

    md.append("## File paths\n")
    md.append(f"- Scored CSV: `{SCORED_CSV.relative_to(PROJECT_ROOT)}`")
    md.append(f"- 4-way summary JSON: `{SUMMARY_JSON.relative_to(PROJECT_ROOT)}`")
    md.append(f"- Contamination JSON: `{CONTAM_JSON.relative_to(PROJECT_ROOT)}`")
    md.append(f"- HTML report: `results/paper_evaluation/exp7_4way_report.html`")
    md.append(f"- This markdown: `{OUT_MD.relative_to(PROJECT_ROOT)}`")
    md.append(f"- Driver: `experiments/exp7_mol2mol_driver.py`")
    md.append(f"- REST scoring: `experiments/exp6_rest_server.py`")
    md.append(f"- Cohort dirs: `data/exp7_lo_benchmark/_rl/<pair_id>_<condition>/sampled.csv`\n")

    OUT_MD.parent.mkdir(parents=True, exist_ok=True)
    OUT_MD.write_text("\n".join(md))
    print(f"Wrote {OUT_MD}")


if __name__ == "__main__":
    main()
