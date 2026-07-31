#!/usr/bin/env python
"""Aggregate per-persona scoring outputs into a single CSV + summary.

Reads 10 persona JSON files from results/paper_evaluation/llm_cohort_output/,
produces aggregated CSV with consensus stats and per-persona sub-scores,
writes a markdown summary.

Usage:
  python experiments/aggregate_llm_cohort.py [--input_dir DIR] [--output_csv FILE] [--summary_md FILE]
"""
from __future__ import annotations
import argparse, json
from collections import Counter
from pathlib import Path
from statistics import mean, stdev

DEFAULT_INPUT = Path("results/paper_evaluation/llm_cohort_output")
DEFAULT_CSV = Path("results/paper_evaluation/llm_cohort_output/llm_cohort_scores.csv")
DEFAULT_MD = Path("results/paper_evaluation/llm_cohort_output/llm_cohort_summary.md")

PERSONA_FILES = {
    "medchem_reviewer":            "medchem_reviewer.json",
    "kinase_chemist":              "kinase_chemist.json",
    "cov_chem_specialist":         "cov_chem_specialist.json",
    "pk_adme_expert":              "pk_adme_expert.json",
    "crystallographer":            "crystallographer.json",
    "selectivity_specialist":      "selectivity_specialist.json",
    "synthesis_chemist":           "synthesis_chemist.json",
    "patent_attorney":             "patent_attorney.json",
    "translational_biologist":     "translational_biologist.json",
    "resistance_specialist":       "resistance_specialist.json",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input_dir", default=DEFAULT_INPUT, type=Path)
    ap.add_argument("--output_csv", default=DEFAULT_CSV, type=Path)
    ap.add_argument("--summary_md", default=DEFAULT_MD, type=Path)
    args = ap.parse_args()

    # Load all persona records
    persona_records = {}  # persona -> list of dicts (one per mol)
    for persona, fname in PERSONA_FILES.items():
        path = args.input_dir / fname
        if not path.exists():
            print(f"WARN: missing {path}")
            continue
        with open(path) as f:
            persona_records[persona] = json.load(f)

    # Index by mol_id
    mol_scores = {}  # mol_id -> {persona -> record}
    for persona, recs in persona_records.items():
        for r in recs:
            mid = r["mol_id"]
            mol_scores.setdefault(mid, {})[persona] = r

    # Aggregate per molecule
    aggregated = []
    for mid, ps in mol_scores.items():
        scores = [ps[p]["score"] for p in PERSONA_FILES if p in ps]
        concerns = [ps[p].get("top_concern", "") for p in PERSONA_FILES if p in ps]
        row = {
            "mol_id": mid,
            "n_personas": len(scores),
            "llm_consensus_score": round(mean(scores), 2),
            "llm_score_std": round(stdev(scores), 2) if len(scores) > 1 else 0.0,
            "llm_min_score": min(scores),
            "llm_max_score": max(scores),
            "llm_score_range": max(scores) - min(scores),
        }
        # Per-persona sub-scores
        for p in PERSONA_FILES:
            row[f"score_{p}"] = ps[p]["score"] if p in ps else None
        # Most common concern keyword (rough)
        concern_text = " ".join(concerns).lower()
        # Extract keywords (very rough)
        keywords = ["gsh", "pKa", "hERG", "logp", "mw", "tpsa", "selectiv", "patent", "fto",
                    "warhead", "geometry", "strain", "michael", "resist", "metabol", "permeab",
                    "synth", "stereo"]
        kw_count = Counter()
        for kw in keywords:
            kw_count[kw] = concern_text.count(kw)
        top_concerns = kw_count.most_common(3)
        row["llm_top_concern"] = ", ".join([f"{k}({v})" for k, v in top_concerns if v > 0]) or "n/a"
        aggregated.append(row)

    # Sort by consensus desc
    aggregated.sort(key=lambda r: -r["llm_consensus_score"])

    # Write CSV
    import csv
    fieldnames = list(aggregated[0].keys())
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with open(args.output_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(aggregated)

    # Write summary
    md_lines = []
    md_lines.append(f"# LLM Cohort Scoring — Aggregated Summary\n")
    md_lines.append(f"Top 10 test cohort, 10 expert personas, N={len(aggregated)} molecules.\n")
    md_lines.append(f"\n## Headline\n")

    strong = [r for r in aggregated if r["llm_consensus_score"] >= 7.0]
    md_lines.append(f"- **{len(strong)} of {len(aggregated)}** mols at consensus ≥ 7.0\n")
    md_lines.append(f"- Mean consensus: {mean([r['llm_consensus_score'] for r in aggregated]):.2f}\n")
    md_lines.append(f"- Max consensus: {max([r['llm_consensus_score'] for r in aggregated]):.2f}\n")

    high_disagreement = sorted(aggregated, key=lambda r: -r["llm_score_std"])[:3]
    md_lines.append(f"\n## Top 3 picks (by consensus)\n")
    for i, r in enumerate(aggregated[:3]):
        md_lines.append(f"{i+1}. **{r['mol_id']}** — consensus={r['llm_consensus_score']} "
                       f"(std={r['llm_score_std']}, range={r['llm_min_score']}-{r['llm_max_score']})\n")
        md_lines.append(f"   - Top concerns: {r['llm_top_concern']}\n")

    md_lines.append(f"\n## Top 3 disagreement cases (highest std — debate-worthy)\n")
    for i, r in enumerate(high_disagreement):
        md_lines.append(f"{i+1}. **{r['mol_id']}** — consensus={r['llm_consensus_score']}, std={r['llm_score_std']}, "
                       f"range={r['llm_min_score']}-{r['llm_max_score']}\n")
        per_persona = {p: r[f"score_{p}"] for p in PERSONA_FILES if r[f"score_{p}"] is not None}
        sorted_p = sorted(per_persona.items(), key=lambda kv: -kv[1])
        md_lines.append(f"   - Highest: {sorted_p[0][0]}={sorted_p[0][1]}; Lowest: {sorted_p[-1][0]}={sorted_p[-1][1]}\n")

    md_lines.append(f"\n## Full ranking\n")
    md_lines.append(f"| Rank | mol_id | consensus | std | min | max |\n")
    md_lines.append(f"|------|--------|-----------|-----|-----|-----|\n")
    for i, r in enumerate(aggregated):
        md_lines.append(f"| {i+1} | {r['mol_id']} | {r['llm_consensus_score']} | "
                       f"{r['llm_score_std']} | {r['llm_min_score']} | {r['llm_max_score']} |\n")

    md_lines.append(f"\n## Per-persona top picks\n")
    for persona in PERSONA_FILES:
        if persona not in persona_records:
            continue
        recs = persona_records[persona]
        sorted_recs = sorted(recs, key=lambda r: -r["score"])
        top3 = sorted_recs[:3]
        picks_str = ", ".join([f"{r['mol_id']}({r['score']})" for r in top3])
        md_lines.append(f"- **{persona}**: {picks_str}\n")

    args.summary_md.parent.mkdir(parents=True, exist_ok=True)
    with open(args.summary_md, "w") as f:
        f.writelines(md_lines)

    print(f"Wrote CSV: {args.output_csv}")
    print(f"Wrote summary: {args.summary_md}")
    print(f"\nTop 3 by consensus:")
    for i, r in enumerate(aggregated[:3]):
        print(f"  {i+1}. {r['mol_id']}: {r['llm_consensus_score']} (std {r['llm_score_std']})")


if __name__ == "__main__":
    main()
