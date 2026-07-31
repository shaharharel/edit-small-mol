"""Build the final cross-stream R1/R2/R3 vs baselines summary report.

Reads per-stream `{r1,r2,r3}_geometry.json` plus the existing baseline cohort
summaries and answers the 4 deliverable questions:
  1. Does R1 fix Stream B's mode collapse? (warhead retention >85%?)
  2. Does R2 match M1a's geometry win?
  3. Does R3 beat M1a on binding affinity (cov-Vina median)?
  4. Recommended next step.

Writes results/paper_evaluation/r_streams_summary.md
"""
from __future__ import annotations
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
PE = ROOT / "results/paper_evaluation"

STREAMS = ["r1", "r2", "r3"]
BASELINES_ORDER = ["m1a", "covft", "rl", "libinvent", "dpo_composite", "dpo_geometry"]


def load_geometry(stream: str):
    p = PE / f"{stream}_geometry.json"
    if not p.exists():
        return None
    return json.loads(p.read_text())


def main():
    data = {s: load_geometry(s) for s in STREAMS}
    missing = [s for s, d in data.items() if d is None]
    if missing:
        print(f"WARNING: missing geometry json for: {missing}")

    # Collect all cohort_summaries (union)
    all_summaries: dict[str, dict] = {}
    for s, d in data.items():
        if d is None:
            continue
        for cn, cs in d["cohort_summaries"].items():
            if cn not in all_summaries:
                all_summaries[cn] = cs

    # Headline questions
    q1 = q2 = q3 = "N/A (data missing)"
    rec = "N/A"

    if data.get("r1") is not None:
        r1 = data["r1"]["cohort_summaries"].get("r1", {})
        wh = r1.get("frac_acrylamide_match")
        if wh is None:
            q1 = "R1 cohort summary missing acrylamide retention metric."
        else:
            verdict = "YES" if wh >= 0.85 else "NO"
            q1 = (f"R1 acrylamide retention = **{wh:.3f}** ({wh*100:.1f}%). "
                  f"Stream B was 0.027 (2.7%). Threshold for 'fix' = >=0.85. **{verdict}**.")

    m1a = all_summaries.get("m1a") or all_summaries.get("m1a", {})
    if data.get("r2") is not None:
        r2 = data["r2"]["cohort_summaries"].get("r2", {})
        if m1a and r2:
            r2_pre = r2.get("pre_reactivity_score_mean")
            m1a_pre = m1a.get("pre_reactivity_score_mean")
            r2_dev = r2.get("planar_dev_deg_median")
            m1a_dev = m1a.get("planar_dev_deg_median")
            # M1a geometry "win" criterion: pre_reactivity_score_mean closely matches or beats M1a's
            if r2_pre is not None and m1a_pre is not None:
                close = abs(r2_pre - m1a_pre) < 0.05
                better = r2_pre >= m1a_pre
                if better:
                    verdict = "YES (matches or exceeds M1a)"
                elif close:
                    verdict = "PARTIAL (within 0.05 of M1a mean)"
                else:
                    verdict = "NO"
                q2 = (f"R2 pre_reactivity mean = **{r2_pre:.3f}** vs M1a = **{m1a_pre:.3f}**; "
                      f"R2 planar_dev_median = {r2_dev:.2f}° vs M1a = {m1a_dev:.2f}°. **{verdict}**.")
            else:
                q2 = "R2 / M1a missing pre_reactivity metric."
        else:
            q2 = "R2 or M1a cohort summary missing."

    if data.get("r3") is not None:
        r3 = data["r3"]["cohort_summaries"].get("r3", {})
        if m1a and r3:
            r3_aff = r3.get("vina_affinity_median")
            m1a_aff = m1a.get("vina_affinity_median")
            if r3_aff is not None and m1a_aff is not None:
                # lower vina_affinity = better
                verdict = "YES" if r3_aff < m1a_aff else "NO"
                delta = m1a_aff - r3_aff
                q3 = (f"R3 cov-Vina median = **{r3_aff:.2f}** vs M1a = **{m1a_aff:.2f}** "
                      f"(lower = better; Δ = {delta:+.2f}). **{verdict}**.")
            else:
                q3 = "R3 or M1a missing vina_affinity_median."
        else:
            q3 = "R3 or M1a cohort summary missing."

    # Recommended next step heuristic
    rec_lines = []
    if "YES" in q1:
        rec_lines.append("- R1 fixes mode collapse: ship it for production DPO runs (replace Stream B).")
    elif "PARTIAL" in q1 or "NO" in q1:
        rec_lines.append("- R1 still partly mode-collapsing; try further tightening Tc band or reducing epochs.")
    if "YES" in q2:
        rec_lines.append("- R2 demonstrates cheap geom reward is sufficient — favor R2 over expensive Boltz-DPO.")
    if "YES" in q3:
        rec_lines.append("- R3 distilled cov-Vina improves binding; investigate ensembling with M1a's pocket-aware decoder.")
    if not rec_lines:
        rec_lines.append("- None of R1/R2/R3 cleanly dominate; revisit reward shaping or move to on-policy RL with on-the-fly cov-Vina.")
    rec = "\n".join(rec_lines)

    # Pairwise verdicts table
    pairwise_table_lines = []
    for s in STREAMS:
        d = data.get(s)
        if d is None:
            continue
        pairwise_table_lines.append(f"\n### {s.upper()} pairwise (positive_axes_count out of 3)\n")
        pairwise_table_lines.append("| baseline | verdict | pos_axes | notes |")
        pairwise_table_lines.append("|---|---|---|---|")
        for key, payload in d["pairwise_tests"].items():
            v = payload["verdict"]
            pairwise_table_lines.append(
                f"| {key.split('_vs_')[1]} | {v['verdict']} | {v['positive_axes_count']}/3 | — |"
            )

    # Per-cohort summary table
    metrics_keys = [
        "n_total", "frac_acrylamide_match",
        "pre_reactivity_score_mean", "pre_reactivity_score_frac_ge_0p5",
        "planar_dev_deg_median",
        "vina_affinity_median", "warhead_sg_distance_median_A", "bd_angle_median_deg",
        "cov_vina_n_attempted", "cov_vina_n_ok",
    ]
    # Order: streams first, then baselines
    cohort_order = [c for c in STREAMS if c in all_summaries] + \
                   [b for b in BASELINES_ORDER if b in all_summaries]
    sum_lines = ["| metric | " + " | ".join(cohort_order) + " |"]
    sum_lines.append("|" + "---|" * (len(cohort_order) + 1))
    for k in metrics_keys:
        row = [k]
        for cn in cohort_order:
            v = all_summaries[cn].get(k)
            if isinstance(v, float):
                row.append(f"{v:.3f}")
            elif v is None:
                row.append("-")
            else:
                row.append(str(v))
        sum_lines.append("| " + " | ".join(row) + " |")

    out = []
    out.append("# R-stream summary: R1 / R2 / R3 vs baselines\n")
    out.append("## Headline answers\n")
    out.append(f"**Q1. Does R1 fix Stream B's mode collapse?**  \n{q1}\n")
    out.append(f"**Q2. Does R2 (cheap RL geom reward) match M1a's geometry win?**  \n{q2}\n")
    out.append(f"**Q3. Does R3 (distilled cov-Vina RL) beat M1a on binding affinity?**  \n{q3}\n")
    out.append(f"**Recommended next step**:\n{rec}\n")
    out.append("\n## Cross-cohort summary\n")
    out.extend(sum_lines)
    out.append("")
    out.append("## Pairwise verdicts (Mann-Whitney + Cliff's δ on 4 axes)\n")
    out.extend(pairwise_table_lines)

    out_md = PE / "r_streams_summary.md"
    out_md.write_text("\n".join(out))
    print(f"Wrote {out_md}")


if __name__ == "__main__":
    main()
