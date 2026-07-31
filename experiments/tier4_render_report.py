#!/usr/bin/env python3
"""Render the Tier 4 section of the cohort-comparison report.

Reads per-cohort JSON metrics from `results/paper_evaluation/tier4_rl_cohorts/`
and emits a markdown report with one subsection per cohort plus a side-by-side
metric table. Each subsection starts with a 1-line configuration description.

Output: `results/paper_evaluation/tier4_rl_cohorts/tier4_report.md`
"""
import json
from pathlib import Path

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
T4 = PROJECT / "results/paper_evaluation/tier4_rl_cohorts"

# Cohort order in the report (matches the order RL ablations should be presented).
ORDER = [
    ("EXP6_v3", "Tier 4a"),
    ("EXP2_V2_RL_v2", "Tier 4b"),
    ("EXP6_v4", "Tier 4c"),
    ("EXP6_v5", "Tier 4d"),
]


def load(name):
    p = T4 / f"{name}_metrics.json"
    if not p.exists():
        return None
    with open(p) as f:
        return json.load(f)


def fmt_pct(x, nd=1):
    return f"{x:.{nd}f}%" if x is not None else "—"


def fmt_num(x, nd=2):
    return f"{x:.{nd}f}" if x is not None else "—"


def subsection(label, name, meta):
    if meta is None:
        return (
            f"### {label} — {name}\n"
            f"_Sampling/metrics not yet available._\n"
        )
    return f"""### {label} — {name}
_{meta['config']}_

- **Prior**: {meta['prior']}
- **RL reward**: {meta['reward']}  ·  **RL steps**: {meta['rl_steps']}
- **N samples** (unique canonical from sampling step): **{meta['n_total']:,}**
- **Valid SMILES**: {meta['n_valid']:,} ({fmt_pct(meta['validity_pct'])})  ·  **Disconnected**: {meta['n_disconnected']:,} ({fmt_pct(meta['disconnected_pct'])})
- **Acrylamide on largest fragment** (strict `[CH2]=[CH]C(=O)[N;!H2]`): **{fmt_pct(meta['acryl_largest_frag_pct'])}**
- **Unique Murcko scaffolds**: {meta['n_unique_murcko_scaffolds']:,}
- **MW**: {fmt_num(meta['mw_mean'])} ± {fmt_num(meta['mw_std'])} Da  ·  10–90th: [{fmt_num(meta['mw_p10'])} – {fmt_num(meta['mw_p90'])}]
- **QED**: {fmt_num(meta['qed_mean'], 3)} ± {fmt_num(meta['qed_std'], 3)}  ·  **MolLogP**: {fmt_num(meta['logp_mean'], 2)}  ·  **TPSA**: {fmt_num(meta['tpsa_mean'])}  ·  **RotB**: {fmt_num(meta['rotbonds_mean'])}
- **Lipinski-ok**: {fmt_pct(meta['lipinski_compliant_pct'])}  ·  **Veber-ok**: {fmt_pct(meta['veber_compliant_pct'])}  ·  **Brenk-alert**: {fmt_pct(meta['brenk_alert_pct'])}
"""


def side_by_side(cohorts):
    """Markdown table comparing all loaded cohorts."""
    have = [(label, name, c) for label, name, c in cohorts if c is not None]
    if not have:
        return "_(no cohort metrics loaded yet)_\n"
    header = "| Metric | " + " | ".join(f"{lbl}<br>{n}" for lbl, n, _ in have) + " |"
    sep = "|---|" + "|".join(["---"] * len(have)) + "|"
    rows = [header, sep]
    def row(key, label, fmt=fmt_num):
        rows.append("| " + label + " | " + " | ".join(
            str(fmt(c.get(key))) if c.get(key) is not None else "—"
            for _, _, c in have
        ) + " |")
    row("n_total", "N samples", lambda x: f"{x:,}")
    row("validity_pct", "Validity %", fmt_pct)
    row("disconnected_pct", "Disconnected %", fmt_pct)
    row("acryl_largest_frag_pct", "**Acryl largest-frag %**", lambda x: f"**{fmt_pct(x)}**")
    row("n_unique_murcko_scaffolds", "Unique Murcko scaffolds", lambda x: f"{x:,}")
    row("mw_mean", "Mean MW (Da)", fmt_num)
    row("mw_p10", "MW 10th pct", fmt_num)
    row("mw_p90", "MW 90th pct", fmt_num)
    row("qed_mean", "Mean QED", lambda x: fmt_num(x, 3))
    row("logp_mean", "Mean MolLogP", lambda x: fmt_num(x, 2))
    row("tpsa_mean", "Mean TPSA", fmt_num)
    row("rotbonds_mean", "Mean RotB", fmt_num)
    row("lipinski_compliant_pct", "Lipinski-ok %", fmt_pct)
    row("veber_compliant_pct", "Veber-ok %", fmt_pct)
    row("brenk_alert_pct", "Brenk-alert %", fmt_pct)
    return "\n".join(rows) + "\n"


def main():
    cohorts = [(label, name, load(name)) for name, label in ORDER]

    # Vina cost estimate (read summary or recompute)
    summary_path = T4 / "summary.json"
    vina_block = ""
    if summary_path.exists():
        with open(summary_path) as f:
            summary = json.load(f)
        v = summary.get("vina_cost_estimate", {})
        if v:
            vina_block = f"""
## AD-CovDock Vina cost estimate

- Mols to dock (across all 4 Tier 4 cohorts): **{v['total_mols']:,}**
- Per-mol time (6 parallel workers on V100): {v['sec_per_mol_6workers']} s
- **Single V100 wallclock**: ~**{v['single_v100_hours']} h**
- **Two V100s in parallel**: ~**{v['two_v100_hours']} h**
- V100 spot $/hr ≈ $0.74; two-V100 parallel cost ≈ **${v['two_v100_hours'] * 0.74 * 2:.2f}**

> Note: this estimate uses the same AD-CovDock pipeline (RDKit 3D embed → meeko CovalentBuilder tether → Vina `--score_only`) as the in-loop RL Vina component. It produces an "input pose" geometry per molecule, useful for d(Cβ-SG) / BD angle metrics in addition to the Vina score.
"""

    md = ["# Tier 4 — RL-Fine-tuned Mol2Mol Cohorts (overnight 2026-06-04→05)\n"]

    # TL;DR up front (per scientific-analyst QA — the warhead-token finding should not be buried)
    md.append("""> **TL;DR (3 findings).**
> 1. **Vina-in-loop RL on a prior without warhead tokens (v4) hijacks the reward**: policy abandoned acrylamide (99%→0.2%, Brenk-alert 100%→9%); meeko CovalentBuilder then failed in 98% of mols → Vina success collapsed 76%→2%. Agent maximized FiLM+QED while Vina went to zero.
> 2. **Same RL recipe on warhead-token prior (v5) keeps 99% acryl AND improves drug-likeness** (QED 0.27→0.50, Lipinski-ok 27%→86%, MW −117 Da, scaffolds 15K→40K). Warhead-token conditioning is a prerequisite for covalent RL with composite rewards.
> 3. **Vina-in-loop is the lever for drug-likeness** (not the SMARTS reward): v2 ablation (no Vina-in-loop, SMARTS-matching only) kept acryl at 100% but **did not** improve drug-likeness (QED stayed at 0.25, MW still 539). So the Vina component drives QED/MW shifts when the warhead is preserved — the SMARTS reward alone is not enough.
""")
    md.append("Each Tier 4 cohort is a 5,000-sample-per-seed × 21-seed expansion of a "
              "Mol2Mol checkpoint, drawn after (or without) RL fine-tuning on ZAP70 covalent "
              "rewards. Seeds are 21 curated ZAP70 acrylamide actives anchored on Mol-1 "
              "(`C=CC(=O)N1Cc2cccc(C(=O)N(C)C3CCN(C(=O)Nc4ccc(N5CCN(C)CC5)nc4)CC3)c2C1`).\n")
    md.append("All Vina-related metrics (d(Cβ-SG), BD angle, AD-CovDock Vina, hinge-bond, "
              "Cβ-SASA) are deferred — see cost estimate at the bottom. Boltz cofold is "
              "also deferred (per-cohort selection pending).\n")
    md.append("## Side-by-side cohort metrics\n")
    md.append(side_by_side(cohorts))
    md.append("""
> **Brenk-alert footnote.** The Brenk filter's highest-priority hit is the **acrylamide Michael acceptor** itself — which is the *intended* covalent warhead. So Brenk-alert is near-100% for any cohort that successfully kept the warhead (v3, v5), and falls below 10% only when the policy *abandons* the warhead (v4). The functional comparison is "Brenk minus acrylamide": v4's 8.6% is the closest proxy here (its warhead-free leftovers tripped 8.6% on *other* liabilities). Adding an explicit `Brenk-ex-acrylamide` column is queued.
""")
    md.append("\n## Per-cohort details\n")
    for label, name, meta in cohorts:
        md.append(subsection(label, name, meta))
    md.append(vina_block)
    md.append("""
## Triage strategy (before paying for the full Vina sweep)

The full-sweep cost above (~$88, ~60 h on two V100s) is hard to justify for 234K random mols. A two-stage triage is much cheaper:

1. **Stage 1 — cheap filter** (CPU, minutes): per cohort, keep only mols with **Lipinski-ok ∩ FiLMDelta pIC50 ≥ 7.5 ∩ acryl-largest-frag ∩ Murcko-scaffold-dedup**. From the table above:
   - v3 → ≤27.3% × 99.6% acryl × scaffold-dedup ≈ **~4-6K mols**
   - v4 → ≤38.5% × 0.2% × dedup ≈ **<100 mols** (mostly empty by design — most lost the warhead)
   - v5 → ≤85.8% × 98.9% × dedup ≈ **~15-25K mols**
   - v2 (TBD) — likely similar to v3/v4 ballpark.
2. **Stage 2 — Vina + d(Cβ-SG) + BD on the filtered set**: ~20-30K mols → **~3-5 h on one V100, ~$5**. Top-100 from each cohort then proceeds to Boltz cofold for confidence-based final ranking.

This way we never spend more than ~$10 on the next experimental cycle, regardless of how the cohorts grow.

""")

    out_path = T4 / "tier4_report.md"
    out_path.write_text("\n".join(md))
    print(f"  → {out_path}", flush=True)
    print(f"  total length: {sum(len(s) for s in md)} chars", flush=True)


if __name__ == "__main__":
    main()
