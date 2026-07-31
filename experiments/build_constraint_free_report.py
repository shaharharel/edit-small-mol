#!/usr/bin/env python3
"""Assemble the constraint_free_eval.md report from A + B + C panel outputs.

Reads:
  - Exp A: <root>/noconstraint_summary.json  (Boltz-no-constraint analysis)
  - Exp B: <root>/etkdg_strain_ks.json + etkdg_strain_panel.csv
  - Exp C: <root>/cohort_chemistry_summary.json

Writes:
  - <root>/constraint_free_eval.md

Usage:
    python experiments/build_constraint_free_report.py \\
        --root data/paper_pair_training/v2_curriculum_clean
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd


COHORTS = ["theta_90", "theta_105", "theta_130", "null_pose"]


def _fmt_ks_table(ks_dict: dict, metric_order: list[str]) -> str:
    """Render KS table as markdown, showing all cohort pairs per metric."""
    if not ks_dict:
        return "_(no KS data)_"
    all_pairs = set()
    for m in metric_order:
        if m in ks_dict:
            all_pairs.update(ks_dict[m].keys())
    all_pairs = sorted(all_pairs)
    lines = [f"| metric | " + " | ".join(all_pairs) + " |",
             "|" + "---|" * (len(all_pairs) + 1)]
    for m in metric_order:
        if m not in ks_dict:
            continue
        cells = [m]
        for p in all_pairs:
            r = ks_dict[m].get(p, {})
            if r.get("p") is None:
                cells.append("—")
                continue
            star = "**" if r["p"] < 0.05 else ""
            cells.append(f"{star}p={r['p']:.3g}{star}<br>D={r['stat']:.2f}")
        lines.append("| " + " | ".join(cells) + " |")
    return "\n".join(lines)


def _experiment_a(root: Path) -> str:
    js = root / "noconstraint_summary.json"
    if not js.exists():
        return ("### Experiment A — Boltz without covalent constraint\n\n"
                "_(cofold summary not yet available: `noconstraint_summary.json` missing)_\n\n")
    data = json.loads(js.read_text())
    lines = ["### Experiment A — Boltz WITHOUT covalent constraint\n"]
    lines.append("**Setup**: identical v1_clean samples (100 per cohort after "
                 "acryl-on-largest-frag filter), same ZAP70 pocket / Mol1 anchor, "
                 "but the YAML `constraints` block is removed. Boltz picks a non-"
                 "covalent pose freely; the warhead β-C is free to be anywhere.\n")
    lines.append("**Headline metric**: `f_BD_ready` = fraction of cofolds with "
                 f"`d_β-Sγ ∈ [{data['config']['d_window'][0]}, {data['config']['d_window'][1]}] Å` "
                 f"AND `θ_BD ∈ [{data['config']['bd_window'][0]}, {data['config']['bd_window'][1]}] °` "
                 "(Bürgi-Dunitz productive window). Bootstrap 95 % CI over 1000 resamples.\n")
    lines.append(f"**Loose metric**: `f_d<{data['config']['d_loose_max']}A` = fraction with "
                 f"`d_β-Sγ ≤ {data['config']['d_loose_max']} Å` (any angle). Distinguishes "
                 "cofolds where the warhead just landed near the pocket at all.\n")
    lines.append("#### Per-cohort summary — NO CONSTRAINT (Boltz free pose)\n")
    lines.append("| cohort | n_ok | median d (Å) | median θ_BD (°) | median iptm | f_BD_ready [95% CI] | f_d≤7Å [95% CI] |")
    lines.append("|---|---|---|---|---|---|---|")
    for c in COHORTS:
        s = data["per_cohort"].get(c, {})
        n_ok = s.get("n_ok", 0)
        d = s.get("d_b_nuc_angstrom", {}).get("median")
        bd = s.get("bd_angle_deg", {}).get("median")
        ip = s.get("complex_iptm", {}).get("median")
        fr = data["f_BD_ready"].get(c, {})
        val = fr.get("value")
        lo = fr.get("ci95_lo")
        hi = fr.get("ci95_hi")
        cell = (f"{val:.3f} [{lo:.3f}, {hi:.3f}]"
                if val is not None and not (isinstance(val, float) and np.isnan(val))
                else "—")
        dw = data.get("f_d_within_7A", {}).get(c, {})
        dw_v = dw.get("value")
        dw_l = dw.get("ci95_lo")
        dw_h = dw.get("ci95_hi")
        cell2 = (f"{dw_v:.3f} [{dw_l:.3f}, {dw_h:.3f}]"
                 if dw_v is not None and not (isinstance(dw_v, float) and np.isnan(dw_v))
                 else "—")
        lines.append(f"| {c} | {n_ok} | "
                     f"{d:.2f} | {bd:.1f} | {ip:.3f} | {cell} | {cell2} |"
                     if d is not None else f"| {c} | {n_ok} | — | — | — | {cell} | {cell2} |")
    lines.append("")
    # Constrained baseline comparison
    if "constrained_baseline" in data and data["constrained_baseline"]:
        lines.append("#### Constrained baseline (for comparison — Boltz WITH `bond` "
                     "constraint enforces d→1.8 Å)\n")
        lines.append("| cohort | n_ok | median d (Å) | median θ_BD (°) | median iptm |")
        lines.append("|---|---|---|---|---|")
        for c in COHORTS:
            b = data["constrained_baseline"].get(c, {})
            if not b:
                lines.append(f"| {c} | — | — | — | — |")
                continue
            n_ok = b.get("n_ok", 0)
            d = b.get("d_b_nuc_angstrom", {}).get("median")
            bd = b.get("bd_angle_deg", {}).get("median")
            ip = b.get("complex_iptm", {}).get("median")
            lines.append(f"| {c} | {n_ok} | {d:.2f} | {bd:.1f} | {ip:.3f} |"
                         if d is not None else f"| {c} | {n_ok} | — | — | — |")
        lines.append("")
        lines.append("_Thermostat effect: with covalent constraint, "
                     "d clamps to ~1.9 Å across every cohort (variance destroyed). "
                     "Non-constrained d distribution — shown above — is what the "
                     "model 'wants' the warhead to do._\n")
    lines.append("#### KS tests (Kolmogorov-Smirnov, p<0.05 marked **)\n")
    metric_order = ["d_b_nuc_angstrom", "bd_angle_deg", "phi_planar_deg",
                    "complex_iptm", "ligand_iptm", "complex_plddt",
                    "mpae_prot_lig_min", "f_BD_ready_bernoulli"]
    lines.append(_fmt_ks_table(data.get("ks", {}), metric_order))
    lines.append("")
    return "\n".join(lines)


def _experiment_b(root: Path) -> str:
    ks_json = root / "etkdg_strain_ks.json"
    csv = root / "etkdg_strain_panel.csv"
    if not ks_json.exists() or not csv.exists():
        return ("### Experiment B — ETKDG conformational strain\n\n"
                "_(ETKDG panel not yet available)_\n\n")
    ks = json.loads(ks_json.read_text())
    df = pd.read_csv(csv)
    ok = df[df["err"].fillna("") == ""]
    lines = ["### Experiment B — ETKDG intrinsic-conformer strain panel\n"]
    lines.append("**Setup**: 50 ETKDG v3 conformers per SMILES, MMFF94s-minimized. "
                 "Purely Boltz-free CPU analysis of the ligand's *intrinsic* "
                 "conformational preference. If v1_clean has learned to produce "
                 "molecules whose conformer ensemble is more BD-competent under one "
                 "pose condition, that shows up here.\n")
    lines.append("#### Per-cohort medians\n")
    lines.append("| cohort | n_ok | median f_reactive_rotamer | median strain (kcal/mol) | median d_β-centroid (Å) | median \|φ_vinyl\| (°) |")
    lines.append("|---|---|---|---|---|---|")
    for c in COHORTS:
        sub = ok[ok["cohort"] == c]
        lines.append(f"| {c} | {len(sub)} | "
                     f"{sub['f_reactive_rotamer'].median():.3f} | "
                     f"{sub['median_strain_kcal'].median():.2f} | "
                     f"{sub['median_d_beta_centroid_A'].median():.2f} | "
                     f"{sub['median_abs_phi_vinyl_deg'].median():.2f} |")
    lines.append("")
    lines.append("#### KS tests\n")
    lines.append(_fmt_ks_table(ks, ["f_reactive_rotamer", "median_strain_kcal",
                                      "median_d_beta_centroid_A",
                                      "median_abs_phi_vinyl_deg"]))
    lines.append("")
    return "\n".join(lines)


def _experiment_c(root: Path) -> str:
    js = root / "cohort_chemistry_summary.json"
    if not js.exists():
        return "### Experiment C — Chemistry differences\n\n_(chemistry summary not available)_\n\n"
    data = json.loads(js.read_text())
    lines = ["### Experiment C — Chemistry differences across cohorts\n"]
    lines.append("**Setup**: pure SMILES topology, no docking or pose. "
                 "Are the cohorts even chemically comparable, or is v1_clean producing "
                 "different molecule *populations* per pose target?\n")
    lines.append("#### Physchem medians\n")
    lines.append("| cohort | MW | logP | TPSA | RotB | HBA | HBD | QED |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for c in COHORTS:
        pc = data["per_cohort"].get(c, {})
        lines.append(f"| {c} | "
                     f"{pc.get('mw',{}).get('median',0):.1f} | "
                     f"{pc.get('logp',{}).get('median',0):.2f} | "
                     f"{pc.get('tpsa',{}).get('median',0):.1f} | "
                     f"{pc.get('rotb',{}).get('median',0):.1f} | "
                     f"{pc.get('hba',{}).get('median',0):.1f} | "
                     f"{pc.get('hbd',{}).get('median',0):.1f} | "
                     f"{pc.get('qed',{}).get('median',0):.3f} |")
    lines.append("")
    lines.append("#### KS on physchem (significant pairs only, p<0.05)\n")
    sig = []
    for m, results in data["ks_vs_theta_90"].items():
        for k, v in results.items():
            if v.get("p") is not None and v["p"] < 0.05:
                sig.append((m, k, v["stat"], v["p"]))
    if not sig:
        lines.append("_None_.")
    else:
        lines.append("| metric | pair | KS stat | p |")
        lines.append("|---|---|---|---|")
        for m, k, s, p in sig:
            lines.append(f"| {m} | {k} | {s:.3f} | {p:.3g} |")
    lines.append("")
    lines.append("#### Scaffold Jaccard (0=disjoint scaffolds; 1=identical)\n")
    lines.append("| pair | Jaccard |")
    lines.append("|---|---|")
    for k, v in data["scaffold_jaccard"].items():
        if v is not None:
            lines.append(f"| {k} | {v:.3f} |")
    lines.append("")
    lines.append("#### Nearest-neighbor Tanimoto (Morgan r=2, 2048)\n")
    lines.append("| cohort | within median | across (theta_90) | across (theta_105) | across (theta_130) | across (null_pose) |")
    lines.append("|---|---|---|---|---|---|")
    for c in COHORTS:
        row = data["nn_tanimoto"][c]
        w = row["within_median"]
        cells = [f"{w:.3f}"]
        for o in COHORTS:
            if o == c:
                cells.append("—")
                continue
            cells.append(f"{row['across'][o]['median']:.3f}")
        lines.append(f"| {c} | " + " | ".join(cells) + " |")
    lines.append("")
    return "\n".join(lines)


def _verdict(root: Path) -> str:
    """Compose the verdict from all available results."""
    lines = ["## Verdict\n"]
    signals = []
    # A
    js_a = root / "noconstraint_summary.json"
    if js_a.exists():
        data = json.loads(js_a.read_text())
        ks_all = data.get("ks", {})
        for m, results in ks_all.items():
            for k, v in results.items():
                if v.get("p") is not None and v["p"] < 0.05:
                    signals.append(("A_boltz_noconstr", m, k, v["stat"], v["p"]))
    # B
    ks_b_json = root / "etkdg_strain_ks.json"
    if ks_b_json.exists():
        ks_b = json.loads(ks_b_json.read_text())
        for m, results in ks_b.items():
            for k, v in results.items():
                if v.get("p") is not None and v["p"] < 0.05:
                    signals.append(("B_etkdg", m, k, v["stat"], v["p"]))
    # C
    js_c = root / "cohort_chemistry_summary.json"
    if js_c.exists():
        data = json.loads(js_c.read_text())
        for m, results in data["ks_vs_theta_90"].items():
            for k, v in results.items():
                if v.get("p") is not None and v["p"] < 0.05:
                    signals.append(("C_chem", m, k, v["stat"], v["p"]))
    if not signals:
        lines.append("**No metric reached p<0.05 KS across any cohort pair in "
                     "any experiment.** v1_clean does not steer pose-conditioned "
                     "response under Boltz-no-constraint, ETKDG-strain, or "
                     "chemistry-topology regimes.")
    else:
        lines.append(f"**{len(signals)} KS tests reached p<0.05** across "
                     "experiments A, B, C. Detail table:\n")
        lines.append("| experiment | metric | pair | KS stat | p |")
        lines.append("|---|---|---|---|---|")
        for e, m, k, s, p in sorted(signals, key=lambda r: r[4]):
            lines.append(f"| {e} | {m} | {k} | {s:.3f} | {p:.3g} |")
        # Break down by pose-condition pair
        pose_conditioned = [s for s in signals if "null_pose" not in s[2]]
        vs_null = [s for s in signals if "null_pose" in s[2]]
        lines.append(f"\n- **Pose-conditioned** (θ_90/105/130 vs each other): "
                     f"{len(pose_conditioned)} significant tests.")
        lines.append(f"- **vs null_pose**: {len(vs_null)} significant tests.")
        if pose_conditioned:
            lines.append("\n**Paper claim supported**: v1_clean shows measurable "
                         "pose-conditioned response on at least one metric under "
                         "constraint-free evaluation.")
        else:
            lines.append("\n**Weaker claim**: v1_clean distinguishes conditioned "
                         "vs unconditioned generation (null_pose baseline), but "
                         "does NOT differentiate between pose targets (90 vs 105 "
                         "vs 130). Consistent with the thermostat interpretation "
                         "that the Boltz covalent constraint hides θ-specific "
                         "signals — but v1_clean is not producing θ-specific "
                         "molecules to begin with.")
    lines.append("")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True)
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()

    root = Path(args.root)
    out_md = Path(args.out_md) if args.out_md else root / "constraint_free_eval.md"

    parts = ["# Constraint-Free Evaluation of v2_curriculum_clean (v1_clean)\n",
             "**Date**: 2026-07-13. **Anchor**: Mol1 (`C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1`). "
             "**Pocket**: ZAP70 Cys346. **Cohorts**: θ_90 / θ_105 / θ_130 / null_pose "
             "(100 acryl-on-largest-frag SMILES each).\n",
             "**Hypothesis**: Prior Boltz-with-covalent-constraint comparison showed "
             "v1_clean does NOT steer `θ_observed` (KS p=0.339 at N=106 vs 100). "
             "The covalent bond constraint acts as a thermostat, forcing d→1.8 Å "
             "regardless of what SMILES suggests. **Does v1_clean show pose-"
             "conditioned response under evaluation regimes that DON'T normalize "
             "the geometry?**\n",
             _experiment_a(root),
             _experiment_b(root),
             _experiment_c(root),
             _verdict(root),
             "---\n",
             "**Files consumed:**\n"
             f"- Experiment A: `{root.relative_to(root.parent.parent.parent) if False else root}/noconstraint_summary.json` "
             f"+ per-cohort track_A_v2curr_clean_NOCONSTR_*.csv\n"
             f"- Experiment B: `etkdg_strain_panel.csv` + `etkdg_strain_ks.json`\n"
             f"- Experiment C: `cohort_chemistry_panel.csv` + `cohort_chemistry_summary.json`\n"]
    out_md.write_text("\n".join(parts))
    print(f"[report] wrote {out_md}")


if __name__ == "__main__":
    main()
