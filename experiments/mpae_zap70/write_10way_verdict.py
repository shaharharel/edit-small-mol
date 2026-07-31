#!/usr/bin/env python3
"""Build verdict_10way_report.md from the 10 proxy_metrics JSONs.

Also merges training histories for training-trajectory panels.

Quality gates (from spec):
  validity ≥ 0.55
  dedup ≥ 0.60
  acryl_lf_pct ≥ 0.80
  Tc ∈ [0.35, 0.75]
  planar ≤ 5°  (deg)
  no NaN
  val_CE ≤ ws_ce + 2  (already enforced during training; report if partial ckpt)

Responsiveness gate:
  |Δd_AC| ≥ 0.1 Å   OR
  |Δθ_AC| ≥ 5°     OR
  |Δφ_AC| ≥ 5°     OR
  (V4) |Δ reg_head_pred AC| ≥ 0.5*std(target)  MONOTONE (A→C in same direction).
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path

import numpy as np

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

TARGETS = ["d", "theta", "phi", "mpae", "composite"]
VARIANTS = [3, 4]


def quality_gate(cohort_A: dict, val_ce_final: float | None, ws_ce_thresh: float = 19.0) -> tuple[bool, list[str]]:
    fails = []
    if cohort_A.get("validity_frac", 0) < 0.55:
        fails.append(f"validity={cohort_A.get('validity_frac'):.3f}<0.55")
    if cohort_A.get("dedup_frac", 0) < 0.60:
        fails.append(f"dedup={cohort_A.get('dedup_frac'):.3f}<0.60")
    if cohort_A.get("acryl_largest_frag_pct", 0) < 0.80:
        fails.append(f"acryl={cohort_A.get('acryl_largest_frag_pct'):.3f}<0.80")
    tc = cohort_A.get("tc_to_mol1_median", float("nan"))
    if not (0.35 <= tc <= 0.75):
        fails.append(f"tc={tc:.3f} not in [0.35,0.75]")
    pl = cohort_A.get("planar_dihedral_median_deg", float("nan"))
    if not (pl <= 5.0):
        fails.append(f"planar={pl:.3f}>5")
    for k in ["validity_frac","dedup_frac","acryl_largest_frag_pct",
                "tc_to_mol1_median","planar_dihedral_median_deg",
                "emitted_d_median","emitted_theta_median","emitted_phi_median"]:
        v = cohort_A.get(k)
        if v is None or (isinstance(v, float) and math.isnan(v)):
            fails.append(f"NaN {k}")
    if val_ce_final is not None and val_ce_final > ws_ce_thresh:
        fails.append(f"val_ce_final={val_ce_final:.3f}>{ws_ce_thresh}")
    return (len(fails) == 0), fails


def responsiveness_gate(resp: dict, target: str, target_std: float,
                          reg_on: bool) -> tuple[bool, list[str], dict]:
    """Return (pass, evidence-strings, per-axis-response-dict).

    Monotone A→C direction: we compare AB and AC signs match (both up OR both down).
    """
    d_AC = resp.get("d_shift_AC", float("nan"))
    d_AB = resp.get("d_shift_AB", float("nan"))
    th_AC = resp.get("theta_shift_AC", float("nan"))
    th_AB = resp.get("theta_shift_AB", float("nan"))
    ph_AC = resp.get("phi_shift_AC", float("nan"))
    ph_AB = resp.get("phi_shift_AB", float("nan"))

    def monotone(ab, ac):
        if math.isnan(ab) or math.isnan(ac): return False
        return (ab * ac) >= 0.0

    evidence = []
    axis_hits = {}
    if abs(d_AC) >= 0.1 and monotone(d_AB, d_AC):
        evidence.append(f"|Δd_AC|={abs(d_AC):.3f}Å≥0.1 monotone")
        axis_hits["d"] = d_AC
    if abs(th_AC) >= 5.0 and monotone(th_AB, th_AC):
        evidence.append(f"|Δθ_AC|={abs(th_AC):.3f}°≥5 monotone")
        axis_hits["theta"] = th_AC
    if abs(ph_AC) >= 5.0 and monotone(ph_AB, ph_AC):
        evidence.append(f"|Δφ_AC|={abs(ph_AC):.3f}°≥5 monotone")
        axis_hits["phi"] = ph_AC
    if reg_on and "reg_pred_shift_AC" in resp:
        rg_AC = resp["reg_pred_shift_AC"]
        rg_AB = resp.get("reg_pred_shift_AB", float("nan"))
        thr = 0.5 * target_std
        if abs(rg_AC) >= thr and monotone(rg_AB, rg_AC):
            evidence.append(f"|Δreg_pred_AC|={abs(rg_AC):.3f}≥{thr:.3f}=0.5σ monotone")
            axis_hits["reg_head"] = rg_AC
    return (len(evidence) > 0), evidence, axis_hits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir",
                    default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_zap70"))
    ap.add_argument("--pose_stats",
                    default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_zap70/pose_stats_zap70.json"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    ps = json.loads(Path(args.pose_stats).read_text())
    target_stds = {k: v["std"] for k, v in ps["target_stats"].items()}

    rows = []
    per_axis_hit_counter = {"d": [], "theta": [], "phi": [], "reg_head": []}
    for target in TARGETS:
        for variant in VARIANTS:
            key = f"v{variant}_{target}"
            proxy_path = out_dir / f"proxy_metrics_v{variant}_{target}.json"
            train_log = out_dir / f"train_v{variant}_{target}.jsonl"
            if not proxy_path.exists():
                rows.append({"key": key, "target": target, "variant": variant,
                              "status": "MISSING", "quality_pass": False,
                              "responsive_pass": False,
                              "quality_fails": ["no proxy json"],
                              "resp_evidence": [], "axis_hits": {}})
                continue
            proxy = json.loads(proxy_path.read_text())
            cA = proxy["cohorts"]["A"]
            resp = proxy["clamp_responsiveness"]
            reg_on = proxy["reg_on"]
            v_ce_final = proxy.get("val_ce_final")
            qpass, qfails = quality_gate(cA, v_ce_final)
            std = target_stds.get(target, 1.0)
            rpass, revid, axis_hits = responsiveness_gate(resp, target, std, reg_on)
            row = {"key": key, "target": target, "variant": variant,
                    "quality_pass": qpass, "quality_fails": qfails,
                    "responsive_pass": rpass, "resp_evidence": revid,
                    "axis_hits": axis_hits,
                    "val_ce_final": v_ce_final,
                    "val_infonce_final": proxy.get("val_infonce_final"),
                    "val_regression_spearman": proxy.get("val_regression_spearman"),
                    "cohort_A": cA, "resp": resp}
            rows.append(row)
            for axis in axis_hits:
                per_axis_hit_counter[axis].append(key)

    # Winner logic:
    #   passes BOTH gates AND
    #   the strongest axis_hit is the SAME axis as its training target
    #     (target 'composite' or 'mpae' with V3 pass if ANY axis fires — no
    #      geometric axis is uniquely on-target for these)
    winners = []
    off_target = []
    for r in rows:
        if not (r["quality_pass"] and r["responsive_pass"]):
            continue
        target = r["target"]
        variant = r["variant"]
        hits = r["axis_hits"]
        if target == "composite" and hits:
            winners.append(r)
            continue
        if target == "mpae" and variant == 3 and hits:
            # V3_mpae has no reg_head; any axis firing counts as on-target
            winners.append(r)
            continue
        # Map training-target name → response-axis name
        target_axis = {"d": "d", "theta": "theta", "phi": "phi",
                        "mpae": "reg_head"}.get(target)
        if target_axis in hits:
            winners.append(r)
        else:
            off_target.append(r)

    # Second-order winner: axis with most models firing
    axis_counts = {k: len(v) for k, v in per_axis_hit_counter.items()}
    top_axis = max(axis_counts, key=axis_counts.get) if any(axis_counts.values()) else None

    # ---- Markdown ----
    md = []
    md.append("# ZAP70 10-way Factorial — Verdict Report")
    md.append("")
    md.append(f"**Data**: {ps.get('csv_path','?')}")
    md.append(f"**N_selected** = {ps['n_selected']} (real_pae={ps['n_real_pae']} substitute={ps['n_substitute']})")
    md.append(f"**Pockets** (ZAP70) = {ps['n_unique_pockets']}")
    md.append(f"**mpae source counts** = {ps.get('mpae_source_counts')}")
    md.append("")
    md.append("## 10-model Headline Matrix")
    md.append("")
    md.append("| target | variant | Quality gate | Responsive gate | Best axis hit(s) | Val CE | Val InfoNCE | Val Reg Spr |")
    md.append("|---|:---:|:---:|:---:|---|---:|---:|---:|")
    for r in rows:
        qmark = "PASS" if r["quality_pass"] else "FAIL"
        rmark = "PASS" if r["responsive_pass"] else "fail"
        axes = ", ".join(f"{k}:{v:+.3f}" for k, v in r["axis_hits"].items()) or "-"
        vce = r.get("val_ce_final")
        vnce = r.get("val_infonce_final")
        vspr = r.get("val_regression_spearman")
        def fmt(x): return f"{x:.3f}" if isinstance(x,(int,float)) and x is not None else "-"
        md.append(f"| {r['target']} | V{r['variant']} | {qmark} | {rmark} | {axes} | "
                    f"{fmt(vce)} | {fmt(vnce)} | {fmt(vspr)} |")

    md.append("")
    md.append("### Quality-gate failure reasons")
    for r in rows:
        if not r["quality_pass"]:
            md.append(f"- **v{r['variant']}_{r['target']}**: {', '.join(r['quality_fails']) or '(no reasons captured)'}")

    md.append("")
    md.append("### Clamp-responsiveness evidence (for those that passed)")
    for r in rows:
        if r["responsive_pass"]:
            md.append(f"- **v{r['variant']}_{r['target']}**: {', '.join(r['resp_evidence'])}")

    md.append("")
    md.append("## Per-axis controllability count (second-order winner)")
    md.append("")
    md.append(f"| axis | # models firing | which |")
    md.append(f"|---|---:|---|")
    for k, v in axis_counts.items():
        keys = per_axis_hit_counter[k]
        md.append(f"| {k} | {v} | {', '.join(keys) or '-'} |")
    md.append("")
    if top_axis and axis_counts[top_axis] > 0:
        md.append(f"**Second-order winner axis**: `{top_axis}` — this axis is controllable in {axis_counts[top_axis]} of 10 models.")
    else:
        md.append(f"**Second-order winner axis**: none — no axis was controllable in any model.")

    md.append("")
    md.append("## HEADLINE VERDICT")
    md.append("")
    if winners:
        md.append(f"**{len(winners)} winner(s) pass BOTH gates with on-target axis response:**")
        for w in winners:
            md.append(f"- **v{w['variant']}_{w['target']}** — axis hits: {w['axis_hits']}")
    else:
        md.append("**NO model passes BOTH quality + responsiveness gates on the correct axis.**")
    if off_target:
        md.append("")
        md.append(f"Also passing BOTH gates but on OFF-TARGET axes: {len(off_target)}:")
        for w in off_target:
            md.append(f"- **v{w['variant']}_{w['target']}** — hits on {list(w['axis_hits'].keys())}")

    md.append("")
    md.append("## 3-way panel — this run vs (a) v2-cond baseline vs (b) previous factorial V4")
    md.append("")
    md.append("The previous factorial (BMX/BTK/EGFR/... panel, WRONG target) produced NO passing model.")
    md.append("The v2-cond autoencoder baseline is CE-only, has no pose steering by construction.")
    md.append("")
    md.append("| model | source | best clamp response | val CE | passes |")
    md.append("|---|---|---|---:|:---:|")
    md.append("| v2-cond autoencoder | prior factorial | none (no clamp mech.) | ~17.4 | trivially fails |")
    md.append("| V4-full (prev factorial, BMX/BTK/... panel) | prev factorial | Δθ_AC=1.7° (below gate) | 17.33 | FAIL both |")
    for r in rows:
        best_axis_val = ""
        if r["axis_hits"]:
            k = max(r["axis_hits"], key=lambda kk: abs(r["axis_hits"][kk]))
            best_axis_val = f"{k}={r['axis_hits'][k]:+.3f}"
        else:
            best_axis_val = "none ≥ threshold"
        pass_str = ("PASS" if (r["quality_pass"] and r["responsive_pass"]) else "FAIL")
        vce = r.get("val_ce_final")
        vce_s = f"{vce:.2f}" if isinstance(vce,(int,float)) and vce is not None else "-"
        md.append(f"| v{r['variant']}_{r['target']} (this run, ZAP70 real-PAE) | 10-way | {best_axis_val} | {vce_s} | {pass_str} |")

    # Write report
    report_path = out_dir / "verdict_10way_report.md"
    report_path.write_text("\n".join(md))
    print(f"[write] {report_path}", flush=True)

    # Also save structured JSON
    struct = {
        "rows": rows,
        "winners": [w["key"] for w in winners],
        "off_target": [w["key"] for w in off_target],
        "axis_counts": axis_counts,
        "top_axis": top_axis,
        "data_stats": {"n_selected": ps["n_selected"],
                        "n_real_pae": ps["n_real_pae"],
                        "n_substitute": ps["n_substitute"],
                        "mpae_source_counts": ps.get("mpae_source_counts")},
    }
    (out_dir / "verdict_10way_raw.json").write_text(json.dumps(struct, indent=2, default=str))


if __name__ == "__main__":
    main()
