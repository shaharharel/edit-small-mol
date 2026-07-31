"""Build the 6-way panel table comparing:
    1. v2-cond baseline    (data/paper_dap_repro/report_v2cond_A_baseline.json)
    2. pose-zero            (data/paper_dap_repro/report_pose_zero.json)
    3. pocket-zero          (data/paper_dap_repro/report_pocket_zero.json)
    4. Scheme B             (data/paper_pair_training/scheme_B/report_scheme_B.json  + _clamps.json)
    5. Scheme A             (data/paper_pair_training/scheme_A/report_scheme_A.json  + _pose_clamps.json)
    6. InfoNCE (w=1)         (data/paper_pair_training/infonce/report_infonce_eval1.json + _eval2.json)

Fields shared across all reports:
    n_valid, validity_frac, acryl_largest_frag_pct, planar_dihedral_median_deg,
    tc_to_mol1_median (Scheme A/B/InfoNCE) OR pIC50_median (paper_dap_repro).

Emits:
    data/paper_pair_training/infonce/panel_6way.json     (structured comparison)
    data/paper_pair_training/infonce/panel_6way.md       (markdown table)
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
V100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else V100_ROOT


def load_json(p: Path):
    if not p.exists():
        print(f"[panel] WARN: missing {p}", file=sys.stderr)
        return {}
    return json.loads(p.read_text())


def get_row_v2cond(report: dict, method_name: str) -> dict:
    """Map paper_dap_repro schema → common schema."""
    return {
        "method": method_name,
        "n_raw": report.get("n_raw"),
        "n_valid": report.get("n_valid_raw", report.get("n_cohort")),
        "validity_frac": report.get("validity_frac"),
        "acryl_largest_frag_pct": report.get("acryl_largest_frag_pct"),
        "planar_dihedral_median_deg": None,  # not in DAP repro
        "tc_to_mol1_median": None,
        "pIC50_median": report.get("pIC50_median"),
        "pIC50_p95": report.get("pIC50_p95"),
        "frac_ge_7": report.get("frac_ge_7"),
    }


def get_row_scheme(report: dict, method_name: str) -> dict:
    """Map Scheme A/B/InfoNCE eval1 schema → common schema."""
    return {
        "method": method_name,
        "n_raw": report.get("n_raw"),
        "n_valid": report.get("n_valid"),
        "validity_frac": report.get("validity_frac"),
        "acryl_largest_frag_pct": report.get("acryl_largest_frag_pct"),
        "planar_dihedral_median_deg": report.get("planar_dihedral_median_deg"),
        "tc_to_mol1_median": report.get("tc_to_mol1_median"),
        "pIC50_median": None,
        "pIC50_p95": None,
        "frac_ge_7": None,
    }


def get_causal_from_clamps(clamp_report: dict, cfg_scale: str | None = None) -> dict:
    """Pull headline Spearman + pose-swap Wasserstein from clamp report.

    Scheme A stores it at top-level.  Scheme B stores at top-level.  InfoNCE
    stores per CFG scale (key = str(w)).
    """
    if cfg_scale is None:
        headline = clamp_report.get("headline_spearman", {})
        wass = clamp_report.get("pose_swap_divergence",
                                  clamp_report.get("pose_swap_wasserstein", {}))
    else:
        cs = clamp_report.get("cfg_scale_headlines", {}).get(cfg_scale, {})
        headline = cs.get("headline_spearman", {})
        wass = cs.get("pose_swap_wasserstein", {})
    return {
        "r_d":       headline.get("r_d"),     "p_d": headline.get("p_d"),
        "r_theta":   headline.get("r_theta"), "p_theta": headline.get("p_theta"),
        "r_phi":     headline.get("r_phi"),   "p_phi": headline.get("p_phi"),
        "wasserstein_d":     wass.get("wasserstein_d"),
        "wasserstein_theta": wass.get("wasserstein_theta"),
        "wasserstein_phi":   wass.get("wasserstein_phi"),
    }


def fmt(x, digits=3):
    if x is None:
        return "—"
    if isinstance(x, (int, float)):
        return f"{x:.{digits}f}"
    return str(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", default=str(PROJECT_ROOT /
                     "data/paper_pair_training/infonce"))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    dap = PROJECT_ROOT / "data/paper_dap_repro"
    ppt = PROJECT_ROOT / "data/paper_pair_training"

    # Eval-1-style rows.
    r_baseline = get_row_v2cond(load_json(dap / "report_v2cond_A_baseline.json"),
                                 "v2-cond baseline")
    r_pose_zero = get_row_v2cond(load_json(dap / "report_pose_zero.json"),
                                   "pose-zero")
    r_pocket_zero = get_row_v2cond(load_json(dap / "report_pocket_zero.json"),
                                     "pocket-zero")
    r_schemeB = get_row_scheme(load_json(ppt / "scheme_B/report_scheme_B.json"),
                                 "Scheme B")
    r_schemeA = get_row_scheme(load_json(ppt / "scheme_A/report_scheme_A.json"),
                                 "Scheme A")
    r_infonce = get_row_scheme(load_json(ppt / "infonce/report_infonce_eval1.json"),
                                 "InfoNCE (w=1)")

    # Causal test rows.
    c_schemeB = get_causal_from_clamps(load_json(
        ppt / "scheme_B/report_scheme_B_clamps.json"))
    c_schemeA = get_causal_from_clamps(load_json(
        ppt / "scheme_A/report_scheme_A_pose_clamps.json"))
    infonce_eval2 = load_json(ppt / "infonce/report_infonce_eval2.json")
    c_infonce_w1 = get_causal_from_clamps(infonce_eval2, cfg_scale="1.0")
    c_infonce_w2 = get_causal_from_clamps(infonce_eval2, cfg_scale="2.0")
    c_infonce_w3 = get_causal_from_clamps(infonce_eval2, cfg_scale="3.0")

    rows_eval1 = [r_baseline, r_pose_zero, r_pocket_zero,
                   r_schemeB, r_schemeA, r_infonce]

    payload = {
        "eval1_panel": rows_eval1,
        "causal_test": {
            "Scheme B":      c_schemeB,
            "Scheme A":      c_schemeA,
            "InfoNCE (w=1)": c_infonce_w1,
            "InfoNCE (w=2)": c_infonce_w2,
            "InfoNCE (w=3)": c_infonce_w3,
        },
        "cfg_dose_response": load_json(
            ppt / "infonce/report_infonce_eval3_cfg_dose.json"),
    }
    (out_dir / "panel_6way.json").write_text(json.dumps(payload, indent=2))

    # ---- Markdown table ----
    md = []
    md.append("# Phase 3 InfoNCE — 6-Way Comparison Panel\n")
    md.append("## Eval-1 quality panel (10K samples, real ZAP70 pocket + Mol1 pose)\n")
    md.append("| Method | validity | acryl (largest frag) | planar median (deg) | Tc-to-Mol1 med | pIC50 med | frac ≥7 |")
    md.append("|---|---|---|---|---|---|---|")
    for r in rows_eval1:
        md.append(f"| {r['method']} | "
                   f"{fmt(r['validity_frac'])} | "
                   f"{fmt(r['acryl_largest_frag_pct'])} | "
                   f"{fmt(r['planar_dihedral_median_deg'])} | "
                   f"{fmt(r['tc_to_mol1_median'])} | "
                   f"{fmt(r['pIC50_median'])} | "
                   f"{fmt(r['frac_ge_7'])} |")

    md.append("\n## Causal pose test (pooled Spearman across A/B/C/D/E clamps, "
               "pose-swap Wasserstein)\n")
    md.append("| Method | r_d (n) | r_θ (n) | r_φ (n) | W(d) | W(θ) | W(φ) |")
    md.append("|---|---|---|---|---|---|---|")
    for name, c in payload["causal_test"].items():
        md.append(f"| {name} | "
                   f"{fmt(c['r_d'])} | "
                   f"{fmt(c['r_theta'])} | "
                   f"{fmt(c['r_phi'])} | "
                   f"{fmt(c['wasserstein_d'])} | "
                   f"{fmt(c['wasserstein_theta'])} | "
                   f"{fmt(c['wasserstein_phi'])} |")

    dose = payload["cfg_dose_response"] or {}
    if dose:
        md.append(f"\n## CFG dose-response (clamp {dose.get('clamp', '?')} "
                   f"— {dose.get('label', '?')})\n")
        planar = dose.get("dose_response_planar_median", {})
        md.append("| w | planar median (deg) |")
        md.append("|---|---|")
        for w in sorted(planar.keys(), key=float):
            md.append(f"| {w} | {fmt(planar[w])} |")

    (out_dir / "panel_6way.md").write_text("\n".join(md))
    print(f"Wrote {out_dir / 'panel_6way.json'}")
    print(f"Wrote {out_dir / 'panel_6way.md'}")
    print("\n" + "\n".join(md))


if __name__ == "__main__":
    main()
