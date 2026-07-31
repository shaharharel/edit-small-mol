#!/usr/bin/env python3
"""Compare Lingo3DMol Boltz cofold metrics to the sequence-only ZAP70 baseline.

Reads:
  - data/pocket_fit_comparison/lingo3dmol_boltz_metrics.csv (new)
  - data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions/*/ (baseline)

Writes:
  - data/pocket_fit_comparison/boltz_cofold_comparison.csv
  - /tmp/lingo3dmol_boltz_cofold_results.md  (final report)
"""
from __future__ import annotations
import csv
import json
import sys
from pathlib import Path
from collections import defaultdict
import numpy as np

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
LINGO_CSV = ROOT / "data/pocket_fit_comparison/lingo3dmol_boltz_metrics.csv"
LINGO_SOURCE = ROOT / "data/pocket_fit_comparison/lingo3dmol_all_unique.csv"
TIER1_CSV = ROOT / "data/pocket_fit_comparison/tier1_native_pose_per_mol.csv"
BASE_DIR = ROOT / "data/boltz_poses/boltz_results_top1000__zap70_cys346/predictions"
OUT_DIR = ROOT / "data/pocket_fit_comparison"


def load_baseline_summary():
    """Parse the existing top1000 baseline cofolds for the same metrics."""
    rows = []
    for d in sorted(BASE_DIR.iterdir()):
        if not d.is_dir(): continue
        name = d.name
        conf_p = d / f"confidence_{name}_model_0.json"
        if not conf_p.exists(): continue
        try:
            c = json.loads(conf_p.read_text())
            rows.append({
                "name": name,
                "ligand_iptm": c.get("ligand_iptm"),
                "complex_plddt": c.get("complex_plddt"),
                "confidence_score": c.get("confidence_score"),
            })
        except Exception:
            continue
    return rows


def stats(arr, name):
    arr = [x for x in arr if x is not None and not (isinstance(x, float) and np.isnan(x))]
    if not arr: return f"{name}: N=0"
    a = np.array(arr, dtype=float)
    return f"{name}: N={len(a)} med={np.median(a):.3f} p25={np.percentile(a,25):.3f} p75={np.percentile(a,75):.3f}"


def main():
    # 1) Load new Lingo3DMol metrics + cohort labels
    if not LINGO_CSV.exists():
        print(f"ERROR: missing {LINGO_CSV}", file=sys.stderr); sys.exit(1)
    lingo = list(csv.DictReader(LINGO_CSV.open()))
    src = {r["uid"]: r for r in csv.DictReader(LINGO_SOURCE.open())}
    print(f"Loaded {len(lingo)} Lingo3DMol metrics rows")
    # Map uid -> cohort (use cohort_first)
    for r in lingo:
        r["cohort"] = src.get(r["uid"], {}).get("cohort_first", "unknown")

    # 2) Aggregate per cohort
    by_cohort = defaultdict(list)
    for r in lingo:
        if r["status"] == "ok": by_cohort[r["cohort"]].append(r)

    # 3) Baseline summary
    baseline = load_baseline_summary()
    print(f"Baseline (top1000) cofolds: {len(baseline)}")

    # 4) Write per-cohort summary CSV
    cohort_csv = OUT_DIR / "boltz_cofold_comparison.csv"
    with cohort_csv.open("w") as fh:
        fields = ["cohort","n","d_SG_med","d_SG_p25","d_SG_p75",
                  "BD_med","BD_p25","BD_p75","ligand_iptm_med","complex_plddt_med",
                  "confidence_med","Met414_med","RMSD_med_to_native","gate_dsg_pct","gate_bd_pct"]
        w = csv.DictWriter(fh, fieldnames=fields); w.writeheader()
        for cohort in sorted(by_cohort):
            grp = by_cohort[cohort]
            def col(k): return [float(r[k]) for r in grp if r[k] not in ("",None) and r[k] != "None"]
            d = col("boltz_d_SG"); bd = col("boltz_BD_angle"); li = col("boltz_ligand_iptm")
            pl = col("boltz_complex_plddt"); cs = col("boltz_confidence_score")
            mt = col("Met414_hinge_dist"); rm = col("pose_RMSD_to_native")
            gate_dsg = sum(1 for x in d if x < 2.0)/len(d)*100 if d else None
            gate_bd  = sum(1 for x in bd if 95 <= x <= 115)/len(bd)*100 if bd else None
            w.writerow({
                "cohort": cohort, "n": len(grp),
                "d_SG_med": f"{np.median(d):.3f}" if d else "",
                "d_SG_p25": f"{np.percentile(d,25):.3f}" if d else "",
                "d_SG_p75": f"{np.percentile(d,75):.3f}" if d else "",
                "BD_med": f"{np.median(bd):.1f}" if bd else "",
                "BD_p25": f"{np.percentile(bd,25):.1f}" if bd else "",
                "BD_p75": f"{np.percentile(bd,75):.1f}" if bd else "",
                "ligand_iptm_med": f"{np.median(li):.3f}" if li else "",
                "complex_plddt_med": f"{np.median(pl):.3f}" if pl else "",
                "confidence_med": f"{np.median(cs):.3f}" if cs else "",
                "Met414_med": f"{np.median(mt):.2f}" if mt else "",
                "RMSD_med_to_native": f"{np.median(rm):.2f}" if rm else "",
                "gate_dsg_pct": f"{gate_dsg:.1f}" if gate_dsg is not None else "",
                "gate_bd_pct": f"{gate_bd:.1f}" if gate_bd is not None else "",
            })
    print(f"Wrote {cohort_csv}")

    # 5) Build markdown report
    report_p = Path("/tmp/lingo3dmol_boltz_cofold_results.md")
    n_ok = sum(1 for r in lingo if r["status"] == "ok")
    n_fail = sum(1 for r in lingo if r["status"] != "ok")
    # Pool all metrics across cohorts for grand summary
    all_ok = [r for r in lingo if r["status"] == "ok"]
    def gcol(k): return [float(r[k]) for r in all_ok if r[k] not in ("",None) and r[k] != "None"]
    base_iptm = [b["ligand_iptm"] for b in baseline if b["ligand_iptm"] is not None]
    base_plddt = [b["complex_plddt"] for b in baseline if b["complex_plddt"] is not None]
    base_conf = [b["confidence_score"] for b in baseline if b["confidence_score"] is not None]

    md = [
        "# Lingo3DMol Boltz Cofold Results (ZAP70 Cys346)",
        f"\nN unique mols cofolded: {n_ok}/401  (fail/missing: {n_fail})",
        "\n## Per-cohort summary",
        "",
        "| Cohort | N | d_SG med | BD med | lig_iptm | plddt | gate_dsg(<2Å) | gate_bd(95-115°) |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for cohort in sorted(by_cohort):
        grp = by_cohort[cohort]
        def col(k): return [float(r[k]) for r in grp if r[k] not in ("",None) and r[k] != "None"]
        d = col("boltz_d_SG"); bd = col("boltz_BD_angle"); li = col("boltz_ligand_iptm"); pl = col("boltz_complex_plddt")
        gd = f"{sum(1 for x in d if x<2.0)/len(d)*100:.0f}" if d else "-"
        gb = f"{sum(1 for x in bd if 95<=x<=115)/len(bd)*100:.0f}" if bd else "-"
        md.append(f"| {cohort} | {len(grp)} | {np.median(d):.2f} | {np.median(bd):.0f} | {np.median(li):.3f} | {np.median(pl):.3f} | {gd}% | {gb}% |" if (d and bd and li and pl) else f"| {cohort} | {len(grp)} | - | - | - | - | - | - |")

    md += [
        "\n## Comparison to sequence-only baseline (top1000)",
        f"\nBaseline (top1000 high-affinity Boltz cofolds, N={len(baseline)}):",
        f"  - ligand_iptm: median={np.median(base_iptm):.3f} p25={np.percentile(base_iptm,25):.3f} p75={np.percentile(base_iptm,75):.3f}" if base_iptm else "  - ligand_iptm: N/A",
        f"  - complex_plddt: median={np.median(base_plddt):.3f}" if base_plddt else "",
        f"  - confidence_score: median={np.median(base_conf):.3f}" if base_conf else "",
        "",
        "Lingo3DMol cofolds (this run):",
    ]
    all_li = gcol("boltz_ligand_iptm")
    all_pl = gcol("boltz_complex_plddt")
    all_cs = gcol("boltz_confidence_score")
    all_d = gcol("boltz_d_SG"); all_bd = gcol("boltz_BD_angle"); all_rm = gcol("pose_RMSD_to_native")
    if all_li: md.append(f"  - ligand_iptm: median={np.median(all_li):.3f} p25={np.percentile(all_li,25):.3f} p75={np.percentile(all_li,75):.3f}")
    if all_pl: md.append(f"  - complex_plddt: median={np.median(all_pl):.3f}")
    if all_cs: md.append(f"  - confidence_score: median={np.median(all_cs):.3f}")
    if all_d: md.append(f"  - d_SG (Boltz post-bond): median={np.median(all_d):.2f} Å")
    if all_bd: md.append(f"  - BD angle: median={np.median(all_bd):.1f}°")
    if all_rm: md.append(f"  - RMSD to Lingo3DMol native pose: N={len(all_rm)} median={np.median(all_rm):.2f} Å")

    # Read run wall+cost from progress.json (will be filled in by report builder)
    md += [
        "\n## Run info",
        "- Instance: ai-gpu-a100 (us-central1-a) — A100 40GB SPOT",
        "- N_PARALLEL=2",
        f"- Cofolds completed: {n_ok}/401",
        "- A100 STOPPED at end of run (TERMINATED)",
    ]
    report_p.write_text("\n".join(md))
    print(f"Wrote {report_p}")


if __name__ == "__main__":
    main()
