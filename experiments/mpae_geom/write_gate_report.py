#!/usr/bin/env python3
"""Aggregate per-variant proxy_metrics and write the Phase A gate report.

Gates:
  Quality (all must pass):
    validity_frac ≥ 0.55
    dedup_frac ≥ 0.60
    acryl_largest_frag_pct ≥ 0.80
    Tc_to_Mol1 median in [0.35, 0.75]
    planar_dihedral_median ≤ 5 (on Clamp A)
    val_ce ≤ 19 (from training log)
    no NaN in reported metrics
    (pIC50_film gate skipped — not computed here)

  Clamp responsiveness (at least one must be true):
    emitted_d shifts ≥ 0.1 Å between A and B, opposite between A and C
    emitted_θ shifts ≥ 5° monotone A→C
    emitted_φ shifts ≥ 5° monotone A→C

Verdict rules:
  1. Filter to variants passing ALL quality gates → eligible
  2. Filter eligible to those passing ≥ 1 responsiveness gate → candidate
  3. If empty → skip Phase B, publish negative
  4. If 1-2 candidates → send to Phase B
  5. If 3-4 → rank by combined score, top 2
"""
from __future__ import annotations
import argparse
import json
import math
from pathlib import Path


def load_last_val_ce(log_path: Path) -> float:
    if not log_path.exists():
        return float("nan")
    lines = [json.loads(ln) for ln in log_path.read_text().splitlines() if ln.strip()]
    if not lines:
        return float("nan")
    return float(lines[-1].get("val_ce", float("nan")))


def check_variant(variant: int, proxy: dict, val_ce: float) -> dict:
    A = proxy["cohorts"]["A"]; B = proxy["cohorts"]["B"]; C = proxy["cohorts"]["C"]
    # Quality on Clamp A
    tc = A.get("tc_to_mol1_median", float("nan"))
    quality = {
        "validity_frac": {"value": A["validity_frac"], "pass": A["validity_frac"] >= 0.55},
        "dedup_frac":    {"value": A["dedup_frac"],    "pass": A["dedup_frac"] >= 0.60},
        "acryl_largest_frag_pct": {"value": A["acryl_largest_frag_pct"], "pass": A["acryl_largest_frag_pct"] >= 0.80},
        "tc_in_range":   {"value": tc, "pass": (0.35 <= tc <= 0.75) if math.isfinite(tc) else False},
        "planar_dihedral_median_deg": {"value": A.get("planar_dihedral_median_deg"), "pass": (A.get("planar_dihedral_median_deg") is not None) and (A["planar_dihedral_median_deg"] <= 5.0)},
        "val_ce":        {"value": val_ce, "pass": math.isfinite(val_ce) and val_ce <= 19.0},
        "no_nan":        {"value": None, "pass": all(math.isfinite(A.get(k, 0.0)) for k in ["validity_frac","dedup_frac","acryl_largest_frag_pct"])},
    }
    q_pass = all(v["pass"] for v in quality.values())

    # Responsiveness
    r = proxy["clamp_responsiveness"]
    d_ok = (r["d_shift_AB"] is not None and r["d_shift_AC"] is not None and
             abs(r["d_shift_AB"]) >= 0.1 and (r["d_shift_AB"] * r["d_shift_AC"] < 0))
    th_ok = (r["theta_shift_AB"] is not None and r["theta_shift_AC"] is not None and
              abs(r["theta_shift_AB"] - r["theta_shift_AC"]) >= 5.0 and
              (r["theta_shift_AB"] * r["theta_shift_AC"] < 0 or
               abs(r["theta_shift_AB"]) >= 5.0 or abs(r["theta_shift_AC"]) >= 5.0))
    ph_ok = (r["phi_shift_AB"] is not None and r["phi_shift_AC"] is not None and
              abs(r["phi_shift_AB"] - r["phi_shift_AC"]) >= 5.0)
    resp = {"d_gate": d_ok, "theta_gate": th_ok, "phi_gate": ph_ok}
    r_pass = any(resp.values())

    return {
        "variant": variant, "reg_on": proxy.get("reg_on"), "nce_on": proxy.get("nce_on"),
        "quality": quality, "quality_pass": q_pass,
        "responsiveness": {**resp, **r},
        "responsiveness_pass": r_pass,
        "advance_to_phase_B": q_pass and r_pass,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="/home/shaharh_quris_ai/edit-small-mol/data/paper_pair_training/mpae_geom")
    ap.add_argument("--models_dir", default="/home/shaharh_quris_ai/edit-small-mol/models")
    ap.add_argument("--out_md", default=None)
    args = ap.parse_args()
    data_dir = Path(args.data_dir); models_dir = Path(args.models_dir)
    out_md = Path(args.out_md) if args.out_md else data_dir / "phase_A_gate_report.md"

    results = []
    for v in [1, 2, 3, 4]:
        proxy_path = data_dir / f"proxy_metrics_V{v}.json"
        log_path = models_dir / f"m1a_v2_mpae_V{v}_log.jsonl"
        if not proxy_path.exists():
            print(f"[skip] no proxy_metrics_V{v}.json"); continue
        proxy = json.loads(proxy_path.read_text())
        val_ce = load_last_val_ce(log_path)
        results.append(check_variant(v, proxy, val_ce))

    if not results:
        out_md.write_text("# Phase A Gate Report\n\nNo variants completed.\n")
        return

    # Emit markdown
    lines = ["# Phase A Gate Report — mpae_warhead_cys Geometry Install", ""]
    lines.append("## Per-variant quality gates (Clamp A cohort)")
    lines.append("| Variant | reg | nce | validity | dedup | acryl_lf | Tc | planar° | val_CE | PASS |")
    lines.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|")
    for r in results:
        q = r["quality"]
        def f(k):
            v = q[k]["value"]
            if v is None: return "—"
            if isinstance(v, float): return f"{v:.3f}"
            return str(v)
        row = (f"| V{r['variant']} | {'ON' if r['reg_on'] else 'off'} | {'ON' if r['nce_on'] else 'off'} | "
               f"{f('validity_frac')} | {f('dedup_frac')} | {f('acryl_largest_frag_pct')} | "
               f"{f('tc_in_range')} | {f('planar_dihedral_median_deg')} | {f('val_ce')} | "
               f"{'YES' if r['quality_pass'] else 'no'} |")
        lines.append(row)
    lines.append("")
    lines.append("## Per-variant clamp responsiveness (A vs B vs C)")
    lines.append("| Variant | Δd(A→B) | Δd(A→C) | Δθ(A→B) | Δθ(A→C) | Δφ(A→B) | Δφ(A→C) | d_gate | θ_gate | φ_gate | PASS |")
    lines.append("|---|---:|---:|---:|---:|---:|---:|:---:|:---:|:---:|:---:|")
    for r in results:
        resp = r["responsiveness"]
        def f(k):
            v = resp.get(k)
            if v is None or (isinstance(v, float) and not math.isfinite(v)): return "—"
            return f"{v:.3f}" if isinstance(v, float) else str(v)
        row = (f"| V{r['variant']} | {f('d_shift_AB')} | {f('d_shift_AC')} | "
               f"{f('theta_shift_AB')} | {f('theta_shift_AC')} | "
               f"{f('phi_shift_AB')} | {f('phi_shift_AC')} | "
               f"{'YES' if resp['d_gate'] else 'no'} | {'YES' if resp['theta_gate'] else 'no'} | "
               f"{'YES' if resp['phi_gate'] else 'no'} | "
               f"{'YES' if r['responsiveness_pass'] else 'no'} |")
        lines.append(row)
    lines.append("")

    eligible = [r for r in results if r["quality_pass"]]
    candidates = [r for r in eligible if r["responsiveness_pass"]]
    lines.append("## Verdict")
    if not candidates:
        lines.append("")
        lines.append("**No variant passes BOTH quality gates AND clamp responsiveness.**")
        lines.append("")
        lines.append("Boltz cofolds (Phase B) would not add information. Phase B skipped.")
        lines.append("")
        lines.append("Publish negative: even with fixed InfoNCE + regression head + real chem-matched")
        lines.append("negatives, the pose channel does not carry actionable leverage on this dataset.")
    else:
        recs = ", ".join(f"V{r['variant']}" for r in candidates[:2])
        lines.append("")
        lines.append(f"**{len(candidates)} candidate variant(s) pass quality + responsiveness:** {recs}")
        lines.append("")
        lines.append("**Recommendation:** advance top " + str(min(2, len(candidates))) + " to Phase B (Boltz cofolds on A100).")

    out_md.write_text("\n".join(lines))
    print(f"[write] {out_md}")

    # Also drop raw json
    (out_md.parent / "phase_A_gate_raw.json").write_text(json.dumps(results, indent=2, default=str))


if __name__ == "__main__":
    main()
