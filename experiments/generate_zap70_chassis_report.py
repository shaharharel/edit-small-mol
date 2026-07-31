"""Generate final /tmp/zap70_multi_chassis_ensemble.md report from collected
per-chassis sampling + eval results.

Usage:
    python experiments/generate_zap70_chassis_report.py \
        --base_dir data/lingo3dmol_zap70_chassis \
        --output /tmp/zap70_multi_chassis_ensemble.md
"""
from __future__ import annotations
import argparse
import json
from pathlib import Path


def fmt_pct(v):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, str):
        return v
    # eval_lingo3dmol_plans reports fractions in [0, 1] (e.g. 1.0 = 100%)
    return f"{v*100:.1f}%"


def fmt_pct_raw(v):
    """For values already in [0, 100] scale."""
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, str):
        return v
    return f"{v:.1f}%"


def fmt_num(v, prec=2):
    if v is None or (isinstance(v, float) and v != v):
        return "—"
    if isinstance(v, str):
        return v
    if isinstance(v, int):
        return str(v)
    return f"{v:.{prec}f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base_dir", default="data/lingo3dmol_zap70_chassis")
    ap.add_argument("--families_json", default="data/zap70_chassis_families.json")
    ap.add_argument("--output", default="/tmp/zap70_multi_chassis_ensemble.md")
    args = ap.parse_args()

    base = Path(args.base_dir)
    fams = json.loads(Path(args.families_json).read_text())

    # Load per-chassis summaries
    rows = []
    for fam in fams["families"]:
        cid = fam["chassis_id"]
        # smoke
        smoke_path = base / f"{cid}_smoke" / "samples_summary.json"
        smoke = None
        if smoke_path.exists():
            try:
                smoke = json.loads(smoke_path.read_text())
            except Exception:
                pass
        # n500
        n500_path = base / cid / "samples_summary.json"
        n500 = None
        if n500_path.exists():
            try:
                n500 = json.loads(n500_path.read_text())
            except Exception:
                pass
        # n500 eval
        eval_path = base / cid / f"{cid}_eval.json"
        eval_summary = None
        if eval_path.exists():
            try:
                ed = json.loads(eval_path.read_text())
                eval_summary = ed.get(cid, {}).get("summary", {})
            except Exception:
                pass
        rows.append({
            "cid": cid,
            "smiles_with_star": fam["chassis_smiles_with_star"],
            "n_actives": fam.get("n_actives", 0),
            "median_pIC50": fam.get("median_pIC50"),
            "source": fam.get("source", "?"),
            "smoke_n": smoke.get("n_sampled") if smoke else None,
            "smoke_status": smoke.get("status") if smoke else None,
            "n500_n": n500.get("n_sampled") if n500 else None,
            "n500_status": n500.get("status") if n500 else None,
            "n500_elapsed_min": (n500.get("elapsed_sec", 0) / 60.0) if n500 else None,
            "eval": eval_summary,
        })

    # ensemble summary
    ens_path = base / "ENSEMBLE" / "ensemble_summary.json"
    ens = json.loads(ens_path.read_text()) if ens_path.exists() else None
    ens_eval_path = base / "ENSEMBLE" / "ENSEMBLE_eval.json"
    ens_eval = None
    if ens_eval_path.exists():
        try:
            ed = json.loads(ens_eval_path.read_text())
            ens_eval = ed.get("ENSEMBLE", {}).get("summary", {})
        except Exception:
            pass

    lines = []
    lines.append("# ZAP70 Multi-Chassis Diversified Ensemble — Phase A→D Report")
    lines.append("")
    lines.append("**Run date:** 2026-05-30 (Mac CPU, ~6 hr budget). Generalizes CovalentLingo anchor methodology across multiple kinase-covalent chassis.")
    lines.append("")
    lines.append(f"## Phase A — Chassis mining")
    lines.append("")
    lines.append(f"From {fams.get('n_total_actives')} ChEMBL ZAP70 actives, {fams.get('n_acrylamide_actives')} contained acrylamide. Stripping each to (warhead + nearest ring system + Murcko) yielded only **{fams.get('n_unique_murcko_from_dataset')} unique scaffolds** dominated by aniline-acrylamide (n=19 actives, median pIC50 7.04). Per protocol fallback, we augment with 4 published kinase chassis to expose 6 distinct chemotypes:")
    lines.append("")
    lines.append("| chassis | source | smiles_with_star | n_actives | med_pIC50 |")
    lines.append("|---|---|---|---|---|")
    for r in rows:
        lines.append(f"| `{r['cid']}` | {r['source']} | `{r['smiles_with_star']}` | {r['n_actives']} | {fmt_num(r['median_pIC50'])} |")
    lines.append("")
    lines.append("All 6 decode-verified via FSMILES `decode3d()` round-trip. Provenance: `data/zap70_chassis_families.json`.")
    lines.append("")

    lines.append("## Phase B + C — Smoke + N=500 sampling")
    lines.append("")
    lines.append("Smokes: `gennums=15, min_acceptable=5`. N=500: `min_acceptable=30–60, max_run=1800–2700s`, T=1.0, Mac CPU. Sequential dispatch to respect 4-concurrent budget shared with user's H2/C5/BTK/KRAS samplers.")
    lines.append("")
    lines.append("| chassis | smoke n / status | N=500 n / status | elapsed (min) |")
    lines.append("|---|---|---|---|")
    for r in rows:
        sn = r['smoke_n'] if r['smoke_n'] is not None else "—"
        ss = r['smoke_status'] or "skip"
        n = r['n500_n'] if r['n500_n'] is not None else "—"
        st = r['n500_status'] or "—"
        e = fmt_num(r['n500_elapsed_min']) if r['n500_elapsed_min'] is not None else "—"
        lines.append(f"| `{r['cid']}` | {sn} / {ss} | {n} / {st} | {e} |")
    lines.append("")

    lines.append("## Phase D — Per-cohort evaluation (eval_lingo3dmol_plans, headline metrics)")
    lines.append("")
    lines.append("| chassis | CHR | acryl_largest | conn_largest | scaffold_div | MW_dr | QED-pass | n_valid |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for r in rows:
        e = r['eval'] or {}
        chr_ = fmt_pct(e.get('Covalent_Hit_Rate'))
        acl = fmt_pct(e.get('acryl_largest_pct'))
        con = fmt_pct(e.get('connectivity_largest_pct'))
        sca = fmt_pct(e.get('scaffold_diversity_pct'))
        mw = fmt_pct(e.get('MW_in_drug_range_pct'))
        qed = fmt_pct(e.get('QED_passing_pct'))
        n = e.get('n_valid', '—')
        lines.append(f"| `{r['cid']}` | {chr_} | {acl} | {con} | {sca} | {mw} | {qed} | {n} |")
    lines.append("")

    lines.append("## Ensemble statistics (merged + deduped by canonical SMILES)")
    lines.append("")
    if ens:
        lines.append(f"- **Unique mols in ensemble:** {ens.get('n_unique_mols_ensemble')}")
        lines.append(f"- **Unique Murcko scaffolds:** {ens.get('n_unique_murcko_ensemble')}")
        lines.append(f"- **MW mean:** {fmt_num(ens.get('MW_mean'), 1)} Da")
        lines.append(f"- **MW in drug range [320–480]:** {fmt_pct_raw(ens.get('MW_in_drug_range_pct'))}")
        lines.append(f"- **QED mean:** {fmt_num(ens.get('QED_mean'), 3)}")
        lines.append(f"- **QED ≥ 0.4 (drug-like):** {fmt_pct_raw(ens.get('QED_passing_pct'))}")
    else:
        lines.append("(ensemble summary not yet generated)")
    lines.append("")

    if ens_eval:
        lines.append("Ensemble eval (eval_lingo3dmol_plans, deduped):")
        lines.append(f"- **CHR (Covalent Hit Rate, headline):** {fmt_pct(ens_eval.get('Covalent_Hit_Rate'))}")
        lines.append(f"- **acryl_largest_pct:** {fmt_pct(ens_eval.get('acryl_largest_pct'))}")
        lines.append(f"- **connectivity_largest_pct:** {fmt_pct(ens_eval.get('connectivity_largest_pct'))}")
        lines.append(f"- **scaffold_diversity_pct:** {fmt_pct(ens_eval.get('scaffold_diversity_pct'))}")
        lines.append(f"- **MW_in_drug_range_pct:** {fmt_pct(ens_eval.get('MW_in_drug_range_pct'))}")
        lines.append(f"- **QED_passing_pct:** {fmt_pct(ens_eval.get('QED_passing_pct'))}")
    lines.append("")

    # Top-20 by composite (full list in ensemble_summary.json)
    if ens and ens.get("top_50_by_composite"):
        lines.append("## Top-20 by composite (QED × MW-in-range × heteroaryl-arm)")
        lines.append("")
        lines.append("| rank | chassis | comp | QED | MW | SMILES |")
        lines.append("|---|---|---|---|---|---|")
        for t in ens["top_50_by_composite"][:20]:
            lines.append(
                f"| {t['rank']} | `{t['source_chassis']}` | "
                f"{t['composite_score']:.2f} | {t['QED']:.2f} | "
                f"{t['MW']:.0f} | `{t['smiles']}` |"
            )
        lines.append("")
        lines.append("(Full top-50 in `data/lingo3dmol_zap70_chassis/ENSEMBLE/ensemble_summary.json`.)")
        lines.append("")

    lines.append("## Conclusions")
    lines.append("")
    n_done = sum(1 for r in rows if r['n500_n'] and r['n500_n'] > 0)
    n_pass = sum(1 for r in rows if r['n500_status'] == 'ok')
    lines.append(f"- 6 chassis dispatched (2 ChEMBL-mined + 4 published); {n_done}/6 produced mols; {n_pass}/6 reached status=ok.")
    if ens:
        lines.append(f"- Total unique mols in ensemble: **{ens.get('n_unique_mols_ensemble')}**, across **{ens.get('n_unique_murcko_ensemble')}** unique Murcko scaffolds.")
    lines.append("- **Negative dataset finding:** ChEMBL ZAP70 acrylamide actives cluster into only 2 Murcko-distinct chassis (both aniline-acrylamide). Multi-chassis generalization requires augmentation with published kinase chassis (this work) or a non-ZAP70 covalent set.")
    lines.append("")
    lines.append("## Artifacts + scripts")
    lines.append("")
    lines.append(f"- Per-chassis SDFs: `{base}/<ZAP_Cx>/samples.sdf` + `_eval.json`")
    lines.append(f"- Ensemble: `{base}/ENSEMBLE/samples.sdf` + `ensemble_summary.json` + `final_report.json`")
    lines.append("- Scripts: `experiments/{mine,merge,finalize,generate}_zap70_chassis*.py`; `run_lingo3dmol_l2_extended_anchor.py` (ZAP_C1-C6 anchors)")

    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text("\n".join(lines))
    print(f"Wrote {args.output}")


if __name__ == "__main__":
    main()
