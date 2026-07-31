"""Score the 6 M1a ablation cohorts (A baseline + B/C/D/E/F variants).

Per-cohort metrics:
  1. validity_rate          — fraction parseable by RDKit
  2. unique_canonical_frac  — fraction unique after canonicalization
  3. top_scaffold_share     — fraction of mols with the most common Murcko scaffold
  4. acryl_retention        — fraction matching `[CH2]=[CH]C(=O)N` SMARTS
  5. pre_reactivity_score   — fraction with planar_dev <= 20° (acrylamide subset)
  6. planar_dev_median      — median deviation of acryl vinyl-amide dihedral from planarity
  7. tanimoto_to_mol1_med   — median Tanimoto (Morgan r=2, 2048b) to Mol1
  8. tanimoto_to_egfr_med   — only for variant E: median Tanimoto to known EGFR
                              acrylamide inhibitor (osimertinib)

Reuses experiments/covft_geometric_options_bc.py._compute_2d_one for
the planar dihedral.

Writes:
  results/paper_evaluation/m1a_pose_sensitivity.json
  results/paper_evaluation/m1a_pose_sensitivity.md
"""
from __future__ import annotations
import argparse
import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger, DataStructs
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.covft_geometric_options_bc import _compute_2d_one  # noqa: E402

MOL1_SMI = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"
# Osimertinib (Tagrisso) — clinically approved EGFR Cys797 covalent inhibitor
OSIMERTINIB_SMI = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1"

ACRYL_SMARTS = "[CH2]=[CH]C(=O)N"

VARIANT_LABELS = {
    "A": "baseline (Mol1 pose + ZAP70)",
    "B": "NULL pose + ZAP70",
    "C": "BD=60° + ZAP70",
    "D": "BD=150° + ZAP70",
    "E": "Mol1 pose + EGFR(5GTY)",
    "F": "Mol1 pose + Cathepsin(1AEC)",
}


def morgan_fp(smi: str, radius: int = 2, n_bits: int = 2048):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def canonicalize(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def murcko_scaffold_smi(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    try:
        sc = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(sc)
    except Exception:
        return None


def acryl_match(smi: str) -> bool:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return False
    patt = Chem.MolFromSmarts(ACRYL_SMARTS)
    return bool(m.GetSubstructMatches(patt))


def per_cohort_basic_metrics(df: pd.DataFrame) -> dict:
    """Validity, uniqueness, top-scaffold share, acrylamide retention, Tanimoto to Mol1
    (and to osimertinib, computed always — caller may keep only for variant E)."""
    smis = df["SMILES"].astype(str).tolist()
    n = len(smis)

    canons = [canonicalize(s) for s in smis]
    n_valid = sum(1 for c in canons if c is not None)

    # Unique canonical fraction among valids
    valid_canons = [c for c in canons if c is not None]
    n_unique = len(set(valid_canons)) if valid_canons else 0

    # Top scaffold share (over valid)
    scs = [murcko_scaffold_smi(s) for s in smis]
    sc_counter = Counter([s for s in scs if s])
    top_sc_share = (sc_counter.most_common(1)[0][1] / n_valid) if (sc_counter and n_valid) else None
    top_sc = sc_counter.most_common(1)[0][0] if sc_counter else None

    # Acrylamide retention
    n_acryl = sum(1 for s in smis if acryl_match(s))

    # Tanimoto-to-Mol1 (over valids)
    fp_mol1 = morgan_fp(MOL1_SMI)
    fp_osi = morgan_fp(OSIMERTINIB_SMI)
    tan_mol1, tan_osi = [], []
    for c in canons:
        if c is None:
            continue
        fp = morgan_fp(c)
        if fp is None:
            continue
        tan_mol1.append(float(DataStructs.TanimotoSimilarity(fp, fp_mol1)))
        tan_osi.append(float(DataStructs.TanimotoSimilarity(fp, fp_osi)))

    return {
        "n_total": int(n),
        "n_valid": int(n_valid),
        "validity_rate": float(n_valid / n) if n else None,
        "n_unique_canonical": int(n_unique),
        "unique_canonical_frac": float(n_unique / n_valid) if n_valid else None,
        "top_scaffold_smi": top_sc,
        "top_scaffold_share": float(top_sc_share) if top_sc_share is not None else None,
        "n_acrylamide_match": int(n_acryl),
        "acryl_retention_rate": float(n_acryl / n_valid) if n_valid else None,
        "tanimoto_to_mol1_median": float(np.median(tan_mol1)) if tan_mol1 else None,
        "tanimoto_to_mol1_mean": float(np.mean(tan_mol1)) if tan_mol1 else None,
        "tanimoto_to_osimertinib_median": float(np.median(tan_osi)) if tan_osi else None,
        "tanimoto_to_osimertinib_mean": float(np.mean(tan_osi)) if tan_osi else None,
        "n_tanimoto_evaluated": int(len(tan_mol1)),
    }


def run_geom_panel(df: pd.DataFrame, cohort_id: str, workers: int) -> pd.DataFrame:
    """Run _compute_2d_one over all SMILES, returning per-mol geometry."""
    tasks = [(i, smi) for i, smi in enumerate(df["SMILES"].astype(str).tolist())]
    print(f"[geom] cohort={cohort_id} N={len(tasks)} workers={workers}", flush=True)
    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futures = [exc.submit(_compute_2d_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"idx": -1, "smi": "", "acryl_match": False,
                              "embed_ok": False, "dihedral_deg": None,
                              "planar_dev_deg": None, "pre_reactivity_score": None,
                              "msg": f"fut_exc:{e}"})
            done += 1
            if done % 1000 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"  [geom {cohort_id}] {done}/{len(tasks)}  "
                       f"{rate:.1f} mol/s  ETA {eta/60:.1f}min", flush=True)
    out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    out["cohort"] = cohort_id
    return out


def geom_summary(df_geom: pd.DataFrame) -> dict:
    acryl_mask = df_geom["acryl_match"].astype(bool)
    geom_ok = df_geom["embed_ok"].astype(bool) & df_geom["pre_reactivity_score"].notna()
    pre = df_geom.loc[geom_ok, "pre_reactivity_score"].astype(float).values
    dev = df_geom.loc[geom_ok, "planar_dev_deg"].astype(float).values
    raw_di = df_geom.loc[geom_ok, "dihedral_deg"].astype(float).values
    return {
        "n_acryl_for_geom": int(acryl_mask.sum()),
        "n_geom_evaluated": int(geom_ok.sum()),
        "pre_reactivity_score_frac_ge_0p5": float((pre >= 0.5).mean()) if len(pre) else None,
        "planar_dev_median_deg": float(np.median(dev)) if len(dev) else None,
        "planar_dev_q25_deg": float(np.percentile(dev, 25)) if len(dev) else None,
        "planar_dev_q75_deg": float(np.percentile(dev, 75)) if len(dev) else None,
        "raw_dihedral_median_deg": float(np.median(np.abs(raw_di))) if len(raw_di) else None,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ablation-dir", default=str(PROJECT_ROOT / "data/m1a_ablation"))
    ap.add_argument("--baseline-csv", default=str(PROJECT_ROOT / "data/m1a_cohorts/cohort_mol1_zap70.csv"))
    ap.add_argument("--out-json", default=str(PROJECT_ROOT / "results/paper_evaluation/m1a_pose_sensitivity.json"))
    ap.add_argument("--out-md", default=str(PROJECT_ROOT / "results/paper_evaluation/m1a_pose_sensitivity.md"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--baseline-n", type=int, default=5000,
                     help="Subsample baseline A to this N for parity (will use first N rows)")
    ap.add_argument("--progress-path", default=str(PROJECT_ROOT / "data/m1a_ablation_progress.json"))
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir)
    out_json = Path(args.out_json)
    out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    # ----- Load all cohorts -----
    cohorts: dict[str, pd.DataFrame] = {}
    print(f"Loading variant A baseline (subsample to {args.baseline_n}) from {args.baseline_csv}", flush=True)
    a_df = pd.read_csv(args.baseline_csv)
    if len(a_df) > args.baseline_n:
        a_df = a_df.head(args.baseline_n).reset_index(drop=True)
    cohorts["A"] = a_df

    for v in ["B", "C", "D", "E", "F"]:
        p = ablation_dir / f"cohort_{v}.csv"
        if not p.exists():
            print(f"WARNING: missing cohort {v} at {p}", flush=True)
            continue
        cohorts[v] = pd.read_csv(p)
        print(f"Loaded variant {v}: {len(cohorts[v])} rows", flush=True)

    # ----- Per-cohort: basic metrics + geometry -----
    summaries: dict[str, dict] = {}
    geom_out_dir = PROJECT_ROOT / "results/paper_evaluation"
    geom_out_dir.mkdir(parents=True, exist_ok=True)

    for vid, df in cohorts.items():
        print(f"\n=== Variant {vid} ({VARIANT_LABELS.get(vid)}) ===", flush=True)
        basic = per_cohort_basic_metrics(df)
        geom_csv = geom_out_dir / f"m1a_pose_sensitivity_geom_{vid}.csv"
        if geom_csv.exists():
            print(f"  [cached] {geom_csv}", flush=True)
            df_geom = pd.read_csv(geom_csv)
        else:
            df_geom = run_geom_panel(df, vid, args.workers)
            df_geom.to_csv(geom_csv, index=False)
        geom = geom_summary(df_geom)
        summaries[vid] = {
            "label": VARIANT_LABELS.get(vid, ""),
            **basic, **geom,
        }
        print(f"  basic: valid={basic['validity_rate']:.3f} unique={basic['unique_canonical_frac']:.3f} "
               f"acryl={basic['acryl_retention_rate']:.3f} top_sc_share={basic['top_scaffold_share']:.3f} "
               f"Tan_Mol1={basic['tanimoto_to_mol1_median']:.3f}", flush=True)
        print(f"  geom : planar_dev_med={geom['planar_dev_median_deg']} pre_react_frac={geom['pre_reactivity_score_frac_ge_0p5']}",
               flush=True)

    # ----- Verdicts -----
    def diff(v: str, key: str) -> float | None:
        a = summaries.get("A", {}).get(key)
        x = summaries.get(v, {}).get(key)
        if a is None or x is None:
            return None
        return float(x - a)

    verdict = {}
    if "B" in summaries and "A" in summaries:
        d_planar = diff("B", "planar_dev_median_deg")
        d_acryl = diff("B", "acryl_retention_rate")
        d_tan = diff("B", "tanimoto_to_mol1_median")
        # Heuristic: pose conditioning is "real" if NULL pose moves planar_dev by >=2°
        # and acryl retention or Tanimoto-to-Mol1 by >=0.05
        b_planar_drift = abs(d_planar) if d_planar is not None else 0.0
        b_acryl_drift = abs(d_acryl) if d_acryl is not None else 0.0
        b_tan_drift = abs(d_tan) if d_tan is not None else 0.0
        verdict["pose_conditioning_used"] = bool(
            b_planar_drift >= 2.0 or b_acryl_drift >= 0.05 or b_tan_drift >= 0.05
        )
        verdict["B_vs_A_diffs"] = {
            "planar_dev_deg": d_planar,
            "acryl_retention": d_acryl,
            "tanimoto_mol1": d_tan,
        }

    if "C" in summaries and "D" in summaries:
        c_planar = summaries["C"].get("planar_dev_median_deg")
        d_planar = summaries["D"].get("planar_dev_median_deg")
        a_planar = summaries.get("A", {}).get("planar_dev_median_deg")
        # Steerable if C is closer to 60° s-trans direction (larger planar_dev)
        # or D is closer to 180° (smaller planar_dev). Vinyl-amide planar dihedrals
        # are usually 0° or 180°; pre_dev <=20° = planar. If model is steerable,
        # C with bd_angle=60° should produce a larger planar_dev than D with bd=150°.
        steerable_signal = None
        if c_planar is not None and d_planar is not None:
            steerable_signal = float(c_planar - d_planar)  # positive = C more bent
        verdict["bd_angle_steerable_signal_deg"] = steerable_signal
        verdict["bd_angle_steerable"] = bool(steerable_signal is not None and abs(steerable_signal) >= 2.0)
        verdict["C_planar_dev"] = c_planar
        verdict["D_planar_dev"] = d_planar
        verdict["A_planar_dev"] = a_planar

    if "E" in summaries and "A" in summaries:
        e_tan_mol1 = summaries["E"].get("tanimoto_to_mol1_median")
        a_tan_mol1 = summaries["A"].get("tanimoto_to_mol1_median")
        e_tan_osi = summaries["E"].get("tanimoto_to_osimertinib_median")
        a_tan_osi = summaries["A"].get("tanimoto_to_osimertinib_median")
        verdict["E_pocket_shift_tanimoto_mol1"] = (
            float(e_tan_mol1 - a_tan_mol1) if (e_tan_mol1 is not None and a_tan_mol1 is not None) else None
        )
        verdict["E_pocket_shift_tanimoto_osimertinib"] = (
            float(e_tan_osi - a_tan_osi) if (e_tan_osi is not None and a_tan_osi is not None) else None
        )
        # Pocket-aware if EGFR shifts Tanimoto-to-Mol1 down by >=0.03 OR
        # shifts Tanimoto-to-osimertinib UP by >=0.01
        verdict["pocket_sensitive"] = bool(
            (verdict["E_pocket_shift_tanimoto_mol1"] is not None and
             verdict["E_pocket_shift_tanimoto_mol1"] <= -0.03) or
            (verdict["E_pocket_shift_tanimoto_osimertinib"] is not None and
             verdict["E_pocket_shift_tanimoto_osimertinib"] >= 0.01)
        )

    if "F" in summaries and "A" in summaries:
        verdict["F_valid_rate"] = summaries["F"].get("validity_rate")
        verdict["F_acryl_retention"] = summaries["F"].get("acryl_retention_rate")

    # ----- Write JSON -----
    out = {
        "config": {
            "ablation_dir": str(ablation_dir),
            "baseline_csv": args.baseline_csv,
            "variant_labels": VARIANT_LABELS,
            "mol1_smi": MOL1_SMI,
            "osimertinib_smi": OSIMERTINIB_SMI,
        },
        "cohort_summaries": summaries,
        "verdicts": verdict,
    }
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_json}", flush=True)

    # ----- Write MD -----
    md = []
    md.append("# M1a pose-sensitivity ablation\n")
    md.append(f"Baseline cohort A: `{args.baseline_csv}` (subsampled to {len(cohorts.get('A', []))} mols)\n")
    md.append(f"Ablation cohorts B/C/D/E/F: `{ablation_dir}/cohort_*.csv` (5000 mols each)\n")
    md.append("\n## Cohort summaries\n")

    columns = list(cohorts.keys())
    def row(label, key, nd=3):
        cells = [label]
        for v in columns:
            x = summaries.get(v, {}).get(key)
            if x is None:
                cells.append("—")
            elif isinstance(x, float):
                cells.append(f"{x:.{nd}f}")
            else:
                cells.append(str(x))
        return "| " + " | ".join(cells) + " |"

    header = "| Metric | " + " | ".join(columns) + " |"
    sep = "|---|" + "|".join(["---"] * len(columns)) + "|"
    md.append(header)
    md.append(sep)
    md.append("| Label | " + " | ".join(VARIANT_LABELS.get(c, "") for c in columns) + " |")
    md.append(row("n_total", "n_total", 0))
    md.append(row("validity_rate", "validity_rate", 3))
    md.append(row("unique_canonical_frac", "unique_canonical_frac", 3))
    md.append(row("top_scaffold_share", "top_scaffold_share", 3))
    md.append(row("acryl_retention_rate", "acryl_retention_rate", 3))
    md.append(row("pre_reactivity_score_frac_ge_0p5", "pre_reactivity_score_frac_ge_0p5", 3))
    md.append(row("planar_dev_median_deg", "planar_dev_median_deg", 2))
    md.append(row("planar_dev_q25/q75", "planar_dev_q25_deg", 2))  # only q25 shown
    md.append(row("tanimoto_to_mol1_median", "tanimoto_to_mol1_median", 3))
    md.append(row("tanimoto_to_osimertinib_median", "tanimoto_to_osimertinib_median", 3))

    md.append("\n## Headline verdicts\n")
    pose_used = verdict.get("pose_conditioning_used")
    md.append(f"### Q1. Did M1a actually use the pose conditioning?\n")
    md.append(f"- **Answer: {'YES (pose conditioning has measurable effect)' if pose_used else 'NO (decoder ignores the pose vector)'}**")
    b_d = verdict.get("B_vs_A_diffs", {})
    md.append(f"- B (NULL pose) vs A (Mol1 pose), ΔZAP70 pocket held constant:")
    md.append(f"   - Δ planar_dev_median = {b_d.get('planar_dev_deg')}")
    md.append(f"   - Δ acryl_retention   = {b_d.get('acryl_retention')}")
    md.append(f"   - Δ tanimoto_to_Mol1  = {b_d.get('tanimoto_mol1')}")
    md.append(f"- Threshold for 'used': Δ planar_dev≥2° OR Δ acryl≥0.05 OR Δ Tan_Mol1≥0.05.\n")

    steer = verdict.get("bd_angle_steerable")
    md.append(f"### Q2. Is M1a steerable by changing the input pose?\n")
    md.append(f"- **Answer: {'YES' if steer else 'NO'}**")
    md.append(f"- C (BD=60°) planar_dev_median = {verdict.get('C_planar_dev')}")
    md.append(f"- D (BD=150°) planar_dev_median = {verdict.get('D_planar_dev')}")
    md.append(f"- A (Mol1 pose, BD~129°) planar_dev_median = {verdict.get('A_planar_dev')}")
    md.append(f"- C - D planar_dev gap = {verdict.get('bd_angle_steerable_signal_deg')}")
    md.append(f"- Threshold for 'steerable': |C-D gap| ≥ 2°.\n")

    poc = verdict.get("pocket_sensitive")
    md.append(f"### Q3. Is M1a sensitive to the input pocket?\n")
    md.append(f"- **Answer: {'YES' if poc else 'NO'}**")
    md.append(f"- E (Mol1 pose + EGFR pocket) Δ Tanimoto_to_Mol1 vs A = {verdict.get('E_pocket_shift_tanimoto_mol1')}")
    md.append(f"- E (Mol1 pose + EGFR pocket) Δ Tanimoto_to_osimertinib vs A = {verdict.get('E_pocket_shift_tanimoto_osimertinib')}")
    md.append(f"- Threshold for 'sensitive': Δ Tan_Mol1 ≤ -0.03 OR Δ Tan_osi ≥ +0.01.\n")
    md.append(f"- F (Mol1 pose + Cathepsin pocket) validity={verdict.get('F_valid_rate')} acryl={verdict.get('F_acryl_retention')} — sanity\n")

    md.append("## One-paragraph headline\n")
    parts = []
    if pose_used:
        parts.append("Pose conditioning IS used by M1a")
    else:
        parts.append("Pose conditioning is effectively IGNORED by M1a — NULL pose ≈ baseline pose")
    if steer:
        parts.append("AND the model is steerable by changing the BD angle input")
    else:
        parts.append("but the model is NOT cleanly steerable by changing the BD angle dimension alone")
    if poc:
        parts.append("AND the model IS sensitive to swapping the input pocket (EGFR vs ZAP70 changes chemistry)")
    else:
        parts.append("AND the model shows NO meaningful sensitivity to swapping the input pocket")
    md.append("**" + ". ".join(parts) + ".**\n")

    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}", flush=True)

    # Update progress file
    Path(args.progress_path).write_text(json.dumps({
        "phase": "done",
        "timestamp": time.time(),
        "current_variant": None,
    }))


if __name__ == "__main__":
    main()
