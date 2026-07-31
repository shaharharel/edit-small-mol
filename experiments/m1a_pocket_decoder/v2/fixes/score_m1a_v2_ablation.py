"""Score the M1a v2 ablation cohorts (A baseline + B/C/D/E/F + optional G).

Reuses the same per-cohort metrics as v1 (score_m1a_ablation.py) and writes a
v2-specific deliverable with side-by-side comparison vs v1:

  results/paper_evaluation/m1a_v2_pose_sensitivity.json
  results/paper_evaluation/m1a_v2_pose_sensitivity.md

Verdicts:
  Q1 Pose channel still used?  (B vs A — should still collapse)
  Q2 BD-angle dim now a control knob? (C vs D — with Fix 1+3 should NOW move
      planar_dev; if not the model still ignores BD)
  Q3 Pocket channel used?      (E vs A; F as non-kinase sanity)
  Q4 (NEW v2) Does E produce EGFR-appropriate chemistry?
      Tanimoto-to-osimertinib >= 0.15 (passes), >= 0.20 (clean win)
  Q5 (NEW v2 — variant G) Is the planar-dihedral dim independently
      steerable? (G vs A)
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
OSIMERTINIB_SMI = "COc1cc(N(C)CCN(C)C)c(NC(=O)C=C)cc1Nc1nccc(-c2cn(C)c3ccccc23)n1"

ACRYL_SMARTS = "[CH2]=[CH]C(=O)N"

VARIANT_LABELS = {
    "A": "baseline (Mol1 cofold pose + ZAP70)",
    "B": "NULL pose + ZAP70",
    "C": "BD=60° + ZAP70",
    "D": "BD=150° + ZAP70",
    "E": "Mol1 pose + EGFR(5GTY)",
    "F": "Mol1 pose + Cathepsin(1AEC)",
    "G": "Planar-dihedral=90° + ZAP70 (NEW v2)",
}

# v1 numbers from results/paper_evaluation/m1a_pose_sensitivity.md for the
# comparison column in the v2 report.
V1_SUMMARIES = {
    "A": {"validity_rate": 0.843, "unique_canonical_frac": 0.479,
          "top_scaffold_share": 0.310, "acryl_retention_rate": 0.919,
          "pre_reactivity_score_frac_ge_0p5": 0.612,
          "planar_dev_median_deg": 2.52,
          "tanimoto_to_mol1_median": 0.613,
          "tanimoto_to_osimertinib_median": 0.175},
    "B": {"validity_rate": 0.504, "unique_canonical_frac": 0.893,
          "top_scaffold_share": 0.061, "acryl_retention_rate": 0.462,
          "pre_reactivity_score_frac_ge_0p5": 0.503,
          "planar_dev_median_deg": 19.70,
          "tanimoto_to_mol1_median": 0.314,
          "tanimoto_to_osimertinib_median": 0.170},
    "C": {"validity_rate": 0.858, "unique_canonical_frac": 0.486,
          "top_scaffold_share": 0.310, "acryl_retention_rate": 0.922,
          "pre_reactivity_score_frac_ge_0p5": 0.614,
          "planar_dev_median_deg": 2.54,
          "tanimoto_to_mol1_median": 0.609,
          "tanimoto_to_osimertinib_median": 0.175},
    "D": {"validity_rate": 0.845, "unique_canonical_frac": 0.490,
          "top_scaffold_share": 0.303, "acryl_retention_rate": 0.924,
          "pre_reactivity_score_frac_ge_0p5": 0.610,
          "planar_dev_median_deg": 2.62,
          "tanimoto_to_mol1_median": 0.609,
          "tanimoto_to_osimertinib_median": 0.175},
    "E": {"validity_rate": 0.532, "unique_canonical_frac": 0.994,
          "top_scaffold_share": 0.020, "acryl_retention_rate": 0.048,
          "pre_reactivity_score_frac_ge_0p5": 0.442,
          "planar_dev_median_deg": 22.40,
          "tanimoto_to_mol1_median": 0.210,
          "tanimoto_to_osimertinib_median": 0.140},
    "F": {"validity_rate": 0.535, "unique_canonical_frac": 0.994,
          "top_scaffold_share": 0.050, "acryl_retention_rate": 0.077,
          "pre_reactivity_score_frac_ge_0p5": 0.464,
          "planar_dev_median_deg": 23.49,
          "tanimoto_to_mol1_median": 0.198,
          "tanimoto_to_osimertinib_median": 0.136},
    # G has no v1 counterpart
}


def morgan_fp(smi, radius=2, n_bits=2048):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=n_bits)


def canonicalize(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    return Chem.MolToSmiles(m)


def murcko_scaffold_smi(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return None
    try:
        return Chem.MolToSmiles(MurckoScaffold.GetScaffoldForMol(m))
    except Exception:
        return None


def acryl_match(smi):
    m = Chem.MolFromSmiles(smi)
    if m is None: return False
    return bool(m.GetSubstructMatches(Chem.MolFromSmarts(ACRYL_SMARTS)))


def per_cohort_basic_metrics(df):
    smis = df["SMILES"].astype(str).tolist()
    n = len(smis)
    canons = [canonicalize(s) for s in smis]
    n_valid = sum(1 for c in canons if c is not None)
    valid_canons = [c for c in canons if c is not None]
    n_unique = len(set(valid_canons)) if valid_canons else 0
    scs = [murcko_scaffold_smi(s) for s in smis]
    sc_counter = Counter([s for s in scs if s])
    top_sc_share = (sc_counter.most_common(1)[0][1] / n_valid) if (sc_counter and n_valid) else None
    top_sc = sc_counter.most_common(1)[0][0] if sc_counter else None
    n_acryl = sum(1 for s in smis if acryl_match(s))
    fp_mol1 = morgan_fp(MOL1_SMI); fp_osi = morgan_fp(OSIMERTINIB_SMI)
    tan_mol1, tan_osi = [], []
    for c in canons:
        if c is None: continue
        fp = morgan_fp(c)
        if fp is None: continue
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


def run_geom_panel(df, cohort_id, workers):
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
                              "planar_dev_deg": None,
                              "pre_reactivity_score": None,
                              "msg": f"fut_exc:{e}"})
            done += 1
            if done % 1000 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(f"  [geom {cohort_id}] {done}/{len(tasks)}  "
                       f"{rate:.1f} mol/s  ETA {eta/60:.1f}min", flush=True)
    return pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)


def geom_summary(df_geom):
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
    ap.add_argument("--ablation_dir", default=str(PROJECT_ROOT /
                     "data/m1a_v2_ablation"))
    ap.add_argument("--out_json", default=str(PROJECT_ROOT /
                     "results/paper_evaluation/m1a_v2_pose_sensitivity.json"))
    ap.add_argument("--out_md", default=str(PROJECT_ROOT /
                     "results/paper_evaluation/m1a_v2_pose_sensitivity.md"))
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--progress_path", default=str(PROJECT_ROOT /
                     "data/m1a_v2_progress.json"))
    args = ap.parse_args()

    ablation_dir = Path(args.ablation_dir)
    out_json = Path(args.out_json); out_md = Path(args.out_md)
    out_json.parent.mkdir(parents=True, exist_ok=True)

    cohorts = {}
    for v in ["A", "B", "C", "D", "E", "F", "G"]:
        p = ablation_dir / f"cohort_{v}.csv"
        if not p.exists():
            print(f"WARNING: missing cohort {v} at {p}", flush=True); continue
        cohorts[v] = pd.read_csv(p)
        print(f"Loaded variant {v}: {len(cohorts[v])} rows", flush=True)

    summaries = {}
    geom_out_dir = PROJECT_ROOT / "results/paper_evaluation"
    geom_out_dir.mkdir(parents=True, exist_ok=True)
    for vid, df in cohorts.items():
        print(f"\n=== Variant {vid} ({VARIANT_LABELS.get(vid)}) ===", flush=True)
        basic = per_cohort_basic_metrics(df)
        geom_csv = geom_out_dir / f"m1a_v2_pose_sensitivity_geom_{vid}.csv"
        if geom_csv.exists():
            print(f"  [cached] {geom_csv}", flush=True)
            df_geom = pd.read_csv(geom_csv)
        else:
            df_geom = run_geom_panel(df, vid, args.workers)
            df_geom.to_csv(geom_csv, index=False)
        geom = geom_summary(df_geom)
        summaries[vid] = {"label": VARIANT_LABELS.get(vid, ""), **basic, **geom}
        bm = basic
        print(f"  basic: valid={bm['validity_rate']:.3f} unique={bm['unique_canonical_frac']:.3f} "
               f"acryl={bm['acryl_retention_rate']:.3f} "
               f"top_sc={bm['top_scaffold_share']:.3f} "
               f"Tan_Mol1={bm['tanimoto_to_mol1_median']:.3f} "
               f"Tan_osi={bm['tanimoto_to_osimertinib_median']:.3f}",
               flush=True)
        print(f"  geom : planar_dev_med={geom['planar_dev_median_deg']} "
               f"pre_react={geom['pre_reactivity_score_frac_ge_0p5']}",
               flush=True)

    # ----- Verdicts -----
    def diff(v, key):
        a = summaries.get("A", {}).get(key); x = summaries.get(v, {}).get(key)
        if a is None or x is None: return None
        return float(x - a)

    verdict = {}
    if "B" in summaries and "A" in summaries:
        d_planar = diff("B", "planar_dev_median_deg")
        d_acryl = diff("B", "acryl_retention_rate")
        d_tan = diff("B", "tanimoto_to_mol1_median")
        verdict["pose_conditioning_used"] = bool(
            (d_planar is not None and abs(d_planar) >= 2.0) or
            (d_acryl is not None and abs(d_acryl) >= 0.05) or
            (d_tan is not None and abs(d_tan) >= 0.05))
        verdict["B_vs_A_diffs"] = {"planar_dev_deg": d_planar,
                                     "acryl_retention": d_acryl,
                                     "tanimoto_mol1": d_tan}

    if "C" in summaries and "D" in summaries:
        c_planar = summaries["C"].get("planar_dev_median_deg")
        d_planar_v = summaries["D"].get("planar_dev_median_deg")
        a_planar = summaries.get("A", {}).get("planar_dev_median_deg")
        steerable_signal = None
        if c_planar is not None and d_planar_v is not None:
            steerable_signal = float(c_planar - d_planar_v)
        verdict["bd_angle_steerable_signal_deg"] = steerable_signal
        verdict["bd_angle_steerable"] = bool(
            steerable_signal is not None and abs(steerable_signal) >= 2.0)
        verdict["C_planar_dev"] = c_planar
        verdict["D_planar_dev"] = d_planar_v
        verdict["A_planar_dev"] = a_planar
        # Also check Tanimoto / acryl shifts
        verdict["C_vs_D_diffs"] = {
            "planar_dev_deg": steerable_signal,
            "tanimoto_mol1": (None if (summaries["C"].get("tanimoto_to_mol1_median") is None
                                         or summaries["D"].get("tanimoto_to_mol1_median") is None)
                                else float(summaries["C"]["tanimoto_to_mol1_median"]
                                            - summaries["D"]["tanimoto_to_mol1_median"])),
        }

    if "E" in summaries and "A" in summaries:
        e_tm = summaries["E"].get("tanimoto_to_mol1_median")
        a_tm = summaries["A"].get("tanimoto_to_mol1_median")
        e_to = summaries["E"].get("tanimoto_to_osimertinib_median")
        a_to = summaries["A"].get("tanimoto_to_osimertinib_median")
        verdict["E_pocket_shift_tanimoto_mol1"] = (
            float(e_tm - a_tm) if (e_tm is not None and a_tm is not None) else None)
        verdict["E_pocket_shift_tanimoto_osimertinib"] = (
            float(e_to - a_to) if (e_to is not None and a_to is not None) else None)
        verdict["pocket_sensitive"] = bool(
            (verdict["E_pocket_shift_tanimoto_mol1"] is not None and
             verdict["E_pocket_shift_tanimoto_mol1"] <= -0.03) or
            (verdict["E_pocket_shift_tanimoto_osimertinib"] is not None and
             verdict["E_pocket_shift_tanimoto_osimertinib"] >= 0.01))
        verdict["E_egfr_appropriate_tan_osi"] = e_to
        verdict["E_egfr_appropriate_passes_0p15"] = bool(e_to is not None and e_to >= 0.15)
        verdict["E_egfr_appropriate_passes_0p20"] = bool(e_to is not None and e_to >= 0.20)

    if "F" in summaries and "A" in summaries:
        verdict["F_valid_rate"] = summaries["F"].get("validity_rate")
        verdict["F_acryl_retention"] = summaries["F"].get("acryl_retention_rate")

    if "G" in summaries and "A" in summaries:
        verdict["G_vs_A_diffs"] = {
            "planar_dev_deg": diff("G", "planar_dev_median_deg"),
            "acryl_retention": diff("G", "acryl_retention_rate"),
            "tanimoto_mol1": diff("G", "tanimoto_to_mol1_median"),
        }
        verdict["planar_dim_steerable"] = bool(
            (verdict["G_vs_A_diffs"]["planar_dev_deg"] is not None and
             abs(verdict["G_vs_A_diffs"]["planar_dev_deg"]) >= 2.0) or
            (verdict["G_vs_A_diffs"]["acryl_retention"] is not None and
             abs(verdict["G_vs_A_diffs"]["acryl_retention"]) >= 0.05))

    # ----- Write JSON -----
    out = {
        "config": {
            "ablation_dir": str(ablation_dir),
            "variant_labels": VARIANT_LABELS,
            "mol1_smi": MOL1_SMI,
            "osimertinib_smi": OSIMERTINIB_SMI,
            "v1_summaries": V1_SUMMARIES,
        },
        "cohort_summaries": summaries,
        "verdicts": verdict,
    }
    out_json.write_text(json.dumps(out, indent=2, default=str))
    print(f"\nWrote {out_json}", flush=True)

    # ----- Write MD -----
    md = []
    md.append("# M1a v2 pose-sensitivity ablation\n")
    md.append("(Pose schema corrected per Fix 1+2+3+4; trained on 9,365-triple v2 corpus.)\n")
    md.append("\n## Cohort summaries (v2)\n")

    cols = list(cohorts.keys())
    def row(label, key, nd=3):
        cells = [label]
        for v in cols:
            x = summaries.get(v, {}).get(key)
            if x is None: cells.append("—")
            elif isinstance(x, float): cells.append(f"{x:.{nd}f}")
            else: cells.append(str(x))
        return "| " + " | ".join(cells) + " |"

    md.append("| Metric | " + " | ".join(cols) + " |")
    md.append("|---|" + "|".join(["---"] * len(cols)) + "|")
    md.append("| Label | " + " | ".join(VARIANT_LABELS.get(c, "") for c in cols) + " |")
    md.append(row("n_total", "n_total", 0))
    md.append(row("validity_rate", "validity_rate"))
    md.append(row("unique_canonical_frac", "unique_canonical_frac"))
    md.append(row("top_scaffold_share", "top_scaffold_share"))
    md.append(row("acryl_retention_rate", "acryl_retention_rate"))
    md.append(row("pre_reactivity_frac_ge_0.5", "pre_reactivity_score_frac_ge_0p5"))
    md.append(row("planar_dev_median_deg", "planar_dev_median_deg", 2))
    md.append(row("tanimoto_to_mol1_median", "tanimoto_to_mol1_median"))
    md.append(row("tanimoto_to_osimertinib_median", "tanimoto_to_osimertinib_median"))

    # v1 comparison column
    md.append("\n## v1 vs v2 comparison (selected variants)\n")
    md.append("| Variant | Metric | v1 | v2 | Δ |")
    md.append("|---|---|---|---|---|")
    for vid in ["A", "B", "C", "D", "E", "F"]:
        if vid not in summaries: continue
        v1 = V1_SUMMARIES.get(vid, {})
        v2 = summaries[vid]
        for metric in ("validity_rate", "acryl_retention_rate",
                        "planar_dev_median_deg", "tanimoto_to_mol1_median",
                        "tanimoto_to_osimertinib_median"):
            a = v1.get(metric); b = v2.get(metric)
            if a is None or b is None: continue
            md.append(f"| {vid} | {metric} | {a:.3f} | {b:.3f} | "
                       f"{b - a:+.3f} |")

    md.append("\n## Headline verdicts (v2)\n")
    pose_used = verdict.get("pose_conditioning_used")
    bd_steer = verdict.get("bd_angle_steerable")
    poc = verdict.get("pocket_sensitive")
    e_osi = verdict.get("E_egfr_appropriate_tan_osi")
    plan_steer = verdict.get("planar_dim_steerable")

    md.append("### Q1. Pose channel still used? (B vs A)\n")
    md.append(f"- **Answer: {'YES' if pose_used else 'NO'}** "
               f"(should remain YES; B = NULL pose collapses chemistry)")
    bd = verdict.get("B_vs_A_diffs", {})
    md.append(f"- Δ planar_dev = {bd.get('planar_dev_deg')}")
    md.append(f"- Δ acryl_retention = {bd.get('acryl_retention')}")
    md.append(f"- Δ tanimoto_to_Mol1 = {bd.get('tanimoto_mol1')}\n")

    md.append("### Q2. BD-angle dim NOW a control knob? (C vs D) — KEY V2 TEST\n")
    md.append(f"- **Answer: {'YES — BD-angle dim is steerable' if bd_steer else 'NO — model still ignores the BD-angle dim'}**")
    md.append(f"- A baseline planar_dev = {verdict.get('A_planar_dev')}")
    md.append(f"- C (BD=60°) planar_dev = {verdict.get('C_planar_dev')}")
    md.append(f"- D (BD=150°) planar_dev = {verdict.get('D_planar_dev')}")
    md.append(f"- C - D gap = {verdict.get('bd_angle_steerable_signal_deg')} "
               f"(threshold for 'steerable': |Δ| ≥ 2°)\n")

    md.append("### Q3. Pocket channel used? (E vs A)\n")
    md.append(f"- **Answer: {'YES' if poc else 'NO'}**")
    md.append(f"- Δ tanimoto_to_Mol1 = {verdict.get('E_pocket_shift_tanimoto_mol1')}")
    md.append(f"- Δ tanimoto_to_osimertinib = {verdict.get('E_pocket_shift_tanimoto_osimertinib')}")
    md.append(f"- F (Cathepsin) validity={verdict.get('F_valid_rate')} "
               f"acryl={verdict.get('F_acryl_retention')} — sanity\n")

    md.append("### Q4. (NEW v2) Does E produce EGFR-appropriate chemistry?\n")
    md.append(f"- Tanimoto-to-osimertinib (E) = {e_osi}")
    md.append(f"- Threshold 0.15: {'PASS' if verdict.get('E_egfr_appropriate_passes_0p15') else 'FAIL'}")
    md.append(f"- Threshold 0.20 (clean win): {'PASS' if verdict.get('E_egfr_appropriate_passes_0p20') else 'FAIL'}\n")

    if "G" in summaries:
        md.append("### Q5. (NEW v2 — variant G) Planar-dihedral dim independently steerable?\n")
        md.append(f"- **Answer: {'YES' if plan_steer else 'NO'}**")
        gd = verdict.get("G_vs_A_diffs", {})
        md.append(f"- Δ planar_dev = {gd.get('planar_dev_deg')}")
        md.append(f"- Δ acryl_retention = {gd.get('acryl_retention')}")
        md.append(f"- Δ tanimoto_to_Mol1 = {gd.get('tanimoto_mol1')}\n")

    md.append("## One-paragraph headline\n")
    parts = []
    if pose_used:
        parts.append("Pose conditioning IS used (NULL pose collapses chemistry)")
    else:
        parts.append("Pose conditioning is IGNORED by v2 (BLOCKER for paper)")
    if bd_steer:
        parts.append("AND the BD-angle dim IS now a steering knob "
                      "(the Fix 1+3 improvements turned the BD-angle channel from "
                      "ignored to load-bearing)")
    else:
        parts.append("BUT the BD-angle dim is STILL not independently steerable")
    if poc:
        parts.append("AND the pocket channel remains pocket-sensitive")
    else:
        parts.append("but pocket sensitivity collapsed")
    if e_osi is not None and e_osi >= 0.15:
        parts.append("AND swapping to the EGFR pocket now produces "
                      f"EGFR-appropriate chemistry (Tan→osimertinib={e_osi:.3f}, "
                      "passes 0.15 threshold)")
    elif e_osi is not None:
        parts.append(f"BUT EGFR-pocketed output still does NOT recover "
                      f"osimertinib-like chemistry (Tan={e_osi:.3f})")
    md.append("**" + ". ".join(parts) + ".**\n")

    out_md.write_text("\n".join(md))
    print(f"Wrote {out_md}", flush=True)

    Path(args.progress_path).write_text(json.dumps({
        "phase": "phase4_done", "timestamp": time.time(),
        "timestamp_iso": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }, indent=2))


if __name__ == "__main__":
    main()
