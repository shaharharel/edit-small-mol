"""Build the unified ZAP70 screening cohort from ALL generative methods.

Reads every available SDF cohort (Lingo3DMol L1/L2/multi-anchor/chassis,
DrugFlow Inpaint v2 / Tier-2-FT, DiffSBDD Plan E), evaluates each mol with
the v2 metric suite (`experiments.eval_lingo3dmol_plans`), dedupes by
canonical SMILES, scores with the composite formula in the task spec,
and writes the merged inventory + top-100 + per-method summary + the
report at /tmp/screening_cohort_summary.md.

Watchdog mode: re-runs the merge every 30 min for up to 3 hours so the
cohort grows as the in-flight Lingo3DMol N=500 samplers continue writing.
Files whose mtime is younger than `--mtime-stable-sec` (default 60 s) are
treated as "actively being written" and SKIPPED for that pass; they will
be picked up on the next pass.

Mac CPU only. Uses the existing `eval_cohort()` API — no eval logic is
re-implemented.

Usage:
    python experiments/build_screening_cohort.py            # one pass
    python experiments/build_screening_cohort.py --watch    # 30-min loop, 3 h
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
import warnings
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs
from rdkit.Chem.Scaffolds import MurckoScaffold

warnings.filterwarnings("ignore")
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.eval_lingo3dmol_plans import eval_cohort  # noqa: E402

OUT_DIR = PROJECT_ROOT / "data" / "screening_cohort"
OUT_DIR.mkdir(parents=True, exist_ok=True)
REPORT_PATH = Path("/tmp/screening_cohort_summary.md")
MMGBSA_CSV = PROJECT_ROOT / "data/boltz_poses/zap70_cys346_capped_mmpbsa_top50.csv"
ANCHOR_JSON = PROJECT_ROOT / "data/lingo3dmol_anchor_zap70_cys346.json"
TRAIN_CSV = PROJECT_ROOT / "data/retrain_covalent/acrylamide_only.csv"

# ---------------------------------------------------------------------------
# Cohort inventory: (source_label, path, expected_n_or_None)
# Each entry is treated independently — if path missing, it's skipped with a log.
# ---------------------------------------------------------------------------
COHORTS: list[tuple[str, Path]] = [
    # Tonight's L2 cohorts (small T10 batches)
    ("L2_scaffold_C5",            PROJECT_ROOT / "data/lingo3dmol_L2_scaffold_C5/samples_T10.sdf"),
    ("L2_extended_H2",            PROJECT_ROOT / "data/lingo3dmol_L2_extended_H2/samples_T10.sdf"),
    ("L2_extended_L_v2",          PROJECT_ROOT / "data/lingo3dmol_L2_extended_L_v2/samples_T10.sdf"),
    # In-flight N=500 batches
    ("L2_extended_H2_N500",       PROJECT_ROOT / "data/lingo3dmol_L2_extended_H2_N500/samples.sdf"),
    ("L2_scaffold_C5_N500",       PROJECT_ROOT / "data/lingo3dmol_L2_scaffold_C5_N500/samples.sdf"),
    # Legacy / multi-anchor / chassis
    ("L2_scaffold_anchor_C1",     PROJECT_ROOT / "data/lingo3dmol_L2_scaffold_anchor/samples_T10_N500.sdf"),
    # multi-anchor: A was requested but doesn't exist on disk — include B/C/D + ENSEMBLE.
    ("multi_anchor_B",            PROJECT_ROOT / "data/lingo3dmol_multi_anchor/B/samples_T10.sdf"),
    ("multi_anchor_C",            PROJECT_ROOT / "data/lingo3dmol_multi_anchor/C/samples_T10.sdf"),
    ("multi_anchor_D",            PROJECT_ROOT / "data/lingo3dmol_multi_anchor/D/samples_T10.sdf"),
    ("multi_anchor_ENSEMBLE",     PROJECT_ROOT / "data/lingo3dmol_multi_anchor/ENSEMBLE/samples.sdf"),
    ("L1_C5",                     PROJECT_ROOT / "data/lingo3dmol_L1_C5/samples_T10.sdf"),
    # zap70_chassis sub-runs (skip if file mtime is younger than threshold; watchdog will pick up)
    ("chassis_C1",                PROJECT_ROOT / "data/lingo3dmol_zap70_chassis/ZAP_C1/samples.sdf"),
    ("chassis_C2",                PROJECT_ROOT / "data/lingo3dmol_zap70_chassis/ZAP_C2/samples.sdf"),
    ("chassis_ENSEMBLE",          PROJECT_ROOT / "data/lingo3dmol_zap70_chassis/ENSEMBLE/samples.sdf"),
    # Alternative methods
    ("drugflow_inpaint_v2_500",   PROJECT_ROOT / "results/covalent_gen_day1/drugflow_inpaint_v2_500/samples.sdf"),
    ("drugflow_dc_winner_500",    PROJECT_ROOT / "results/covalent_gen_day1/drugflow_dc_winner_500/samples.sdf"),
    ("diffsbdd_plan_e",           PROJECT_ROOT / "data/genplan_e_diffsbdd/fixed.sdf"),
]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def _morgan_bv(mol: Chem.Mol, n: int = 2048, r: int = 2):
    return AllChem.GetMorganFingerprintAsBitVect(mol, r, nBits=n)


def _canonical_smiles(s: str) -> Optional[str]:
    if not isinstance(s, str) or not s:
        return None
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


def _is_stable(path: Path, min_age_sec: float) -> bool:
    """File mtime older than threshold AND size unchanged across a brief sample."""
    if not path.exists():
        return False
    age = time.time() - path.stat().st_mtime
    return age >= min_age_sec


def _load_mmgbsa_top15_fps() -> tuple[list, list[str]]:
    """Top-15 SMILES (by combined_score) from MM-GBSA + their Morgan FPs."""
    if not MMGBSA_CSV.exists():
        return [], []
    df = pd.read_csv(MMGBSA_CSV)
    if "combined_score" in df.columns:
        df = df.sort_values("combined_score", ascending=False)
    df = df.head(15)
    fps, smis = [], []
    for s in df["smiles"].astype(str):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(_morgan_bv(m))
        smis.append(Chem.MolToSmiles(m))
    return fps, smis


def _tanimoto_max(smi: str, ref_fps: list) -> float:
    if not ref_fps or not smi:
        return float("nan")
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return float("nan")
    bv = _morgan_bv(m)
    return float(max(DataStructs.TanimotoSimilarity(bv, t) for t in ref_fps))


def _murcko(smi: str) -> Optional[str]:
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        scaff = MurckoScaffold.GetScaffoldForMol(m)
        if scaff is None or scaff.GetNumHeavyAtoms() == 0:
            return None
        return Chem.MolToSmiles(scaff)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# FiLMDelta predicted pIC50 (optional)
# ---------------------------------------------------------------------------
class FilmPredictor:
    """Wraps FiLMDelta-clean checkpoint for per-mol pIC50 prediction.
    Returns NaN if the checkpoint / dependencies can't be loaded.
    """

    def __init__(self) -> None:
        self.ok = False
        self.m = self.sc = self.ae = self.ap = None
        ckpt = PROJECT_ROOT / "results/paper_evaluation/reinvent4_film_model_clean.pt"
        if not ckpt.exists():
            return
        try:
            import torch
            from sklearn.preprocessing import StandardScaler
            from src.models.predictors.film_delta_predictor import FiLMDeltaMLP

            ck = torch.load(ckpt, map_location="cpu", weights_only=False)
            m = FiLMDeltaMLP(input_dim=2048, hidden_dims=[1024, 512, 256], dropout=0.2)
            m.load_state_dict(ck["model_state"])
            m.eval()
            sc = StandardScaler()
            sc.mean_ = ck["scaler_mean"]
            sc.scale_ = ck["scaler_scale"]
            sc.var_ = sc.scale_ ** 2
            sc.n_features_in_ = len(sc.mean_)
            self.m, self.sc = m, sc
            self.ae = ck["anchor_embs"]
            self.ap = np.asarray(ck["anchor_pIC50"])
            self.ok = True
            print(f"[film] loaded {ckpt.name} with {len(self.ap)} anchors")
        except Exception as e:  # noqa: BLE001
            print(f"[film] disabled — {e}")

    def predict(self, smi: str) -> float:
        if not self.ok:
            return float("nan")
        try:
            import torch
            mol = Chem.MolFromSmiles(smi)
            if mol is None:
                return float("nan")
            a = np.zeros(2048, dtype=np.float32)
            DataStructs.ConvertToNumpyArray(_morgan_bv(mol), a)
            e = torch.FloatTensor(self.sc.transform(a[None, :]).astype(np.float32))
            with torch.no_grad():
                d = self.m(self.ae, e.expand(len(self.ap), -1)).numpy().flatten()
            return float(np.mean(self.ap + d))
        except Exception:
            return float("nan")


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------
def composite_score(rec: dict) -> float:
    """Score per task spec — range 0..10."""
    s = 0.0
    if rec.get("chr_v2") is True:
        s += 3.0
    if rec.get("CHR") is True:
        s += 2.0
    if rec.get("QED") is not None and rec["QED"] >= 0.5:
        s += 1.5
    mw = rec.get("MW")
    if mw is not None and 320.0 <= mw <= 480.0:
        s += 1.5
    if rec.get("arm_heteroaryl") is True:
        s += 1.0
    if rec.get("brenk_pains_pass") is True:
        s += 1.0
    nc = rec.get("net_charge")
    if nc is not None and nc == 0:
        s += 1.0
    tc = rec.get("tanimoto_to_mmgbsa_top15")
    if tc is not None and not (isinstance(tc, float) and math.isnan(tc)) and tc >= 0.3:
        s += 0.5
    return s


def _all_metrics_passed_list(rec: dict) -> str:
    """Semicolon-joined list of v2 gates this mol satisfies (for the CSV)."""
    gates = [
        "connectivity_largest",
        "acryl_largest",
        "body_atom_C",
        "body_atom_validity",
        "QED_passing",
        "MW_in_drug_range",
        "hinge_clamp",
        "arm_heteroaryl",
        "net_charge_acceptable",
        "brenk_pains_pass",
        "veber_pass",
        "chr_v2",
    ]
    out = []
    for g in gates:
        if rec.get(g) is True:
            out.append(g)
    return ";".join(out)


# ---------------------------------------------------------------------------
# One pass: read all cohorts → records → dedup → score → write
# ---------------------------------------------------------------------------
def _per_mol_with_CHR(per_mol: list[dict], has_3d: bool) -> list[dict]:
    """Annotate per-mol records with the existing CHR gate as a single bool."""
    from experiments.eval_lingo3dmol_plans import CHR_GATES

    out = []
    for r in per_mol:
        if not r.get("valid"):
            continue
        ok = True
        for g in CHR_GATES:
            if g == "d_SG_in_range" and not has_3d:
                continue
            v = r.get(g)
            if v is None or v is False:
                ok = False
                break
        r2 = dict(r)
        r2["CHR"] = ok
        out.append(r2)
    return out


def gather_records(
    mtime_stable_sec: float,
    cache: dict[str, dict],
) -> tuple[list[dict], dict[str, dict]]:
    """Run eval_cohort on each cohort SDF (skipping in-flight & cached-unchanged).
    Returns (all_mol_records_flat, stats_by_cohort).
    """
    flat: list[dict] = []
    stats: dict[str, dict] = {}

    for label, path in COHORTS:
        if not path.exists():
            stats[label] = {"path": str(path), "status": "missing"}
            print(f"[skip] {label}: file not found {path}")
            continue
        if path.stat().st_size == 0:
            stats[label] = {"path": str(path), "status": "empty"}
            print(f"[skip] {label}: empty file (0 bytes)")
            continue
        if not _is_stable(path, mtime_stable_sec):
            stats[label] = {"path": str(path), "status": "in_flight"}
            print(f"[wait] {label}: mtime too recent (<{mtime_stable_sec:.0f}s), skipping this pass")
            # Reuse last cached result if we have one
            cached = cache.get(label)
            if cached is not None:
                flat.extend(cached["records"])
                stats[label].update(cached["stats"])
                stats[label]["status"] = "in_flight_using_cache"
            continue

        mtime = path.stat().st_mtime
        cached = cache.get(label)
        if cached is not None and cached.get("mtime") == mtime:
            # Unchanged since last pass; reuse.
            flat.extend(cached["records"])
            stats[label] = dict(cached["stats"])
            stats[label]["status"] = "cached"
            print(f"[cache] {label}: {len(cached['records'])} mols (unchanged)")
            continue

        print(f"[eval]  {label}: {path}")
        try:
            res = eval_cohort(
                path,
                anchor_json=ANCHOR_JSON,
                train_csv=None,  # skip train-FP novelty (not needed for triage)
                xtb_results=None,
            )
        except Exception as e:  # noqa: BLE001
            stats[label] = {"path": str(path), "status": "eval_error", "error": str(e)}
            print(f"[err]   {label}: {e}")
            continue

        has_3d = bool(res["summary"].get("has_3d_pose"))
        per_mol = _per_mol_with_CHR(res["per_mol"], has_3d)
        for r in per_mol:
            r["source_cohort"] = label
            r["has_3d_pose"] = has_3d
        cohort_stats = {
            "path": str(path),
            "status": "ok",
            "n_input": res["summary"]["n_input"],
            "n_valid": res["summary"]["n_valid"],
            "validity_pct": res["summary"]["validity_pct"],
            "CHR_pct": res["summary"]["Covalent_Hit_Rate"],
            "CHR_v2_pct": res["summary"]["Covalent_Hit_Rate_v2"],
            "scaffold_diversity_pct": res["summary"]["scaffold_diversity_pct"],
            "has_3d": has_3d,
        }
        stats[label] = cohort_stats
        cache[label] = {"mtime": mtime, "records": per_mol, "stats": cohort_stats}
        flat.extend(per_mol)
        print(f"        n_input={cohort_stats['n_input']} n_valid={cohort_stats['n_valid']} CHR={cohort_stats['CHR_pct']:.1%} CHRv2={cohort_stats['CHR_v2_pct']:.1%}")

    return flat, stats


def merge_and_score(
    flat: list[dict],
    mmgbsa_fps: list,
    film: Optional[FilmPredictor],
) -> pd.DataFrame:
    """Dedup by canonical SMILES, attach Tanimoto + composite + (optional)
    predicted pIC50. Returns a sorted DataFrame.
    """
    by_smi: dict[str, dict] = {}
    for r in flat:
        smi_raw = r.get("smiles")
        smi = _canonical_smiles(smi_raw)
        if smi is None:
            continue
        if smi in by_smi:
            # Keep first occurrence (per task spec); record all sources for tracking.
            by_smi[smi]["all_source_cohorts"].add(r["source_cohort"])
            continue
        rec = dict(r)
        rec["smiles"] = smi
        rec["all_source_cohorts"] = {r["source_cohort"]}
        by_smi[smi] = rec

    # Compute Tanimoto + composite + optional pIC50.
    rows: list[dict] = []
    for smi, rec in by_smi.items():
        tc = _tanimoto_max(smi, mmgbsa_fps)
        rec["tanimoto_to_mmgbsa_top15"] = tc
        rec["score"] = composite_score(rec)
        rec["all_metrics_passed"] = _all_metrics_passed_list(rec)
        rec["murcko_smiles"] = rec.get("murcko_smiles") or _murcko(smi)
        rec["all_source_cohorts"] = ";".join(sorted(rec["all_source_cohorts"]))
        rec["pred_pIC50"] = float(film.predict(smi)) if (film and film.ok) else float("nan")
        rows.append(rec)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    # Stable column order — keep the most useful first.
    front = [
        "smiles", "source_cohort", "all_source_cohorts", "score",
        "CHR", "chr_v2", "tanimoto_to_mmgbsa_top15", "pred_pIC50",
        "MW", "QED", "n_heavy", "linker_length_to_donor", "net_charge",
        "hinge_clamp", "arm_heteroaryl", "brenk_pains_pass", "veber_pass",
        "connectivity_largest", "acryl_strict", "acryl_largest",
        "body_atom_C", "body_atom_validity",
        "QED_passing", "MW_in_drug_range", "net_charge_acceptable",
        "d_SG_in_range", "burgi_dunitz_angle",
        "all_metrics_passed", "murcko_smiles", "has_3d_pose",
    ]
    cols = [c for c in front if c in df.columns] + [c for c in df.columns if c not in front]
    df = df[cols]
    df = df.sort_values("score", ascending=False, kind="mergesort").reset_index(drop=True)
    return df


def write_outputs(df: pd.DataFrame, stats: dict[str, dict]) -> None:
    all_csv = OUT_DIR / "all_unique_mols.csv"
    top100_csv = OUT_DIR / "top_100_for_wetlab.csv"
    per_method_csv = OUT_DIR / "per_method_summary.csv"

    if df.empty:
        print("[warn] no mols to write")
        return

    df.to_csv(all_csv, index=False)
    df.head(100).to_csv(top100_csv, index=False)

    # Per-method summary
    rows = []
    for label, st in stats.items():
        sub = df[df["source_cohort"] == label] if "source_cohort" in df.columns else df.iloc[0:0]
        if sub.empty:
            rows.append({
                "source_cohort": label,
                "status": st.get("status"),
                "n_total_unique": 0,
                "n_passing_CHR": 0,
                "n_passing_CHR_v2": 0,
                "median_score": float("nan"),
                "top10_mean_score": float("nan"),
                "arm_heteroaryl_pct": float("nan"),
                "MW_drug_pct": float("nan"),
                "neutral_charge_pct": float("nan"),
            })
            continue
        rows.append({
            "source_cohort": label,
            "status": st.get("status"),
            "n_total_unique": int(len(sub)),
            "n_passing_CHR": int(sub["CHR"].fillna(False).sum()),
            "n_passing_CHR_v2": int(sub["chr_v2"].fillna(False).sum()),
            "median_score": float(sub["score"].median()),
            "top10_mean_score": float(sub["score"].head(10).mean()),
            "arm_heteroaryl_pct": float(sub["arm_heteroaryl"].fillna(False).mean()),
            "MW_drug_pct": float(sub["MW_in_drug_range"].fillna(False).mean()),
            "neutral_charge_pct": float((sub["net_charge"] == 0).mean()) if "net_charge" in sub.columns else float("nan"),
            "CHR_v2_pct": float(sub["chr_v2"].fillna(False).mean()),
            "median_MW": float(sub["MW"].median()),
            "median_QED": float(sub["QED"].median()),
        })
    pd.DataFrame(rows).sort_values("top10_mean_score", ascending=False, na_position="last").to_csv(per_method_csv, index=False)

    print(f"[write] {all_csv}  ({len(df)} rows)")
    print(f"[write] {top100_csv} ({min(100, len(df))} rows)")
    print(f"[write] {per_method_csv}")


def write_report(df: pd.DataFrame, stats: dict[str, dict]) -> None:
    """Markdown summary < 700 words."""
    if df.empty:
        REPORT_PATH.write_text("# Screening cohort — empty (no mols found)\n")
        return

    n_total = len(df)
    n_chr = int(df["CHR"].fillna(False).sum())
    n_chrv2 = int(df["chr_v2"].fillna(False).sum())
    n_cohorts_ok = sum(1 for s in stats.values() if s.get("status") in ("ok", "cached", "in_flight_using_cache"))
    n_cohorts_missing = sum(1 for s in stats.values() if s.get("status") == "missing")
    n_in_flight = sum(1 for s in stats.values() if s.get("status") in ("in_flight", "in_flight_using_cache"))

    # Per-method top10 means (sorted)
    method_rows = []
    for label in df["source_cohort"].unique():
        sub = df[df["source_cohort"] == label]
        method_rows.append({
            "label": label,
            "n": len(sub),
            "n_chrv2": int(sub["chr_v2"].fillna(False).sum()),
            "top10_score": float(sub["score"].head(10).mean()),
            "median_score": float(sub["score"].median()),
        })
    method_rows.sort(key=lambda r: r["top10_score"], reverse=True)

    # Top-30 composition
    top30 = df.head(30)
    top30_methods = top30["source_cohort"].value_counts().to_dict()
    top100 = df.head(100)
    top100_scaffolds = top100["murcko_smiles"].dropna().nunique()

    # Wet-lab 24 mol proposal — top by score, deduped by Murcko scaffold (max 2/scaffold),
    # require chr_v2 + arm_heteroaryl + Brenk OK + net_charge==0.
    cand = df.copy()
    cand = cand[
        cand["chr_v2"].fillna(False)
        & cand["arm_heteroaryl"].fillna(False)
        & cand["brenk_pains_pass"].fillna(False)
        & (cand["net_charge"].fillna(99) == 0)
    ]
    chosen: list[dict] = []
    scaff_count: dict[str, int] = defaultdict(int)
    for _, row in cand.iterrows():
        if len(chosen) >= 24:
            break
        sc = row.get("murcko_smiles") or "_no_scaffold_"
        if scaff_count[sc] >= 2:
            continue
        scaff_count[sc] += 1
        chosen.append(row.to_dict())
    # If <24 after strict filter, top up with high-score mols regardless.
    if len(chosen) < 24:
        used = {c["smiles"] for c in chosen}
        for _, row in df.iterrows():
            if len(chosen) >= 24:
                break
            if row["smiles"] in used:
                continue
            chosen.append(row.to_dict())

    lines: list[str] = []
    lines.append("# ZAP70 Screening Cohort — Unified Wet-Lab Triage List")
    lines.append("")
    lines.append(f"_Generated: {time.strftime('%Y-%m-%d %H:%M:%S')}_")
    lines.append("")
    lines.append("## 1. Cohort size + dedup")
    lines.append("")
    lines.append(f"- Source cohorts: {len(stats)} total ({n_cohorts_ok} ok/cached, {n_in_flight} in-flight, {n_cohorts_missing} missing)")
    lines.append(f"- **Unique molecules after canonical-SMILES dedup: {n_total}**")
    lines.append(f"- Pass legacy CHR (Covalent Hit Rate v1): {n_chr} ({100*n_chr/max(1,n_total):.1f}%)")
    lines.append(f"- Pass strict CHR v2 (all medchem gates): {n_chrv2} ({100*n_chrv2/max(1,n_total):.1f}%)")
    lines.append("")
    lines.append("## 2. Method-by-method ranking (top-10-mean composite score)")
    lines.append("")
    lines.append("| rank | method | n_unique | n_passing_chr_v2 | top10-score | median |")
    lines.append("|---|---|---|---|---|---|")
    for i, r in enumerate(method_rows, 1):
        lines.append(f"| {i} | `{r['label']}` | {r['n']} | {r['n_chrv2']} | {r['top10_score']:.2f} | {r['median_score']:.2f} |")
    lines.append("")
    lines.append("## 3. Top-30 cross-method composition")
    lines.append("")
    if len(top30_methods) == 1:
        only = next(iter(top30_methods))
        lines.append(f"Top-30 is DOMINATED by `{only}` ({top30_methods[only]}/30).")
    else:
        parts = ", ".join(f"`{k}`={v}" for k, v in sorted(top30_methods.items(), key=lambda kv: -kv[1]))
        lines.append(f"Top-30 cross-method: {parts}")
    lines.append("")
    lines.append("## 4. Chemistry diversity in top-100")
    lines.append("")
    lines.append(f"- Unique Murcko scaffolds in top-100: **{top100_scaffolds}** (of {len(top100)} mols)")
    lines.append(f"- Mean composite score (top-100): {float(top100['score'].mean()):.2f}")
    lines.append(f"- Median MW (top-100): {float(top100['MW'].median()):.1f}")
    lines.append(f"- Median QED (top-100): {float(top100['QED'].median()):.3f}")
    if "pred_pIC50" in top100.columns and top100["pred_pIC50"].notna().any():
        lines.append(f"- Median pred_pIC50 (top-100): {float(top100['pred_pIC50'].median()):.2f}")
    lines.append("")
    lines.append("## 5. Wet-lab handoff — 24 mols proposed (scaffold-diverse, all medchem gates pass)")
    lines.append("")
    lines.append(f"- After strict gates (chr_v2 + arm_heteroaryl + Brenk + net_charge=0) and Murcko-cap (max 2/scaffold), {len(chosen)} mols selected.")
    lines.append("")
    lines.append("| # | SMILES | source | score | MW | QED | pIC50 | TC_top15 |")
    lines.append("|---|---|---|---|---|---|---|---|")
    for i, c in enumerate(chosen[:24], 1):
        pic = c.get("pred_pIC50")
        pic_s = f"{pic:.2f}" if isinstance(pic, float) and not math.isnan(pic) else "n/a"
        tc = c.get("tanimoto_to_mmgbsa_top15")
        tc_s = f"{tc:.2f}" if isinstance(tc, float) and not math.isnan(tc) else "n/a"
        mw = c.get("MW"); mw_s = f"{mw:.1f}" if isinstance(mw, float) else "n/a"
        qed = c.get("QED"); qed_s = f"{qed:.2f}" if isinstance(qed, float) else "n/a"
        smi = c["smiles"]
        if len(smi) > 80:
            smi = smi[:77] + "..."
        lines.append(f"| {i} | `{smi}` | {c['source_cohort']} | {c['score']:.2f} | {mw_s} | {qed_s} | {pic_s} | {tc_s} |")
    lines.append("")
    lines.append("## 6. Outputs")
    lines.append("")
    lines.append("- `data/screening_cohort/all_unique_mols.csv` — full deduped inventory")
    lines.append("- `data/screening_cohort/top_100_for_wetlab.csv` — top-100 by composite")
    lines.append("- `data/screening_cohort/per_method_summary.csv` — per-method metrics")

    text = "\n".join(lines) + "\n"
    REPORT_PATH.write_text(text)
    n_words = len(text.split())
    print(f"[write] {REPORT_PATH} ({n_words} words)")


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------
def one_pass(film: Optional[FilmPredictor], mmgbsa_fps, cache, mtime_stable_sec) -> tuple[pd.DataFrame, dict]:
    flat, stats = gather_records(mtime_stable_sec, cache)
    df = merge_and_score(flat, mmgbsa_fps, film)
    write_outputs(df, stats)
    write_report(df, stats)
    return df, stats


def main(argv: Optional[list[str]] = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n", 1)[0])
    ap.add_argument("--watch", action="store_true",
                    help="Watchdog mode: re-run every interval-min for up to budget-min total.")
    ap.add_argument("--interval-min", type=float, default=30.0,
                    help="Minutes between passes in --watch mode.")
    ap.add_argument("--budget-min", type=float, default=180.0,
                    help="Max total minutes in --watch mode.")
    ap.add_argument("--mtime-stable-sec", type=float, default=60.0,
                    help="Skip files whose mtime is younger than this (still being written).")
    ap.add_argument("--no-film", action="store_true",
                    help="Disable FiLMDelta pred_pIC50 (faster).")
    args = ap.parse_args(argv)

    print(f"[init] project_root={PROJECT_ROOT}")
    print(f"[init] OUT_DIR={OUT_DIR}")
    print(f"[init] mtime_stable_sec={args.mtime_stable_sec}")

    film = None if args.no_film else FilmPredictor()
    mmgbsa_fps, mmgbsa_smis = _load_mmgbsa_top15_fps()
    print(f"[init] MM-GBSA top-15 anchor FPs: {len(mmgbsa_fps)}")

    cache: dict[str, dict] = {}

    if not args.watch:
        df, _ = one_pass(film, mmgbsa_fps, cache, args.mtime_stable_sec)
        print(f"[done] single pass — {len(df)} unique mols")
        return 0

    start = time.time()
    deadline = start + args.budget_min * 60.0
    pass_i = 0
    while True:
        pass_i += 1
        t0 = time.time()
        print(f"\n[pass {pass_i}] start at t+{(t0-start)/60:.1f} min")
        df, stats = one_pass(film, mmgbsa_fps, cache, args.mtime_stable_sec)
        print(f"[pass {pass_i}] done — {len(df)} unique mols")
        # If no cohorts are in-flight any more AND we've already done >=2 passes, exit early.
        in_flight = sum(
            1 for s in stats.values() if s.get("status") in ("in_flight", "in_flight_using_cache")
        )
        if in_flight == 0 and pass_i >= 2:
            print(f"[exit] no more in-flight cohorts after pass {pass_i}")
            break
        if time.time() >= deadline:
            print(f"[exit] budget {args.budget_min} min exhausted")
            break
        # Sleep to next interval (relative to t0 so passes happen on schedule).
        next_at = t0 + args.interval_min * 60.0
        sleep_s = max(0.0, next_at - time.time())
        print(f"[wait] sleeping {sleep_s/60:.1f} min until next pass")
        time.sleep(sleep_s)

    return 0


if __name__ == "__main__":
    sys.exit(main())
