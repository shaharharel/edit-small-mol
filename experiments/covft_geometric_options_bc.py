"""Geometric comparisons B and C for the covalent-pre-reactivity question.

Two new cohorts, both with high warhead retention (so MW tests have power):
  - RL'd mol2mol: ~10k Mol1-seeded samples from exp2_v2_rl_v2 checkpoint
  - LibInvent baseline: ~10k samples with Mol1's acrylamide-isoindoline core fixed

Plus we re-use the previously computed cohorts:
  - covFT mol2mol (results/paper_evaluation/covft_geometric_*_covft.csv)
  - warhead_tokens mol2mol (new — generated here from samples_warhead_tokens.csv)

Two verdicts:
  Option B (3-way, warhead retention high in all three arms):
    LibInvent vs covFT vs RL'd — does FT or RL improve geometry over a
    warhead-fixed baseline?
  Option C (covFT vs warhead_tokens):
    Does the explicit warhead-class token improve geometry over implicit FT?

Outputs:
  results/paper_evaluation/geom_option_b_libinvent.json
  results/paper_evaluation/geom_option_c_warhead_tokens.json
  results/paper_evaluation/geom_options_bc_summary.md
  results/paper_evaluation/covft_geometric_2d_<cohort>.csv     (per-mol 2D)
  results/paper_evaluation/covft_geometric_covvina_<cohort>.csv (per-mol cov-Vina)
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import mannwhitneyu

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT_ROOT))

QURIS_BIN = "/opt/miniconda3/envs/quris/bin"
if QURIS_BIN not in os.environ.get("PATH", ""):
    os.environ["PATH"] = QURIS_BIN + os.pathsep + os.environ.get("PATH", "")

from experiments.run_covalent_docking import (  # noqa: E402
    CYS346_SG,
    build_cov_tethered_pdbqt,
    find_warhead_atoms,
    prepare_stripped_receptor,
    read_tethered_geom,
    vina_dock,
    pose_valid_covalent,
)

# ------------------------------------------------------------------
OUT_DIR = PROJECT_ROOT / "results/paper_evaluation"
WORK_DIR = PROJECT_ROOT / "data/covft_geometric_work"

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
PLANARITY_THRESH_DEG = 20.0
N_COV_VINA = 300
SUBSAMPLE_SEED = 0
WORKERS_2D = 6
WORKERS_COVDOCK = 4

# Cohort SMILES sources
COHORT_SOURCES = {
    "covft":          PROJECT_ROOT / "experiments/exp_covft_value/samples_covft.csv",
    "warhead_tokens": PROJECT_ROOT / "experiments/exp_covft_value/samples_warhead_tokens.csv",
    "rl":             PROJECT_ROOT / "experiments/exp_geom_bc/samples_rl.csv",
    "libinvent":      PROJECT_ROOT / "experiments/exp_geom_bc/samples_libinvent.csv",
}

# ------------------------------------------------------------------
def _planar_dev(d_deg: float) -> float:
    d = abs(d_deg)
    return min(d, abs(180.0 - d))


def _compute_2d_one(args):
    idx, smi = args
    out = {
        "idx": idx,
        "smi": smi,
        "acryl_match": False,
        "embed_ok": False,
        "dihedral_deg": None,
        "planar_dev_deg": None,
        "pre_reactivity_score": None,
        "msg": "",
    }
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out["msg"] = "smi_parse_fail"
            return out
        patt = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            out["msg"] = "no_acrylamide"
            return out
        out["acryl_match"] = True
        m = matches[0]
        b_idx, a_idx, c_idx, n_idx = int(m[0]), int(m[1]), int(m[2]), int(m[4])

        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        rc = AllChem.EmbedMolecule(mol_h, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol_h, p)
            if rc != 0:
                out["msg"] = "embed_fail"
                return out
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        out["embed_ok"] = True
        conf = mol_h.GetConformer()
        from rdkit.Chem.rdMolTransforms import GetDihedralDeg
        d = GetDihedralDeg(conf, b_idx, a_idx, c_idx, n_idx)
        out["dihedral_deg"] = float(d)
        dev = _planar_dev(d)
        out["planar_dev_deg"] = float(dev)
        out["pre_reactivity_score"] = 1.0 if dev <= PLANARITY_THRESH_DEG else 0.0
        out["msg"] = "ok"
    except Exception as e:
        out["msg"] = f"exc:{type(e).__name__}:{str(e)[:80]}"
    return out


def run_2d_panel(df: pd.DataFrame, cohort: str, workers: int) -> pd.DataFrame:
    tasks = [(i, smi) for i, smi in enumerate(df["SMILES"].tolist())]
    print(f"[2D] cohort={cohort} N={len(tasks)} workers={workers}", flush=True)
    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futures = [exc.submit(_compute_2d_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({
                    "idx": -1, "smi": "", "acryl_match": False, "embed_ok": False,
                    "dihedral_deg": None, "planar_dev_deg": None,
                    "pre_reactivity_score": None, "msg": f"fut_exc:{e}",
                })
            done += 1
            if done % 1000 == 0:
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                print(
                    f"  [2D {cohort}] {done}/{len(tasks)} "
                    f"{rate:.1f} mol/s  ETA {eta/60:.1f} min",
                    flush=True,
                )
    out = pd.DataFrame(rows).sort_values("idx").reset_index(drop=True)
    out["cohort"] = cohort
    return out


# ------------------------------------------------------------------
def _cov_vina_one(args):
    cohort, sub_idx, original_idx, smi, work_root_str = args
    work_root = Path(work_root_str) / cohort
    work_root.mkdir(parents=True, exist_ok=True)
    row = {
        "cohort": cohort, "subsample_idx": sub_idx, "original_idx": original_idx,
        "smi": smi, "warhead_found": False, "lig_prep_ok": False,
        "vina_ok": False, "vina_affinity": None,
        "warhead_sg_distance_A": None, "bd_angle_deg": None,
        "pose_converged": False, "msg": "", "dt_s": 0.0,
    }
    t0 = time.time()
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            row["msg"] = "smi_parse_fail"
            return row
        mol_h = Chem.AddHs(mol)
        p = AllChem.ETKDGv3()
        p.randomSeed = 42
        rc = AllChem.EmbedMolecule(mol_h, p)
        if rc != 0:
            p.useRandomCoords = True
            rc = AllChem.EmbedMolecule(mol_h, p)
            if rc != 0:
                row["msg"] = "embed_fail"
                return row
        try:
            AllChem.MMFFOptimizeMolecule(mol_h, maxIters=200)
        except Exception:
            pass
        info = find_warhead_atoms(smi)
        if info is None:
            row["msg"] = "no_warhead"
            return row
        row["warhead_found"] = True

        lig_pdbqt = work_root / f"{cohort}_{sub_idx:05d}.pdbqt"
        prep = build_cov_tethered_pdbqt(mol_h, smi, lig_pdbqt)
        if not prep.get("ok"):
            row["msg"] = f"prep_fail:{prep.get('msg')}"
            return row
        row["lig_prep_ok"] = True

        g = read_tethered_geom(lig_pdbqt, smi)
        row["warhead_sg_distance_A"] = g.get("d_sg")
        row["bd_angle_deg"] = g.get("bd_angle")

        pose_path = work_root / f"{cohort}_{sub_idx:05d}_pose.pdbqt"
        dock = vina_dock(lig_pdbqt, pose_path, mode="score_only", threads=1)
        if not dock.get("ok"):
            row["msg"] = f"vina_fail:{dock.get('msg')}"
            return row
        row["vina_ok"] = True
        row["vina_affinity"] = dock.get("score")
        row["pose_converged"] = bool(pose_valid_covalent(
            row["warhead_sg_distance_A"], row["bd_angle_deg"]
        ))
        row["msg"] = "ok"
        try:
            if sub_idx >= 200:
                lig_pdbqt.unlink(missing_ok=True)
        except Exception:
            pass
    except Exception as e:
        row["msg"] = f"exc:{type(e).__name__}:{str(e)[:120]}"
    finally:
        row["dt_s"] = time.time() - t0
    return row


def run_cov_vina(df_acryl: pd.DataFrame, cohort: str, n: int, seed: int, workers: int) -> pd.DataFrame:
    if len(df_acryl) == 0:
        return pd.DataFrame()
    rng = np.random.default_rng(seed)
    take = min(n, len(df_acryl))
    sample_idx = rng.choice(len(df_acryl), size=take, replace=False)
    sample = df_acryl.iloc[sample_idx].reset_index(drop=True)
    sample["subsample_idx"] = np.arange(len(sample))

    print(f"[CovVina] cohort={cohort} subsample={len(sample)} workers={workers}", flush=True)
    tasks = [
        (cohort, int(r["subsample_idx"]), int(r["idx"]), r["smi"], str(WORK_DIR))
        for _, r in sample.iterrows()
    ]
    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=workers) as exc:
        futures = [exc.submit(_cov_vina_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({
                    "cohort": cohort, "subsample_idx": -1, "original_idx": -1,
                    "smi": "", "warhead_found": False, "lig_prep_ok": False,
                    "vina_ok": False, "vina_affinity": None,
                    "warhead_sg_distance_A": None, "bd_angle_deg": None,
                    "pose_converged": False, "msg": f"fut_exc:{e}", "dt_s": 0.0,
                })
            done += 1
            if done % 20 == 0 or done == len(tasks):
                dt = time.time() - t0
                rate = done / max(dt, 1e-6)
                eta = (len(tasks) - done) / max(rate, 1e-6)
                ok = sum(1 for r in rows if r.get("vina_ok"))
                print(
                    f"  [CovVina {cohort}] {done}/{len(tasks)} ok={ok} "
                    f"{rate:.2f} mol/s  ETA {eta/60:.1f} min",
                    flush=True,
                )
    return pd.DataFrame(rows).sort_values("subsample_idx").reset_index(drop=True)


# ------------------------------------------------------------------
def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ via scipy Mann-Whitney U. δ = 2U/(n*m) - 1.

    Critical: must use a tie-aware U statistic. We delegate to scipy's
    mannwhitneyu (which handles ties by counting ties as 0.5 contribution),
    not ordinal ranks (which silently break for binary/categorical inputs).
    Positive δ => x tends to be larger than y.
    """
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    try:
        u, _ = mannwhitneyu(x, y, alternative="two-sided")
    except Exception:
        return float("nan")
    return float(2.0 * u / (n * m) - 1.0)


def mw_summary(a: np.ndarray, b: np.ndarray, label_a: str, label_b: str,
               alternative: str = "two-sided") -> dict:
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    out = {
        "label_a": label_a, "label_b": label_b,
        "n_a": int(len(a)), "n_b": int(len(b)),
        "median_a": float(np.median(a)) if len(a) else None,
        "median_b": float(np.median(b)) if len(b) else None,
        "U": None, "p_value": None,
        "cliffs_delta_a_vs_b": None,
        "alternative": alternative,
    }
    if len(a) < 5 or len(b) < 5:
        return out
    try:
        u, p = mannwhitneyu(a, b, alternative=alternative)
        out["U"] = float(u); out["p_value"] = float(p)
    except Exception as e:
        out["err"] = str(e)
    out["cliffs_delta_a_vs_b"] = cliffs_delta(a, b)
    return out


def iqr(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (None, None)
    return (float(np.percentile(x, 25)), float(np.percentile(x, 75)))


def cohort_summary(df_2d: pd.DataFrame, df_cov: pd.DataFrame) -> dict:
    n_total = len(df_2d)
    acryl_mask = df_2d["acryl_match"].astype(bool)
    geom_ok = df_2d["embed_ok"].astype(bool) & df_2d["pre_reactivity_score"].notna()
    pre = df_2d.loc[geom_ok, "pre_reactivity_score"].astype(float).values
    dev = df_2d.loc[geom_ok, "planar_dev_deg"].astype(float).values

    cov_ok = df_cov[df_cov["vina_ok"] == True] if len(df_cov) else df_cov  # noqa: E712
    aff = cov_ok["vina_affinity"].astype(float).values if len(cov_ok) else np.array([])
    dist = cov_ok["warhead_sg_distance_A"].astype(float).values if len(cov_ok) else np.array([])
    bd = cov_ok["bd_angle_deg"].astype(float).values if len(cov_ok) else np.array([])
    conv = (df_cov["pose_converged"].astype(bool).mean() if len(df_cov) else None)

    return {
        "n_total": int(n_total),
        "n_acrylamide_match": int(acryl_mask.sum()),
        "frac_acrylamide_match": float(acryl_mask.mean()) if n_total else None,
        "n_geom_evaluated": int(geom_ok.sum()),
        "pre_reactivity_score_mean": float(pre.mean()) if len(pre) else None,
        "pre_reactivity_score_std": float(pre.std(ddof=1)) if len(pre) > 1 else None,
        "pre_reactivity_score_median": float(np.median(pre)) if len(pre) else None,
        "pre_reactivity_score_frac_ge_0p5": float((pre >= 0.5).mean()) if len(pre) else None,
        "planar_dev_deg_median": float(np.median(dev)) if len(dev) else None,
        "planar_dev_deg_iqr": iqr(dev),
        "cov_vina_n_attempted": int(len(df_cov)),
        "cov_vina_n_ok": int(len(cov_ok)),
        "vina_affinity_median": float(np.median(aff)) if len(aff) else None,
        "vina_affinity_iqr": iqr(aff),
        "vina_affinity_mean": float(aff.mean()) if len(aff) else None,
        "vina_affinity_std": float(aff.std(ddof=1)) if len(aff) > 1 else None,
        "warhead_sg_distance_median_A": float(np.median(dist)) if len(dist) else None,
        "warhead_sg_distance_iqr_A": iqr(dist),
        "bd_angle_median_deg": float(np.median(bd)) if len(bd) else None,
        "bd_angle_iqr_deg": iqr(bd),
        "pose_converged_rate": float(conv) if conv is not None else None,
    }


# ------------------------------------------------------------------
def load_or_compute_cohort(cohort: str, csv_path: Path, workers_2d: int, workers_cov: int,
                            n_cov: int, seed: int, force: bool = False) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load existing per-mol CSVs if they exist (and force=False); else compute."""
    out_2d = OUT_DIR / f"covft_geometric_2d_{cohort}.csv"
    out_cov = OUT_DIR / f"covft_geometric_covvina_{cohort}.csv"
    if (not force) and out_2d.exists() and out_cov.exists():
        print(f"[load] cohort={cohort} from cached CSVs", flush=True)
        df_2d = pd.read_csv(out_2d)
        df_cov = pd.read_csv(out_cov)
        # Coerce dtypes that may have been written as text
        for col in ("acryl_match", "embed_ok"):
            if col in df_2d.columns:
                df_2d[col] = df_2d[col].astype(bool)
        for col in ("warhead_found", "lig_prep_ok", "vina_ok", "pose_converged"):
            if col in df_cov.columns:
                df_cov[col] = df_cov[col].astype(bool)
        return df_2d, df_cov
    print(f"[compute] cohort={cohort} from {csv_path}", flush=True)
    df = pd.read_csv(csv_path)
    assert "SMILES" in df.columns, f"missing SMILES col in {csv_path}"
    df_2d = run_2d_panel(df, cohort, workers_2d)
    df_2d.to_csv(out_2d, index=False)
    acryl = df_2d[df_2d["acryl_match"] == True].reset_index(drop=True)  # noqa: E712
    df_cov = run_cov_vina(acryl, cohort, n_cov, seed, workers_cov)
    df_cov.to_csv(out_cov, index=False)
    return df_2d, df_cov


def pairwise_stats(name_a: str, df2d_a: pd.DataFrame, dfcov_a: pd.DataFrame,
                   name_b: str, df2d_b: pd.DataFrame, dfcov_b: pd.DataFrame) -> dict:
    pre_a = df2d_a.loc[df2d_a["pre_reactivity_score"].notna(),
                        "pre_reactivity_score"].astype(float).values
    pre_b = df2d_b.loc[df2d_b["pre_reactivity_score"].notna(),
                        "pre_reactivity_score"].astype(float).values
    dev_a = df2d_a.loc[df2d_a["planar_dev_deg"].notna(),
                        "planar_dev_deg"].astype(float).values
    dev_b = df2d_b.loc[df2d_b["planar_dev_deg"].notna(),
                        "planar_dev_deg"].astype(float).values

    aff_a = dfcov_a.loc[dfcov_a["vina_ok"].astype(bool),
                         "vina_affinity"].astype(float).values if len(dfcov_a) else np.array([])
    aff_b = dfcov_b.loc[dfcov_b["vina_ok"].astype(bool),
                         "vina_affinity"].astype(float).values if len(dfcov_b) else np.array([])
    dist_a = dfcov_a.loc[dfcov_a["vina_ok"].astype(bool),
                          "warhead_sg_distance_A"].astype(float).values if len(dfcov_a) else np.array([])
    dist_b = dfcov_b.loc[dfcov_b["vina_ok"].astype(bool),
                          "warhead_sg_distance_A"].astype(float).values if len(dfcov_b) else np.array([])

    return {
        "name_a": name_a, "name_b": name_b,
        "mw_pre_reactivity_score": mw_summary(pre_a, pre_b, name_a, name_b),
        "mw_planar_dev_deg":       mw_summary(dev_a, dev_b, name_a, name_b),
        "mw_vina_affinity":        mw_summary(aff_a, aff_b, name_a, name_b),
        "mw_warhead_sg_distance":  mw_summary(dist_a, dist_b, name_a, name_b),
    }


def pairwise_verdict(stats: dict, label_a: str, label_b: str) -> dict:
    """Verdict for 'a is more geometrically pre-reactive than b'."""
    alpha = 0.05
    p_pre = stats["mw_pre_reactivity_score"]["p_value"]
    d_pre = stats["mw_pre_reactivity_score"]["cliffs_delta_a_vs_b"]
    p_dev = stats["mw_planar_dev_deg"]["p_value"]
    d_dev = stats["mw_planar_dev_deg"]["cliffs_delta_a_vs_b"]
    p_aff = stats["mw_vina_affinity"]["p_value"]
    d_aff = stats["mw_vina_affinity"]["cliffs_delta_a_vs_b"]

    pre_better = (p_pre is not None) and (p_pre < alpha) and (d_pre is not None) and (d_pre > 0)
    dev_better = (p_dev is not None) and (p_dev < alpha) and (d_dev is not None) and (d_dev < 0)
    aff_better = (p_aff is not None) and (p_aff < alpha) and (d_aff is not None) and (d_aff < 0)
    pos = int(pre_better) + int(dev_better) + int(aff_better)
    if pos >= 2:
        verdict = f"{label_a} IS significantly more geometrically pre-reactive than {label_b}"
    elif pos == 1:
        verdict = f"Mixed: {label_a} significantly better on one axis only vs {label_b}"
    else:
        verdict = f"{label_a} is NOT significantly more geometrically pre-reactive than {label_b}"
    return {
        "pre_reactivity_significant_and_favors_a": bool(pre_better),
        "planar_dev_significant_and_favors_a": bool(dev_better),
        "vina_affinity_significant_and_favors_a": bool(aff_better),
        "positive_axes_count": pos,
        "alpha": alpha,
        "verdict": verdict,
    }


# ------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers-2d", type=int, default=WORKERS_2D)
    ap.add_argument("--workers-cov", type=int, default=WORKERS_COVDOCK)
    ap.add_argument("--n-cov-vina", type=int, default=N_COV_VINA)
    ap.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    ap.add_argument("--cohorts", nargs="+", default=["covft", "warhead_tokens", "rl", "libinvent"],
                    help="Which cohorts to load/compute. covft has cached results already.")
    ap.add_argument("--force-recompute", action="store_true",
                    help="Recompute even if cached CSVs exist.")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    prepare_stripped_receptor()

    # Verify all source files exist
    for cohort in args.cohorts:
        src = COHORT_SOURCES.get(cohort)
        if src is None:
            raise SystemExit(f"unknown cohort {cohort}")
        if not src.exists():
            raise SystemExit(f"missing cohort source: {src}")

    # Load/compute each
    cohort_data: dict[str, tuple[pd.DataFrame, pd.DataFrame]] = {}
    for cohort in args.cohorts:
        df2d, dfcov = load_or_compute_cohort(
            cohort, COHORT_SOURCES[cohort],
            args.workers_2d, args.workers_cov,
            args.n_cov_vina, args.seed,
            force=args.force_recompute,
        )
        cohort_data[cohort] = (df2d, dfcov)

    # Per-cohort summary
    cohort_summaries = {
        c: cohort_summary(df2d, dfcov) for c, (df2d, dfcov) in cohort_data.items()
    }

    # ---------- Option B (3-way LibInvent / covFT / RL) ----------
    pairs_b = [
        ("covft", "libinvent"),    # FT vs baseline
        ("rl",    "libinvent"),    # RL vs baseline
        ("rl",    "covft"),        # RL vs FT
    ]
    option_b = {
        "config": {
            "cohorts": ["libinvent", "covft", "rl"],
            "alpha": 0.05,
            "cov_vina_subsample_n": args.n_cov_vina,
            "subsample_seed": args.seed,
            "acrylamide_smarts": ACRYLAMIDE_SMARTS,
            "planarity_threshold_deg": PLANARITY_THRESH_DEG,
        },
        "cohort_summaries": {k: cohort_summaries[k] for k in ("libinvent", "covft", "rl") if k in cohort_summaries},
        "pairwise_tests": {},
        "headline_verdict": None,
    }
    pos_axes_count_per_pair: dict[str, int] = {}
    for a, b in pairs_b:
        if a not in cohort_data or b not in cohort_data:
            continue
        df2d_a, dfcov_a = cohort_data[a]
        df2d_b, dfcov_b = cohort_data[b]
        stats = pairwise_stats(a, df2d_a, dfcov_a, b, df2d_b, dfcov_b)
        verdict = pairwise_verdict(stats, a, b)
        key = f"{a}_vs_{b}"
        option_b["pairwise_tests"][key] = {"stats": stats, "verdict": verdict}
        pos_axes_count_per_pair[key] = verdict["positive_axes_count"]

    # Headline for B: does FT or RL beat LibInvent?
    covft_vs_lib = option_b["pairwise_tests"].get("covft_vs_libinvent", {}).get("verdict")
    rl_vs_lib    = option_b["pairwise_tests"].get("rl_vs_libinvent", {}).get("verdict")
    rl_vs_covft  = option_b["pairwise_tests"].get("rl_vs_covft", {}).get("verdict")
    def _pos(v):  # safely extract positive_axes_count
        return (v or {}).get("positive_axes_count", 0)
    n_pos_covft = _pos(covft_vs_lib)
    n_pos_rl    = _pos(rl_vs_lib)
    if (n_pos_covft >= 2) or (n_pos_rl >= 2):
        head = (
            "FT and/or RL DO produce geometrically more covalent-pre-reactive poses "
            "than a warhead-fixed (LibInvent) baseline (positive axes: "
            f"covFT-vs-LibInvent={n_pos_covft}/3, RL-vs-LibInvent={n_pos_rl}/3)."
        )
    elif (n_pos_covft == 1) or (n_pos_rl == 1):
        head = (
            "Mixed signal: FT and/or RL beat LibInvent on only one axis "
            f"(covFT-vs-LibInvent={n_pos_covft}/3, RL-vs-LibInvent={n_pos_rl}/3). "
            "Soften the 'geometric pre-reactivity' claim."
        )
    else:
        head = (
            "No evidence that FT or RL produce geometrically more pre-reactive "
            "poses than a warhead-fixed (LibInvent) baseline."
        )
    option_b["headline_verdict"] = {
        "summary": head,
        "covft_vs_libinvent_positive_axes": n_pos_covft,
        "rl_vs_libinvent_positive_axes": n_pos_rl,
        "rl_vs_covft_positive_axes": _pos(rl_vs_covft),
    }

    # ---------- Option C (covFT vs warhead_tokens) ----------
    option_c = {
        "config": {
            "cohorts": ["covft", "warhead_tokens"],
            "alpha": 0.05,
            "cov_vina_subsample_n": args.n_cov_vina,
            "subsample_seed": args.seed,
            "acrylamide_smarts": ACRYLAMIDE_SMARTS,
            "planarity_threshold_deg": PLANARITY_THRESH_DEG,
        },
        "cohort_summaries": {k: cohort_summaries[k] for k in ("covft", "warhead_tokens") if k in cohort_summaries},
        "pairwise_tests": {},
        "headline_verdict": None,
    }
    if "warhead_tokens" in cohort_data and "covft" in cohort_data:
        df2d_wt, dfcov_wt = cohort_data["warhead_tokens"]
        df2d_ft, dfcov_ft = cohort_data["covft"]
        # Test direction: warhead_tokens vs covft (does explicit token improve over implicit FT?)
        stats = pairwise_stats("warhead_tokens", df2d_wt, dfcov_wt, "covft", df2d_ft, dfcov_ft)
        verdict = pairwise_verdict(stats, "warhead_tokens", "covft")
        option_c["pairwise_tests"]["warhead_tokens_vs_covft"] = {"stats": stats, "verdict": verdict}
        n_pos = verdict["positive_axes_count"]
        if n_pos >= 2:
            head_c = (
                f"Explicit warhead-class token (warhead_tokens) IS geometrically "
                f"more pre-reactive than implicit FT (covFT) — positive axes={n_pos}/3."
            )
        elif n_pos == 1:
            head_c = (
                f"Mixed: warhead_tokens beats covFT on only one axis ({n_pos}/3). "
                "Soften the 'explicit token improves geometry' claim."
            )
        else:
            head_c = (
                "No evidence that the explicit warhead-class token improves "
                "geometric pre-reactivity over implicit FT."
            )
        option_c["headline_verdict"] = {
            "summary": head_c,
            "warhead_tokens_vs_covft_positive_axes": n_pos,
        }

    out_b = OUT_DIR / "geom_option_b_libinvent.json"
    out_c = OUT_DIR / "geom_option_c_warhead_tokens.json"
    out_b.write_text(json.dumps(option_b, indent=2, default=str))
    out_c.write_text(json.dumps(option_c, indent=2, default=str))
    print(f"Wrote {out_b}")
    print(f"Wrote {out_c}")

    # ----- Markdown summary -----
    md = build_markdown(option_b, option_c, cohort_summaries)
    out_md = OUT_DIR / "geom_options_bc_summary.md"
    out_md.write_text(md)
    print(f"Wrote {out_md}")

    print("\n" + "=" * 60)
    print("OPTION B headline:", option_b["headline_verdict"]["summary"])
    print("=" * 60)
    if option_c["headline_verdict"]:
        print("OPTION C headline:", option_c["headline_verdict"]["summary"])
    print("=" * 60)


def fmt(v, nd=3):
    if v is None or (isinstance(v, float) and not np.isfinite(v)):
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def fmt_iqr(t, nd=3):
    if t is None or (isinstance(t, (list, tuple)) and (t[0] is None or t[1] is None)):
        return "n/a"
    return f"[{fmt(t[0], nd)}, {fmt(t[1], nd)}]"


def _cohort_row(name: str, s: dict) -> str:
    return (
        f"| {name} | {s['n_total']} | {s['n_acrylamide_match']} | "
        f"{fmt(s['frac_acrylamide_match'], 4)} | "
        f"{s['n_geom_evaluated']} | "
        f"{fmt(s['pre_reactivity_score_mean'])} ± {fmt(s['pre_reactivity_score_std'])} | "
        f"{fmt(s['pre_reactivity_score_median'])} | "
        f"{fmt(s['planar_dev_deg_median'], 2)}° | "
        f"{fmt_iqr(s['planar_dev_deg_iqr'], 2)}° | "
        f"{s['cov_vina_n_ok']}/{s['cov_vina_n_attempted']} | "
        f"{fmt(s['vina_affinity_median'], 2)} | "
        f"{fmt_iqr(s['vina_affinity_iqr'], 2)} | "
        f"{fmt(s['warhead_sg_distance_median_A'], 3)} | "
        f"{fmt(s['pose_converged_rate'], 3)} |"
    )


def _pair_row(name: str, stats: dict) -> str:
    return (
        f"| {name} | {stats['n_a']} | {stats['n_b']} | "
        f"{fmt(stats['median_a'])} | {fmt(stats['median_b'])} | "
        f"{fmt(stats['U'], 1)} | {fmt(stats['p_value'], 4)} | "
        f"{fmt(stats['cliffs_delta_a_vs_b'], 3)} |"
    )


def build_markdown(option_b: dict, option_c: dict, cohort_summaries: dict) -> str:
    md = []
    md.append("# Geometric Pre-Reactivity — Options B and C\n")
    md.append("Question: do covFT and/or RL place the warhead in geometrically better poses than a")
    md.append("warhead-fixed baseline (Option B), and does the explicit warhead-class token improve")
    md.append("geometry over implicit FT (Option C)?\n")

    md.append("## Inputs and method\n")
    md.append("- 2D pre-reactivity: per-mol RDKit ETKDG + MMFF on every SMILES matching "
              f"`{ACRYLAMIDE_SMARTS}`; dihedral over Cβ-Cα-C(=O)-N; "
              f"planar deviation = min(|d|, |180-|d||); score=1 if dev ≤ {PLANARITY_THRESH_DEG}°.\n")
    md.append(f"- Cov-Vina (N={N_COV_VINA}/cohort, seed={SUBSAMPLE_SEED}): meeko CovalentBuilder "
              "tethers warhead Cβ to Cys346 CB; Vina --score_only against the Cys346-stripped "
              "ZAP70 (4K2R) receptor.\n")

    md.append("## Cohort summaries\n")
    md.append("| Cohort | N | N(acryl) | frac(acryl) | N(2D) | "
              "PreReact mean ± std | PreReact median | PlanarDev median | PlanarDev IQR | "
              "CovVina ok/att | Vina median | Vina IQR | warhead-SG median (Å) | pose_conv |")
    md.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for cohort in ("libinvent", "covft", "rl", "warhead_tokens"):
        if cohort in cohort_summaries:
            md.append(_cohort_row(cohort, cohort_summaries[cohort]))
    md.append("")

    md.append("## Option B — LibInvent (warhead-fixed) vs covFT vs RL\n")
    md.append("Cliff's δ is signed (a − b); positive δ means a has larger values.\n")
    for pair_key, payload in option_b.get("pairwise_tests", {}).items():
        st = payload["stats"]
        v = payload["verdict"]
        md.append(f"### {pair_key}\n")
        md.append("| Metric | n_a | n_b | median_a | median_b | U | p-value | Cliff's δ |")
        md.append("|---|---|---|---|---|---|---|---|")
        md.append(_pair_row("pre_reactivity_score (higher=more aligned)",
                            st["mw_pre_reactivity_score"]))
        md.append(_pair_row("planar_dev_deg (lower=more aligned)",
                            st["mw_planar_dev_deg"]))
        md.append(_pair_row("vina_affinity kcal/mol (lower=better)",
                            st["mw_vina_affinity"]))
        md.append(_pair_row("warhead-SG distance Å",
                            st["mw_warhead_sg_distance"]))
        md.append("")
        md.append(f"- Pre-reactivity favors {st['mw_pre_reactivity_score']['label_a']}: "
                  f"**{v['pre_reactivity_significant_and_favors_a']}**")
        md.append(f"- Planar deviation favors {st['mw_planar_dev_deg']['label_a']}: "
                  f"**{v['planar_dev_significant_and_favors_a']}**")
        md.append(f"- Vina-cov affinity favors {st['mw_vina_affinity']['label_a']}: "
                  f"**{v['vina_affinity_significant_and_favors_a']}**")
        md.append(f"- Positive axes count: {v['positive_axes_count']} / 3 — *{v['verdict']}*\n")

    hv = option_b.get("headline_verdict") or {}
    md.append(f"**Option B headline verdict**: {hv.get('summary','n/a')}\n")

    md.append("## Option C — covFT vs warhead_tokens\n")
    for pair_key, payload in option_c.get("pairwise_tests", {}).items():
        st = payload["stats"]
        v = payload["verdict"]
        md.append(f"### {pair_key}\n")
        md.append("| Metric | n_a | n_b | median_a | median_b | U | p-value | Cliff's δ |")
        md.append("|---|---|---|---|---|---|---|---|")
        md.append(_pair_row("pre_reactivity_score (higher=more aligned)",
                            st["mw_pre_reactivity_score"]))
        md.append(_pair_row("planar_dev_deg (lower=more aligned)",
                            st["mw_planar_dev_deg"]))
        md.append(_pair_row("vina_affinity kcal/mol (lower=better)",
                            st["mw_vina_affinity"]))
        md.append(_pair_row("warhead-SG distance Å",
                            st["mw_warhead_sg_distance"]))
        md.append("")
        md.append(f"- Pre-reactivity favors {st['mw_pre_reactivity_score']['label_a']}: "
                  f"**{v['pre_reactivity_significant_and_favors_a']}**")
        md.append(f"- Planar deviation favors {st['mw_planar_dev_deg']['label_a']}: "
                  f"**{v['planar_dev_significant_and_favors_a']}**")
        md.append(f"- Vina-cov affinity favors {st['mw_vina_affinity']['label_a']}: "
                  f"**{v['vina_affinity_significant_and_favors_a']}**")
        md.append(f"- Positive axes count: {v['positive_axes_count']} / 3 — *{v['verdict']}*\n")

    hvc = option_c.get("headline_verdict") or {}
    md.append(f"**Option C headline verdict**: {hvc.get('summary','n/a')}\n")
    return "\n".join(md)


if __name__ == "__main__":
    sys.exit(main())
