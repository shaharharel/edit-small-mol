"""Paired geometric-covalent comparison: covFT vs base mol2mol cohorts.

Two metrics, two cohorts:
  A. 2D acrylamide pre-reactivity score (RDKit ETKDG + MMFF, all 10k each).
     - Dihedral over Cβ-Cα-C(=O)-N (atoms 0,1,2,3 of [CH2]=[CH]C(=O)N).
     - Planar deviation = min(|d|, |180 - |d||).  ≤20° => score 1, else 0.
  B. Covalent Vina with Cys346 restraint (N=300 random subsample per cohort, seed=0).
     - Uses experiments.run_covalent_docking primitives (meeko CovalentBuilder
       tether + Vina --score_only against Cys346-stripped 4K2R receptor).
     - Records vina_affinity, warhead_sg_distance_A, pose_converged.

Cohort-level summary stats + Mann-Whitney U + Cliff's δ for each metric.
Headline question: covFT cohort more pre-reactive / better cov-Vina than base?

Outputs:
  results/paper_evaluation/covft_geometric_comparison.json
  results/paper_evaluation/covft_geometric_comparison.md
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

# Force quris env's bin onto PATH for Vina + meeko subprocesses
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
# Configuration
# ------------------------------------------------------------------

BASE_CSV = PROJECT_ROOT / "experiments/exp_covft_value/samples_base.csv"
COVFT_CSV = PROJECT_ROOT / "experiments/exp_covft_value/samples_covft.csv"
OUT_DIR = PROJECT_ROOT / "results/paper_evaluation"
OUT_JSON = OUT_DIR / "covft_geometric_comparison.json"
OUT_MD = OUT_DIR / "covft_geometric_comparison.md"
WORK_DIR = PROJECT_ROOT / "data/covft_geometric_work"

ACRYLAMIDE_SMARTS = "[CH2]=[CH]C(=O)N"
PLANARITY_THRESH_DEG = 20.0
N_COV_VINA = 300
SUBSAMPLE_SEED = 0
WORKERS_2D = 6
WORKERS_COVDOCK = 4


# ------------------------------------------------------------------
# 2D pre-reactivity (per-mol worker)
# ------------------------------------------------------------------

def _planar_dev(d_deg: float) -> float:
    """Distance of dihedral to nearest planar conformer (s-cis=0° or s-trans=180°)."""
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
        # SMARTS [CH2]=[CH]C(=O)N matches 5 atoms: (β=CH2, α=CH, C, O, N).
        # We want dihedral over β - α - C(=O) - N => atoms 0, 1, 2, 4.
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
    """Run 2D pre-reactivity on every SMILES in df. Returns per-mol DataFrame."""
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
# Cov-Vina worker (single mol)
# ------------------------------------------------------------------

def _cov_vina_one(args):
    cohort, sub_idx, original_idx, smi, work_root_str = args
    work_root = Path(work_root_str) / cohort
    work_root.mkdir(parents=True, exist_ok=True)
    row = {
        "cohort": cohort,
        "subsample_idx": sub_idx,
        "original_idx": original_idx,
        "smi": smi,
        "warhead_found": False,
        "lig_prep_ok": False,
        "vina_ok": False,
        "vina_affinity": None,
        "warhead_sg_distance_A": None,
        "bd_angle_deg": None,
        "pose_converged": False,
        "msg": "",
        "dt_s": 0.0,
    }
    t0 = time.time()
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            row["msg"] = "smi_parse_fail"
            return row

        # Generate 3D conformer (CovalentBuilder will re-align anyway)
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

        # Build covalently tethered PDBQT via meeko CovalentBuilder.
        lig_pdbqt = work_root / f"{cohort}_{sub_idx:05d}.pdbqt"
        prep = build_cov_tethered_pdbqt(mol_h, smi, lig_pdbqt)
        if not prep.get("ok"):
            row["msg"] = f"prep_fail:{prep.get('msg')}"
            return row
        row["lig_prep_ok"] = True

        # Read geometry directly from the tethered PDBQT (canonical AD-CovDock).
        g = read_tethered_geom(lig_pdbqt, smi)
        row["warhead_sg_distance_A"] = g.get("d_sg")
        row["bd_angle_deg"] = g.get("bd_angle")

        # Vina --score_only against the Cys346-stripped receptor.
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

        # Clean tethered pdbqt to keep disk usage bounded (keep last 200 for debug).
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
    """Run covalent-Vina on a random subsample of acrylamide-matching molecules."""
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
# Stats
# ------------------------------------------------------------------

def cliffs_delta(x: np.ndarray, y: np.ndarray) -> float:
    """Cliff's δ via rank trick: δ = 2U/(n*m) - 1 where U is M-W U on x>y."""
    x = np.asarray(x, dtype=float); y = np.asarray(y, dtype=float)
    x = x[np.isfinite(x)]; y = y[np.isfinite(y)]
    n, m = len(x), len(y)
    if n == 0 or m == 0:
        return float("nan")
    # Pairwise sign count via O(n log n) using sorting.
    combined = np.concatenate([x, y])
    ranks = np.argsort(np.argsort(combined)) + 1
    rx = ranks[:n].sum()
    U_xy = rx - n * (n + 1) / 2.0  # # pairs where x[i] > y[j], ties=0.5
    delta = (2.0 * U_xy) / (n * m) - 1.0
    return float(delta)


def mw_summary(a: np.ndarray, b: np.ndarray, alternative: str = "two-sided") -> dict:
    """Mann-Whitney U test on finite values of a vs b. Returns p, U, n_a, n_b, cliffs_delta."""
    a = np.asarray(a, dtype=float); b = np.asarray(b, dtype=float)
    a = a[np.isfinite(a)]; b = b[np.isfinite(b)]
    out = {
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
        out["U"] = None; out["p_value"] = None; out["err"] = str(e)
    out["cliffs_delta_a_vs_b"] = cliffs_delta(a, b)
    return out


def iqr(x):
    x = np.asarray(x, dtype=float); x = x[np.isfinite(x)]
    if len(x) == 0:
        return (None, None)
    return (float(np.percentile(x, 25)), float(np.percentile(x, 75)))


def cohort_summary(df_2d: pd.DataFrame, df_cov: pd.DataFrame) -> dict:
    """Per-cohort summary numbers."""
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


def headline_verdict(stats: dict) -> dict:
    """Compose verdict on 'covFT more covalent-pre-reactive than base'."""
    p_pre = stats["mw_pre_reactivity_score"].get("p_value")
    delta_pre = stats["mw_pre_reactivity_score"].get("cliffs_delta_a_vs_b")
    p_dev = stats["mw_planar_dev_deg"].get("p_value")
    delta_dev = stats["mw_planar_dev_deg"].get("cliffs_delta_a_vs_b")
    p_aff = stats["mw_vina_affinity"].get("p_value")
    delta_aff = stats["mw_vina_affinity"].get("cliffs_delta_a_vs_b")
    p_dist = stats["mw_warhead_sg_distance"].get("p_value")
    delta_dist = stats["mw_warhead_sg_distance"].get("cliffs_delta_a_vs_b")

    alpha = 0.05

    # covFT > base on pre-reactivity score (higher = more s-cis aligned)
    pre_better = (p_pre is not None) and (p_pre < alpha) and (delta_pre is not None) and (delta_pre > 0)
    # planar deviation: smaller = better; covFT - base should be NEGATIVE
    dev_better = (p_dev is not None) and (p_dev < alpha) and (delta_dev is not None) and (delta_dev < 0)
    # vina affinity: lower (more negative) is better => covFT < base => delta < 0
    aff_better = (p_aff is not None) and (p_aff < alpha) and (delta_aff is not None) and (delta_aff < 0)
    # warhead-SG distance: closer to 1.85 Å is better; lower if base is far,
    # but tethered AD-CovDock always uses meeko's exact tether (so values near
    # 1.85 by construction).  We still report it but don't weight it heavily.
    dist_signal = (p_dist is not None) and (p_dist < alpha)

    # Headline: average effect-size sign of the two "is more covalent" axes.
    # Pre-reactivity dominates (it's the 10k-mol 2D test); cov-Vina is the
    # confirmatory 300-mol sample.
    headline = {
        "pre_reactivity_significant_and_favors_covft": bool(pre_better),
        "planar_dev_significant_and_favors_covft": bool(dev_better),
        "vina_affinity_significant_and_favors_covft": bool(aff_better),
        "warhead_sg_distance_test_significant": bool(dist_signal),
        "alpha": alpha,
    }
    # Interpretation:
    pos = int(pre_better) + int(dev_better) + int(aff_better)
    if pos >= 2:
        verdict = "covFT IS significantly more covalent-pre-reactive than base"
        recommend = "Defensible to retain a 'more covalent' claim in the paper."
    elif pos == 1:
        verdict = "Mixed signal: covFT significantly better on one axis only"
        recommend = (
            "Soften the 'more covalent' claim in the paper; "
            "report only the axis on which the effect is significant."
        )
    else:
        verdict = "covFT is NOT significantly more covalent-pre-reactive than base"
        recommend = "Drop the 'more covalent' claim from the paper."
    headline["verdict"] = verdict
    headline["recommend"] = recommend
    headline["positive_axes_count"] = pos
    return headline


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-csv", default=str(BASE_CSV))
    ap.add_argument("--covft-csv", default=str(COVFT_CSV))
    ap.add_argument("--n-cov-vina", type=int, default=N_COV_VINA)
    ap.add_argument("--seed", type=int, default=SUBSAMPLE_SEED)
    ap.add_argument("--workers-2d", type=int, default=WORKERS_2D)
    ap.add_argument("--workers-cov", type=int, default=WORKERS_COVDOCK)
    ap.add_argument("--skip-cov-vina", action="store_true", help="2D only, no cov-vina")
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    WORK_DIR.mkdir(parents=True, exist_ok=True)

    print(f"PROJECT_ROOT={PROJECT_ROOT}")
    print(f"BASE_CSV={args.base_csv}")
    print(f"COVFT_CSV={args.covft_csv}")
    print(f"OUT_JSON={OUT_JSON}")
    print(f"OUT_MD={OUT_MD}")

    # Receptor must exist; build it if missing.
    prepare_stripped_receptor()

    # Load cohorts.
    df_base = pd.read_csv(args.base_csv)
    df_covft = pd.read_csv(args.covft_csv)
    assert "SMILES" in df_base.columns and "SMILES" in df_covft.columns
    print(f"base N={len(df_base)}  covft N={len(df_covft)}")

    # -------- A. 2D pre-reactivity for entire cohorts --------
    t0 = time.time()
    df_2d_base = run_2d_panel(df_base, "base", args.workers_2d)
    df_2d_covft = run_2d_panel(df_covft, "covft", args.workers_2d)
    print(f"[2D] DONE in {(time.time()-t0)/60:.1f} min")

    # Cache full per-mol tables to CSV for downstream inspection.
    (OUT_DIR / "covft_geometric_2d_base.csv").write_text(df_2d_base.to_csv(index=False))
    (OUT_DIR / "covft_geometric_2d_covft.csv").write_text(df_2d_covft.to_csv(index=False))

    # -------- B. Cov-Vina on N=300 random acrylamide-matching mols per cohort --------
    df_cov_base = pd.DataFrame()
    df_cov_covft = pd.DataFrame()
    if not args.skip_cov_vina:
        acryl_base = df_2d_base[df_2d_base["acryl_match"] == True].reset_index(drop=True)  # noqa: E712
        acryl_covft = df_2d_covft[df_2d_covft["acryl_match"] == True].reset_index(drop=True)  # noqa: E712
        t1 = time.time()
        df_cov_base = run_cov_vina(acryl_base, "base", args.n_cov_vina, args.seed, args.workers_cov)
        df_cov_covft = run_cov_vina(acryl_covft, "covft", args.n_cov_vina, args.seed, args.workers_cov)
        print(f"[CovVina] DONE in {(time.time()-t1)/60:.1f} min")
        (OUT_DIR / "covft_geometric_covvina_base.csv").write_text(df_cov_base.to_csv(index=False))
        (OUT_DIR / "covft_geometric_covvina_covft.csv").write_text(df_cov_covft.to_csv(index=False))

    # -------- C. Cohort summaries --------
    cohort_base = cohort_summary(df_2d_base, df_cov_base)
    cohort_covft = cohort_summary(df_2d_covft, df_cov_covft)

    # -------- D. Two-cohort Mann-Whitney + Cliff's δ --------
    # Sign convention: stats compare (covft - base) for all metrics, named
    # "a_vs_b" where a=covft, b=base.

    pre_base = df_2d_base.loc[df_2d_base["pre_reactivity_score"].notna(),
                              "pre_reactivity_score"].astype(float).values
    pre_covft = df_2d_covft.loc[df_2d_covft["pre_reactivity_score"].notna(),
                                "pre_reactivity_score"].astype(float).values
    dev_base = df_2d_base.loc[df_2d_base["planar_dev_deg"].notna(),
                              "planar_dev_deg"].astype(float).values
    dev_covft = df_2d_covft.loc[df_2d_covft["planar_dev_deg"].notna(),
                                "planar_dev_deg"].astype(float).values

    if not args.skip_cov_vina and len(df_cov_base) and len(df_cov_covft):
        cov_base_ok = df_cov_base[df_cov_base["vina_ok"] == True]  # noqa: E712
        cov_covft_ok = df_cov_covft[df_cov_covft["vina_ok"] == True]  # noqa: E712
        aff_base = cov_base_ok["vina_affinity"].astype(float).values
        aff_covft = cov_covft_ok["vina_affinity"].astype(float).values
        dist_base = cov_base_ok["warhead_sg_distance_A"].astype(float).values
        dist_covft = cov_covft_ok["warhead_sg_distance_A"].astype(float).values
    else:
        aff_base = aff_covft = dist_base = dist_covft = np.array([])

    stats = {
        # Pre-reactivity: covft > base means "more aligned"
        "mw_pre_reactivity_score": mw_summary(pre_covft, pre_base, alternative="two-sided"),
        # Planar deviation: covft < base means "more aligned" (smaller deviation)
        "mw_planar_dev_deg": mw_summary(dev_covft, dev_base, alternative="two-sided"),
        # Vina affinity: more negative = better, covft < base = better
        "mw_vina_affinity": mw_summary(aff_covft, aff_base, alternative="two-sided"),
        # Warhead-SG distance: closer to 1.85 Å is "more covalent"
        "mw_warhead_sg_distance": mw_summary(dist_covft, dist_base, alternative="two-sided"),
    }
    stats["mw_pre_reactivity_score"]["covft_minus_base_median"] = (
        (stats["mw_pre_reactivity_score"]["median_a"] or 0)
        - (stats["mw_pre_reactivity_score"]["median_b"] or 0)
    ) if (
        stats["mw_pre_reactivity_score"]["median_a"] is not None
        and stats["mw_pre_reactivity_score"]["median_b"] is not None
    ) else None

    verdict = headline_verdict(stats)

    # -------- E. Assemble JSON --------
    payload = {
        "config": {
            "base_csv": str(args.base_csv),
            "covft_csv": str(args.covft_csv),
            "acrylamide_smarts": ACRYLAMIDE_SMARTS,
            "planarity_threshold_deg": PLANARITY_THRESH_DEG,
            "cov_vina_subsample_n": args.n_cov_vina,
            "subsample_seed": args.seed,
            "receptor_pdb": str(PROJECT_ROOT / "data/docking_500/4K2R.pdb"),
            "receptor_pdbqt_stripped": str(
                PROJECT_ROOT / "data/docking_500/receptor_cys346_stripped.pdbqt"
            ),
            "vina_mode": "score_only (AD-CovDock tether via meeko CovalentBuilder)",
            "cys346_sg_coords_A": list(map(float, CYS346_SG)),
        },
        "cohort_base": cohort_base,
        "cohort_covft": cohort_covft,
        "two_cohort_tests_covft_vs_base": stats,
        "headline_verdict": verdict,
    }

    OUT_JSON.write_text(json.dumps(payload, indent=2, default=str))
    print(f"\nWrote {OUT_JSON}")

    # -------- F. Markdown summary --------
    md = build_markdown(payload)
    OUT_MD.write_text(md)
    print(f"Wrote {OUT_MD}")

    # -------- G. Echo verdict --------
    print("\n" + "=" * 60)
    print("HEADLINE VERDICT")
    print("=" * 60)
    print(verdict["verdict"])
    print(verdict["recommend"])


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


def build_markdown(p: dict) -> str:
    b = p["cohort_base"]; c = p["cohort_covft"]
    s = p["two_cohort_tests_covft_vs_base"]
    v = p["headline_verdict"]

    md = []
    md.append("# CovFT vs Base — Geometric Covalent Pre-Reactivity Comparison\n")
    md.append("Inputs:")
    md.append(f"- base: `{p['config']['base_csv']}`")
    md.append(f"- covft: `{p['config']['covft_csv']}`")
    md.append("")
    md.append("Method:")
    md.append("- **2D pre-reactivity**: per-mol RDKit ETKDG + MMFF on every SMILES "
              "matching `[CH2]=[CH]C(=O)N`. Dihedral over Cβ-Cα-C(=O)-N; "
              f"planar deviation = min(|d|, |180-|d||); score=1 if dev≤{p['config']['planarity_threshold_deg']}°.")
    md.append(f"- **Cov-Vina** ({p['config']['cov_vina_subsample_n']}/cohort, "
              f"seed={p['config']['subsample_seed']}): meeko CovalentBuilder tethers "
              "warhead Cβ to Cys346 CB; Vina --score_only against the Cys346-stripped "
              "ZAP70 (4K2R) receptor.")
    md.append("")

    md.append("## Cohort summaries\n")
    md.append("| Metric | base | covft |")
    md.append("|---|---|---|")
    md.append(f"| n_total | {b['n_total']} | {c['n_total']} |")
    md.append(f"| n_acrylamide_match | {b['n_acrylamide_match']} | {c['n_acrylamide_match']} |")
    md.append(f"| frac_acrylamide_match | {fmt(b['frac_acrylamide_match'], 4)} | {fmt(c['frac_acrylamide_match'], 4)} |")
    md.append(f"| n_geom_evaluated | {b['n_geom_evaluated']} | {c['n_geom_evaluated']} |")
    md.append(f"| pre_reactivity_score mean ± std | {fmt(b['pre_reactivity_score_mean'])} ± {fmt(b['pre_reactivity_score_std'])} | {fmt(c['pre_reactivity_score_mean'])} ± {fmt(c['pre_reactivity_score_std'])} |")
    md.append(f"| pre_reactivity_score median | {fmt(b['pre_reactivity_score_median'])} | {fmt(c['pre_reactivity_score_median'])} |")
    md.append(f"| pre_reactivity frac ≥ 0.5 | {fmt(b['pre_reactivity_score_frac_ge_0p5'])} | {fmt(c['pre_reactivity_score_frac_ge_0p5'])} |")
    md.append(f"| planar_dev_deg median | {fmt(b['planar_dev_deg_median'], 2)}° | {fmt(c['planar_dev_deg_median'], 2)}° |")
    md.append(f"| planar_dev_deg IQR | {fmt_iqr(b['planar_dev_deg_iqr'], 2)}° | {fmt_iqr(c['planar_dev_deg_iqr'], 2)}° |")
    md.append(f"| cov-vina n attempted | {b['cov_vina_n_attempted']} | {c['cov_vina_n_attempted']} |")
    md.append(f"| cov-vina n successful | {b['cov_vina_n_ok']} | {c['cov_vina_n_ok']} |")
    md.append(f"| vina_affinity median (kcal/mol) | {fmt(b['vina_affinity_median'], 2)} | {fmt(c['vina_affinity_median'], 2)} |")
    md.append(f"| vina_affinity IQR | {fmt_iqr(b['vina_affinity_iqr'], 2)} | {fmt_iqr(c['vina_affinity_iqr'], 2)} |")
    md.append(f"| vina_affinity mean ± std | {fmt(b['vina_affinity_mean'], 2)} ± {fmt(b['vina_affinity_std'], 2)} | {fmt(c['vina_affinity_mean'], 2)} ± {fmt(c['vina_affinity_std'], 2)} |")
    md.append(f"| warhead-SG distance median (Å) | {fmt(b['warhead_sg_distance_median_A'], 3)} | {fmt(c['warhead_sg_distance_median_A'], 3)} |")
    md.append(f"| BD angle median (°) | {fmt(b['bd_angle_median_deg'], 2)} | {fmt(c['bd_angle_median_deg'], 2)} |")
    md.append(f"| pose_converged_rate | {fmt(b['pose_converged_rate'])} | {fmt(c['pose_converged_rate'])} |")
    md.append("")

    md.append("## Two-cohort Mann-Whitney U tests (covft vs base)\n")
    md.append("Cliff's δ is signed (covft − base). Positive δ = covft has larger values.\n")
    md.append("| Metric | n_covft | n_base | median_covft | median_base | U | p-value | Cliff's δ |")
    md.append("|---|---|---|---|---|---|---|---|")
    for name, key in [
        ("pre_reactivity_score (higher=more aligned)", "mw_pre_reactivity_score"),
        ("planar_dev_deg (lower=more aligned)", "mw_planar_dev_deg"),
        ("vina_affinity kcal/mol (lower=better)", "mw_vina_affinity"),
        ("warhead-SG distance Å (closer to 1.85)", "mw_warhead_sg_distance"),
    ]:
        r = s[key]
        md.append(
            f"| {name} | {r['n_a']} | {r['n_b']} | {fmt(r['median_a'])} | "
            f"{fmt(r['median_b'])} | {fmt(r['U'], 1)} | {fmt(r['p_value'], 4)} | "
            f"{fmt(r['cliffs_delta_a_vs_b'], 3)} |"
        )
    md.append("")
    md.append("## Headline verdict\n")
    md.append(f"- Pre-reactivity significant & favors covFT: **{v['pre_reactivity_significant_and_favors_covft']}**")
    md.append(f"- Planar-deviation significant & favors covFT: **{v['planar_dev_significant_and_favors_covft']}**")
    md.append(f"- Vina-cov affinity significant & favors covFT: **{v['vina_affinity_significant_and_favors_covft']}**")
    md.append(f"- Positive axes count: {v['positive_axes_count']} / 3")
    md.append("")
    md.append(f"**Verdict**: {v['verdict']}")
    md.append("")
    md.append(f"**Recommendation**: {v['recommend']}")
    md.append("")
    return "\n".join(md)


if __name__ == "__main__":
    sys.exit(main())
