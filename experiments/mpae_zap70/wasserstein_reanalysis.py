#!/usr/bin/env python3
"""Track C — Wasserstein reanalysis of the 10-way factorial's clamp samples.

For each (variant, target) pair we already have ~299 SMILES per clamp (A/B/C).
This script:
  1. Embeds each SMILES with ETKDGv3 (single conformer, seed=42), matching
     `experiments/mpae_zap70/sample_and_proxy_eval_zap70.py:measure_emitted_pose`.
  2. Computes 1-Wasserstein on each axis (d, theta, phi):
       - W(A, B), W(A, C)
       - Noise floor via bootstrap: randomly split A into halves 10x with
         different seeds, take max of W(A1, A2).
       - Detection ratio: W(A, C) / W(A1, A2) — >2.0 means real shift.
  3. Also computes Wasserstein against a ChEMBL kinase-active null (500 random
     ligands sampled from `labeled_mols_zap70.npz`) and fraction of Clamp C
     samples > 1σ from the Clamp A distribution mean.

Outputs:
  wasserstein_matrix.csv
  wasserstein_report.md

Local CPU only. ~2h budget.
"""
from __future__ import annotations
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from scipy.stats import wasserstein_distance

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
SAMPLES_DIR = ROOT / "data/paper_pair_training/mpae_zap70"
OUT_DIR = SAMPLES_DIR / "wasserstein_reanalysis"
OUT_DIR.mkdir(parents=True, exist_ok=True)

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=O)[N]")

VARIANTS = [("v3", "d"), ("v3", "theta"), ("v3", "phi"), ("v3", "mpae"),
            ("v3", "composite"),
            ("v4", "d"), ("v4", "theta"), ("v4", "phi"), ("v4", "mpae"),
            ("v4", "composite")]

CLAMPS = ["A", "B", "C"]

N_BOOTSTRAP = 10
NULL_N = 500
CACHE_JSON = OUT_DIR / "_etkdg_cache.json"  # keyed by canonical SMILES


# --------------- Emitted-pose measurement (same recipe as eval) ---------------
def measure_one(smi: str, seed: int = 42):
    """Return (d, theta, phi_wrapped) triple or (nan, nan, nan)."""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return (np.nan, np.nan, np.nan)
    matches = m.GetSubstructMatches(ACRYL_SMARTS)
    if not matches:
        return (np.nan, np.nan, np.nan)
    a1, a2, a3, _a_o, a_n = matches[0]
    try:
        mH = Chem.AddHs(m)
        p = AllChem.ETKDGv3()
        p.randomSeed = seed
        cid = AllChem.EmbedMolecule(mH, p)
        if cid < 0:
            return (np.nan, np.nan, np.nan)
        conf = mH.GetConformer(cid)
        p1 = np.array(conf.GetAtomPosition(a1))
        p3 = np.array(conf.GetAtomPosition(a3))
        d = float(np.linalg.norm(p1 - p3))
        th = float(AllChem.GetAngleDeg(conf, a1, a2, a3))
        phi = float(AllChem.GetDihedralDeg(conf, a1, a2, a3, a_n))
        phi_wrap = ((phi + 180.0) % 360.0) - 180.0
        phi_planar = min(abs(phi_wrap), abs(180.0 - abs(phi_wrap)))
        return (d, th, phi_planar)
    except Exception:
        return (np.nan, np.nan, np.nan)


def canonicalize(smi: str) -> str | None:
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def measure_batch(smis: list[str], cache: dict, seed: int = 42):
    """Return d, theta, phi arrays. Uses/updates in-memory cache keyed on
    canonical SMILES to avoid re-embedding shared molecules."""
    n = len(smis)
    d = np.full(n, np.nan)
    th = np.full(n, np.nan)
    ph = np.full(n, np.nan)
    for i, smi in enumerate(smis):
        c = canonicalize(smi)
        if c is None:
            continue
        if c in cache:
            d[i], th[i], ph[i] = cache[c]
        else:
            triple = measure_one(smi, seed=seed)
            cache[c] = triple
            d[i], th[i], ph[i] = triple
    return d, th, ph


# ------------------- Wasserstein + tail metrics -------------------
def finite(arr: np.ndarray) -> np.ndarray:
    return arr[np.isfinite(arr)]


def wd(a: np.ndarray, b: np.ndarray) -> float:
    a = finite(a); b = finite(b)
    if len(a) < 3 or len(b) < 3:
        return float("nan")
    return float(wasserstein_distance(a, b))


def noise_floor(a: np.ndarray, n_boot: int = N_BOOTSTRAP) -> float:
    a = finite(a)
    if len(a) < 20:
        return float("nan")
    ws = []
    for k in range(n_boot):
        rng = np.random.default_rng(1000 + k)
        idx = rng.permutation(len(a))
        half = len(a) // 2
        a1 = a[idx[:half]]
        a2 = a[idx[half:2*half]]
        ws.append(float(wasserstein_distance(a1, a2)))
    return float(max(ws))


def frac_beyond_sigma(a_dist: np.ndarray, c_dist: np.ndarray) -> float:
    a = finite(a_dist); c = finite(c_dist)
    if len(a) < 5 or len(c) < 1:
        return float("nan")
    mu = float(np.mean(a)); sigma = float(np.std(a))
    if sigma == 0:
        return float("nan")
    return float(np.mean(np.abs(c - mu) > sigma))


# ------------------- Main pipeline -------------------
def load_samples(variant: str, target: str, clamp: str) -> list[str]:
    p = SAMPLES_DIR / f"samples_{variant}_{target}_clamp{clamp}.txt"
    if not p.exists():
        return []
    return [ln.strip() for ln in p.read_text().splitlines() if ln.strip()]


def load_null_smiles(n: int = NULL_N, seed: int = 7) -> list[str]:
    npz = np.load(SAMPLES_DIR / "labeled_mols_zap70.npz", allow_pickle=True)
    smis = list(npz["smiles"])
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(smis), size=min(n, len(smis)), replace=False)
    return [str(smis[i]) for i in idx]


def load_cache() -> dict:
    if CACHE_JSON.exists():
        try:
            raw = json.loads(CACHE_JSON.read_text())
            return {k: tuple(v) for k, v in raw.items()}
        except Exception:
            return {}
    return {}


def save_cache(cache: dict):
    serializable = {k: list(v) for k, v in cache.items()}
    CACHE_JSON.write_text(json.dumps(serializable))


def main():
    cache = load_cache()
    print(f"[cache] loaded {len(cache)} previously embedded SMILES")

    # ---- Kinase-active null ----
    print("[null] embedding ChEMBL kinase-active reference (n=500)...")
    t0 = time.time()
    null_smis = load_null_smiles()
    null_d, null_th, null_ph = measure_batch(null_smis, cache)
    print(f"  null done in {time.time()-t0:.1f}s "
          f"({np.sum(np.isfinite(null_d))}/{len(null_smis)} measured)")
    save_cache(cache)

    rows = []
    md_lines = ["# Wasserstein Reanalysis — 10-way ZAP70 factorial",
                "",
                "**Purpose**: Check whether ANY of the 10 variants shows a "
                "distributional shift in emitted pose (d, theta, phi) beyond a "
                "bootstrap noise floor. Median-shift thresholds (5° theta, "
                "0.1 A d) may miss tail shifts.",
                "",
                "**Detection rule**: `detection_ratio = W(A, C) / max_bootstrap(W(A1, A2))`. ",
                "`>2.0` on any axis flags a real shift the median missed.",
                "",
                "| Variant | Target | Axis | W(A,B) | W(A,C) | NoiseFloor | DetRatio_AC | W(A,null) | frac_C_beyond_1sigma |",
                "|---|---|---|---|---|---|---|---|---|"]

    triggered = []  # list of (variant, target, axis, ratio)

    for variant, target in VARIANTS:
        print(f"\n=== {variant} / {target} ===")
        pose_arrs = {}
        for clamp in CLAMPS:
            smis = load_samples(variant, target, clamp)
            print(f"  clamp {clamp}: {len(smis)} SMILES")
            if not smis:
                pose_arrs[clamp] = (np.array([]), np.array([]), np.array([]))
                continue
            pose_arrs[clamp] = measure_batch(smis, cache)
        save_cache(cache)

        A_d, A_th, A_ph = pose_arrs["A"]
        B_d, B_th, B_ph = pose_arrs["B"]
        C_d, C_th, C_ph = pose_arrs["C"]

        # W(A,B), W(A,C) per axis
        W_d_AB = wd(A_d, B_d); W_d_AC = wd(A_d, C_d)
        W_th_AB = wd(A_th, B_th); W_th_AC = wd(A_th, C_th)
        W_ph_AB = wd(A_ph, B_ph); W_ph_AC = wd(A_ph, C_ph)

        # Noise floors
        nf_d = noise_floor(A_d); nf_th = noise_floor(A_th); nf_ph = noise_floor(A_ph)

        det_d = W_d_AC / nf_d if nf_d and np.isfinite(nf_d) and nf_d > 0 else float("nan")
        det_th = W_th_AC / nf_th if nf_th and np.isfinite(nf_th) and nf_th > 0 else float("nan")
        det_ph = W_ph_AC / nf_ph if nf_ph and np.isfinite(nf_ph) and nf_ph > 0 else float("nan")

        # W(A, null) for context
        W_d_null = wd(A_d, null_d); W_th_null = wd(A_th, null_th); W_ph_null = wd(A_ph, null_ph)

        # Fraction C beyond 1sigma of A
        f_d = frac_beyond_sigma(A_d, C_d)
        f_th = frac_beyond_sigma(A_th, C_th)
        f_ph = frac_beyond_sigma(A_ph, C_ph)

        # Track any >2.0 detections
        for axis_name, ratio in [("d", det_d), ("theta", det_th), ("phi", det_ph)]:
            if np.isfinite(ratio) and ratio > 2.0:
                triggered.append((variant, target, axis_name, ratio))

        rows.append({
            "variant": variant, "target": target,
            "W_d_AB": W_d_AB, "W_d_AC": W_d_AC,
            "W_theta_AB": W_th_AB, "W_theta_AC": W_th_AC,
            "W_phi_AB": W_ph_AB, "W_phi_AC": W_ph_AC,
            "W_d_noisefloor": nf_d,
            "W_theta_noisefloor": nf_th,
            "W_phi_noisefloor": nf_ph,
            "detection_ratio_d": det_d,
            "detection_ratio_theta": det_th,
            "detection_ratio_phi": det_ph,
            "W_d_null": W_d_null,
            "W_theta_null": W_th_null,
            "W_phi_null": W_ph_null,
            "frac_C_beyond_1sigma_d": f_d,
            "frac_C_beyond_1sigma_theta": f_th,
            "frac_C_beyond_1sigma_phi": f_ph,
        })

        for axis, (ab, ac, nf, dr, wn, fc) in [
            ("d", (W_d_AB, W_d_AC, nf_d, det_d, W_d_null, f_d)),
            ("theta", (W_th_AB, W_th_AC, nf_th, det_th, W_th_null, f_th)),
            ("phi", (W_ph_AB, W_ph_AC, nf_ph, det_ph, W_ph_null, f_ph)),
        ]:
            md_lines.append(
                f"| {variant} | {target} | {axis} | "
                f"{ab:.3f} | {ac:.3f} | {nf:.3f} | {dr:.2f} | "
                f"{wn:.3f} | {fc:.3f} |"
            )

    # Write outputs
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "wasserstein_matrix.csv", index=False)
    print(f"\n[write] {OUT_DIR/'wasserstein_matrix.csv'}")

    # Verdict
    md_lines.append("")
    md_lines.append("## Verdict")
    md_lines.append("")
    if triggered:
        md_lines.append(f"**{len(triggered)} axis-variant combinations "
                        f"exceed detection_ratio > 2.0:**")
        for v, t, ax, r in triggered:
            md_lines.append(f"- {v}/{t} axis={ax}: detection_ratio={r:.2f}")
        md_lines.append("")
        md_lines.append("**Wasserstein reveals shifts the median-threshold "
                        "verdict missed. The 10-way \"no control\" call "
                        "should be reopened for these variants.**")
    else:
        md_lines.append("**No variant shows detection_ratio > 2.0 on ANY axis.**")
        md_lines.append("")
        md_lines.append("Wasserstein confirms the median-threshold verdict: "
                        "no clamp-driven distributional shift beyond bootstrap "
                        "noise on any of the 10 variants. Methodological null "
                        "is consistent across both metrics.")

    (OUT_DIR / "wasserstein_report.md").write_text("\n".join(md_lines))
    print(f"[write] {OUT_DIR/'wasserstein_report.md'}")
    save_cache(cache)

    return triggered


if __name__ == "__main__":
    triggered = main()
    print(f"\n[final] triggered={len(triggered)} axis-variant combinations")
    for v, t, ax, r in triggered:
        print(f"  {v}/{t} axis={ax}: detection_ratio={r:.2f}")
