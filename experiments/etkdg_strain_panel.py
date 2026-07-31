#!/usr/bin/env python3
"""Experiment B: ETKDG conformational strain panel across v2_curriculum_clean cohorts.

Per SMILES:
  1. Generate N_CONF ETKDG (v3) conformers with AllChem.EmbedMultipleConfs.
  2. MMFF94 (or UFF fallback) energy minimize each conformer.
  3. For each conformer compute:
       - phi_vinyl_deg = dihedral N-C(=O)-Calpha=Cbeta (planarity of Michael acceptor)
       - d_beta_to_anchor_centroid = distance from acryl beta-C to the mol's
         aromatic centroid (proxy for intramolecular tether stretch)
       - strain_kcal = E_conf - min(E_ensemble)
  4. Per molecule: f_reactive_rotamer = fraction with |phi_vinyl| < 30 deg
     (planar acrylamide = reactive geometry).
  5. Per cohort: distribution of f_reactive_rotamer and median internal strain.
  6. KS tests between cohorts (theta_90 vs theta_130 vs null_pose vs theta_105).

Purely CPU; no Boltz. Runs in ~15 min on a laptop.

Usage:
    python experiments/etkdg_strain_panel.py \
        --samples_dir data/paper_pair_training/v2_curriculum_clean/steering_samples_merged \
        --out_csv data/paper_pair_training/v2_curriculum_clean/etkdg_strain_panel.csv \
        --n_per_cohort 100 --n_conf 50
"""
from __future__ import annotations
import argparse
import json
import multiprocessing as mp
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

# Acryl beta-C = terminal CH2; alpha-C = CH; carbonyl C; N of amide
ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH][C](=O)[N]")


def _get_aromatic_centroid(mol: Chem.Mol, conf) -> np.ndarray | None:
    """Compute centroid of aromatic atoms (proxy for scaffold core)."""
    ar_idxs = [a.GetIdx() for a in mol.GetAtoms() if a.GetIsAromatic()]
    if not ar_idxs:
        return None
    positions = np.array([[conf.GetAtomPosition(i).x,
                            conf.GetAtomPosition(i).y,
                            conf.GetAtomPosition(i).z] for i in ar_idxs])
    return positions.mean(axis=0)


def _dihedral(p0, p1, p2, p3) -> float:
    """Compute dihedral p0-p1-p2-p3 in degrees, wrapped to [-180, 180]."""
    b0 = p1 - p0
    b1 = p2 - p1
    b2 = p3 - p2
    b1_norm = b1 / (np.linalg.norm(b1) + 1e-12)
    v = b0 - np.dot(b0, b1_norm) * b1_norm
    w = b2 - np.dot(b2, b1_norm) * b1_norm
    x = np.dot(v, w)
    y = np.dot(np.cross(b1_norm, v), w)
    ang = np.degrees(np.arctan2(y, x))
    return float(ang)


def _per_mol(args):
    """Compute per-molecule ETKDG panel: returns dict."""
    idx, smi, cohort, n_conf, seed = args
    row = {"idx": idx, "cohort": cohort, "smiles": smi,
           "n_conf": 0, "n_min": 0,
           "f_reactive_rotamer": np.nan,
           "median_strain_kcal": np.nan,
           "median_d_beta_centroid_A": np.nan,
           "median_abs_phi_vinyl_deg": np.nan,
           "min_phi_vinyl_deg": np.nan,
           "err": ""}
    try:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            row["err"] = "invalid"
            return row
        # Largest fragment
        frags = Chem.GetMolFrags(mol, asMols=True)
        if len(frags) > 1:
            mol = max(frags, key=lambda x: x.GetNumHeavyAtoms())
        # Find acryl match
        match = mol.GetSubstructMatch(ACRYL_SMARTS)
        if not match:
            row["err"] = "no_acryl"
            return row
        beta_i, alpha_i, carbonyl_i, oxy_i, n_i = match  # from smarts atom order
        # Note: SMARTS [CH2]=[CH][C](=O)[N] atom order in match is:
        #   0: CH2 (beta), 1: CH (alpha), 2: C(=O), 3: O, 4: N
        mol_h = Chem.AddHs(mol)
        # Re-find match on mol_h (RDKit preserves heavy atom indices)
        match_h = mol_h.GetSubstructMatch(ACRYL_SMARTS)
        if not match_h:
            row["err"] = "no_acryl_h"
            return row
        beta_i, alpha_i, carbonyl_i, oxy_i, n_i = match_h
        # Embed
        params = AllChem.ETKDGv3()
        params.randomSeed = seed
        params.numThreads = 1
        params.useRandomCoords = True
        try:
            conf_ids = AllChem.EmbedMultipleConfs(mol_h, numConfs=n_conf, params=params)
        except Exception as e:
            row["err"] = f"embed:{e.__class__.__name__}"
            return row
        conf_ids = list(conf_ids)
        row["n_conf"] = len(conf_ids)
        if not conf_ids:
            row["err"] = "no_conf"
            return row
        # Minimize; keep energies
        energies = []
        ok_confs = []
        for cid in conf_ids:
            try:
                # Try MMFF94, fall back to UFF
                mp = AllChem.MMFFGetMoleculeProperties(mol_h, mmffVariant="MMFF94s")
                if mp is not None:
                    ff = AllChem.MMFFGetMoleculeForceField(mol_h, mp, confId=cid)
                else:
                    ff = AllChem.UFFGetMoleculeForceField(mol_h, confId=cid)
                if ff is None:
                    continue
                ff.Minimize(maxIts=500)
                e = ff.CalcEnergy()
                energies.append(e)
                ok_confs.append(cid)
            except Exception:
                continue
        row["n_min"] = len(ok_confs)
        if not ok_confs:
            row["err"] = "no_min"
            return row
        e_arr = np.array(energies)
        e_min = e_arr.min()
        strains = e_arr - e_min  # kcal/mol relative
        # Compute geometry per conformer
        phis = []
        ds = []
        for cid in ok_confs:
            conf = mol_h.GetConformer(cid)
            p_beta = np.array([conf.GetAtomPosition(beta_i).x,
                                conf.GetAtomPosition(beta_i).y,
                                conf.GetAtomPosition(beta_i).z])
            p_alpha = np.array([conf.GetAtomPosition(alpha_i).x,
                                 conf.GetAtomPosition(alpha_i).y,
                                 conf.GetAtomPosition(alpha_i).z])
            p_carb = np.array([conf.GetAtomPosition(carbonyl_i).x,
                                conf.GetAtomPosition(carbonyl_i).y,
                                conf.GetAtomPosition(carbonyl_i).z])
            p_n = np.array([conf.GetAtomPosition(n_i).x,
                             conf.GetAtomPosition(n_i).y,
                             conf.GetAtomPosition(n_i).z])
            # Dihedral N-C(=O)-Calpha-Cbeta = vinyl-amide planar dihedral
            phi = _dihedral(p_n, p_carb, p_alpha, p_beta)
            phis.append(phi)
            centroid = _get_aromatic_centroid(mol_h, conf)
            if centroid is not None:
                ds.append(float(np.linalg.norm(p_beta - centroid)))
        phis = np.array(phis)
        ds = np.array(ds) if ds else np.array([np.nan])
        # For a planar Michael acceptor: |phi| ~ 0 or ~180 both count as planar.
        # Wrap |phi| into [0, 90]: reactive_dist = min(|phi|, 180 - |phi|)
        abs_phi = np.abs(phis)
        reactive_dist = np.minimum(abs_phi, 180.0 - abs_phi)
        row["f_reactive_rotamer"] = float(np.mean(reactive_dist < 30.0))
        row["median_abs_phi_vinyl_deg"] = float(np.median(reactive_dist))
        row["min_phi_vinyl_deg"] = float(reactive_dist.min())
        row["median_strain_kcal"] = float(np.median(strains))
        row["median_d_beta_centroid_A"] = float(np.nanmedian(ds))
    except Exception as e:
        row["err"] = f"top:{e.__class__.__name__}:{e}"
    return row


def _load_cohort(csv_path: Path, n_max: int) -> list[str]:
    df = pd.read_csv(csv_path)
    smis = df["SMILES"].astype(str).tolist()
    # Filter to unique valid canonical SMILES (dedup)
    seen = set()
    out = []
    for s in smis:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            continue
        can = Chem.MolToSmiles(mol)
        if can in seen:
            continue
        seen.add(can)
        out.append(can)
        if len(out) >= n_max:
            break
    return out


def _ks(a, b):
    from scipy.stats import ks_2samp
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    a = a[~np.isnan(a)]
    b = b[~np.isnan(b)]
    if len(a) < 3 or len(b) < 3:
        return {"stat": float("nan"), "p": float("nan"), "n_a": len(a), "n_b": len(b)}
    r = ks_2samp(a, b)
    return {"stat": float(r.statistic), "p": float(r.pvalue), "n_a": len(a), "n_b": len(b)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--samples_dir", required=True)
    ap.add_argument("--out_csv", required=True)
    ap.add_argument("--out_ks_json", default=None)
    ap.add_argument("--n_per_cohort", type=int, default=100)
    ap.add_argument("--n_conf", type=int, default=50)
    ap.add_argument("--workers", type=int, default=max(1, mp.cpu_count() - 1))
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    sdir = Path(args.samples_dir)
    cohorts = ["theta_90", "theta_105", "theta_130", "null_pose"]
    all_tasks = []
    for c in cohorts:
        smis = _load_cohort(sdir / f"samples_{c}.csv", args.n_per_cohort)
        print(f"[cohort {c}] {len(smis)} unique SMILES")
        for i, s in enumerate(smis):
            all_tasks.append((i, s, c, args.n_conf, args.seed))

    t0 = time.time()
    print(f"[etkdg] processing {len(all_tasks)} SMILES × {args.n_conf} confs "
          f"with {args.workers} workers")
    if args.workers > 1:
        with mp.Pool(args.workers) as pool:
            rows = []
            for i, r in enumerate(pool.imap_unordered(_per_mol, all_tasks, chunksize=4)):
                rows.append(r)
                if (i + 1) % 25 == 0:
                    dt = time.time() - t0
                    rate = (i + 1) / (dt / 60)
                    eta = (len(all_tasks) - i - 1) / rate if rate > 0 else 0
                    print(f"[etkdg] {i+1}/{len(all_tasks)} rate={rate:.1f}/min eta={eta:.1f}min")
    else:
        rows = [_per_mol(t) for t in all_tasks]
    df = pd.DataFrame(rows)
    out_csv = Path(args.out_csv)
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_csv, index=False)
    print(f"[etkdg] wrote {len(df)} rows to {out_csv} in {(time.time()-t0)/60:.1f}min")

    # Summary
    print("\n=== Cohort summary ===")
    print(f"{'cohort':<12} {'n':<4} {'ok':<4} "
          f"{'f_react_med':<12} {'strain_med':<11} "
          f"{'d_med':<7} {'phi_med':<8}")
    for c in cohorts:
        sub = df[df["cohort"] == c]
        ok = sub[sub["err"] == ""]
        print(f"{c:<12} {len(sub):<4} {len(ok):<4} "
              f"{ok['f_reactive_rotamer'].median():<12.3f} "
              f"{ok['median_strain_kcal'].median():<11.2f} "
              f"{ok['median_d_beta_centroid_A'].median():<7.2f} "
              f"{ok['median_abs_phi_vinyl_deg'].median():<8.2f}")

    # KS tests: theta_90 vs each
    ok = df[df["err"] == ""]
    metrics = ["f_reactive_rotamer", "median_strain_kcal",
               "median_d_beta_centroid_A", "median_abs_phi_vinyl_deg"]
    ks_out = {}
    print("\n=== KS tests (metric | pair | stat | p | n_a | n_b) ===")
    pairs = [("theta_90", "theta_130"),
             ("theta_90", "null_pose"),
             ("theta_105", "theta_130"),
             ("theta_90", "theta_105"),
             ("null_pose", "theta_130")]
    for m in metrics:
        ks_out[m] = {}
        for (a, b) in pairs:
            r = _ks(ok[ok["cohort"] == a][m], ok[ok["cohort"] == b][m])
            ks_out[m][f"{a}_vs_{b}"] = r
            print(f"{m:<32} {a:>10}_vs_{b:<10} stat={r['stat']:.3f} "
                  f"p={r['p']:.4g} n={r['n_a']}/{r['n_b']}")

    if args.out_ks_json:
        Path(args.out_ks_json).write_text(json.dumps(ks_out, indent=2))
        print(f"[etkdg] KS results → {args.out_ks_json}")


if __name__ == "__main__":
    main()
