#!/usr/bin/env python
"""Phase B — covalent docking metrics 5-9 for ZAP70 generation cohorts.

Metrics computed per cohort (mean over docked mols):
  5. d(Cβ-SG)              (Å, target 1.85)
  6. Bürgi-Dunitz angle    (°, target 107)
  7. AD-CovDock Vina score (kcal/mol, ↓)
  8. Hinge H-bond hits     (% mols with d_min < HBOND_MAX)
  9. Cβ SASA               (Å², ↑) — placeholder if FreeSASA not avail

Inputs:
  Cohort SMILES (subsampled to N_PER_COHORT for RL cohorts by FiLMDelta pIC50).

Outputs:
  results/paper_evaluation/cohort_eval/phase_b_per_mol.csv
  results/paper_evaluation/cohort_eval/phase_b_per_cohort.csv
"""
from __future__ import annotations
import argparse, csv, gc, json, math, os, sys, time, traceback, warnings
from multiprocessing import Pool, cpu_count
from pathlib import Path

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(PROJECT))

# Re-use existing helpers from run_covalent_docking
from experiments.run_covalent_docking import (
    build_cov_tethered_pdbqt, vina_dock, read_tethered_geom,
    pose_valid_covalent, parse_pdbqt_models, find_warhead_atoms,
    CYS346_SG, D_SG_LO, D_SG_HI, BD_LO, BD_HI,
)

OUT_DIR = PROJECT / "results/paper_evaluation/cohort_eval"
WORK_DIR = PROJECT / "data/cohort_eval/phase_b_work"
OUT_DIR.mkdir(parents=True, exist_ok=True)
WORK_DIR.mkdir(parents=True, exist_ok=True)

# Hinge atoms for ZAP70 4K2R (corrected per specialist QA, 2026-06-03).
# Met414 is the GATEKEEPER, NOT the hinge. Canonical kinase hinge is gk+1 (Glu415)
# and gk+3 (Ala417). Atoms read directly from 4K2R.pdb (verified residue identities).
# - ligand donor → hinge acceptor: Glu415 backbone O, Ala417 backbone O
# - ligand acceptor → hinge donor: Ala417 backbone N
RECEPTOR_PDB = PROJECT / "data/docking_500/4K2R.pdb"
GLU415_O = np.array([3.160, -2.094, -24.172])    # hinge acceptor
ALA417_N = np.array([4.749,  0.508, -22.256])    # hinge donor
ALA417_O = np.array([7.828,  0.930, -21.100])    # hinge acceptor
HINGE_DONORS = [ALA417_N]                    # ligand-O → hinge N-H
HINGE_ACCEPTORS = [GLU415_O, ALA417_O]        # ligand-N-H → hinge C=O
HINGE_HBOND_DIST_MAX = 5.0                    # Å (heavy-atom-to-heavy-atom cutoff)

# ---------- Cohort manifest ----------
N_PER_COHORT = 250  # cap per cohort to keep wallclock < 5h. RL cohorts subsample top-N by pIC50.
COHORTS = [
    ("OLD_LibInvent_NoCovalent",    "csv", PROJECT/"results/paper_evaluation/reinvent4/libinvent/libinvent_rgroup_1.csv",       "SMILES", "FiLMDelta pIC50 (raw)"),
    ("OLD_Amine_Replacements",      "csv", PROJECT/"results/paper_evaluation/aichem_tier2_scaled/products_top50k.csv",          "smiles", "pIC50_film"),
    ("REINVENT4_DeNovo_RL",         "csv", PROJECT/"results/paper_evaluation/reinvent4/reinvent/reinvent_denovo_1.csv",          "SMILES", "FiLMDelta pIC50 (raw)"),
    ("REINVENT4_Mol2Mol_RL",        "csv", PROJECT/"results/paper_evaluation/reinvent4/mol2mol/mol2mol_optimize_1.csv",          "SMILES", "FiLMDelta pIC50 (raw)"),
    ("REINVENT4_Mol2Mol_RL_late_only", "csv", PROJECT/"results/paper_evaluation/reinvent4/mol2mol_late/mol2mol_optimize_late_1.csv", "SMILES", "FiLMDelta pIC50 (raw)"),
    ("EXP1_LibInvent_DoubleLocked", "smi", PROJECT/"data/reinvent4_libinvent_hybrid_double_locked/samples.smi", None, None),
    ("EXP2_Mol2Mol_CovInDB_FT",     "smi", PROJECT/"data/reinvent4_mol2mol_covalent_ft_samples/samples.smi", None, None),
    ("EXP3_Reactivity_RL",          "csv", PROJECT/"results/paper_evaluation/reinvent4/denovo_reactivity/denovo_reactivity_1.csv","SMILES", "FiLMDelta pIC50 (raw)"),
    ("EXP4_Warhead_Prefix",         "smi", PROJECT/"data/reinvent4_warhead_prefix_samples/samples.smi", None, None),
    ("EXP6_Warhead_Tokens",         "smi", PROJECT/"data/reinvent4_mol2mol_warhead_tokens_samples/samples.smi", None, None),
    ("Lingo_BC_ONLY_N500",          "sdf", PROJECT/"data/cohort_eval/BC_ONLY/samples.sdf", None, None),
    ("Lingo_NP09_N200",             "sdf", PROJECT/"data/cohort_eval/NP09/samples.sdf", None, None),
]


HINGE_DONOR_ACCEPTOR_COORDS = HINGE_DONORS + HINGE_ACCEPTORS


def load_cohort_smiles(fmt, path, smiles_col, pic50_col, max_n):
    """Return list of (mol_idx, smiles, prescored_pIC50, sdf_path_or_None, sdf_mol_idx_or_None).

    For SDF cohorts we preserve the file path + mol index so the dock worker can
    re-load the native 3D conformer (NOT re-embed via ETKDG).
    """
    rows = []
    if fmt == "smi":
        for i, line in enumerate(path.read_text().splitlines()):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            tok = line.split()
            if not tok or tok[0].lower() in ("smiles",):
                continue
            rows.append((i, tok[0], None, None, None))
    elif fmt == "csv":
        with open(path, newline="") as f:
            for i, row in enumerate(csv.DictReader(f)):
                smi = row.get(smiles_col, "").strip()
                if not smi:
                    continue
                try:
                    p = float(row.get(pic50_col, "") or "nan")
                    if p == 0.0:
                        p = float("nan")  # filter-rejected sentinel
                except ValueError:
                    p = float("nan")
                rows.append((i, smi, p, None, None))
    elif fmt == "sdf":
        supp = Chem.SDMolSupplier(str(path), sanitize=False, removeHs=False)
        for i, mol in enumerate(supp):
            if mol is None:
                continue
            try:
                smi = Chem.MolToSmiles(mol)
            except Exception:
                continue
            if smi:
                rows.append((i, smi, None, str(path), i))
    else:
        raise ValueError(fmt)

    # Deduplicate canonical SMILES + FILTER for warhead-bearing molecules
    # (covalent docking is meaningless without a warhead — and RL cohorts at
    #  0.07-6% acryl retention would otherwise be sampled as ~all no_warhead).
    seen = set()
    deduped = []
    for i, smi, p, sdf_path, sdf_idx in rows:
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        try:
            can = Chem.MolToSmiles(mol)
        except Exception:
            continue
        if can in seen:
            continue
        # Filter: keep only mols with a recognizable Michael acceptor warhead
        if find_warhead_atoms(can) is None:
            continue
        seen.add(can)
        deduped.append((i, can, p, sdf_path, sdf_idx))

    # Subsample: top by pIC50 if available, else first N
    has_pic50 = pic50_col and any(r[2] is not None and not math.isnan(r[2]) for r in deduped)
    if has_pic50:
        scored = [r for r in deduped if r[2] is not None and not math.isnan(r[2])]
        scored.sort(key=lambda r: -r[2])
        sample = scored[:max_n]
    else:
        sample = deduped[:max_n]
    return sample


def compute_hinge_hbond_distance(lig_pdbqt: Path) -> tuple[float | None, int | None]:
    """For each ligand polar heavy atom (N, O), find min distance to any Met414 hinge donor/acceptor.
    Returns (d_min, n_within_threshold)."""
    if not lig_pdbqt.exists():
        return None, None
    poses = parse_pdbqt_models(lig_pdbqt)
    if not poses:
        return None, None
    p = poses[0]
    coords = p["coords"]
    elems = p.get("elements") or []
    d_min = float("inf")
    for j in range(len(coords)):
        el = ""
        if j < len(elems):
            el = (elems[j] or "").upper()[:1]
        if el not in ("N", "O"):
            continue
        for hxyz in HINGE_DONOR_ACCEPTOR_COORDS:
            d = float(np.linalg.norm(coords[j] - hxyz))
            if d < d_min:
                d_min = d
    if d_min == float("inf"):
        return None, 0
    n_hits = 1 if d_min < HINGE_HBOND_DIST_MAX else 0
    return d_min, n_hits


def compute_cb_sasa_in_complex(lig_pdbqt: Path, b_idx: int) -> float | None:
    """SASA at Cβ in the LIGAND-RECEPTOR COMPLEX (not free ligand).

    Concatenates the ligand pose PDB with the receptor PDB and computes per-atom
    SASA via FreeSASA on the complex. Reports the Cβ atom's SASA in pocket-aware
    context (low → buried in pocket; high → solvent-exposed).
    """
    try:
        import freesasa
    except ImportError:
        return None
    if not lig_pdbqt.exists() or not RECEPTOR_PDB.exists() or b_idx < 0:
        return None
    try:
        # 1. Parse ligand heavy-atom coords from the tethered PDBQT
        poses = parse_pdbqt_models(lig_pdbqt)
        if not poses:
            return None
        coords = poses[0]["coords"]
        elems = poses[0].get("elements") or []
        if b_idx >= len(coords):
            return None
        # AutoDock-type → PDB chemical-element mapping (freesasa needs PDB standards)
        AD_TO_ELEM = {"A": "C", "OA": "O", "NA": "N", "NS": "N",
                      "SA": "S", "HD": "H", "HS": "H", "C": "C",
                      "N": "N", "O": "O", "S": "S", "F": "F",
                      "Cl": "Cl", "Br": "Br", "I": "I", "P": "P"}
        # 2. Write a combined PDB: receptor + ligand HETATM block
        from tempfile import NamedTemporaryFile
        with NamedTemporaryFile(suffix=".pdb", delete=False, mode="w") as f:
            # Receptor lines (ATOM only — skip waters/cofactors for cleanliness)
            for line in RECEPTOR_PDB.read_text().splitlines():
                if line.startswith(("ATOM", "TER")):
                    f.write(line + "\n")
            f.write("TER\n")
            # Ligand atoms as HETATM, preserving original order
            for j, (x, y, z) in enumerate(coords):
                ad = (elems[j] if j < len(elems) else "C") or "C"
                el = AD_TO_ELEM.get(ad, ad[0] if ad else "C")
                f.write(f"HETATM{j+1:5d}  {el:<3s} LIG L{1:4d}    "
                        f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00          {el:>2s}\n")
            f.write("END\n")
            complex_pdb = f.name
        try:
            structure = freesasa.Structure(complex_pdb)
            # freesasa 2.2 API: top-level calc() function, NOT Calc class.
            result = freesasa.calc(structure)
            n_total = structure.nAtoms()
            # Cβ is the LAST receptor-atoms + b_idx-th ligand atom.
            # Receptor atom count = n_total - len(coords).
            receptor_atoms = n_total - len(coords)
            cb_global_idx = receptor_atoms + b_idx
            if 0 <= cb_global_idx < n_total:
                return float(result.atomArea(cb_global_idx))
        finally:
            try: os.unlink(complex_pdb)
            except Exception: pass
    except Exception:
        return None
    return None


def dock_one(args):
    cohort, mol_idx, smi, sdf_path, sdf_idx = args
    work = WORK_DIR / cohort
    work.mkdir(parents=True, exist_ok=True)
    lig_pdbqt = work / f"{cohort}_{mol_idx}_cov_lig.pdbqt"
    pose_pdbqt = work / f"{cohort}_{mol_idx}_cov_pose.pdbqt"

    t0 = time.time()
    base = {"cohort": cohort, "mol_idx": mol_idx, "smi": smi,
            "AD_CovDock_score": None, "d_cb_sg_postrelax": None, "bd_angle_postrelax": None,
            "pose_valid": 0, "hinge_dmin": None, "hinge_hbond_top1": None,
            "cb_sasa_in_pocket": None, "msg": "", "dt_s": 0.0}

    # 1. Load mol — prefer SDF native conformer if available (Lingo cohorts).
    mol = None
    if sdf_path and sdf_idx is not None:
        try:
            supp = Chem.SDMolSupplier(str(sdf_path), sanitize=True, removeHs=False)
            if 0 <= sdf_idx < len(supp):
                mol = supp[sdf_idx]
        except Exception:
            mol = None
    if mol is None:
        mol = Chem.MolFromSmiles(smi)
    if mol is None:
        base["msg"] = "smiles_parse_fail"; base["dt_s"] = time.time() - t0
        return base

    info = find_warhead_atoms(smi)
    if info is None:
        base["msg"] = "no_warhead"; base["dt_s"] = time.time() - t0
        return base

    prep = build_cov_tethered_pdbqt(mol, smi, lig_pdbqt)
    if not prep["ok"]:
        base["msg"] = f"prep_fail:{prep['msg']}"; base["dt_s"] = time.time() - t0
        return base

    # 2. score_only mode (canonical AD-CovDock-Vina recipe). Note that
    #    d_cb_sg ≈ 1.85 Å and BD ≈ 107° by construction for ALL successfully
    #    tethered mols — they're tether constants, NOT cohort metrics. We
    #    record them for QA but do not use them to rank cohorts.
    dock = vina_dock(lig_pdbqt, pose_pdbqt, mode="score_only", threads=1)
    if not dock["ok"]:
        base["msg"] = f"score_fail:{dock['msg']}"; base["dt_s"] = time.time() - t0
        return base

    # 3. Geometry from the tethered PDBQT (score_only leaves no relaxed pose).
    g = read_tethered_geom(lig_pdbqt, smi)
    base["d_cb_sg_postrelax"] = g.get("d_sg")   # ≈ 1.85 Å (tether constant)
    base["bd_angle_postrelax"] = g.get("bd_angle")  # ≈ 107° (tether constant)
    base["pose_valid"] = pose_valid_covalent(g.get("d_sg"), g.get("bd_angle"))

    base["AD_CovDock_score"] = dock.get("score")

    # 4. Hinge H-bond + Cβ SASA on the TETHERED ligand pose (since score_only
    #    doesn't write a relaxed pose). The non-Cβ atoms still convey pocket fit.
    d_min, n_hits = compute_hinge_hbond_distance(lig_pdbqt)
    base["hinge_dmin"] = d_min
    base["hinge_hbond_top1"] = n_hits

    base["cb_sasa_in_pocket"] = compute_cb_sasa_in_complex(
        lig_pdbqt, info.get("b_idx", -1),
    )
    base["msg"] = "ok"
    base["dt_s"] = round(time.time() - t0, 2)
    return base


def aggregate_per_cohort(rows):
    by_cohort = {}
    for r in rows:
        by_cohort.setdefault(r["cohort"], []).append(r)
    out = []
    for c, lst in by_cohort.items():
        ok = [r for r in lst if r["msg"] == "ok"]
        no_warhead = [r for r in lst if r["msg"] == "no_warhead"]
        prep_fail = [r for r in lst if r["msg"].startswith("prep_fail")]
        score_fail = [r for r in lst if r["msg"].startswith("score_fail")]
        n_attempted = len(lst)
        n_warhead_eligible = n_attempted - len(no_warhead)
        n_ok = len(ok)
        def mean(key, items=ok):
            vals = [r[key] for r in items if r.get(key) is not None]
            return float(np.mean(vals)) if vals else None
        n_hinge = [r for r in ok if r.get("hinge_hbond_top1") is not None]
        hinge_pct = (100.0 * sum(1 for r in n_hinge if r["hinge_hbond_top1"] == 1) / len(n_hinge)) if n_hinge else None
        out.append({
            "cohort": c,
            "n_attempted": n_attempted,
            "n_no_warhead": len(no_warhead),
            "n_warhead_eligible": n_warhead_eligible,
            "n_prep_fail": len(prep_fail),
            "n_score_fail": len(score_fail),
            "n_docked_ok": n_ok,
            "dock_success_pct_of_warhead": round(100.0 * n_ok / n_warhead_eligible, 1) if n_warhead_eligible else 0.0,
            "mean_AD_CovDock_score":   round(mean("AD_CovDock_score"), 3) if mean("AD_CovDock_score") is not None else None,
            "mean_d_cb_sg_postrelax":  round(mean("d_cb_sg_postrelax"), 3) if mean("d_cb_sg_postrelax") is not None else None,
            "mean_bd_angle_postrelax": round(mean("bd_angle_postrelax"), 1) if mean("bd_angle_postrelax") is not None else None,
            "pct_pose_valid":          round(100.0 * sum(r["pose_valid"] for r in ok) / n_ok, 1) if n_ok else 0.0,
            "hinge_hbond_pct":         round(hinge_pct, 1) if hinge_pct is not None else None,
            "mean_cb_sasa_in_pocket":  round(mean("cb_sasa_in_pocket"), 2) if mean("cb_sasa_in_pocket") is not None else None,
        })
    out.sort(key=lambda r: r["mean_AD_CovDock_score"] if r["mean_AD_CovDock_score"] is not None else 1e9)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(2, cpu_count() - 2))
    ap.add_argument("--max_per_cohort", type=int, default=N_PER_COHORT)
    ap.add_argument("--per_mol_csv", default=str(OUT_DIR/"phase_b_per_mol.csv"))
    ap.add_argument("--per_cohort_csv", default=str(OUT_DIR/"phase_b_per_cohort.csv"))
    args = ap.parse_args()

    all_jobs = []
    for name, fmt, path, smi_col, pic50_col in COHORTS:
        rows = load_cohort_smiles(fmt, path, smi_col, pic50_col, args.max_per_cohort)
        print(f"[load] {name}: {len(rows)} mols queued (cap={args.max_per_cohort})")
        for mol_idx, smi, _p, sdf_path, sdf_idx in rows:
            all_jobs.append((name, mol_idx, smi, sdf_path, sdf_idx))

    print(f"\n[phase_b] {len(all_jobs)} dockings on {args.workers} workers")
    t0 = time.time()
    results = []
    with Pool(args.workers) as pool:
        for i, r in enumerate(pool.imap_unordered(dock_one, all_jobs, chunksize=4)):
            results.append(r)
            if (i + 1) % 50 == 0:
                elapsed = time.time() - t0
                rate = (i + 1) / elapsed
                eta_min = (len(all_jobs) - (i + 1)) / rate / 60 if rate > 0 else 0
                print(f"  [progress] {i+1}/{len(all_jobs)} ({rate*60:.1f}/min, ETA {eta_min:.1f} min)", flush=True)

    # Per-mol CSV
    fields = list(results[0].keys()) if results else []
    with open(args.per_mol_csv, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(results)
    print(f"\nWrote per-mol: {args.per_mol_csv}")

    # Per-cohort summary
    summary = aggregate_per_cohort(results)
    if summary:
        with open(args.per_cohort_csv, "w", newline="") as f:
            w = csv.DictWriter(f, fieldnames=list(summary[0].keys()))
            w.writeheader()
            w.writerows(summary)
    print(f"Wrote per-cohort: {args.per_cohort_csv}")

    # Console table
    print("\n| cohort | n_warhead | n_ok | dock%_of_warhead | AD_score | d_cb_sg_postrelax | bd_postrelax | pose_valid% | hinge_hbond% | cb_sasa_pocket |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in summary:
        print(f"| {r['cohort']} | {r['n_warhead_eligible']} | {r['n_docked_ok']} | "
              f"{r['dock_success_pct_of_warhead']} | "
              f"{r['mean_AD_CovDock_score']} | {r['mean_d_cb_sg_postrelax']} | "
              f"{r['mean_bd_angle_postrelax']} | {r['pct_pose_valid']} | "
              f"{r['hinge_hbond_pct']} | {r['mean_cb_sasa_in_pocket']} |")


if __name__ == "__main__":
    main()
