"""Compute the unified evaluation-metric panel for CovaCraft paper cohorts.

Covers §3.1 tab:covft (base vs covft) and §3.3 tab:m1a_geometry (covft, covft+RL,
LibInvent, v2). Outputs JSON at results/paper_evaluation/metric_panel_all_cohorts.json.

Run:  conda run -n quris python scripts/compute_metric_panel_cohorts.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
RESULTS = ROOT / "results" / "paper_evaluation"
OUT_JSON = RESULTS / "metric_panel_all_cohorts.json"

MOL1_PATH = ROOT / "experiments" / "exp_covft_value" / "mol1_only.smi"

# Cohort source SMILES files (full 10k or 500)
COHORT_SMILES = {
    "base": (ROOT / "experiments/exp_covft_value/samples_base.csv", "SMILES", None),
    "covft": (ROOT / "experiments/exp_covft_value/samples_covft.csv", "SMILES", None),
    "covft_rl": (ROOT / "experiments/exp_geom_bc/samples_rl.csv", "SMILES", None),
    "libinvent": (ROOT / "experiments/exp_geom_bc/samples_libinvent.csv", "SMILES", None),
    "v2": (ROOT / "data/m1a_v2_vs_covft/per_mol_scored.csv", "smiles", "v2"),
}

# 2D geometry CSVs (pre-computed pre-reactivity + planar deviation)
GEOM_2D = {
    "base": RESULTS / "covft_geometric_2d_base.csv",
    "covft": RESULTS / "covft_geometric_2d_covft.csv",
    "covft_rl": RESULTS / "covft_geometric_2d_rl.csv",
    "libinvent": RESULTS / "covft_geometric_2d_libinvent.csv",
    "v2": RESULTS / "covft_geometric_2d_m1a.csv",
}

# Cov-Vina CSVs (subsampled)
COVVINA = {
    "base": RESULTS / "covft_geometric_covvina_base.csv",
    "covft": RESULTS / "covft_geometric_covvina_covft.csv",
    "covft_rl": RESULTS / "covft_geometric_covvina_rl.csv",
    "libinvent": RESULTS / "covft_geometric_covvina_libinvent.csv",
    "v2": RESULTS / "covft_geometric_covvina_m1a.csv",
}

# Reference values from the current tables (§3.1 + §3.3)
REFERENCE = {
    "base": {
        "validity": 0.996, "scaffold_uniqueness": 0.079,
        "acryl_retention_largest_frag": 0.014,
        "prereact_ge_0p5": 0.46, "planar_dihedral_median_deg": 63.0,
        "cov_vina_median": 152.0,
    },
    "covft": {
        "validity": 0.989, "scaffold_uniqueness": 0.092,
        "unique_canonical": 0.330, "top_scaffold_share": 0.114,
        "acryl_retention_largest_frag": 0.939,  # from §3.3 (§3.1 says 97.2)
        "prereact_ge_0p5": 0.478, "planar_dihedral_median_deg": 61.78,
        "cov_vina_median": 150.0, "tc_median_to_mol1": 0.55,
    },
    "covft_rl": {
        "validity": 0.996, "unique_canonical": 0.314, "top_scaffold_share": 0.127,
        "acryl_retention_largest_frag": 0.978,
        "prereact_ge_0p5": 0.413, "planar_dihedral_median_deg": 62.91,
        "tc_median_to_mol1": 0.55,
    },
    "libinvent": {
        "validity": 0.994, "unique_canonical": 0.287, "top_scaffold_share": 0.141,
        "acryl_retention_largest_frag": 0.986,
        "prereact_ge_0p5": 0.604, "planar_dihedral_median_deg": 2.43,
        "tc_median_to_mol1": 0.65,
    },
    "v2": {
        "validity": 0.767, "unique_canonical": 0.531, "top_scaffold_share": 0.365,
        "acryl_retention_largest_frag": 0.944,
        "prereact_ge_0p5": 0.582, "planar_dihedral_median_deg": 3.06,
        "tc_median_to_mol1": 0.64,
    },
}


ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")


def canonical(smiles):
    """Return canonical SMILES or None if unparseable."""
    if smiles is None or (isinstance(smiles, float) and np.isnan(smiles)):
        return None
    try:
        m = Chem.MolFromSmiles(smiles)
    except Exception:
        return None
    if m is None:
        return None
    return Chem.MolToSmiles(m)


def bemis_murcko(mol):
    try:
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        if scaf is None or scaf.GetNumAtoms() == 0:
            return ""
        return Chem.MolToSmiles(scaf)
    except Exception:
        return ""


def morgan_fp(mol, radius=2, n_bits=2048):
    try:
        return AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    except Exception:
        return None


def largest_frag(mol):
    """Return the largest fragment (by heavy atom count)."""
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def acryl_on_largest(mol):
    if mol is None:
        return False
    lf = largest_frag(mol)
    return lf.HasSubstructMatch(ACRYL_SMARTS)


def load_smiles(csv_path: Path, col: str, filter_val: str | None):
    df = pd.read_csv(csv_path)
    if filter_val is not None:
        df = df[df["cohort"] == filter_val].reset_index(drop=True)
    return df[col].astype(str).tolist()


def compute_core_metrics(smiles_list, mol1_fp):
    """Compute validity, uniqueness, scaffold stats, QED, Tc-to-mol1, acryl retention."""
    total = len(smiles_list)
    mols = []
    canon = []
    for s in smiles_list:
        c = canonical(s)
        if c is None:
            continue
        m = Chem.MolFromSmiles(c)
        if m is None:
            continue
        mols.append(m)
        canon.append(c)

    n_valid = len(mols)
    validity = n_valid / total if total else 0.0

    # Unique canonical
    unique_canonical = len(set(canon)) / n_valid if n_valid else 0.0

    # Scaffolds
    scafs = [bemis_murcko(m) for m in mols]
    scafs_nonempty = [s for s in scafs if s]
    n_scaf = len(scafs_nonempty)
    scaf_unique = len(set(scafs_nonempty)) / n_scaf if n_scaf else 0.0
    if n_scaf:
        from collections import Counter
        top = Counter(scafs_nonempty).most_common(1)[0][1]
        top_share = top / n_scaf
    else:
        top_share = 0.0

    # QED
    qeds = []
    for m in mols:
        try:
            qeds.append(QED.qed(m))
        except Exception:
            pass
    qed_med = float(np.median(qeds)) if qeds else float("nan")

    # Tc to Mol1
    tcs = []
    for m in mols:
        fp = morgan_fp(m)
        if fp is None:
            continue
        tcs.append(DataStructs.TanimotoSimilarity(fp, mol1_fp))
    tc_med = float(np.median(tcs)) if tcs else float("nan")

    # Acrylamide on largest fragment
    acryl = [acryl_on_largest(m) for m in mols]
    acryl_frac = float(np.mean(acryl)) if acryl else 0.0

    return {
        "n_total": total,
        "n_valid": n_valid,
        "validity": round(validity, 4),
        "unique_canonical": round(unique_canonical, 4),
        "scaffold_uniqueness": round(scaf_unique, 4),
        "top_scaffold_share": round(top_share, 4),
        "qed_median": round(qed_med, 4) if not np.isnan(qed_med) else None,
        "tc_median_to_mol1": round(tc_med, 4) if not np.isnan(tc_med) else None,
        "tc_dist_from_0p5": round(abs(tc_med - 0.5), 4) if not np.isnan(tc_med) else None,
        "acryl_retention_largest_frag": round(acryl_frac, 4),
    }


def compute_geom_2d(csv_path: Path):
    """Pull pre-reactivity ≥0.5 fraction and median planar deviation from 2D geom CSV."""
    df = pd.read_csv(csv_path)
    # Use rows where an acrylamide was found (rest of molecules do not enter geom stats)
    df_acryl = df[df["acryl_match"] == True]
    # planar_dev_deg is the |dihedral-90| deviation (small=poised, large=twisted)
    # BUT the paper's "planar-dihedral median" for covft is ~62° and for libinvent ~2.4°,
    # so it aligns with `planar_dev_deg` (deviation from planar 90°? or 0°?).
    # Check the numbers: libinvent median 2.43° means planar; covft 62° means twisted.
    # -> `planar_dev_deg` is the metric matching the table.
    planar_median = float(np.median(df_acryl["planar_dev_deg"].dropna())) if not df_acryl.empty else float("nan")
    prereact = df_acryl["pre_reactivity_score"].dropna()
    frac_ge_05 = float((prereact >= 0.5).mean()) if not prereact.empty else float("nan")
    return {
        "planar_dihedral_median_deg": round(planar_median, 2) if not np.isnan(planar_median) else None,
        "prereact_ge_0p5": round(frac_ge_05, 4) if not np.isnan(frac_ge_05) else None,
        "prereact_n": int(len(prereact)),
    }


def compute_covvina(csv_path: Path):
    if not csv_path.exists():
        return {"cov_vina_median": None, "n_vina": 0}
    df = pd.read_csv(csv_path)
    if "vina_affinity" not in df.columns:
        return {"cov_vina_median": None, "n_vina": 0}
    vals = df["vina_affinity"].dropna()
    if vals.empty:
        return {"cov_vina_median": None, "n_vina": 0}
    return {"cov_vina_median": round(float(np.median(vals)), 3), "n_vina": int(len(vals))}


def compute_v2_boltz(csv_path: Path):
    df = pd.read_csv(csv_path)
    df = df[df["cohort"] == "v2"]
    scored = df[df["scored"] == True]
    out = {
        "boltz_iptm_median": round(float(np.median(scored["iptm"].dropna())), 4) if not scored.empty else None,
        "boltz_mpae_min_median": round(float(np.median(scored["mPAE_min"].dropna())), 4) if not scored.empty else None,
        "n_boltz_scored": int(len(scored)),
    }
    # Pose deviations (only where pose converged)
    conv = scored[scored["pose_converged"] == 1.0]
    if not conv.empty:
        out["cofold_bd_deviation_median"] = round(float(np.median((conv["bd_angle_deg"] - 105).abs().dropna())), 3)
        out["cofold_planar_deviation_median"] = round(float(np.median(conv["planar_dihedral_deg"].abs().dropna())), 3)
        out["cofold_sgcb_deviation_median"] = round(float(np.median((conv["d_SG_Cb_A"] - 1.85).abs().dropna())), 4)
        out["n_pose_converged"] = int(len(conv))
    else:
        out["cofold_bd_deviation_median"] = None
        out["cofold_planar_deviation_median"] = None
        out["cofold_sgcb_deviation_median"] = None
        out["n_pose_converged"] = 0
    return out


def main():
    mol1_smiles = MOL1_PATH.read_text().strip().splitlines()[0].strip()
    mol1_mol = Chem.MolFromSmiles(mol1_smiles)
    assert mol1_mol is not None, f"Mol1 failed to parse: {mol1_smiles}"
    mol1_fp = morgan_fp(mol1_mol)

    result = {
        "mol1_smiles": mol1_smiles,
        "mol1_canonical": Chem.MolToSmiles(mol1_mol),
        "cohorts": {},
        "comparison_to_current_table": {},
    }

    for cohort, (csv_path, col, filt) in COHORT_SMILES.items():
        print(f"\n=== {cohort} ===", flush=True)
        smiles = load_smiles(csv_path, col, filt)
        print(f"  N SMILES: {len(smiles)}", flush=True)

        core = compute_core_metrics(smiles, mol1_fp)
        print(f"  core: validity={core['validity']}, uniq_canon={core['unique_canonical']}, "
              f"scaf_uniq={core['scaffold_uniqueness']}, "
              f"acryl_lf={core['acryl_retention_largest_frag']}, "
              f"tc_med={core['tc_median_to_mol1']}", flush=True)

        geom = compute_geom_2d(GEOM_2D[cohort]) if cohort in GEOM_2D else {}
        print(f"  geom: {geom}", flush=True)

        vina = compute_covvina(COVVINA[cohort]) if cohort in COVVINA else {}
        print(f"  vina: {vina}", flush=True)

        cohort_metrics = {**core, **geom, **vina}
        cohort_metrics["notes"] = (
            "pred_pIC50 SKIPPED (FiLM predictor requires ~5 min setup with checkpoint + embedding cache); "
            "planar-dihedral metric uses `planar_dev_deg` col from 2D geom CSV; "
            "acrylamide check on largest fragment via SMARTS [CH2]=[CH]C(=O)N"
        )

        if cohort == "v2":
            boltz = compute_v2_boltz(COHORT_SMILES["v2"][0])
            print(f"  boltz: {boltz}", flush=True)
            cohort_metrics.update(boltz)

        result["cohorts"][cohort] = cohort_metrics

        # Comparison
        if cohort in REFERENCE:
            comp = {}
            for k, ref_v in REFERENCE[cohort].items():
                if k == "planar_dihedral_median_deg":
                    got = cohort_metrics.get("planar_dihedral_median_deg")
                    tol = 1.5
                elif k == "cov_vina_median":
                    got = cohort_metrics.get("cov_vina_median")
                    tol = 5.0
                elif k in ("validity", "scaffold_uniqueness", "unique_canonical",
                           "top_scaffold_share", "acryl_retention_largest_frag",
                           "prereact_ge_0p5", "tc_median_to_mol1"):
                    got = cohort_metrics.get(k)
                    tol = 0.02  # 2 percentage points
                else:
                    got = cohort_metrics.get(k)
                    tol = 0.02
                if got is None:
                    comp[k] = {"current": ref_v, "recomputed": None, "match": False,
                               "reason": "not computed"}
                else:
                    diff = abs(got - ref_v)
                    comp[k] = {
                        "current": ref_v, "recomputed": got,
                        "abs_diff": round(diff, 4),
                        "match": bool(diff <= tol),
                    }
            result["comparison_to_current_table"][cohort] = comp

    # Sanity: mismatches summary
    n_mismatch = 0
    total_checks = 0
    for cohort, comps in result["comparison_to_current_table"].items():
        for k, v in comps.items():
            total_checks += 1
            if not v["match"]:
                n_mismatch += 1

    result["_summary"] = {
        "total_checks": total_checks,
        "n_mismatches": n_mismatch,
    }

    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    with open(OUT_JSON, "w") as f:
        json.dump(result, f, indent=2)

    print(f"\n=== SUMMARY ===")
    print(f"Wrote {OUT_JSON}")
    print(f"Total ref-value checks: {total_checks}")
    print(f"Mismatches: {n_mismatch}")
    if n_mismatch:
        print(f"\nMismatched metrics:")
        for cohort, comps in result["comparison_to_current_table"].items():
            for k, v in comps.items():
                if not v["match"]:
                    print(f"  {cohort}.{k}: current={v['current']}, recomputed={v['recomputed']}, diff={v.get('abs_diff')}")


if __name__ == "__main__":
    main()
