"""Compute metric panel for v2-cond Mol1 cohort_A 10k, mirroring the 5k pipeline
(scripts/compute_metric_panel_cohorts.py). Reuses covft_geometric_comparison._compute_2d_one
for planar-dihedral / pre-reactivity to guarantee identical geometry math.

Outputs: results/paper_evaluation/v2_cond_10k_metric_panel.json
"""
from __future__ import annotations

import json
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, DataStructs, QED
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
sys.path.insert(0, str(ROOT))

# Reuse exact planar-dev routine from the 5k reference pipeline
from experiments.covft_geometric_comparison import _compute_2d_one  # noqa: E402

COHORT_CSV = ROOT / "data/m1a_v2_ablation/cohort_A_10k.csv"
MOL1_PATH = ROOT / "experiments/exp_covft_value/mol1_only.smi"
OUT_JSON = ROOT / "results/paper_evaluation/v2_cond_10k_metric_panel.json"

# 5k v2 reference (from REFERENCE["v2"] in compute_metric_panel_cohorts.py)
REF_5K = {
    "validity": 0.767,
    "scaffold_uniqueness": 0.263,      # user-reported "scaffold_uniq 26.3%"
    "top_scaffold_share": 0.365,
    "qed_median": 0.86,
    "tc_median_to_mol1": 0.63,
    "acryl_retention_largest_frag": 0.944,
    "planar_dihedral_median_deg": 3.06,
    "prereact_ge_0p5": 0.582,
}

ACRYL_SMARTS = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
WORKERS = 8


def canonical(smiles):
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
    frags = Chem.GetMolFrags(mol, asMols=True)
    if not frags:
        return mol
    return max(frags, key=lambda m: m.GetNumHeavyAtoms())


def acryl_on_largest(mol):
    if mol is None:
        return False
    return largest_frag(mol).HasSubstructMatch(ACRYL_SMARTS)


def compute_core(smiles_list, mol1_fp):
    total = len(smiles_list)
    mols, canon = [], []
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
    unique_canonical = len(set(canon)) / n_valid if n_valid else 0.0

    scafs = [bemis_murcko(m) for m in mols]
    scafs_ne = [s for s in scafs if s]
    n_scaf = len(scafs_ne)
    scaf_uniq = len(set(scafs_ne)) / n_scaf if n_scaf else 0.0
    top_share = (Counter(scafs_ne).most_common(1)[0][1] / n_scaf) if n_scaf else 0.0

    qeds = []
    for m in mols:
        try:
            qeds.append(QED.qed(m))
        except Exception:
            pass
    qed_med = float(np.median(qeds)) if qeds else float("nan")

    tcs = []
    for m in mols:
        fp = morgan_fp(m)
        if fp is None:
            continue
        tcs.append(DataStructs.TanimotoSimilarity(fp, mol1_fp))
    tc_med = float(np.median(tcs)) if tcs else float("nan")

    acryl = [acryl_on_largest(m) for m in mols]
    acryl_frac = float(np.mean(acryl)) if acryl else 0.0

    return {
        "n_total": total,
        "n_valid": n_valid,
        "validity": round(validity, 4),
        "unique_canonical": round(unique_canonical, 4),
        "scaffold_uniqueness": round(scaf_uniq, 4),
        "top_scaffold_share": round(top_share, 4),
        "qed_median": round(qed_med, 4) if not np.isnan(qed_med) else None,
        "tc_median_to_mol1": round(tc_med, 4) if not np.isnan(tc_med) else None,
        "acryl_retention_largest_frag": round(acryl_frac, 4),
    }


def compute_geom(smiles_list):
    tasks = [(i, smi) for i, smi in enumerate(smiles_list)]
    print(f"[2D geom] N={len(tasks)} workers={WORKERS}", flush=True)
    rows = []
    t0 = time.time()
    done = 0
    with ProcessPoolExecutor(max_workers=WORKERS) as exc:
        futures = [exc.submit(_compute_2d_one, t) for t in tasks]
        for fut in as_completed(futures):
            try:
                rows.append(fut.result())
            except Exception as e:
                rows.append({"acryl_match": False, "pre_reactivity_score": None,
                             "planar_dev_deg": None, "msg": f"fut_exc:{e}"})
            done += 1
            if done % 1000 == 0:
                el = time.time() - t0
                print(f"  {done}/{len(tasks)} in {el:.0f}s ({done/max(el,1e-6):.0f}/s)",
                      flush=True)
    df = pd.DataFrame(rows)
    df_ac = df[df["acryl_match"] == True]
    planar_med = float(np.median(df_ac["planar_dev_deg"].dropna())) if not df_ac.empty else float("nan")
    pre = df_ac["pre_reactivity_score"].dropna()
    frac05 = float((pre >= 0.5).mean()) if not pre.empty else float("nan")
    return {
        "planar_dihedral_median_deg": round(planar_med, 2) if not np.isnan(planar_med) else None,
        "prereact_ge_0p5": round(frac05, 4) if not np.isnan(frac05) else None,
        "prereact_n": int(len(pre)),
        "acryl_match_n": int(len(df_ac)),
    }


def main():
    mol1_smi = MOL1_PATH.read_text().strip().splitlines()[0].strip()
    mol1 = Chem.MolFromSmiles(mol1_smi)
    mol1_fp = morgan_fp(mol1)

    df = pd.read_csv(COHORT_CSV)
    print(f"Loaded {len(df)} rows from {COHORT_CSV}", flush=True)
    smiles = df["SMILES"].astype(str).tolist()

    print("Computing core metrics...", flush=True)
    core = compute_core(smiles, mol1_fp)
    print(f"  core: {json.dumps(core, indent=2)}", flush=True)

    print("Computing 2D geometry (planar dihedral + pre-reactivity)...", flush=True)
    geom = compute_geom(smiles)
    print(f"  geom: {geom}", flush=True)

    metrics = {**core, **geom}

    # Compare to 5k
    comparison = {}
    for k, ref in REF_5K.items():
        got = metrics.get(k)
        if got is None:
            comparison[k] = {"ref_5k": ref, "recomputed_10k": None,
                             "delta_pct": None, "within_2pct": False}
            continue
        abs_diff = got - ref
        delta_pct = 100.0 * abs_diff / max(abs(ref), 1e-9)
        # For "within 2%" apply an absolute tolerance of 2 percentage points
        # for fractions in [0,1] and 2 relative % for planar (deg). Report both
        # abs and relative.
        within = abs(abs_diff) <= 0.02 if ref <= 1.0 else abs(delta_pct) <= 2.0
        comparison[k] = {
            "ref_5k": ref,
            "recomputed_10k": got,
            "abs_delta": round(abs_diff, 4),
            "delta_pct": round(delta_pct, 2),
            "within_2pct": bool(within),
        }

    out = {
        "cohort_csv": str(COHORT_CSV),
        "mol1_smiles": mol1_smi,
        "mol1_canonical": Chem.MolToSmiles(mol1),
        "n_sampled": len(df),
        "metrics_10k": metrics,
        "reference_5k": REF_5K,
        "comparison": comparison,
    }
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(out, indent=2))
    print(f"\nWrote {OUT_JSON}")
    print(json.dumps(out["metrics_10k"], indent=2))
    print("\nCOMPARISON (5k vs 10k):")
    for k, v in comparison.items():
        flag = "OK" if v["within_2pct"] else "!!"
        print(f"  [{flag}] {k}: 5k={v['ref_5k']}, 10k={v['recomputed_10k']}, "
              f"abs_delta={v['abs_delta']}, delta_pct={v['delta_pct']}%")


if __name__ == "__main__":
    main()
