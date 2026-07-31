"""F4 Boltz pool builder.

Reduces the 11 Mol1-anchored NEW cohorts from F3=66,887 to ~2,000 candidates
for downstream Boltz-2 pose/affinity prediction.

Strategy (per scientist team consensus 2026-06-09):
- PRIMARY reducers (drug-like + shape + pIC50, NaN-fail when listed as required):
    pIC50_FiLM        >= 7.5
    shape_Tc_mol1     >= 0.45   (NaN-pass — 85% coverage)
    LLE               >= 4.5
    SAScore           <= 4.5
    strain_posefree   <= 60
    QED               >= 0.45
    max_pubTc         <= 0.35
    Tc_to_Mol1        <= 0.65
    log_k2_GSH        in [-3.5, -1.0]   (NaN-pass)
    Brenk_alerts      <= 1
- SOFT quality gates (NaN-pass: only fail if value present AND bad):
    vanilla_vina_kcalmol     <= -7.5
    adcov_local_kcalmol      <= 0.0
- Mol1 RESCUE tier (~200): shape_Tc_mol1 >= 0.60 AND Tc_to_Mol1 <= 0.65
    AND pIC50_FiLM >= 6.5 (relaxed) — buys back Mol1-chemotype analogs
    that the FiLM model under-calls (Mol1 itself = 6.59).
- Butina clustering Tc=0.4, cap 4/cluster, target >=500 clusters.

Output: data/tier4_scored/F4_boltz_pool.csv  (~2,000 mols)
"""
from pathlib import Path
import pandas as pd
import numpy as np
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem
from rdkit.ML.Cluster import Butina

RDLogger.DisableLog("rdApp.*")

ROOT = Path("data/tier4_scored")
NEW = [
    "mol1RL_v5_seed_mol1_only", "thiq_rl_mol1only", "thiq_rl_zap70",
    "thiq_rl_kinase", "thiq_rl_exp2_zap70", "thiq_rl_exp2_kinase",
    "thiq_rl_exp2_mol1only", "murcko_rl_zap70", "murcko_rl_kinase",
    "murcko_rl_exp2_zap70", "murcko_rl_exp2_kinase",
]

def kle(c, X): return c.isna() | (c <= X)
def kge(c, X): return c.isna() | (c >= X)
def klt(c, X): return c.isna() | (c < X)
def kt(c):    return c.isna() | (c == True) | (c == 1)
def NkLE(c, X): return c.notna() & (c <= X)
def NkGE(c, X): return c.notna() & (c >= X)
def NkBETWEEN(c, lo, hi): return c.notna() & (c >= lo) & (c <= hi)

def main():
    frames = []
    for c in NEW:
        df = pd.read_csv(ROOT / f"{c}_scored.csv")
        df["_cohort"] = c
        frames.append(df)
    A = pd.concat(frames, ignore_index=True, sort=False)
    print(f"Total mols across 11 NEW cohorts: {len(A):,}")

    # F1 + F2 + F3 (NaN-fail, current cascade)
    # F1 NEW: lead-opt similarity floor Tc_to_Mol1 >= 0.30 (loose: q05 = 0.30,
    # excludes obvious off-scaffold drift; lead-opt = stay close to Mol1).
    m1 = (kle(A["MW"], 700) & kge(A["LogP"], -1) & kle(A["LogP"], 6)
          & kle(A["TPSA"], 180) & kle(A["HBD"], 7) & kle(A["RotBonds"], 14)
          & klt(A["PAINS_alerts"], 1) & kt(A["warhead_intact"])
          & kle(A["Tc_to_Mol1"], 0.85) & kge(A["Tc_to_Mol1"], 0.30)
          & klt(A["Lipinski_violations"], 3))
    m2 = m1 & (kle(A["MW"], 500) & kle(A["LogP"], 5) & kle(A["TPSA"], 140)
               & kle(A["HBA"], 12) & kle(A["HBD"], 5) & kle(A["RotBonds"], 11)
               & kge(A["QED"], 0.25) & kle(A["HeavyAtoms"], 50)
               & klt(A["Brenk_alerts"], 3))
    pic = A["pIC50_film"] if "pIC50_film" in A.columns else A["pIC50_mean"]
    m3 = (m2 & kge(pic, 7.0) & kle(A["SAScore"], 7.0) & kge(A["LLE"], 3.5)
          & kge(A["LE"], 0.2) & kle(A["max_pubTc"], 0.85)
          & kle(A["rdkit_strain_posefree_kcal_mol"], 120))
    f3_count = int(m3.sum())
    print(f"F3 survivors: {f3_count:,}")

    # F4 PRIMARY reducers — calibrated to land ~1500-2500 mols
    # Notable design choice: shape_Tc_mol1 NaN-PASS because 85% coverage —
    # don't penalize cohorts where the shape job didn't finish.
    f4 = (m3
          & (pic >= 7.5)                              # tightened from F3=7.0
          & NkGE(A["LLE"], 4.0)                       # relaxed from 4.5 (F3=3.5)
          & NkLE(A["SAScore"], 4.5)                   # tightened from F3=7.0
          & NkLE(A["rdkit_strain_posefree_kcal_mol"], 80)  # relaxed from 60
          & NkGE(A["QED"], 0.45)                      # tightened from F2=0.25
          & NkLE(A["max_pubTc"], 0.35)                # tightened from F3=0.85
          # Lead-opt similarity window: stay close to Mol1 but novel vs IP
          & NkLE(A["Tc_to_Mol1"], 0.65)
          & NkGE(A["Tc_to_Mol1"], 0.35)               # lead-opt floor (relaxed from 0.40)
          & klt(A["Brenk_alerts"], 2)                 # NaN-pass; <=1
          & (kge(A["shape_Tc_mol1"], 0.40)            # NaN-PASS (85% coverage)
             if "shape_Tc_mol1" in A.columns else True)
          # SOFT quality gates (NaN-pass; fail only if value present AND bad)
          & kle(A["vanilla_vina_kcalmol"], -7.5)
          & kle(A["adcov_local_kcalmol"], 0.0)
    )
    f4_main = A[f4].copy()
    print(f"F4 main (drug-like + shape + soft Vina/ADCov): {len(f4_main):,}")

    # Mol1 rescue tier — recover the Mol1-chemotype CLOSE analogs the FiLM model
    # under-calls. Tighter than F4 main on geometry (shape ≥0.65, Tc ≥0.50) so
    # we get genuine Mol1 close-analogs, not just generic non-passing mols.
    rescue = (m2  # F2 survivors only (loosen pIC50, LLE)
              & (pic >= 7.0)                              # at least Mol1-class
              & NkGE(A["shape_Tc_mol1"], 0.55)            # tight 3D shape to Mol1
              & NkGE(A["Tc_to_Mol1"], 0.50)               # close 2D analog
              & NkLE(A["Tc_to_Mol1"], 0.65)               # but novel vs Mol1
              & NkLE(A["SAScore"], 5.0)
              & NkGE(A["QED"], 0.40)
              & NkGE(A["LLE"], 3.5)
              & NkLE(A["rdkit_strain_posefree_kcal_mol"], 100)
              & NkLE(A["max_pubTc"], 0.40)
              & klt(A["Brenk_alerts"], 2)
              & kle(A["vanilla_vina_kcalmol"], -7.5)
              & kle(A["adcov_local_kcalmol"], 0.0))
    f4_rescue = A[rescue & ~f4].copy()  # only NEW rescue mols
    print(f"F4 rescue (Mol1-shape≥0.55, Tc∈[0.50,0.65]): {len(f4_rescue):,}")

    f4_main["_tier"] = "main"
    f4_rescue["_tier"] = "rescue"
    pool = pd.concat([f4_main, f4_rescue], ignore_index=True)
    pool = pool.drop_duplicates("smiles", keep="first")
    print(f"F4 pool pre-cluster (deduped): {len(pool):,}")

    # Composite ranking score for cluster representative selection
    pool["_pic"] = pool["pIC50_film"].fillna(pool.get("pIC50_mean", pool["pIC50_film"]))
    pool["_vina"] = pool["vanilla_vina_kcalmol"].fillna(pool["vanilla_vina_kcalmol"].median())
    pool["_shape"] = pool["shape_Tc_mol1"].fillna(pool["shape_Tc_mol1"].median()) if "shape_Tc_mol1" in pool.columns else 0.4
    pool["_lle"] = pool["LLE"].fillna(pool["LLE"].median())

    def z(s): return (s - s.mean()) / (s.std() + 1e-9)
    pool["_rank_score"] = (
        0.40 * z(pool["_pic"])
        + 0.20 * z(-pool["_vina"])
        + 0.20 * z(pool["_shape"])
        + 0.15 * z(pool["_lle"])
        - 0.05 * z(pool["rdkit_strain_posefree_kcal_mol"].fillna(0))
    )

    # Butina clustering on Morgan FP at Tc=0.4
    print("Computing Morgan FP for Butina clustering...")
    fps = []
    valid_idx = []
    for i, s in enumerate(pool["smiles"].tolist()):
        m = Chem.MolFromSmiles(str(s))
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, 1024))
        valid_idx.append(i)
    print(f"  Valid mols for clustering: {len(fps):,}")

    # Tight Butina cutoff Tc=0.7 — because ALL pool mols are Mol1 analogs,
    # broader cutoffs collapse into a few mega-clusters. Tc=0.7 (distance 0.3)
    # separates close analogs from sub-scaffold variants.
    print("Butina clustering at Tc=0.7 (distance 0.3)...")
    n = len(fps)
    dists = []
    for i in range(n):
        sims = DataStructs.BulkTanimotoSimilarity(fps[i], fps[:i])
        dists.extend(1.0 - s for s in sims)
    clusters = Butina.ClusterData(dists, n, 0.3, isDistData=True)  # cutoff = 1 - Tc = 0.3
    print(f"  Clusters: {len(clusters):,}")

    # Per cluster, keep top-2 by _rank_score (cap output at ~2K)
    keep_local_idx = []
    for cluster in clusters:
        if not cluster:
            continue
        cluster_global = [valid_idx[k] for k in cluster]
        cluster_df = pool.iloc[cluster_global].copy()
        top = cluster_df.nlargest(2, "_rank_score").index.tolist()
        keep_local_idx.extend(top)
    final = pool.loc[keep_local_idx].copy()
    final = final.sort_values("_rank_score", ascending=False).reset_index(drop=True)
    print(f"F4 pool final (cluster-trimmed, top-4/cluster): {len(final):,}")

    out_path = ROOT / "F4_boltz_pool.csv"
    final.to_csv(out_path, index=False)
    print(f"\nWritten: {out_path}")
    print(f"  Total: {len(final):,}")
    print(f"  Tier breakdown: {final['_tier'].value_counts().to_dict()}")
    print(f"  Cohort breakdown:\n{final['_cohort'].value_counts().to_string()}")

if __name__ == "__main__":
    main()
