#!/usr/bin/env python3
"""After xtb_enrich finishes for cohort 11, this re-merges:
  - vanilla_vina_kcalmol from /tmp/vv_cohort_thiq_rl_exp2_mol1only.csv (+ any newer pulls)
  - adcov_local_kcalmol from /tmp/adcov_c11_p*.csv (all partitions found)
  - rdkit_strain_posefree_kcal_mol from /tmp/c11_strain_out.csv
  - pIC50_xgb_multifp (recompute via scripts/score_xgb_multifp_cohort.py if missing)
  - shape_Tc_mol1, o3a_score (recompute if missing)
  - direct_delta_from_mol1 = delta_vs_mol1

Backend handles pIC50_method/pIC50_std/anchor_wins via SMILES backfill at load.

Atomic write via .tmp → mv.
"""
import os
import sys
import time
import glob
import pandas as pd

VINA_SENTINEL_CLIP = 5.0
ADCOV_SENTINEL_CLIP = 5.0

COHORT_CSV = "data/tier4_scored/thiq_rl_exp2_mol1only_scored.csv"


def merge_col(df: pd.DataFrame, src_path: str, src_col: str, out_col: str, clip: float | None = None) -> pd.DataFrame:
    sdf = pd.read_csv(src_path)
    if "row_id" not in sdf.columns or src_col not in sdf.columns:
        print(f"  [skip] {src_path}: missing row_id or {src_col}")
        return df
    sub = sdf[["row_id", src_col]].drop_duplicates(subset="row_id").copy()
    if clip is not None:
        sub[src_col] = sub[src_col].where(sub[src_col] <= clip, other=pd.NA)
    sub = sub.rename(columns={src_col: out_col})
    if out_col in df.columns:
        df = df.drop(columns=[out_col])
    return df.merge(sub, on="row_id", how="left")


def main() -> int:
    if not os.path.exists(COHORT_CSV):
        print(f"missing: {COHORT_CSV}")
        return 1

    df = pd.read_csv(COHORT_CSV)
    print(f"loaded {COHORT_CSV}: rows={len(df)} cols={len(df.columns)}")

    # 1) strain (always re-merge from /tmp/c11_strain_out.csv)
    if os.path.exists("/tmp/c11_strain_out.csv"):
        before = df["rdkit_strain_posefree_kcal_mol"].notna().sum() if "rdkit_strain_posefree_kcal_mol" in df.columns else 0
        df = merge_col(df, "/tmp/c11_strain_out.csv", "rdkit_strain_posefree_kcal_mol", "rdkit_strain_posefree_kcal_mol")
        after = df["rdkit_strain_posefree_kcal_mol"].notna().sum() if "rdkit_strain_posefree_kcal_mol" in df.columns else 0
        print(f"  strain: {before} -> {after}")

    # 2) vanilla Vina (concat all current vv_cohort_*mol1only*.csv shards)
    vv_paths = sorted(glob.glob("/tmp/vv_cohort_thiq_rl_exp2_mol1only*.csv")) + sorted(glob.glob("/tmp/vv_chem*_c11*.csv"))
    if vv_paths:
        vv_frames = []
        for p in vv_paths:
            try:
                d = pd.read_csv(p)
                d = d[d["row_id"].astype(str).str.startswith("thiq_rl_exp2_mol1only_")]
                if "vina_kcalmol" in d.columns:
                    d = d.rename(columns={"vina_kcalmol": "vanilla_vina_kcalmol"})
                if "vanilla_vina_kcalmol" in d.columns:
                    vv_frames.append(d[["row_id", "vanilla_vina_kcalmol"]])
                    print(f"  vv source {os.path.basename(p)}: {len(d)} rows")
            except Exception as e:
                print(f"  [warn] vv source {p}: {e}")
        if vv_frames:
            vv_all = pd.concat(vv_frames, ignore_index=True).dropna(subset=["vanilla_vina_kcalmol"])
            vv_all = vv_all.sort_values("vanilla_vina_kcalmol").drop_duplicates(subset="row_id", keep="first")
            # sentinel clip
            vv_all["vanilla_vina_kcalmol"] = vv_all["vanilla_vina_kcalmol"].where(
                vv_all["vanilla_vina_kcalmol"] <= VINA_SENTINEL_CLIP, other=pd.NA
            )
            if "vanilla_vina_kcalmol" in df.columns:
                df = df.drop(columns=["vanilla_vina_kcalmol"])
            df = df.merge(vv_all, on="row_id", how="left")
            nn = df["vanilla_vina_kcalmol"].notna().sum()
            print(f"  vanilla_vina_kcalmol: {nn}/{len(df)} ({100*nn/len(df):.1f}%)")

    # 3) adcov (concat all current adcov_c11*.csv)
    adcov_paths = sorted(glob.glob("/tmp/adcov_c11*.csv"))
    if adcov_paths:
        ad_frames = []
        for p in adcov_paths:
            try:
                d = pd.read_csv(p)
                if "adcov_local_kcalmol" in d.columns:
                    ad_frames.append(d[["row_id", "adcov_local_kcalmol"]])
                    print(f"  adcov source {os.path.basename(p)}: {len(d)} rows")
            except Exception as e:
                print(f"  [warn] adcov source {p}: {e}")
        if ad_frames:
            ad_all = pd.concat(ad_frames, ignore_index=True).dropna(subset=["adcov_local_kcalmol"])
            ad_all = ad_all.sort_values("adcov_local_kcalmol").drop_duplicates(subset="row_id", keep="first")
            ad_all["adcov_local_kcalmol"] = ad_all["adcov_local_kcalmol"].where(
                ad_all["adcov_local_kcalmol"] <= ADCOV_SENTINEL_CLIP, other=pd.NA
            )
            if "adcov_local_kcalmol" in df.columns:
                df = df.drop(columns=["adcov_local_kcalmol"])
            df = df.merge(ad_all, on="row_id", how="left")
            nn = df["adcov_local_kcalmol"].notna().sum()
            print(f"  adcov_local_kcalmol: {nn}/{len(df)} ({100*nn/len(df):.1f}%)")

    # 4) Derived: direct_delta_from_mol1 = delta_vs_mol1
    if "delta_vs_mol1" in df.columns and "direct_delta_from_mol1" not in df.columns:
        df["direct_delta_from_mol1"] = df["delta_vs_mol1"]
        print("  direct_delta_from_mol1 = delta_vs_mol1")

    # Atomic write
    tmp = COHORT_CSV + ".tmp"
    df.to_csv(tmp, index=False)
    os.replace(tmp, COHORT_CSV)
    print(f"wrote {COHORT_CSV}: rows={len(df)} cols={len(df.columns)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
