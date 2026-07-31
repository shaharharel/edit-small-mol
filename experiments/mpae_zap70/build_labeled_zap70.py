#!/usr/bin/env python3
"""Build labeled_mols_zap70.npz + esm_cache_zap70.npz for the 10-way factorial.

Reads:
    ZAP70 harvest CSV (extended preferred, fallback to base). Filters to real-PAE
    sources; if <2,500 rows, ALSO includes substitute rows with per-source z-score
    normalization for mpae only.

Cross-references with the pose cache
    data/m1a_triples_v2/esm2_cache_posefix_v3.npz
where the ZAP70 pockets/embeddings + z-scored poses live under source='boltz_zap70'.

Outputs (mpae_zap70 dir):
    labeled_mols_zap70.npz  — smiles, pose_boltz (UNNORMALIZED), pose_norm (normalized),
        residues_seq_idx (index into ESM pocket bank), plus 5 target columns:
            d_b_nuc_angstrom, bd_angle_deg, phi_planar_deg (raw)
            target_d, target_theta, target_phi, target_mpae, target_composite (lower=better)
            mpae_source (for provenance)
    esm_cache_zap70.npz  — residues_emb, residues_mask (K unique pockets), row_seq_idx
    pose_stats_zap70.json  — pose_mean/std + composite_z stats
    data_selection_report.json — real vs substitute breakdown

Env-independent path handling.
"""
from __future__ import annotations
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

# Try to detect local vs A100.
LOCAL_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
A100_ROOT = Path("/home/shaharh_quris_ai/edit-small-mol")
PROJECT_ROOT = LOCAL_ROOT if LOCAL_ROOT.exists() else A100_ROOT

REAL_PAE_SOURCES = {"pae_tensor", "pae_tensor_split_b_recovered",
                     "pae_tensor_m1a_v2_vs_covft"}
IDEAL_BD_ANGLE_DEG = 105.0


def compute_targets(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out["target_d"] = out["d_b_nuc_angstrom"].astype(float)
    out["target_theta"] = (out["bd_angle_deg"].astype(float) - IDEAL_BD_ANGLE_DEG).abs()
    phi = out["phi_planar_deg"].astype(float)
    out["target_phi"] = np.minimum.reduce([
        phi.abs(), (phi - 180.0).abs(), (phi + 180.0).abs()
    ])
    out["target_mpae"] = out["mpae_warhead_cys"].astype(float)
    return out


def per_source_zscore(vals: np.ndarray, sources: np.ndarray) -> np.ndarray:
    """z-score `vals` per source group."""
    z = np.zeros_like(vals, dtype=np.float32)
    for src in np.unique(sources):
        mask = sources == src
        v = vals[mask]
        mu = v.mean(); sig = v.std() + 1e-6
        z[mask] = (v - mu) / sig
    return z


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None,
                    help="ZAP70 harvest CSV. If None, auto-detects extended → base.")
    ap.add_argument("--pose_cache",
                    default=str(PROJECT_ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"))
    ap.add_argument("--out_dir",
                    default=str(PROJECT_ROOT / "data/paper_pair_training/mpae_zap70"))
    ap.add_argument("--min_real_rows", type=int, default=2500,
                    help="If real-PAE row count is below this, include substitute rows with per-source mpae z-norm.")
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)

    csv_path = None
    if args.csv:
        csv_path = Path(args.csv)
    else:
        extended = PROJECT_ROOT / "data/paper_pair_training/zap70_cofold_harvest/zap70_acryl_mpae_extended.csv"
        base = PROJECT_ROOT / "data/paper_pair_training/zap70_cofold_harvest/zap70_acryl_mpae.csv"
        csv_path = extended if extended.exists() else base
    print(f"[in] {csv_path}  exists={csv_path.exists()}", flush=True)
    df = pd.read_csv(csv_path)
    print(f"[in] {len(df)} rows, columns={list(df.columns)}", flush=True)

    # Filter to rows with usable pose + mpae
    df = df.dropna(subset=["mpae_warhead_cys", "canonical_smiles",
                            "d_b_nuc_angstrom", "bd_angle_deg", "phi_planar_deg"])
    print(f"[filt] {len(df)} rows with all pose+mpae fields", flush=True)

    # Filter to real-PAE first
    real = df[df["mpae_source"].isin(REAL_PAE_SOURCES)].copy()
    print(f"[real-PAE] {len(real)} rows", flush=True)
    print(f"[real-PAE by source] {real['mpae_source'].value_counts().to_dict()}", flush=True)

    if len(real) >= args.min_real_rows:
        selected = real
        selected["is_real_pae"] = True
        use_substitute = False
        print(f"[decision] using {len(selected)} real-PAE rows only "
               f"(>= {args.min_real_rows})", flush=True)
    else:
        substitute = df[df["mpae_source"] == "complex_ipde_substitute"].copy()
        substitute["is_real_pae"] = False
        real["is_real_pae"] = True
        selected = pd.concat([real, substitute], ignore_index=True)
        # Per-source z-score for mpae only
        src = selected["mpae_source"].to_numpy()
        raw_mpae = selected["mpae_warhead_cys"].to_numpy(dtype=np.float32)
        z_mpae = per_source_zscore(raw_mpae, src)
        selected["mpae_zscored"] = z_mpae
        use_substitute = True
        print(f"[decision] using {len(real)} real + {len(substitute)} substitute "
               f"rows, per-source z-normalizing mpae only "
               f"(real-PAE count {len(real)} < {args.min_real_rows}).", flush=True)
        # Report per-source mpae stats
        for s in selected["mpae_source"].unique():
            m = selected[selected["mpae_source"] == s]["mpae_warhead_cys"]
            print(f"    {s}: n={len(m)}  median={m.median():.3f}  "
                   f"std={m.std():.3f}", flush=True)

    # Cross-reference with pose cache for pocket embeddings
    print(f"[cache] loading {args.pose_cache}", flush=True)
    d = np.load(args.pose_cache, allow_pickle=True)
    cache_smiles = np.array([str(s) for s in d["smiles"]])
    cache_sources = np.array([str(s) for s in d["sources"]])
    cache_row_seq_idx = d["row_seq_idx"]
    residues_emb_all = d["residues_emb"]
    residues_mask_all = d["residues_mask"]

    # Restrict cache to boltz_zap70 rows to pull pocket embeddings
    zap_mask = cache_sources == "boltz_zap70"
    zap_smiles = cache_smiles[zap_mask]
    zap_seq_idx = cache_row_seq_idx[zap_mask]

    # Build smi -> seq_idx map (dedup: pick first occurrence)
    smi_to_seq = {}
    for smi, sq in zip(zap_smiles, zap_seq_idx):
        if smi not in smi_to_seq:
            smi_to_seq[smi] = int(sq)
    print(f"[cache] {len(smi_to_seq)} unique ZAP70 smiles in pose cache", flush=True)

    # Filter selected to those with pocket embedding
    smi_arr = selected["canonical_smiles"].astype(str).to_numpy()
    has_pocket = np.array([s in smi_to_seq for s in smi_arr])
    print(f"[join] {int(has_pocket.sum())}/{len(selected)} selected rows have cached "
           f"pocket embeddings", flush=True)
    selected = selected.loc[has_pocket].reset_index(drop=True)
    if len(selected) == 0:
        raise SystemExit("[FATAL] no selected rows have pocket embeddings — cannot train")

    # Map per-row pocket seq_idx (in ORIGINAL cache indexing)
    row_seq_orig = np.array([smi_to_seq[s] for s in selected["canonical_smiles"].astype(str)])
    # Remap to compact index of unique ZAP70 pockets
    unique_pkt = np.unique(row_seq_orig)
    # Preserve cache indexing for embeddings but compact for storage
    pkt_remap = {int(p): i for i, p in enumerate(unique_pkt)}
    row_seq_compact = np.array([pkt_remap[int(p)] for p in row_seq_orig], dtype=np.int32)
    zap_res_emb = residues_emb_all[unique_pkt].astype(np.float32)
    zap_res_mask = residues_mask_all[unique_pkt].astype(bool)
    print(f"[pocket] {len(unique_pkt)} unique ZAP70 pockets carried, "
           f"emb shape={zap_res_emb.shape}", flush=True)

    # Compute all targets
    selected = compute_targets(selected)

    # Composite: z-score sum of all 4 individual targets
    #   z(d) + z(|θ-105|) + z(|φ|_mod) + z(mpae or z-normalized mpae)
    d_arr = selected["target_d"].to_numpy(dtype=np.float32)
    th_arr = selected["target_theta"].to_numpy(dtype=np.float32)
    ph_arr = selected["target_phi"].to_numpy(dtype=np.float32)
    if use_substitute:
        mp_arr = selected["mpae_zscored"].to_numpy(dtype=np.float32)  # already z per source
    else:
        mp_arr = selected["target_mpae"].to_numpy(dtype=np.float32)

    def z(x):
        return (x - x.mean()) / (x.std() + 1e-6)
    composite = z(d_arr) + z(th_arr) + z(ph_arr) + z(mp_arr)
    selected["target_composite"] = composite

    # Pose input tensor (UNNORMALIZED), and normalization from cache
    pose_boltz = selected[["d_b_nuc_angstrom", "bd_angle_deg",
                              "phi_planar_deg"]].to_numpy(dtype=np.float32)
    pose_mean_cache = d["pose_mean"].astype(np.float32)
    pose_std_cache = d["pose_std"].astype(np.float32)
    pose_norm = (pose_boltz - pose_mean_cache) / np.clip(pose_std_cache, 1e-6, None)

    # Save
    out_npz = out_dir / "labeled_mols_zap70.npz"
    np.savez_compressed(
        out_npz,
        smiles=selected["canonical_smiles"].astype(str).to_numpy(),
        row_seq_idx=row_seq_compact,  # index into zap_res_emb
        pose_boltz=pose_boltz,
        pose_norm=pose_norm.astype(np.float32),
        mpae_warhead_cys=selected["mpae_warhead_cys"].to_numpy(dtype=np.float32),
        mpae_source=selected["mpae_source"].astype(str).to_numpy(),
        is_real_pae=selected["is_real_pae"].to_numpy(dtype=bool),
        target_d=d_arr,
        target_theta=th_arr,
        target_phi=ph_arr,
        target_mpae=mp_arr,  # z-normalized if substitute used, else raw mpae
        target_composite=composite.astype(np.float32),
    )
    print(f"[write] {out_npz}", flush=True)

    esm_out = out_dir / "esm_cache_zap70.npz"
    np.savez_compressed(esm_out, residues_emb=zap_res_emb, residues_mask=zap_res_mask)
    print(f"[write] {esm_out}", flush=True)

    stats = {
        "n_selected": int(len(selected)),
        "n_real_pae": int(selected["is_real_pae"].sum()),
        "n_substitute": int((~selected["is_real_pae"]).sum()),
        "use_substitute": bool(use_substitute),
        "csv_path": str(csv_path),
        "pose_mean": pose_mean_cache.tolist(),
        "pose_std": pose_std_cache.tolist(),
        "pose_names": ["d_b_nuc", "bd_angle_deg", "phi_planar_deg"],
        "mpae_source_counts": {k: int(v) for k, v in
                                 selected["mpae_source"].value_counts().to_dict().items()},
        "n_unique_pockets": int(len(unique_pkt)),
        "target_stats": {
            "d":     {"mean": float(d_arr.mean()),  "std": float(d_arr.std()),
                       "q25": float(np.quantile(d_arr, 0.25)),
                       "q75": float(np.quantile(d_arr, 0.75))},
            "theta": {"mean": float(th_arr.mean()), "std": float(th_arr.std()),
                       "q25": float(np.quantile(th_arr, 0.25)),
                       "q75": float(np.quantile(th_arr, 0.75))},
            "phi":   {"mean": float(ph_arr.mean()), "std": float(ph_arr.std()),
                       "q25": float(np.quantile(ph_arr, 0.25)),
                       "q75": float(np.quantile(ph_arr, 0.75))},
            "mpae":  {"mean": float(mp_arr.mean()), "std": float(mp_arr.std()),
                       "q25": float(np.quantile(mp_arr, 0.25)),
                       "q75": float(np.quantile(mp_arr, 0.75))},
            "composite": {"mean": float(composite.mean()),
                           "std": float(composite.std()),
                           "q25": float(np.quantile(composite, 0.25)),
                           "q75": float(np.quantile(composite, 0.75))},
        },
    }
    stats_path = out_dir / "pose_stats_zap70.json"
    stats_path.write_text(json.dumps(stats, indent=2))
    print(f"[write] {stats_path}", flush=True)
    print(f"[SUMMARY] N_selected={len(selected)}  N_real_pae={stats['n_real_pae']}  "
           f"N_substitute={stats['n_substitute']}", flush=True)


if __name__ == "__main__":
    main()
