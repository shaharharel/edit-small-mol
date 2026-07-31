"""Merge Boltz confidence metrics + Vina rescore + cov-Vina rescore into the
3,597-mol Boltz cohort CSV.

Reads:
  data/boltz_results/cohort_3597_full/from_*/<row_id>/confidence_<row_id>_model_0.json
  data/tier4_scored/vina_rescore_boltz_intermediate.csv
  data/tier4_scored/covvina_rescore_boltz_intermediate.csv

Writes (in-place with .bak):
  data/tier4_scored/boltz2_cohort_A_relaxed.csv

New columns added:
  boltz_confidence_score, boltz_ptm, boltz_iptm, boltz_ligand_iptm, boltz_protein_iptm,
  boltz_complex_plddt, boltz_complex_iplddt, boltz_complex_pde, boltz_complex_ipde,
  vina_rescore_affinity_kcalmol, vina_rescore_inter_kcalmol, vina_rescore_intra_kcalmol, vina_rescore_torsions_kcalmol,
  covvina_rescore_affinity_kcalmol, covvina_rescore_inter_kcalmol, covvina_rescore_intra_kcalmol, covvina_rescore_torsions_kcalmol,
  boltz_pose_locally_available (bool)

Run: python scripts/merge_boltz_metrics_into_cohort.py
"""
from __future__ import annotations
import json
import sys
import time
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
COFOLD_ROOT = PROJECT_ROOT / "data/boltz_results/cohort_3597_full"
VINA_CSV = PROJECT_ROOT / "data/tier4_scored/vina_rescore_boltz_intermediate.csv"
COVVINA_CSV = PROJECT_ROOT / "data/tier4_scored/covvina_rescore_boltz_intermediate.csv"
COHORT_CSV = PROJECT_ROOT / "data/tier4_scored/boltz2_cohort_A_relaxed.csv"
# Geometry + contacts + mPAE-proxy metrics (written by scripts/compute_full_boltz_metrics.py)
FULL_METRICS_CSV = PROJECT_ROOT / "data/tier4_scored/boltz_full_metrics_3597.csv"

# Columns to pull from FULL_METRICS_CSV (skip ones already in cohort).
FULL_METRICS_KEEP = [
    "warhead_class", "warhead_atom_name",
    "d_SG", "geom_ok", "burgi_dunitz_dev_deg",
    "n_h_bonds", "n_covalent", "n_salt_bridges", "n_pi_pi",
    "n_stabilizing_contacts", "n_contacts_total",
    "pocket_occupancy_pct", "atp_pocket_fraction", "hinge_hbond",
    "mPAE_paper", "mPAE_full", "mPAE_interface", "mPAE_min",
    "mPAE_proxy_kind", "mPAE_min_available",
]

# Pose-aware strain (computed by master agent run 2026-06-06 PM, master result CSV)
STRAIN_CSV = PROJECT_ROOT / "data/tier4_scored/rdkit_strain_kcal_mol_3597.csv"
# v2 pose-aware strain (orchestrator run 2026-06-07 on extended 3,267 cofold set)
STRAIN_CSV_V2 = PROJECT_ROOT / "data/tier4_scored/rdkit_strain_pose_3597_v2.csv"

# 3D shape + USRCAT (ESP proxy) similarity vs Mol1 anchor (computed by
# scripts/compute_shape_sim_3597.py — 2026-06-06)
SHAPE_SIM_CSV = PROJECT_ROOT / "data/tier4_scored/shape_sim_3597.csv"

# PROPKA3 pKa_Cys346 (v1: 2010 rows; v2: extended to 3,267 cofolds by orchestrator 2026-06-07)
PROPKA_CSV_V1 = PROJECT_ROOT / "data/tier4_scored/propka_pKa_Cys346_3597.csv"
PROPKA_CSV_V2 = PROJECT_ROOT / "data/tier4_scored/propka_cys346_3597_v2.csv"

# xTB orbital energies (HOMO/LUMO/q_Cb/gap/omega/fukui), 100% coverage
XTB_ORBITAL_CSV = PROJECT_ROOT / "data/tier4_scored/xtb_orbital_energies_3597.csv"


def collect_boltz_confidence() -> dict[int, dict]:
    """Walk every from_* dir, parse each confidence JSON, return row_id → metrics."""
    out: dict[int, dict] = {}
    n_seen = n_parsed = n_err = 0
    t0 = time.perf_counter()
    for cof_dir in COFOLD_ROOT.rglob("confidence_*_model_0.json"):
        n_seen += 1
        try:
            # row_id is the parent dir name
            row_id = int(cof_dir.parent.name)
        except ValueError:
            continue
        try:
            d = json.loads(cof_dir.read_text())
        except Exception:
            n_err += 1
            continue
        out[row_id] = {
            "boltz_confidence_score": d.get("confidence_score"),
            "boltz_ptm": d.get("ptm"),
            "boltz_iptm": d.get("iptm"),
            "boltz_ligand_iptm": d.get("ligand_iptm"),
            "boltz_protein_iptm": d.get("protein_iptm"),
            "boltz_complex_plddt": d.get("complex_plddt"),
            "boltz_complex_iplddt": d.get("complex_iplddt"),
            "boltz_complex_pde": d.get("complex_pde"),
            "boltz_complex_ipde": d.get("complex_ipde"),
        }
        n_parsed += 1
    print(f"[boltz] seen={n_seen} parsed={n_parsed} err={n_err} elapsed={time.perf_counter()-t0:.1f}s")
    return out


def main() -> int:
    print("=== Boltz cohort metric merge ===")
    if not COHORT_CSV.exists():
        print(f"Cohort CSV missing: {COHORT_CSV}")
        return 1

    cohort = pd.read_csv(COHORT_CSV, low_memory=False)
    print(f"Cohort: {len(cohort):,} rows, {len(cohort.columns)} cols")

    # Backup
    bak = COHORT_CSV.with_suffix(".csv.bak2")
    cohort.to_csv(bak, index=False)
    print(f"  backup → {bak}")

    # 1. Boltz confidence
    boltz_metrics = collect_boltz_confidence()
    boltz_df = pd.DataFrame.from_dict(boltz_metrics, orient="index").reset_index()
    boltz_df.columns = ["row_id"] + list(boltz_df.columns[1:])

    # 2. Vanilla Vina rescore
    print(f"\n[vina_rescore] reading {VINA_CSV.name}")
    vina = pd.read_csv(VINA_CSV)
    vina_keep = vina[["row_id"]].copy()
    vina_keep["vina_rescore_affinity_kcalmol"] = vina["vina_affinity_kcalmol"]
    vina_keep["vina_rescore_inter_kcalmol"] = vina["vina_inter_kcalmol"]
    vina_keep["vina_rescore_intra_kcalmol"] = vina["vina_intra_kcalmol"]
    vina_keep["vina_rescore_torsions_kcalmol"] = vina["vina_torsions_kcalmol"]
    # Dedup row_id (keep last in case of retries)
    vina_keep = vina_keep.drop_duplicates(subset="row_id", keep="last")
    print(f"  {len(vina_keep):,} rows")

    # 3. Cov-Vina rescore
    print(f"\n[covvina_rescore] reading {COVVINA_CSV.name}")
    cov = pd.read_csv(COVVINA_CSV)
    cov_keep = cov[["row_id"]].copy()
    cov_keep["covvina_rescore_affinity_kcalmol"] = cov["covvina_affinity_kcalmol"]
    cov_keep["covvina_rescore_inter_kcalmol"] = cov["covvina_inter_kcalmol"]
    cov_keep["covvina_rescore_intra_kcalmol"] = cov["covvina_intra_kcalmol"]
    cov_keep["covvina_rescore_torsions_kcalmol"] = cov["covvina_torsions_kcalmol"]
    cov_keep = cov_keep.drop_duplicates(subset="row_id", keep="last")
    print(f"  {len(cov_keep):,} rows")

    # 4. Merge all four by row_id
    cohort["row_id"] = cohort["row_id"].astype(int)
    boltz_df["row_id"] = boltz_df["row_id"].astype(int)
    vina_keep["row_id"] = vina_keep["row_id"].astype(int)
    cov_keep["row_id"] = cov_keep["row_id"].astype(int)

    # Geometry + contacts + mPAE-proxy table
    full_keep = None
    if FULL_METRICS_CSV.exists():
        print(f"\n[full_metrics] reading {FULL_METRICS_CSV.name}")
        full = pd.read_csv(FULL_METRICS_CSV, low_memory=False)
        full["row_id"] = full["row_id"].astype(int)
        keep_cols = ["row_id"] + [c for c in FULL_METRICS_KEEP if c in full.columns]
        full_keep = full[keep_cols].drop_duplicates(subset="row_id", keep="last")
        # Drop overlap with existing cohort cols (e.g. mPAE_full if pre-existing) — we
        # always prefer the freshly computed values, so rename overlapping cohort cols
        # to *_legacy before the merge so we don't end up with _x/_y junk.
        overlap = [c for c in full_keep.columns if c != "row_id" and c in cohort.columns]
        if overlap:
            print(f"  overlap with cohort (cohort cols renamed *_legacy): {overlap}")
            cohort = cohort.rename(columns={c: f"{c}_legacy" for c in overlap})
        print(f"  {len(full_keep):,} rows × {len(full_keep.columns)} cols (incl row_id)")
    else:
        print(f"\n[full_metrics] {FULL_METRICS_CSV.name} not found — skipping (geometry/contacts/mPAE columns will not be added)")

    # Shape + USRCAT (ESP proxy) similarity vs Mol1 anchor
    shape_keep = None
    if SHAPE_SIM_CSV.exists():
        print(f"\n[shape_sim] reading {SHAPE_SIM_CSV.name}")
        shape = pd.read_csv(SHAPE_SIM_CSV)
        shape["row_id"] = shape["row_id"].astype(int)
        shape_keep = shape[["row_id", "shape_Tc_seed", "esp_sim_seed"]].drop_duplicates(
            subset="row_id", keep="last"
        )
        # Cohort already carries empty `shape_Tc_seed` and `esp_sim_seed` columns;
        # drop them so the merge populates with the freshly computed values.
        for c in ("shape_Tc_seed", "esp_sim_seed"):
            if c in cohort.columns:
                cohort = cohort.drop(columns=[c])
        print(
            f"  {len(shape_keep):,} rows "
            f"(success_flag={int(shape['success_flag'].astype(bool).sum()):,})"
        )

    # ── Extra v2 sources (pose-strain, PROPKA, xTB orbital energies) ─────────
    strain_v2_keep = None
    # CAVEAT 2026-06-07: v1 strain (rdkit_strain_kcal_mol_3597.csv, 2010 rows) and
    # v2 strain (rdkit_strain_pose_3597_v2.csv, 3337 rows) computed e_bound on
    # DIFFERENTLY-prepared PDBs (v1 from boltz_poses/top1000 pipeline w/ H-relax,
    # v2 from on-the-fly gemmi CIF→PDB). v1 median strain ≈ 295; v2 ≈ 35.
    # e_free is identical → difference is purely in pose H-prep. We use v2 ONLY
    # for cohort-wide consistency (a uniform metric across all 3,337 cofolds beats
    # a bimodal one). f5_strain_pose cutoff is >500 (default-OFF); v2 catches the
    # same outlier shape, just at a tighter absolute threshold if needed.
    if STRAIN_CSV_V2.exists():
        strain_src = STRAIN_CSV_V2
        print(f"\n[strain_v2] reading {strain_src.name} (v2: 3,337-cohort consistent, takes precedence over v1)")
        s_df = pd.read_csv(strain_src)
        colmap = {
            "strain_kcal_mol": "rdkit_strain_kcal_mol",
            "vdw_interaction_kcal_mol": "vdw_interaction_kcal_mol",
            "e_bound_kcal_mol": "rdkit_e_bound_kcal_mol",
            "e_free_kcal_mol": "rdkit_e_free_kcal_mol",
        }
        keep = [c for c in colmap if c in s_df.columns]
        s_df = s_df[["row_id"] + keep].rename(columns=colmap).copy()
        s_df["row_id"] = s_df["row_id"].astype(int)
        s_df = s_df.drop_duplicates(subset="row_id", keep="last")
        strain_v2_keep = s_df
        print(f"  {len(s_df):,} rows from {strain_src.name}")
    elif STRAIN_CSV.exists():
        # Fallback to v1 if v2 missing.
        strain_src = STRAIN_CSV
        print(f"\n[strain_v1] reading {strain_src.name} (v1 fallback — v2 not present)")
        s_df = pd.read_csv(strain_src)
        colmap = {"strain_kcal_mol": "rdkit_strain_kcal_mol",
                  "e_bound_kcal_mol": "rdkit_e_bound_kcal_mol",
                  "e_free_kcal_mol": "rdkit_e_free_kcal_mol"}
        keep = [c for c in colmap if c in s_df.columns]
        s_df = s_df[["row_id"] + keep].rename(columns=colmap).copy()
        s_df["row_id"] = s_df["row_id"].astype(int)
        s_df = s_df.drop_duplicates(subset="row_id", keep="last")
        strain_v2_keep = s_df
        print(f"  {len(s_df):,} rows from {strain_src.name}")

    propka_v2_keep = None
    propka_rows: list = []
    for propka_src in (PROPKA_CSV_V2, PROPKA_CSV_V1):
        if propka_src.exists():
            print(f"\n[propka_v2] reading {propka_src.name}")
            p_df = pd.read_csv(propka_src)
            if "row_id" in p_df.columns and "pKa_Cys346" in p_df.columns:
                p_keep = p_df[["row_id", "pKa_Cys346"]].copy()
                if "success_flag" in p_df.columns:
                    # Only keep successful rows
                    p_keep = p_keep[p_df["success_flag"].astype(str).isin({"1", "True", "true"})]
                p_keep["row_id"] = p_keep["row_id"].astype(int)
                p_keep = p_keep.drop_duplicates(subset="row_id", keep="last")
                propka_rows.append(p_keep)
                print(f"  {len(p_keep):,} successful rows from {propka_src.name}")
    if propka_rows:
        propka_v2_keep = propka_rows[0]
        for extra in propka_rows[1:]:
            # Add v1 rows that aren't in v2 (extra is older v1 — prefer v2 over v1)
            mask_new = ~extra["row_id"].isin(propka_v2_keep["row_id"])
            propka_v2_keep = pd.concat([propka_v2_keep, extra[mask_new]], ignore_index=True)
        print(f"  PROPKA combined: {len(propka_v2_keep):,} unique row_ids")

    xtb_keep = None
    if XTB_ORBITAL_CSV.exists():
        print(f"\n[xtb_orbital] reading {XTB_ORBITAL_CSV.name}")
        x_df = pd.read_csv(XTB_ORBITAL_CSV)
        x_cols = ["HOMO_eV", "LUMO_eV", "gap_eV", "omega_eV", "q_Cb", "fukui_plus_Cb",
                  "pred_log_k2_GSH"]
        x_cols = [c for c in x_cols if c in x_df.columns]
        xtb_keep = x_df[["row_id"] + x_cols].copy()
        xtb_keep["row_id"] = xtb_keep["row_id"].astype(int)
        xtb_keep = xtb_keep.drop_duplicates(subset="row_id", keep="last")
        print(f"  {len(xtb_keep):,} rows × {len(x_cols)} xTB cols")

    merged = cohort.merge(boltz_df, on="row_id", how="left", suffixes=("", "_boltz_new"))
    merged = merged.merge(vina_keep, on="row_id", how="left", suffixes=("", "_vina_new"))
    merged = merged.merge(cov_keep, on="row_id", how="left", suffixes=("", "_cov_new"))
    if full_keep is not None:
        merged = merged.merge(full_keep, on="row_id", how="left", suffixes=("", "_full_new"))
    if shape_keep is not None:
        merged = merged.merge(shape_keep, on="row_id", how="left", suffixes=("", "_shape_new"))
    if strain_v2_keep is not None:
        merged = merged.merge(strain_v2_keep, on="row_id", how="left", suffixes=("", "_strain_new"))
    if propka_v2_keep is not None:
        merged = merged.merge(propka_v2_keep, on="row_id", how="left", suffixes=("", "_propka_new"))
    if xtb_keep is not None:
        merged = merged.merge(xtb_keep, on="row_id", how="left", suffixes=("", "_xtb_new"))

    # boltz_pose_locally_available
    merged["boltz_pose_locally_available"] = merged["row_id"].isin(boltz_metrics.keys())

    # Coalesce-prefer-newer: for every *_<src>_new column, fill the original col
    # from the newer one wherever the original is NaN. This is the "additive,
    # don't replace the original 2010-row merge" pattern from the orchestrator brief.
    suffixes = ("_boltz_new", "_vina_new", "_cov_new", "_full_new", "_shape_new",
                "_strain_new", "_propka_new", "_xtb_new")
    new_suffix_cols = [c for c in merged.columns if c.endswith(suffixes)]
    n_coalesced = 0
    for new_c in new_suffix_cols:
        for suf in suffixes:
            if new_c.endswith(suf):
                base = new_c[: -len(suf)]
                break
        if base in merged.columns:
            # Prefer-newer: take new value when present, else keep old.
            new_vals = merged[new_c]
            try:
                mask_has_new = new_vals.notna()
            except Exception:
                mask_has_new = pd.Series([v is not None for v in new_vals], index=merged.index)
            merged.loc[mask_has_new, base] = new_vals[mask_has_new]
            n_coalesced += 1
    print(f"  coalesced {n_coalesced} new columns into existing cohort cols (prefer-existing-non-null)")

    # Drop the *_new suffix columns now that we've coalesced them in.
    drop_cols = [c for c in merged.columns if c.endswith(suffixes)]
    if drop_cols:
        print(f"  dropping {len(drop_cols)} now-redundant suffix-collision cols")
        merged = merged.drop(columns=drop_cols)

    # 5. Summary stats
    print("\n=== Coverage ===")
    new_cols = [
        "boltz_confidence_score", "boltz_ptm", "boltz_iptm", "boltz_complex_plddt",
        "vina_rescore_affinity_kcalmol", "covvina_rescore_affinity_kcalmol",
        "boltz_pose_locally_available",
        # Geometry / contacts / mPAE proxies
        "d_SG", "burgi_dunitz_dev_deg", "n_h_bonds",
        "n_stabilizing_contacts", "pocket_occupancy_pct", "atp_pocket_fraction",
        "hinge_hbond", "geom_ok",
        "mPAE_paper", "mPAE_full", "mPAE_interface",
        # Mol1-anchor shape + USRCAT (ESP proxy)
        "shape_Tc_seed", "esp_sim_seed",
        # v2 sources
        "rdkit_strain_kcal_mol", "pKa_Cys346",
        "HOMO_eV", "LUMO_eV", "gap_eV", "omega_eV", "q_Cb", "fukui_plus_Cb",
    ]
    for c in new_cols:
        if c not in merged.columns:
            print(f"  {c}: MISSING")
            continue
        filled = merged[c].notna().sum() if merged[c].dtype != bool else merged[c].sum()
        print(f"  {c}: {filled:,}/{len(merged):,} ({100*filled/len(merged):.1f}%)")

    # 6. Quick stats on key metrics
    print("\n=== Key metric distributions (mols with Boltz pose) ===")
    sub = merged[merged["boltz_pose_locally_available"]]
    for c, ascending in [
        ("boltz_confidence_score", False), ("boltz_iptm", False), ("boltz_complex_plddt", False),
        ("vina_rescore_affinity_kcalmol", True), ("covvina_rescore_affinity_kcalmol", True),
    ]:
        if c not in sub.columns:
            continue
        s = pd.to_numeric(sub[c], errors="coerce").dropna()
        if len(s) == 0:
            continue
        print(f"  {c}: n={len(s):,}  median={s.median():.3f}  p10={s.quantile(0.10):.3f}  p90={s.quantile(0.90):.3f}  min={s.min():.3f}  max={s.max():.3f}")

    # 7. Write
    merged.to_csv(COHORT_CSV, index=False)
    print(f"\nWrote {COHORT_CSV} ({len(merged):,} rows, {len(merged.columns)} cols)")

    # 8. Standalone summary CSV (just the new boltz/vina cols)
    summary_cols = ["row_id", "smiles"] + new_cols + [
        "boltz_ligand_iptm", "boltz_protein_iptm", "boltz_complex_iplddt",
        "boltz_complex_pde", "boltz_complex_ipde",
        "vina_rescore_inter_kcalmol", "vina_rescore_intra_kcalmol", "vina_rescore_torsions_kcalmol",
        "covvina_rescore_inter_kcalmol", "covvina_rescore_intra_kcalmol", "covvina_rescore_torsions_kcalmol",
        "warhead_class", "warhead_atom_name", "n_covalent", "n_salt_bridges", "n_pi_pi",
        "mPAE_min", "mPAE_proxy_kind", "mPAE_min_available",
    ]
    summary_cols = [c for c in summary_cols if c in merged.columns]
    summary = merged[summary_cols]
    summary_path = PROJECT_ROOT / "data/tier4_scored/boltz_cohort_metrics_summary.csv"
    summary.to_csv(summary_path, index=False)
    print(f"Summary → {summary_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
