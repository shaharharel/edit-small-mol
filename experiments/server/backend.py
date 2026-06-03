#!/usr/bin/env python3
"""
Flask backend for live filtering of 498K Tier 2 SCALED candidates.

Loads products_scored_full.csv into memory at startup, serves DataTables
server-side AJAX requests with SearchBuilder filter conditions parsed into
pandas queries. Plus quick chip presets, structure-on-demand SVG rendering.

Run:
    cd ~/Documents/github/edit-small-mol
    conda run -n quris python experiments/server/backend.py
    # listens on http://localhost:5000

Frontend HTML lives at: results/paper_evaluation/overnight_method_report_live.html
                        (Tier 2 SCALED section uses serverSide: true with this backend)
"""

import json
import sys
from pathlib import Path
from io import BytesIO
import warnings

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
warnings.filterwarnings("ignore")

from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Draw
RDLogger.DisableLog('rdApp.*')

DATA_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
LEGACY_FILE = PROJECT_ROOT / "results" / "paper_evaluation" / "aichem_tier2_scaled" / "products_scored_full.csv"

app = Flask(__name__)
CORS(app)

print(f"Loading {DATA_FILE} ...")
if DATA_FILE.exists():
    DF = pd.read_csv(DATA_FILE)
else:
    print(f"  Falling back to {LEGACY_FILE}")
    DF = pd.read_csv(LEGACY_FILE)
    DF["method"] = "Tier 2 SCALED — Fragment Replacement (498K)"
    if "pIC50_film" in DF.columns and "pIC50_method" not in DF.columns:
        DF["pIC50_method"] = DF["pIC50_film"]
print(f"Loaded {len(DF):,} candidates with columns: {list(DF.columns)}")
if "method" in DF.columns:
    print(f"Methods: {DF['method'].value_counts().to_dict()}")
DF = DF.reset_index(drop=True)
DF["row_id"] = DF.index
# ── Method unification + tier grouping (per user 2026-05-04 spec) ──
#  Tier 1: "Medchem Rules"  ← Tier 1 + Tier 1.5 unified
#  Tier 2: "Amine Replacements"  ← Tier 2 + Tier 2 SCALED unified
#  Tier 3: "Constrained Generative"  ← LibInvent locked / Mol2Mol+wh / De Novo+wh / v2 (kept as sub-methods)
#  Tier 4: removed for now (Tier 4 De Novo / Tier 4 Mol2Mol / Method A / Method B filtered out)
if "method" in DF.columns:
    DF["method"] = DF["method"].replace({
        "Tier 1 — Med-Chem Playbook (rule-based)": "Medchem Rules",
        "Tier 1.5 — Warhead Controls + Med-Chem Tricks": "Medchem Rules",
        "Tier 2 — Fragment Replacement (curated 204)": "Amine Replacements",
        "Tier 2 SCALED — Fragment Replacement (498K)": "Amine Replacements",
        "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)": "Amine Replacements",
        # Tier 3 v2 + Tier 3 v3 sub-methods kept as-is, will group in frontend
    })
    # Drop Tier 4 + Methods A/B (generate-and-filter family) for now
    DROPPED = [
        "Tier 4 — De Novo unconstrained",
        "Tier 4 — Mol2Mol unconstrained",
        "Method A — De Novo FiLMDelta-driven",
        "Method B — Mol2Mol FiLMDelta-driven",
    ]
    pre = len(DF)
    DF = DF[~DF["method"].isin(DROPPED)].reset_index(drop=True)
    print(f"  Filtered out Tier 4 / Methods A/B: {pre:,} -> {len(DF):,}")
    # Drop warhead-modifying controls — every reported candidate must have warhead intact
    if "warhead_intact" in DF.columns:
        pre = len(DF)
        DF = DF[DF["warhead_intact"] == True].reset_index(drop=True)
        print(f"  Filtered out warhead-MODIFIED candidates: {pre:,} -> {len(DF):,}")
    # Drop disconnected-fragment SMILES (Tier 1 ReplaceSubstructs sometimes produces
    # mol1.fragment2 SMILES where a bond was inadvertently broken; these pass the warhead
    # SMARTS via the first fragment but aren't real candidates).
    if "smiles" in DF.columns:
        pre = len(DF)
        DF = DF[~DF["smiles"].astype(str).str.contains(".", regex=False, na=False)].reset_index(drop=True)
        print(f"  Filtered out disconnected (multi-fragment) SMILES: {pre:,} -> {len(DF):,}")
    DF["row_id"] = DF.index
    print(f"  Methods after unification: {sorted(DF['method'].unique().tolist())}")
# Use pIC50_method as the primary pIC50 (uniform across methods)
if "pIC50_method" in DF.columns and "pIC50_film" not in DF.columns:
    DF["pIC50_film"] = DF["pIC50_method"]
elif "pIC50_film" not in DF.columns and "pIC50_method" in DF.columns:
    DF["pIC50_film"] = DF["pIC50_method"]

# ── Merge Boltz cofold metrics ──────────────────────────────────────────────
# Pre-2026-05-09: previous Cys560-targeted cofolds were deprecated after a QA
# audit found Cys346 (P-loop, GxGxxG motif) is the literature-validated
# ZAP70 covalent target (PMID 33845236, 37594408), not Cys560. Old data is
# under _DEPRECATED_cys560_*; we wait for the Cys346 redo to repopulate
# the Boltz columns. Until then, the columns are simply NaN.
TOP1000_MAN = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
ENERGY_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_energy_scores.csv"
if TOP1000_MAN.exists():
    _b = json.loads(TOP1000_MAN.read_text())
    boltz_rows = []
    for rid_str, m in _b.items():
        boltz_rows.append({
            "row_id": int(rid_str),
            "yaml_name": m.get("yaml_name"),
            "mPAE": m["mPAE"],
            "iptm": m["iptm"],
            "ligand_iptm": m["ligand_iptm"],
            "boltz_plddt": m["complex_plddt"],
            "boltz_pde": m["complex_pde"],
            "boltz_confidence": m["confidence_score"],
            "combined_score": m["combined_score"],
        })
    boltz_df = pd.DataFrame(boltz_rows)
    DF = DF.merge(boltz_df, on="row_id", how="left")
    n_with = DF["mPAE"].notna().sum()
    print(f"Merged Cys346 Boltz cofold metrics for {n_with} of {len(DF)} rows")
else:
    print(f"No Cys346 cofold manifest yet at {TOP1000_MAN.name}; Boltz columns will be NaN until cofold redo completes")

# ── MM-GBSA energy merge is DISABLED until scorer bugs are fixed ────────────
# QA audit (2026-05-21) flagged 4 critical bugs in score_cofold_energy.py:
#   1. MMFFOptimizeMolecule scrambles the Boltz pose before scoring (HETEROSCEDASTIC)
#   2. GB radius patch (Sγ=1.80) is no-op on GBSAOBCForce (silent)
#   3. CustomBondForce restraint energy contaminates dG_bind
#   4. Stage-2 minimization releases all restraints (not just near-ligand)
# Expected rank correlation with combined_score: |ρ| < 0.15. Killing this path.
# When a fixed scorer is built, re-enable the merge below. ENERGY_CSV exists
# but has only 3 untrustworthy rows from the killed run.
ENABLE_ENERGY_MERGE = False
if ENABLE_ENERGY_MERGE and ENERGY_CSV.exists():
    _e = pd.read_csv(ENERGY_CSV)
    _e = _e[_e["success_flag"] == 1].copy()
    if len(_e):
        _e_merge = _e.rename(columns={"name": "yaml_name"})[
            ["yaml_name", "dG_bind_kcalmol", "ligand_strain_kcalmol",
             "rmsd_min_A", "sg_cb_dist_A"]
        ].drop_duplicates(subset=["yaml_name"], keep="last")
        if "yaml_name" in DF.columns:
            DF = DF.merge(_e_merge, on="yaml_name", how="left")
            n_e = DF["dG_bind_kcalmol"].notna().sum()
            print(f"Merged MM-GBSA energy for {n_e} of {len(DF)} rows  ({len(_e)} successful)")
else:
    print("MM-GBSA energy merge disabled (scorer needs rebuild per QA audit)")

# ── Expose extra Boltz geometry/confidence fields from top1000 manifest ─────
# d_SG (warhead-Cys distance), ligand_iptm, complex_plddt already populated
# above as 'boltz_plddt'. d_SG is the defensible warhead-geometry signal we
# add now in lieu of broken energy terms.
if TOP1000_MAN.exists() and "row_id" in DF.columns:
    extra_rows = []
    for rid_str, m in _b.items():
        extra_rows.append({
            "row_id": int(rid_str),
            "d_SG": m.get("d_SG"),
            "geom_ok": m.get("geom_ok"),
            "n_h_bonds": m.get("n_h_bonds"),
            "n_stabilizing_contacts": m.get("n_stabilizing_contacts"),
            "pocket_occupancy_pct": m.get("pocket_occupancy_pct"),
        })
    extra_df = pd.DataFrame(extra_rows)
    DF = DF.merge(extra_df, on="row_id", how="left")
    print(f"Exposed d_SG, geom_ok, contact counts, pocket_occupancy for {DF['d_SG'].notna().sum()} rows")

# ── Merge PROPKA3 Cys346 pKa per-pose ─────────────────────────────────────
# Per Olsson 2011 / Søndergaard 2011, PROPKA3 estimates a per-pose,
# ligand-environment-aware pKa for the Cys346 nucleophile. Together with
# xTB warhead electrophilicity, this completes the covalent-rate story
# (nucleophile activation + electrophile). Computed by
# experiments/propka_cys346_batch.py over the 997 Boltz cofolds.
PROPKA_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_propka.csv"
if PROPKA_CSV.exists() and "row_id" in DF.columns:
    _pk = pd.read_csv(PROPKA_CSV)
    _pk_merge = _pk[_pk["success_flag"] == 1][["row_id", "pKa_Cys346"]].drop_duplicates(
        subset=["row_id"], keep="last"
    )
    DF = DF.merge(_pk_merge, on="row_id", how="left")
    n_pk = DF["pKa_Cys346"].notna().sum()
    print(f"Merged PROPKA3 pKa_Cys346 for {n_pk} of {len(DF)} rows")
else:
    print(f"No PROPKA3 pKa CSV at {PROPKA_CSV}; pKa_Cys346 will be NaN")

# ── Merge pubTc v3 panel similarity (300-lead, primary; v1/v2 removed) ────
# Computed by experiments/compute_pubtc_panel_v3.py from the curated 300-lead
# panel of published kinase covalent inhibitors (experiments/pubtc_panel_v3.py).
# Covers ≥20 target classes (BTK/EGFR/KRAS-G12C/HER2/SYK/JAK/FGFR/MEK/BMX/TEC/
# TYK2/ITK/ALK/MET/ABL/RIPK1/KIT/CDK7-9/MELK/AURKA/GAK/ERK/RSK + warhead probes).
# Replaces both the v1 (11-lead) and v2 (30-lead) panels — those columns are
# *dropped*, not renamed, per user instruction.
#
# Columns merged (primary, unsuffixed):
#   max_pubTc            max Tanimoto across 300 leads
#   mean_pubTc           mean Tanimoto
#   median_pubTc         median Tanimoto
#   top10_mean_pubTc     mean of the top-10 Tanimotos (broader than v1/v2 top-3)
#   closest_lead         drug_name of highest-Tanimoto lead
PUBTC_V3_CSV = PROJECT_ROOT / "data" / "paper_evaluation" / "pubtc_scores_v3.csv"

if PUBTC_V3_CSV.exists():
    _pt3 = pd.read_csv(PUBTC_V3_CSV)
    DF = DF.merge(_pt3, on="row_id", how="left")
    n_with_v3 = DF["max_pubTc"].notna().sum()
    print(f"Merged pubTc v3 (300-lead) panel similarities for {n_with_v3} rows")
else:
    print(f"No pubTc v3 panel scores at {PUBTC_V3_CSV}; pubTc columns will be NaN")

# ── Merge classic medchem ligand-efficiency metrics ─────────────────────────
# Computed by experiments/compute_efficiency_metrics.py from existing columns
# (pIC50_method, MW, LogP, HeavyAtoms, TPSA, HBA, HBD) + RDKit fsp3.
# Exposes: LLE, LE, BEI, SEI, SILE, fsp3, Lipinski_violations
# Citations: Hopkins 2014 (LLE); Hopkins, Groom, Alex 2004 (LE);
#   Abad-Zapatero & Metz 2005 (BEI, SEI); Reynolds 2008 (SILE);
#   Lovering 2009 (fsp3); Lipinski 1997 (rule-of-5).
EFFICIENCY_CSV = PROJECT_ROOT / "data" / "paper_evaluation" / "efficiency_metrics.csv"
if EFFICIENCY_CSV.exists():
    _eff = pd.read_csv(EFFICIENCY_CSV)
    DF = DF.merge(_eff, on="row_id", how="left")
    n_with = DF["LLE"].notna().sum()
    print(f"Merged efficiency metrics (LLE/LE/BEI/SEI/SILE/fsp3/Lipinski_violations) for {n_with} rows")
else:
    print(f"No efficiency metrics at {EFFICIENCY_CSV}; efficiency columns will be NaN")

# ── xTB GFN2 warhead electrophilicity descriptors ───────────────────────────
# k_inact (covalent reaction-rate) signal — the missing half of
# dG_total = dG_recognition + ΔG_react.  Computed by
# experiments/xtb_warhead_electrophilicity.py for all 997 Cys346 cofold
# ligands (acrylamide [CH2]=[CH]-[C](=O)-[N] Cβ identified via SMARTS,
# free-state ETKDG+MMFF geometry, xTB --gfn 2 --vfukui single point).
# Exposed columns:
#   LUMO_eV         frontier orbital energy (lower = more electrophilic)
#   HOMO_eV         frontier occupied orbital energy
#   gap_eV          HOMO-LUMO gap
#   omega_eV        Parr global electrophilicity ω = (H+L)²/(2·gap)
#   q_Cb            Mulliken charge on Cβ (xTB scale; more-negative = slower)
#   fukui_plus_Cb   local electrophilic Fukui f+ on Cβ
#   pred_log_k2_GSH heuristic log k2 for GSH addition (Flanagan JMC 2014
#                   style; calibrated to span ~-3 → +1 in literature units)
# Note: xTB GFN2 LUMOs are systematically ~4-5 eV lower than DFT/B3LYP
# (internal calibration). Use these for ranking, not absolute kinetic
# interpretation. Rank correlation with combined_score is modest
# (|Spearman| ≈ 0.18-0.24), confirming orthogonality from K_I signal.
WARHEAD_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_warhead_reactivity.csv"
if WARHEAD_CSV.exists() and "row_id" in DF.columns:
    _w = pd.read_csv(WARHEAD_CSV)
    _w = _w[_w["success_flag"] == 1].copy()
    _w_merge = _w[[
        "row_id", "LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV",
        "q_Cb", "fukui_plus_Cb", "pred_log_k2_GSH",
    ]].drop_duplicates(subset=["row_id"], keep="last")
    DF = DF.merge(_w_merge, on="row_id", how="left")
    n_w = DF["LUMO_eV"].notna().sum()
    print(f"Merged xTB warhead electrophilicity (LUMO/omega/q_Cb/f+/log_k2_GSH) for {n_w} of {len(DF)} rows")
else:
    print(f"No xTB warhead reactivity file at {WARHEAD_CSV.name}; descriptors will be NaN")

# ── AutoDock Vina rescore (score_only) for all 997 Cys346 cofold poses ─────
# Trott & Olson 2010 knowledge-based scoring function. Orthogonal physics to
# MM-GBSA (force field) and complementary to Boltz ML confidence — catches
# MM-GBSA pathologies (entropy neglect, clash mis-scoring). Computed by
# experiments/vina_rescore_batch.py from the existing Boltz cofold poses
# (no re-docking; --score_only on the pre-positioned ligand).
# Exposed columns:
#   vina_kcalmol         Estimated free energy of binding (kcal/mol); primary
#   vina_inter_kcalmol   Final intermolecular energy (ligand-receptor)
#   vina_intra_kcalmol   Final total internal energy (ligand torsions)
# Real kinase covalent inhibitors typically score -7 to -10 in Vina; values
# below -2 indicate plausible binders, above 0 indicates significant clash.
VINA_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_vina_rescore.csv"
if VINA_CSV.exists() and "row_id" in DF.columns:
    _v = pd.read_csv(VINA_CSV)
    _v = _v[_v["success_flag"] == 1].copy()
    _v_merge = _v[[
        "row_id", "vina_kcalmol", "vina_inter_kcalmol", "vina_intra_kcalmol",
    ]].drop_duplicates(subset=["row_id"], keep="last")
    DF = DF.merge(_v_merge, on="row_id", how="left")
    n_v = DF["vina_kcalmol"].notna().sum()
    print(f"Merged Vina rescore (vina_kcalmol / inter / intra) for {n_v} of {len(DF)} rows")
else:
    print(f"No Vina rescore file at {VINA_CSV.name}; vina_kcalmol will be NaN")

# ── RDKit MMFF ligand strain + UFF vdW (cofold) ────────────────────────────
# H-only relaxation (heavy atoms fixed) under MMFF94s, then MMFF SP energy of
# the bound pose, minus best-of-5 free-conformer reference. Pose-aware strain
# for the 997 Cys346 Boltz cofolds. vdW interaction (UFF Lennard-Jones,
# 8 A cutoff, 2.5 A min-dist to exclude warhead-Cys bond partner) is an
# orthogonal physics signal complementary to Vina.
# Computed by experiments/score_rdkit_strain_batch.py --mode cofold.
RDKIT_STRAIN_COFOLD_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_rdkit_strain_cofold.csv"
if RDKIT_STRAIN_COFOLD_CSV.exists() and "row_id" in DF.columns:
    _rs = pd.read_csv(RDKIT_STRAIN_COFOLD_CSV)
    _rs = _rs[_rs["success_flag"] == 1].copy()
    _rs_merge = _rs[[
        "row_id", "strain_kcal_mol", "vdw_interaction_kcal_mol",
    ]].rename(columns={
        "strain_kcal_mol": "rdkit_strain_kcal_mol",
    }).drop_duplicates(subset=["row_id"], keep="last")
    DF = DF.merge(_rs_merge, on="row_id", how="left")
    n_rs = DF["rdkit_strain_kcal_mol"].notna().sum()
    print(f"Merged RDKit MMFF strain + vdW (cofold) for {n_rs} of {len(DF)} rows")
else:
    print(f"No RDKit strain cofold file at {RDKIT_STRAIN_COFOLD_CSV.name}; rdkit strain columns will be NaN")

# ── RDKit pose-free constitutional strain (~520K rows) ─────────────────────
# Single ETKDGv3 conformer ("bound proxy", H-only relax) MMFF SP minus
# best-of-5 free conformer MMFF energy. Scaffold-independent intrinsic
# strain proxy for all dashboard candidates (covers methods that lack cofold
# poses). Computed by experiments/score_rdkit_strain_batch.py --mode posefree.
# Supports the canonical single file OR a directory of shard CSVs written
# by independent --shard idx/total invocations (concatenated in-memory).
RDKIT_STRAIN_POSEFREE_CSV = PROJECT_ROOT / "data" / "paper_evaluation" / "rdkit_strain_posefree.csv"
RDKIT_STRAIN_POSEFREE_SHARDS = PROJECT_ROOT / "data" / "paper_evaluation" / "posefree_shards"

_rp_df = None
if RDKIT_STRAIN_POSEFREE_CSV.exists() and RDKIT_STRAIN_POSEFREE_CSV.stat().st_size > 0:
    try:
        _rp_df = pd.read_csv(RDKIT_STRAIN_POSEFREE_CSV)
    except pd.errors.EmptyDataError:
        _rp_df = None
elif RDKIT_STRAIN_POSEFREE_SHARDS.exists() and RDKIT_STRAIN_POSEFREE_SHARDS.is_dir():
    shard_dfs = []
    for shard in sorted(RDKIT_STRAIN_POSEFREE_SHARDS.glob("shard_*.csv")):
        if shard.stat().st_size == 0:
            continue
        try:
            shard_dfs.append(pd.read_csv(shard))
        except pd.errors.EmptyDataError:
            continue
    if shard_dfs:
        _rp_df = pd.concat(shard_dfs, ignore_index=True)
        print(f"  (loaded {len(shard_dfs)} pose-free shards: {len(_rp_df)} rows)")

if _rp_df is not None and "row_id" in DF.columns and "success_flag" in _rp_df.columns:
    _rp = _rp_df[_rp_df["success_flag"] == 1].copy()
    _rp_merge = _rp[["row_id", "strain_kcal_mol"]].rename(
        columns={"strain_kcal_mol": "rdkit_strain_posefree_kcal_mol"}
    ).drop_duplicates(subset=["row_id"], keep="last")
    DF = DF.merge(_rp_merge, on="row_id", how="left")
    n_rp = DF["rdkit_strain_posefree_kcal_mol"].notna().sum()
    print(f"Merged RDKit pose-free strain for {n_rp} of {len(DF)} rows")
else:
    print(f"No RDKit pose-free strain available yet (file={RDKIT_STRAIN_POSEFREE_CSV.name}, "
          f"shards={RDKIT_STRAIN_POSEFREE_SHARDS.name}); pose-free strain will be NaN")

# ── Merge MM-GBSA dG_recognition (capped-analog, MD trajectory) ─────────────
# Cheng et al. JCTC 2017 capped-analog protocol: acrylamide is replaced with
# propionamide BEFORE topology build to compute the non-covalent recognition
# step (K_I component) of the irreversible-inhibitor mechanism. AmberTools
# MMPBSA.py.MPI (np=4), ff14SB + GAFF2, igb=8 GBn2, mbondi3 radii,
# saltcon=0.15M, 200 frames sampled from 1 ns NPT MD. Joined on row_id.
# CSV is growing (V100 batch still adding rows); how="left" so missing rows
# stay null until the next backend restart picks them up.
MDMMGBSA_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_capped_mmpbsa_top50.csv"
if MDMMGBSA_CSV.exists() and "row_id" in DF.columns:
    _md = pd.read_csv(MDMMGBSA_CSV)
    if "success_flag" in _md.columns:
        _md = _md[_md["success_flag"] == 1].copy()
    _md_merge = _md[[
        "row_id", "dG_recognition_md_kcalmol", "dG_recognition_md_std",
        "ggas_kcalmol", "gsolv_kcalmol",
    ]].drop_duplicates(subset=["row_id"], keep="last")
    DF = DF.merge(_md_merge, on="row_id", how="left")
    n_md = DF["dG_recognition_md_kcalmol"].notna().sum()
    print(f"[merge] mdmmgbsa: {n_md} of {len(DF)} rows matched")
else:
    print(f"No MD-MM-GBSA capped-analog CSV at {MDMMGBSA_CSV.name}; "
          f"dG_recognition_md_* will be NaN")

# ── Brenk structural alerts (count) ─────────────────────────────────────────
# Brenk et al. ChemMedChem 2008 — reactive/toxic substructure filter,
# complementary to PAINS (which targets pan-assay interference). Catches
# acyl ureas, anhydrides, Michael acceptors outside the intended warhead,
# disulfides, etc. RDKit FilterCatalog with BRENK catalog (~110 SMARTS).
# Cached to data/paper_evaluation/brenk_alerts.csv (row_id, Brenk_alerts).
# First-time cost ~2 min on 520K rows; subsequent loads are instant.
BRENK_CACHE = PROJECT_ROOT / "data" / "paper_evaluation" / "brenk_alerts.csv"
_brenk_loaded_from_cache = False
if BRENK_CACHE.exists() and "row_id" in DF.columns:
    try:
        _bk = pd.read_csv(BRENK_CACHE)
        # Cache validity check: must cover every row_id currently in DF.
        if set(DF["row_id"]).issubset(set(_bk["row_id"])):
            DF = DF.merge(
                _bk[["row_id", "Brenk_alerts"]].drop_duplicates(subset=["row_id"]),
                on="row_id", how="left",
            )
            n_bk = DF["Brenk_alerts"].notna().sum()
            print(f"[brenk] loaded cache: {n_bk} of {len(DF)} rows from {BRENK_CACHE.name}")
            _brenk_loaded_from_cache = True
        else:
            print(f"[brenk] cache stale (missing rows); will recompute")
    except Exception as _e:
        print(f"[brenk] cache load failed ({_e}); will recompute")

if not _brenk_loaded_from_cache and "smiles" in DF.columns:
    import time as _time
    from rdkit.Chem import FilterCatalog as _FilterCatalog
    _bparams = _FilterCatalog.FilterCatalogParams()
    _bparams.AddCatalog(_FilterCatalog.FilterCatalogParams.FilterCatalogs.BRENK)
    _brenk_cat = _FilterCatalog.FilterCatalog(_bparams)

    def _count_brenk(smi):
        if not isinstance(smi, str):
            return None
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        return int(len(_brenk_cat.GetMatches(m)))

    _t0 = _time.time()
    DF["Brenk_alerts"] = DF["smiles"].apply(_count_brenk)
    _dt = _time.time() - _t0
    n_bk = DF["Brenk_alerts"].notna().sum()
    print(f"[brenk] computed for {n_bk} of {len(DF)} rows in {_dt:.1f}s")
    # Write cache for next startup (best-effort; ignore disk errors).
    try:
        BRENK_CACHE.parent.mkdir(parents=True, exist_ok=True)
        DF[["row_id", "Brenk_alerts"]].to_csv(BRENK_CACHE, index=False)
        print(f"[brenk] cached -> {BRENK_CACHE.name}")
    except Exception as _e:
        print(f"[brenk] cache write skipped ({_e})")

NUMERIC_COLS = [c for c in DF.columns if pd.api.types.is_numeric_dtype(DF[c])]
print(f"Numeric columns: {NUMERIC_COLS}")


# ── Cohort diversity (observe-only, mode-collapse alarm) ────────────────────
# Pre-computed by experiments/cohort_diversity_backfill.py for the cohort-
# comparison CSV cohorts (Lingo / Mol2Mol / DeNovo / LibInvent / Amine).
# Schema (cohort, group, n_total, n_unique, diversity_ratio,
# mean_intra_nn_tanimoto, max_intra_tanimoto, n_bemis_murcko_scaffolds,
# mean_pairwise_dissim). Loaded once at startup, exposed via /api/cohort_diversity.
# Method-level diversity (computed on the live DF's per-method SMILES) is
# computed lazily on first request to /api/methods?with_diversity=1 and cached
# in process memory (METHOD_DIVERSITY_CACHE).
COHORT_DIVERSITY_CSV = PROJECT_ROOT / "results" / "paper_evaluation" / "cohort_comparison" / "cohort_diversity_metrics.csv"
COHORT_DIVERSITY: pd.DataFrame | None = None
if COHORT_DIVERSITY_CSV.exists():
    try:
        COHORT_DIVERSITY = pd.read_csv(COHORT_DIVERSITY_CSV)
        print(f"[diversity] loaded {len(COHORT_DIVERSITY)} cohort rows from {COHORT_DIVERSITY_CSV.name}")
    except Exception as _e:
        print(f"[diversity] failed to load {COHORT_DIVERSITY_CSV.name}: {_e}")
else:
    print(f"[diversity] no precomputed cohort table at {COHORT_DIVERSITY_CSV}; run experiments/cohort_diversity_backfill.py")

METHOD_DIVERSITY_CACHE: dict[str, dict] | None = None


def _compute_method_diversity_cache() -> dict[str, dict]:
    """Lazy per-method diversity (one row per `method` in the live DF).

    Reuses src.utils.diversity.compute_diversity_metrics. Result is cached so
    that /api/methods stays responsive after first call. Costs ~1–2 min on
    520K rows the first time (Morgan FP + Bemis-Murcko per unique SMILES,
    over each method group); subsequent calls are O(1).
    """
    global METHOD_DIVERSITY_CACHE
    if METHOD_DIVERSITY_CACHE is not None:
        return METHOD_DIVERSITY_CACHE
    sys.path.insert(0, str(PROJECT_ROOT))
    from src.utils.diversity import compute_diversity_metrics
    out: dict[str, dict] = {}
    if "method" not in DF.columns or "smiles" not in DF.columns:
        METHOD_DIVERSITY_CACHE = out
        return out
    for m, sub in DF.groupby("method"):
        smis = sub["smiles"].dropna().tolist()
        div = compute_diversity_metrics(smis)
        out[m] = div
        print(f"[diversity] method={m}: n_total={div['n_total']}, n_unique={div['n_unique']}, "
              f"ratio={div['diversity_ratio']:.3f}, mean_nn_Tc="
              f"{div['mean_intra_nn_tanimoto']:.3f}, BM_scaf={div['n_bemis_murcko_scaffolds']}")
    METHOD_DIVERSITY_CACHE = out
    return out


# ── Filtering & Ranking Spec (single source of truth) ───────────────────────
# Mirrors docs/filtering_ranking_spec.md. Both the spec doc and the UI render
# from this object via /api/spec. Edit here = edit the spec.

FILTER_SPEC_VERSION = "0.1"

# Each filter entry:
#   id        : stable identifier (used in localStorage + API keys)
#   col       : DF column the filter reads
#   op        : one of "<", ">", "<=", ">=", "==", "!="
#   cutoff    : numeric or bool; the threshold
#   default_on: bool; whether this filter is on by default
#   active_if : "always" | "col_populated" — F3-5 placeholders only apply when col is non-null
#   label     : short UI label
#   rationale : one-liner citation; full reasoning in docs/filtering_ranking_spec.md
#
# A molecule FAILS this filter when (df[col] OP cutoff) evaluates True.
# I.e. "reject if MW > 700" => op=">" cutoff=700.

FILTER_SPEC = {
    "version": FILTER_SPEC_VERSION,
    "layer1": {
        "label": "Filter 1: Hard exclude",
        "description": "Layer 1 — hard medchem exclusions. ALL ON by default.",
        "filters": [
            {"id": "f1_pic50_mean",       "col": "pIC50_mean",          "op": "<",  "cutoff": 5.5,   "default_on": True, "active_if": "always",        "label": "pIC50_mean < 5.5",           "rationale": "Sub-µM only; Leeson NRDD 2007 cellular ≥6."},
            {"id": "f1_warhead",          "col": "warhead_intact",      "op": "==", "cutoff": False, "default_on": True, "active_if": "col_populated", "label": "warhead intact = False",     "rationale": "Acrylamide required for covalent design."},
            {"id": "f1_mw",               "col": "MW",                  "op": ">",  "cutoff": 700,   "default_on": True, "active_if": "always",        "label": "MW > 700",                   "rationale": "Oral kinase max ≈650 (Patel 2020)."},
            {"id": "f1_logp_hi",          "col": "LogP",                "op": ">",  "cutoff": 6.0,   "default_on": True, "active_if": "always",        "label": "LogP > 6",                   "rationale": "Lipinski + 1; CYP3A4/hERG (Hughes 3/75)."},
            {"id": "f1_logp_lo",          "col": "LogP",                "op": "<",  "cutoff": -1.0,  "default_on": True, "active_if": "always",        "label": "LogP < −1",                  "rationale": "Permeability floor."},
            {"id": "f1_tpsa",             "col": "TPSA",                "op": ">",  "cutoff": 180,   "default_on": True, "active_if": "always",        "label": "TPSA > 180",                 "rationale": "Veber 2002 ≤140."},
            {"id": "f1_hbd",              "col": "HBD",                 "op": ">",  "cutoff": 7,     "default_on": True, "active_if": "always",        "label": "HBD > 7",                    "rationale": "Lipinski 1997 ≤5."},
            {"id": "f1_rotbonds",         "col": "RotBonds",            "op": ">",  "cutoff": 14,    "default_on": True, "active_if": "always",        "label": "RotBonds > 14",              "rationale": "Veber ≤10."},
            {"id": "f1_lipinski_viol",    "col": "Lipinski_violations", "op": ">=", "cutoff": 3,     "default_on": True, "active_if": "always",        "label": "Lipinski_violations ≥ 3",    "rationale": "3 violations = not oral."},
            {"id": "f1_sascore",          "col": "SAScore",             "op": ">",  "cutoff": 7.0,   "default_on": True, "active_if": "always",        "label": "SAScore > 7",                "rationale": "Ertl 2009 — unsynthesisable."},
            {"id": "f1_pains",            "col": "PAINS_alerts",        "op": ">=", "cutoff": 1,     "default_on": True, "active_if": "always",        "label": "PAINS_alerts ≥ 1",           "rationale": "Warhead is NOT PAINS — any hit real."},
            {"id": "f1_dsg",              "col": "d_SG",                "op": ">",  "cutoff": 6.0,   "default_on": True, "active_if": "col_populated", "label": "d_SG > 6 Å",                 "rationale": "Michael-TS vdW ≤3.5; >6 unproductive."},
        ],
    },
    "layer2": {
        "label": "Filter 2: Soft cuts",
        "description": "Layer 2 — softer medchem / structural cuts. ALL ON by default.",
        "filters": [
            {"id": "f2_pic50_method",     "col": "pIC50_method",        "op": "<",  "cutoff": 6.0,   "default_on": True, "active_if": "always",        "label": "pIC50_method < 6.0",         "rationale": "Fallback single-seed potency."},
            {"id": "f2_pic50_std",        "col": "pIC50_std",           "op": ">",  "cutoff": 0.6,   "default_on": True, "active_if": "col_populated", "label": "pIC50_std > 0.6",            "rationale": "Ensemble disagreement."},
            {"id": "f2_anchor_wins",      "col": "anchor_wins",         "op": "<",  "cutoff": 80,    "default_on": True, "active_if": "col_populated", "label": "anchor_wins < 80",           "rationale": "<30% wins of 280 anchors."},
            {"id": "f2_mw",               "col": "MW",                  "op": ">",  "cutoff": 550,   "default_on": True, "active_if": "always",        "label": "MW > 550",                   "rationale": "Oral kinase upper bound."},
            {"id": "f2_logp",             "col": "LogP",                "op": ">",  "cutoff": 5.0,   "default_on": True, "active_if": "always",        "label": "LogP > 5",                   "rationale": "Lipinski."},
            {"id": "f2_tpsa",             "col": "TPSA",                "op": ">",  "cutoff": 140,   "default_on": True, "active_if": "always",        "label": "TPSA > 140",                 "rationale": "Veber."},
            {"id": "f2_hba",              "col": "HBA",                 "op": ">",  "cutoff": 12,    "default_on": True, "active_if": "always",        "label": "HBA > 12",                   "rationale": "Lipinski +2."},
            {"id": "f2_hbd",              "col": "HBD",                 "op": ">",  "cutoff": 5,     "default_on": True, "active_if": "always",        "label": "HBD > 5",                    "rationale": "Lipinski."},
            {"id": "f2_rotbonds",         "col": "RotBonds",            "op": ">",  "cutoff": 11,    "default_on": True, "active_if": "always",        "label": "RotBonds > 11",              "rationale": "Veber +1."},
            {"id": "f2_qed",              "col": "QED",                 "op": "<",  "cutoff": 0.25,  "default_on": True, "active_if": "always",        "label": "QED < 0.25",                 "rationale": "Bickerton 2012 floor."},
            {"id": "f2_heavy",            "col": "HeavyAtoms",          "op": ">",  "cutoff": 50,    "default_on": True, "active_if": "always",        "label": "HeavyAtoms > 50",            "rationale": "Clinical median ~40."},
            {"id": "f2_brenk",            "col": "Brenk_alerts",        "op": ">=", "cutoff": 3,     "default_on": True, "active_if": "col_populated", "label": "Brenk_alerts ≥ 3",           "rationale": "Warhead = 1; ≥3 = +2 extras."},
            {"id": "f2_lle",              "col": "LLE",                 "op": "<",  "cutoff": 1.0,   "default_on": True, "active_if": "col_populated", "label": "LLE < 1.0",                  "rationale": "Leeson NRDD 2007 floor."},
            {"id": "f2_le",               "col": "LE",                  "op": "<",  "cutoff": 0.20,  "default_on": True, "active_if": "col_populated", "label": "LE < 0.20",                  "rationale": "Hopkins-Groom-Alex 2004."},
            {"id": "f2_tc_mol1",          "col": "Tc_to_Mol1",          "op": ">",  "cutoff": 0.85,  "default_on": True, "active_if": "always",        "label": "Tc_to_Mol1 > 0.85",          "rationale": "Novelty / IP flag."},
            {"id": "f2_max_pubtc",        "col": "max_pubTc",           "op": ">",  "cutoff": 0.85,  "default_on": True, "active_if": "col_populated", "label": "max_pubTc > 0.85",           "rationale": "Freedom-to-operate vs 300-lead panel."},
            {"id": "f2_warhead_dev",      "col": "warhead_dev_deg",     "op": ">",  "cutoff": 60,    "default_on": True, "active_if": "col_populated", "label": "warhead_dev_deg > 60°",      "rationale": "Warhead must aim at Cys346."},
            {"id": "f2_n_hbonds",         "col": "n_h_bonds",           "op": "==", "cutoff": 0,     "default_on": True, "active_if": "col_populated", "label": "n_h_bonds == 0",             "rationale": "Hinge engagement required."},
            {"id": "f2_mpae",             "col": "mPAE",                "op": ">",  "cutoff": 10.0,  "default_on": True, "active_if": "col_populated", "label": "mPAE > 10 Å",                "rationale": "AF-multimer FP cutoff (Evans 2022)."},
            {"id": "f2_logk2_hi",         "col": "pred_log_k2_GSH",     "op": ">",  "cutoff": 0.0,   "default_on": True, "active_if": "col_populated", "label": "pred_log_k2_GSH > 0",        "rationale": "Promiscuous off-target rate."},
            {"id": "f2_logk2_lo",         "col": "pred_log_k2_GSH",     "op": "<",  "cutoff": -3.0,  "default_on": True, "active_if": "col_populated", "label": "pred_log_k2_GSH < −3",       "rationale": "Too sluggish for on-target."},
        ],
    },
    "layer3": {
        "label": "Filter 3: Compute gate (placeholder)",
        "description": "Applies only to rows with Vina + xTB + PROPKA populated.",
        "filters": [
            {"id": "f3_vina",             "col": "vina_kcalmol",        "op": ">",  "cutoff": 0.0,   "default_on": True, "active_if": "col_populated", "label": "vina_kcalmol > 0",           "rationale": "Positive Vina = clash."},
            {"id": "f3_vina_intra",       "col": "vina_intra_kcalmol",  "op": ">",  "cutoff": 8.0,   "default_on": True, "active_if": "col_populated", "label": "vina_intra > 8",             "rationale": "Ligand strain proxy."},
            {"id": "f3_strain_pose",      "col": "rdkit_strain_kcal_mol","op": ">", "cutoff": 20.0,  "default_on": True, "active_if": "col_populated", "label": "rdkit strain > 20",          "rationale": "Pose-aware ligand strain."},
            {"id": "f3_strain_posefree",  "col": "rdkit_strain_posefree_kcal_mol","op": ">","cutoff":120.0,"default_on":True,"active_if":"col_populated","label":"posefree strain > 120",      "rationale": "Distribution p90 ≈102."},
            {"id": "f3_pka",              "col": "pKa_Cys346",          "op": ">",  "cutoff": 10.5,  "default_on": True, "active_if": "col_populated", "label": "pKa_Cys346 > 10.5",          "rationale": "<0.1% thiolate at pH 7.4 (Olsson 2011)."},
        ],
    },
    "layer4": {
        "label": "Filter 4: Pose gate (cofolded only)",
        "description": "Applies only to rows with Boltz cofold metrics populated.",
        "filters": [
            {"id": "f4_ligand_iptm",      "col": "ligand_iptm",         "op": "<",  "cutoff": 0.55,  "default_on": True, "active_if": "col_populated", "label": "ligand_iptm < 0.55",         "rationale": "Boltz ligand-specific iptm floor."},
            {"id": "f4_iptm",             "col": "iptm",                "op": "<",  "cutoff": 0.60,  "default_on": True, "active_if": "col_populated", "label": "iptm < 0.60",                "rationale": "AF-Multimer FP threshold."},
        ],
    },
    "layer5": {
        "label": "Filter 5: Energy gate (placeholder)",
        "description": "Applies only to rows with MM-GBSA dG_recognition populated.",
        "filters": [
            {"id": "f5_mmgbsa",           "col": "dG_recognition_md_kcalmol","op": ">","cutoff":-25.0,"default_on": True, "active_if": "col_populated", "label": "dG_recognition > −25",       "rationale": "Cheng JCTC 2017 calibration band."},
        ],
    },
    "observe_only": [
        "fsp3", "Rings", "closest_lead", "shape_Tc_seed", "esp_sim_seed",
        "max_Tc_train", "mean_top10_Tc_train", "pocket_occupancy_pct",
        "LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb", "fukui_plus_Cb",
        "BEI", "SEI", "SILE", "vdw_interaction_kcal_mol",
        "vina_inter_kcalmol", "anchor_wins_ge7",
        # ── Cohort/method diversity (observe-only, NO filter cut) ───────────
        # Surface on the dashboard per method/cohort to flag mode collapse.
        # Pre-computed in cohort_diversity_metrics.csv for the cohort_comparison
        # cohorts; computed lazily per `method` on the live DF via
        # /api/methods?with_diversity=1. NEVER used as a hard filter cut.
        "diversity_ratio", "mean_intra_nn_tanimoto", "max_intra_tanimoto",
        "n_bemis_murcko_scaffolds", "mean_pairwise_dissim",
    ],
    "simple_sort": [
        {"col": "pIC50_mean",        "dir": "desc"},
        {"col": "ligand_iptm",       "dir": "desc"},
        {"col": "LE",                "dir": "desc"},
        {"col": "d_SG",              "dir": "asc"},
        {"col": "high_score_count",  "dir": "desc"},
    ],
    "composite": {
        "description": "PCA-decorrelated, winsorised, block-weighted z. Apply Layer 1 first.",
        "blocks": [
            {"name": "Potency",              "weight": 0.30, "features": [{"col": "pIC50_mean", "sign": "+"}, {"col": "pIC50_std", "sign": "-", "factor": 0.5}]},
            {"name": "Pose trust",           "weight": 0.20, "features": [{"col": "ligand_iptm", "sign": "+"}, {"col": "mPAE", "sign": "-"}, {"col": "iptm", "sign": "+"}]},
            {"name": "Covalent feasibility", "weight": 0.20, "features": [{"col": "d_SG", "sign": "-"}, {"col": "pKa_Cys346", "sign": "-"}, {"col": "warhead_dev_deg", "sign": "-"}]},
            {"name": "Binding energy",       "weight": 0.10, "features": [{"col": "vina_kcalmol", "sign": "-"}, {"col": "max_pubTc", "sign": "+"}]},
            {"name": "Drug-likeness",        "weight": 0.10, "features": [{"col": "LLE", "sign": "+"}, {"col": "LE", "sign": "+"}, {"col": "QED", "sign": "+"}, {"col": "rdkit_strain_kcal_mol", "sign": "-"}]},
            {"name": "Warhead reactivity",   "weight": 0.10, "features": [{"col": "pred_log_k2_GSH", "sign": "parabolic_around_0"}]},
        ],
        "validation": {"holdout": "19 measured ZAP70 mols", "metric": "Spearman ρ", "accept": 0.45, "refit_threshold": 0.30},
    },
}


def _all_filters():
    out = []
    for layer in ("layer1", "layer2", "layer3", "layer4", "layer5"):
        for f in FILTER_SPEC[layer]["filters"]:
            out.append((layer, f))
    return out


def _eval_filter_fail(series: pd.Series, op: str, cutoff):
    """Return boolean mask of FAILING rows for filter (series op cutoff is True)."""
    if op == "<":
        return series < float(cutoff)
    if op == "<=":
        return series <= float(cutoff)
    if op == ">":
        return series > float(cutoff)
    if op == ">=":
        return series >= float(cutoff)
    if op == "==":
        # cutoff may be False/True or numeric
        if isinstance(cutoff, bool):
            return series.astype("boolean") == cutoff
        return series == cutoff
    if op == "!=":
        return series != cutoff
    return pd.Series(False, index=series.index)


def apply_filter_cascade(df: pd.DataFrame, settings: dict):
    """Walk Layers 1→5 and return cascade counts + final mask + per-row failures.

    settings: {filter_id: {"on": bool, "cutoff": float|bool}} overrides.
              Defaults come from FILTER_SPEC.
    """
    counts = {"total": len(df)}
    failed_per_row = pd.Series([[] for _ in range(len(df))], index=df.index)
    cumulative_mask = pd.Series(True, index=df.index)

    for layer in ("layer1", "layer2", "layer3", "layer4", "layer5"):
        layer_mask = pd.Series(True, index=df.index)
        for f in FILTER_SPEC[layer]["filters"]:
            fid = f["id"]
            cfg = settings.get(fid, {})
            on = cfg.get("on", f["default_on"])
            if not on:
                continue
            cutoff = cfg.get("cutoff", f["cutoff"])
            col = f["col"]
            if col not in df.columns:
                continue
            series = df[col]
            populated = series.notna()
            fail_mask = _eval_filter_fail(series, f["op"], cutoff)
            fail_mask = fail_mask.fillna(False)
            # NaN handling depends on active_if:
            #   "always"        → required column; rows with NaN FAIL (predictable cutoff semantics)
            #   "col_populated" → optional column; rows with NaN are skipped (legacy behaviour)
            active_if = f.get("active_if", "always")
            if active_if == "always":
                fail_mask = fail_mask | (~populated)
            else:
                fail_mask = fail_mask & populated
            layer_mask = layer_mask & ~fail_mask
            # Record per-row failure label
            for idx in df.index[fail_mask]:
                failed_per_row.at[idx].append(f["label"])
        cumulative_mask = cumulative_mask & layer_mask
        counts[f"after_{layer}"] = int(cumulative_mask.sum())

    counts["visible"] = int(cumulative_mask.sum())
    return cumulative_mask, failed_per_row, counts


# ── Composite ranking ───────────────────────────────────────────────────────

def _winsorise(s: pd.Series, lo=0.01, hi=0.99) -> pd.Series:
    if s.notna().sum() < 10:
        return s
    qlo = s.quantile(lo)
    qhi = s.quantile(hi)
    return s.clip(qlo, qhi)


def _robust_z(s: pd.Series) -> pd.Series:
    med = s.median()
    mad = (s - med).abs().median()
    if mad is None or mad == 0 or pd.isna(mad):
        std = s.std()
        if std and std > 0:
            return (s - med) / std
        return s * 0
    return (s - med) / (1.4826 * mad)


def compute_composite_score(df: pd.DataFrame) -> pd.Series:
    """Block-weighted z. Missing block features collapse to mean of present.
    Missing block entirely => block weight re-normalised over present blocks."""
    z_cache = {}
    for block in FILTER_SPEC["composite"]["blocks"]:
        for feat in block["features"]:
            col = feat["col"]
            if col in df.columns and col not in z_cache:
                z_cache[col] = _robust_z(_winsorise(df[col]))

    block_vals = []
    block_weights = []
    for block in FILTER_SPEC["composite"]["blocks"]:
        comps = []
        for feat in block["features"]:
            col = feat["col"]
            if col not in z_cache:
                continue
            z = z_cache[col]
            sign = feat.get("sign", "+")
            factor = feat.get("factor", 1.0)
            if sign == "+":
                comps.append(z * factor)
            elif sign == "-":
                comps.append(-z * factor)
            elif sign == "parabolic_around_0":
                comps.append(-(z ** 2) * factor)
        if not comps:
            continue
        block_z = pd.concat(comps, axis=1).mean(axis=1, skipna=True)
        block_vals.append(block_z)
        block_weights.append(block["weight"])

    if not block_vals:
        return pd.Series([float("nan")] * len(df), index=df.index)
    w_sum = sum(block_weights)
    # Renormalise weights over present blocks
    scaled = sum(v * (w / w_sum) for v, w in zip(block_vals, block_weights))
    return scaled


def compute_high_score_count(df: pd.DataFrame) -> pd.Series:
    """Number of green cells across the 9 anchored axes per the spec doc."""
    rules = [
        ("pIC50_mean",         ">=", 7.0),
        ("ligand_iptm",        ">=", 0.85),
        ("d_SG",               "<=", 3.5),
        ("LE",                 ">=", 0.30),
        ("LLE",                ">=", 4.0),
        ("vina_kcalmol",       "<=", -6.0),
        ("pKa_Cys346",         "<=", 9.0),
        ("Brenk_alerts",       "<=", 1),
    ]
    score = pd.Series(0, index=df.index, dtype=int)
    for col, op, cut in rules:
        if col not in df.columns:
            continue
        s = df[col]
        if op == ">=":
            score = score + (s >= cut).fillna(False).astype(int)
        elif op == "<=":
            score = score + (s <= cut).fillna(False).astype(int)
    # pred_log_k2_GSH sweet spot [-2, 0]
    if "pred_log_k2_GSH" in df.columns:
        s = df["pred_log_k2_GSH"]
        score = score + ((s >= -2.0) & (s <= 0.0)).fillna(False).astype(int)
    return score


# ── SearchBuilder condition parser ───────────────────────────────────────────

def apply_searchbuilder(df: pd.DataFrame, sb_payload: dict) -> pd.DataFrame:
    """Apply SearchBuilder JSON conditions to a pandas DataFrame.

    Payload structure (from DataTables SearchBuilder):
      {
        "criteria": [
          {"data": "MW", "condition": "<", "value": ["400"]},
          {"data": "QED", "condition": ">", "value": ["0.5"]},
          ...
        ],
        "logic": "AND"
      }
    """
    if not sb_payload:
        return df
    criteria = sb_payload.get("criteria", [])
    logic = sb_payload.get("logic", "AND").upper()
    if not criteria:
        return df

    # Boolean-aware comparison helper. Bug history (2026-06-01):
    # SearchBuilder sends string "true"/"false" for bool columns like
    # warhead_intact. The old "float(vals[0])" path raised ValueError and fell
    # back to `df[col].astype(str) == "true"`, but pandas stringifies bool as
    # "True"/"False" (capitalized), so the comparison matched 0 rows. This made
    # the LibInvent table appear empty whenever a user enabled a
    # warhead_intact=True chip/filter, even though 11,363/11,367 rows are True.
    _BOOL_TRUE = {"true", "1", "yes", "t", "y"}
    _BOOL_FALSE = {"false", "0", "no", "f", "n"}

    def _coerce_value(col_series, raw):
        """Coerce a SearchBuilder string value to match the column dtype.
        Returns (coerced_value, is_bool_compare). is_bool_compare signals that
        the caller should use a pandas-bool comparison (no astype(str))."""
        s = str(raw).strip()
        sl = s.lower()
        if col_series.dtype == bool or str(col_series.dtype) == "boolean":
            if sl in _BOOL_TRUE: return True, True
            if sl in _BOOL_FALSE: return False, True
        # Try numeric
        try:
            return float(s), False
        except (ValueError, TypeError):
            # Accept literal "true"/"false" even for object cols (mixed bools)
            if sl in _BOOL_TRUE: return True, True
            if sl in _BOOL_FALSE: return False, True
            return s, False

    masks = []
    for c in criteria:
        col = c.get("data") or c.get("origData")
        cond = c.get("condition")
        vals = c.get("value", [])
        if col not in df.columns:
            continue
        try:
            if cond in ("<", "<="):
                mask = df[col] <= float(vals[0]) if cond == "<=" else df[col] < float(vals[0])
            elif cond in (">", ">="):
                mask = df[col] >= float(vals[0]) if cond == ">=" else df[col] > float(vals[0])
            elif cond in ("=", "==", "equals"):
                coerced, is_bool = _coerce_value(df[col], vals[0])
                if is_bool:
                    mask = df[col] == coerced
                elif isinstance(coerced, float):
                    mask = df[col] == coerced
                else:
                    mask = df[col].astype(str) == coerced
            elif cond in ("!=", "≠", "not"):
                coerced, is_bool = _coerce_value(df[col], vals[0])
                if is_bool:
                    mask = df[col] != coerced
                elif isinstance(coerced, float):
                    mask = df[col] != coerced
                else:
                    mask = df[col].astype(str) != coerced
            elif cond == "between":
                lo, hi = float(vals[0]), float(vals[1])
                mask = (df[col] >= lo) & (df[col] <= hi)
            elif cond in ("starts", "starts with"):
                mask = df[col].astype(str).str.startswith(str(vals[0]))
            elif cond in ("ends", "ends with"):
                mask = df[col].astype(str).str.endswith(str(vals[0]))
            elif cond in ("contains",):
                mask = df[col].astype(str).str.contains(str(vals[0]), na=False, regex=False)
            elif cond in ("null", "isnull"):
                mask = df[col].isna()
            elif cond in ("notnull",):
                mask = df[col].notna()
            else:
                continue
        except Exception as e:
            print(f"  SB error on {col}/{cond}/{vals}: {e}")
            continue
        masks.append(mask)

    if not masks:
        return df
    if logic == "AND":
        combined = masks[0]
        for m in masks[1:]:
            combined = combined & m
    else:
        combined = masks[0]
        for m in masks[1:]:
            combined = combined | m
    return df[combined]


# ── Quick filter chips ──────────────────────────────────────────────────────

CHIP_PRESETS = {
    "lipinski": lambda d: d[(d["MW"] < 500) & (d["LogP"] < 5) & (d["HBD"] <= 5) & (d["HBA"] <= 10)],
    "leadlike": lambda d: d[(d["MW"] < 350) & (d["SAScore"] < 4) & (d["QED"] > 0.5)],
    "potent": lambda d: d[d["pIC50_film"] >= 7.0],
    "ultra_potent": lambda d: d[d["pIC50_film"] >= 8.0],
    "in_distribution": lambda d: d[d["max_Tc_train"] >= 0.3],
    "warhead_intact": lambda d: d[d["warhead_intact"] == True],
    "pains_clean": lambda d: d[d["PAINS_alerts"] == 0],
    "synthesizable": lambda d: d[d["SAScore"] < 4.0],
    "drug_like": lambda d: d[d["QED"] >= 0.5],
}


@app.route("/api/data", methods=["POST", "GET"])
def api_data():
    """DataTables server-side endpoint."""
    if request.method == "POST":
        payload = request.get_json() or request.form.to_dict() or {}
    else:
        payload = request.args.to_dict()
    # DEBUG: log non-trivial payload keys to find SearchBuilder integration bug
    _interesting = {k: v for k, v in payload.items() if k not in ("draw", "start", "length")}
    if _interesting:
        print(f"[REQ] keys={list(payload.keys())} interesting={_interesting}", flush=True)
    # Standard DataTables params
    draw = int(payload.get("draw", 1))
    start = int(payload.get("start", 0))
    length = int(payload.get("length", 20))
    if length < 0: length = len(DF)

    df = DF
    # Method filter (selects one method)
    method_filter = payload.get("method", "")
    if method_filter and method_filter != "_all" and "method" in df.columns:
        df = df[df["method"] == method_filter]
    n_total = len(df)

    # Apply quick chips
    chips_str = payload.get("chips", "")
    if chips_str:
        chips = [c.strip() for c in chips_str.split(",") if c.strip()]
        for chip in chips:
            if chip in CHIP_PRESETS:
                df = CHIP_PRESETS[chip](df)

    # Apply SearchBuilder
    sb_str = payload.get("searchBuilder", "") or payload.get("sb", "")
    if sb_str:
        try:
            sb = json.loads(sb_str) if isinstance(sb_str, str) else sb_str
            print(f"[SB] method={method_filter!r} payload={sb}", flush=True)
            before = len(df)
            df = apply_searchbuilder(df, sb)
            print(f"[SB] {before} -> {len(df)}", flush=True)
        except Exception as e:
            print(f"SB parse error: {e}")

    # Free-text search across SMILES
    text = payload.get("search", "") or payload.get("q", "")
    if text:
        df = df[df["smiles"].astype(str).str.contains(text, case=False, na=False)]

    n_filtered = len(df)

    # Sorting
    order_col = payload.get("order_col", "pIC50_film")
    order_dir = payload.get("order_dir", "desc")
    if order_col in df.columns:
        df = df.sort_values(order_col, ascending=(order_dir == "asc"), na_position="last")

    # Pagination
    df_page = df.iloc[start:start + length]

    # CRITICAL: replace NaN with None — Python's json.dumps emits literal `NaN`
    # tokens which are invalid JSON; browsers' JSON.parse rejects them and
    # jQuery routes the response to the error handler, causing the table to
    # render "No data available" even on a 200 OK with valid filtered data.
    df_page = df_page.astype(object).where(df_page.notna(), None)
    rows = df_page.to_dict(orient="records")
    return jsonify({
        "draw": draw,
        "recordsTotal": n_total,
        "recordsFiltered": n_filtered,
        "data": rows,
    })


# ── Boltz / AlphaFold cofolded poses (Cys346 cohort) ──────────────────────
# Old Cys560-targeted poses are under _DEPRECATED_cys560_* and explicitly
# NOT loaded here. Cys346 manifests will populate when the redo completes.
MEDCHEM10_ROOT = PROJECT_ROOT / "data" / "boltz_poses" / ("medchem_top10__zap70_cys346")
TOP1000_ROOT   = PROJECT_ROOT / "data" / "boltz_poses" / ("boltz_results_top1000__zap70_cys346") / "predictions"

POSES_MANIFEST = {}
_man = MEDCHEM10_ROOT / "manifest.json"
if _man.exists():
    POSES_MANIFEST.update(json.loads(_man.read_text()))
    print(f"Loaded {len(POSES_MANIFEST)} medchem-top10 poses (Cys346)")
else:
    print(f"No Cys346 medchem-top10 manifest yet")

TOP1000_MANIFEST = {}
if TOP1000_MAN.exists():
    TOP1000_MANIFEST = json.loads(TOP1000_MAN.read_text())
    print(f"Loaded {len(TOP1000_MANIFEST)} top-1000 Cys346 cofold poses")


def _resolve_pose(row_id: int):
    """Return (cif_text, metadata) for the given row_id, or (None, None)."""
    key = str(row_id)
    # Prefer top-1000 (newer + has full Boltz metrics)
    if key in TOP1000_MANIFEST:
        m = TOP1000_MANIFEST[key]
        cif = TOP1000_ROOT / m["yaml_name"] / f"{m['yaml_name']}_model_0.cif"
        if cif.exists():
            meta = dict(m)
            meta["pose_name"] = m["yaml_name"]
            meta["pose_source"] = "top1000_boltz"
            return cif.read_text(), meta
    # Fallback: medchem10 batch
    if key in POSES_MANIFEST:
        m = POSES_MANIFEST[key]
        cif = MEDCHEM10_ROOT / "predictions" / m["pose_name"] / f"{m['pose_name']}_model_0.cif"
        if cif.exists():
            meta = dict(m)
            meta["pose_source"] = "medchem10_boltz"
            return cif.read_text(), meta
    return None, None


@app.route("/api/pose/<int:row_id>")
def api_pose(row_id: int):
    """Return the cofolded protein-ligand mmCIF + metadata for row_id."""
    cif, meta = _resolve_pose(row_id)
    if cif is None:
        return jsonify({"available": False, "row_id": row_id})
    return jsonify({"available": True, "row_id": row_id, "cif": cif, "metadata": meta})


@app.route("/api/poses_index")
def api_poses_index():
    """Return all row_ids that have cofolded poses available."""
    keys = set(POSES_MANIFEST.keys()) | set(TOP1000_MANIFEST.keys())
    return jsonify({"row_ids": sorted(int(k) for k in keys)})


@app.route("/api/top_combined")
def api_top_combined():
    """Top-N candidates ranked by combined_score (FiLMDelta pIC50 + Boltz confidence).

    2026-05-26: Each manifest row is enriched with ALL DF columns (pubTc panel,
    vina rescore, RDKit strain, propka, xTB, efficiency, …) so the Top-20
    leaderboard can render the SAME 33-column 7-family layout as the per-method
    tables. Manifest values win on overlap (manifest has the Boltz cofold
    metrics keyed by yaml_name; DF row_id is the JOIN key).
    """
    n = int(request.args.get("n", 20))
    rows = list(TOP1000_MANIFEST.values())
    # combined_score can be null in some manifest entries; coerce to -1e9 so they sort last
    rows.sort(key=lambda r: (r.get("combined_score") if r.get("combined_score") is not None else -1e9), reverse=True)
    top = rows[:n]
    enriched = []
    for r in top:
        rid = r.get("row_id")
        merged = dict(r)
        try:
            rid_int = int(rid) if rid is not None else None
        except (TypeError, ValueError):
            rid_int = None
        if rid_int is not None:
            df_row = DF[DF["row_id"] == rid_int]
            if not df_row.empty:
                df_dict = df_row.iloc[0].replace({np.nan: None}).to_dict()
                # DF columns fill in everything; manifest then overrides for
                # the cofold-keyed fields (mPAE, iptm, …, combined_score).
                merged = {**df_dict, **r}
        enriched.append(merged)
    return jsonify({
        "n_total": len(rows),
        "rows": enriched,
    })


# ── Seed Mol 1 cofold (Cys346) — pending redo ──────────────────────────
# Old Cys560-targeted seed poses are deprecated. New Cys346 versions will
# populate at data/boltz_poses/mol1__zap70_cys346/ when ready.
SEED_POSE_DIR = PROJECT_ROOT / "data" / "boltz_poses" / "mol1__zap70_cys346"
SEED_POSE_META = None
_seed_meta_path = SEED_POSE_DIR / "mol1_manifest.json"
if _seed_meta_path.exists():
    SEED_POSE_META = json.loads(_seed_meta_path.read_text()).get("mol1")
    print(f"Loaded seed Mol 1 cofold pose (Cys346)")
else:
    print(f"No Cys346 seed Mol 1 cofold yet")


@app.route("/api/pose_seed")
def api_pose_seed():
    """Return the cofolded Mol 1 (seed) protein-ligand mmCIF + metadata.

    Query param `variant`:
      - 'v1' (default): no pocket constraint — covalent bond to Cys560 forms,
        molecule sits at activation loop, NO hinge H-bonds
      - 'v2': pocket constraint at hinge (Met414/Glu415/Met416) — molecule
        binds canonically at ATP pocket with hinge H-bonds, but warhead can't
        reach Cys560 so covalent bond is NOT formed
    """
    variant = request.args.get("variant", "v1")
    if variant == "v2":
        v2_dir = PROJECT_ROOT / "data" / "boltz_poses" / "mol1_v2" / "mol1_v2_pocket"
        cif_path = v2_dir / "mol1_v2_pocket_model_0.cif"
        conf_path = v2_dir / "confidence_mol1_v2_pocket_model_0.json"
        if not cif_path.exists():
            return jsonify({"available": False})
        meta = {
            "pose_name": "mol1_v2_pocket",
            "smiles": "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1",
            "method": "Seed (Mol 1) — pocket-constrained",
            "variant": "v2",
            **(json.loads(conf_path.read_text()) if conf_path.exists() else {}),
        }
        return jsonify({
            "available": True,
            "cif": cif_path.read_text(),
            "metadata": meta,
        })

    # default: v1
    cif_path = SEED_POSE_DIR / "mol1" / "mol1_model_0.cif"
    if not cif_path.exists() or SEED_POSE_META is None:
        return jsonify({"available": False})
    meta = dict(SEED_POSE_META)
    meta["variant"] = "v1"
    return jsonify({
        "available": True,
        "cif": cif_path.read_text(),
        "metadata": meta,
    })


MOL1_SMILES = "C=CC(=O)N1Cc2cccc(C(=O)Nc3cn(C(C)C)cn3)c2C1"


@app.route("/api/svg/<row_id>")
def api_svg(row_id: str):
    """Render a small SVG for one row's molecule.

    Special-case `row_id == "M1"` → render from MOL1_SMILES so the Mol1 anchor
    row in the dashboard can use the same SVG endpoint as the leaderboard.
    Otherwise expects an integer row_id indexed into the bulk-scored CSV.
    """
    if row_id == "M1":
        smi = MOL1_SMILES
    else:
        try:
            rid = int(row_id)
        except ValueError:
            return "", 404
        if rid < 0 or rid >= len(DF):
            return "", 404
        smi = DF.iloc[rid]["smiles"]
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return "", 404
    AllChem.Compute2DCoords(mol)
    w = int(request.args.get("w", 140))
    h = int(request.args.get("h", 100))
    drawer = Draw.MolDraw2DSVG(w, h)
    drawer.drawOptions().bondLineWidth = 1.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    return svg, 200, {"Content-Type": "image/svg+xml"}

@app.route("/api/seed_svg")
def api_seed_svg():
    w = int(request.args.get("w", 200))
    h = int(request.args.get("h", 140))
    mol = Chem.MolFromSmiles(MOL1_SMILES)
    AllChem.Compute2DCoords(mol)
    drawer = Draw.MolDraw2DSVG(w, h)
    drawer.drawOptions().bondLineWidth = 1.0
    drawer.DrawMolecule(mol)
    drawer.FinishDrawing()
    svg = drawer.GetDrawingText().replace("<?xml version='1.0' encoding='iso-8859-1'?>", "")
    return svg, 200, {"Content-Type": "image/svg+xml"}


# ── Mol1 (seed / design anchor) — one-row reference for the dashboard ─────
# Returns ONE row with the SAME column keys as the Top-20 leaderboard /
# bulk CSV manifest entries, plus the enriched columns (pKa_Cys346,
# vina_kcalmol, LLE, pubTc panel, rdkit_strain, xTB descriptors). Used by
# the new "Mol1 — Design Anchor" reference table above the leaderboard.
# Mol1 is NOT in all_methods_bulk_scored_v4.csv (it's the seed) so all
# fields are computed/loaded fresh here.
MOL1_ANCHOR_CACHE: dict | None = None

def _build_mol1_anchor() -> dict:
    """Compute Mol1 anchor row. Cached at first call."""
    global MOL1_ANCHOR_CACHE
    if MOL1_ANCHOR_CACHE is not None:
        return MOL1_ANCHOR_CACHE

    from rdkit.Chem import Descriptors, Crippen, Lipinski, QED, rdMolDescriptors
    from rdkit.Chem import DataStructs, FilterCatalog, RDConfig

    smi = MOL1_SMILES
    mol = Chem.MolFromSmiles(smi)
    row: dict = {
        "row_id": "M1",
        "smiles": smi,
        "method": "Seed (Mol 1) — Design Anchor",
        "yaml_name": "mol1",
    }

    # ── RDKit physchem ─────────────────────────────────────────────────
    MW = float(Descriptors.MolWt(mol))
    LogP = float(Crippen.MolLogP(mol))
    TPSA = float(Descriptors.TPSA(mol))
    HBA = int(Lipinski.NumHAcceptors(mol))
    HBD = int(Lipinski.NumHDonors(mol))
    RotBonds = int(Lipinski.NumRotatableBonds(mol))
    HeavyAtoms = int(mol.GetNumHeavyAtoms())
    Rings = int(rdMolDescriptors.CalcNumRings(mol))
    fsp3 = float(rdMolDescriptors.CalcFractionCSP3(mol))
    qed_val = float(QED.qed(mol))
    row.update({
        "MW": MW, "LogP": LogP, "TPSA": TPSA, "HBA": HBA, "HBD": HBD,
        "RotBonds": RotBonds, "HeavyAtoms": HeavyAtoms, "Rings": Rings,
        "fsp3": fsp3, "QED": qed_val,
    })

    # Lipinski violations (matches compute_efficiency_metrics.py)
    viol = (int(MW > 500) + int(LogP > 5) + int(HBA > 10) + int(HBD > 5))
    row["Lipinski_violations"] = viol

    # PAINS
    try:
        params = FilterCatalog.FilterCatalogParams()
        params.AddCatalog(FilterCatalog.FilterCatalogParams.FilterCatalogs.PAINS)
        cat = FilterCatalog.FilterCatalog(params)
        row["PAINS_alerts"] = int(len(cat.GetMatches(mol)))
    except Exception as e:
        print(f"  PAINS failed: {e}")
        row["PAINS_alerts"] = None

    # SAScore (Ertl-Schuffenhauer; ships with RDKit Contrib)
    try:
        sa_path = Path(RDConfig.RDContribDir) / "SA_Score"
        if str(sa_path) not in sys.path:
            sys.path.append(str(sa_path))
        import sascorer  # type: ignore
        row["SAScore"] = float(sascorer.calculateScore(mol))
    except Exception as e:
        print(f"  SAScore failed: {e}")
        row["SAScore"] = None

    # Warhead intact (acrylamide) — by definition true for Mol1
    row["warhead_intact"] = True
    row["Tc_to_Mol1"] = 1.0

    # max Tc to training set (Mol1 vs 280 ZAP70 training mols)
    try:
        from experiments.run_zap70_v3 import load_zap70_molecules
        train_df, _ = load_zap70_molecules()
        mol1_fp = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        train_fps = []
        for s in train_df["smiles"].tolist():
            m = Chem.MolFromSmiles(s)
            if m is None:
                continue
            train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
        sims = np.array(DataStructs.BulkTanimotoSimilarity(mol1_fp, train_fps))
        row["max_Tc_train"] = float(sims.max())
        top10 = np.partition(sims, -10)[-10:]
        row["mean_top10_Tc_train"] = float(top10.mean())
    except Exception as e:
        print(f"  Tc-train failed: {e}")
        row["max_Tc_train"] = None
        row["mean_top10_Tc_train"] = None

    # ── pubTc v3 panel (300-lead) ─────────────────────────────────────
    try:
        sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
        from pubtc_panel_v3 import PANEL_SMILES  # type: ignore
        panel_fps, panel_names = [], []
        for name, ps in PANEL_SMILES.items():
            pm = Chem.MolFromSmiles(ps)
            if pm is None:
                continue
            panel_fps.append(AllChem.GetMorganFingerprintAsBitVect(pm, 2, nBits=2048))
            panel_names.append(name)
        mol1_fp_v3 = AllChem.GetMorganFingerprintAsBitVect(mol, 2, nBits=2048)
        sims = np.array(DataStructs.BulkTanimotoSimilarity(mol1_fp_v3, panel_fps))
        row["max_pubTc"] = float(sims.max())
        row["median_pubTc"] = float(np.median(sims))
        row["mean_pubTc"] = float(sims.mean())
        k = min(10, len(sims))
        top10 = np.partition(sims, -k)[-k:]
        row["top10_mean_pubTc"] = float(top10.mean())
        row["closest_lead"] = panel_names[int(np.argmax(sims))]
    except Exception as e:
        print(f"  pubTc v3 panel failed: {e}")
        for k in ("max_pubTc", "median_pubTc", "mean_pubTc", "top10_mean_pubTc"):
            row[k] = None
        row["closest_lead"] = None

    # ── Boltz cofold metrics (from mol1_manifest.json) ───────────────
    seed_meta_path = PROJECT_ROOT / "data" / "boltz_poses" / "mol1__zap70_cys346" / "mol1_manifest.json"
    if seed_meta_path.exists():
        sm = json.loads(seed_meta_path.read_text()).get("mol1", {})
        # Map manifest field names → leaderboard field names
        row["mPAE"] = None  # not in mol1 manifest; only confidence/iptm/etc
        row["iptm"] = sm.get("boltz_iptm")
        row["ligand_iptm"] = sm.get("boltz_ligand_iptm")
        row["boltz_plddt"] = sm.get("boltz_complex_plddt")
        row["complex_plddt"] = sm.get("boltz_complex_plddt")
        row["boltz_pde"] = sm.get("boltz_complex_pde")
        row["boltz_confidence"] = sm.get("boltz_confidence_score")
        row["combined_score"] = None
        row["rank_score"] = None
        row["warhead_atom_name"] = sm.get("warhead_atom_name")
    else:
        for k in ("mPAE", "iptm", "ligand_iptm", "boltz_plddt", "complex_plddt",
                  "boltz_pde", "boltz_confidence", "combined_score", "rank_score",
                  "warhead_atom_name"):
            row[k] = None

    # ── Enriched per-pose computations (Vina, PROPKA, xTB, RDKit strain, ─
    # cofold geometry) loaded from cache JSON written by
    # experiments/enrich_mol1_anchor.py. If the cache is missing every
    # column falls back to None (the previous behaviour).
    enriched_path = PROJECT_ROOT / "data" / "boltz_poses" / "mol1_anchor_enriched.json"
    if enriched_path.exists():
        try:
            enr = json.loads(enriched_path.read_text())
        except Exception as e:
            print(f"  failed to load mol1_anchor_enriched.json: {e}")
            enr = {}
    else:
        enr = {}

    # Cofold geometry (overrides manifest's None if present)
    for k in ("d_SG", "geom_ok", "n_h_bonds", "n_stabilizing_contacts",
              "pocket_occupancy_pct"):
        row[k] = enr.get(k)
    # If the cache derived a better warhead atom name keep it.
    if enr.get("warhead_atom_name"):
        row["warhead_atom_name"] = enr["warhead_atom_name"]

    # ── pIC50 prediction for Mol1 ─────────────────────────────────────
    # Mol1 is the anchor used to predict Δ on candidates; we don't have a
    # FiLMDelta pIC50 prediction for Mol1 itself stored on disk. It also
    # has no measured ChEMBL pIC50 (it's a designed seed, not in the 280
    # ZAP70 anchor set). Leave as null with a clear note in the
    # dashboard.
    row["pIC50_method"] = None
    row["pIC50_mean"] = None
    row["pIC50_std"] = None
    row["pIC50_film"] = None
    row["delta_vs_mol1"] = 0.0  # by definition (Mol1 vs itself)
    row["direct_delta_from_mol1"] = 0.0  # by definition

    # ── Efficiency metrics (need pIC50 → all NaN for Mol1) ───────────
    # LLE/LE/BEI/SEI/SILE all require a pIC50 anchor value. Mol1's pIC50
    # is unknown (no in-house assay), so these stay null.
    row["LLE"] = None
    row["LE"] = None
    row["BEI"] = None
    row["SEI"] = None
    row["SILE"] = None

    # ── Vina rescore for Mol1 pose (from cache) ──────────────────────
    row["vina_kcalmol"] = enr.get("vina_kcalmol")
    row["vina_inter_kcalmol"] = enr.get("vina_inter_kcalmol")
    row["vina_intra_kcalmol"] = enr.get("vina_intra_kcalmol")

    # ── PROPKA pKa_Cys346 / xTB warhead reactivity / RDKit strain (cache) ──
    for k in ("pKa_Cys346", "LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV",
              "q_Cb", "fukui_plus_Cb", "pred_log_k2_GSH",
              "rdkit_strain_kcal_mol", "vdw_interaction_kcal_mol",
              "rdkit_strain_posefree_kcal_mol",
              "shape_Tc_seed", "esp_sim_seed", "warhead_dev_deg"):
        row[k] = enr.get(k)

    # Anchor-vs-itself wins are n/a (we never compare Mol1 to itself).
    row["anchor_wins"] = None
    row["anchor_wins_ge7"] = None

    # Convert any remaining numpy types → Python builtins for clean JSON
    out: dict = {}
    for k, v in row.items():
        if isinstance(v, (np.floating,)):
            out[k] = float(v) if not np.isnan(v) else None
        elif isinstance(v, (np.integer,)):
            out[k] = int(v)
        elif isinstance(v, (np.bool_,)):
            out[k] = bool(v)
        else:
            out[k] = v

    MOL1_ANCHOR_CACHE = out
    return out


@app.route("/api/mol1_anchor")
def api_mol1_anchor():
    """Mol1 (design anchor) as a one-row reference for the dashboard."""
    try:
        row = _build_mol1_anchor()
        return jsonify({"available": True, "row": row})
    except Exception as e:
        import traceback
        print(f"mol1_anchor error: {e}\n{traceback.format_exc()}")
        return jsonify({"available": False, "error": str(e)}), 500


@app.route("/api/methods")
def api_methods():
    """List methods + their candidate counts + per-method stats.

    Query params:
        with_diversity=1  Include per-method diversity panel (lazy, cached
                          after first call: n_unique, diversity_ratio,
                          mean_intra_nn_tanimoto, max_intra_tanimoto,
                          n_bemis_murcko_scaffolds, mean_pairwise_dissim).
                          Default off so the existing report.html init stays
                          fast.
    """
    if "method" not in DF.columns:
        return jsonify({"methods": [{"name": "All", "n": len(DF)}]})
    with_div = str(request.args.get("with_diversity", "0")).lower() in ("1", "true", "yes", "on")
    div_cache = _compute_method_diversity_cache() if with_div else {}
    rows = []
    for m, sub in DF.groupby("method"):
        pic = sub["pIC50_film"].dropna() if "pIC50_film" in sub.columns else pd.Series(dtype=float)
        row = {
            "name": m,
            "n": int(len(sub)),
            "max_pIC50": float(pic.max()) if len(pic) else None,
            "median_pIC50": float(pic.median()) if len(pic) else None,
            "n_potent_7": int((pic >= 7).sum()),
            "n_potent_8": int((pic >= 8).sum()),
            "n_warhead": int(sub["warhead_intact"].sum()) if "warhead_intact" in sub.columns else None,
        }
        if with_div and m in div_cache:
            d = div_cache[m]
            row.update({
                "n_unique": int(d["n_unique"]),
                "diversity_ratio": d["diversity_ratio"],
                "mean_intra_nn_tanimoto": d["mean_intra_nn_tanimoto"],
                "max_intra_tanimoto": d["max_intra_tanimoto"],
                "n_bemis_murcko_scaffolds": int(d["n_bemis_murcko_scaffolds"]),
                "mean_pairwise_dissim": d["mean_pairwise_dissim"],
            })
        rows.append(row)
    return jsonify({"methods": rows, "total": len(DF)})


@app.route("/api/cohort_diversity")
def api_cohort_diversity():
    """Per-cohort diversity metrics (observe-only, mode-collapse alarm).

    Returns the pre-computed `cohort_diversity_metrics.csv` rows as JSON.
    Columns: cohort, group, n_total, n_unique, diversity_ratio,
             mean_intra_nn_tanimoto, max_intra_tanimoto,
             n_bemis_murcko_scaffolds, mean_pairwise_dissim.
    Source: experiments/cohort_diversity_backfill.py over
    `results/paper_evaluation/cohort_comparison/all_cohorts_metrics.csv`.
    """
    if COHORT_DIVERSITY is None:
        return jsonify({"rows": [], "source": str(COHORT_DIVERSITY_CSV), "available": False}), 200
    rows = COHORT_DIVERSITY.astype(object).where(COHORT_DIVERSITY.notna(), None).to_dict(orient="records")
    return jsonify({
        "rows": rows,
        "source": str(COHORT_DIVERSITY_CSV.relative_to(PROJECT_ROOT)),
        "available": True,
        "n_cohorts": len(rows),
    })


@app.route("/api/spec")
def api_spec():
    """Return the filter+ranking spec JSON. Single source of truth (the docs
    page and the dashboard UI both render from this)."""
    return jsonify(FILTER_SPEC)


def _parse_filter_settings(payload: dict) -> dict:
    """Pull per-filter on/cutoff overrides out of a JSON or query payload.

    Accepts either:
      - JSON: {"filters": {"f1_mw": {"on": true, "cutoff": 700}, ...}}
      - Query: ?f1_mw_on=1&f1_mw_cutoff=700&...
      - Layer-level shortcuts: ?layer1_enabled=1&layer2_enabled=0
    """
    settings: dict = {}
    # JSON body
    if isinstance(payload.get("filters"), dict):
        settings.update(payload["filters"])

    # Layer-level shortcuts
    for layer in ("layer1", "layer2", "layer3", "layer4", "layer5"):
        enabled_key = f"{layer}_enabled"
        if enabled_key in payload:
            v = payload[enabled_key]
            on = str(v).lower() in ("1", "true", "yes", "on")
            for f in FILTER_SPEC[layer]["filters"]:
                settings.setdefault(f["id"], {})
                settings[f["id"]]["on"] = on

    # Per-filter query params: <id>_on=1, <id>_cutoff=700
    for layer in ("layer1", "layer2", "layer3", "layer4", "layer5"):
        for f in FILTER_SPEC[layer]["filters"]:
            fid = f["id"]
            if f"{fid}_on" in payload:
                v = payload[f"{fid}_on"]
                on = str(v).lower() in ("1", "true", "yes", "on")
                settings.setdefault(fid, {})["on"] = on
            if f"{fid}_cutoff" in payload:
                raw = payload[f"{fid}_cutoff"]
                try:
                    settings.setdefault(fid, {})["cutoff"] = float(raw)
                except (TypeError, ValueError):
                    if isinstance(raw, bool):
                        settings.setdefault(fid, {})["cutoff"] = raw
    return settings


@app.route("/api/filter", methods=["GET", "POST"])
def api_filter():
    """Apply F1-F5 cascade + sort + (optional) composite scoring.

    Query/JSON params:
      layer{1..5}_enabled : 0/1 — bulk toggle a layer
      <filter_id>_on      : 0/1 — toggle one filter
      <filter_id>_cutoff  : float — override cutoff
      sort                : comma-separated 'col:dir' pairs (overrides default simple sort)
      composite           : 0/1 — sort by composite score column
      method              : restrict to one method (else _all)
      start, length       : pagination (length=-1 returns everything)
    """
    if request.method == "POST":
        payload = request.get_json(silent=True) or request.form.to_dict() or {}
    else:
        payload = request.args.to_dict()

    df = DF
    method_filter = payload.get("method", "")
    if method_filter and method_filter != "_all" and "method" in df.columns:
        df = df[df["method"] == method_filter]

    settings = _parse_filter_settings(payload)
    mask, failed, counts = apply_filter_cascade(df, settings)

    # Compose / sort
    visible = df[mask].copy()
    visible["_failed_filters"] = failed.loc[visible.index].apply(lambda x: list(x))
    visible["_pass_count"] = 0  # will be computed below
    # Build a parallel "fail count over ALL active filters" per visible row
    # (always 0 for survivors but useful for the ✗N badge on the table-side).
    # For unfiltered rows (mask=False) we attach their failure list separately
    # — but we only return surviving rows here. UI sees the dropped count via cascade.

    # high_score_count is reused both as a sort column and as part of the
    # simple-sort default. Compute on the visible subset to avoid scoring the
    # full 520K when only a few k are visible.
    if "high_score_count" not in visible.columns:
        visible["high_score_count"] = compute_high_score_count(visible)

    composite = str(payload.get("composite", "0")).lower() in ("1", "true", "yes", "on")
    if composite:
        visible["_score"] = compute_composite_score(visible)

    # Sort: comma list "col:dir,col:dir,..."
    sort_str = payload.get("sort", "")
    if composite:
        sort_pairs = [("_score", "desc")]
    elif sort_str:
        sort_pairs = []
        for token in sort_str.split(","):
            if ":" in token:
                col, d = token.split(":", 1)
                sort_pairs.append((col.strip(), d.strip().lower()))
            else:
                sort_pairs.append((token.strip(), "desc"))
    else:
        sort_pairs = [(s["col"], s["dir"]) for s in FILTER_SPEC["simple_sort"]]

    sort_cols = [c for c, _ in sort_pairs if c in visible.columns]
    sort_dirs = [d == "asc" for c, d in sort_pairs if c in visible.columns]
    if sort_cols:
        visible = visible.sort_values(sort_cols, ascending=sort_dirs, na_position="last")

    # Pagination
    start = int(payload.get("start", 0))
    length = int(payload.get("length", 200))
    n_visible = len(visible)
    if length < 0:
        length = n_visible
    page = visible.iloc[start:start + length]

    # JSON-safe NaN → None
    page = page.astype(object).where(page.notna(), None)
    rows = page.to_dict(orient="records")

    # Attach a per-row pass badge: green if no failures in *active* filters.
    # Since visible rows survived the cascade, _failed_filters is [] for them.
    # We also attach a *_failed_filters* for the FAILED rows in the broader df
    # — but only as aggregate counts in the cascade, not row-by-row, to keep
    # the response small.

    return jsonify({
        "counts": counts,
        "n_visible": n_visible,
        "start": start,
        "length": length,
        "rows": rows,
        "sort": [{"col": c, "dir": d} for c, d in sort_pairs],
        "composite": composite,
    })


@app.route("/api/health")
def api_health():
    return jsonify({"ok": True, "n_rows": len(DF), "columns": list(DF.columns),
                    "methods": list(DF["method"].unique()) if "method" in DF.columns else []})


@app.route("/")
def index():
    """Serve the report HTML directly so a single Flask process handles everything."""
    html_path = Path(__file__).parent / "report.html"
    if html_path.exists():
        return send_file(html_path)
    return jsonify({"endpoints": ["/api/data", "/api/svg/<row_id>", "/api/health"]})


@app.route("/tier2")
@app.route("/tier2_scaled")
def tier2_scaled_page():
    """Serve the Tier 2 SCALED page (reduced column layout, pose-free cohort)."""
    html_path = Path(__file__).parent / "tier2_scaled_page.html"
    if html_path.exists():
        return send_file(html_path)
    return jsonify({"error": "tier2_scaled_page.html not found"}), 404


if __name__ == "__main__":
    import os
    host = os.environ.get("BACKEND_HOST", "127.0.0.1")
    port = int(os.environ.get("BACKEND_PORT", "5001"))
    print(f"Listening on {host}:{port}")
    app.run(host=host, port=port, debug=False)
