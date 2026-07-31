#!/usr/bin/env python3
"""Select top-N molecules from the bulk-scored cohort and prep LLM-scoring inputs.

Reproduces the backend.py data pipeline (filters + enrichment merges), applies
the canonical simple-rank (pIC50_mean DESC -> ligand_iptm DESC -> LE DESC ->
d_SG ASC), and writes a single JSON with all 73 columns + the 5 nearest ChEMBL
ZAP70 actives (Tanimoto on Morgan FP) per molecule. Also writes a markdown
manifest summarising the selection.

Usage:
    conda run -n quris python experiments/select_top_N_for_llm_cohort.py \
        --n 10 \
        --output results/paper_evaluation/llm_cohort_input/top10_test.json \
        --manifest results/paper_evaluation/llm_cohort_input/manifest_top10.md \
        [--input results/paper_evaluation/all_methods_bulk_scored_v4.csv] \
        [--label "TEST/DEV"]

Production:
    conda run -n quris python experiments/select_top_N_for_llm_cohort.py \
        --n 100 \
        --output results/paper_evaluation/llm_cohort_input/top100.json \
        --manifest results/paper_evaluation/llm_cohort_input/manifest_top100.md \
        --label "PRODUCTION"
"""
from __future__ import annotations

import argparse
import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from rdkit import Chem, RDLogger  # noqa: E402
from rdkit.Chem import AllChem, DataStructs  # noqa: E402

RDLogger.DisableLog("rdApp.*")


# -----------------------------------------------------------------------------
# Data loading (mirrors experiments/server/backend.py up to NUMERIC_COLS)
# -----------------------------------------------------------------------------

DEFAULT_INPUT = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"

TOP1000_MAN = PROJECT_ROOT / "data" / "boltz_poses" / "top1000_manifest__zap70_cys346.json"
PROPKA_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_propka.csv"
PUBTC_V3_CSV = PROJECT_ROOT / "data" / "paper_evaluation" / "pubtc_scores_v3.csv"
EFFICIENCY_CSV = PROJECT_ROOT / "data" / "paper_evaluation" / "efficiency_metrics.csv"
WARHEAD_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_warhead_reactivity.csv"
VINA_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_vina_rescore.csv"
RDKIT_STRAIN_COFOLD_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_rdkit_strain_cofold.csv"
RDKIT_STRAIN_POSEFREE_CSV = PROJECT_ROOT / "data" / "paper_evaluation" / "rdkit_strain_posefree.csv"
MDMMGBSA_CSV = PROJECT_ROOT / "data" / "boltz_poses" / "zap70_cys346_capped_mmpbsa_top50.csv"
BRENK_CACHE = PROJECT_ROOT / "data" / "paper_evaluation" / "brenk_alerts.csv"

DROPPED_METHODS = [
    "Tier 4 — De Novo unconstrained",
    "Tier 4 — Mol2Mol unconstrained",
    "Method A — De Novo FiLMDelta-driven",
    "Method B — Mol2Mol FiLMDelta-driven",
]

METHOD_RENAME = {
    "Tier 1 — Med-Chem Playbook (rule-based)": "Medchem Rules",
    "Tier 1.5 — Warhead Controls + Med-Chem Tricks": "Medchem Rules",
    "Tier 2 — Fragment Replacement (curated 204)": "Amine Replacements",
    "Tier 2 SCALED — Fragment Replacement (498K)": "Amine Replacements",
    "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)": "Amine Replacements",
}


def load_and_enrich(input_csv: Path) -> pd.DataFrame:
    """Load the bulk CSV and apply the same filters + merges as backend.py."""
    print(f"[load] {input_csv}")
    DF = pd.read_csv(input_csv)
    DF = DF.reset_index(drop=True)
    DF["row_id"] = DF.index

    if "method" in DF.columns:
        DF["method"] = DF["method"].replace(METHOD_RENAME)
        pre = len(DF)
        DF = DF[~DF["method"].isin(DROPPED_METHODS)].reset_index(drop=True)
        print(f"  drop Tier4/MethodA/B: {pre:,} -> {len(DF):,}")

    if "warhead_intact" in DF.columns:
        pre = len(DF)
        DF = DF[DF["warhead_intact"] == True].reset_index(drop=True)
        print(f"  drop warhead-modified: {pre:,} -> {len(DF):,}")

    if "smiles" in DF.columns:
        pre = len(DF)
        DF = DF[~DF["smiles"].astype(str).str.contains(".", regex=False, na=False)].reset_index(drop=True)
        print(f"  drop disconnected SMILES: {pre:,} -> {len(DF):,}")

    DF["row_id"] = DF.index

    if "pIC50_method" in DF.columns and "pIC50_film" not in DF.columns:
        DF["pIC50_film"] = DF["pIC50_method"]

    # ── Boltz cofold metrics ────────────────────────────────────────────
    if TOP1000_MAN.exists():
        _b = json.loads(TOP1000_MAN.read_text())
        boltz_rows = []
        for rid_str, m in _b.items():
            boltz_rows.append({
                "row_id": int(rid_str),
                "yaml_name": m.get("yaml_name"),
                "mPAE": m.get("mPAE"),
                "iptm": m.get("iptm"),
                "ligand_iptm": m.get("ligand_iptm"),
                "boltz_plddt": m.get("complex_plddt"),
                "boltz_pde": m.get("complex_pde"),
                "boltz_confidence": m.get("confidence_score"),
                "combined_score": m.get("combined_score"),
                "d_SG": m.get("d_SG"),
                "geom_ok": m.get("geom_ok"),
                "n_h_bonds": m.get("n_h_bonds"),
                "n_stabilizing_contacts": m.get("n_stabilizing_contacts"),
                "pocket_occupancy_pct": m.get("pocket_occupancy_pct"),
            })
        boltz_df = pd.DataFrame(boltz_rows)
        DF = DF.merge(boltz_df, on="row_id", how="left")
        n_with = DF["mPAE"].notna().sum()
        print(f"  + Boltz Cys346 cofold metrics: {n_with} / {len(DF)} rows")

    # ── PROPKA pKa_Cys346 ───────────────────────────────────────────────
    if PROPKA_CSV.exists():
        _pk = pd.read_csv(PROPKA_CSV)
        _pk = _pk[_pk["success_flag"] == 1][["row_id", "pKa_Cys346"]].drop_duplicates("row_id", keep="last")
        DF = DF.merge(_pk, on="row_id", how="left")
        print(f"  + PROPKA pKa_Cys346: {DF['pKa_Cys346'].notna().sum()} rows")

    # ── pubTc v3 panel ──────────────────────────────────────────────────
    if PUBTC_V3_CSV.exists():
        _pt3 = pd.read_csv(PUBTC_V3_CSV)
        DF = DF.merge(_pt3, on="row_id", how="left")
        print(f"  + pubTc v3 panel: {DF['max_pubTc'].notna().sum()} rows")

    # ── Efficiency metrics ──────────────────────────────────────────────
    if EFFICIENCY_CSV.exists():
        _eff = pd.read_csv(EFFICIENCY_CSV)
        DF = DF.merge(_eff, on="row_id", how="left")
        print(f"  + Efficiency LLE/LE/BEI/SEI/SILE/fsp3/Lipinski: {DF['LLE'].notna().sum()} rows")

    # ── xTB warhead reactivity ──────────────────────────────────────────
    if WARHEAD_CSV.exists():
        _w = pd.read_csv(WARHEAD_CSV)
        _w = _w[_w["success_flag"] == 1][[
            "row_id", "LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV",
            "q_Cb", "fukui_plus_Cb", "pred_log_k2_GSH",
        ]].drop_duplicates("row_id", keep="last")
        DF = DF.merge(_w, on="row_id", how="left")
        print(f"  + xTB warhead reactivity: {DF['LUMO_eV'].notna().sum()} rows")

    # ── Vina rescore ────────────────────────────────────────────────────
    if VINA_CSV.exists():
        _v = pd.read_csv(VINA_CSV)
        _v = _v[_v["success_flag"] == 1][[
            "row_id", "vina_kcalmol", "vina_inter_kcalmol", "vina_intra_kcalmol",
        ]].drop_duplicates("row_id", keep="last")
        DF = DF.merge(_v, on="row_id", how="left")
        print(f"  + Vina rescore: {DF['vina_kcalmol'].notna().sum()} rows")

    # ── RDKit strain (cofold) ───────────────────────────────────────────
    if RDKIT_STRAIN_COFOLD_CSV.exists():
        _rs = pd.read_csv(RDKIT_STRAIN_COFOLD_CSV)
        _rs = _rs[_rs["success_flag"] == 1][[
            "row_id", "strain_kcal_mol", "vdw_interaction_kcal_mol",
        ]].rename(columns={"strain_kcal_mol": "rdkit_strain_kcal_mol"}).drop_duplicates("row_id", keep="last")
        DF = DF.merge(_rs, on="row_id", how="left")
        print(f"  + RDKit strain (cofold) + vdW: {DF['rdkit_strain_kcal_mol'].notna().sum()} rows")

    # ── RDKit strain (pose-free) ────────────────────────────────────────
    if RDKIT_STRAIN_POSEFREE_CSV.exists():
        try:
            _rp = pd.read_csv(RDKIT_STRAIN_POSEFREE_CSV)
            _rp = _rp[_rp["success_flag"] == 1][["row_id", "strain_kcal_mol"]].rename(
                columns={"strain_kcal_mol": "rdkit_strain_posefree_kcal_mol"}
            ).drop_duplicates("row_id", keep="last")
            DF = DF.merge(_rp, on="row_id", how="left")
            print(f"  + RDKit strain (pose-free): {DF['rdkit_strain_posefree_kcal_mol'].notna().sum()} rows")
        except Exception as e:
            print(f"  ! RDKit strain pose-free load failed: {e}")

    # ── MD-MM-GBSA capped ───────────────────────────────────────────────
    if MDMMGBSA_CSV.exists():
        _md = pd.read_csv(MDMMGBSA_CSV)
        if "success_flag" in _md.columns:
            _md = _md[_md["success_flag"] == 1]
        cols = [c for c in ["row_id", "dG_recognition_md_kcalmol", "dG_recognition_md_std",
                            "ggas_kcalmol", "gsolv_kcalmol"] if c in _md.columns]
        _md = _md[cols].drop_duplicates("row_id", keep="last")
        DF = DF.merge(_md, on="row_id", how="left")
        if "dG_recognition_md_kcalmol" in DF.columns:
            print(f"  + MD-MM-GBSA capped: {DF['dG_recognition_md_kcalmol'].notna().sum()} rows")

    # ── Brenk alerts ────────────────────────────────────────────────────
    if BRENK_CACHE.exists():
        _bk = pd.read_csv(BRENK_CACHE)
        if set(DF["row_id"]).issubset(set(_bk["row_id"])):
            DF = DF.merge(_bk[["row_id", "Brenk_alerts"]].drop_duplicates("row_id"), on="row_id", how="left")
            print(f"  + Brenk alerts: {DF['Brenk_alerts'].notna().sum()} rows")

    print(f"[load] final shape: {DF.shape}, columns: {len(DF.columns)}")
    return DF


# -----------------------------------------------------------------------------
# Ranking
# -----------------------------------------------------------------------------

def apply_simple_rank(df: pd.DataFrame, n: int) -> pd.DataFrame:
    """Apply: pIC50_mean DESC -> ligand_iptm DESC -> LE DESC -> d_SG ASC."""
    needed = ["pIC50_mean", "ligand_iptm", "LE", "d_SG"]
    missing = [c for c in needed if c not in df.columns]
    if missing:
        raise SystemExit(f"Missing required ranking columns: {missing}")

    sub = df.dropna(subset=["pIC50_mean", "ligand_iptm", "d_SG"]).copy()
    print(f"[rank] non-null pIC50_mean+ligand_iptm+d_SG: {len(sub):,} of {len(df):,}")

    # If LE has NaN, fill with median so rows still rank (not be excluded entirely)
    if sub["LE"].isna().any():
        n_na = sub["LE"].isna().sum()
        med = sub["LE"].median()
        sub["LE_for_rank"] = sub["LE"].fillna(med)
        print(f"[rank] LE NaN filled with median ({med:.3f}) for {n_na} rows")
    else:
        sub["LE_for_rank"] = sub["LE"]

    ranked = sub.sort_values(
        by=["pIC50_mean", "ligand_iptm", "LE_for_rank", "d_SG"],
        ascending=[False, False, False, True],
        kind="mergesort",
    ).head(n).drop(columns=["LE_for_rank"])
    return ranked


# -----------------------------------------------------------------------------
# Anchor / cohort inference
# -----------------------------------------------------------------------------

ANCHOR_TOKENS = ["H2", "H3", "L0", "L1_FT_H2", "L1", "C1", "C5", "Mol1", "M1"]


def infer_anchor_variant(method: str | None, yaml_name: str | None, smiles: str | None) -> str:
    """Best-effort anchor inference from method label + yaml_name."""
    m = (method or "").lower()
    y = (yaml_name or "")
    # First, scan yaml_name for an explicit anchor token (case-sensitive)
    for tok in ANCHOR_TOKENS:
        if tok in y:
            return tok
    # Tier 1 (Med-chem Rules) and Tier 2 (Amine Replacements) ARE Mol1-anchored MMPs
    if "medchem rules" in m:
        return "Mol1"
    if "amine replacements" in m:
        return "Mol1"
    # Tier 3 generative: explicit anchor in name, else "sequence baseline"
    if "libinvent" in m:
        return "LibInvent_locked"
    if "de novo" in m:
        return "sequence_baseline"
    if "mol2mol" in m:
        return "Mol2Mol_warhead"
    if "constrained generative" in m:
        return "Tier3_v2"
    return "unknown"


def infer_cohort_source(method: str | None) -> str:
    """Short cohort tag for the dashboard."""
    m = (method or "").lower()
    if "medchem rules" in m:
        return "medchem_rules"
    if "amine replacements" in m:
        return "amine_replacements"
    if "libinvent" in m:
        return "reinvent4_libinvent"
    if "de novo" in m:
        return "reinvent4_denovo"
    if "mol2mol" in m:
        return "reinvent4_mol2mol"
    if "constrained generative" in m:
        return "tier3_v2"
    return (method or "unknown")


# -----------------------------------------------------------------------------
# Nearest ChEMBL ZAP70 active neighbors
# -----------------------------------------------------------------------------

def load_zap70_actives_pool(min_pIC50: float = 7.0) -> pd.DataFrame:
    """Load the canonical 280-mol ZAP70 set used by run_zap70_v3 and filter to actives."""
    from experiments.run_zap70_v3 import load_zap70_molecules  # type: ignore
    train_df, _ = load_zap70_molecules()
    pool = train_df[train_df["pIC50"] >= min_pIC50].copy().reset_index(drop=True)
    print(f"[pool] ZAP70 actives (pIC50>={min_pIC50}): {len(pool)} / {len(train_df)} mols")
    return pool


def compute_morgan_fps(smiles_list: list[str], radius: int = 2, nbits: int = 2048):
    fps = []
    valid_idx = []
    for i, s in enumerate(smiles_list):
        m = Chem.MolFromSmiles(s)
        if m is None:
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius, nBits=nbits))
        valid_idx.append(i)
    return fps, valid_idx


def top5_neighbors(query_smiles: str, pool_df: pd.DataFrame, pool_fps: list) -> list[dict]:
    """Return up to 5 nearest ChEMBL actives for one query SMILES."""
    qm = Chem.MolFromSmiles(query_smiles)
    if qm is None:
        return []
    qfp = AllChem.GetMorganFingerprintAsBitVect(qm, 2, nBits=2048)
    sims = np.array(DataStructs.BulkTanimotoSimilarity(qfp, pool_fps))
    order = np.argsort(-sims)[:5]
    out = []
    for idx in order:
        row = pool_df.iloc[int(idx)]
        out.append({
            "chembl_id": str(row["molecule_chembl_id"]),
            "smiles": str(row["smiles"]),
            "pIC50": float(row["pIC50"]),
            "tanimoto": float(sims[int(idx)]),
        })
    return out


# -----------------------------------------------------------------------------
# Serialisation
# -----------------------------------------------------------------------------

def _scrub(v):
    """Convert numpy / pandas scalars to JSON-safe Python builtins."""
    if v is None:
        return None
    if isinstance(v, float) and np.isnan(v):
        return None
    if isinstance(v, np.floating):
        f = float(v)
        return None if np.isnan(f) else f
    if isinstance(v, np.integer):
        return int(v)
    if isinstance(v, (np.bool_, bool)):
        return bool(v)
    if isinstance(v, (pd.Timestamp,)):
        return str(v)
    return v


def row_to_record(row: pd.Series, neighbors: list[dict]) -> dict:
    cols: dict = {c: _scrub(row[c]) for c in row.index}
    return {
        "mol_id": f"row_{int(row['row_id'])}",
        "row_id": int(row["row_id"]),
        "smiles": str(row["smiles"]),
        "method": str(row.get("method", "")),
        "yaml_name": str(row.get("yaml_name", "") or ""),
        "anchor_variant": infer_anchor_variant(row.get("method"), row.get("yaml_name"), row.get("smiles")),
        "cohort_source": infer_cohort_source(row.get("method")),
        "columns": cols,
        "nearest_chembl_zap70_actives": neighbors,
    }


# -----------------------------------------------------------------------------
# Manifest writer
# -----------------------------------------------------------------------------

def write_manifest(out_path: Path, label: str, n: int, picked: pd.DataFrame,
                   input_csv: Path, source_total: int) -> None:
    cohort_counts = picked["method"].value_counts().to_dict()
    anchor_counts: dict = {}
    for _, r in picked.iterrows():
        a = infer_anchor_variant(r.get("method"), r.get("yaml_name"), r.get("smiles"))
        anchor_counts[a] = anchor_counts.get(a, 0) + 1

    med_pic50 = float(picked["pIC50_mean"].median())
    mean_pic50 = float(picked["pIC50_mean"].mean())
    med_iptm = float(picked["ligand_iptm"].median())
    med_dsg = float(picked["d_SG"].median())
    med_le = float(picked["LE"].median()) if "LE" in picked.columns else None

    cohort_lines = "\n".join(f"  - **{k}**: {v}" for k, v in cohort_counts.items())
    anchor_lines = "\n".join(f"  - **{k}**: {v}" for k, v in anchor_counts.items())

    md = f"""# LLM Cohort Input Manifest

**Label**: {label}
**N selected**: {n}
**Source CSV**: `{input_csv.relative_to(PROJECT_ROOT)}`
**Source rows (after backend filters)**: {source_total:,}

## Selection rule

Simple-rank, in this exact order (mergesort, NaN excluded for required keys):

1. `pIC50_mean` DESC
2. `ligand_iptm` DESC
3. `LE` DESC (NaN filled with median for ranking)
4. `d_SG` ASC

Rows missing `pIC50_mean`, `ligand_iptm`, or `d_SG` are excluded from ranking
(only ~997 cofolded rows have `ligand_iptm`/`d_SG`, so the ranking pool is
effectively the cofold subset).

## Selected cohort summary

- Median pIC50_mean: **{med_pic50:.3f}**
- Mean pIC50_mean: {mean_pic50:.3f}
- Median ligand_iptm: {med_iptm:.3f}
- Median d_SG (Å): {med_dsg:.3f}
- Median LE: {med_le:.3f}

## Source method distribution

{cohort_lines}

## Anchor variant distribution

{anchor_lines}

## Per-molecule payload

Each entry in the companion JSON contains:

- `mol_id`, `row_id`, `smiles`, `method`, `yaml_name`
- `anchor_variant` (best-effort: parses yaml_name for tokens H2/H3/L/C1/C5/L0/L1_FT_H2/Mol1; falls back to method-based heuristic)
- `cohort_source` (short tag: `medchem_rules` / `amine_replacements` / `reinvent4_libinvent` / `reinvent4_denovo` / `reinvent4_mol2mol` / `tier3_v2`)
- `columns` — every column from the enriched DataFrame (all 73+ scoring columns: pIC50_mean/std, ligand_iptm, mPAE, d_SG, warhead_dev_deg, vina_kcalmol, pKa_Cys346, pred_log_k2_GSH, MW/LogP/TPSA, SAScore, PAINS_alerts, Brenk_alerts, LLE/LE/BEI/SEI/SILE, fsp3, max_Tc_train, max_pubTc, closest_lead, xTB descriptors, RDKit strain, etc.)
- `nearest_chembl_zap70_actives` — top-5 by Tanimoto on Morgan FP r2/2048, drawn from the canonical 280-mol ZAP70 set (`experiments.run_zap70_v3.load_zap70_molecules`) filtered to `pIC50 >= 7.0` (the only on-disk pIC50-annotated ZAP70 active pool). Each neighbor has `chembl_id`, `smiles`, `pIC50`, `tanimoto`.

> NOTE: the ZAP70 set carries `CHEMBL4899` in older overlapping-assay extracts;
> `load_zap70_molecules()` resolves to the corrected `CHEMBL2803` mols (per
> memory note `target_identity_correction.md`).
"""
    out_path.write_text(md)
    print(f"[write] manifest -> {out_path}")


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Select top-N for LLM cohort scoring")
    ap.add_argument("--n", type=int, required=True, help="Number of molecules to select")
    ap.add_argument("--input", type=Path, default=DEFAULT_INPUT, help="Bulk-scored CSV input")
    ap.add_argument("--output", type=Path, required=True, help="Output JSON path")
    ap.add_argument("--manifest", type=Path, required=True, help="Output markdown manifest path")
    ap.add_argument("--label", type=str, default="UNLABELED", help="Free-text label for the manifest")
    ap.add_argument("--min-active-pIC50", type=float, default=7.0, help="ZAP70 active pIC50 cutoff for neighbor pool")
    args = ap.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.manifest.parent.mkdir(parents=True, exist_ok=True)

    df = load_and_enrich(args.input)
    source_total = len(df)
    picked = apply_simple_rank(df, args.n)
    print(f"[rank] picked {len(picked)} rows")

    # Build neighbor pool
    pool = load_zap70_actives_pool(min_pIC50=args.min_active_pIC50)
    pool_fps, _ = compute_morgan_fps(pool["smiles"].tolist())
    if len(pool_fps) != len(pool):
        # Drop any rows whose SMILES failed to parse so indices align
        keep_idx = []
        for i, s in enumerate(pool["smiles"].tolist()):
            if Chem.MolFromSmiles(s) is not None:
                keep_idx.append(i)
        pool = pool.iloc[keep_idx].reset_index(drop=True)

    records = []
    for _, row in picked.iterrows():
        nbrs = top5_neighbors(row["smiles"], pool, pool_fps)
        records.append(row_to_record(row, nbrs))

    args.output.write_text(json.dumps(records, indent=2))
    print(f"[write] JSON -> {args.output} ({len(records)} records, "
          f"{args.output.stat().st_size / 1024:.1f} KB)")

    write_manifest(
        out_path=args.manifest,
        label=args.label,
        n=len(records),
        picked=picked,
        input_csv=args.input,
        source_total=source_total,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
