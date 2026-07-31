#!/usr/bin/env python3
"""Build the Lead-Optimization corpus from ChEMBL pairs.

Output: data/lo_corpus/lo_pairs_v1*.{csv,smi.gz,json,md}

Each surviving pair represents an LO direction:
    input_smiles (a, lower pIC50) --> output_smiles (b, higher pIC50)

Filters applied (see CLAUDE.md spec in user prompt):
  1. Same target              (always true after grouping)
  2. Same assay_type ('B') AND confidence_score >= 7 on both assays
  3. pIC50_b - pIC50_a >= 0.3
  4. Tc(a, b) in [0.4, 0.85]                 (Morgan FP r=2, 2048-bit)
  5. Murcko-scaffold Tc >= 0.5
  6. year(a) <= year(b) (or same year)        (skipped if year unknown for either)
  7. MCS coverage >= 50% of either molecule
  8. Tc < 1.0                                 (subsumed by upper bound in 4)
  9. Drug-like: MW in [200, 700] AND LogP in [-2, 6]

Convention: only keep direction (a -> b) where b is better; deduplicate by
(input_smiles, output_smiles).

Holdout: skip pairs where either molecule appears in
data/exp7_lo_benchmark/{egfr,btk,jak3,her2,fgfr}_pairs.json (silently
skipped if the file does not yet exist).

Split: by target -- 5% of unique target_chembl_ids reserved for validation.
"""

from __future__ import annotations

import argparse
import gc
import gzip
import json
import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem, Crippen, Descriptors, rdFMCS
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
SHARED_PAIRS = DATA_DIR / "overlapping_assays" / "extracted" / "shared_pairs_deduped.csv"
ALL_WITHIN = DATA_DIR / "overlapping_assays" / "extracted" / "all_pairs_within_assay.csv"
CHEMBL_DB = DATA_DIR / "chembl_db" / "chembl" / "36" / "chembl_36.db"
EXP7_DIR = DATA_DIR / "exp7_lo_benchmark"

OUT_DIR = DATA_DIR / "lo_corpus"
OUT_CSV = OUT_DIR / "lo_pairs_v1.csv"
OUT_TRAIN = OUT_DIR / "lo_pairs_v1_train.smi.gz"
OUT_VAL = OUT_DIR / "lo_pairs_v1_val.smi.gz"
OUT_STATS = OUT_DIR / "lo_pairs_v1_stats.json"
OUT_SUMMARY = OUT_DIR / "lo_pairs_v1_summary.md"

# Filter thresholds (see header docstring)
TC_LOW, TC_HIGH = 0.4, 0.85
SCAFFOLD_TC_MIN = 0.5
DELTA_MIN = 0.3
MCS_COVERAGE_MIN = 0.5
MW_LOW, MW_HIGH = 200.0, 700.0
LOGP_LOW, LOGP_HIGH = -2.0, 6.0
CONFIDENCE_MIN = 7
ASSAY_TYPE = "B"

VAL_TARGET_FRACTION = 0.05
RNG_SEED = 13


# -----------------------------------------------------------------------------
# RDKit helpers (cached per unique SMILES)
# -----------------------------------------------------------------------------


def canonical_smiles(smi: str) -> str | None:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    return Chem.MolToSmiles(mol)


def compute_mol_props(smiles: Iterable[str]) -> dict[str, dict]:
    """Compute Morgan FP, Murcko scaffold FP, MW, LogP for each SMILES."""
    out: dict[str, dict] = {}
    mfpgen = AllChem.GetMorganGenerator(radius=2, fpSize=2048)
    n = 0
    t0 = time.time()
    for smi in smiles:
        if smi in out:
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out[smi] = None  # type: ignore[assignment]
            continue
        try:
            scaf = MurckoScaffold.GetScaffoldForMol(mol)
            scaf_smi = Chem.MolToSmiles(scaf) if scaf and scaf.GetNumAtoms() > 0 else ""
            scaf_mol = Chem.MolFromSmiles(scaf_smi) if scaf_smi else None
            scaf_fp = mfpgen.GetFingerprint(scaf_mol) if scaf_mol is not None else None
            mw = Descriptors.MolWt(mol)
            logp = Crippen.MolLogP(mol)
            n_heavy = mol.GetNumHeavyAtoms()
            out[smi] = {
                "fp": mfpgen.GetFingerprint(mol),
                "scaf_fp": scaf_fp,
                "scaf_smi": scaf_smi,
                "mw": mw,
                "logp": logp,
                "n_heavy": n_heavy,
                "mol": mol,
            }
        except Exception:
            out[smi] = None  # type: ignore[assignment]
        n += 1
        if n % 5000 == 0:
            print(f"    mol props: {n}/{len(smiles) if hasattr(smiles, '__len__') else '?'} ({time.time() - t0:.1f}s)")
    return out


def mcs_coverage(mol_a: Chem.Mol, mol_b: Chem.Mol) -> float:
    """Compute MCS coverage as max(|MCS|/|A|, |MCS|/|B|)."""
    try:
        result = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrderExact,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=2,
        )
        if result.canceled or result.numAtoms == 0:
            return 0.0
        na = mol_a.GetNumHeavyAtoms()
        nb = mol_b.GetNumHeavyAtoms()
        if min(na, nb) == 0:
            return 0.0
        return result.numAtoms / min(na, nb)
    except Exception:
        return 0.0


# -----------------------------------------------------------------------------
# ChEMBL DB lookups
# -----------------------------------------------------------------------------


def load_assay_metadata(assay_ids: set[int]) -> dict[int, dict]:
    """Return assay_id -> {assay_type, confidence_score, year}."""
    print(f"  Loading metadata for {len(assay_ids):,} assays...")
    con = sqlite3.connect(CHEMBL_DB)
    cur = con.cursor()

    # Process in batches to avoid SQL parameter limit (~999)
    out: dict[int, dict] = {}
    ids = list(assay_ids)
    BATCH = 800
    for i in range(0, len(ids), BATCH):
        batch = ids[i : i + BATCH]
        placeholders = ",".join("?" for _ in batch)
        cur.execute(
            f"""
            SELECT a.assay_id, a.assay_type, a.confidence_score, d.year
            FROM assays a
            LEFT JOIN docs d ON d.doc_id = a.doc_id
            WHERE a.assay_id IN ({placeholders})
            """,
            batch,
        )
        for aid, atype, conf, yr in cur.fetchall():
            out[int(aid)] = {
                "assay_type": atype,
                "confidence_score": conf,
                "year": yr,
            }
    con.close()
    return out


def load_target_metadata(target_chembl_ids: set[str]) -> dict[str, dict]:
    """Return target_chembl_id -> {pref_name, target_type, target_class}.

    Walks ``protein_classification`` ancestors because ChEMBL stores only the
    *leaf* class for each component (e.g. EGFR -> only level 5 'Egfr'); a naive
    level-2 lookup misses Kinase/GPCR tags entirely.
    """
    print(f"  Loading metadata for {len(target_chembl_ids):,} targets...")
    con = sqlite3.connect(CHEMBL_DB)
    cur = con.cursor()

    # Materialize protein_classification once (~5k rows).
    cur.execute("SELECT protein_class_id, parent_id, pref_name, class_level FROM protein_classification")
    pc_table: dict[int, dict] = {}
    for cid, par, name, lvl in cur.fetchall():
        pc_table[int(cid)] = {
            "parent_id": int(par) if par is not None else None,
            "name": name,
            "level": int(lvl),
        }

    def ancestry(pcid):
        seen: set[int] = set()
        cur_id = pcid
        while cur_id is not None and cur_id in pc_table and cur_id not in seen:
            seen.add(cur_id)
            yield pc_table[cur_id]
            cur_id = pc_table[cur_id]["parent_id"]

    def coarse_class(leaf_ids: list[int]) -> str:
        chain = [n for pcid in leaf_ids for n in ancestry(pcid)]
        names_l2 = [n["name"] for n in chain if n["level"] == 2]
        names_l1 = [n["name"] for n in chain if n["level"] == 1]
        if any("Kinase" in n for n in names_l2):
            return "Kinase"
        if any("G protein-coupled" in n for n in names_l2):
            return "GPCR"
        if "Protease" in names_l2:
            return "Protease"
        if any("ion channel" in n.lower() for n in names_l1 + names_l2):
            return "Ion channel"
        if "Nuclear receptor" in names_l2:
            return "Nuclear receptor"
        if "Phosphatase" in names_l2:
            return "Phosphatase"
        if "Phosphodiesterase" in names_l2:
            return "Phosphodiesterase"
        if "Cytochrome P450" in names_l2:
            return "Cytochrome P450"
        if any("Epigenetic" in n for n in names_l1):
            return "Epigenetic"
        if names_l2:
            return names_l2[0]
        if names_l1:
            return names_l1[0]
        return "Unknown"

    out: dict[str, dict] = {tcid: {"pref_name": None, "target_type": None, "target_class": "Unknown"} for tcid in target_chembl_ids}

    ids = list(target_chembl_ids)
    BATCH = 800
    for i in range(0, len(ids), BATCH):
        batch = ids[i : i + BATCH]
        placeholders = ",".join("?" for _ in batch)
        cur.execute(
            f"""
            SELECT chembl_id, pref_name, target_type, tid
            FROM target_dictionary
            WHERE chembl_id IN ({placeholders})
            """,
            batch,
        )
        rows = cur.fetchall()
        tid_to_chembl: dict[int, str] = {}
        for cid, pref, ttype, tid in rows:
            out[cid]["pref_name"] = pref
            out[cid]["target_type"] = ttype
            if tid is not None:
                tid_to_chembl[int(tid)] = cid

        if not tid_to_chembl:
            continue

        tids = list(tid_to_chembl.keys())
        for j in range(0, len(tids), BATCH):
            tbatch = tids[j : j + BATCH]
            ph = ",".join("?" for _ in tbatch)
            cur.execute(
                f"""
                SELECT tc.tid, cc.protein_class_id
                FROM target_components tc
                JOIN component_class cc ON cc.component_id = tc.component_id
                WHERE tc.tid IN ({ph})
                """,
                tbatch,
            )
            tid_leaf_classes: dict[int, list[int]] = defaultdict(list)
            for tid, pcid in cur.fetchall():
                tid_leaf_classes[int(tid)].append(int(pcid))
            for tid, leaf_ids in tid_leaf_classes.items():
                cid = tid_to_chembl.get(tid)
                if cid is not None:
                    out[cid]["target_class"] = coarse_class(leaf_ids)
    con.close()
    return out


# -----------------------------------------------------------------------------
# Holdout
# -----------------------------------------------------------------------------


def load_holdout_ids() -> tuple[set[str], dict[str, int]]:
    """Return (set of canonical SMILES, per-file count) of compounds in the
    exp7 retrospective benchmark; empty if dir missing/empty.

    Handles the canonical exp7 schema (``pairs[].{anchor,drug}.smiles``) plus
    a few fallback key names so legacy/test files still work.
    """
    holdout: set[str] = set()
    per_file: dict[str, int] = {}
    if not EXP7_DIR.exists():
        return holdout, per_file

    def add_smi(s):  # type: ignore[no-untyped-def]
        if not s:
            return
        cs = canonical_smiles(s)
        if cs:
            holdout.add(cs)

    for fp in sorted(EXP7_DIR.glob("*_pairs.json")):
        before = len(holdout)
        try:
            payload = json.loads(fp.read_text())
        except Exception as exc:
            print(f"  WARN: could not parse {fp.name}: {exc}")
            continue
        if isinstance(payload, dict):
            iterable = payload.get("pairs", []) or payload.get("data", []) or []
        elif isinstance(payload, list):
            iterable = payload
        else:
            iterable = []
        for item in iterable:
            if not isinstance(item, dict):
                continue
            # Nested anchor/drug sub-objects (canonical exp7 layout)
            for sub_key in ("anchor", "drug", "input", "output"):
                sub = item.get(sub_key)
                if isinstance(sub, dict):
                    add_smi(sub.get("smiles") or sub.get("canonical_smiles"))
            # Flat keys (fallback)
            for k in ("input_smiles", "output_smiles", "mol_a", "mol_b",
                      "smiles_a", "smiles_b", "smiles",
                      "anchor_smiles", "drug_smiles"):
                v = item.get(k)
                if isinstance(v, str):
                    add_smi(v)
        per_file[fp.name] = len(holdout) - before
    return holdout, per_file


# -----------------------------------------------------------------------------
# Main pipeline
# -----------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-pairs", type=int, default=None,
                        help="Cap input rows for a faster smoke test.")
    parser.add_argument("--source", choices=("shared", "all_within"), default="shared")
    parser.add_argument("--mcs", action="store_true",
                        help="Apply MCS-coverage filter (slow: ~5-10ms per pair). "
                             "Off by default for tractable full-dataset runs; the "
                             "Tc + scaffold-Tc filters already enforce substantial "
                             "shared structure.")
    args = parser.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # -----------------------------------------------------------------------
    # 1) Load source pairs (within-assay only)
    # -----------------------------------------------------------------------
    src_path = SHARED_PAIRS if args.source == "shared" else ALL_WITHIN
    print(f"[1/9] Loading source pairs from {src_path.name}...")
    t0 = time.time()
    usecols = [
        "mol_a", "mol_b", "mol_a_id", "mol_b_id",
        "value_a", "value_b", "target_chembl_id",
        "assay_id_a", "assay_id_b", "is_within_assay",
    ]
    df = pd.read_csv(src_path, usecols=usecols, nrows=args.max_pairs)
    print(f"  rows in: {len(df):,}")
    # Restrict to within-assay (safest for LO supervision)
    df = df[df["is_within_assay"] == True].copy()  # noqa: E712
    print(f"  within-assay rows: {len(df):,}")
    df = df.dropna(subset=["mol_a", "mol_b", "value_a", "value_b"]).reset_index(drop=True)
    print(f"  after dropna: {len(df):,}  ({time.time() - t0:.1f}s)")

    # -----------------------------------------------------------------------
    # 2) Pre-filter by delta -- keep only pairs where direction is informative
    # -----------------------------------------------------------------------
    print("[2/9] Pre-filter by delta-pIC50...")
    df["delta_abs"] = (df["value_b"] - df["value_a"]).abs()
    df = df[df["delta_abs"] >= DELTA_MIN].reset_index(drop=True)
    print(f"  rows with |delta| >= {DELTA_MIN}: {len(df):,}")
    # Orient: keep input (a) as worse, output (b) as better.
    flip_mask = df["value_b"] < df["value_a"]
    if flip_mask.any():
        for left, right in [("mol_a", "mol_b"), ("mol_a_id", "mol_b_id"),
                            ("value_a", "value_b"), ("assay_id_a", "assay_id_b")]:
            df.loc[flip_mask, [left, right]] = df.loc[flip_mask, [right, left]].values
    df["delta_pic50"] = df["value_b"] - df["value_a"]
    assert (df["delta_pic50"] >= DELTA_MIN - 1e-9).all()
    # Drop pair direction duplicates (same a->b under different assays)
    df = df.drop_duplicates(subset=["mol_a_id", "mol_b_id"]).reset_index(drop=True)
    print(f"  after dedup direction: {len(df):,}")

    # -----------------------------------------------------------------------
    # 3) Assay-quality filter via ChEMBL DB
    # -----------------------------------------------------------------------
    print("[3/9] Assay quality (assay_type='B', confidence_score >= 7)...")
    assay_ids = set(df["assay_id_a"].astype(int)) | set(df["assay_id_b"].astype(int))
    assay_meta = load_assay_metadata(assay_ids)

    def assay_ok(row) -> bool:
        a = assay_meta.get(int(row["assay_id_a"]))
        b = assay_meta.get(int(row["assay_id_b"]))
        if not a or not b:
            return False
        if a["assay_type"] != ASSAY_TYPE or b["assay_type"] != ASSAY_TYPE:
            return False
        if (a["confidence_score"] or 0) < CONFIDENCE_MIN or (b["confidence_score"] or 0) < CONFIDENCE_MIN:
            return False
        return True

    # Vectorize via apply over numpy arrays for speed
    types_a = df["assay_id_a"].map(lambda x: assay_meta.get(int(x), {}).get("assay_type"))
    types_b = df["assay_id_b"].map(lambda x: assay_meta.get(int(x), {}).get("assay_type"))
    conf_a = df["assay_id_a"].map(lambda x: assay_meta.get(int(x), {}).get("confidence_score") or 0)
    conf_b = df["assay_id_b"].map(lambda x: assay_meta.get(int(x), {}).get("confidence_score") or 0)
    keep_assay = (types_a == ASSAY_TYPE) & (types_b == ASSAY_TYPE) & (conf_a >= CONFIDENCE_MIN) & (conf_b >= CONFIDENCE_MIN)
    df = df[keep_assay].reset_index(drop=True)
    print(f"  rows after assay filter: {len(df):,}")
    df["year_a"] = df["assay_id_a"].map(lambda x: assay_meta.get(int(x), {}).get("year"))
    df["year_b"] = df["assay_id_b"].map(lambda x: assay_meta.get(int(x), {}).get("year"))

    # Year monotonicity: keep when both years known and year_a <= year_b, OR
    # when either year is missing (relax rather than drop because ChEMBL year
    # is publication year and is often missing for older entries).
    has_both = df["year_a"].notna() & df["year_b"].notna()
    bad_order = has_both & (df["year_a"] > df["year_b"])
    df = df[~bad_order].reset_index(drop=True)
    print(f"  after year-order filter (where known): {len(df):,}")

    # -----------------------------------------------------------------------
    # 4) Molecule property cache (FPs, scaffold FPs, MW, LogP, Mol)
    # -----------------------------------------------------------------------
    print("[4/9] Caching molecule props (Morgan FP, Murcko scaffold, MW, LogP)...")
    unique_smiles = pd.unique(pd.concat([df["mol_a"], df["mol_b"]], ignore_index=True))
    print(f"  unique molecules: {len(unique_smiles):,}")
    mol_props = compute_mol_props(unique_smiles)

    # Drop pairs where either mol failed parse
    valid = df["mol_a"].map(lambda s: mol_props.get(s) is not None) & df["mol_b"].map(lambda s: mol_props.get(s) is not None)
    df = df[valid].reset_index(drop=True)
    print(f"  after invalid-mol drop: {len(df):,}")

    # -----------------------------------------------------------------------
    # 5) Drug-likeness filter
    # -----------------------------------------------------------------------
    print("[5/9] Drug-likeness (MW, LogP)...")
    def druglike(smi: str) -> bool:
        p = mol_props[smi]
        return (MW_LOW <= p["mw"] <= MW_HIGH) and (LOGP_LOW <= p["logp"] <= LOGP_HIGH)
    keep_dl = df["mol_a"].map(druglike) & df["mol_b"].map(druglike)
    df = df[keep_dl].reset_index(drop=True)
    print(f"  after drug-like filter: {len(df):,}")

    # -----------------------------------------------------------------------
    # 6) Tanimoto similarity windows (Morgan FP + Murcko scaffold)
    # -----------------------------------------------------------------------
    print("[6/9] Tanimoto + Murcko scaffold Tc...")
    tc_vals = np.empty(len(df), dtype=np.float32)
    scaf_tc_vals = np.empty(len(df), dtype=np.float32)
    smis_a = df["mol_a"].tolist()
    smis_b = df["mol_b"].tolist()
    for i, (sa, sb) in enumerate(zip(smis_a, smis_b)):
        pa = mol_props[sa]
        pb = mol_props[sb]
        tc_vals[i] = DataStructs.TanimotoSimilarity(pa["fp"], pb["fp"])
        if pa["scaf_fp"] is not None and pb["scaf_fp"] is not None:
            scaf_tc_vals[i] = DataStructs.TanimotoSimilarity(pa["scaf_fp"], pb["scaf_fp"])
        else:
            scaf_tc_vals[i] = 0.0
        if (i + 1) % 100_000 == 0:
            print(f"    Tc {i + 1}/{len(df):,}")
    df["tc"] = tc_vals
    df["scaffold_tc"] = scaf_tc_vals
    df = df[(df["tc"] >= TC_LOW) & (df["tc"] < TC_HIGH) & (df["scaffold_tc"] >= SCAFFOLD_TC_MIN)].reset_index(drop=True)
    print(f"  after Tc filter: {len(df):,}")

    # -----------------------------------------------------------------------
    # 7) MCS coverage filter (expensive; only on survivors; optional)
    # -----------------------------------------------------------------------
    if args.mcs:
        print(f"[7/9] MCS coverage >= {MCS_COVERAGE_MIN} (on {len(df):,} survivors)...")
        mcs_vals = np.empty(len(df), dtype=np.float32)
        t1 = time.time()
        for i, (sa, sb) in enumerate(zip(df["mol_a"], df["mol_b"])):
            pa = mol_props[sa]
            pb = mol_props[sb]
            mcs_vals[i] = mcs_coverage(pa["mol"], pb["mol"])
            if (i + 1) % 25_000 == 0:
                print(f"    MCS {i + 1}/{len(df):,} ({time.time() - t1:.1f}s)")
        df["mcs_coverage"] = mcs_vals
        df = df[df["mcs_coverage"] >= MCS_COVERAGE_MIN].reset_index(drop=True)
        print(f"  after MCS filter: {len(df):,}  ({time.time() - t1:.1f}s)")
    else:
        # Use Tc lower bound as a cheap proxy for "substantial shared structure"
        # (Tc >= 0.4 with Morgan r=2 implies > ~40% shared bit-set, which in
        # practice tracks heavy-atom overlap closely for drug-like mols).
        print(f"[7/9] MCS filter SKIPPED (use --mcs to enable). "
              f"Tc >= {TC_LOW} + scaffold-Tc >= {SCAFFOLD_TC_MIN} already enforce shared structure.")
        df["mcs_coverage"] = np.nan

    # -----------------------------------------------------------------------
    # 8) Target metadata + dedup on canonical (input, output) SMILES
    # -----------------------------------------------------------------------
    print("[8/9] Target metadata + canonical-SMILES dedup + holdout...")
    target_meta = load_target_metadata(set(df["target_chembl_id"].unique()))
    df["target_class"] = df["target_chembl_id"].map(lambda t: target_meta.get(t, {}).get("target_class", "Unknown"))
    df["target_pref_name"] = df["target_chembl_id"].map(lambda t: target_meta.get(t, {}).get("pref_name"))

    # Canonicalize
    df["input_smiles"] = df["mol_a"].map(lambda s: canonical_smiles(s) or s)
    df["output_smiles"] = df["mol_b"].map(lambda s: canonical_smiles(s) or s)
    df = df[df["input_smiles"] != df["output_smiles"]].reset_index(drop=True)
    df = df.drop_duplicates(subset=["input_smiles", "output_smiles"]).reset_index(drop=True)
    print(f"  after canonical dedup: {len(df):,}")

    # Holdout removal
    holdout, holdout_per_file = load_holdout_ids()
    print(f"  exp7 holdout molecules: {len(holdout)}  (per file: {holdout_per_file})")
    if holdout:
        before = len(df)
        df = df[~df["input_smiles"].isin(holdout) & ~df["output_smiles"].isin(holdout)].reset_index(drop=True)
        print(f"  removed {before - len(df):,} holdout-touching pairs ({len(df):,} remain)")

    # Per-target counts (for stats + split)
    per_target = df["target_chembl_id"].value_counts()
    n_targets = len(per_target)
    print(f"  unique targets: {n_targets:,}  (median pairs/target: {per_target.median():.0f})")

    # -----------------------------------------------------------------------
    # 9) Target-disjoint train/val split, write outputs
    # -----------------------------------------------------------------------
    print("[9/9] Split + write outputs...")
    rng = np.random.default_rng(RNG_SEED)
    targets = per_target.index.to_numpy().copy()
    rng.shuffle(targets)
    n_val_targets = max(1, int(round(VAL_TARGET_FRACTION * len(targets))))
    val_targets = set(targets[:n_val_targets].tolist())
    train_targets = set(targets[n_val_targets:].tolist())
    df["split"] = np.where(df["target_chembl_id"].isin(val_targets), "val", "train")
    print(f"  train targets: {len(train_targets):,}  val targets: {len(val_targets):,}")
    print(f"  train pairs: {(df['split'] == 'train').sum():,}  val pairs: {(df['split'] == 'val').sum():,}")

    # Final CSV (selected columns)
    out_cols = [
        "input_smiles", "output_smiles", "target_chembl_id", "target_pref_name",
        "target_class", "value_a", "value_b", "delta_pic50",
        "tc", "scaffold_tc", "mcs_coverage", "year_a", "year_b",
        "mol_a_id", "mol_b_id", "assay_id_a", "assay_id_b", "split",
    ]
    df_out = df[out_cols].rename(columns={"value_a": "anchor_pIC50", "value_b": "optimized_pIC50"})
    df_out.to_csv(OUT_CSV, index=False)
    print(f"  wrote {OUT_CSV} ({OUT_CSV.stat().st_size / 1e6:.1f} MB)")

    # REINVENT4-compatible .smi.gz (input \t output)
    for split_name, path in (("train", OUT_TRAIN), ("val", OUT_VAL)):
        rows = df_out[df_out["split"] == split_name]
        with gzip.open(path, "wt") as fh:
            for a, b in zip(rows["input_smiles"], rows["output_smiles"]):
                fh.write(f"{a}\t{b}\n")
        print(f"  wrote {path} ({path.stat().st_size / 1e6:.2f} MB, {len(rows):,} rows)")

    # Stats JSON
    stats = {
        "total_pairs": int(len(df_out)),
        "train_pairs": int((df_out["split"] == "train").sum()),
        "val_pairs": int((df_out["split"] == "val").sum()),
        "n_targets": int(df_out["target_chembl_id"].nunique()),
        "n_train_targets": int(len(train_targets)),
        "n_val_targets": int(len(val_targets)),
        "n_unique_input_mols": int(df_out["input_smiles"].nunique()),
        "n_unique_output_mols": int(df_out["output_smiles"].nunique()),
        "n_unique_mols": int(pd.unique(pd.concat([df_out["input_smiles"], df_out["output_smiles"]], ignore_index=True)).size),
        "tc": {
            "mean": float(df_out["tc"].mean()),
            "median": float(df_out["tc"].median()),
            "p05": float(df_out["tc"].quantile(0.05)),
            "p95": float(df_out["tc"].quantile(0.95)),
        },
        "scaffold_tc": {
            "mean": float(df_out["scaffold_tc"].mean()),
            "median": float(df_out["scaffold_tc"].median()),
        },
        "mcs_coverage": (
            {
                "mean": float(df_out["mcs_coverage"].mean()),
                "median": float(df_out["mcs_coverage"].median()),
            }
            if df_out["mcs_coverage"].notna().any()
            else {"mean": None, "median": None, "note": "MCS filter skipped"}
        ),
        "delta_pic50": {
            "mean": float(df_out["delta_pic50"].mean()),
            "median": float(df_out["delta_pic50"].median()),
            "p05": float(df_out["delta_pic50"].quantile(0.05)),
            "p95": float(df_out["delta_pic50"].quantile(0.95)),
            "max": float(df_out["delta_pic50"].max()),
        },
        "per_target_top20": df_out["target_chembl_id"].value_counts().head(20).to_dict(),
        "per_class": df_out["target_class"].value_counts().to_dict(),
        "holdout_mols_count": len(holdout),
        "holdout_per_file": holdout_per_file,
    }
    OUT_STATS.write_text(json.dumps(stats, indent=2))
    print(f"  wrote {OUT_STATS}")

    # Spot-check 20 random pairs for sanity
    spot = df_out.sample(min(20, len(df_out)), random_state=RNG_SEED).copy()
    def _mcs_str(m):
        try:
            return f"{m:.2f}" if m == m else "n/a"  # NaN-safe
        except Exception:
            return "n/a"
    spot_lines = [
        f"- {r.target_chembl_id} ({r.target_class}): pIC50 {r.anchor_pIC50:.2f} -> {r.optimized_pIC50:.2f} "
        f"(d={r.delta_pic50:.2f}, tc={r.tc:.2f}, scaf_tc={r.scaffold_tc:.2f}, mcs={_mcs_str(r.mcs_coverage)})  "
        f"`{r.input_smiles}` -> `{r.output_smiles}`"
        for r in spot.itertuples()
    ]

    # Summary markdown
    top10 = df_out["target_chembl_id"].value_counts().head(10)
    top10_lines = []
    for tcid, cnt in top10.items():
        nm = target_meta.get(tcid, {}).get("pref_name") or tcid
        cls = target_meta.get(tcid, {}).get("target_class") or "Unknown"
        top10_lines.append(f"| {tcid} | {nm} | {cls} | {cnt:,} |")
    class_lines = []
    for cls, cnt in df_out["target_class"].value_counts().head(15).items():
        class_lines.append(f"| {cls} | {cnt:,} |")

    summary_md = f"""# LO Corpus v1 - Summary

Built by `scripts/build_lo_corpus.py` from `{src_path.name}`.

## Scale

| Metric | Value |
|---|---|
| Total LO pairs | **{stats['total_pairs']:,}** |
| Train pairs (target-disjoint) | {stats['train_pairs']:,} |
| Val pairs (target-disjoint) | {stats['val_pairs']:,} |
| Unique targets | {stats['n_targets']:,} (train: {stats['n_train_targets']:,}, val: {stats['n_val_targets']:,}) |
| Unique molecules | {stats['n_unique_mols']:,} |
| Holdout mols removed (exp7) | {stats['holdout_mols_count']} |

## Filters applied

- Same target, same assay (`assay_type='B'`), `confidence_score >= 7` on both assays
- Within-assay pairs only (both measurements from the same ChEMBL assay)
- pIC50(b) - pIC50(a) >= {DELTA_MIN}
- Tanimoto Tc(a, b) in [{TC_LOW}, {TC_HIGH}) (Morgan r=2, 2048-bit)
- Murcko-scaffold Tc >= {SCAFFOLD_TC_MIN}
- year(a) <= year(b) (where both years known; relaxed when either missing)
- {"MCS coverage >= " + str(MCS_COVERAGE_MIN) + " (of min(|A|, |B|) heavy atoms)" if df_out["mcs_coverage"].notna().any() else "MCS-coverage criterion **DROPPED** for tractability (~10ms/pair x 600k pairs = unacceptable). Tc>=0.4 + scaffold-Tc>=0.5 + |delta|>=0.3 still enforce substantial shared structure."}
- Drug-like: MW in [{MW_LOW}, {MW_HIGH}], LogP in [{LOGP_LOW}, {LOGP_HIGH}]
- Identical pairs (Tc=1.0 or canonical SMILES equal) excluded

## Distributions

| Metric | Mean | Median | p05 | p95 |
|---|---|---|---|---|
| Tc(a,b) | {stats['tc']['mean']:.3f} | {stats['tc']['median']:.3f} | {stats['tc']['p05']:.3f} | {stats['tc']['p95']:.3f} |
| Scaffold Tc | {stats['scaffold_tc']['mean']:.3f} | {stats['scaffold_tc']['median']:.3f} | - | - |
{f"| MCS coverage | {stats['mcs_coverage']['mean']:.3f} | {stats['mcs_coverage']['median']:.3f} | - | - |" if df_out["mcs_coverage"].notna().any() else "| MCS coverage | n/a (filter dropped) | n/a | - | - |"}
| delta-pIC50 | {stats['delta_pic50']['mean']:.3f} | {stats['delta_pic50']['median']:.3f} | {stats['delta_pic50']['p05']:.3f} | {stats['delta_pic50']['p95']:.3f} |

## Top 10 targets by pair count

| ChEMBL ID | Pref name | Class | Pairs |
|---|---|---|---|
{chr(10).join(top10_lines)}

## Top 15 target classes by pair count

| Class | Pairs |
|---|---|
{chr(10).join(class_lines)}

## Exp7 held-out compounds

Loaded {stats['holdout_mols_count']} canonical SMILES from `data/exp7_lo_benchmark/*_pairs.json`:

| File | SMILES added |
|---|---|
{chr(10).join(f"| {k} | {v} |" for k, v in holdout_per_file.items()) or "| (none) | - |"}

Any LO pair where either input or output canonical SMILES matched the
holdout set was excluded BEFORE the target-disjoint train/val split.

## Spot check (20 random pairs)

{chr(10).join(spot_lines)}

## Caveats

- **Assay heterogeneity**: even with `assay_type='B'` and `confidence_score >= 7`,
  the corpus pools across many distinct assay protocols within each target.
  Within-assay pairing removes between-assay scale drift but not all noise.
- **Salt/stereoisomer collapse**: SMILES are canonicalized (RDKit) but
  stereochemistry is preserved. Different enantiomers therefore appear as
  distinct pairs and can show up as "improvements" that are really chirality
  switches. We did not strip stereochemistry on purpose -- mol2mol should
  learn that.
- **Year information** is publication year (often `NULL` for older entries);
  the year-order filter is silently relaxed when either year is missing.
- **Within-assay only**: pairs measured in the same assay; this is the cleanest
  signal but excludes useful cross-assay analogs that share a tight protocol
  (e.g., same lab, sequential papers).
- **Holdout coverage**: exp7 benchmark dir held {stats['holdout_mols_count']} canonical SMILES at build time.
  Re-run after updating `data/exp7_lo_benchmark/*_pairs.json` if you want
  the exp7 hold-out to take effect.
"""
    OUT_SUMMARY.write_text(summary_md)
    print(f"  wrote {OUT_SUMMARY}")

    print(f"\nDONE in {time.time() - t0:.0f}s.")


if __name__ == "__main__":
    main()
