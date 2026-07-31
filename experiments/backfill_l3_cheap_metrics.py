"""
Backfill 5 cheap metrics on the full 756K backend DF for the dashboard's
filter-cascade evaluation.

Five target columns (all keyed on the post-filter `row_id = DF.index`):

  1. SAScore                          — Ertl 2009 synthetic accessibility
  2. LLE_method                       — pIC50_method - LogP
  3. LE_method                        — 1.4 * pIC50_method / HeavyAtoms
  4. max_pubTc                        — max Tanimoto vs the 300-lead panel
  5. rdkit_strain_posefree_kcal_mol   — pose-free constitutional strain

Workflow:
  - Build the same DF the backend builds (legacy v4 + Tier 4 concat, drop
    methods + warhead-modified + disconnected SMILES, reset_index, row_id =
    index).
  - For each metric: identify rows missing the value; compute; write into a
    single unified CSV.

The backend currently merges some of these by `row_id`, but the cached CSVs
were written against pre-filter indices that no longer line up.  This script
recomputes everything from the rebuilt DF so the output CSV is keyed by the
NEW row_id and merging is safe.

Output:
  data/paper_evaluation/l3_cheap_backfill.csv
    columns: row_id, smiles, SAScore, LLE_method, LE_method, max_pubTc,
             rdkit_strain_posefree_kcal_mol

Usage:
    python experiments/backfill_l3_cheap_metrics.py \
        [--workers 8] [--skip rdkit_strain] [--limit 1000]
"""

from __future__ import annotations

import argparse
import gc
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
sys.path.insert(0, str(PROJECT_ROOT / "external" / "DiffSBDD" / "analysis" / "SA_Score"))

from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

OUT_PATH = PROJECT_ROOT / "data" / "paper_evaluation" / "l3_cheap_backfill.csv"
V4_PATH = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
T4_PATH = PROJECT_ROOT / "results" / "paper_evaluation" / "tier4_overnight_bulk.csv"

PUBTC_V3_CACHE = PROJECT_ROOT / "data" / "paper_evaluation" / "pubtc_scores_v3.csv"
STRAIN_CACHE = PROJECT_ROOT / "data" / "paper_evaluation" / "rdkit_strain_posefree.csv"


# ────────────────────────────────────────────────────────────────────────────
# DF rebuild (mirrors backend.py)
# ────────────────────────────────────────────────────────────────────────────

def build_backend_df() -> pd.DataFrame:
    """Exact replica of backend.py's DF construction up to row_id assignment."""
    print(f"[build] reading {V4_PATH.name} ...")
    DF = pd.read_csv(V4_PATH, low_memory=False)

    print(f"[build] reading {T4_PATH.name} ...")
    DF_T4 = pd.read_csv(T4_PATH, low_memory=False)
    for c in DF.columns:
        if c not in DF_T4.columns:
            DF_T4[c] = pd.NA
    for c in DF_T4.columns:
        if c not in DF.columns:
            DF[c] = pd.NA
    DF_T4 = DF_T4[DF.columns]
    DF = pd.concat([DF, DF_T4], ignore_index=True)
    del DF_T4
    gc.collect()

    DF = DF.reset_index(drop=True)
    DF["row_id"] = DF.index

    DF["method"] = DF["method"].replace({
        "Tier 1 — Med-Chem Playbook (rule-based)": "Medchem Rules",
        "Tier 1.5 — Warhead Controls + Med-Chem Tricks": "Medchem Rules",
        "Tier 2 — Fragment Replacement (curated 204)": "Amine Replacements",
        "Tier 2 SCALED — Fragment Replacement (498K)": "Amine Replacements",
        "Tier 2 SCALED — Fragment Replacement (498K from ChEMBL 35)": "Amine Replacements",
    })
    DROPPED = [
        "Tier 4 — De Novo unconstrained",
        "Tier 4 — Mol2Mol unconstrained",
        "Method A — De Novo FiLMDelta-driven",
        "Method B — Mol2Mol FiLMDelta-driven",
    ]
    DF = DF[~DF["method"].isin(DROPPED)].reset_index(drop=True)
    DF = DF[DF["warhead_intact"] == True].reset_index(drop=True)
    DF = DF[~DF["smiles"].astype(str).str.contains(".", regex=False, na=False)].reset_index(drop=True)
    DF["row_id"] = DF.index
    print(f"[build] final DF: {len(DF):,} rows  (row_id 0..{len(DF)-1})")
    return DF


# ────────────────────────────────────────────────────────────────────────────
# Stream 1 — SAScore
# ────────────────────────────────────────────────────────────────────────────

# Worker globals (set in init_worker)
_SASCORER = None


def _sa_init_worker():
    global _SASCORER
    import sascorer  # noqa: E402
    _SASCORER = sascorer


def _sa_worker(chunk):
    """chunk = list[(row_id, smiles)] -> list[(row_id, sascore)]."""
    out = []
    for rid, smi in chunk:
        if not isinstance(smi, str) or not smi:
            out.append((rid, np.nan))
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append((rid, np.nan))
            continue
        try:
            out.append((rid, float(_SASCORER.calculateScore(m))))
        except Exception:
            out.append((rid, np.nan))
    return out


def compute_sascore(df_missing: pd.DataFrame, workers: int) -> pd.Series:
    """Returns a Series indexed by row_id (only missing rows)."""
    if len(df_missing) == 0:
        return pd.Series(dtype=float, name="SAScore")
    print(f"[SAScore] {len(df_missing):,} rows to compute, {workers} workers")
    pairs = list(zip(df_missing["row_id"].tolist(), df_missing["smiles"].tolist()))
    chunk_size = 2000
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_sa_init_worker) as pool:
        for i, chunk_out in enumerate(pool.imap_unordered(_sa_worker, chunks, chunksize=1)):
            results.extend(chunk_out)
            if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
                done = len(results)
                rate = done / (time.time() - t0)
                eta = (len(pairs) - done) / rate if rate > 0 else 0
                print(f"  [SAScore] chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  "
                      f"rate={rate:.0f}/s  eta={eta:.0f}s")
    rids, vals = zip(*results)
    s = pd.Series(vals, index=rids, name="SAScore")
    n_ok = s.notna().sum()
    print(f"[SAScore] done: {n_ok:,}/{len(s):,} success in {time.time()-t0:.1f}s "
          f"(mean={np.nanmean(s):.2f}, min={np.nanmin(s):.2f}, max={np.nanmax(s):.2f})")
    return s


# ────────────────────────────────────────────────────────────────────────────
# Stream 2 + 3 — LLE_method & LE_method  (vectorized)
# ────────────────────────────────────────────────────────────────────────────

def compute_lle_le(df: pd.DataFrame) -> pd.DataFrame:
    """Vectorized — returns DataFrame indexed by row_id with LLE_method, LE_method."""
    print(f"[LLE/LE] computing on {len(df):,} rows (vectorized)")
    t0 = time.time()
    pic = pd.to_numeric(df["pIC50_method"], errors="coerce")
    logp = pd.to_numeric(df["LogP"], errors="coerce")
    ha = pd.to_numeric(df["HeavyAtoms"], errors="coerce")

    lle = pic - logp
    le = 1.4 * pic / ha.where(ha > 0)

    out = pd.DataFrame({
        "LLE_method": lle.values,
        "LE_method": le.values,
    }, index=df["row_id"].values)
    out.index.name = "row_id"
    n_lle = out["LLE_method"].notna().sum()
    n_le = out["LE_method"].notna().sum()
    print(f"[LLE/LE] LLE non-null: {n_lle:,}  LE non-null: {n_le:,}  in {time.time()-t0:.2f}s")
    print(f"  LLE  mean={out['LLE_method'].mean():.3f}  min={out['LLE_method'].min():.3f}  max={out['LLE_method'].max():.3f}")
    print(f"  LE   mean={out['LE_method'].mean():.3f}  min={out['LE_method'].min():.3f}  max={out['LE_method'].max():.3f}")
    return out


# ────────────────────────────────────────────────────────────────────────────
# Stream 4 — max_pubTc (Tanimoto vs 300-lead panel)
# ────────────────────────────────────────────────────────────────────────────

_PT_PANEL_FPS = []


def _pubtc_init_worker(panel_fps_bytes):
    global _PT_PANEL_FPS
    _PT_PANEL_FPS = [DataStructs.CreateFromBinaryText(b) for b in panel_fps_bytes]


def _pubtc_worker(chunk):
    out = []
    for rid, smi in chunk:
        if not isinstance(smi, str) or not smi:
            out.append((rid, np.nan))
            continue
        m = Chem.MolFromSmiles(smi)
        if m is None:
            out.append((rid, np.nan))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048)
        sims = DataStructs.BulkTanimotoSimilarity(fp, _PT_PANEL_FPS)
        out.append((rid, float(max(sims))))
    return out


def compute_max_pubtc(df_missing: pd.DataFrame, workers: int) -> pd.Series:
    if len(df_missing) == 0:
        return pd.Series(dtype=float, name="max_pubTc")
    print(f"[pubTc] importing v3 panel ...")
    from pubtc_panel_v3 import PANEL_SMILES  # type: ignore
    panel_fps = []
    for name, smi in PANEL_SMILES.items():
        m = Chem.MolFromSmiles(smi)
        if m is None:
            raise RuntimeError(f"panel SMILES failed: {name}")
        panel_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, radius=2, nBits=2048))
    print(f"[pubTc] {len(panel_fps)} panel FPs, {len(df_missing):,} candidate rows, {workers} workers")
    panel_bytes = [DataStructs.BitVectToBinaryText(p) for p in panel_fps]

    pairs = list(zip(df_missing["row_id"].tolist(), df_missing["smiles"].tolist()))
    chunk_size = 1000
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_pubtc_init_worker, initargs=(panel_bytes,)) as pool:
        for i, chunk_out in enumerate(pool.imap_unordered(_pubtc_worker, chunks, chunksize=1)):
            results.extend(chunk_out)
            if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
                done = len(results)
                rate = done / (time.time() - t0)
                eta = (len(pairs) - done) / rate if rate > 0 else 0
                print(f"  [pubTc] chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  "
                      f"rate={rate:.0f}/s  eta={eta:.0f}s")
    rids, vals = zip(*results)
    s = pd.Series(vals, index=rids, name="max_pubTc")
    n_ok = s.notna().sum()
    print(f"[pubTc] done: {n_ok:,}/{len(s):,} success in {time.time()-t0:.1f}s "
          f"(mean={np.nanmean(s):.3f}, min={np.nanmin(s):.3f}, max={np.nanmax(s):.3f})")
    return s


# ────────────────────────────────────────────────────────────────────────────
# Stream 5 — rdkit_strain_posefree_kcal_mol
# ────────────────────────────────────────────────────────────────────────────

def _strain_init_worker():
    # Lazy import in worker; src is on sys.path via PROJECT_ROOT
    global _SCORE_POSEFREE
    from src.utils.rdkit_strain import score_posefree  # noqa: E402
    _SCORE_POSEFREE = score_posefree


def _strain_worker(chunk):
    out = []
    for rid, smi in chunk:
        if not isinstance(smi, str) or not smi:
            out.append((rid, np.nan))
            continue
        try:
            res = _SCORE_POSEFREE(smi, num_conf_free=2, seed=42, free_max_its=100)
            if res.get("success_flag") == 1:
                out.append((rid, float(res["strain_kcal_mol"])))
            else:
                out.append((rid, np.nan))
        except Exception:
            out.append((rid, np.nan))
    return out


def compute_strain(df_missing: pd.DataFrame, workers: int) -> pd.Series:
    if len(df_missing) == 0:
        return pd.Series(dtype=float, name="rdkit_strain_posefree_kcal_mol")
    print(f"[strain] {len(df_missing):,} rows to compute, {workers} workers")
    pairs = list(zip(df_missing["row_id"].tolist(), df_missing["smiles"].tolist()))
    chunk_size = 200
    chunks = [pairs[i:i + chunk_size] for i in range(0, len(pairs), chunk_size)]

    t0 = time.time()
    results = []
    ctx = mp.get_context("spawn")
    with ctx.Pool(workers, initializer=_strain_init_worker) as pool:
        for i, chunk_out in enumerate(pool.imap_unordered(_strain_worker, chunks, chunksize=1)):
            results.extend(chunk_out)
            if (i + 1) % 10 == 0 or (i + 1) == len(chunks):
                done = len(results)
                rate = done / (time.time() - t0)
                eta = (len(pairs) - done) / rate if rate > 0 else 0
                print(f"  [strain] chunk {i+1}/{len(chunks)}  rows={done:,}/{len(pairs):,}  "
                      f"rate={rate:.1f}/s  eta={eta:.0f}s")
    rids, vals = zip(*results)
    s = pd.Series(vals, index=rids, name="rdkit_strain_posefree_kcal_mol")
    n_ok = s.notna().sum()
    print(f"[strain] done: {n_ok:,}/{len(s):,} success in {time.time()-t0:.1f}s "
          f"(mean={np.nanmean(s):.2f}, min={np.nanmin(s):.2f}, max={np.nanmax(s):.2f})")
    return s


# ────────────────────────────────────────────────────────────────────────────
# Backfill from cached CSVs by canonical SMILES match
# ────────────────────────────────────────────────────────────────────────────

def fill_from_cached_v4(DF: pd.DataFrame, out: pd.DataFrame) -> None:
    """Lift legacy SAScore from v4 source (already in DF directly)."""
    n_pre = out["SAScore"].notna().sum()
    out.loc[DF["SAScore"].notna(), "SAScore"] = DF.loc[DF["SAScore"].notna(), "SAScore"].values
    n_post = out["SAScore"].notna().sum()
    print(f"[SAScore cache] lifted from v4 column: +{n_post - n_pre:,} → {n_post:,} filled")


def fill_pubtc_strain_from_smiles(DF: pd.DataFrame, out: pd.DataFrame) -> None:
    """Use canonical SMILES to lift cached pubTc/strain values where possible.

    The cached CSVs are indexed by stale row_ids that no longer align with
    the post-filter DF. Build SMILES→value maps from the cached CSVs (using
    the corresponding pre-filter SMILES) and re-lookup on the new DF.

    For pubTc: cache is built against v4 row_ids 0..620714. Map row_id → smiles
    via the v4 CSV directly. Then keyed lookup.
    """
    # ── pubTc v3 ──
    if PUBTC_V3_CACHE.exists():
        t0 = time.time()
        pt = pd.read_csv(PUBTC_V3_CACHE, usecols=["row_id", "max_pubTc"])
        # Map the cache row_id → v4 SMILES (pre-filter)
        v4_smi = pd.read_csv(V4_PATH, usecols=["row_id", "smiles"])
        v4_smi = v4_smi.rename(columns={"row_id": "v4_row_id"})
        pt = pt.rename(columns={"row_id": "v4_row_id"})
        smi_to_pubtc = pt.merge(v4_smi, on="v4_row_id", how="left")
        smi_to_pubtc = smi_to_pubtc.dropna(subset=["smiles", "max_pubTc"]).drop_duplicates("smiles")
        lookup = dict(zip(smi_to_pubtc["smiles"], smi_to_pubtc["max_pubTc"]))
        # Apply on current DF
        mapped = DF["smiles"].map(lookup)
        n_pre = out["max_pubTc"].notna().sum()
        out.loc[mapped.notna(), "max_pubTc"] = mapped[mapped.notna()].values
        n_post = out["max_pubTc"].notna().sum()
        print(f"[pubTc cache] lifted by SMILES: +{n_post - n_pre:,} → {n_post:,} filled "
              f"({time.time()-t0:.1f}s)")
        del pt, v4_smi, smi_to_pubtc, lookup, mapped
        gc.collect()

    # ── rdkit_strain_posefree ──
    if STRAIN_CACHE.exists():
        t0 = time.time()
        st = pd.read_csv(STRAIN_CACHE, usecols=["row_id", "strain_kcal_mol", "success_flag"])
        st = st[st["success_flag"] == 1][["row_id", "strain_kcal_mol"]]
        v4_smi = pd.read_csv(V4_PATH, usecols=["row_id", "smiles"])
        v4_smi = v4_smi.rename(columns={"row_id": "v4_row_id"})
        st = st.rename(columns={"row_id": "v4_row_id"})
        smi_to_strain = st.merge(v4_smi, on="v4_row_id", how="left")
        smi_to_strain = smi_to_strain.dropna(subset=["smiles", "strain_kcal_mol"]).drop_duplicates("smiles")
        lookup = dict(zip(smi_to_strain["smiles"], smi_to_strain["strain_kcal_mol"]))
        mapped = DF["smiles"].map(lookup)
        n_pre = out["rdkit_strain_posefree_kcal_mol"].notna().sum()
        out.loc[mapped.notna(), "rdkit_strain_posefree_kcal_mol"] = mapped[mapped.notna()].values
        n_post = out["rdkit_strain_posefree_kcal_mol"].notna().sum()
        print(f"[strain cache] lifted by SMILES: +{n_post - n_pre:,} → {n_post:,} filled "
              f"({time.time()-t0:.1f}s)")
        del st, v4_smi, smi_to_strain, lookup, mapped
        gc.collect()


# ────────────────────────────────────────────────────────────────────────────
# main
# ────────────────────────────────────────────────────────────────────────────

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--out", type=Path, default=OUT_PATH)
    ap.add_argument("--limit", type=int, default=None,
                    help="Debug: limit each compute stream to first N missing rows")
    ap.add_argument("--skip", nargs="*", default=[],
                    choices=["sascore", "lle_le", "pubtc", "strain"],
                    help="Skip selected streams (debug)")
    args = ap.parse_args()

    DF = build_backend_df()

    # Initialize output container — one row per DF row_id
    out = pd.DataFrame({
        "row_id": DF["row_id"].values,
        "smiles": DF["smiles"].values,
        "SAScore": np.nan,
        "LLE_method": np.nan,
        "LE_method": np.nan,
        "max_pubTc": np.nan,
        "rdkit_strain_posefree_kcal_mol": np.nan,
    })

    # ── seed from existing columns / caches by SMILES ──
    fill_from_cached_v4(DF, out)
    fill_pubtc_strain_from_smiles(DF, out)

    # ── Stream 2+3: LLE/LE (vectorized, always rebuild) ──
    if "lle_le" not in args.skip:
        ll = compute_lle_le(DF)
        out["LLE_method"] = ll["LLE_method"].values
        out["LE_method"] = ll["LE_method"].values

    # ── Stream 1: SAScore (missing rows only) ──
    if "sascore" not in args.skip:
        miss = DF[out["SAScore"].isna()].copy()
        if args.limit:
            miss = miss.head(args.limit)
        s = compute_sascore(miss[["row_id", "smiles"]], args.workers)
        out.loc[out["row_id"].isin(s.index), "SAScore"] = (
            out.loc[out["row_id"].isin(s.index), "row_id"].map(s.to_dict())
        )

    # ── Stream 4: max_pubTc (missing rows only) ──
    if "pubtc" not in args.skip:
        miss = DF[out["max_pubTc"].isna()].copy()
        if args.limit:
            miss = miss.head(args.limit)
        s = compute_max_pubtc(miss[["row_id", "smiles"]], args.workers)
        out.loc[out["row_id"].isin(s.index), "max_pubTc"] = (
            out.loc[out["row_id"].isin(s.index), "row_id"].map(s.to_dict())
        )

    # ── Stream 5: rdkit_strain_posefree (missing rows only) ──
    if "strain" not in args.skip:
        miss = DF[out["rdkit_strain_posefree_kcal_mol"].isna()].copy()
        if args.limit:
            miss = miss.head(args.limit)
        s = compute_strain(miss[["row_id", "smiles"]], args.workers)
        out.loc[out["row_id"].isin(s.index), "rdkit_strain_posefree_kcal_mol"] = (
            out.loc[out["row_id"].isin(s.index), "row_id"].map(s.to_dict())
        )

    # ── write ──
    args.out.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(args.out, index=False)
    print()
    print(f"[done] wrote {len(out):,} rows → {args.out}")
    print(f"  coverage:")
    for c in ("SAScore", "LLE_method", "LE_method", "max_pubTc",
              "rdkit_strain_posefree_kcal_mol"):
        n = out[c].notna().sum()
        pct = 100 * n / len(out)
        print(f"    {c:35s}: {n:>8,} ({pct:5.1f}%)  "
              f"mean={out[c].mean():7.3f}  min={out[c].min():7.3f}  max={out[c].max():7.3f}")


if __name__ == "__main__":
    # Prevent BLAS/OpenMP oversubscription
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    mp.freeze_support()
    main()
