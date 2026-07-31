"""Enrich tier4_scored CSVs with pubTc panel + xTB warhead descriptors.

For each tier4_scored CSV:
  1. Compute the 5 pubTc columns (max_pubTc, mean_pubTc, median_pubTc,
     top10_mean_pubTc, closest_lead) against the 300-lead v3 panel.
  2. For rows with warhead_intact == True (acrylamide), compute xTB
     descriptors (LUMO_eV, HOMO_eV, gap_eV, omega_eV, q_Cb,
     fukui_plus_Cb, pred_log_k2_GSH).
  3. Save enriched CSV back to the same path.

Parallelised via multiprocessing.Pool. Designed for the ai-chem 32-CPU box.

Usage:
    python experiments/enrich_tier4_pubtc_xtb.py \
        --cohort_csv data/tier4_scored/foo_scored.csv \
        --workers 30
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "experiments"))

from pubtc_panel_v3 import PANEL_SMILES  # type: ignore
from xtb_warhead_electrophilicity import (  # type: ignore
    ACRYLAMIDE_SMARTS,
    compute_warhead_descriptors,
)

# ── PubTc worker globals (rehydrated in each worker via init_worker) ─────────
_PANEL_FPS: list = []
_PANEL_NAMES: list[str] = []


def build_panel_fps_bytes() -> tuple[list[bytes], list[str]]:
    fps_bytes, names = [], []
    for name, smi in PANEL_SMILES.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            raise RuntimeError(f"Panel SMILES failed to parse: {name}")
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        fps_bytes.append(DataStructs.BitVectToBinaryText(fp))
        names.append(name)
    return fps_bytes, names


def pubtc_init_worker(panel_bytes: list[bytes], panel_names: list[str]) -> None:
    global _PANEL_FPS, _PANEL_NAMES
    _PANEL_FPS = [DataStructs.CreateFromBinaryText(b) for b in panel_bytes]
    _PANEL_NAMES = list(panel_names)


def pubtc_worker(chunk: list[tuple[int, str]]) -> list[tuple]:
    out = []
    K = 10
    for row_id, smi in chunk:
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            out.append((row_id, np.nan, np.nan, np.nan, np.nan, ""))
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append((row_id, np.nan, np.nan, np.nan, np.nan, ""))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        sims = np.array(
            DataStructs.BulkTanimotoSimilarity(fp, _PANEL_FPS), dtype=np.float32
        )
        k = min(K, len(sims))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_vals = sims[top_idx]
        out.append((
            int(row_id),
            float(sims.max()),
            float(np.median(sims)),
            float(sims.mean()),
            float(top_vals.mean()),
            _PANEL_NAMES[int(np.argmax(sims))],
        ))
    return out


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def run_pubtc(df: pd.DataFrame, workers: int, chunk_size: int = 500) -> pd.DataFrame:
    """Add pubTc columns to df (in place via merge). Returns enriched df."""
    panel_bytes, panel_names = build_panel_fps_bytes()
    print(f"[pubtc] Built {len(panel_bytes)} panel FPs", flush=True)

    pairs = list(zip(df.index.tolist(), df["smiles"].tolist()))
    chunks = list(chunked(pairs, chunk_size))
    print(f"[pubtc] {len(pairs):,} rows in {len(chunks):,} chunks of {chunk_size}", flush=True)

    t0 = time.time()
    results: list[tuple] = []
    with mp.Pool(workers, initializer=pubtc_init_worker, initargs=(panel_bytes, panel_names)) as pool:
        for i, chunk_out in enumerate(pool.imap_unordered(pubtc_worker, chunks, chunksize=1)):
            results.extend(chunk_out)
            if (i + 1) % 20 == 0 or (i + 1) == len(chunks):
                el = time.time() - t0
                rate = len(results) / el if el > 0 else 0.0
                eta = (len(pairs) - len(results)) / rate if rate > 0 else 0.0
                print(
                    f"[pubtc]   {len(results):,}/{len(pairs):,}  "
                    f"rate={rate:.0f}/s  eta={eta/60:.1f}m",
                    flush=True,
                )
    el = time.time() - t0
    print(f"[pubtc] done in {el:.1f}s ({len(results)/el:.0f}/s)", flush=True)

    pubtc_df = pd.DataFrame(
        results,
        columns=[
            "_idx_pubtc",
            "max_pubTc",
            "median_pubTc",
            "mean_pubTc",
            "top10_mean_pubTc",
            "closest_lead",
        ],
    ).set_index("_idx_pubtc")
    pubtc_df.index.name = None
    # Reorder via reindex
    pubtc_df = pubtc_df.reindex(df.index)
    for c in pubtc_df.columns:
        df[c] = pubtc_df[c].values
    return df


# ── xTB worker ───────────────────────────────────────────────────────────────
def xtb_worker(job: tuple[int, str]) -> dict:
    idx, smi = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")
    os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
    os.environ.setdefault("XTB_NUM_THREADS", "1")
    rec: dict = {
        "_idx": idx,
        "LUMO_eV": np.nan,
        "HOMO_eV": np.nan,
        "gap_eV": np.nan,
        "omega_eV": np.nan,
        "q_Cb": np.nan,
        "fukui_plus_Cb": np.nan,
        "pred_log_k2_GSH": np.nan,
        "xtb_status": "skipped",
    }
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return rec
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        rec["xtb_status"] = "parse_fail"
        return rec
    if not mol.HasSubstructMatch(ACRYLAMIDE_SMARTS):
        rec["xtb_status"] = "no_acrylamide"
        return rec
    try:
        out = compute_warhead_descriptors(smi)
    except Exception as e:
        rec["xtb_status"] = f"err:{type(e).__name__}"
        return rec
    if int(out.get("success_flag", 0)) == 1:
        for k in ("LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb",
                  "fukui_plus_Cb", "pred_log_k2_GSH"):
            v = out.get(k, np.nan)
            try:
                rec[k] = float(v)
            except Exception:
                rec[k] = np.nan
        rec["xtb_status"] = "ok"
    else:
        rec["xtb_status"] = "failed"
    return rec


def run_xtb(df: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Add xTB descriptor columns to df. Only acrylamide-bearing rows get
    real values; the rest are NaN. We dispatch ALL rows so the worker
    classifies internally (cheap RDKit check vs. expensive xtb call)."""
    # Pre-filter to rows where warhead_intact == True for efficiency
    if "warhead_intact" in df.columns:
        mask = df["warhead_intact"].fillna(False).astype(bool)
    else:
        mask = pd.Series([True] * len(df), index=df.index)
    sub = df[mask]
    print(f"[xtb] dispatching {len(sub):,} warhead_intact rows of {len(df):,}", flush=True)

    jobs = list(zip(sub.index.tolist(), sub["smiles"].tolist()))
    if not jobs:
        for c in ("LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb",
                  "fukui_plus_Cb", "pred_log_k2_GSH"):
            df[c] = np.nan
        df["xtb_status"] = "no_jobs"
        return df

    t0 = time.time()
    rows: list[dict] = []
    flush_every = max(200, len(jobs) // 40)
    with mp.Pool(workers) as pool:
        for i, rec in enumerate(pool.imap_unordered(xtb_worker, jobs, chunksize=1), 1):
            rows.append(rec)
            if i % flush_every == 0 or i == len(jobs):
                el = time.time() - t0
                rate = i / el if el > 0 else 0.0
                eta = (len(jobs) - i) / rate if rate > 0 else 0.0
                n_ok = sum(1 for r in rows if r["xtb_status"] == "ok")
                print(
                    f"[xtb]   {i:,}/{len(jobs):,}  ok={n_ok:,}  rate={rate:.1f}/s  "
                    f"eta={eta/60:.1f}m",
                    flush=True,
                )

    xtb_df = pd.DataFrame(rows).set_index("_idx")
    xtb_df = xtb_df.reindex(df.index)
    for c in ("LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV", "q_Cb",
              "fukui_plus_Cb", "pred_log_k2_GSH", "xtb_status"):
        df[c] = xtb_df[c].values
    el = time.time() - t0
    n_ok = int((df["xtb_status"] == "ok").sum())
    print(f"[xtb] done in {el/60:.1f}m  ok={n_ok:,}/{len(jobs):,}", flush=True)
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cohort_csv", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=30)
    ap.add_argument("--skip_pubtc", action="store_true")
    ap.add_argument("--skip_xtb", action="store_true")
    ap.add_argument("--xtb_workers", type=int, default=0,
                    help="Override worker count for xtb (default = --workers)")
    args = ap.parse_args()

    csv_path = args.cohort_csv
    if not csv_path.exists():
        raise SystemExit(f"Missing: {csv_path}")

    print(f"\n=== enrich {csv_path.name} ===", flush=True)
    df = pd.read_csv(csv_path)
    print(f"  loaded {len(df):,} rows × {len(df.columns)} cols", flush=True)
    if "smiles" not in df.columns:
        raise SystemExit("Need 'smiles' column")

    t_start = time.time()
    if not args.skip_pubtc:
        df = run_pubtc(df, workers=args.workers)
    if not args.skip_xtb:
        xw = args.xtb_workers or args.workers
        df = run_xtb(df, workers=xw)

    df.to_csv(csv_path, index=False)
    el = time.time() - t_start
    print(f"\n[summary] {csv_path.name}: enriched in {el/60:.1f}m  → {csv_path}", flush=True)

    # QA report
    new_cols = ["max_pubTc", "median_pubTc", "mean_pubTc", "top10_mean_pubTc",
                "closest_lead", "LUMO_eV", "HOMO_eV", "gap_eV", "omega_eV",
                "q_Cb", "fukui_plus_Cb", "pred_log_k2_GSH"]
    print("[summary] new columns non-NaN fraction:")
    for c in new_cols:
        if c in df.columns:
            if c == "closest_lead":
                frac = (df[c].astype(str).str.len() > 0).mean()
            else:
                frac = df[c].notna().mean()
            print(f"   {c:24s}  {100*frac:5.1f}%")
    if "warhead_intact" in df.columns:
        n_acryl = int(df["warhead_intact"].fillna(False).astype(bool).sum())
        if n_acryl > 0:
            sub = df[df["warhead_intact"].fillna(False).astype(bool)]
            for c in ("LUMO_eV", "pred_log_k2_GSH"):
                if c in sub.columns:
                    frac = sub[c].notna().mean()
                    print(f"   {c} (acryl-only)         {100*frac:5.1f}% of {n_acryl:,}")


if __name__ == "__main__":
    main()
