"""Compute pubTc panel similarity scores against the v3 300-lead panel.

Replaces v1 (11 leads) and v2 (30 leads). For each of ~620K candidate molecules
in ``all_methods_bulk_scored_v4.csv`` we compute the Tanimoto similarity
(Morgan radius=2, 2048-bit) to every one of the 300 published kinase covalent
inhibitor leads, then derive the following metrics:

    max_pubTc          : maximum Tanimoto across the 300 leads
    mean_pubTc         : mean Tanimoto
    median_pubTc       : median Tanimoto
    top10_mean_pubTc   : mean of the 10 highest Tanimotos (NEW vs v1/v2 top3)
    closest_lead       : drug_name of the highest-Tanimoto lead (string)

Why ``top10_mean_pubTc`` instead of ``top3_mean_pubTc``?
    With a 300-lead panel the broader top-K mean smooths over single-scaffold
    quirks; top-10 is ~3% of the panel and rewards molecules near a cluster of
    leads, not just a lucky single match.

Output: ``data/paper_evaluation/pubtc_scores_v3.csv``

This script is the v3 replacement; the v1 (``pubtc_scores.csv``) and v2
(``pubtc_scores_v2.csv``) files are dropped from the dashboard backend per the
user's instruction.
"""

from __future__ import annotations

import argparse
import multiprocessing as mp
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, DataStructs, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_INPUT = (
    PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "data" / "paper_evaluation" / "pubtc_scores_v3.csv"


# ── Worker globals (set in init_worker) ──────────────────────────────────────
_PANEL_FPS: list = []
_PANEL_NAMES: list[str] = []


def build_panel_fps(panel_smiles: dict[str, str]) -> tuple[list, list[str]]:
    """Build Morgan FPs for every panel member. Raises on any parse failure.

    The panel module itself already hard-raises at import on bad SMILES; the
    redundant check here is defensive in case someone hot-edits the SMILES
    dict before calling this function.
    """
    fps, names, bad = [], [], []
    for name, smi in panel_smiles.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(name)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
        names.append(name)
    if bad:
        raise RuntimeError(
            f"Panel SMILES failed to parse: {bad}. "
            "Fix experiments/pubtc_panel_v3.py before running."
        )
    return fps, names


def init_worker(panel_fps_bytes: list[bytes], panel_names: list[str]) -> None:
    """Pool initializer: rehydrate panel FPs from serialized bytes + names."""
    global _PANEL_FPS, _PANEL_NAMES
    _PANEL_FPS = [DataStructs.CreateFromBinaryText(b) for b in panel_fps_bytes]
    _PANEL_NAMES = list(panel_names)


def worker_score_chunk(args: tuple) -> list[tuple]:
    """Score a chunk of (row_id, smiles) tuples → list of result tuples."""
    chunk = args
    out = []
    K = 10  # top-10 mean
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
        # argmax & top-K (avoid full sort; np.partition is O(n))
        k = min(K, len(sims))
        top_idx = np.argpartition(-sims, k - 1)[:k]
        top_vals = sims[top_idx]
        # mean over top-K is order-agnostic, so no need to sort top_vals
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


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional: only score the first N rows (for testing)",
    )
    args = p.parse_args()

    # Import the v3 panel module — verifies SMILES at import; hard-raises on
    # any parse failure / coverage shortfall.
    sys.path.insert(0, str(PROJECT_ROOT / "experiments"))
    from pubtc_panel_v3 import PANEL_SMILES, PANEL_V3  # type: ignore

    print(f"Loaded v3 panel: {len(PANEL_V3)} members")
    print(f"Output → {args.output}")

    panel_fps, panel_names = build_panel_fps(PANEL_SMILES)
    print(f"Built {len(panel_fps)} Morgan(r=2,2048) panel FPs")

    print(f"Loading {args.input} ...")
    df = pd.read_csv(args.input, usecols=lambda c: c in ("row_id", "smiles"))
    if "row_id" not in df.columns:
        df = df.reset_index(drop=True)
        df["row_id"] = df.index
    print(f"  Loaded {len(df):,} rows")
    if args.limit:
        df = df.head(args.limit).copy()
        print(f"  Truncated to first {len(df):,} rows (--limit)")

    pairs = list(zip(df["row_id"].tolist(), df["smiles"].tolist()))
    chunks = list(chunked(pairs, args.chunk_size))
    print(f"  {len(chunks):,} chunks of size {args.chunk_size}")

    # Serialize panel FPs for the workers (bytes are stable across versions).
    panel_bytes = [DataStructs.BitVectToBinaryText(p) for p in panel_fps]

    t0 = time.time()
    results: list[tuple] = []
    with mp.Pool(
        args.workers, initializer=init_worker, initargs=(panel_bytes, panel_names)
    ) as pool:
        for i, chunk_out in enumerate(
            pool.imap_unordered(worker_score_chunk, chunks, chunksize=1)
        ):
            results.extend(chunk_out)
            if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
                elapsed = time.time() - t0
                done = len(results)
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(pairs) - done) / rate if rate > 0 else float("inf")
                print(
                    f"  chunk {i+1:,}/{len(chunks):,}  rows={done:,}/{len(pairs):,}  "
                    f"rate={rate:.0f}/s  eta={eta/60:.1f} min",
                    flush=True,
                )
    print(f"Scored {len(results):,} rows in {time.time()-t0:.1f}s")

    out_df = (
        pd.DataFrame(
            results,
            columns=[
                "row_id",
                "max_pubTc",
                "median_pubTc",
                "mean_pubTc",
                "top10_mean_pubTc",
                "closest_lead",
            ],
        )
        .sort_values("row_id")
        .reset_index(drop=True)
    )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df):,} rows → {args.output}")

    # Distribution stats so the user sees something useful in the log
    print("\n── Distribution stats ──")
    for col in ("max_pubTc", "median_pubTc", "mean_pubTc", "top10_mean_pubTc"):
        s = out_df[col].dropna()
        if not len(s):
            continue
        qs = s.quantile([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]).values
        print(
            f"  {col:18s}  min={qs[0]:.3f} p10={qs[1]:.3f} p25={qs[2]:.3f} "
            f"p50={qs[3]:.3f} p75={qs[4]:.3f} p90={qs[5]:.3f} max={qs[6]:.3f}"
        )
    above = (out_df["max_pubTc"] >= 0.5).sum()
    print(f"  rows with max_pubTc ≥ 0.5 : {above:,} ({100.0*above/len(out_df):.2f}%)")

    print("\n── closest_lead histogram (top 30) ──")
    print(out_df["closest_lead"].value_counts().head(30).to_string())


if __name__ == "__main__":
    main()
