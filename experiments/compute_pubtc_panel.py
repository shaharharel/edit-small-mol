"""Compute pubTc panel similarity scores for the dashboard candidates.

Curated panel of 11 published kinase covalent (and 1 generic acrylamide probe)
leads. For each input molecule (~520K rows) we compute the Tanimoto similarity
(Morgan radius=2, 2048-bit) to every panel member, then derive four metrics:

    max_pubTc        : maximum Tanimoto over the 11 panel members
    median_pubTc     : median Tanimoto over the 11 panel members
    mean_pubTc       : mean Tanimoto over the 11 panel members
    top3_mean_pubTc  : mean of the top-3 highest Tanimotos

We also store `closest_lead` — the panel member with the highest Tanimoto for
that row (useful for human inspection of which scaffolds dominate).

Output: data/paper_evaluation/pubtc_scores.csv with columns
    row_id, max_pubTc, median_pubTc, mean_pubTc, top3_mean_pubTc, closest_lead

This is intentionally a stand-alone script: the panel SMILES is curated inline
so the file is self-contained and does not depend on the /tmp scout prototype.
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
DEFAULT_INPUT = PROJECT_ROOT / "results" / "paper_evaluation" / "all_methods_bulk_scored_v4.csv"
# Default output depends on --panel; resolved in main().
DEFAULT_OUTPUT_V1 = PROJECT_ROOT / "data" / "paper_evaluation" / "pubtc_scores.csv"
DEFAULT_OUTPUT_V2 = PROJECT_ROOT / "data" / "paper_evaluation" / "pubtc_scores_v2.csv"


# ── v1: Curated 11-lead panel (+ 1 generic acrylamide probe = 12 total fps) ─
# SMILES from PubChem / ChEMBL canonical structures where available.
PANEL_SMILES_V1: dict[str, str] = {
    # BTK covalent inhibitors (Cys481)
    "ibrutinib":      "C=CC(=O)N1CCC[C@@H](C1)n2c(c3cc(ccc3n2)Oc4ccccc4)c5cnc(nc5)N",
    "acalabrutinib":  "CC#CC(=O)N1CCC[C@@H](C1)n2c(nc3c2ncnc3N)c4ccc(cc4)C(=O)Nc5ccccn5",
    "zanubrutinib":   "C=CC(=O)N1CCC[C@@H]1n2c(c3cc(ccc3n2)Oc4ccc(cc4)F)c5cnc(nc5)N",
    "evobrutinib":    "C=CC(=O)N1CCC(CC1)Oc1nc2[nH]ccc2c(n1)c1ccc(cc1)Oc1ccccc1",
    "spebrutinib":    "C=CC(=O)Nc1ccc2c(c1)ncnc2NCc3ccc(cc3)F",
    # SYK family
    "R406":           "COc1cc(Nc2ncc(F)c(Nc3ccc(C(=O)NC4CCOC4)cc3OC)n2)cc(OC)c1OC",
    "entospletinib":  "CC(C)n1nccc1Nc2ncc(N)c(C#Cc3ccccc3)n2",
    "lanraplenib":    "COc1cc(Nc2ncc3c(n2)n(c(=O)n3c4ccc(cn4)N5CCOCC5)C)ccc1",
    "cerdulatinib":   "CCN1CCN(CC1)c2cnc(c(c2)F)Nc3ncc(c(n3)Nc4cc(F)c(F)cc4)C(C)C",
    "TAK-659":        "Nc1ccc(cc1)C(=O)Nc2c3CCCc3nc4cc(ccc24)C(F)(F)F",
    # JAK3 covalent
    "PF-06651600":    "C=CC(=O)N1CC[C@@H](C1)NC(=O)c2cncc3c2cccc3",
    # Generic acrylamide probe (lower bound / "what does the warhead alone match?")
    "acrylamide_probe":  "C=CC(=O)Nc1ccc(cc1)c2cnc3[nH]ncc3c2",
}

# v2: 30-lead expanded panel — loaded from experiments/pubtc_panel_v2.py
# (verified at module-import time; hard-raises on parse failure / dupes / cap
# violations). Imported lazily inside main() so v1 runs don't need the module.

# These two globals are set by main() based on --panel choice and then read
# by build_panel_fps / worker_score_chunk.
PANEL_SMILES: dict[str, str] = {}
PANEL_NAMES: list[str] = []


# ── Worker globals (set in init_worker, used in worker_score_chunk) ─────────
_PANEL_FPS: list = []  # list of ExplicitBitVect
_PANEL_NAMES: list[str] = []  # parallel list of panel member names


def build_panel_fps() -> list:
    """Build Morgan fingerprints for every panel member.

    Raises if any SMILES fails to parse, so the panel is never silently
    truncated (which would corrupt the `closest_lead` index mapping).
    """
    fps = []
    bad = []
    for name, smi in PANEL_SMILES.items():
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            bad.append(name)
            continue
        fps.append(AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048))
    if bad:
        raise RuntimeError(
            f"Panel SMILES failed to parse: {bad}. Fix inline before running."
        )
    assert len(fps) == len(PANEL_NAMES), (
        f"panel fp/name length mismatch: {len(fps)} vs {len(PANEL_NAMES)}"
    )
    return fps


def init_worker(panel_fps_bytes: list[bytes], panel_names: list[str]) -> None:
    """Pool initializer: rehydrate panel FPs from serialized bytes + names."""
    global _PANEL_FPS, _PANEL_NAMES
    _PANEL_FPS = [DataStructs.CreateFromBinaryText(b) for b in panel_fps_bytes]
    _PANEL_NAMES = list(panel_names)


def worker_score_chunk(args: tuple) -> list[tuple]:
    """Score a chunk of (row_id, smiles) tuples. Returns list of result tuples."""
    chunk = args
    out = []
    for row_id, smi in chunk:
        if smi is None or (isinstance(smi, float) and np.isnan(smi)):
            out.append((row_id, np.nan, np.nan, np.nan, np.nan, ""))
            continue
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            out.append((row_id, np.nan, np.nan, np.nan, np.nan, ""))
            continue
        fp = AllChem.GetMorganFingerprintAsBitVect(mol, radius=2, nBits=2048)
        tcs = np.array(
            [DataStructs.TanimotoSimilarity(fp, p) for p in _PANEL_FPS],
            dtype=np.float32,
        )
        order = np.argsort(-tcs)
        top3_mean = float(tcs[order[: min(3, len(tcs))]].mean())
        out.append((
            int(row_id),
            float(tcs.max()),
            float(np.median(tcs)),
            float(tcs.mean()),
            top3_mean,
            _PANEL_NAMES[int(order[0])],
        ))
    return out


def chunked(seq, n):
    for i in range(0, len(seq), n):
        yield seq[i : i + n]


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    p.add_argument("--output", type=Path, default=None,
                   help="Output CSV path (defaults depend on --panel)")
    p.add_argument("--panel", choices=("v1", "v2"), default="v1",
                   help="Which curated panel to use (v1=11 leads, v2=30 leads)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--chunk_size", type=int, default=1000)
    p.add_argument("--limit", type=int, default=None,
                   help="Optional: only score the first N rows (for testing)")
    args = p.parse_args()

    # Wire the chosen panel into module-level globals (used by build/worker fns)
    global PANEL_SMILES, PANEL_NAMES
    if args.panel == "v1":
        PANEL_SMILES = dict(PANEL_SMILES_V1)
        if args.output is None:
            args.output = DEFAULT_OUTPUT_V1
    else:
        # Import the v2 panel module (verifies SMILES at import; hard-raises on
        # parse failure / duplicates / cap violations).
        from pubtc_panel_v2 import PANEL_SMILES as PANEL_SMILES_V2
        PANEL_SMILES = dict(PANEL_SMILES_V2)
        if args.output is None:
            args.output = DEFAULT_OUTPUT_V2
    PANEL_NAMES = list(PANEL_SMILES.keys())
    print(f"Selected panel: {args.panel}  ({len(PANEL_NAMES)} members)")
    print(f"Output will be written to: {args.output}")

    print(f"Building panel FPs ({len(PANEL_SMILES)} members)...")
    panel_fps = build_panel_fps()
    print(f"  Built {len(panel_fps)} fps; panel order: {PANEL_NAMES}")

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

    # Serialize panel FPs for the workers (bit-vectors aren't directly picklable
    # in a stable cross-version way; bytes are).
    panel_bytes = [DataStructs.BitVectToBinaryText(p) for p in panel_fps]

    t0 = time.time()
    results: list[tuple] = []
    with mp.Pool(args.workers, initializer=init_worker,
                 initargs=(panel_bytes, PANEL_NAMES)) as pool:
        for i, chunk_out in enumerate(pool.imap_unordered(worker_score_chunk, chunks, chunksize=1)):
            results.extend(chunk_out)
            if (i + 1) % 50 == 0 or (i + 1) == len(chunks):
                elapsed = time.time() - t0
                done = len(results)
                rate = done / elapsed if elapsed > 0 else 0.0
                eta = (len(pairs) - done) / rate if rate > 0 else float("inf")
                print(f"  chunk {i+1:,}/{len(chunks):,}  rows={done:,}/{len(pairs):,}  "
                      f"rate={rate:.0f}/s  eta={eta/60:.1f} min", flush=True)
    print(f"Scored {len(results):,} rows in {time.time()-t0:.1f}s")

    out_df = pd.DataFrame(
        results,
        columns=["row_id", "max_pubTc", "median_pubTc", "mean_pubTc",
                 "top3_mean_pubTc", "closest_lead"],
    ).sort_values("row_id").reset_index(drop=True)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    out_df.to_csv(args.output, index=False)
    print(f"Wrote {len(out_df):,} rows -> {args.output}")

    # Brief stats so the user sees something useful in the log
    print("\n── Distribution stats (across all rows) ──")
    for col in ("max_pubTc", "median_pubTc", "mean_pubTc", "top3_mean_pubTc"):
        s = out_df[col].dropna()
        if not len(s):
            continue
        qs = s.quantile([0.0, 0.10, 0.25, 0.50, 0.75, 0.90, 1.0]).values
        print(f"  {col:18s}  min={qs[0]:.3f} p10={qs[1]:.3f} p25={qs[2]:.3f} "
              f"p50={qs[3]:.3f} p75={qs[4]:.3f} p90={qs[5]:.3f} max={qs[6]:.3f}")
    above = (out_df["max_pubTc"] >= 0.5).sum()
    print(f"  rows with max_pubTc >= 0.5 : {above:,} ({100.0*above/len(out_df):.2f}%)")
    print("\n── closest_lead histogram (top hits) ──")
    print(out_df["closest_lead"].value_counts().to_string())


if __name__ == "__main__":
    main()
