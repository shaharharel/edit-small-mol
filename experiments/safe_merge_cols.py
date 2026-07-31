#!/usr/bin/env python3
"""safe_merge_cols.py — additive, loss-free in-place CSV column enrichment.

USE THIS MODULE FROM ANY SCRIPT THAT ADDS COLUMNS TO AN EXISTING tier4_scored
CSV. It guarantees:

  1. The on-disk file is re-read fresh inside the helper (no stale snapshots).
  2. Existing non-NaN values are NEVER overwritten by NaN.
  3. Columns the caller did not touch are NEVER dropped.
  4. The write is atomic: .csv.tmp then os.replace().

Typical use::

    from experiments.safe_merge_cols import safe_merge_inplace

    # Build a new-cols DataFrame keyed on a stable id (row_id or smiles).
    new = pd.DataFrame({
        "row_id":  [...],
        "LLE":     [...],
        "LE":      [...],
    })

    safe_merge_inplace(
        csv_path="data/tier4_scored/foo_scored.csv",
        new_df=new,
        key="row_id",       # or "smiles"
    )

If the key column is missing from the on-disk CSV, the helper raises rather
than silently appending row-aligned (which is what caused the 00:31 data loss).

For column-aligned (no key) use you can pass ``align="rowidx"`` BUT the helper
will then refuse to run unless ``len(new_df) == len(existing)`` to avoid the
silent truncation footgun.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


def safe_merge_inplace(
    csv_path: str | Path,
    new_df: pd.DataFrame,
    key: Optional[str] = "row_id",
    align: str = "key",
    cols_to_write: Optional[Iterable[str]] = None,
    prefer_existing: bool = True,
) -> dict:
    """Additively merge ``new_df`` into ``csv_path`` without losing columns.

    Args:
        csv_path: target CSV (read fresh, written atomically)
        new_df:   DataFrame with the new columns. Must contain ``key`` unless
                  align="rowidx".
        key:      join key column (default "row_id"). Must exist in BOTH on-disk
                  CSV and ``new_df`` when align="key".
        align:    "key"     -> merge on ``key`` (safe; default)
                  "rowidx"  -> position-align (refuses when row counts differ)
        cols_to_write: explicit list of new cols to merge. None = all new cols
                  in ``new_df`` except the key.
        prefer_existing: True means existing non-NaN values are preserved; only
                  NaN cells get filled from ``new_df``. False = overwrite blindly.

    Returns a small summary dict.
    """
    p = Path(csv_path)
    if not p.exists():
        raise FileNotFoundError(p)

    existing = pd.read_csv(p)
    n_existing = len(existing)

    if cols_to_write is None:
        cols_to_write = [c for c in new_df.columns if c != key]
    else:
        cols_to_write = list(cols_to_write)

    if align == "key":
        if key is None:
            raise ValueError("align='key' requires a key column name")
        if key not in existing.columns:
            raise KeyError(
                f"{p.name} has no '{key}' column; cannot safely merge. "
                f"Available cols: {list(existing.columns)[:10]}..."
            )
        if key not in new_df.columns:
            raise KeyError(f"new_df has no '{key}' column")
        # Merge: existing left join new_df on key
        merge_cols = [key] + [c for c in cols_to_write if c in new_df.columns]
        merged = existing.merge(new_df[merge_cols], on=key, how="left",
                                suffixes=("", "__NEW"))
    elif align == "rowidx":
        if len(new_df) != n_existing:
            raise ValueError(
                f"rowidx alignment refused: len(new_df)={len(new_df)} != "
                f"len(existing)={n_existing}. Use key-based merge instead."
            )
        merged = existing.copy()
        for c in cols_to_write:
            if c in new_df.columns:
                merged[c + "__NEW"] = new_df[c].values
    else:
        raise ValueError(f"unknown align={align!r}")

    # Resolve __NEW vs existing column with prefer_existing semantics
    filled = []
    overwritten = []
    added = []
    for c in cols_to_write:
        new_col = c + "__NEW"
        if new_col not in merged.columns:
            continue
        if c in existing.columns:
            if prefer_existing:
                # Only fill NaN cells in existing column
                mask_nan = merged[c].isna()
                merged.loc[mask_nan, c] = merged.loc[mask_nan, new_col]
                filled.append((c, int(mask_nan.sum())))
            else:
                merged[c] = merged[new_col]
                overwritten.append(c)
        else:
            merged[c] = merged[new_col]
            added.append(c)
        merged.drop(columns=[new_col], inplace=True)

    # Sanity: row count must equal existing
    if len(merged) != n_existing:
        raise RuntimeError(
            f"merge inflated rows: {len(merged)} != {n_existing}. "
            f"Check key uniqueness in new_df."
        )

    # Atomic write
    tmp = p.with_suffix(".csv.tmp")
    merged.to_csv(tmp, index=False)
    os.replace(tmp, p)

    return {
        "csv": str(p),
        "rows": n_existing,
        "added_cols": added,
        "filled_in_nan_cols": filled,
        "overwritten_cols": overwritten,
        "final_cols": len(merged.columns),
    }


if __name__ == "__main__":
    import argparse
    import json

    ap = argparse.ArgumentParser(description="Self-test: safe_merge_inplace")
    ap.add_argument("--demo", action="store_true")
    args = ap.parse_args()

    if args.demo:
        import tempfile

        with tempfile.TemporaryDirectory() as tmp:
            t = Path(tmp) / "demo.csv"
            pd.DataFrame({"row_id": [1, 2, 3], "smiles": ["C", "CC", "CCC"],
                          "MW": [12, 26, 40], "LLE": [None, 4.0, None]}).to_csv(t, index=False)
            new = pd.DataFrame({"row_id": [1, 2, 3], "LLE": [3.0, 9.9, 5.0],
                                "LE": [0.5, 0.4, 0.3]})
            summary = safe_merge_inplace(t, new, key="row_id")
            print(json.dumps(summary, indent=2))
            print("After merge:")
            print(pd.read_csv(t))
