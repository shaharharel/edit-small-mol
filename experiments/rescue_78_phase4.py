"""Rescue-78 Phase 4 driver: produce rescue_78_full.csv with column-complete schema.

Reads data/tier4_scored/rescue_78_working.csv (Phase 1+3 results) and ensures it
has all the columns the 838-survivor view consumes (defined by COLS list in
report_light.html and F4_boltz_full.csv schema). Fills NaN where the underlying
metric was not computed and documents missing columns.

Output: data/tier4_scored/rescue_78_full.csv
"""
from __future__ import annotations
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
WORKING = ROOT / "data/tier4_scored/rescue_78_working.csv"
FULL = ROOT / "data/tier4_scored/rescue_78_full.csv"
F4 = ROOT / "data/tier4_scored/F4_boltz_full.csv"
STATE = ROOT / "data/tier4_scored/rescue_78_state.json"
LOG = ROOT / "data/tier4_scored/rescue_78_progress.log"


def now_z() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def append_log(msg: str) -> None:
    with LOG.open("a") as fh:
        fh.write(f"[{now_z()}] {msg}\n")


def update_state(phase: str, status: str, **extra) -> None:
    s = json.loads(STATE.read_text())
    s["phases"][phase] = {"status": status, "ts": now_z(), **extra}
    STATE.write_text(json.dumps(s, indent=2))


def main():
    df = pd.read_csv(WORKING)
    n = len(df)
    append_log(f"Phase 4: loaded {n} rows × {len(df.columns)} cols")

    # Read F4 schema (columns) as the canonical target schema
    f4_cols = pd.read_csv(F4, nrows=1).columns.tolist()
    append_log(f"F4 reference schema has {len(f4_cols)} cols")

    # Aliases / legacy keys the light-view reads (mirrors the patches in backend.py
    # that map confidence.json keys → backend column names).
    alias_map = {
        "boltz_iptm":         ["iptm"],
        "boltz_ligand_iptm":  ["ligand_iptm"],
        "boltz_complex_plddt":["boltz_plddt", "complex_plddt"],
        "boltz_complex_pde":  ["boltz_pde", "complex_pde"],
        "boltz_confidence_score": ["boltz_confidence"],
        "mPAE_paper":         ["mPAE"],
    }
    for canonical, aliases in alias_map.items():
        if canonical in df.columns:
            for a in aliases:
                if a not in df.columns:
                    df[a] = df[canonical]

    # Ensure boltz_pose_locally_available is set
    if "boltz_pose_locally_available" not in df.columns:
        df["boltz_pose_locally_available"] = df.get("boltz_iptm").notna() if "boltz_iptm" in df.columns else False

    # Make sure every column the F4 view uses exists (NaN-fill if not)
    n_added = 0
    for c in f4_cols:
        if c not in df.columns:
            df[c] = np.nan
            n_added += 1
    append_log(f"Added {n_added} missing F4 columns as NaN")

    # row_id alias — the light-view uses 'row_id' as a primary key for SVG fetch;
    # for rescue-78 we use rescue_row_id as both. Keep both so the SVG endpoint
    # has something stable to look up.
    if "row_id" not in df.columns and "rescue_row_id" in df.columns:
        df["row_id"] = df["rescue_row_id"]

    # Stamp _source_cohort if missing (came from input)
    if "_source_cohort" not in df.columns:
        df["_source_cohort"] = "rescue_78"

    # mol1_murcko_smarts_match — present in input, ensure it's bool-like
    for c in ("warhead_intact", "thiq_core", "mol1_murcko_match", "mol1_murcko_smarts_match", "acryl_match"):
        if c in df.columns:
            df[c] = df[c].map(lambda v: True if str(v).lower() in ("true", "1") or v is True else (False if str(v).lower() in ("false", "0") or v is False else v))

    # Coerce numeric columns
    for c in df.columns:
        if df[c].dtype == "object":
            num = pd.to_numeric(df[c], errors="ignore")
            if num.dtype != "object":
                df[c] = num

    df.to_csv(FULL, index=False)
    append_log(f"Phase 4.1: wrote {FULL.name} with {len(df)} rows × {len(df.columns)} cols")
    update_state("4.1_full_csv", "done",
                 rows=len(df), cols=len(df.columns),
                 n_with_boltz=int(df["boltz_iptm"].notna().sum()) if "boltz_iptm" in df.columns else 0,
                 n_with_mmgbsa=int(df["dG_GB_kcalmol"].notna().sum()) if "dG_GB_kcalmol" in df.columns else 0,
                 n_with_pKa=int(df["pKa_Cys346"].notna().sum()) if "pKa_Cys346" in df.columns else 0)


if __name__ == "__main__":
    main()
