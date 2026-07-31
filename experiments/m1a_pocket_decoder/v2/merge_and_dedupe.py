#!/usr/bin/env python3
"""Phase 3a: Merge all triple sources + dedupe.

Sources:
  - Original (boltz_zap70 + covindb_pdb): data/m1a_triples/triples.parquet (2,512)
  - CovInDB v2 metadata-driven:            data/m1a_triples_v2/covindb_v2_triples.parquet
  - CovBinderInPDB:                        data/m1a_triples_v2/covbinder_inpdb_triples.parquet

Dedupe key: (canon_smi, nucleophile_resid, pdb_id) - keep first occurrence.
"""
from __future__ import annotations
import json
import sys
from pathlib import Path
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
SOURCES = [
    PROJECT_ROOT / "data/m1a_triples/triples.parquet",          # original
    PROJECT_ROOT / "data/m1a_triples_v2/covindb_v2_triples.parquet",
    PROJECT_ROOT / "data/m1a_triples_v2/covbinder_inpdb_triples.parquet",
]
OUT = PROJECT_ROOT / "data/m1a_triples_v2/triples.parquet"


def main():
    dfs = []
    for sp in SOURCES:
        if not sp.exists():
            print(f"MISSING: {sp}", flush=True); continue
        df = pd.read_parquet(sp)
        print(f"{sp.name}: {len(df)} rows  cols={list(df.columns)}", flush=True)
        # Normalize: ensure these columns exist
        for col in ("warhead_class", "nucleophile_resname", "target_name"):
            if col not in df.columns:
                df[col] = None
        if "canon_smi" not in df.columns:
            df["canon_smi"] = df["smiles"].apply(
                lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s)) if (
                    isinstance(s, str) and Chem.MolFromSmiles(s)) else None)
        dfs.append(df)

    merged = pd.concat(dfs, ignore_index=True, sort=False)
    print(f"\nMerged: {len(merged)} rows", flush=True)
    print("By source:")
    print(merged["source"].value_counts())

    # SMARTS-based warhead-class backfill: legacy boltz+covindb_pdb triples
    # never got a warhead_class column populated (they were built by the older
    # pipeline). Apply the same SMARTS-first classifier here so every row has a
    # SMARTS-verified warhead_class label at merge time.
    sys.path.insert(0, str(PROJECT_ROOT / "experiments/m1a_pocket_decoder/v2"))
    from build_triples_v2 import find_warhead_atoms_smarts
    n_backfill = int(merged["warhead_class"].isna().sum())
    if n_backfill > 0:
        print(f"\nSMARTS backfill: classifying {n_backfill} rows without warhead_class...")
        def _classify(smi):
            if not isinstance(smi, str): return None
            m = Chem.MolFromSmiles(smi)
            if m is None: return None
            info = find_warhead_atoms_smarts(m, None)
            return info["name"] if info is not None else "unclassified"
        mask = merged["warhead_class"].isna()
        merged.loc[mask, "warhead_class"] = merged.loc[mask, "smiles"].apply(_classify)
        # Also backfill warhead_class_source_label so it exists on every row
        if "warhead_class_source_label" not in merged.columns:
            merged["warhead_class_source_label"] = None
        merged.loc[mask, "warhead_class_source_label"] = "backfilled_by_smarts"
        print(f"  Post-backfill warhead_class distribution (top 10):")
        print(merged.loc[mask, "warhead_class"].value_counts().head(10).to_string())

    # Drop rows with missing canon_smi or pose
    before_clean = len(merged)
    merged = merged.dropna(subset=["canon_smi", "warhead_pose_6d"])
    print(f"\nAfter cleaning NaN canon_smi/pose: {len(merged)} (dropped {before_clean - len(merged)})")

    # Chain-aware dedupe: same SMILES bound to same residue in same chain in
    # same PDB = duplicate (truly identical geometry). Different chains in the
    # same PDB = different pockets (kept). For Boltz the struct_id encodes the
    # cofold run so each is unique.
    def _chain_key(s):
        if not isinstance(s, str) or "_" not in s:
            return ""
        parts = s.split("_")
        # build_triples_v2 struct_id pattern: <pdb>_<chain>_<lig>_<nuc_posi>
        # legacy covindb_pdb:                <pdb>_<chain>_<resname>_<resid>
        # boltz:                              <stem> (no underscore parsing)
        return parts[1] if len(parts) >= 2 else ""
    merged["chain_key"] = merged["struct_id"].apply(_chain_key)
    merged["dedupe_pdb"] = merged.apply(
        lambda r: r["pdb_id"] if (isinstance(r["pdb_id"], str) and r["pdb_id"])
        else r.get("struct_id", ""),
        axis=1)
    merged["dedupe_nuc"] = merged["nucleophile_resid"].fillna(-1).astype(int)
    before_dedup = len(merged)
    merged = merged.drop_duplicates(
        subset=["canon_smi", "dedupe_nuc", "dedupe_pdb", "chain_key"],
        keep="first"
    )
    print(f"After dedupe: {len(merged)} (dropped {before_dedup - len(merged)})")
    merged = merged.drop(columns=["dedupe_pdb", "dedupe_nuc", "chain_key"])

    # Diagnostics
    print()
    print("=== Final by source ===")
    print(merged["source"].value_counts().to_string())
    print()
    print("=== Warhead class (top 20) ===")
    print(merged["warhead_class"].value_counts().head(20).to_string())
    print()
    print("=== Nucleophile (top 10) ===")
    print(merged["nucleophile_resname"].value_counts().head(10).to_string())
    print()
    print(f"Unique canon SMILES: {merged['canon_smi'].nunique()}")
    print(f"Unique PDBs: {merged['pdb_id'].nunique()}")

    OUT.parent.mkdir(parents=True, exist_ok=True)
    merged.to_parquet(OUT, index=False)
    print(f"\nWrote {len(merged)} triples to {OUT}", flush=True)


if __name__ == "__main__":
    main()
