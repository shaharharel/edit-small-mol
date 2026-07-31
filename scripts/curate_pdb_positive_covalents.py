"""Curate PDB co-crystal covalent kinase inhibitors as positives for future Variant 4 DPO.

Filter chain (data-prep only, no training):
  1. Start from ``data/m1a_triples_v2/triples_posefix_v3.parquet`` (already
     canonicalised + pose-normalised covalent adduct triples with pdb_id +
     warhead_class + nucleophile_resname).
  2. Keep rows whose pdb_id is in the curated kinase PDB list
     (``data/covbinder/kinase_pdb_ids.json``, 12,616 kinase PDB IDs).
  3. Keep only Cys-anchored covalent adducts (``nucleophile_resname == 'CYS'``).
  4. Prefer Michael acceptors (acrylamide / michael_acceptor) as PRIMARY;
     include chloroacetamide/haloacetamide, vinyl_sulfone/vinyl_sulfonamide,
     bromoacetyl, chloroacetyl, acrylyl as SECONDARY.
  5. Canonicalise SMILES with RDKit and deduplicate.
  6. Cross-check pdb_id against the ESM pocket-embedding cache
     ``data/m1a_triples_v2/esm2_cache_posefix_v3.npz`` — since the cache is
     row-indexed via ``triples_posefix_v3.parquet['struct_id']``, every kept
     row has a pocket embedding by construction (we still record the flag
     per-item for downstream code).

Output: ``results/paper_evaluation/pdb_positive_covalents.json``.

Run under conda env ``quris``.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
TRIPLES = ROOT / "data/m1a_triples_v2/triples_posefix_v3.parquet"
ESM_CACHE = ROOT / "data/m1a_triples_v2/esm2_cache_posefix_v3.npz"
KINASE_PDBS = ROOT / "data/covbinder/kinase_pdb_ids.json"
CHASSIS_LABELS = ROOT / "data/covindb_chassis_labels.csv"
OUT_JSON = ROOT / "results/paper_evaluation/pdb_positive_covalents.json"

PRIMARY_WARHEADS = {"acrylamide", "michael_acceptor", "acrylyl", "acrylate"}
SECONDARY_WARHEADS = {
    "chloroacetamide", "haloacetamide", "chloroacetyl", "bromoacetyl",
    "vinyl_sulfone", "vinyl_sulfonamide", "aryl_sulfone",
}
KEEP_WARHEADS = PRIMARY_WARHEADS | SECONDARY_WARHEADS

# regex to classify target family from target_name (post-filter, for summary only)
TARGET_FAMILY_REGEX: list[tuple[str, re.Pattern[str]]] = [
    ("EGFR/ERBB",     re.compile(r"epidermal growth factor|erbB|EGFR", re.I)),
    ("BTK",           re.compile(r"\bBTK\b|Bruton", re.I)),
    ("JAK",           re.compile(r"\bJAK\d?\b", re.I)),
    ("SRC-family",    re.compile(r"\bSrc\b|proto-oncogene tyrosine-protein kinase src", re.I)),
    ("BMX/TEC",       re.compile(r"\bBMX\b|\bTEC\b|\bITK\b|\bTXK\b", re.I)),
    ("MAPK/ERK",      re.compile(r"MAP kinase|MAPK|Mitogen-activated protein kinase(?! kinase)|\bERK\b", re.I)),
    ("MAP2K/MEK",     re.compile(r"mitogen-activated protein kinase kinase(?! kinase)|\bMEK\b|MAP2K|dual specificity mitogen", re.I)),
    ("MAP3K/TAK",     re.compile(r"kinase kinase kinase|MAP3K|TAK1", re.I)),
    ("CDK",           re.compile(r"cyclin-dependent kinase|\bCDK\d*\b", re.I)),
    ("PDK1/PDPK1",    re.compile(r"3-phosphoinositide-dependent|PDPK1|PDK1", re.I)),
    ("Aurora",        re.compile(r"aurora kinase", re.I)),
    ("GSK",           re.compile(r"glycogen synthase kinase|GSK-?3", re.I)),
    ("Akt",           re.compile(r"RAC-\w+ serine/threonine|\bAkt\b", re.I)),
    ("MLKL/RIPK",     re.compile(r"mixed lineage kinase|\bRIPK\b|\bMLKL\b", re.I)),
    ("ZAP70",         re.compile(r"ZAP-?70", re.I)),
    ("S6K/RSK/p90",   re.compile(r"ribosomal protein S6 kinase|RPS6", re.I)),
    ("FGFR",          re.compile(r"FGFR|fibroblast growth factor receptor", re.I)),
    ("PI3K",          re.compile(r"phosphatidylinositol.*3-kinase|PI3K", re.I)),
    ("MET/HGFR",      re.compile(r"hepatocyte growth factor receptor|\bMET\b(?!hyl)", re.I)),
    ("ALK/LTK",       re.compile(r"\bALK\b(?!yl)|leukocyte tyrosine kinase", re.I)),
]


def target_family(name: str | float) -> str:
    if pd.isna(name):
        return "unknown"
    s = str(name)
    for fam, pat in TARGET_FAMILY_REGEX:
        if pat.search(s):
            return fam
    if "kinase" in s.lower():
        return "other-kinase"
    # UniProt-only IDs (P00533 etc.) with no readable name
    if re.fullmatch(r"[A-Z0-9]{6,10}", s.strip()):
        return "uniprot-only"
    return "other"


def canonical_smiles(smi: str) -> str | None:
    if smi is None or (isinstance(smi, float) and np.isnan(smi)):
        return None
    try:
        mol = Chem.MolFromSmiles(str(smi))
        if mol is None:
            return None
        return Chem.MolToSmiles(mol, canonical=True)
    except Exception:
        return None


def priority(warhead: str) -> str:
    if warhead in PRIMARY_WARHEADS:
        return "primary"
    if warhead in SECONDARY_WARHEADS:
        return "secondary"
    return "excluded"


def main() -> None:
    print(f"[1/6] Loading triples: {TRIPLES}")
    df = pd.read_parquet(TRIPLES)
    print(f"      rows: {len(df):,}")

    print(f"[2/6] Loading kinase PDB whitelist: {KINASE_PDBS}")
    kinase_set = {p.upper() for p in json.loads(KINASE_PDBS.read_text())}
    print(f"      kinase PDBs: {len(kinase_set):,}")

    df = df[df["pdb_id"].notna() & (df["pdb_id"] != "None")].copy()
    df["pdb_id"] = df["pdb_id"].astype(str).str.upper()
    print(f"      rows with real PDB id: {len(df):,}")

    df = df[df["nucleophile_resname"] == "CYS"].copy()
    print(f"[3/6] Cys-anchored rows: {len(df):,}")

    df = df[df["pdb_id"].isin(kinase_set)].copy()
    print(f"[4/6] Kinase PDB intersection: {len(df):,} rows, "
          f"{df['pdb_id'].nunique():,} unique PDBs")

    df = df[df["warhead_class"].isin(KEEP_WARHEADS)].copy()
    print(f"      after warhead filter: {len(df):,} rows")
    print("      warhead breakdown:")
    for k, v in df["warhead_class"].value_counts().items():
        print(f"        {k}: {v}")

    # Prefer canon_smi if present, fall back to smiles
    smi_series = df["canon_smi"].where(df["canon_smi"].notna(), df["smiles"])
    df["_canonical_smiles"] = smi_series.map(canonical_smiles)
    df = df[df["_canonical_smiles"].notna()].copy()
    print(f"      after RDKit canonicalisation: {len(df):,} rows")

    # Enrich target-gene guesses from chassis labels (optional secondary annotation)
    print(f"[5/6] Loading chassis labels for gene annotation: {CHASSIS_LABELS}")
    chassis = pd.read_csv(CHASSIS_LABELS)
    chassis["pdb_id"] = chassis["pdb_id"].astype(str).str.upper()
    chassis_pdb_to_class = dict(zip(chassis["pdb_id"], chassis["warhead_class"]))

    # ESM cache flag: struct_ids in cache is row-index-based, and every row in
    # this parquet corresponds to a cached row by construction.  We record the
    # flag anyway in case any row is missing.
    with np.load(ESM_CACHE, allow_pickle=True) as npz:
        cached_struct_ids = {str(x) for x in npz["struct_ids"]}
    print(f"      ESM cache holds {len(cached_struct_ids):,} struct_ids")

    df["_in_esm_cache"] = df["struct_id"].astype(str).isin(cached_struct_ids)
    print(f"      rows already in ESM cache: {int(df['_in_esm_cache'].sum())} / {len(df)}")

    df["target_family"] = df["target_name"].map(target_family)
    df["_priority"] = df["warhead_class"].map(priority)

    # Deduplicate by (canonical_smiles, pdb_id): keep first occurrence with
    # richest metadata.  Prefer PRIMARY over SECONDARY, then keep in-cache.
    df["_prio_rank"] = df["_priority"].map({"primary": 0, "secondary": 1, "excluded": 2})
    df["_cache_rank"] = (~df["_in_esm_cache"]).astype(int)
    df = df.sort_values(["_prio_rank", "_cache_rank"])
    dedup = df.drop_duplicates(subset=["_canonical_smiles", "pdb_id"], keep="first").copy()
    # Also collapse SMILES-only duplicates across PDBs (keep best-priority + in-cache),
    # remembering the extra PDB ids in a list.
    smiles_groups = dedup.groupby("_canonical_smiles").agg(
        pdb_ids=("pdb_id", lambda s: sorted(set(s))),
    )
    dedup = dedup.drop_duplicates(subset=["_canonical_smiles"], keep="first").copy()
    dedup = dedup.merge(smiles_groups, on="_canonical_smiles", how="left")
    print(f"[6/6] After SMILES dedup: {len(dedup):,} unique molecules")

    items = []
    for _, row in dedup.iterrows():
        pdb_id = row["pdb_id"]
        extra_pdbs = [p for p in row["pdb_ids"] if p != pdb_id]
        notes: list[str] = [f"priority={row['_priority']}"]
        if extra_pdbs:
            notes.append(f"also_in_pdbs={','.join(extra_pdbs)}")
        chassis_class = chassis_pdb_to_class.get(pdb_id)
        if isinstance(chassis_class, str) and chassis_class != row["warhead_class"]:
            notes.append(f"chassis_warhead_class={chassis_class}")
        items.append({
            "smiles": row["smiles"],
            "canonical_smiles": row["_canonical_smiles"],
            "pdb_id": pdb_id,
            "target_gene": row["target_name"] if isinstance(row["target_name"], str) else None,
            "target_family": row["target_family"],
            "warhead_class": row["warhead_class"],
            "warhead_priority": row["_priority"],
            "nucleophile_resname": row["nucleophile_resname"],
            "nucleophile_resid": int(row["nucleophile_resid"]) if pd.notna(row["nucleophile_resid"]) else None,
            "struct_id": row["struct_id"],
            "in_esm_cache": bool(row["_in_esm_cache"]),
            "source_file": row["source"],
            "notes": "; ".join(notes),
        })

    warhead_breakdown = dict(Counter(x["warhead_class"] for x in items).most_common())
    priority_breakdown = dict(Counter(x["warhead_priority"] for x in items).most_common())
    family_breakdown = dict(Counter(x["target_family"] for x in items).most_common())
    cache_count = sum(1 for x in items if x["in_esm_cache"])
    unique_pdbs = sorted({x["pdb_id"] for x in items})

    summary = {
        "total_count": len(items),
        "unique_pdbs": len(unique_pdbs),
        "warhead_class_breakdown": warhead_breakdown,
        "warhead_priority_breakdown": priority_breakdown,
        "target_family_breakdown": family_breakdown,
        "in_esm_cache_count": cache_count,
        "not_in_esm_cache_count": len(items) - cache_count,
        "sources": {
            "triples_parquet": str(TRIPLES.relative_to(ROOT)),
            "kinase_pdb_whitelist": str(KINASE_PDBS.relative_to(ROOT)),
            "esm_cache": str(ESM_CACHE.relative_to(ROOT)),
            "chassis_labels": str(CHASSIS_LABELS.relative_to(ROOT)),
        },
        "filter_criteria": {
            "nucleophile_resname": "CYS",
            "pdb_in_kinase_whitelist": True,
            "primary_warheads": sorted(PRIMARY_WARHEADS),
            "secondary_warheads": sorted(SECONDARY_WARHEADS),
        },
    }

    payload = {"summary": summary, "positives": items}
    OUT_JSON.parent.mkdir(parents=True, exist_ok=True)
    OUT_JSON.write_text(json.dumps(payload, indent=2))
    print()
    print(f"[out] Wrote {OUT_JSON}")
    print(f"      total curated positives: {len(items)}")
    print(f"      unique PDBs: {len(unique_pdbs)}")
    print(f"      in ESM cache: {cache_count} / {len(items)}")
    print(f"      warhead class breakdown: {warhead_breakdown}")
    print(f"      target family breakdown: {family_breakdown}")


if __name__ == "__main__":
    main()
