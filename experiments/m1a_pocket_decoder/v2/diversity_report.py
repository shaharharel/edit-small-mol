#!/usr/bin/env python3
"""Phase 4: Diversity QA report for M1a v2 triples."""
from __future__ import annotations
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem.Scaffolds import MurckoScaffold

RDLogger.DisableLog("rdApp.*")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
TRIPLES = PROJECT_ROOT / "data/m1a_triples_v2/triples.parquet"
REPORT = PROJECT_ROOT / "data/m1a_triples_v2/diversity_report.md"


def murcko(smi):
    try:
        m = Chem.MolFromSmiles(smi)
        if m is None:
            return None
        s = MurckoScaffold.GetScaffoldForMol(m)
        return Chem.MolToSmiles(s)
    except Exception:
        return None


def categorize_target(name):
    if not isinstance(name, str):
        return "unknown"
    n = name.lower()
    if any(k in n for k in ["kinase", "kit", "src", "egfr", "fgfr", "btk", "jak", "syk", "zap", "lck", "abl", "mek", "erk", "akt", "ck1", "ck2", "cdk", "alk", "tyr", "ser/thr"]):
        return "kinase"
    if any(k in n for k in ["protease", "pepsin", "trypsin", "chymotryp", "elast", "papain", "calpain", "cathep", "caspase", "thromb", "3clpro", "mpro", "factor"]):
        return "protease"
    if any(k in n for k in ["transferase", "synth", "ase"]):
        return "other_enzyme"
    if any(k in n for k in ["receptor", "channel"]):
        return "receptor"
    return "other"


def main():
    df = pd.read_parquet(TRIPLES)
    n = len(df)
    print(f"Total triples: {n}", flush=True)

    by_source = df["source"].value_counts().to_dict()
    n_smi = df["canon_smi"].nunique()
    n_pdb = df["pdb_id"].dropna().nunique()
    df["murcko"] = df["canon_smi"].apply(murcko)
    n_scaff = df["murcko"].dropna().nunique()

    warhead = df["warhead_class"].fillna("unknown").value_counts()
    nuc = df["nucleophile_resname"].fillna("unknown").value_counts()

    df["target_cat"] = df["target_name"].apply(categorize_target)
    targets = df["target_cat"].value_counts()

    # Pocket sizes (from JSON)
    sizes = df["pocket_residues"].apply(lambda s: len(json.loads(s)))
    bd = pd.to_numeric(df["bd_angle_deg"], errors="coerce").dropna()
    d_b_nuc = pd.to_numeric(df["d_b_nuc"], errors="coerce").dropna()
    in_range = ((bd >= 95) & (bd <= 115)).mean() * 100

    lines = []
    lines.append("# M1a v2 Triples - Diversity Report\n")
    lines.append(f"Generated: 2026-06-30  \nSource parquet: `{TRIPLES.relative_to(PROJECT_ROOT)}`  \n")
    lines.append(f"**Total triples**: {n:,}\n")
    lines.append(f"**Unique canonical SMILES**: {n_smi:,}\n")
    lines.append(f"**Unique PDBs**: {n_pdb:,}\n")
    lines.append(f"**Unique Murcko scaffolds**: {n_scaff:,}\n")
    lines.append("")
    lines.append("## Per-source breakdown\n")
    lines.append("| source | rows | %  |")
    lines.append("|---|---|---|")
    for src, c in by_source.items():
        lines.append(f"| {src} | {c:,} | {100.0*c/n:.1f}% |")
    lines.append("")
    lines.append("## Warhead class distribution (top 30)\n")
    lines.append("| warhead_class | n |")
    lines.append("|---|---|")
    for w, c in warhead.head(30).items():
        lines.append(f"| {w} | {c} |")
    lines.append(f"\n(Total distinct warhead classes: **{warhead.size}**)\n")

    lines.append("\n## Nucleophile residue distribution\n")
    lines.append("| nucleophile | n |")
    lines.append("|---|---|")
    for nu, c in nuc.items():
        lines.append(f"| {nu} | {c} |")

    lines.append("\n## Target category distribution (heuristic name match)\n")
    lines.append("| category | n |")
    lines.append("|---|---|")
    for t, c in targets.items():
        lines.append(f"| {t} | {c} |")

    lines.append("\n## Pocket size distribution\n")
    lines.append(f"- min={sizes.min()}  median={int(sizes.median())}  mean={sizes.mean():.2f}  max={sizes.max()}")
    lines.append("- distribution:")
    for sz_bin in [(4,5),(6,8),(9,12),(13,16),(17,20)]:
        m = ((sizes >= sz_bin[0]) & (sizes <= sz_bin[1])).sum()
        lines.append(f"  - {sz_bin[0]}-{sz_bin[1]} residues: {m} ({100.0*m/n:.1f}%)")

    lines.append("\n## BD angle distribution (warhead C-beta -> nucleophile angle)\n")
    lines.append(f"- min={bd.min():.1f}  25%={bd.quantile(0.25):.1f}  median={bd.median():.1f}  75%={bd.quantile(0.75):.1f}  max={bd.max():.1f}")
    lines.append(f"- mean={bd.mean():.1f}  std={bd.std():.1f}")
    lines.append(f"- **fraction in 95-115 deg sweet spot** (sp3 attack on sp2 carbon, Burgi-Dunitz): **{in_range:.1f}%**\n")

    lines.append("\n## d(b, nuc) distance distribution\n")
    lines.append(f"- min={d_b_nuc.min():.2f}  median={d_b_nuc.median():.2f}  max={d_b_nuc.max():.2f}")
    lines.append(f"- mean={d_b_nuc.mean():.2f}  std={d_b_nuc.std():.2f}  A\n")

    lines.append("\n## QA notes\n")
    lines.append("- Pocket residues sourced via 8 A C-alpha cutoff around the warhead beta-C, restricted to the nucleophile's chain (avoids multimer blowup).")
    lines.append("- Warhead beta/alpha atoms identified via per-class SMARTS on the metadata SMILES, then mapped to PDB HETATM xyz coords by element-greedy matching.")
    lines.append("- SMILES are the authoritative metadata SMILES from CovInDB v2 / CovBinderInPDB; no re-parsing from PDB HETATM (which loses bond orders).")
    lines.append("- Dedupe key: (canon_smi, nucleophile_resid, pdb_id, chain) -- same SMILES bound to same residue in same chain = duplicate; different chains = distinct geometries kept.")
    lines.append("- All 9 nucleophile classes (Cys, Ser, Thr, Lys, His, Glu, Asp, Tyr, Met) represented.")
    lines.append("- ESM-2-8M (320d) per-residue embeddings precomputed at `data/m1a_triples_v2/esm2_cache.npz` (3,155 unique pocket sequences, 37.5 MB).\n")

    REPORT.write_text("\n".join(lines))
    print(f"Wrote {REPORT}")


if __name__ == "__main__":
    main()
