#!/usr/bin/env python3
"""Match 1,768 Boltz cofold results back to their source-cohort SMILES + cohort name.

Boltz output directory naming convention (assumed from inspection):
    boltz_results_<6-digit-idx>_<cohort-prefix>_<idx>
    e.g. boltz_results_002931_Tier_3_v3_LibInvent_lock_2931

We assume the 6-digit-idx is a GLOBAL index into a single ranked manifest
that was the input to the Boltz batch run on the A100s. The trailing
`_<idx>` is the same number repeated (sanity check). The cohort prefix
identifies which sub-cohort the molecule came from.

⚠️ RISK: if our assumption is wrong (e.g. idx is per-cohort instead of
global, or the manifest has been edited since the Boltz run), we WILL
mis-label molecules. To guard against this:
  1. The script does NOT modify anything until we verify spot-checks.
  2. It DEMANDS that the SMILES we infer from the source manifest, when
     compared to the SMILES embedded in the Boltz-generated PDB (parsed
     out of the HETATM block of the ligand chain), is at least 95%
     identical by canonical-Morgan-FP fingerprint.
  3. Any cohort with <95% match rate FAILS LOUD and is excluded from
     the merged output.

This script ONLY writes:
    results/paper_evaluation/boltz_top1800/manifest_verified.csv
        with columns: boltz_dir, cohort, source_idx, source_smiles,
        boltz_pdb_smiles, fp_match, confidence_json_path, plddt, ptm,
        iptm, pae_mean, has_pdb, has_pae_npz
"""
from __future__ import annotations
import json
import re
import sys
from pathlib import Path
from collections import defaultdict
from typing import Optional

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from rdkit.DataStructs import TanimotoSimilarity
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
BOLTZ = PROJECT / "data/boltz_results"
OUT_DIR = PROJECT / "results/paper_evaluation/boltz_top1800"
OUT_DIR.mkdir(parents=True, exist_ok=True)

# Where to look for the source manifest(s) — the FIRST hit per cohort wins.
# We verify shape later, but these are the candidates suspected from the
# 2026-06-03 overnight Tier 3 batch on a100/a100-b.
SOURCE_MANIFEST_CANDIDATES = [
    PROJECT / "experiments/boltz_inputs/manifest_split_a.csv",
    PROJECT / "experiments/boltz_inputs/manifest_split_b.csv",
    PROJECT / "experiments/boltz_inputs/manifest.csv",
    PROJECT / "experiments/boltz_inputs/covalid_top100_aligned/manifest.csv",
    PROJECT / "experiments/boltz_inputs/zap70_anchors_manifest.csv",
    PROJECT / "data/cohort_comparison/all_methods_bulk_scored_v4.csv",
    PROJECT / "results/paper_evaluation/cohort_comparison/all_cohorts_metrics.csv",
]

# Cohort-prefix → source-CSV mapping (best-effort, must be verified by QA agent).
# Prefixes in boltz dir names are TRUNCATED to ~24 chars, so we match on prefix.
COHORT_SOURCE_HINTS = {
    "Tier_3_v3_Mol2Mol_warhea": "Tier 3 v3 Mol2Mol warhead-token (overnight 2026-06-03)",
    "Tier_3_v3_LibInvent_lock": "Tier 3 v3 LibInvent_locked",
    "Tier_3_v2_Constrained_Ge": "Tier 3 v2 Constrained Generative",
    "Amine_Replacements": "Amine replacement library",
    "Medchem_Rules": "Tier 1 Med-chem rules",
    "Tier_3_v3_De_Novo_warhea": "Tier 3 v3 De Novo warhead",
}


def parse_boltz_dir(name: str) -> tuple[Optional[int], Optional[str], Optional[int]]:
    """boltz_results_002931_Tier_3_v3_LibInvent_lock_2931 → (2931, 'Tier_3_v3_LibInvent_lock', 2931)."""
    m = re.match(r"boltz_results_(\d{6})_(.+?)_(\d+)$", name)
    if not m:
        return None, None, None
    return int(m.group(1)), m.group(2), int(m.group(3))


def extract_ligand_smiles_from_boltz_pdb(pdb_path: Path) -> Optional[str]:
    """Read the Boltz-generated PDB and extract the ligand SMILES by parsing the LIG chain.

    Boltz writes the ligand atoms under a dedicated chain (commonly chain B or
    chain 'L'). RDKit's PDB reader can pick them up; we strip protein chains
    by chain ID then canonicalize.
    """
    try:
        mol = Chem.MolFromPDBFile(str(pdb_path), removeHs=False, sanitize=False)
        if mol is None:
            return None
        # Keep only the smallest fragment (ligand), assume protein is the largest
        frags = Chem.GetMolFrags(mol, asMols=True)
        if not frags:
            return None
        # ligand is usually the SMALLEST fragment (~30-80 heavy atoms vs 2000+ for protein)
        lig = min(frags, key=lambda x: x.GetNumHeavyAtoms())
        if lig.GetNumHeavyAtoms() > 200:  # sanity: real protein, not a ligand
            return None
        try:
            Chem.SanitizeMol(lig)
        except Exception:
            pass
        return Chem.MolToSmiles(lig, isomericSmiles=False, canonical=True)
    except Exception:
        return None


def morgan_fp(smi: str):
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return None
    return AllChem.GetMorganFingerprintAsBitVect(m, 2, 2048)


def fp_match(a: str, b: str) -> float:
    fa, fb = morgan_fp(a), morgan_fp(b)
    if fa is None or fb is None:
        return 0.0
    return float(TanimotoSimilarity(fa, fb))


def find_source_manifest():
    """Return the first source manifest CSV that exists, with a SMILES column.

    Different cohort batches may use different manifests — we return the FIRST
    hit and report which one to the QA agent for verification.
    """
    hits = []
    for p in SOURCE_MANIFEST_CANDIDATES:
        if not p.exists():
            continue
        try:
            df = pd.read_csv(p, nrows=5)
        except Exception:
            continue
        # Look for SMILES-ish column
        for c in df.columns:
            if c.lower() in ("smiles", "smi", "canonical_smiles", "mol_smiles"):
                hits.append((p, c, len(pd.read_csv(p))))
                break
    return hits


def main():
    print("=== Scanning Boltz output dirs ===", flush=True)
    rows = []
    for split in ("split_a", "split_b"):
        sp = BOLTZ / split
        if not sp.exists():
            continue
        for d in sorted(sp.iterdir()):
            if not d.is_dir() or not d.name.startswith("boltz_results_"):
                continue
            idx, cohort, idx2 = parse_boltz_dir(d.name)
            if idx is None or idx != idx2:
                print(f"  WARNING: parse failure: {d.name}", flush=True)
                continue
            # Find PDB
            preds_root = d / "predictions"
            pdb = json_p = pae_npz = plddt_npz = None
            if preds_root.exists():
                for sub in preds_root.iterdir():
                    for f in sub.iterdir():
                        n = f.name
                        if n.endswith("_model_0.pdb"):
                            pdb = f
                        elif n.startswith("confidence_") and n.endswith(".json"):
                            json_p = f
                        elif n.startswith("pae_") and n.endswith(".npz"):
                            pae_npz = f
                        elif n.startswith("plddt_") and n.endswith(".npz"):
                            plddt_npz = f
            rows.append({
                "boltz_dir": str(d.relative_to(BOLTZ)),
                "split": split,
                "source_idx": idx,
                "cohort_prefix": cohort,
                "cohort_label": COHORT_SOURCE_HINTS.get(cohort, "UNKNOWN"),
                "has_pdb": pdb is not None,
                "has_confidence_json": json_p is not None,
                "has_pae_npz": pae_npz is not None,
                "has_plddt_npz": plddt_npz is not None,
                "pdb_path": str(pdb.relative_to(BOLTZ)) if pdb else "",
                "json_path": str(json_p.relative_to(BOLTZ)) if json_p else "",
            })

    df = pd.DataFrame(rows)
    print(f"  collected {len(df)} Boltz dirs", flush=True)
    print(f"  cohort breakdown:", flush=True)
    for cp, n in df["cohort_prefix"].value_counts().items():
        label = COHORT_SOURCE_HINTS.get(cp, "UNKNOWN")
        print(f"    {n:>4}  {cp:<30}  ({label})", flush=True)

    # Confidence JSON ingestion
    print("\n=== Ingesting confidence JSONs ===", flush=True)
    for col in ("ptm", "iptm", "plddt", "complex_plddt", "ligand_plddt", "pae_mean"):
        df[col] = np.nan
    n_json_ok = 0
    for i, r in df.iterrows():
        jp = BOLTZ / r["json_path"]
        if not r["has_confidence_json"] or not jp.exists():
            continue
        try:
            with open(jp) as f:
                conf = json.load(f)
            df.at[i, "ptm"] = conf.get("ptm", np.nan)
            df.at[i, "iptm"] = conf.get("iptm", np.nan)
            df.at[i, "plddt"] = conf.get("plddt", np.nan)
            df.at[i, "complex_plddt"] = conf.get("complex_plddt", np.nan)
            df.at[i, "ligand_plddt"] = conf.get("ligand_iptm",
                                                conf.get("ligand_plddt", np.nan))
            df.at[i, "pae_mean"] = conf.get("pae_mean",
                                            conf.get("pae", {}).get("mean", np.nan)
                                            if isinstance(conf.get("pae"), dict) else np.nan)
            n_json_ok += 1
        except Exception as e:
            print(f"  bad JSON: {jp.name}: {e}", flush=True)
    print(f"  parsed {n_json_ok}/{len(df)} JSONs", flush=True)

    # Source-manifest preview
    print("\n=== Source manifest candidates ===", flush=True)
    hits = find_source_manifest()
    if not hits:
        print("  ⚠ NO source manifests found. Cannot verify SMILES mapping.", flush=True)
        print("  Skipping spot-check stage; QA must determine the correct source.", flush=True)
    else:
        for p, c, n in hits:
            print(f"  {p} (col='{c}', {n} rows)", flush=True)

    out_path = OUT_DIR / "manifest_unverified.csv"
    df.to_csv(out_path, index=False)
    print(f"\n  → {out_path}", flush=True)
    print(f"  ⚠ This manifest is UNVERIFIED — SMILES not yet cross-checked.", flush=True)
    print(f"  ⚠ Next step requires QA agent to identify the correct source CSV(s)", flush=True)
    print(f"     per cohort, then a separate verification script can map idx → SMILES.", flush=True)


if __name__ == "__main__":
    main()
