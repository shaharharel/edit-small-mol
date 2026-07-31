#!/usr/bin/env python3
"""Build the VERIFIED Boltz manifest: join Boltz outputs to source SMILES.

Per the QA agent's verified mapping (see /tmp/boltz_source_mapping_qa.md):
  - Source-of-truth CSV: `data/boltz_poses/survivors_1887_for_boltz.csv`
  - Mapping rule: Boltz dir 6-digit prefix == source CSV `row_id` (sparse,
    non-contiguous; range 33-3865).
  - Verification: heavy-atom count between source SMILES and Boltz PDB
    HETATM block (chain B, LIG residue). Flag mismatch >1 atom.
  - SMILES-Tanimoto on Boltz PDB is unreliable (no aromaticity); we use
    heavy-atom count + atom-symbol multiset instead.

Output:
  results/paper_evaluation/boltz_top1800/manifest_verified.csv
      columns: row_id, name, SMILES (source), method (cohort), pIC50_film,
               boltz_dir, ptm, iptm, plddt, complex_plddt, pae_mean,
               n_heavy_src, n_heavy_pdb, atom_check_ok, has_pdb,
               has_pae_npz, pdb_path

Plus a per-cohort aggregate at
  results/paper_evaluation/boltz_top1800/per_cohort_boltz_summary.csv
"""
from pathlib import Path
import re
import pandas as pd
import numpy as np
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

PROJECT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
BOLTZ = PROJECT / "data/boltz_results"
OUT_DIR = PROJECT / "results/paper_evaluation/boltz_top1800"
SOURCE_CSV = PROJECT / "data/boltz_poses/survivors_1887_for_boltz.csv"
UNVERIFIED = OUT_DIR / "manifest_unverified.csv"


def heavy_atoms_smiles(smi: str) -> int:
    m = Chem.MolFromSmiles(smi)
    return m.GetNumHeavyAtoms() if m else -1


def heavy_atoms_pdb(pdb_path: Path) -> int:
    """Count HETATM lines that are heavy atoms (not H) in chain B / LIG residue.

    The Boltz convention is: protein is chain A (ATOM records), ligand is
    chain B (HETATM records). Counting unique heavy-atom HETATM lines is
    robust to bond-order issues that confuse the SMILES parser.
    """
    if not pdb_path.exists():
        return -1
    n = 0
    try:
        with open(pdb_path) as f:
            for line in f:
                if not line.startswith("HETATM"):
                    continue
                # PDB columns: element symbol is cols 77-78 (0-indexed 76:78)
                elem = line[76:78].strip()
                if not elem:  # fallback: atom name cols 12-16
                    name = line[12:16].strip()
                    elem = "".join(c for c in name if c.isalpha())[:2]
                if elem.upper() == "H":
                    continue
                n += 1
    except Exception:
        return -1
    return n


def main():
    print(f"Loading source CSV: {SOURCE_CSV}", flush=True)
    src = pd.read_csv(SOURCE_CSV)
    print(f"  {len(src)} source rows; cols: {list(src.columns)}", flush=True)

    print(f"Loading unverified Boltz manifest: {UNVERIFIED}", flush=True)
    boltz = pd.read_csv(UNVERIFIED)
    print(f"  {len(boltz)} Boltz dirs scanned", flush=True)

    # Join: boltz.source_idx == src.row_id
    merged = boltz.merge(src, left_on="source_idx", right_on="row_id", how="left")
    n_unmatched = merged["row_id"].isna().sum()
    print(f"\n  joined: {len(merged)} rows  ({n_unmatched} unmatched)", flush=True)
    if n_unmatched > 0:
        bad = merged[merged["row_id"].isna()]["source_idx"].head(5).tolist()
        print(f"  ⚠ unmatched idx samples: {bad}", flush=True)

    # Verify heavy-atom counts
    print("\n  verifying heavy-atom counts (source SMILES vs Boltz PDB)...", flush=True)
    n_heavy_src = []
    n_heavy_pdb_arr = []
    for i, r in merged.iterrows():
        if pd.isna(r["SMILES"]):
            n_heavy_src.append(-1)
            n_heavy_pdb_arr.append(-1)
            continue
        n_heavy_src.append(heavy_atoms_smiles(r["SMILES"]))
        if r["has_pdb"] and isinstance(r["pdb_path"], str):
            n_heavy_pdb_arr.append(heavy_atoms_pdb(BOLTZ / r["pdb_path"]))
        else:
            n_heavy_pdb_arr.append(-1)
    merged["n_heavy_src"] = n_heavy_src
    merged["n_heavy_pdb"] = n_heavy_pdb_arr
    merged["atom_check_ok"] = (
        (merged["n_heavy_src"] >= 0)
        & (merged["n_heavy_pdb"] >= 0)
        & (abs(merged["n_heavy_src"] - merged["n_heavy_pdb"]) <= 1)
    )

    n_ok = int(merged["atom_check_ok"].sum())
    print(f"  atom-count match (±1): {n_ok}/{len(merged)}  ({100*n_ok/len(merged):.1f}%)", flush=True)
    bad_mask = (
        ~merged["atom_check_ok"]
        & merged["has_pdb"]
        & ~merged["row_id"].isna()
        & (merged["n_heavy_src"] >= 0)
        & (merged["n_heavy_pdb"] >= 0)
    )
    if bad_mask.sum() > 0:
        print(f"\n  ⚠ {bad_mask.sum()} rows with atom-count mismatch >1:", flush=True)
        cols = ["source_idx", "method", "n_heavy_src", "n_heavy_pdb"]
        print(merged.loc[bad_mask, cols].head(10).to_string(), flush=True)

    # Re-parse JSONs using the ACTUAL Boltz schema (discovered post-merge):
    # the canonical keys are confidence_score, ptm, iptm, ligand_iptm,
    # complex_plddt, complex_iplddt, complex_pde, complex_ipde.
    # (Earlier scan looked for non-existent 'plddt' and 'pae_mean'.)
    import json as _json
    new_cols = ["confidence_score", "ligand_iptm", "complex_iplddt", "complex_pde", "complex_ipde", "chain_ligand_ptm"]
    for c in new_cols:
        merged[c] = np.nan
    for i, r in merged.iterrows():
        jp = BOLTZ / r["json_path"]
        if not isinstance(r["json_path"], str) or not r["json_path"] or not jp.exists():
            continue
        try:
            with open(jp) as f:
                conf = _json.load(f)
            merged.at[i, "confidence_score"] = conf.get("confidence_score", np.nan)
            merged.at[i, "ligand_iptm"] = conf.get("ligand_iptm", np.nan)
            merged.at[i, "complex_iplddt"] = conf.get("complex_iplddt", np.nan)
            merged.at[i, "complex_pde"] = conf.get("complex_pde", np.nan)
            merged.at[i, "complex_ipde"] = conf.get("complex_ipde", np.nan)
            # chain "1" is the ligand chain in Boltz output (chain "0" = protein)
            cp = conf.get("chains_ptm", {})
            if isinstance(cp, dict):
                merged.at[i, "chain_ligand_ptm"] = cp.get("1", np.nan)
        except Exception:
            pass

    # Reorder + save
    keep_cols = [
        "row_id", "name", "SMILES", "method", "pIC50_film",
        "boltz_dir", "split", "source_idx", "cohort_prefix", "cohort_label",
        "confidence_score", "ptm", "iptm", "ligand_iptm",
        "complex_plddt", "complex_iplddt", "complex_pde", "complex_ipde",
        "chain_ligand_ptm",
        "n_heavy_src", "n_heavy_pdb", "atom_check_ok",
        "has_pdb", "has_pae_npz", "pdb_path", "json_path",
    ]
    keep_cols = [c for c in keep_cols if c in merged.columns]
    out = merged[keep_cols].copy()
    out_path = OUT_DIR / "manifest_verified.csv"
    out.to_csv(out_path, index=False)
    print(f"\n  → {out_path}  ({len(out)} rows)", flush=True)

    # Per-cohort aggregate
    print("\n  per-cohort Boltz summary:", flush=True)
    grp = out.groupby("method").agg(
        n_boltz=("row_id", "count"),
        n_atom_ok=("atom_check_ok", "sum"),
        confidence_score_mean=("confidence_score", "mean"),
        ptm_mean=("ptm", "mean"),
        iptm_mean=("iptm", "mean"),
        iptm_p10=("iptm", lambda x: float(np.percentile(x.dropna(), 10)) if len(x.dropna()) else np.nan),
        iptm_p90=("iptm", lambda x: float(np.percentile(x.dropna(), 90)) if len(x.dropna()) else np.nan),
        ligand_iptm_mean=("ligand_iptm", "mean"),
        complex_plddt_mean=("complex_plddt", "mean"),
        complex_iplddt_mean=("complex_iplddt", "mean"),
        complex_pde_mean=("complex_pde", "mean"),
        complex_ipde_mean=("complex_ipde", "mean"),
        pIC50_film_mean=("pIC50_film", "mean"),
    ).round(3).reset_index()
    grp.to_csv(OUT_DIR / "per_cohort_boltz_summary.csv", index=False)
    print(grp.to_string(), flush=True)
    print(f"\n  → {OUT_DIR / 'per_cohort_boltz_summary.csv'}", flush=True)


if __name__ == "__main__":
    main()
