#!/usr/bin/env python3
"""Phase 0: Diagnose why CovInDB v2 PDB extraction yields only 27%.

Runs the EXISTING extraction logic from build_triples.parse_pdb_for_triple
against ALL 3,445 PDBs and records the failure reason per file. Categorises:

 (i)   no_cys_sg           - no Cys SG found in PDB
 (ii)  no_close_het        - HET atoms exist but no carbon within 2.5 A of any SG
 (iii) het_too_small       - all HET candidates have <8 or >80 heavy atoms
 (iv)  pocket_size_bad     - pocket <4 or >80 residues at 8A
 (v)   pdb_parse_error     - file unreadable
 (vi)  no_smiles_recovered - geometry OK but RDKit PDB->SMILES failed
 (vii) ok                  - extraction succeeded
 (viii) ok_no_smiles       - geometry OK but no SMILES recovered (dropped by current code)

Also cross-references with Covalent_Complex_Records.csv to identify what we
*should* have extracted:
  - residue category mismatch (e.g. metadata says SER145 but we only check CYS)
  - warhead class (e.g. Phosphonate is recorded but extractor only handles
    acrylamide/chloroacetamide/vinyl_sulfonamide/alpha_keto_amide)
"""
from __future__ import annotations
import json
import os
import sys
import warnings
from pathlib import Path
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
PDB_DIR = PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB"
META_CSV = PROJECT_ROOT / "data/covbinder/raw_covindb2/Covalent_Complex_Records.csv"
OUT_DIR = PROJECT_ROOT / "data/m1a_triples_v2"
OUT_DIR.mkdir(parents=True, exist_ok=True)

POCKET_RADIUS = 8.0
NUC_ATOMS = {
    "CYS": ("SG",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "LYS": ("NZ",),
    "TYR": ("OH",),
    "HIS": ("ND1", "NE2"),
}
SKIP_HET = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "MG", "CA", "ZN", "MN",
    "FE", "K", "BR", "EDO", "GOL", "PEG", "DMS", "DTT", "BME", "NAG", "MAN",
    "FUC", "BMA", "GAL", "GLC", "ACT", "TRS", "IPA", "FMT", "MES", "BCT",
}


def diagnose_pdb(pdb_path_str: str) -> dict:
    """Return dict with status + per-step counts."""
    pdb_path = Path(pdb_path_str)
    pdb_id = pdb_path.stem
    out = {"pdb_id": pdb_id, "status": "unknown",
           "n_residues": 0, "n_het_groups": 0, "n_sg": 0,
           "n_het_close_to_sg": 0, "n_het_with_size_ok": 0,
           "n_triples_geom": 0, "n_triples_with_smi": 0}
    residues = {}
    hetatoms = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        atom_name = line[12:16].strip()
                        resname = line[17:20].strip()
                        chain = line[21]
                        resid = int(line[22:26])
                        x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                        elem = line[76:78].strip() if len(line) > 78 else atom_name[0]
                        # altloc filter: only accept blank or "A"
                        altloc = line[16]
                        if altloc not in (" ", "A"):
                            continue
                    except (ValueError, IndexError):
                        continue
                    xyz = np.array([x, y, z])
                    if line.startswith("ATOM"):
                        key = (chain, resid)
                        if key not in residues:
                            residues[key] = {"aa": resname, "idx": resid,
                                              "chain": chain, "atoms": {}}
                        residues[key]["atoms"][atom_name] = xyz
                    else:
                        if resname in SKIP_HET:
                            continue
                        key = (chain, resname, resid)
                        hetatoms.setdefault(key, []).append(
                            {"name": atom_name, "elem": elem, "xyz": xyz})
    except Exception as e:
        out["status"] = "pdb_parse_error"
        out["error"] = str(e)[:100]
        return out

    out["n_residues"] = len(residues)
    out["n_het_groups"] = len(hetatoms)
    if not residues:
        out["status"] = "no_protein"
        return out
    if not hetatoms:
        out["status"] = "no_het"
        return out

    # Count Cys SGs
    sg_atoms = []
    for r in residues.values():
        if r["aa"] == "CYS" and "SG" in r["atoms"]:
            sg_atoms.append((r["chain"], r["idx"], r["atoms"]["SG"]))
    out["n_sg"] = len(sg_atoms)
    if not sg_atoms:
        out["status"] = "no_cys_sg"
        return out

    n_close = 0
    n_size_ok = 0
    n_geom_triples = 0
    for (chain, resname, resid), atoms in hetatoms.items():
        heavy = [a for a in atoms if a["elem"] not in ("H", "D")]
        if not (8 <= len(heavy) <= 80):
            continue
        n_size_ok += 1
        # Closest carbon to any SG
        best = None
        for ha in heavy:
            if ha["elem"] not in ("C", "S"):
                continue
            for (_, _, sg_xyz) in sg_atoms:
                d = float(np.linalg.norm(ha["xyz"] - sg_xyz))
                if d < 2.5 and (best is None or d < best[0]):
                    best = (d, ha)
        if best is None:
            continue
        n_close += 1
        # Pocket size check
        b_xyz = best[1]["xyz"]
        pocket = []
        for r in residues.values():
            if "CA" not in r["atoms"]:
                continue
            d_ca = float(np.linalg.norm(r["atoms"]["CA"] - b_xyz))
            if d_ca <= POCKET_RADIUS:
                pocket.append(r)
        if 4 <= len(pocket) <= 80:
            n_geom_triples += 1

    out["n_het_close_to_sg"] = n_close
    out["n_het_with_size_ok"] = n_size_ok
    out["n_triples_geom"] = n_geom_triples

    if n_size_ok == 0:
        out["status"] = "het_too_small_or_big"
    elif n_close == 0:
        out["status"] = "no_close_het_to_sg"
    elif n_geom_triples == 0:
        out["status"] = "pocket_size_bad"
    else:
        out["status"] = "ok_geom"  # geometry succeeded; SMILES recovery is separate
    return out


def main():
    pdb_files = sorted(PDB_DIR.glob("*.pdb"))
    print(f"Diagnosing {len(pdb_files)} PDB files...", flush=True)

    n_workers = os.cpu_count() or 8
    results = []
    with ProcessPoolExecutor(max_workers=n_workers) as ex:
        futures = {ex.submit(diagnose_pdb, str(p)): p for p in pdb_files}
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(pdb_files)}", flush=True)

    df = pd.DataFrame(results)
    df.to_csv(OUT_DIR / "diagnosis_per_pdb.csv", index=False)
    print()
    print("=== Status breakdown ===")
    print(df["status"].value_counts())

    # Cross-reference with metadata
    meta = pd.read_csv(META_CSV)
    meta["PDB"] = meta["PDB"].astype(str).str.lower()
    df["pdb_id"] = df["pdb_id"].str.lower()
    merged = df.merge(meta, left_on="pdb_id", right_on="PDB", how="left")

    # Status x warhead
    print()
    print("=== Status x Warhead class (top warheads) ===")
    top_warheads = meta["Warhead"].value_counts().head(15).index.tolist()
    cross = merged[merged["Warhead"].isin(top_warheads)].pivot_table(
        index="Warhead", columns="status", values="pdb_id", aggfunc="count", fill_value=0)
    print(cross.to_string())

    # Status x nucleophile residue
    print()
    print("=== Status x Nucleophile residue ===")
    cross_nuc = merged.pivot_table(
        index="Resi_name", columns="status", values="pdb_id", aggfunc="count", fill_value=0)
    print(cross_nuc.to_string())

    # Sample 5 PDBs per failure category
    print()
    print("=== Example PDBs per failure category ===")
    examples = {}
    for status in df["status"].unique():
        sub = df[df["status"] == status]
        ex_list = sub.head(5)["pdb_id"].tolist()
        m_for_status = merged[merged["status"] == status].head(5)
        examples[status] = {
            "count": len(sub),
            "examples": ex_list,
            "metadata_for_examples": m_for_status[["pdb_id", "Warhead", "Resi_name",
                                                    "Resi_chain", "Resi_posi",
                                                    "Ligand_name"]].to_dict(orient="records"),
        }
        print(f"\n[{status}] N={len(sub)}")
        for e in examples[status]["metadata_for_examples"]:
            print(f"  {e['pdb_id']}: warhead={e['Warhead']} nuc={e.get('Resi_name')}{e.get('Resi_posi')} chain={e.get('Resi_chain')} lig={e.get('Ligand_name')}")

    # What we COULD potentially recover with metadata
    print()
    print("=== Potential recovery using metadata ===")
    # PDBs where metadata says SER/THR/LYS/TYR/HIS nucleophile (extractor only does CYS!)
    non_cys = meta[~meta["Resi_name"].isin(["CYS", "MSE", "SEC"])]
    print(f"Records with non-Cys nucleophile: {len(non_cys)} ({len(non_cys.PDB.unique())} unique PDBs)")
    print(f"  Their warheads:")
    print(non_cys["Warhead"].value_counts().head(10).to_string())

    # Save
    summary = {
        "total_pdbs": len(df),
        "status_breakdown": df["status"].value_counts().to_dict(),
        "examples_per_status": {k: {"count": v["count"], "examples": v["examples"]}
                                  for k, v in examples.items()},
    }
    with open(OUT_DIR / "diagnosis_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSaved: {OUT_DIR / 'diagnosis_summary.json'}")
    print(f"Saved: {OUT_DIR / 'diagnosis_per_pdb.csv'}")


if __name__ == "__main__":
    main()
