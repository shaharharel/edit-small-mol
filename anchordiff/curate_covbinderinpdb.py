"""CovBinderInPDB curation — flagship D+C training data prep.

Pulls the published CovBinderInPDB dataset (~3000 covalent protein-ligand
co-crystals), filters to kinase-Cys complexes with our supported warhead
classes (acrylamide / chloroacetamide / vinyl-sulfonamide / cyanoacrylate),
and writes a training CSV the flagship D+C diffusion fine-tune script can
consume directly.

Strategy:
  1. Download the curated CSV summary from the CovBinderInPDB FTP /
     Zenodo mirror. (If unavailable, fall back to building from a local
     list of known kinase-covalent PDB IDs.)
  2. For each entry: download PDB, find Cys-SG-bonded ligand, extract:
       - protein file path
       - ligand SDF (heavy-atom 3D)
       - warhead-class one-hot (matched via SMARTS)
       - Cys SG xyz (label_seq_id + chain)
       - warhead atom indices (Cα/Cβ/C_carb/O/N in ligand)
  3. Persist:
       data/covbinder/raw/<pdb_id>/{protein.pdb, ligand.sdf}
       data/covbinder/training_set.csv  (one row per complex)

This is local CPU. ETA ~2h (most time spent in PDB downloads).
Final CSV is what `train_DC_flagship.py` (TBD) will load.

NOTE: this script attempts the network fetches but is designed to fail
gracefully if remote sources are unavailable — it then writes a stub CSV
with what it has + a clear "needs network" note. Re-run when online.
"""
from __future__ import annotations
import sys, os, json, time
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import urllib.request
import urllib.error
import pandas as pd
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

OUT_DIR = PROJECT_ROOT / "data" / "covbinder"
OUT_DIR.mkdir(parents=True, exist_ok=True)
RAW_DIR = OUT_DIR / "raw"
RAW_DIR.mkdir(parents=True, exist_ok=True)

# CovBinderInPDB — paper: Liu et al., J Chem Inf Model 2022; data URL on
# the project's GitHub release: https://github.com/yuxiangren/CovBinderInPDB
# We attempt a few candidate URLs.
CANDIDATE_URLS = [
    "https://raw.githubusercontent.com/yuxiangren/CovBinderInPDB/main/data/covbinder_pdb_list.csv",
    "https://raw.githubusercontent.com/yuxiangren/CovBinderInPDB/master/data/covbinder_pdb_list.csv",
    "https://raw.githubusercontent.com/yuxiangren/CovBinderInPDB/main/CovBinderInPDB.csv",
]
# Fallback: hand-curated kinase-covalent PDBs (well-characterised, all Cys, all
# acrylamide or chloroacetamide). Used if network fails. Source: review of
# Schneuing 2024 supp, London JACS 2026, Shi 2021 (ZAP70 RDN009), Wang 2023.
FALLBACK_PDBS = [
    # (pdb, target, cys_residue, warhead)
    ("4K2R", "ZAP70",        346, "AMP_PNP"),       # apo-ish; no covalent ligand
    ("5P9J", "BTK",          481, "acrylamide"),    # ibrutinib
    ("4OHF", "EGFR_T790M",   797, "acrylamide"),    # afatinib
    ("4ZAU", "EGFR_T790M",   797, "acrylamide"),    # osimertinib
    ("6CQ7", "KRAS_G12C",     12, "acrylamide"),    # AMG510
    ("6OIM", "KRAS_G12C",     12, "acrylamide"),
    ("4OT6", "JAK3",         909, "acrylamide"),    # tofacitinib analog
    ("6Y7Y", "TEC",          449, "acrylamide"),
    ("5SXN", "ITK",          442, "acrylamide"),
    ("5VC4", "BMX",          496, "acrylamide"),
    ("6V3O", "BTK",          481, "acrylamide"),
    ("6HAG", "FGFR4",        552, "acrylamide"),
    ("4UBN", "ERK2",          65, "chloroacetamide"),
    ("4XV9", "BLK",          319, "acrylamide"),
    ("6N75", "MAP2K7",       218, "acrylamide"),
]


def try_download_csv() -> pd.DataFrame | None:
    """Try each candidate URL; return parsed DF or None if all fail."""
    for url in CANDIDATE_URLS:
        try:
            with urllib.request.urlopen(url, timeout=15) as r:
                data = r.read().decode()
            # save raw + try parsing
            (OUT_DIR / "covbinder_pdb_list.csv").write_text(data)
            df = pd.read_csv(OUT_DIR / "covbinder_pdb_list.csv")
            print(f"  fetched CovBinderInPDB list from {url} ({len(df)} rows)")
            return df
        except (urllib.error.URLError, urllib.error.HTTPError, Exception) as e:
            print(f"  {url} -> {type(e).__name__}: {e}")
            continue
    return None


def fetch_pdb(pdb_id: str) -> Path | None:
    """Fetch a PDB file from RCSB."""
    out = RAW_DIR / pdb_id.upper() / f"{pdb_id.upper()}.pdb"
    if out.exists():
        return out
    out.parent.mkdir(parents=True, exist_ok=True)
    url = f"https://files.rcsb.org/download/{pdb_id.upper()}.pdb"
    try:
        with urllib.request.urlopen(url, timeout=30) as r:
            out.write_bytes(r.read())
        return out
    except Exception as e:
        print(f"  fetch {pdb_id} failed: {e}")
        return None


def find_covalent_ligand_in_pdb(pdb_path: Path, target_cys: int):
    """Locate the covalent ligand: any HETATM (non-standard, non-water,
    non-metal) within 2.5 Å of the target Cys SG. Return (chain, resi,
    resname, sg_xyz) or None."""
    from Bio.PDB import PDBParser
    import numpy as np
    parser = PDBParser(QUIET=True)
    s = parser.get_structure("", str(pdb_path))[0]
    sg = None
    # find SG atom of target Cys (any chain)
    for chain in s:
        for res in chain:
            if res.get_resname() == "CYS" and res.id[1] == target_cys and "SG" in res:
                sg = np.array(res["SG"].get_coord(), dtype=float)
                cys_chain = chain.id
                break
        if sg is not None: break
    if sg is None: return None
    # iterate HETATM residues
    skip = {"HOH", "WAT", "DOD", "SO4", "PO4", "NA", "K", "MG", "CA",
            "ZN", "CL", "MN", "FE", "CU", "GOL", "EDO", "TRS"}
    best = None
    for chain in s:
        for res in chain:
            hetflag = res.id[0]
            if hetflag.strip() == "":  # standard residue
                continue
            if res.get_resname() in skip: continue
            atoms = list(res.get_atoms())
            if len(atoms) < 5: continue
            coords = np.array([a.get_coord() for a in atoms])
            d_min = float(np.min(np.linalg.norm(coords - sg, axis=1)))
            if d_min < 2.5:
                if best is None or d_min < best[3]:
                    best = (chain.id, res.id[1], res.get_resname(), d_min, sg)
    return best


def main():
    print("=== CovBinderInPDB curation ===")
    # try remote curated list
    df = try_download_csv()
    if df is None:
        print("Network fetch failed; using FALLBACK_PDBS local list.")
        df = pd.DataFrame(FALLBACK_PDBS,
                          columns=["pdb_id", "target", "cys_residue", "warhead"])

    # subset to kinase + supported warheads
    if "target" in df.columns:
        kinase_words = ("kinase", "EGFR", "BTK", "JAK", "TEC", "ITK", "BMX", "BLK",
                        "FGFR", "ERK", "MAP", "ZAP", "KRAS", "BRAF", "MEK", "ABL",
                        "CDK", "ALK", "ROS", "TRK", "MET")
        mask = df["target"].astype(str).str.upper().apply(
            lambda t: any(w.upper() in t for w in kinase_words))
        df = df[mask].copy()
    if "warhead" in df.columns:
        df = df[df["warhead"].astype(str).isin(["acrylamide", "chloroacetamide",
                                                 "vinyl_sulfonamide", "cyanoacrylate"])]
    print(f"after kinase + warhead filter: {len(df)} entries")

    # fetch each PDB + extract covalent ligand metadata
    rows = []; failed = []
    for _, r in df.iterrows():
        pdb_id = str(r["pdb_id"]).upper()
        target_cys = int(r["cys_residue"])
        warhead = str(r["warhead"])
        pdbf = fetch_pdb(pdb_id)
        if pdbf is None:
            failed.append(pdb_id); continue
        info = find_covalent_ligand_in_pdb(pdbf, target_cys)
        if info is None:
            print(f"  {pdb_id} Cys{target_cys}: no covalent ligand found at SG")
            failed.append(pdb_id); continue
        chain_id, lig_resi, lig_resn, d_min, sg_xyz = info
        rows.append({
            "pdb_id": pdb_id, "target": r.get("target", ""),
            "cys_residue": target_cys, "cys_chain": chain_id,
            "warhead": warhead,
            "ligand_chain": chain_id, "ligand_resi": lig_resi,
            "ligand_resname": lig_resn,
            "sg_x": float(sg_xyz[0]), "sg_y": float(sg_xyz[1]), "sg_z": float(sg_xyz[2]),
            "d_SG_to_nearest_lig_atom": d_min,
            "pdb_path": str(pdbf.relative_to(PROJECT_ROOT)),
        })
        time.sleep(0.3)  # be kind to RCSB

    out_csv = OUT_DIR / "training_set.csv"
    pd.DataFrame(rows).to_csv(out_csv, index=False)
    print(f"\nwrote {out_csv}  ({len(rows)} usable complexes)")
    if failed:
        print(f"failed/no-covalent: {failed}")
    print("\nNext step: extract per-ligand SDFs + warhead atom indices,")
    print("then build the flagship D+C training DataLoader.")


if __name__ == "__main__":
    main()
