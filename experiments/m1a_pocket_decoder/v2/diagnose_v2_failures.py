#!/usr/bin/env python3
"""Diagnose why v2 extractor still misses ~2,000 rows.

Returns per-row failure step:
  smiles_invalid, missing_pdb, no_protein, no_nuc_residue, no_nuc_atom,
  no_het_lig, het_size_bad, no_warhead_match, no_b_xyz_mapping,
  d_b_nuc_too_far, pocket_too_small, pocket_too_big, ok
"""
from __future__ import annotations
import os, sys, warnings
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger

sys.path.insert(0, str(Path(__file__).parent))
from build_triples_v2 import (
    parse_pdb, find_warhead_atoms_smarts, smiles_to_pdb_atom_at_b,
    WARHEAD_NORM, NUC_ATOM_NAMES, HET_MIN, HET_MAX, POCKET_MIN, POCKET_MAX,
    D_B_NUC_MAX, POCKET_RADIUS, ONE_LETTER,
)

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
META_CSV = PROJECT_ROOT / "data/covbinder/raw_covindb2/Covalent_Complex_Records.csv"
PDB_DIR = PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB"
OUT_DIR = PROJECT_ROOT / "data/m1a_triples_v2"


def diagnose_row(args):
    try:
        return _diagnose_row_inner(args)
    except Exception as e:
        return ("exception", args[1].get("PDB"), args[1].get("Warhead"), str(e)[:80])


def _diagnose_row_inner(args):
    pdb_path_str, row = args
    pdb_path = Path(pdb_path_str)
    if not pdb_path.exists():
        return ("missing_pdb", row.get("PDB"), row.get("Warhead"))
    smi = row.get("SMILES")
    if not isinstance(smi, str) or len(smi) < 3:
        return ("smiles_invalid", row.get("PDB"), row.get("Warhead"))
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return ("smiles_invalid", row.get("PDB"), row.get("Warhead"))
    warhead_raw = row.get("Warhead")
    warhead_class = WARHEAD_NORM.get(warhead_raw,
                                       str(warhead_raw).lower().replace(" ", "_"))
    nuc_aa = row.get("Resi_name")
    try:
        nuc_posi = int(float(row.get("Resi_posi")))
    except (TypeError, ValueError):
        return ("smiles_invalid", row.get("PDB"), warhead_raw)
    nuc_chain = str(row.get("Resi_chain") or "").strip()[:1] or "A"
    lig_name = str(row.get("Ligand_name") or "").strip().upper()
    lig_chain = str(row.get("Ligand_chain") or "").strip()[:1] or nuc_chain
    try:
        lig_posi = int(float(row.get("Ligand_position")))
    except (TypeError, ValueError):
        lig_posi = None
    parsed = parse_pdb(pdb_path)
    if parsed is None:
        return ("no_protein", row.get("PDB"), warhead_raw)
    residues = parsed["residues"]
    hetatoms = parsed["hetatoms"]
    nuc_key = (nuc_chain, nuc_posi)
    nuc_xyz = None
    if nuc_key in residues:
        r = residues[nuc_key]
        atom_names = NUC_ATOM_NAMES.get(nuc_aa, ("CA",))
        for an in atom_names:
            if an in r["atoms"]:
                nuc_xyz = r["atoms"][an]; break
        if nuc_xyz is None and "CA" in r["atoms"]:
            nuc_xyz = r["atoms"]["CA"]
    if nuc_xyz is None:
        for (ch, rid), r in residues.items():
            if rid == nuc_posi and r["aa"] == nuc_aa:
                atom_names = NUC_ATOM_NAMES.get(nuc_aa, ("CA",))
                for an in atom_names:
                    if an in r["atoms"]:
                        nuc_xyz = r["atoms"][an]; nuc_chain = ch; break
                if nuc_xyz is not None: break
    if nuc_xyz is None:
        return ("no_nuc_residue", row.get("PDB"), warhead_raw)
    het_atoms_for_lig = None
    if lig_posi is not None:
        key = (lig_chain, lig_name, lig_posi)
        if key in hetatoms:
            het_atoms_for_lig = hetatoms[key]
    if het_atoms_for_lig is None:
        for (ch, rn, rp), atoms in hetatoms.items():
            if rn == lig_name:
                het_atoms_for_lig = atoms; break
    if het_atoms_for_lig is None:
        return ("no_het_lig", row.get("PDB"), warhead_raw)
    heavy = [a for a in het_atoms_for_lig if a["elem"] not in ("H", "D")]
    if not (HET_MIN <= len(heavy) <= HET_MAX):
        return ("het_size_bad", row.get("PDB"), warhead_raw)
    info = find_warhead_atoms_smarts(mol, warhead_class)
    bxyz = axyz = None
    if info is not None:
        ba = smiles_to_pdb_atom_at_b(mol, het_atoms_for_lig, info["b_idx"], info["a_idx"])
        if ba is not None:
            bxyz, axyz = ba
    if bxyz is None:
        best = None
        for ha in heavy:
            if ha["elem"] not in ("C", "B", "P", "S"):
                continue
            d = float(np.linalg.norm(ha["xyz"] - nuc_xyz))
            if best is None or d < best[0]:
                best = (d, ha)
        if best is None:
            return ("no_b_xyz_mapping", row.get("PDB"), warhead_raw)
        bxyz = best[1]["xyz"]
        nbrs = [(float(np.linalg.norm(a["xyz"] - bxyz)), a) for a in heavy
                if a is not best[1] and a["elem"] in ("C", "N", "O", "S")]
        if not nbrs:
            return ("no_b_xyz_mapping", row.get("PDB"), warhead_raw)
        nbrs.sort()
        axyz = nbrs[0][1]["xyz"]
    d_b_nuc = float(np.linalg.norm(bxyz - nuc_xyz))
    if d_b_nuc > D_B_NUC_MAX:
        return ("d_b_nuc_too_far", row.get("PDB"), warhead_raw, d_b_nuc)
    pocket = []
    for (ch, rid), r in residues.items():
        if ch != nuc_chain: continue
        if "CA" not in r["atoms"]: continue
        d_ca = float(np.linalg.norm(r["atoms"]["CA"] - bxyz))
        if d_ca <= POCKET_RADIUS:
            pocket.append(r)
    if len(pocket) < POCKET_MIN:
        return ("pocket_too_small", row.get("PDB"), warhead_raw, len(pocket))
    if len(pocket) > POCKET_MAX:
        return ("pocket_too_big", row.get("PDB"), warhead_raw, len(pocket))
    return ("ok", row.get("PDB"), warhead_raw)


def main():
    meta = pd.read_csv(META_CSV)
    work = []
    for _, row in meta.iterrows():
        pdb_id = str(row["PDB"]).lower()
        pdb_path = PDB_DIR / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            pdb_path = PDB_DIR / f"{pdb_id.upper()}.pdb"
        work.append((str(pdb_path), row.to_dict()))

    print(f"Diagnosing {len(work)} rows...", flush=True)
    results = []
    with ProcessPoolExecutor(max_workers=os.cpu_count()) as ex:
        futures = [ex.submit(diagnose_row, w) for w in work]
        done = 0
        for fut in as_completed(futures):
            results.append(fut.result())
            done += 1
            if done % 500 == 0:
                print(f"  {done}/{len(work)}", flush=True)

    statuses = Counter(r[0] for r in results)
    print()
    print("=== Status breakdown ===")
    for k, v in sorted(statuses.items(), key=lambda x: -x[1]):
        print(f"  {k:24s} {v}")

    # Per-step recovery potential
    rows = []
    for r in results:
        rows.append({"status": r[0], "pdb": r[1], "warhead": r[2]})
    df = pd.DataFrame(rows)
    df.to_csv(OUT_DIR / "v2_failure_diagnosis.csv", index=False)

    # Cross with warhead
    print()
    print("=== Status x warhead (failures only) ===")
    fail = df[df["status"] != "ok"]
    cross = fail.pivot_table(index="warhead", columns="status",
                              values="pdb", aggfunc="count", fill_value=0)
    print(cross.to_string())


if __name__ == "__main__":
    main()
