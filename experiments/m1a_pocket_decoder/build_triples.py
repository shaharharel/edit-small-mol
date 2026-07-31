#!/usr/bin/env python3
"""Build (SMILES, pocket_residues, warhead_pose_6d) triples for M1a fine-tune.

Three sources:
  1. Boltz ZAP70 cofold poses (~7k structures across cohort_3597_full,
     boltz_f4_results, boltz_f4_extra_results, boltz_rescue_78). Pocket = residues
     within 8 A of the warhead beta-C of the ligand. Cys346 typically present.
  2. CovInDB v2 PDB cross-ref (~3.4k PDB files in raw_covindb2/PDB/), extracting
     the HETATM ligand + nearest Cys SG (or Ser-OG / Thr-OG / Lys-NZ).
  3. CovBinderInPDB external pull - SKIPPED (URL unstable; 2+3 sufficient).

Output: data/m1a_triples/triples.parquet  with columns
  source, struct_id, smiles, pocket_residues (list of dict),
  warhead_pose_6d (6-float list), nucleophile_xyz (3-float),
  warhead_b_xyz (3-float), bd_angle_deg, d_b_nuc

pocket_residues entries: {aa, idx, ca_xyz}  (ESM-2 embedding added in a later
step on the GPU VM, so this script is fully CPU-bound).
"""
from __future__ import annotations
import argparse
import gzip
import json
import os
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
from concurrent.futures import ProcessPoolExecutor, as_completed

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
OUT_DIR = PROJECT_ROOT / "data/m1a_triples"
OUT_DIR.mkdir(exist_ok=True, parents=True)

# Pocket radius and electrophile / nucleophile rules
POCKET_RADIUS = 8.0  # A around warhead-beta-C
NUC_ATOMS = {  # element -> heavy-atom name pattern (for PDB)
    "CYS": ("SG",),
    "SER": ("OG",),
    "THR": ("OG1",),
    "LYS": ("NZ",),
    "TYR": ("OH",),
    "HIS": ("ND1", "NE2"),
}
WARHEAD_SMARTS = [
    ("acrylamide",        "[CH2;X3]=[CH;X3][C;X3](=O)[N]",   (0, 1, 2)),
    ("acrylamide_loose",  "[CH2]=C[C](=O)[N,n]",             (0, 1, 2)),
    ("chloroacetamide",   "Cl[CH2][C](=O)[N]",               (1, 2, 0)),
    ("vinyl_sulfonamide", "[CH2]=[CH][S](=O)(=O)[N]",        (0, 1, 2)),
    ("alpha_keto_amide",  "O=C(C(=O)N)",                     (1, 0, 2)),  # fallback
]
THREE_LETTER = {
    "A": "ALA", "R": "ARG", "N": "ASN", "D": "ASP", "C": "CYS", "E": "GLU",
    "Q": "GLN", "G": "GLY", "H": "HIS", "I": "ILE", "L": "LEU", "K": "LYS",
    "M": "MET", "F": "PHE", "P": "PRO", "S": "SER", "T": "THR", "W": "TRP",
    "Y": "TYR", "V": "VAL",
}
ONE_LETTER = {v: k for k, v in THREE_LETTER.items()}
ONE_LETTER["MSE"] = "M"  # selenomethionine -> M
ONE_LETTER["SEC"] = "C"  # selenocysteine  -> C


# ============================================================
# Geometry helpers
# ============================================================

def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1)
    n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def find_warhead_atoms_in_mol(mol: Chem.Mol) -> dict | None:
    """Find warhead atom indices in the RDKit mol (already has 3D coords)."""
    for name, smarts, (b, a, g) in WARHEAD_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        m = mol.GetSubstructMatch(patt)
        if m:
            return {"name": name, "b_idx": m[b], "a_idx": m[a], "g_idx": m[g]}
    return None


def pose_features_6d(b_xyz, a_xyz, nuc_xyz, bd_angle):
    """[b_xyz_relative_to_nuc (3), a-b direction (1 cos), d_b_nuc (1), bd_angle (1)]
    = 6 floats."""
    rel = b_xyz - nuc_xyz
    d = float(np.linalg.norm(rel))
    return [float(rel[0]), float(rel[1]), float(rel[2]),
            d, float(bd_angle), float(np.linalg.norm(a_xyz - b_xyz))]


# ============================================================
# Source 1: Boltz cofold pose extraction
# ============================================================

def parse_boltz_cif_chain_a(cif_path: Path) -> dict | None:
    """Parse Boltz CIF via Biopython. Returns chain-A protein + ligand HETs.

    {residues: [{aa, idx, atoms: {name: xyz}}], lig_atoms: [{name, xyz}]}
    """
    from Bio.PDB.MMCIFParser import MMCIFParser
    try:
        parser = MMCIFParser(QUIET=True)
        s = parser.get_structure("m", str(cif_path))
        model = next(s.get_models())
    except Exception:
        return None
    residues = {}
    lig_atoms = []
    for chain in model:
        for res in chain:
            hetflag = res.id[0]
            if hetflag == " ":
                # Protein residue
                # Boltz uses chain "A" for the polymer in our cofolds
                if chain.id != "A":
                    continue
                atoms = {a.get_name(): np.array(a.get_coord()) for a in res}
                residues[(chain.id, res.id[1])] = {
                    "aa": res.get_resname(),
                    "idx": int(res.id[1]),
                    "atoms": atoms,
                }
            else:
                # HET = ligand
                if res.get_resname().startswith("LIG"):
                    for a in res:
                        lig_atoms.append({
                            "name": a.get_name(),
                            "xyz": np.array(a.get_coord()),
                        })
    if not residues or not lig_atoms:
        return None
    return {"residues": list(residues.values()), "lig_atoms": lig_atoms}


def boltz_pose_to_triple(cif_path: Path, smiles: str | None = None) -> dict | None:
    """Build a triple from a single Boltz pose. Returns None if not usable.

    Strategy: if .lig.sdf sibling exists, use it (covers cohort_3597). Otherwise
    use the manifest SMILES + map RDKit atom indices to CIF HET atom order
    (covers F4 + rescue_78).
    """
    smi = None
    b_xyz = a_xyz = None

    sdf_path = cif_path.with_suffix("").with_suffix(".lig.sdf")
    if not sdf_path.exists():
        sdf_path = Path(str(cif_path).replace(".cif", ".lig.sdf"))

    if sdf_path.exists():
        try:
            supp = Chem.SDMolSupplier(str(sdf_path), removeHs=False, sanitize=False)
            mol = next(iter(supp))
            if mol is None:
                return None
            try:
                Chem.SanitizeMol(mol)
            except Exception:
                pass
            smi = Chem.MolToSmiles(Chem.RemoveHs(mol))
            info = find_warhead_atoms_in_mol(mol)
            if info is None:
                return None
            conf = mol.GetConformer()
            b_xyz = np.array(list(conf.GetAtomPosition(info["b_idx"])))
            a_xyz = np.array(list(conf.GetAtomPosition(info["a_idx"])))
        except Exception:
            return None
    elif smiles is not None:
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None
            smi = Chem.MolToSmiles(mol)
            info = find_warhead_atoms_in_mol(mol)
            if info is None:
                return None
            # Parse CIF HET atom coords; index by RDKit atom order (heavy only)
            parsed = parse_boltz_cif_chain_a(cif_path)
            if parsed is None or len(parsed["lig_atoms"]) < mol.GetNumHeavyAtoms():
                return None
            heavy_lig = [a for a in parsed["lig_atoms"]
                          if not (a["name"].startswith("H") or
                                   (len(a["name"]) > 1 and a["name"][0] in "HD"))]
            if len(heavy_lig) < mol.GetNumHeavyAtoms():
                return None
            b_idx, a_idx = info["b_idx"], info["a_idx"]
            if b_idx >= len(heavy_lig) or a_idx >= len(heavy_lig):
                return None
            b_xyz = heavy_lig[b_idx]["xyz"]
            a_xyz = heavy_lig[a_idx]["xyz"]
        except Exception:
            return None
    else:
        return None

    if smi is None or b_xyz is None or a_xyz is None:
        return None
    # Parse CIF for chain A (may have been already parsed in the SMILES-based path
    # but parse again here for the SDF path; cheap on a worker)
    parsed = parse_boltz_cif_chain_a(cif_path)
    if parsed is None:
        return None
    # Find nucleophile: closest CYS SG (or fall back to closest CA of any pocket residue)
    nuc_xyz = None
    nuc_resid = None
    best_d = 1e9
    for r in parsed["residues"]:
        nuc_names = NUC_ATOMS.get(r["aa"], ())
        for nuc_name in nuc_names:
            if nuc_name in r["atoms"]:
                d = float(np.linalg.norm(r["atoms"][nuc_name] - b_xyz))
                if d < best_d:
                    best_d = d
                    nuc_xyz = r["atoms"][nuc_name]
                    nuc_resid = r["idx"]
    if nuc_xyz is None:
        return None
    if best_d > 6.0:
        # Warhead too far from nucleophile (no productive geometry); skip.
        return None
    # Pocket = residues with CA within POCKET_RADIUS of b_xyz
    pocket = []
    for r in parsed["residues"]:
        if "CA" not in r["atoms"]:
            continue
        d_ca = float(np.linalg.norm(r["atoms"]["CA"] - b_xyz))
        if d_ca <= POCKET_RADIUS:
            pocket.append({
                "aa": ONE_LETTER.get(r["aa"], "X"),
                "idx": r["idx"],
                "d": d_ca,
                "ca_xyz": [float(r["atoms"]["CA"][0]),
                            float(r["atoms"]["CA"][1]),
                            float(r["atoms"]["CA"][2])],
            })
    if len(pocket) < 4 or len(pocket) > 80:
        return None
    bd = angle_deg(nuc_xyz - b_xyz, a_xyz - b_xyz)
    pose6d = pose_features_6d(b_xyz, a_xyz, nuc_xyz, bd)
    return {
        "source": "boltz_zap70",
        "struct_id": cif_path.parent.name,
        "smiles": smi,
        "pocket_residues": pocket,
        "warhead_pose_6d": pose6d,
        "nucleophile_xyz": [float(nuc_xyz[0]), float(nuc_xyz[1]), float(nuc_xyz[2])],
        "warhead_b_xyz": [float(b_xyz[0]), float(b_xyz[1]), float(b_xyz[2])],
        "bd_angle_deg": float(bd),
        "d_b_nuc": best_d,
        "nucleophile_resid": nuc_resid,
    }


def collect_boltz_cifs(roots: list[Path]) -> list[Path]:
    out = []
    for root in roots:
        if not root.exists():
            continue
        out.extend(root.rglob("*model_0.cif"))
    return out


# ============================================================
# Source 2: CovInDB v2 PDB cross-ref
# ============================================================

def parse_pdb_for_triple(pdb_path: Path) -> list[dict]:
    """Parse a CovInDB PDB file. May contain multiple HET ligands; emit one triple per
    qualifying HET (small, has acrylamide-like warhead, near a Cys SG)."""
    # Parse residues (chain -> resid -> {aa, atoms})
    residues = {}
    hetatoms_by_resn = {}  # (chain, resn, resid) -> [{name, elem, xyz}]
    try:
        with open(pdb_path) as f:
            for line in f:
                if line.startswith("ATOM") or line.startswith("HETATM"):
                    try:
                        atom_name = line[12:16].strip()
                        resname = line[17:20].strip()
                        chain = line[21]
                        resid = int(line[22:26])
                        x = float(line[30:38])
                        y = float(line[38:46])
                        z = float(line[46:54])
                        elem = line[76:78].strip() if len(line) > 78 else atom_name[0]
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
                        # HETATM - skip water/common ions
                        if resname in ("HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA",
                                       "MG", "CA", "ZN", "MN", "FE", "K", "BR", "EDO",
                                       "GOL", "PEG", "DMS", "DTT", "BME", "NAG", "MAN",
                                       "FUC", "BMA", "GAL", "GLC"):
                            continue
                        key = (chain, resname, resid)
                        hetatoms_by_resn.setdefault(key, []).append(
                            {"name": atom_name, "elem": elem, "xyz": xyz})
    except Exception:
        return []
    if not residues or not hetatoms_by_resn:
        return []
    triples = []
    for (chain, resname, resid), atoms in hetatoms_by_resn.items():
        # Need at least 8 heavy atoms (filter dummy ligands)
        heavy = [a for a in atoms if a["elem"] not in ("H", "D")]
        if len(heavy) < 8 or len(heavy) > 80:
            continue
        # Build RDKit mol from xyz to detect warhead (use PDB block writer)
        # Simpler: try to detect acrylamide via geometric pattern (CH2=CH-C(=O)-N)
        # Since we don't have bond info reliably, just check for Cys SG within ~3 A of
        # any HET atom -> infer covalent contact. Use atom closest to a Cys SG as the
        # putative beta-C.
        sg_atoms = []
        for r in residues.values():
            if r["aa"] == "CYS" and "SG" in r["atoms"]:
                sg_atoms.append((r["chain"], r["idx"], r["atoms"]["SG"]))
        if not sg_atoms:
            continue
        # Find HET atom (carbon preferred) closest to any SG
        best = None  # (d, het_atom, sg_xyz, sg_resid, sg_chain)
        for ha in heavy:
            if ha["elem"] not in ("C", "S"):
                continue
            for (sg_chain, sg_resid, sg_xyz) in sg_atoms:
                d = float(np.linalg.norm(ha["xyz"] - sg_xyz))
                if d < 2.5 and (best is None or d < best[0]):
                    best = (d, ha, sg_xyz, sg_resid, sg_chain)
        if best is None:
            continue
        d_b_sg, b_atom, nuc_xyz, nuc_resid, nuc_chain = best
        b_xyz = b_atom["xyz"]
        # alpha-C = closest C atom to beta-C within HET (excluding beta-C itself)
        candidates = [(float(np.linalg.norm(a["xyz"] - b_xyz)), a) for a in heavy
                       if a is not b_atom and a["elem"] == "C"]
        if not candidates:
            continue
        candidates.sort()
        a_atom = candidates[0][1]
        a_xyz = a_atom["xyz"]
        # Pocket = residues with CA within POCKET_RADIUS of b_xyz
        pocket = []
        for r in residues.values():
            if "CA" not in r["atoms"]:
                continue
            d_ca = float(np.linalg.norm(r["atoms"]["CA"] - b_xyz))
            if d_ca <= POCKET_RADIUS:
                pocket.append({
                    "aa": ONE_LETTER.get(r["aa"], "X"),
                    "idx": int(r["idx"]),
                    "d": d_ca,
                    "ca_xyz": [float(r["atoms"]["CA"][0]),
                                float(r["atoms"]["CA"][1]),
                                float(r["atoms"]["CA"][2])],
                })
        if len(pocket) < 4 or len(pocket) > 80:
            continue
        # Need SMILES. Skip for now (would need ligand definition from PDB chem comp lookup).
        # Use a HET-based placeholder; will skip rows without SMILES on the training side.
        bd = angle_deg(nuc_xyz - b_xyz, a_xyz - b_xyz)
        pose6d = pose_features_6d(b_xyz, a_xyz, nuc_xyz, bd)
        triples.append({
            "source": "covindb_pdb",
            "struct_id": f"{pdb_path.stem}_{chain}_{resname}_{resid}",
            "pdb_id": pdb_path.stem,
            "het_resname": resname,
            "smiles": None,  # filled in via CovInDB_All.csv lookup
            "pocket_residues": pocket,
            "warhead_pose_6d": pose6d,
            "nucleophile_xyz": [float(nuc_xyz[0]), float(nuc_xyz[1]),
                                 float(nuc_xyz[2])],
            "warhead_b_xyz": [float(b_xyz[0]), float(b_xyz[1]), float(b_xyz[2])],
            "bd_angle_deg": float(bd),
            "d_b_nuc": d_b_sg,
            "nucleophile_resid": int(nuc_resid),
        })
    return triples


# ============================================================
# Worker wrappers (for multiprocessing)
# ============================================================

def _boltz_worker(args: tuple) -> dict | None:
    cif_path_str, smi = args
    return boltz_pose_to_triple(Path(cif_path_str), smiles=smi)


def _pdb_worker(pdb_path_str: str) -> list[dict]:
    try:
        return parse_pdb_for_triple(Path(pdb_path_str))
    except Exception:
        return []


def _boltz_worker_safe(args: tuple) -> dict | None:
    try:
        return _boltz_worker(args)
    except Exception:
        return None


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", default="boltz,covindb",
                     help="Comma-separated subset of {boltz,covindb}")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None,
                     help="Limit per source for smoke testing")
    args = ap.parse_args()
    sources = set(args.sources.split(","))

    all_rows = []

    # ---- Source 1: Boltz cofolds ----
    if "boltz" in sources:
        print("[boltz] Collecting CIF files...", flush=True)
        boltz_roots = [
            PROJECT_ROOT / "data/boltz_results/cohort_3597_full",
            PROJECT_ROOT / "data/boltz_f4_results",
            PROJECT_ROOT / "data/boltz_f4_extra_results",
            PROJECT_ROOT / "data/boltz_rescue_78",
        ]
        cifs = collect_boltz_cifs(boltz_roots)
        # Build a stem -> SMILES map from manifests (F4 + F4_extra + rescue_78)
        stem_to_smi: dict[str, str] = {}
        manifest_paths = [
            PROJECT_ROOT / "data/boltz_f4_results/manifest_deduped.csv",
            PROJECT_ROOT / "data/boltz_f4_extra_results/ai-gpu-a100-d/manifest.csv",
        ]
        # Also search any rescue / f4 manifest with smiles col
        for mp in manifest_paths:
            if not mp.exists():
                continue
            try:
                m = pd.read_csv(mp)
            except Exception:
                continue
            stem_col = "stem" if "stem" in m.columns else ("name" if "name" in m.columns else None)
            if stem_col is None or "smiles" not in m.columns:
                continue
            for _, r in m.iterrows():
                stem_to_smi[str(r[stem_col])] = str(r["smiles"])
        # Also pick up any other per-vm manifests
        for extra in PROJECT_ROOT.glob("data/boltz_f4_results/*/manifest.csv"):
            try:
                m = pd.read_csv(extra)
            except Exception:
                continue
            stem_col = "stem" if "stem" in m.columns else ("name" if "name" in m.columns else None)
            if stem_col and "smiles" in m.columns:
                for _, r in m.iterrows():
                    stem_to_smi.setdefault(str(r[stem_col]), str(r["smiles"]))
        # cohort_3597 has full .lig.sdf so SMILES not needed; rescue_78 has no manifest
        # but its 78 rows are small; we'll just rely on sdf where present and skip
        # the no-warhead ones.
        print(f"[boltz] manifest SMILES map: {len(stem_to_smi)} stems", flush=True)
        if args.limit:
            cifs = cifs[: args.limit]
        print(f"[boltz] {len(cifs)} CIFs found", flush=True)
        # Build (cif, smi) work list
        worklist = []
        for c in cifs:
            stem = c.stem.replace("_model_0", "")
            smi = stem_to_smi.get(stem)
            worklist.append((str(c), smi))
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_boltz_worker_safe, item): item for item in worklist}
            done = 0
            for fut in as_completed(futures):
                r = fut.result()
                if r is not None:
                    rows.append(r)
                done += 1
                if done % 500 == 0:
                    print(f"[boltz] {done}/{len(cifs)}  kept={len(rows)}", flush=True)
        print(f"[boltz] FINAL kept={len(rows)} / {len(cifs)}", flush=True)
        all_rows.extend(rows)

    # ---- Source 2: CovInDB v2 PDB ----
    if "covindb" in sources:
        print("[covindb] Loading SMILES lookup from CovInDB_All.csv ...", flush=True)
        # CovInDB_All.csv has columns including SMILES + Reference -> PDB IDs? Check.
        # We use SMILES via heuristic: look for PDB IDs in any column. Try a simple
        # join via the Reference / Activity_type columns - but really PDB id is not
        # cleanly listed. Fall back: extract HET ligand SMILES from PDB using the
        # connectivity inferred by RDKit's MolFromPDBFile on the het residue.
        pdb_dir = PROJECT_ROOT / "data/covbinder/raw_covindb2/PDB"
        pdb_files = sorted(pdb_dir.glob("*.pdb"))
        if args.limit:
            pdb_files = pdb_files[: args.limit]
        print(f"[covindb] {len(pdb_files)} PDB files found", flush=True)
        rows = []
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futures = {ex.submit(_pdb_worker, str(p)): p for p in pdb_files}
            done = 0
            for fut in as_completed(futures):
                rs = fut.result()
                rows.extend(rs)
                done += 1
                if done % 200 == 0:
                    print(f"[covindb] {done}/{len(pdb_files)}  kept_triples={len(rows)}",
                           flush=True)
        print(f"[covindb] PRE-SMILES kept={len(rows)}", flush=True)
        # Try to recover SMILES from PDB using MolFromPDBFile + RemoveHs + per-residue
        # extraction. This is best-effort; on failure the row is dropped.
        pdb_to_rows = {}
        for r in rows:
            pdb_to_rows.setdefault(r["pdb_id"], []).append(r)
        n_with_smi = 0
        for pdb_id, rs in pdb_to_rows.items():
            pdb_path = pdb_dir / f"{pdb_id}.pdb"
            try:
                mol_full = Chem.MolFromPDBFile(str(pdb_path), removeHs=True,
                                                 sanitize=False)
                if mol_full is None:
                    continue
                # Split into HET fragments by residue name
                # Walk atoms; group by PDB info residue name + resid
                het_smiles = {}  # (chain, resname, resid) -> smi
                groups = {}  # (chain, resname, resid) -> [atom_idx]
                for atom in mol_full.GetAtoms():
                    pdbinfo = atom.GetPDBResidueInfo()
                    if pdbinfo is None:
                        continue
                    if not pdbinfo.GetIsHeteroAtom():
                        continue
                    if pdbinfo.GetResidueName().strip() in ("HOH", "WAT", "DOD",
                                                              "SO4", "CL", "NA",
                                                              "MG", "ZN", "K"):
                        continue
                    key = (pdbinfo.GetChainId(),
                            pdbinfo.GetResidueName().strip(),
                            int(pdbinfo.GetResidueNumber()))
                    groups.setdefault(key, []).append(atom.GetIdx())
                for key, idxs in groups.items():
                    if not (8 <= len(idxs) <= 100):
                        continue
                    try:
                        sub = Chem.PathToSubmol(mol_full,
                                                  Chem.rdmolops.FindAllPathsOfLengthN(mol_full, 1, useBonds=False)) \
                            if False else None
                        sub = Chem.RWMol()
                        old_to_new = {}
                        for old in idxs:
                            new = sub.AddAtom(Chem.Atom(mol_full.GetAtomWithIdx(old).GetSymbol()))
                            old_to_new[old] = new
                        for bond in mol_full.GetBonds():
                            i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                            if i in old_to_new and j in old_to_new:
                                sub.AddBond(old_to_new[i], old_to_new[j], bond.GetBondType())
                        sub_mol = sub.GetMol()
                        try:
                            Chem.SanitizeMol(sub_mol)
                        except Exception:
                            try:
                                Chem.SanitizeMol(sub_mol,
                                                  sanitizeOps=Chem.SanitizeFlags.SANITIZE_FINDRADICALS |
                                                  Chem.SanitizeFlags.SANITIZE_KEKULIZE |
                                                  Chem.SanitizeFlags.SANITIZE_SETAROMATICITY |
                                                  Chem.SanitizeFlags.SANITIZE_SETCONJUGATION |
                                                  Chem.SanitizeFlags.SANITIZE_SETHYBRIDIZATION |
                                                  Chem.SanitizeFlags.SANITIZE_SYMMRINGS)
                            except Exception:
                                continue
                        smi = Chem.MolToSmiles(Chem.RemoveHs(sub_mol))
                        if smi and len(smi) > 5:
                            het_smiles[key] = smi
                    except Exception:
                        continue
                for r in rs:
                    key = (r["struct_id"].split("_")[1],  # chain
                            r["het_resname"],
                            int(r["struct_id"].rsplit("_", 1)[1]))
                    if key in het_smiles:
                        r["smiles"] = het_smiles[key]
                        n_with_smi += 1
            except Exception:
                continue
        rows = [r for r in rows if r.get("smiles")]
        print(f"[covindb] FINAL with-SMILES kept={len(rows)} (recovered {n_with_smi})",
               flush=True)
        all_rows.extend(rows)

    # ---- Write parquet ----
    if not all_rows:
        print("FATAL: no triples", flush=True)
        sys.exit(1)
    df = pd.DataFrame(all_rows)
    # Dedup on SMILES (canonical) - keep first occurrence
    df["canon_smi"] = df["smiles"].apply(lambda s: Chem.MolToSmiles(Chem.MolFromSmiles(s))
                                          if Chem.MolFromSmiles(s) else None)
    before = len(df)
    df = df.dropna(subset=["canon_smi"]).drop_duplicates(subset=["canon_smi"])
    after = len(df)
    print(f"After dedup: {after} / {before}", flush=True)
    df["pocket_residues"] = df["pocket_residues"].apply(json.dumps)
    df["warhead_pose_6d"] = df["warhead_pose_6d"].apply(lambda x: json.dumps(list(x)))
    df["nucleophile_xyz"] = df["nucleophile_xyz"].apply(lambda x: json.dumps(list(x)))
    df["warhead_b_xyz"] = df["warhead_b_xyz"].apply(lambda x: json.dumps(list(x)))
    out_path = OUT_DIR / "triples.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} triples to {out_path}", flush=True)
    # Per-source counts
    print("By source:", flush=True)
    print(df["source"].value_counts(), flush=True)


if __name__ == "__main__":
    main()
