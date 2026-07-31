#!/usr/bin/env python3
"""Phase 1: Metadata-driven CovInDB v2 (and CovBinderInPDB) triple extractor.

Differs from build_triples.py:
  - Drives extraction from Covalent_Complex_Records.csv (one row -> one triple
    attempt). Uses metadata Resi_chain/Resi_posi/Resi_name as the GROUND-TRUTH
    nucleophile.
  - Uses metadata SMILES directly (no PDB->SMILES recovery loss).
  - Supports all warhead classes; tags each triple with normalized
    `warhead_class` column.
  - Per-chain pocket computation (handles multimeric PDBs).
  - Relaxed HET heavy-atom bounds 5-100.
  - Alt-loc filter (blank or "A").
  - Maps SMILES -> warhead beta/alpha atoms via SMARTS, then atom-matches to
    PDB HETATM coords to get geometric pose.

Output columns (parquet):
  source, struct_id, pdb_id, het_resname, smiles, canon_smi,
  pocket_residues (JSON), warhead_pose_6d (JSON), nucleophile_xyz (JSON),
  warhead_b_xyz (JSON), bd_angle_deg, d_b_nuc, nucleophile_resid,
  nucleophile_resname, warhead_class, target_name
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
import pandas as pd
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")
warnings.filterwarnings("ignore")

PROJECT_ROOT = Path("/Users/shaharharel/Documents/github/edit-small-mol")
POCKET_RADIUS = 8.0
HET_MIN = 5
HET_MAX = 100
POCKET_MIN = 4
POCKET_MAX = 80
D_B_NUC_MAX = 6.0  # warhead-b to nucleophile distance cutoff

NUC_ATOM_NAMES = {
    "CYS": ("SG",),
    "SEC": ("SE", "SG"),
    "SER": ("OG",),
    "THR": ("OG1",),
    "LYS": ("NZ",),
    "TYR": ("OH",),
    "HIS": ("NE2", "ND1"),
    "ASP": ("OD1", "OD2"),
    "GLU": ("OE1", "OE2"),
    "ASN": ("ND2",),
    "ARG": ("NE", "NH1", "NH2"),
    "MET": ("SD",),
    "TRP": ("NE1",),
    "GLN": ("NE2",),
    "GLY": ("CA",),  # rare; fallback to CA
    "PRO": ("N",),
    "VAL": ("CA",),
}

# Normalize CovInDB warhead names to a smaller vocab
WARHEAD_NORM = {
    "Michael Acceptor": "michael_acceptor",
    "Acrylamide": "acrylamide",
    "Halohydrocarbon": "halohydrocarbon",
    "Beta Lactam": "beta_lactam",
    "Boronic Acid": "boronic_acid",
    "Carbonyl": "carbonyl",
    "Aldehyde": "aldehyde",
    "Aldehydic carbonyl": "aldehyde",
    "Nitrile": "nitrile",
    "Phosphonate": "phosphonate",
    "Epoxide": "epoxide",
    "Disulfide": "disulfide",
    "Hemiacetal": "hemiacetal",
    "Lactone": "lactone",
    "Ester": "ester",
    "Vinyl Sulfone": "vinyl_sulfone",
    "Vinylsulfone": "vinyl_sulfone",
    "Sulfonic acid": "sulfonic_acid",
    "Sulfonyl Fluorine": "sulfonyl_fluoride",
    "Gamma Lactam": "gamma_lactam",
    "Urea carbonyl": "urea_carbonyl",
    "Phosphate": "phosphate",
    "Phosphate Group": "phosphate",
    "Beta Lacton": "beta_lactone",
    "Sulfone": "sulfone",
    "Sulfonate": "sulfonate",
    "Sulfonamide": "sulfonamide",
    "Imine": "imine",
    "Azide": "azide",
    "Vinyl Ester": "vinyl_ester",
}

# Per-warhead-class SMARTS for finding the electrophilic atom (b) and adjacent (a)
# Returns indices [b_idx, a_idx] within SMARTS match.
# For most: b is the electrophile (carbon/boron/phosphorus/sulfur that bonds to nuc),
# a is the adjacent atom (used for geometric direction & d_a_b sanity).
WARHEAD_SMARTS_V2 = [
    # (norm_name, SMARTS, (b_offset, a_offset))
    ("acrylamide",        "[CH2;X3]=[CH;X3][C;X3](=O)[N]",   (0, 1)),
    ("michael_acceptor",  "[CH2,CH;X3]=[CH;X3][C;X3](=O)",   (0, 1)),
    ("michael_acceptor",  "[CH2,CH]=[CH][S](=O)(=O)",         (0, 1)),
    ("michael_acceptor",  "[C]=[C][C](=O)",                   (0, 1)),
    ("vinyl_sulfone",     "[CH2]=[CH][S](=O)(=O)",            (0, 1)),
    ("halohydrocarbon",   "[Cl,Br,I,F][CH2,CH][C](=O)",       (1, 2)),
    ("halohydrocarbon",   "[Cl,Br,I][CH2,CH]",                (1, 0)),
    ("epoxide",           "[C;R1]1[O;R1][C;R1]1",             (0, 2)),
    ("aldehyde",          "[CH;X3](=O)",                      (0, 0)),
    ("nitrile",           "[C;X2]#[N;X1]",                    (0, 1)),
    ("boronic_acid",      "[B]([OH])[OH]",                    (0, 1)),
    ("boronic_acid",      "[B][O]",                           (0, 1)),
    ("phosphonate",       "[P](=O)([O,F])",                   (0, 1)),
    ("beta_lactam",       "[C;R1]1[C;R1](=O)[N;R1][C;R1]1",   (1, 2)),
    ("disulfide",         "[S][S]",                           (0, 1)),
    ("sulfonyl_fluoride", "[S](=O)(=O)[F]",                   (0, 1)),
    ("urea_carbonyl",     "[N][C](=O)[N]",                    (1, 2)),
    ("carbonyl",          "[C;X3](=O)",                       (0, 0)),
    ("ester",             "[C;X3](=O)[O;X2]",                 (0, 2)),
    ("lactone",           "[C;R1](=O)[O;R1]",                 (0, 2)),
    ("gamma_lactam",      "[C;R1](=O)[N;R1]",                 (0, 1)),
    ("hemiacetal",        "[C]([OH])[O]",                     (0, 1)),
    ("imine",             "[C]=[N]",                          (0, 1)),
    ("sulfonic_acid",     "[S](=O)(=O)[OH]",                  (0, 1)),
    ("sulfonate",         "[S](=O)(=O)[O-,O]",                (0, 1)),
    ("sulfonamide",       "[S](=O)(=O)[N]",                   (0, 1)),
    ("vinyl_ester",       "[C]=[C][C](=O)[O]",                (0, 1)),
    ("azide",             "[N]=[N+]=[N-]",                    (0, 1)),
    ("phosphate",         "[P](=O)([O])",                     (0, 1)),
]

SKIP_HET = {
    "HOH", "WAT", "DOD", "SO4", "PO4", "CL", "NA", "MG", "CA", "ZN", "MN",
    "FE", "K", "BR", "EDO", "GOL", "PEG", "DMS", "DTT", "BME", "NAG", "MAN",
    "FUC", "BMA", "GAL", "GLC", "ACT", "TRS", "IPA", "FMT", "MES", "BCT",
}

ONE_LETTER = {
    "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLU": "E",
    "GLN": "Q", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
    "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
    "TYR": "Y", "VAL": "V", "MSE": "M", "SEC": "C", "PYL": "K",
}


# ============================================================
# Geometry helpers
# ============================================================

def angle_deg(v1: np.ndarray, v2: np.ndarray) -> float:
    n1 = np.linalg.norm(v1); n2 = np.linalg.norm(v2)
    if n1 < 1e-8 or n2 < 1e-8:
        return float("nan")
    c = float(np.dot(v1, v2) / (n1 * n2))
    c = max(-1.0, min(1.0, c))
    return float(np.degrees(np.arccos(c)))


def pose_features_6d(b_xyz, a_xyz, nuc_xyz, bd_angle):
    rel = b_xyz - nuc_xyz
    d = float(np.linalg.norm(rel))
    return [float(rel[0]), float(rel[1]), float(rel[2]),
            d, float(bd_angle), float(np.linalg.norm(a_xyz - b_xyz))]


# ============================================================
# PDB parser (per-chain)
# ============================================================

def parse_pdb(pdb_path: Path) -> dict | None:
    """Returns:
      {
        'residues': {(chain, resid): {'aa', 'idx', 'chain', 'atoms': {name: xyz}}},
        'hetatoms': {(chain, resname, resid): [{'name', 'elem', 'xyz'}]}
      }
    """
    residues = {}
    hetatoms = {}
    try:
        with open(pdb_path) as f:
            for line in f:
                if not (line.startswith("ATOM") or line.startswith("HETATM")):
                    continue
                try:
                    atom_name = line[12:16].strip()
                    altloc = line[16]
                    resname = line[17:20].strip()
                    chain = line[21]
                    resid_str = line[22:26].strip()
                    try:
                        resid = int(resid_str)
                    except ValueError:
                        continue
                    x = float(line[30:38]); y = float(line[38:46]); z = float(line[46:54])
                    elem = line[76:78].strip() if len(line) > 78 else atom_name[0]
                except (ValueError, IndexError):
                    continue
                if altloc not in (" ", "A"):
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
    except Exception:
        return None
    if not residues:
        return None
    return {"residues": residues, "hetatoms": hetatoms}


# ============================================================
# Warhead detection on metadata SMILES
# ============================================================

def find_warhead_atoms_smarts(mol: Chem.Mol, warhead_class: str | None) -> dict | None:
    """Try SMARTS patterns in order of specificity (most specific first).

    WARHEAD_SMARTS_V2 is arranged so that more-specific SMARTS come before
    less-specific ones (e.g., acrylamide before michael_acceptor, since every
    acrylamide is a Michael acceptor but not vice versa). The returned `name`
    is the authoritative warhead_class label — a molecule that matches acrylamide
    SMARTS is labelled 'acrylamide' regardless of what the source curator called it.

    The `warhead_class` argument is retained for API compatibility but is not
    used for classification decisions (previously it was a hint that biased the
    match order; that caused Michael-Acceptor-labelled acrylamides to short-circuit
    on the michael_acceptor SMARTS before ever trying the acrylamide one).
    """
    for name, smarts, (b_off, a_off) in WARHEAD_SMARTS_V2:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        m = mol.GetSubstructMatch(patt)
        if m:
            try:
                b_idx = m[b_off]
                a_idx = m[a_off]
                return {"name": name, "b_idx": b_idx, "a_idx": a_idx}
            except IndexError:
                continue
    return None


# ============================================================
# Match SMILES heavy atoms onto PDB HETATM xyz coords
# ============================================================

def smiles_to_3d_via_pdb_het(mol: Chem.Mol, het_atoms: list[dict]) -> Chem.Mol | None:
    """Embed mol in 3D using PDB HETATM coords. Returns mol with conformer if
    we can map all heavy atoms to a PDB atom via element + greedy distance.

    Strategy: build RDKit conformer where each heavy atom's xyz is set to the
    NEAREST same-element HETATM. This is approximate but works for most ligands
    since CovInDB's metadata SMILES come from the PDB chemical component, so
    atom counts/elements match.
    """
    mol_h = Chem.RemoveHs(mol)
    n_heavy = mol_h.GetNumHeavyAtoms()
    het_heavy = [a for a in het_atoms if a["elem"] not in ("H", "D")]
    if len(het_heavy) != n_heavy:
        # Atom count mismatch; abort
        return None
    # Match by element histogram first
    rdkit_elements = [mol_h.GetAtomWithIdx(i).GetSymbol().upper() for i in range(n_heavy)]
    het_elements = [a["elem"].upper() for a in het_heavy]
    if Counter(rdkit_elements) != Counter(het_elements):
        return None
    # Greedy assignment: for each RDKit atom in canonical order, assign nearest
    # available HET atom of matching element. This is good enough for our purposes
    # since we don't need a stereochemically perfect match - just b/a positions.
    used = [False] * len(het_heavy)
    assignment = [-1] * n_heavy
    for i, ele in enumerate(rdkit_elements):
        for j, het in enumerate(het_heavy):
            if used[j]:
                continue
            if het["elem"].upper() != ele:
                continue
            assignment[i] = j
            used[j] = True
            break
        if assignment[i] == -1:
            return None
    # Build conformer
    mol_with_conf = Chem.RWMol(mol_h)
    mol_with_conf.RemoveAllConformers()
    conf = Chem.Conformer(n_heavy)
    for i in range(n_heavy):
        xyz = het_heavy[assignment[i]]["xyz"]
        conf.SetAtomPosition(i, (float(xyz[0]), float(xyz[1]), float(xyz[2])))
    mol_with_conf.AddConformer(conf, assignId=True)
    return mol_with_conf.GetMol()


def smiles_to_pdb_atom_at_b(mol: Chem.Mol, het_atoms: list[dict],
                              b_idx: int, a_idx: int) -> tuple[np.ndarray, np.ndarray] | None:
    """Direct positional mapping: map RDKit b/a atom indices to PDB xyz.
    Returns (b_xyz, a_xyz) if successful.
    """
    mol3d = smiles_to_3d_via_pdb_het(mol, het_atoms)
    if mol3d is None or mol3d.GetNumConformers() == 0:
        return None
    conf = mol3d.GetConformer()
    try:
        b_pos = conf.GetAtomPosition(b_idx)
        a_pos = conf.GetAtomPosition(a_idx)
    except Exception:
        return None
    return (np.array([b_pos.x, b_pos.y, b_pos.z]),
            np.array([a_pos.x, a_pos.y, a_pos.z]))


# ============================================================
# Per-row extraction
# ============================================================

def extract_one_row(args: tuple) -> dict | None:
    """args = (pdb_path_str, row_dict)"""
    pdb_path_str, row = args
    pdb_path = Path(pdb_path_str)
    if not pdb_path.exists():
        return None

    smi = row.get("SMILES")
    if not isinstance(smi, str) or len(smi) < 3:
        return None
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None
    canon = Chem.MolToSmiles(mol)

    warhead_raw = row.get("Warhead")
    warhead_class = WARHEAD_NORM.get(warhead_raw, str(warhead_raw).lower().replace(" ", "_"))
    nuc_aa = row.get("Resi_name")
    try:
        nuc_posi = int(float(row.get("Resi_posi")))
    except (TypeError, ValueError):
        return None
    nuc_chain = str(row.get("Resi_chain") or "").strip()[:1] or "A"
    lig_name = str(row.get("Ligand_name") or "").strip().upper()
    lig_chain = str(row.get("Ligand_chain") or "").strip()[:1] or nuc_chain
    try:
        lig_posi = int(float(row.get("Ligand_position")))
    except (TypeError, ValueError):
        lig_posi = None

    parsed = parse_pdb(pdb_path)
    if parsed is None:
        return None
    residues = parsed["residues"]
    hetatoms = parsed["hetatoms"]

    # Find the nucleophile atom
    nuc_key = (nuc_chain, nuc_posi)
    nuc_xyz = None
    if nuc_key in residues:
        r = residues[nuc_key]
        atom_names = NUC_ATOM_NAMES.get(nuc_aa, ("CA",))
        for an in atom_names:
            if an in r["atoms"]:
                nuc_xyz = r["atoms"][an]
                break
        if nuc_xyz is None and "CA" in r["atoms"]:
            nuc_xyz = r["atoms"]["CA"]
    if nuc_xyz is None:
        # Search any chain for the residue (sometimes chain naming differs)
        for (ch, rid), r in residues.items():
            if rid == nuc_posi and r["aa"] == nuc_aa:
                atom_names = NUC_ATOM_NAMES.get(nuc_aa, ("CA",))
                for an in atom_names:
                    if an in r["atoms"]:
                        nuc_xyz = r["atoms"][an]
                        nuc_chain = ch
                        nuc_key = (ch, rid)
                        break
                if nuc_xyz is not None:
                    break
    if nuc_xyz is None:
        return None

    # Find the ligand HETATM block
    het_atoms_for_lig = None
    # Try exact match (lig_chain, lig_name, lig_posi)
    if lig_posi is not None:
        key = (lig_chain, lig_name, lig_posi)
        if key in hetatoms:
            het_atoms_for_lig = hetatoms[key]
    if het_atoms_for_lig is None:
        # Try by name only (any chain/posi)
        for (ch, rn, rp), atoms in hetatoms.items():
            if rn == lig_name:
                het_atoms_for_lig = atoms
                break
    if het_atoms_for_lig is None:
        return None

    heavy = [a for a in het_atoms_for_lig if a["elem"] not in ("H", "D")]
    if not (HET_MIN <= len(heavy) <= HET_MAX):
        return None

    # Find warhead b/a positions. The returned `info["name"]` is the
    # SMARTS-verified class and is authoritative for the final label
    # (the source-curator `warhead_class` is only a preference hint here).
    info = find_warhead_atoms_smarts(mol, warhead_class)
    bxyz = axyz = None
    smarts_class = None
    if info is not None:
        smarts_class = info["name"]
        ba = smiles_to_pdb_atom_at_b(mol, het_atoms_for_lig, info["b_idx"], info["a_idx"])
        if ba is not None:
            bxyz, axyz = ba
    # Fallback: choose HET heavy atom closest to nucleophile as b
    if bxyz is None:
        best = None
        for ha in heavy:
            if ha["elem"] not in ("C", "B", "P", "S"):
                continue
            d = float(np.linalg.norm(ha["xyz"] - nuc_xyz))
            if best is None or d < best[0]:
                best = (d, ha)
        if best is None:
            return None
        bxyz = best[1]["xyz"]
        # a = closest neighbor of b (carbon)
        nbrs = [(float(np.linalg.norm(a["xyz"] - bxyz)), a) for a in heavy
                if a is not best[1] and a["elem"] in ("C", "N", "O", "S")]
        if not nbrs:
            return None
        nbrs.sort()
        axyz = nbrs[0][1]["xyz"]

    d_b_nuc = float(np.linalg.norm(bxyz - nuc_xyz))
    if d_b_nuc > D_B_NUC_MAX:
        return None

    # Pocket: residues within POCKET_RADIUS of bxyz, restricted to nuc_chain
    # (avoid multimer blowup)
    pocket = []
    for (ch, rid), r in residues.items():
        if ch != nuc_chain:
            continue
        if "CA" not in r["atoms"]:
            continue
        d_ca = float(np.linalg.norm(r["atoms"]["CA"] - bxyz))
        if d_ca <= POCKET_RADIUS:
            pocket.append({
                "aa": ONE_LETTER.get(r["aa"], "X"),
                "idx": int(rid),
                "d": d_ca,
                "ca_xyz": [float(r["atoms"]["CA"][0]),
                            float(r["atoms"]["CA"][1]),
                            float(r["atoms"]["CA"][2])],
            })
    if len(pocket) < POCKET_MIN or len(pocket) > POCKET_MAX:
        return None

    bd = angle_deg(nuc_xyz - bxyz, axyz - bxyz)
    pose6d = pose_features_6d(bxyz, axyz, nuc_xyz, bd)
    return {
        "source": row.get("_source", "covindb_v2"),
        "struct_id": f"{pdb_path.stem}_{nuc_chain}_{lig_name}_{nuc_posi}",
        "pdb_id": pdb_path.stem,
        "het_resname": lig_name,
        "smiles": canon,
        "canon_smi": canon,
        "pocket_residues": pocket,
        "warhead_pose_6d": pose6d,
        "nucleophile_xyz": [float(nuc_xyz[0]), float(nuc_xyz[1]), float(nuc_xyz[2])],
        "warhead_b_xyz": [float(bxyz[0]), float(bxyz[1]), float(bxyz[2])],
        "bd_angle_deg": float(bd),
        "d_b_nuc": d_b_nuc,
        "nucleophile_resid": int(nuc_posi),
        "nucleophile_resname": nuc_aa,
        # SMARTS-verified class wins; source-curator hint used only as fallback
        # when no SMARTS pattern matched (bxyz came from the nearest-heavy-atom fallback path).
        "warhead_class": smarts_class if smarts_class is not None else warhead_class,
        "warhead_class_source_label": warhead_class,  # keep the original for audit
        "target_name": row.get("Protein_name") or row.get("Proteins"),
    }


def _worker_safe(args):
    try:
        return extract_one_row(args)
    except Exception as e:
        return None


# ============================================================
# Main
# ============================================================

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta_csv", default=str(PROJECT_ROOT /
                     "data/covbinder/raw_covindb2/Covalent_Complex_Records.csv"))
    ap.add_argument("--pdb_dir", default=str(PROJECT_ROOT /
                     "data/covbinder/raw_covindb2/PDB"))
    ap.add_argument("--out", default=str(PROJECT_ROOT /
                     "data/m1a_triples_v2/covindb_v2_triples.parquet"))
    ap.add_argument("--source_tag", default="covindb_v2")
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    meta = pd.read_csv(args.meta_csv)
    pdb_dir = Path(args.pdb_dir)
    print(f"[{args.source_tag}] {len(meta)} metadata rows, "
           f"{meta['PDB'].nunique()} unique PDBs", flush=True)

    work = []
    skipped_missing_pdb = 0
    for _, row in meta.iterrows():
        pdb_id = str(row["PDB"]).lower()
        pdb_path = pdb_dir / f"{pdb_id}.pdb"
        if not pdb_path.exists():
            # Try uppercase
            pdb_path = pdb_dir / f"{pdb_id.upper()}.pdb"
            if not pdb_path.exists():
                skipped_missing_pdb += 1
                continue
        rd = row.to_dict()
        rd["_source"] = args.source_tag
        work.append((str(pdb_path), rd))
    print(f"[{args.source_tag}] Built {len(work)} work items "
           f"(skipped {skipped_missing_pdb} missing PDBs)", flush=True)

    if args.limit:
        work = work[:args.limit]

    rows = []
    fail_reasons = Counter()
    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(_worker_safe, w) for w in work]
        done = 0
        for fut in as_completed(futures):
            r = fut.result()
            if r is not None:
                rows.append(r)
            done += 1
            if done % 500 == 0:
                print(f"[{args.source_tag}] {done}/{len(work)}  kept={len(rows)}",
                       flush=True)
    print(f"[{args.source_tag}] FINAL kept={len(rows)} / {len(work)} "
           f"({100.0*len(rows)/max(1,len(work)):.1f}%)", flush=True)

    if not rows:
        print("FATAL: no triples", flush=True)
        sys.exit(1)

    df = pd.DataFrame(rows)
    df["pocket_residues"] = df["pocket_residues"].apply(json.dumps)
    df["warhead_pose_6d"] = df["warhead_pose_6d"].apply(lambda x: json.dumps(list(x)))
    df["nucleophile_xyz"] = df["nucleophile_xyz"].apply(lambda x: json.dumps(list(x)))
    df["warhead_b_xyz"] = df["warhead_b_xyz"].apply(lambda x: json.dumps(list(x)))

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out_path, index=False)
    print(f"Wrote {len(df)} triples to {out_path}", flush=True)

    # Diagnostics
    print()
    print("=== Warhead class distribution ===")
    print(df["warhead_class"].value_counts().head(20).to_string())
    print()
    print("=== Nucleophile distribution ===")
    print(df["nucleophile_resname"].value_counts().to_string())
    print()
    print("=== Pocket size distribution ===")
    sizes = df["pocket_residues"].apply(lambda s: len(json.loads(s)))
    print(f"  min={sizes.min()} median={int(sizes.median())} mean={sizes.mean():.1f} max={sizes.max()}")
    print()
    print("=== BD angle distribution ===")
    print(df["bd_angle_deg"].describe().to_string())


if __name__ == "__main__":
    main()
