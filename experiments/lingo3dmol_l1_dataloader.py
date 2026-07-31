"""L1 Phase 2 — Lingo3DMol covalent fine-tune dataloader.

Streams real (pocket, ligand) pairs out of CovalentInDB 2.0 cocrystal
records and feeds them into the Phase 1 training scaffold
(`run_lingo3dmol_l1_train.py`).

The dataset reads `Covalent_Complex_Records.csv`, looks up each row's
crystal in `pdb_dir`, extracts:

  - ligand HETATM xyz (matched by Ligand_chain + Ligand_position +
    Ligand_name)
  - reactive Cys CA xyz (matched by Resi_chain + Resi_posi + Resi_name)
  - a heavy-atom pocket crop within `pocket_radius` Å of the ligand,
    written as a temporary protein-only PDB and fed to
    `PocketCode.pocketCodeNCI`.

For tokenisation we use the proper FSMILES tokenizer from
`lingo3dmol_fsmiles_encoder.py` (Phase 3). It emits ring-size-aware
vocab tokens (`C_0`, `C_6`, `c_6`, `[nH]_6`, ...) that match what the
pretrained encoder expects, and is round-trip-verified through the
shipped `FragmolUtil.decode3d` on the L1 fine-tune corpus.

Failures (missing PDB, SMILES parse error, ligand HETATM not found,
Cys not found, SMILES > T tokens, grid OOB, etc.) are logged and the
row is skipped at the dataset level.

CPU only. No GPU.
"""

from __future__ import annotations

import hashlib
import os
import re
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import torch
import torch.utils.data as tud
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# ---------------------------------------------------------------------------
# Tokenizer / vocab shared with Phase 1 smoke script.
# ---------------------------------------------------------------------------
# We import the smoke module to reuse its FSMILES vocab, regex tokenizer, the
# pocket encoder wrapper, and the xyz->grid helper. The smoke module already
# inserts `external/Lingo3DMol` into sys.path via the L1 train entrypoint.
from run_lingo3dmol_l1_smoke import (  # noqa: E402
    V,
    PAD_ID,
    START_ID,
    SEP_ID,
    _xyz_to_grid,
    _encode_pocket,
    _embed_smiles_3d,
)
# Phase 3: proper FSMILES tokenizer (round-trip verified, 99.5% on
# acrylamide_only / 97% on CovInDB after filtering unsupported atoms).
from lingo3dmol_fsmiles_encoder import (
    tokenize_smiles_with_xyz as _fsmiles_tokenize,
)


# ---------------------------------------------------------------------------
# PDB helpers
# ---------------------------------------------------------------------------
# Parse a PDB ATOM/HETATM record using fixed-column widths (PDB spec).
# Columns (1-indexed): 13-16 atom name, 17 altLoc, 18-20 resName, 22 chainID,
# 23-26 resSeq, 31-38 x, 39-46 y, 47-54 z.
def _parse_pdb_line(line: str) -> dict | None:
    if not (line.startswith("ATOM") or line.startswith("HETATM")):
        return None
    try:
        return {
            "record": line[0:6].strip(),
            "atom_name": line[12:16].strip(),
            "alt_loc": line[16:17].strip(),
            "res_name": line[17:20].strip(),
            "chain": line[21:22].strip(),
            "res_seq": int(line[22:26].strip()),
            "x": float(line[30:38]),
            "y": float(line[38:46]),
            "z": float(line[46:54]),
            "element": line[76:78].strip() if len(line) >= 78 else "",
            "raw": line,
        }
    except ValueError:
        return None


def _read_pdb_records(pdb_path: str) -> list[dict]:
    records = []
    with open(pdb_path) as f:
        for line in f:
            r = _parse_pdb_line(line)
            if r is not None:
                records.append(r)
    return records


def extract_ligand_xyz(
    records: list[dict],
    chain: str,
    res_seq: int,
    res_name: str | None,
) -> np.ndarray | None:
    """Pull the (n_atoms, 3) xyz of one ligand instance from a PDB record list.

    Matches HETATM records by (chain, res_seq, optionally res_name).
    Excludes 'H' atoms.
    Returns None if no atoms match.
    """
    out = []
    for r in records:
        if r["record"] != "HETATM":
            continue
        if r["chain"] != chain:
            continue
        if r["res_seq"] != res_seq:
            continue
        if res_name is not None and r["res_name"] != res_name:
            continue
        # skip hydrogens (most cocrystal PDBs have none, but defend anyway)
        if r["element"] == "H" or r["atom_name"].startswith("H"):
            continue
        out.append([r["x"], r["y"], r["z"]])
    if not out:
        return None
    return np.asarray(out, dtype=np.float64)


def extract_cys_ca_xyz(
    records: list[dict],
    chain: str,
    res_seq: int,
    res_name: str = "CYS",
) -> np.ndarray | None:
    """Find the CA atom of the labeled Cys (or other) residue. (3,) array."""
    for r in records:
        if r["record"] != "ATOM":
            continue
        if r["chain"] != chain:
            continue
        if r["res_seq"] != res_seq:
            continue
        if r["res_name"] != res_name:
            continue
        if r["atom_name"] != "CA":
            continue
        return np.asarray([r["x"], r["y"], r["z"]], dtype=np.float64)
    return None


def extract_cys_atom_xyz(
    records: list[dict],
    chain: str,
    res_seq: int,
    atom_name: str,
    res_name: str = "CYS",
) -> np.ndarray | None:
    """Find a specific atom (e.g. 'SG', 'CB', 'CA') of the labeled residue."""
    for r in records:
        if r["record"] != "ATOM":
            continue
        if r["chain"] != chain:
            continue
        if r["res_seq"] != res_seq:
            continue
        if r["res_name"] != res_name:
            continue
        if r["atom_name"] != atom_name:
            continue
        return np.asarray([r["x"], r["y"], r["z"]], dtype=np.float64)
    return None


# ---------------------------------------------------------------------------
# Bürgi-Dunitz geometry: where the warhead Cβ should land relative to Cys-SG.
# ---------------------------------------------------------------------------
def compute_bd_target_coord(
    sg_xyz: np.ndarray,
    ca_xyz: np.ndarray | None,
    bd_distance: float = 1.85,
    bd_angle_deg: float = 107.0,
) -> np.ndarray | None:
    """Compute the IDEAL position of the warhead Cβ (electrophile carbon) such
    that, upon Michael addition, the new C-S bond would form at `bd_distance`
    Å along the Bürgi-Dunitz attack vector — i.e. `bd_angle_deg` from the
    S->Cα backbone vector.

    Recipe (matches `data/lingo3dmol_anchor_zap70_cys346.json`):
      1. attack_axis = -unit(Cα - SG) rotated into the plane such that the
         angle SG->target--SG->Cα is bd_angle_deg.
      Practical shortcut used by the project's anchor builder: rotate the
      unit vector (SG - Cα) around an arbitrary in-plane axis by
      (180 - bd_angle_deg) ≈ 73° toward the "out of pocket" side; we use
      a simple, deterministic in-plane rotation around the perpendicular to
      (SG - Cα) and the Cys CB vector.

    Args:
        sg_xyz: (3,) Cys SG coord in Å.
        ca_xyz: (3,) Cys CA coord in Å (None falls back to a default axis).
        bd_distance: target SG-to-Cβ' distance in Å. Default 1.85.
        bd_angle_deg: target SG-Cα ∠ SG-Cβ' angle in degrees. Default 107°.

    Returns:
        (3,) target xyz, or None if degenerate (sg == ca).
    """
    sg = np.asarray(sg_xyz, dtype=np.float64)
    if ca_xyz is None:
        # No CA — fall back: place target 1.85 Å along +z. Not ideal but
        # the geometry loss still trains "move warhead to sg+δ".
        return sg + np.array([0.0, 0.0, bd_distance])
    ca = np.asarray(ca_xyz, dtype=np.float64)
    v_sg_to_ca = ca - sg
    n = np.linalg.norm(v_sg_to_ca)
    if n < 1e-6:
        return None
    e_sg_ca = v_sg_to_ca / n  # unit vector S -> Cα (the backbone direction)

    # Build an orthonormal frame {e_sg_ca, e_perp, e_third}.
    # e_perp: any unit vector perpendicular to e_sg_ca. We pick the most
    # numerically stable one by choosing the world axis least aligned with
    # e_sg_ca.
    world = np.eye(3)
    align = np.abs(e_sg_ca @ world.T)
    least_aligned = int(np.argmin(align))
    a = world[least_aligned]
    e_perp = a - (a @ e_sg_ca) * e_sg_ca
    e_perp_norm = np.linalg.norm(e_perp)
    if e_perp_norm < 1e-6:
        return None
    e_perp = e_perp / e_perp_norm

    # Target unit vector at `bd_angle_deg` from e_sg_ca, in the plane
    # spanned by (e_sg_ca, e_perp).  cos(angle) along e_sg_ca, sin(angle)
    # along e_perp.  (We rotate AWAY from Cα — i.e. the target sits on the
    # OUTER side of SG relative to the backbone.  We achieve this by using
    # the NEGATIVE of e_sg_ca as the "forward" direction.)
    theta = np.deg2rad(bd_angle_deg)
    # Direction from SG to target: rotate (-e_sg_ca) by (180 - theta) toward e_perp.
    # Equivalently: target_dir = cos(theta)*e_sg_ca_neg + sin(theta)*e_perp
    # where e_sg_ca_neg points OUT of the backbone (away from Cα).
    e_out = -e_sg_ca
    # We want the angle between (SG->Cα) and (SG->target) to be bd_angle_deg.
    # vec(SG->target) = cos(theta) * (SG->Cα unit) + sin(theta) * e_perp
    # is the canonical construction (theta measured from S->Cα toward e_perp).
    target_dir = np.cos(theta) * e_sg_ca + np.sin(theta) * e_perp
    target_dir = target_dir / (np.linalg.norm(target_dir) + 1e-12)
    target = sg + bd_distance * target_dir
    return target


# ---------------------------------------------------------------------------
# Warhead atom identification in a SMILES.
# ---------------------------------------------------------------------------
# For acrylamides (C=CC(=O)N-...), the warhead's β-carbon — the electrophilic
# vinyl carbon that forms the new C-S bond — is mol atom index 1
# (the second C in C=CC(=O)N). Atom 0 is the terminal CH2 (the leaving end).
# Wait: re-check the canonical convention used by the project.  In
# `experiments/run_lingo3dmol_l2_inpaint.py` the comment notes
# "atom 0 = terminal CH2 (the leaving end)" — i.e. the CH2= end SITS at the
# Cβ_target (1.85 Å from SG along the attack vector). After Michael addition,
# the SG-C bond forms with atom 0 (the terminal CH2 becomes the new Cβ).
#
# So the warhead atom we want to anchor at the BD target is **mol atom 0** —
# the terminal CH2 in C=CC(=O)N. We confirm against the anchor JSON which
# names cb_pos_target as "the IDEAL position of the warhead's β-carbon (the
# CH2= end of CH2=CH-C(=O)-NH-...)".
#
# For other warhead classes (carbonyl aldehydes, nitriles, ...) the
# electrophilic atom index differs. Acrylamide-only is fine for Phase 1.
def _find_acrylamide_warhead_atom_idx(smiles: str) -> int | None:
    """Return the SMILES atom index of the warhead Cβ (terminal CH2 in
    C=CC(=O)N-...) or None if the molecule lacks an acrylamide warhead.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    # SMARTS: [CH2]=[CH][C](=O)[N] — terminal CH2 of acrylamide.
    patt = Chem.MolFromSmarts("[CH2;X3]=[CH;X3][C;X3](=O)[N]")
    matches = mol.GetSubstructMatches(patt)
    if not matches:
        # Looser pattern (allow sub'd β-C or charged amide N).
        patt2 = Chem.MolFromSmarts("[CH2]=C[C](=O)[N,n]")
        matches = mol.GetSubstructMatches(patt2)
    if not matches:
        return None
    # First match's atom 0 = terminal CH2 = warhead Cβ.
    return int(matches[0][0])


# ---------------------------------------------------------------------------
# Multi-warhead detection (Exp 2, 2026-05-29).
# ---------------------------------------------------------------------------
# Goal: replace the acrylamide-only `_find_acrylamide_warhead_atom_idx` with
# a multi-class detector that returns the electrophilic atom (warhead Cβ)
# for any of {acrylamide, chloroacetamide, vinyl-sulfonamide}.
#
# Convention: the electrophilic atom is the one that bonds to Cys-SG after
# the reaction. We follow the same "atom 1 in SMARTS" pattern used for
# acrylamide:
#   Acrylamide SMARTS:        [CH2]=[CH][C](=O)[N]    → atom 0 = terminal CH2
#   Chloroacetamide SMARTS:   Cl[CH2][C](=O)[N]       → atom 1 = the CH2 (Cα'
#                                                       after Cl leaves)
#   Vinyl-sulfonamide SMARTS: [CH2]=[CH][S](=O)(=O)[N]→ atom 0 = terminal CH2
#
# All three share BD distance 1.85 Å and BD angle 107°.
_WARHEAD_SMARTS: list[tuple[str, str, int, str]] = [
    # (name, smarts, electrophile_atom_idx_in_match, fallback_smarts)
    ("acrylamide",        "[CH2;X3]=[CH;X3][C;X3](=O)[N]",   0, "[CH2]=C[C](=O)[N,n]"),
    ("chloroacetamide",   "Cl[CH2][C](=O)[N]",                1, "Cl[CH2][C](=O)[N,n]"),
    ("vinyl_sulfonamide", "[CH2]=[CH][S](=O)(=O)[N]",         0, "[CH2]=C[S](=O)(=O)[N,n]"),
]


def _find_warhead_atom_idx_multi(smiles: str) -> tuple[int, str] | None:
    """Find the electrophilic warhead atom for any of:
    acrylamide, chloroacetamide, vinyl-sulfonamide.

    Returns (atom_idx, warhead_name) or None.

    Priority order: acrylamide → chloroacetamide → vinyl-sulfonamide
    (acrylamide is the dominant CovInDB class and most reliable detector).
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    for name, smarts, electro_pos, fallback in _WARHEAD_SMARTS:
        patt = Chem.MolFromSmarts(smarts)
        if patt is None:
            continue
        matches = mol.GetSubstructMatches(patt)
        if not matches:
            pat2 = Chem.MolFromSmarts(fallback)
            if pat2 is None:
                continue
            matches = mol.GetSubstructMatches(pat2)
        if matches:
            return int(matches[0][electro_pos]), name
    return None


def write_pocket_crop(
    records: list[dict],
    ligand_xyz: np.ndarray,
    out_pdb: str,
    pocket_radius: float = 15.0,
    max_atoms: int = 480,
) -> int:
    """Write a protein-only PDB containing protein heavy atoms inside
    `pocket_radius` Å of any ligand atom, capped at `max_atoms` (the
    `pocketCodeNCI` encoder pads to 500; we leave headroom for the
    occasional residue trickle past the radius cut). Atoms are sorted
    by distance-to-ligand ascending before capping.

    Excludes HETATM (would re-include the ligand).
    Returns the number of atoms written.
    """
    protein = [r for r in records if r["record"] == "ATOM"]
    if not protein:
        return 0
    protein_xyz = np.asarray([[r["x"], r["y"], r["z"]] for r in protein],
                             dtype=np.float64)
    diff = protein_xyz[:, None, :] - ligand_xyz[None, :, :]
    d = np.linalg.norm(diff, axis=-1)
    min_d = d.min(axis=1)

    # First filter by radius
    in_radius = np.where(min_d <= pocket_radius)[0]
    if in_radius.size == 0:
        return 0
    # Sort by distance and cap at max_atoms (the encoder pads to 500)
    order = in_radius[np.argsort(min_d[in_radius])]
    if order.size > max_atoms:
        order = order[:max_atoms]
    keep_set = set(order.tolist())

    with open(out_pdb, "w") as f:
        for i, r in enumerate(protein):
            if i in keep_set:
                f.write(r["raw"])  # already contains newline
        f.write("END\n")
    return int(order.size)


# ---------------------------------------------------------------------------
# Ligand SMILES -> per-token coords (Phase 3: uses proper FSMILES encoder)
# ---------------------------------------------------------------------------
def tokenize_ligand_with_xyz(
    smiles: str,
    atom_xyz_grid: np.ndarray,
    max_tokens: int = 100,
) -> tuple[list[int], list[list[int]], list[int | None]] | None:
    """Tokenize SMILES via the FSMILES (ring-size-aware) encoder. Each atom
    token gets the matching mol atom's grid coord; non-atom tokens carry
    the previous atom's coord.

    Returns (tokens, per_tok_coords, atom_idx_per_tok) or None on vocab /
    unsupported / too-long failure. `atom_idx_per_tok[i]` is the mol atom
    index that token i refers to, or None for non-atom tokens.
    """
    out = _fsmiles_tokenize(smiles, atom_xyz_grid, max_tokens=max_tokens)
    if out is None:
        return None
    return (
        out["token_ids"],
        out["per_token_coords"],
        out["atom_idx_per_token"],
    )


def _ligand_xyz_for_smiles(
    smi: str,
    pdb_lig_xyz: np.ndarray | None,
    cys_ca_xyz: np.ndarray | None,
    pocket_center: np.ndarray,
    seed: int = 42,
) -> tuple[Chem.Mol, np.ndarray] | tuple[None, None]:
    """Return (mol3d, xyz_in_pocket_frame) for the ligand.

    Strategy:
      1. If `pdb_lig_xyz` supplied AND the SMILES atom count matches
         the PDB heavy-atom count, use the crystallographic pose directly.
         (Bond order may differ from the SMILES; we still align by atom order
          since both should match the HET record order.)
      2. Otherwise ETKDG-embed, then translate the embedded ligand so its
         centroid sits at the Cys CA (paper-faithful: ligand anchored near
         the reactive Cys) — falling back to pocket center if no Cys.
    """
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return None, None

    # Pose strategy 1: crystallographic pose — works only if the SMILES atom
    # count matches the HETATM heavy-atom count (no canonicalisation needed,
    # we are taking the PDB as ground truth).
    if pdb_lig_xyz is not None:
        n_heavy_smi = mol.GetNumHeavyAtoms()
        if n_heavy_smi == pdb_lig_xyz.shape[0]:
            # Use the PDB pose. Build a conformer-less mol; the per-atom
            # xyz tensor is all we need for tokenisation.
            return mol, pdb_lig_xyz.copy()
        # Otherwise fall through to ETKDG.

    # Pose strategy 2: ETKDG embed + translate to Cys (or pocket centroid).
    mol3d, xyz = _embed_smiles_3d(smi, seed=seed)
    if xyz is None:
        return None, None
    anchor = cys_ca_xyz if cys_ca_xyz is not None else pocket_center
    ligand_centroid = xyz.mean(axis=0)
    xyz_translated = xyz - ligand_centroid + anchor
    return mol3d, xyz_translated


# ---------------------------------------------------------------------------
# Canonical-SMILES helper (used for reactivity-label lookup)
# ---------------------------------------------------------------------------
def _canon_smiles(s: str) -> str | None:
    """Return RDKit canonical SMILES (no stereo flags removed) or None on
    parse failure. Used to robustly key the reactivity-label lookup."""
    try:
        m = Chem.MolFromSmiles(s)
        if m is None:
            return None
        return Chem.MolToSmiles(m)
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
class CovalentInDB2Dataset(tud.Dataset):
    """Streaming (pocket, ligand) dataset over CovalentInDB 2.0 cocrystals.

    Args:
        complex_csv_path: path to `Covalent_Complex_Records.csv`.
        pdb_dir: directory containing `<PDB_ID>.pdb` files.
        vocab: FSMILES vocab dict (unused here — kept for API symmetry; the
               smoke module's `V` is the canonical reference).
        T: padded FSMILES sequence length.
        max_complexes: if set, truncate the dataset for dev runs.
        cache_dir: if set, cache pre-processed samples here as `.npz`.
        pocket_radius: Å radius for cropping protein atoms around the ligand.
        verbose: log skip reasons.

    Failures inside `__getitem__` are converted to RuntimeErrors so the
    DataLoader's worker can either skip-retry (we provide a `safe_getitem`)
    or surface the error. We use `safe_getitem` via a wrapper that retries
    on the next valid index when a row fails.
    """

    SKIP_REASONS = (
        "missing_pdb",
        "smiles_parse_fail",
        "ligand_not_in_pdb",
        "cys_not_in_pdb",
        "pocket_crop_empty",
        "etkdg_failed",
        "smiles_too_long",
        "grid_oob",
        "tokenizer_failed",
        "pocket_encode_failed",
    )

    def __init__(
        self,
        complex_csv_path: str,
        pdb_dir: str,
        vocab: dict | None = None,
        T: int = 80,
        max_complexes: int | None = None,
        cache_dir: str | None = None,
        pocket_radius: float = 15.0,
        verbose: bool = True,
        chassis_labels_csv: str | None = None,
        reactivity_csv_path: str | None = None,
    ):
        self.T = T
        self.pdb_dir = Path(pdb_dir)
        self.cache_dir = Path(cache_dir) if cache_dir else None
        if self.cache_dir is not None:
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.pocket_radius = pocket_radius
        self.verbose = verbose

        # ---- Reactivity labels (CovalentLingo Full) ----
        # Optional CSV from `experiments/compute_covindb_xtb_reactivity.py`.
        # Keyed by canonical SMILES; NaN where no acrylamide / xTB failed.
        self.smi_to_log_k2: dict[str, float] = {}
        self.reactivity_csv_path = reactivity_csv_path
        if reactivity_csv_path is not None:
            try:
                rdf = pd.read_csv(reactivity_csv_path)
                if "smiles" not in rdf.columns or "log_k2_GSH" not in rdf.columns:
                    raise ValueError(
                        f"reactivity_csv must have 'smiles' and 'log_k2_GSH' "
                        f"columns; got {list(rdf.columns)}"
                    )
                for _, r in rdf.iterrows():
                    s = str(r["smiles"]).strip()
                    if not s:
                        continue
                    val = (float(r["log_k2_GSH"]) if pd.notna(r["log_k2_GSH"])
                           else float("nan"))
                    can = _canon_smiles(s)
                    if can is not None:
                        self.smi_to_log_k2[can] = val
                    # Also key by raw to be robust to canonicalisation drift.
                    self.smi_to_log_k2.setdefault(s, val)
                n_non_nan = sum(1 for v in self.smi_to_log_k2.values()
                                if v == v)  # NaN != NaN
                print(f"[ds]   reactivity labels: {reactivity_csv_path}\n"
                      f"[ds]     {len(self.smi_to_log_k2)} entries, "
                      f"{n_non_nan} non-NaN")
            except Exception as e:
                print(f"[ds]   WARNING: failed to load reactivity_csv: {e}")
                self.smi_to_log_k2 = {}

        # ---- Chassis-prior labels ----
        # Optional: load (pdb_id, ligand_smiles) -> chassis_family_id mapping.
        # Used by the chassis-prior head in TransformerModel. If unset OR row
        # has no entry, emit chassis_label = -1 (CE ignore_index).
        self.chassis_lookup: dict[tuple[str, str], int] = {}
        self.chassis_lookup_by_pdb: dict[str, int] = {}
        if chassis_labels_csv is not None:
            chassis_df = pd.read_csv(chassis_labels_csv)
            for _, lab_row in chassis_df.iterrows():
                pid = str(lab_row["pdb_id"]).strip().upper()
                smi = str(lab_row["ligand_smiles"]).strip()
                fid = int(lab_row["chassis_family_id"])
                self.chassis_lookup[(pid, smi)] = fid
                # First-write-wins by pdb_id (collisions are rare but possible
                # for multiple ligand chains per PDB).
                self.chassis_lookup_by_pdb.setdefault(pid, fid)
            n_in_top = sum(1 for v in self.chassis_lookup.values() if v >= 0)
            print(f"[ds]   chassis labels loaded: {len(self.chassis_lookup)} rows, "
                  f"{n_in_top} in-top-N (rest = -1).")

        df = pd.read_csv(complex_csv_path)
        # ---- Schema normalisation -------------------------------------
        # Two CSV schemas are in use in this project:
        #   (a) raw CovInDB `Covalent_Complex_Records.csv` — UPPER-case
        #       columns: PDB, Ligand_chain, Ligand_position, Ligand_name,
        #       Resi_chain, Resi_posi, Resi_name, SMILES.
        #   (b) curated `covind_smoke_subset.csv` / `covind_training_set.csv`
        #       — LOWER-case columns + PRE-EXTRACTED sg_x/y/z, cb_x/y/z and
        #       anchor_atom_idx_in_ligand. Schema (b) lets us skip the PDB
        #       parse for SG extraction.
        # Detect schema and rename columns so the rest of the code can use
        # the upper-case names uniformly.
        rename_map = {
            "pdb_id":           "PDB",
            "ligand_chain":     "Ligand_chain",
            "ligand_resi":      "Ligand_position",
            "ligand_resname":   "Ligand_name",
            "cys_chain":        "Resi_chain",
            "cys_resi":         "Resi_posi",
            "warhead_class":    "Warhead",  # unused but kept for completeness
            "smiles":           "SMILES",
        }
        for old, new in rename_map.items():
            if old in df.columns and new not in df.columns:
                df = df.rename(columns={old: new})
        # If lower-case schema, synthesize a Resi_name column.
        if "Resi_name" not in df.columns:
            df["Resi_name"] = "CYS"
        # Sanity-check key columns
        for c in ("PDB", "Ligand_chain", "Ligand_position", "Ligand_name",
                  "Resi_chain", "Resi_posi", "Resi_name", "SMILES"):
            if c not in df.columns:
                raise ValueError(
                    f"complex_csv missing required column '{c}'. "
                    f"Got: {list(df.columns)}"
                )
        # Drop rows with missing critical fields.
        df = df.dropna(subset=["PDB", "SMILES", "Ligand_chain",
                                "Ligand_position", "Resi_chain", "Resi_posi"])
        # Quick existence + length filter: drop rows where the PDB file is
        # missing or the SMILES obviously exceeds T (chars > 3*T is a cheap
        # upper bound that admits all real SMILES that fit in T tokens).
        usable_idx: list[int] = []
        skip_counts: Counter = Counter()
        for i, row in df.reset_index(drop=True).iterrows():
            pdb_id = str(row["PDB"]).strip().upper()
            pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
            if not pdb_path.exists():
                skip_counts["missing_pdb"] += 1
                continue
            smi = str(row["SMILES"]).strip()
            if not smi or smi.lower() in ("nan", "none"):
                skip_counts["smiles_parse_fail"] += 1
                continue
            if len(smi) > 4 * T:
                # cheap pre-filter; real check happens at __getitem__
                skip_counts["smiles_too_long"] += 1
                continue
            usable_idx.append(i)
        self.df = df.reset_index(drop=True).iloc[usable_idx].reset_index(drop=True)
        self.precheck_skips = skip_counts

        if max_complexes is not None:
            self.df = self.df.iloc[:max_complexes].reset_index(drop=True)

        # Per-getitem skip counter (warmed during __getitem__).
        self.getitem_skips: Counter = Counter()
        self._fail_seen: set[int] = set()

        print(
            f"[ds] CovalentInDB2Dataset: complex_csv={complex_csv_path}\n"
            f"[ds]   total rows in csv: {len(df)}\n"
            f"[ds]   precheck skips:    {dict(self.precheck_skips)}\n"
            f"[ds]   usable after precheck: {len(self.df)}\n"
            f"[ds]   max_complexes:     {max_complexes}\n"
            f"[ds]   pdb_dir:           {self.pdb_dir}\n"
            f"[ds]   pocket_radius:     {self.pocket_radius} A\n"
            f"[ds]   T (seq_len):       {self.T}\n"
            f"[ds]   cache_dir:         {self.cache_dir}"
        )

    def __len__(self) -> int:
        return len(self.df)

    # ----- internal: look up chassis label for a row ----------------
    def _chassis_label_for(self, pdb_id: str, smi: str) -> int:
        """Return the chassis_family_id for (pdb_id, smi), or -1 if not in
        the top-N families / not in lookup table at all."""
        if not self.chassis_lookup:
            return -1
        v = self.chassis_lookup.get((pdb_id, smi))
        if v is not None:
            return int(v)
        # Fallback: pdb_id-only match (handles minor SMILES canonicalization drift)
        v = self.chassis_lookup_by_pdb.get(pdb_id)
        if v is not None:
            return int(v)
        return -1

    def _log_k2_for(self, smi: str) -> float:
        """Look up the xTB log_k2_GSH label for this SMILES (canonical, then
        raw fallback). Returns NaN if absent or no reactivity csv loaded."""
        if not self.smi_to_log_k2:
            return float("nan")
        can = _canon_smiles(smi)
        if can is not None and can in self.smi_to_log_k2:
            return float(self.smi_to_log_k2[can])
        if smi in self.smi_to_log_k2:
            return float(self.smi_to_log_k2[smi])
        return float("nan")

    # ----- internal: process one row ---------------------------------
    def _process_row(self, row: pd.Series) -> dict | None:
        pdb_id = str(row["PDB"]).strip().upper()
        pdb_path = self.pdb_dir / f"{pdb_id}.pdb"
        smi = str(row["SMILES"]).strip()
        lig_chain = str(row["Ligand_chain"]).strip()
        lig_pos = int(row["Ligand_position"])
        lig_name = str(row["Ligand_name"]).strip() if not pd.isna(row.get("Ligand_name")) else None
        resi_chain = str(row["Resi_chain"]).strip()
        resi_posi = int(row["Resi_posi"])
        resi_name = str(row.get("Resi_name", "CYS")).strip() if not pd.isna(row.get("Resi_name")) else "CYS"

        # Resolve chassis label up-front (cheap dict lookup).
        chassis_lbl = self._chassis_label_for(pdb_id, smi)

        # Cache hit?
        # NB: bumped `v2` suffix on 2026-05-29 when we added bd_target_coord /
        # warhead_atom_idx / warhead_valid fields. Old caches missing these
        # are STILL loadable (we backfill on read) — this just lets fresh
        # caches use a deterministic, distinct file.
        # NB: bumped `v3` suffix on 2026-05-31 to invalidate all stale caches
        # after the CRIT-1/CRIT-2 fixes (target_coords[0] anchor mismatch +
        # contact_idx always 0).
        # NB: bumped `v4` suffix on 2026-05-29 for Exp 2 multi-warhead
        # support — caches now include warhead_class detected via the
        # multi-class detector (acrylamide / chloroacetamide / vinyl-SO2).
        if self.cache_dir is not None:
            key = hashlib.md5(
                f"{pdb_id}|{lig_chain}|{lig_pos}|{lig_name}|{resi_chain}|"
                f"{resi_posi}|{resi_name}|T{self.T}|R{self.pocket_radius}|v4".encode()
            ).hexdigest()[:12]
            cache_path = self.cache_dir / f"{pdb_id}_{key}.npz"
            if cache_path.exists():
                npz = np.load(cache_path, allow_pickle=False)
                loaded = {k: torch.from_numpy(npz[k].copy()) for k in npz.files}
                # Backfill missing keys for forward compatibility with old caches.
                if "bd_target_coord" not in loaded:
                    loaded["bd_target_coord"] = torch.zeros((3,), dtype=torch.int64)
                if "warhead_atom_idx" not in loaded:
                    loaded["warhead_atom_idx"] = torch.tensor(-1, dtype=torch.int64)
                if "warhead_valid" not in loaded:
                    loaded["warhead_valid"] = torch.tensor(0.0)
                # Chassis label: always re-resolve from runtime lookup so cache
                # rebuilds aren't required when chassis_labels_csv changes.
                loaded["chassis_label"] = torch.tensor(chassis_lbl, dtype=torch.int64)
                # Reactivity label: always re-resolve so cache rebuilds aren't
                # required when reactivity_csv_path changes.
                loaded["log_k2_target"] = torch.tensor(
                    self._log_k2_for(smi), dtype=torch.float32
                )
                return loaded
        else:
            cache_path = None

        records = _read_pdb_records(str(pdb_path))
        if not records:
            self.getitem_skips["missing_pdb"] += 1
            return None

        pdb_lig_xyz = extract_ligand_xyz(records, lig_chain, lig_pos, lig_name)
        if pdb_lig_xyz is None:
            self.getitem_skips["ligand_not_in_pdb"] += 1
            return None

        cys_ca_xyz = extract_cys_ca_xyz(records, resi_chain, resi_posi, resi_name)
        # cys_ca optional — None is recoverable (we just won't translate to Cys)
        if cys_ca_xyz is None:
            self.getitem_skips["cys_not_in_pdb"] += 1
            # not fatal — proceed (covalent assignment is for r/theta/phi heads, TODO Phase 3)

        # Cys SG xyz: needed for the L_anchor_geometry loss (Phase 4-geom).
        # Try the row first (curated CSV may pre-cache sg_x/y/z); else parse PDB.
        sg_xyz_aa: np.ndarray | None = None
        if all(c in row.index for c in ("sg_x", "sg_y", "sg_z")):
            try:
                sg_xyz_aa = np.asarray([float(row["sg_x"]),
                                         float(row["sg_y"]),
                                         float(row["sg_z"])],
                                        dtype=np.float64)
                if not np.isfinite(sg_xyz_aa).all():
                    sg_xyz_aa = None
            except (TypeError, ValueError):
                sg_xyz_aa = None
        if sg_xyz_aa is None:
            sg_xyz_aa = extract_cys_atom_xyz(records, resi_chain, resi_posi,
                                              "SG", resi_name)

        # Crop pocket to a temp PDB.
        # We dump the crop into the cache_dir if available, else /tmp.
        crop_dir = self.cache_dir if self.cache_dir is not None else Path("/tmp")
        crop_dir.mkdir(parents=True, exist_ok=True)
        crop_path = crop_dir / f"_pocket_{pdb_id}_{lig_chain}{lig_pos}.pdb"
        n_pocket_atoms = write_pocket_crop(
            records, pdb_lig_xyz, str(crop_path),
            pocket_radius=self.pocket_radius,
        )
        if n_pocket_atoms == 0:
            self.getitem_skips["pocket_crop_empty"] += 1
            return None

        # Encode pocket (heavy lift — RDKit MolFromPDBFile is slow on big inputs;
        # we cropped to ~few-hundred atoms so this is OK).
        try:
            pocket = _encode_pocket(str(crop_path), lig_xyz=pdb_lig_xyz.tolist())
        except Exception as e:
            if self.verbose:
                print(f"[ds] pocket encode failed on {pdb_id}: {e}")
            self.getitem_skips["pocket_encode_failed"] += 1
            return None
        pocket_center = pocket["center"]

        # Ligand pose -> grid coords.
        mol, lig_xyz = _ligand_xyz_for_smiles(
            smi,
            pdb_lig_xyz=pdb_lig_xyz,
            cys_ca_xyz=cys_ca_xyz,
            pocket_center=pocket_center,
        )
        if lig_xyz is None:
            self.getitem_skips["etkdg_failed"] += 1
            return None

        grid_xyz = _xyz_to_grid(lig_xyz, pocket_center)
        if grid_xyz.min() < 0 or grid_xyz.max() >= 240:
            self.getitem_skips["grid_oob"] += 1
            return None

        # Tokenize via the proper FSMILES (ring-size-aware) encoder.
        # The encoder canonicalises the SMILES internally and outputs tokens
        # in canonical atom order; the per-atom xyz must follow the SAME
        # order. For the PDB-pose path we feed PDB-order xyz, which matches
        # the input-SMILES atom order via the encoder's _smilesAtomOutputOrder
        # bookkeeping.
        result = tokenize_ligand_with_xyz(smi, grid_xyz, max_tokens=self.T - 2)
        if result is None:
            self.getitem_skips["tokenizer_failed"] += 1
            return None
        toks, per_tok_coords, atom_idx_per_tok = result

        toks = [START_ID] + toks + [SEP_ID]
        first_coord = grid_xyz[0].tolist() if len(grid_xyz) else [0, 0, 0]
        last_coord = per_tok_coords[-1] if per_tok_coords else first_coord
        per_tok_coords = [list(first_coord)] + per_tok_coords + [list(last_coord)]
        # atom_idx_per_tok aligned to the token list (None for non-atom)
        atom_idx_per_tok = [None] + atom_idx_per_tok + [None]

        if len(toks) > self.T:
            self.getitem_skips["smiles_too_long"] += 1
            return None

        target_token = np.full((self.T,), PAD_ID, dtype=np.int64)
        target_coords = np.zeros((self.T, 3), dtype=np.int64)
        target_mask = np.zeros((self.T,), dtype=np.float32)
        L = len(toks)
        target_token[:L] = np.asarray(toks, dtype=np.int64)
        target_coords[:L, :] = np.asarray(per_tok_coords, dtype=np.int64)
        target_mask[:L] = 1.0

        # Phase 3 (3.2): build smi_map / root_coords for the r/theta/phi heads.
        # `smi_map[i]` = position of the previous atom that token i links back
        # to (0 if no parent), mirroring the autoregressive forward's
        # `find_root_smi_cur` logic at preprocessing time. We use the simpler
        # "previous atom in SMILES order" heuristic — for L1 this is close
        # enough to the paper's bonded-parent for the gradient to be useful.
        # See encoder docstring for caveats; r/theta/phi are bonus heads.
        is_atom_per_tok = [a is not None for a in atom_idx_per_tok]
        smi_map = np.zeros((self.T,), dtype=np.int64)
        smi_map_n1 = np.zeros((self.T,), dtype=np.int64)
        smi_map_n2 = np.zeros((self.T,), dtype=np.int64)
        # Walk forward, tracking previous atom token positions
        last_atom_pos = 0  # position of last atom token (start_0 -> coord at idx 0)
        last_atom_pos_n1 = 0
        last_atom_pos_n2 = 0
        for i in range(L):
            if i == 0:
                continue
            if is_atom_per_tok[i]:
                # parent = previous atom token
                smi_map[i] = last_atom_pos
                smi_map_n1[i] = last_atom_pos_n1
                smi_map_n2[i] = last_atom_pos_n2
                last_atom_pos_n2 = last_atom_pos_n1
                last_atom_pos_n1 = last_atom_pos
                last_atom_pos = i
            else:
                # non-atom token: parent = last_atom_pos
                smi_map[i] = last_atom_pos
                smi_map_n1[i] = last_atom_pos_n1
                smi_map_n2[i] = last_atom_pos_n2

        # Per-position root coords (the GT coord at smi_map[i]).
        root_coords = target_coords[smi_map]               # (T, 3)
        root_coords_n1 = target_coords[smi_map_n1]         # (T, 3)
        root_coords_n2 = target_coords[smi_map_n2]         # (T, 3)
        # is_ele flag per position (atom tokens flow through r/theta/phi loss)
        is_ele = np.zeros((self.T,), dtype=np.float32)
        is_ele[:L] = np.asarray(is_atom_per_tok, dtype=np.float32)

        # contact_idx: prefer the pocket atom nearest the Cys SG (deterministic,
        # meaningful pocket-side anchor that matches the autoregressive
        # forward()'s `coords[batch_ind, contact_idx]` semantics).
        # Fall back to argmax(contact_arr) and then 0 only if no SG available.
        # BUG FIX 2026-05-31 (forensic CRIT-2): cached samples had contact_idx==0
        # for all 2409 rows because `pocket["contact"]` summed to 0 (the
        # cropped-pocket contact-channel computation in pocketCodeNCI is
        # silently empty). Using SG-nearest gives a real anchor.
        contact_arr = pocket["contact"]
        cidx = 0
        if sg_xyz_aa is not None:
            # pocket["coords"] is voxel-grid (int) — recompute its Å position
            # via pocket_center: pocket_coords_aa = (vox - 120) / 10 + center.
            # But we only need RELATIVE nearness, so we can also use the
            # original `new_coords`-equivalent. Simpler: compare grid distance.
            pocket_coords_grid = pocket["coords"]   # (500, 3) int voxel
            sg_grid = _xyz_to_grid(sg_xyz_aa.reshape(1, 3),
                                    pocket_center).reshape(3)
            # mask out padded atoms (mask==0)
            sm = pocket["mask"].astype(np.float32)
            d2 = ((pocket_coords_grid.astype(np.float64)
                   - sg_grid.astype(np.float64)) ** 2).sum(axis=1)
            # set padded atoms to +inf so they aren't chosen
            d2 = np.where(sm > 0.5, d2, np.inf)
            if np.isfinite(d2).any():
                cidx = int(np.argmin(d2))
        if cidx == 0 and contact_arr.sum() > 0:
            cidx = int(np.argmax(contact_arr))

        # ------------------------------------------------------------------
        # BUG FIX 2026-05-31 (forensic CRIT-1): align `target_coords[:, 0]`
        # with the inference contract.  The autoregressive `forward()` sets
        #   gt_coords[:, 0] = coords[batch_ind, contact_idx]
        # at line 235 of transformer_v1_res_fac2.py — i.e. slot 0 is the
        # POCKET-SIDE ANCHOR voxel, NOT the first ligand atom voxel.  The
        # dataloader previously placed the first ligand atom's grid coord at
        # slot 0, causing a hard train/inference mismatch (verified on all
        # 2409 cached samples — never matched).  Overwrite slot 0 with
        # `pocket["coords"][cidx]`, and re-derive root_coords (which reads
        # from target_coords).  Token CE loss already excludes slot 0
        # (`loss_mask = target_mask[:, 1:]`) so no extra loss bookkeeping.
        # ------------------------------------------------------------------
        target_coords[0, :] = pocket["coords"][cidx]
        root_coords = target_coords[smi_map]                  # re-derive
        root_coords_n1 = target_coords[smi_map_n1]
        root_coords_n2 = target_coords[smi_map_n2]

        # ------------------------------------------------------------------
        # L_anchor_geometry bookkeeping (new — Phase 4-geom).
        #
        #   bd_target_coord  (3,) int64   — grid coords of the prereactive
        #                                   Cβ position (SG + 1.85 Å along
        #                                   the Bürgi-Dunitz attack vector).
        #   warhead_atom_idx scalar int64 — token index (0..T-1) of the
        #                                   warhead Cβ atom.  -1 = invalid
        #                                   (no acrylamide, or token >= T).
        #   warhead_valid    scalar float — 1.0 if both above are usable.
        # ------------------------------------------------------------------
        # 1. BD target in Å — prefer pre-cached cb_x/y/z from the curated
        #    CSV (already the BD-computed target for that record); else
        #    compute on the fly from SG + Cα.
        bd_aa: np.ndarray | None = None
        if all(c in row.index for c in ("cb_x", "cb_y", "cb_z")):
            try:
                # NOTE: in `covind_*` curated CSVs, cb_x/y/z is the actual
                # Cys-side Cβ atom (read from PDB), NOT the BD-target Cβ'.
                # We still need to apply the BD geometry formula on top to
                # get the warhead's IDEAL Cβ' position.
                pass  # keep falling through to the from-SG path
            except (TypeError, ValueError):
                pass
        if sg_xyz_aa is not None:
            bd_aa = compute_bd_target_coord(sg_xyz_aa, cys_ca_xyz)

        # 2. SMILES warhead atom -> token index.
        warhead_atom_idx_int = -1
        warhead_valid_flag = 0.0
        bd_grid_in_range = False
        bd_grid: np.ndarray | None = None
        if bd_aa is not None:
            # Map BD target (Å) into the same voxel grid as the ligand coords.
            bd_grid = _xyz_to_grid(bd_aa.reshape(1, 3),
                                    pocket_center).reshape(3)
            if bd_grid.min() >= 0 and bd_grid.max() < 240:
                bd_grid_in_range = True
            if bd_grid_in_range:
                # Find the warhead Cβ atom in the SMILES.
                # For curated CSV, `anchor_atom_idx_in_ligand` is pre-computed.
                wh_atom_in_smi: int | None = None
                if "anchor_atom_idx_in_ligand" in row.index:
                    try:
                        v = row["anchor_atom_idx_in_ligand"]
                        if pd.notna(v) and str(v).strip() != "":
                            wh_atom_in_smi = int(v)
                    except (TypeError, ValueError):
                        wh_atom_in_smi = None
                if wh_atom_in_smi is None:
                    # Exp 2 multi-warhead: try acrylamide, then chloroacetamide,
                    # then vinyl-sulfonamide. BD geometry (1.85 Å / 107°) is the
                    # same across all three.
                    multi_hit = _find_warhead_atom_idx_multi(smi)
                    if multi_hit is not None:
                        wh_atom_in_smi, _wh_class = multi_hit

                if wh_atom_in_smi is not None and wh_atom_in_smi >= 0:
                    # Map mol-atom index -> token position via atom_idx_per_tok.
                    # Walk the (already-padded with start/sep) token list.
                    tok_pos = None
                    for ti, a in enumerate(atom_idx_per_tok):
                        if a is None:
                            continue
                        if a == wh_atom_in_smi:
                            tok_pos = ti
                            break
                    if tok_pos is not None and tok_pos < self.T:
                        warhead_atom_idx_int = int(tok_pos)
                        warhead_valid_flag = 1.0
                        # Overwrite the warhead token's target_coord to the
                        # IDEAL BD position. The model is now trained to
                        # place the warhead at the prereactive pose, NOT at
                        # the crystallographic post-reaction pose (which
                        # already sits at SG + 1.85 Å for covalent crystals,
                        # i.e. very close to bd_grid — so this re-write is
                        # a small correction in practice, NOT a regime change).
                        target_coords[tok_pos, :] = bd_grid.astype(np.int64)
                        # Also re-populate root_coords entries that point
                        # back to this token (they read from target_coords).
                        # We already built smi_map BEFORE the rewrite, so
                        # re-derive the rooted coords.
                        root_coords = target_coords[smi_map]
                        root_coords_n1 = target_coords[smi_map_n1]
                        root_coords_n2 = target_coords[smi_map_n2]

        # Fall-through: if warhead not found, bd_target_coord stays at 0,0,0
        # and warhead_valid=0 so the loss masks it out.
        if bd_grid_in_range and bd_grid is not None:
            bd_target_coord_arr = bd_grid.astype(np.int64)
        else:
            bd_target_coord_arr = np.zeros((3,), dtype=np.int64)

        sample = {
            "coords": pocket["coords"].astype(np.int64),       # (500, 3)
            "type": pocket["type"].astype(np.int64),           # (500,)
            "residue": pocket["residue"].astype(np.int64),     # (500,)
            "critical_anchor": np.zeros((500,), dtype=np.int64),
            "src_mask": pocket["mask"].astype(np.float32),     # (500,)
            "target_token": target_token,
            "target_coords": target_coords,
            "target_mask": target_mask,
            "contact_idx": np.asarray(cidx, dtype=np.int64),
            # Phase 3 r/theta/phi bookkeeping
            "smi_map":        smi_map,                          # (T,)
            "smi_map_n1":     smi_map_n1,                       # (T,)
            "smi_map_n2":     smi_map_n2,                       # (T,)
            "root_coords":    root_coords.astype(np.int64),     # (T, 3)
            "root_coords_n1": root_coords_n1.astype(np.int64),  # (T, 3)
            "root_coords_n2": root_coords_n2.astype(np.int64),  # (T, 3)
            "is_ele":         is_ele.astype(np.float32),        # (T,)
            # L_anchor_geometry bookkeeping
            "bd_target_coord":  bd_target_coord_arr,                        # (3,)
            "warhead_atom_idx": np.asarray(warhead_atom_idx_int,
                                            dtype=np.int64),                # scalar
            "warhead_valid":    np.asarray(warhead_valid_flag,
                                            dtype=np.float32),              # scalar
            # Chassis-prior head: family id (0..n_classes-1) or -1 (ignore).
            "chassis_label":    np.asarray(chassis_lbl, dtype=np.int64),    # scalar
            # CovalentLingo Full reactivity head: xTB log_k2_GSH (NaN = no label).
            "log_k2_target":    np.asarray(self._log_k2_for(smi),
                                            dtype=np.float32),               # scalar
        }

        if cache_path is not None:
            try:
                np.savez_compressed(cache_path, **sample)
            except Exception as e:
                if self.verbose:
                    print(f"[ds] cache save failed for {pdb_id}: {e}")

        return {k: torch.from_numpy(v) if v.ndim > 0 else torch.from_numpy(np.array(v))
                for k, v in sample.items()}

    # ----- public Dataset API ----------------------------------------
    def __getitem__(self, idx: int) -> dict:
        # Bounded retry to dodge transient bad rows. Walk forward in the
        # dataset so we never repeat the same failing index.
        n = len(self.df)
        original = idx
        for attempt in range(min(n, 32)):
            row = self.df.iloc[idx]
            try:
                out = self._process_row(row)
            except Exception as e:
                pdb_id = str(row.get("PDB", "?"))
                if self.verbose and idx not in self._fail_seen:
                    print(f"[ds] exception on row idx={idx} pdb={pdb_id}: {e}")
                    self._fail_seen.add(idx)
                self.getitem_skips["exception"] += 1
                out = None
            if out is not None:
                return out
            idx = (idx + 1) % n
        raise RuntimeError(
            f"CovalentInDB2Dataset: could not produce a sample after 32 "
            f"attempts starting at idx={original}. "
            f"skip counts: {dict(self.getitem_skips)}"
        )


# ---------------------------------------------------------------------------
# Collate
# ---------------------------------------------------------------------------
def collate_fn(batch: list[dict]) -> dict:
    """Stack list[dict] of per-sample tensors into a batch dict matching
    `TransformerModel.forward_train` kwargs."""
    out: dict[str, torch.Tensor] = {}
    keys = batch[0].keys()
    scalar_keys = ("contact_idx", "warhead_atom_idx", "warhead_valid",
                   "chassis_label", "log_k2_target")
    for k in keys:
        if k in scalar_keys:
            # (B,) — stack 0-d tensors
            out[k] = torch.stack([b[k].view(()) for b in batch])
        else:
            out[k] = torch.stack([b[k] for b in batch], dim=0)
    # Cast dtype to forward_train expectations.
    out["coords"] = out["coords"].long()
    out["type"] = out["type"].long()
    out["residue"] = out["residue"].long()
    out["critical_anchor"] = out["critical_anchor"].long()
    out["src_mask"] = out["src_mask"].float()
    out["target_token"] = out["target_token"].long()
    out["target_coords"] = out["target_coords"].long()
    out["target_mask"] = out["target_mask"].float()
    out["contact_idx"] = out["contact_idx"].long()
    # Phase 3 r/theta/phi bookkeeping
    for k in ("smi_map", "smi_map_n1", "smi_map_n2"):
        if k in out:
            out[k] = out[k].long()
    for k in ("root_coords", "root_coords_n1", "root_coords_n2"):
        if k in out:
            out[k] = out[k].long()
    if "is_ele" in out:
        out["is_ele"] = out["is_ele"].float()
    # L_anchor_geometry casts
    if "bd_target_coord" in out:
        out["bd_target_coord"] = out["bd_target_coord"].long()       # (B, 3)
    if "warhead_atom_idx" in out:
        out["warhead_atom_idx"] = out["warhead_atom_idx"].long()      # (B,)
    if "warhead_valid" in out:
        out["warhead_valid"] = out["warhead_valid"].float()           # (B,)
    if "chassis_label" in out:
        out["chassis_label"] = out["chassis_label"].long()            # (B,)
    if "log_k2_target" in out:
        out["log_k2_target"] = out["log_k2_target"].float()           # (B,)
    return out


# ---------------------------------------------------------------------------
# CLI: smoke probe — build the dataset, pull a few samples, print shapes.
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--complex_csv", required=True)
    p.add_argument("--pdb_dir", required=True)
    p.add_argument("--cache_dir", default=None)
    p.add_argument("--max_complexes", type=int, default=8)
    p.add_argument("--T", type=int, default=80)
    p.add_argument("--pocket_radius", type=float, default=15.0)
    args = p.parse_args()

    # CPU shim must be installed before any Lingo3DMol import (we already
    # transitively pulled it through `run_lingo3dmol_l1_smoke`, but be safe).
    import lingo3dmol_cpu_shim  # noqa: F401

    ds = CovalentInDB2Dataset(
        complex_csv_path=args.complex_csv,
        pdb_dir=args.pdb_dir,
        T=args.T,
        max_complexes=args.max_complexes,
        cache_dir=args.cache_dir,
        pocket_radius=args.pocket_radius,
    )
    print(f"[ds-cli] dataset len = {len(ds)}")
    for i in range(min(4, len(ds))):
        t0 = time.time()
        s = ds[i]
        dt = time.time() - t0
        print(f"[ds-cli] sample {i}: dt={dt:.2f}s "
              f"toks={int(s['target_mask'].sum().item())} "
              f"pocket_atoms={int(s['src_mask'].sum().item())} "
              f"contact_idx={int(s['contact_idx'].item())}")
    print(f"[ds-cli] getitem_skips: {dict(ds.getitem_skips)}")
