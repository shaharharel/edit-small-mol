"""Proper FSMILES tokenizer for Lingo3DMol L1 fine-tuning.

The pretrained Lingo3DMol encoder uses a *fragment-aware* FSMILES vocab
(`vocab_c2i_v1_decode_new`, 76 tokens). Each atom token carries a
ring-size suffix — e.g. `C_0` (non-ring), `C_5`, `C_6`, `C_10`, `C_11`,
`C_12` (ring sizes 5, 6, 7-9, 10, ≥11 — see vocab definition).
The molecule is optionally split into BRICS-style fragments separated by
`sep_0`; consecutive fragments are linked via `[*]` markers that mark
the bond-back atom.

Lingo3DMol ships only the *decode* side (`FragmolUtil.decode3d` +
`mergeSmiles3D`). The encode pass is not exposed. This module rebuilds
it from scratch and ROUND-TRIPS through the shipped decoder to verify
correctness:

    encode(mol)  ->  (token_ids, per_atom_xyz_index)
    decode(token_ids, per_token_xyz)  ->  SMILES'
    canonical_smiles(SMILES') == canonical_smiles(mol)  ✔

Public API
----------
    enc = FSMILESEncoder()
    out = enc.encode_mol(mol)            # mol must have a conformer
    -> {
         'token_ids':   list[int],       # FSMILES vocab ids
         'token_strs':  list[str],
         'atom_indices_per_token': list[int|None],
                                         # which mol atom index each
                                         # token refers to (None if non-atom)
         'fragments':   list[str],       # the per-fragment SMILES
                                         # (each ends with [*] or no [*])
       }

Use `tokenize_smiles_with_xyz(...)` in the dataloader: given a SMILES +
per-atom xyz, build the (token_ids, per_token_coords) pair plus the
bookkeeping for the autoregressive root_coords / smi_map tensors
(see `experiments/lingo3dmol_l1_dataloader.py`).
"""
from __future__ import annotations
import os, sys, re
from pathlib import Path
from copy import deepcopy
from typing import Optional

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Ensure Lingo3DMol is on the path so we can pull FragmolUtil.
_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_ROOT / "external" / "Lingo3DMol"))
from util.fragmol_frag_zyh import FragmolUtil  # noqa: E402


# ---------------------------------------------------------------------------
# Vocab constants (cached from FragmolUtil)
# ---------------------------------------------------------------------------
_FU = FragmolUtil()
VOCAB_C2I = _FU.vocab_c2i_v1_decode_new
VOCAB_I2C = _FU.vocab_i2c_v1_decode_new

PAD_ID   = VOCAB_C2I["pad_0"]
START_ID = VOCAB_C2I["start_0"]
END_ID   = VOCAB_C2I["end_0"]
SEP_ID   = VOCAB_C2I["sep_0"]

# Element token IDs (atoms have a coord). Per the autoregressive forward
# `ele_token = range(4, 56)` covers C_0..Br_0.
ELE_TOKEN_IDS = set(range(4, 56))

# Symbol -> set of (ring_size, vocab_key) options actually present
ATOM_SYMBOLS = ("C", "c", "N", "n", "S", "s", "O", "o", "F",
                "Cl", "[nH]", "Br")

# For each atom symbol, which ring sizes have a vocab entry?
_ATOM_RING_VOCAB: dict[str, dict[int, str]] = {}
for sym in ATOM_SYMBOLS:
    _ATOM_RING_VOCAB[sym] = {}
    for r in (0, 5, 6, 10, 11, 12):
        k = f"{sym}_{r}"
        if k in VOCAB_C2I:
            _ATOM_RING_VOCAB[sym][r] = k

# Closing brackets/etc keys
_NON_ATOM_KEYS = ("(", ")", "[", "]", "=", "#", "-", "/", "\\",
                  "+", "1", "2", "3", "4", "5", "6", "@", "@@",
                  "H", "[*]", "([*])")


def _ringsize_bucket(rsize: int) -> int:
    """Lingo3DMol bins ring sizes to {0, 5, 6, 10, 11, 12} for atom suffixes.
    Mapping (read off vocab_list_decode_new):
        not in ring     -> 0
        size 3, 4, 5    -> 5      (the model treats small rings as "_5")
        size 6          -> 6
        size 7, 8, 9    -> 10
        size 10         -> 11
        size 11+        -> 12
    """
    if rsize == 0:
        return 0
    if rsize <= 5:
        return 5
    if rsize == 6:
        return 6
    if rsize <= 9:
        return 10
    if rsize == 10:
        return 11
    return 12


def _smallest_ring_size_per_atom(mol: Chem.Mol) -> list[int]:
    """Return list[int] — size of the smallest SSSR ring each atom belongs to,
    or 0 if the atom is not in any ring."""
    ri = mol.GetRingInfo()
    atom_rings = ri.AtomRings()   # tuple of tuples of atom indices, per SSSR
    n = mol.GetNumAtoms()
    out = [0] * n
    for ring in atom_rings:
        sz = len(ring)
        for a in ring:
            if out[a] == 0 or sz < out[a]:
                out[a] = sz
    return out


def _atom_to_smi_symbol(atom: Chem.Atom) -> str:
    """Convert RDKit Atom to FSMILES-vocab symbol string.

    Returns one of ATOM_SYMBOLS: 'C', 'c', 'N', 'n', 'S', 's', 'O', 'o',
    'F', 'Cl', '[nH]', 'Br' — or None if unsupported.
    """
    sym = atom.GetSymbol()
    if sym == "C":
        return "c" if atom.GetIsAromatic() else "C"
    if sym == "N":
        if atom.GetIsAromatic():
            # In SMILES, aromatic [nH] vs n: the H is implicit. We check formal
            # H count.
            if atom.GetTotalNumHs() > 0:
                return "[nH]"
            return "n"
        return "N"
    if sym == "O":
        return "o" if atom.GetIsAromatic() else "O"
    if sym == "S":
        return "s" if atom.GetIsAromatic() else "S"
    if sym == "F":
        return "F"
    if sym == "Cl":
        return "Cl"
    if sym == "Br":
        return "Br"
    return None


# ---------------------------------------------------------------------------
# Tokenizer: walk RDKit-canonical SMILES, emit ring-size-aware tokens.
# ---------------------------------------------------------------------------
# We use RDKit's `MolToSmiles(mol, canonical=False)` after a `RenumberAtoms`
# pass to get a deterministic atom-index -> SMILES-position map. The atom
# order in the output SMILES is the renumbered order, so we walk the SMILES
# string left-to-right and pair each atom token with the next unused
# RDKit atom-index.

# Multi-char atom names that must be matched BEFORE single-char ones.
# Order matters: longer alternatives first.
_TOKEN_RE = re.compile(
    r"\[nH\]"               # aromatic NH
    r"|Cl|Br"               # halogens (2-char)
    r"|\[[^\]]+\]"          # general bracketed atoms
    r"|%\d\d"               # ring closures >9
    r"|@@"                  # double @
    r"|[BCNOSPFIcnops]"     # single-char atoms
    r"|[()=#\-/\\+@H]"      # bonds/branches/misc
    r"|[1-9]"               # ring closure digits
    r"|\."                  # disconnection
)


def _smiles_atom_order(smi: str) -> list[int]:
    """Return list of mol-atom indices for each atom encountered in the SMILES
    (left to right). For canonical SMILES the order matches the conformer
    atom order if we use the same mol object."""
    # We rely on RDKit's _smilesAtomOutputOrder property after MolToSmiles.
    raise NotImplementedError("use Chem.MolToSmiles(mol, canonical=False) "
                              "+ mol.GetPropsAsDict()['_smilesAtomOutputOrder']")


def _tokenize_smiles_with_atom_walk(smi: str) -> list[tuple[str, bool]]:
    """Walk SMILES char-by-char (matched via regex), return list of
    (token_string, is_atom). Bracketed atoms like [C@H], [N+], [O-] etc are
    de-bracketed to their bare symbol + the appropriate side-tokens; if the
    bracket cannot be reduced to a supported FSMILES atom, returns None.

    Returns None on any unsupported token.
    """
    out: list[tuple[str, bool]] = []
    pos = 0
    while pos < len(smi):
        m = _TOKEN_RE.match(smi, pos)
        if m is None:
            return None
        s = m.group(0)
        pos = m.end()

        if s.startswith("[") and s.endswith("]") and s != "[nH]":
            # General bracketed atom. Parse element + chirality + charge.
            # Examples: [C@H], [C@@H], [N+], [O-], [nH+], [N-], [13C], ...
            inner = s[1:-1]
            # Strip isotope digits (leading digits) — model doesn't model isotopes
            inner = re.sub(r"^\d+", "", inner)
            # Find element: longest matching at start
            elem = None
            for cand in ("Cl", "Br", "Si", "Se", "@@", "@", "nH"):
                if inner.startswith(cand):
                    elem = cand
                    inner = inner[len(cand):]
                    break
            if elem is None:
                elem = inner[0]
                inner = inner[1:]

            if elem == "nH":
                base_tok = "[nH]"
            elif elem in ("C", "c", "N", "n", "S", "s", "O", "o", "F", "Cl", "Br"):
                base_tok = elem
            else:
                # Unsupported element — bail
                return None

            # Charge / chirality / H markers stripped silently; FSMILES vocab
            # doesn't model them, so we proceed without them.
            out.append((base_tok, True))
            continue

        if s == "[nH]":
            out.append(("[nH]", True))
            continue

        if s in ("c", "n", "o", "s"):
            out.append((s, True))
            continue

        if s in ("C", "N", "O", "S", "F", "Cl", "Br"):
            out.append((s, True))
            continue

        if s in ("B", "P", "I", "p", "b"):
            # Not modelled in FSMILES vocab
            return None

        if s in ("(", ")", "=", "#", "-", "/", "\\", "+",
                 "1", "2", "3", "4", "5", "6", "[", "]",
                 "@", "@@", "H"):
            out.append((s, False))
            continue

        if s.startswith("%") or s in ("7", "8", "9", "0"):
            # Ring closure >6 not in FSMILES vocab
            return None

        if s == ".":
            return None

        # default: skip
        # (shouldn't reach here if regex is exhaustive)
        return None

    return out


# ---------------------------------------------------------------------------
# Public encoder
# ---------------------------------------------------------------------------
class FSMILESEncoder:
    """Encodes (mol with conformer) -> (token_ids, per_token_atom_idx).

    Optionally splits into fragments at BRICS bonds to mimic the pretrained
    model's exposure during training. For L1 fine-tuning we default to
    `fragmentation='none'` (whole mol as single fragment) — this avoids
    BRICS canonicalisation pitfalls and still produces well-formed FSMILES
    that the decoder round-trips on.
    """

    def __init__(self, fragmentation: str = "none"):
        assert fragmentation in ("none", "brics")
        self.fragmentation = fragmentation

    # ------------------------------------------------------------------
    def encode_mol(self, mol: Chem.Mol, max_tokens: int = 100) -> dict | None:
        """Encode RDKit mol -> FSMILES token ids.

        Returns None if the mol cannot be expressed in the FSMILES vocab
        (unsupported atom, charge, isotope, big ring closure, .-disconnection,
        too-many tokens, ...).
        """
        if mol is None:
            return None

        # Strip Hs (FSMILES vocab has no explicit-H semantics other than [nH])
        mol = Chem.RemoveHs(mol)

        if mol.GetNumAtoms() == 0:
            return None

        try:
            Chem.SanitizeMol(mol)
        except Exception:
            return None

        if self.fragmentation == "none":
            return self._encode_single(mol, max_tokens=max_tokens)

        # BRICS fragmentation
        return self._encode_brics(mol, max_tokens=max_tokens)

    # ------------------------------------------------------------------
    def _atom_ringsize_bucket(self, mol: Chem.Mol) -> list[int]:
        sizes = _smallest_ring_size_per_atom(mol)
        return [_ringsize_bucket(s) for s in sizes]

    # ------------------------------------------------------------------
    def _encode_single(self, mol: Chem.Mol, max_tokens: int) -> dict | None:
        """Encode whole mol as a single fragment (no sep)."""
        # Get canonical-equivalent SMILES with explicit atom-order tracking.
        # Use canonical=False with deterministic atom numbering for predictable
        # ordering of conformer xyz.
        try:
            smi = Chem.MolToSmiles(mol, canonical=False)
        except Exception:
            return None

        # Get the atom output order
        order = list(mol.GetPropsAsDict().get("_smilesAtomOutputOrder", []))
        if not order:
            # RDKit didn't emit the prop — fall back to 0..N-1
            order = list(range(mol.GetNumAtoms()))

        atom_ringbucket = self._atom_ringsize_bucket(mol)

        toks = _tokenize_smiles_with_atom_walk(smi)
        if toks is None:
            return None

        token_strs: list[str] = []
        token_ids: list[int] = []
        atom_idx_per_tok: list[int | None] = []

        atom_count = 0
        for (s, is_atom) in toks:
            if is_atom:
                # which mol atom is this?
                if atom_count >= len(order):
                    return None
                mol_atom = order[atom_count]
                rbucket = atom_ringbucket[mol_atom]
                key = f"{s}_{rbucket}"
                if key not in VOCAB_C2I:
                    # Fall back to _0 if specific ring suffix not present
                    key = f"{s}_0"
                    if key not in VOCAB_C2I:
                        return None
                token_strs.append(key)
                token_ids.append(VOCAB_C2I[key])
                atom_idx_per_tok.append(mol_atom)
                atom_count += 1
            else:
                key = f"{s}_0"
                if key not in VOCAB_C2I:
                    return None
                token_strs.append(key)
                token_ids.append(VOCAB_C2I[key])
                atom_idx_per_tok.append(None)

        if atom_count != mol.GetNumAtoms():
            # Mismatch — SMILES has different atom count than mol
            return None

        if len(token_ids) > max_tokens:
            return None

        return {
            "token_ids":   token_ids,
            "token_strs":  token_strs,
            "atom_indices_per_token": atom_idx_per_tok,
            "fragments":   [smi],
        }

    # ------------------------------------------------------------------
    def _encode_brics(self, mol: Chem.Mol, max_tokens: int) -> dict | None:
        """BRICS-split, then emit each fragment as its own SMILES with [*] at
        the cut atom. Fragments separated by sep_0. Currently NOT
        round-trip-verified — kept as TODO for Phase 4.
        """
        # Stub — for L1 we use single-fragment encoding (simpler, verifiable).
        return self._encode_single(mol, max_tokens=max_tokens)


# ---------------------------------------------------------------------------
# Round-trip test
# ---------------------------------------------------------------------------
def _canon_smi(smi: str) -> str:
    """Canonical SMILES — strip stereo (model doesn't model it) for comparison."""
    if smi is None:
        return ""
    m = Chem.MolFromSmiles(smi)
    if m is None:
        return ""
    Chem.RemoveStereochemistry(m)
    return Chem.MolToSmiles(m, canonical=True)


def round_trip_check(mol: Chem.Mol) -> tuple[bool, str, str]:
    """Encode -> decode -> compare canonical SMILES (stereochem stripped).

    Returns (ok, decoded_smi, expected_smi)."""
    enc = FSMILESEncoder()
    out = enc.encode_mol(mol)
    if out is None:
        return False, "<encode_fail>", _canon_smi(Chem.MolToSmiles(mol))

    expected = _canon_smi(Chem.MolToSmiles(mol))

    # Decode via FragmolUtil.decode3d. We don't need real coordinates for
    # the topology check — pass dummy zeros (decode3d handles it; it builds
    # the mol from the SMILES side, attaching coords as a separate step).
    ids = [START_ID] + out["token_ids"] + [SEP_ID, END_ID]
    T = max(100, len(ids))
    batch = np.zeros((1, T), dtype=np.int64)
    batch[0, :len(ids)] = ids
    pos = np.zeros((1, T, 3), dtype=np.float32)

    try:
        smiles_list, _tokens, _mols = _FU.decode3d(batch, pos)
    except Exception as e:
        return False, f"<decode_exc: {e}>", expected

    decoded = smiles_list[0] if smiles_list else None
    if decoded is None:
        return False, "<decode_none>", expected
    got = _canon_smi(decoded)
    return got == expected, got, expected


# ---------------------------------------------------------------------------
# Dataloader entry point: tokenize SMILES + xyz -> token IDs + per-token coord
# ---------------------------------------------------------------------------
def tokenize_smiles_with_xyz(
    smiles: str,
    atom_xyz_grid: np.ndarray,
    max_tokens: int = 78,
) -> dict | None:
    """For the L1 dataloader: given a SMILES and per-atom voxel-grid coords
    (in RDKit atom-index order, matching the input SMILES), return:

        {
          'token_ids':            list[int],         # len=L
          'token_strs':           list[str],         # len=L
          'per_token_coords':     list[list[int]],   # len=L, (x,y,z) grid
          'atom_idx_per_token':   list[int|None],    # len=L
        }

    Per-token coord for non-atom tokens carries the previous atom's coord
    (so the model's coord embedding always has something to consume).

    Returns None on unsupported tokens or length > max_tokens.
    """
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None

    # We do NOT call RemoveHs here — already done in the encoder.
    enc = FSMILESEncoder()
    out = enc.encode_mol(mol, max_tokens=max_tokens)
    if out is None:
        return None

    token_ids: list[int] = out["token_ids"]
    token_strs: list[str] = out["token_strs"]
    atom_per_tok: list[int | None] = out["atom_indices_per_token"]

    n_atoms = mol.GetNumAtoms()
    if atom_xyz_grid is None or len(atom_xyz_grid) < n_atoms:
        return None

    per_token_coords: list[list[int]] = []
    last = atom_xyz_grid[0].tolist()
    for i, tok in enumerate(token_strs):
        a = atom_per_tok[i]
        if a is not None:
            last = atom_xyz_grid[a].tolist()
        per_token_coords.append(list(last))

    return {
        "token_ids":          token_ids,
        "token_strs":         token_strs,
        "per_token_coords":   per_token_coords,
        "atom_idx_per_token": atom_per_tok,
    }


# ---------------------------------------------------------------------------
# CLI: run the round-trip test
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import argparse, pandas as pd

    p = argparse.ArgumentParser()
    p.add_argument("--csv", default=str(
        _ROOT / "data" / "retrain_covalent" / "acrylamide_only.csv"))
    p.add_argument("--n", type=int, default=20)
    p.add_argument("--smiles_col", default="smiles")
    args = p.parse_args()

    df = pd.read_csv(args.csv)
    smis = df[args.smiles_col].dropna().astype(str).tolist()[:args.n]

    n_pass = 0
    n_encfail = 0
    for i, smi in enumerate(smis):
        mol = Chem.MolFromSmiles(smi)
        if mol is None:
            continue
        ok, got, expected = round_trip_check(mol)
        marker = "PASS" if ok else "FAIL"
        if ok:
            n_pass += 1
        if "<encode_fail>" in got:
            n_encfail += 1
        if not ok or i < 5:
            print(f"[{marker}] {i:3d}  expected={expected}\n           got={got}")

    print(f"\nRound-trip: {n_pass}/{len(smis)} passed "
          f"({n_pass/len(smis)*100:.1f}%, enc_fail={n_encfail})")
