"""L1 Phase 1 smoke-batch builder.

Exports `build_smoke_batch(...)` that the L1 train script imports to assemble
a single CPU batch:

  - load N SMILES from `data_csv` (`smiles` column)
  - ETKDG+MMFF embed each SMILES; skip rows that fail embedding
  - tokenize SMILES into FSMILES ids using FragmolUtil.vocab_c2i_v1_decode_new
    (start_0 prepended, sep_0 appended)
  - per-token grid coords: atom tokens carry the atom's voxel coord,
    non-atom tokens (=, (, ), [*], ring digits, etc) carry the most recent
    atom's coord — same recipe as L2 v3 (run_lingo3dmol_l2_inpaint.py).
  - right-pad to T tokens; warn+skip if SMILES exceeds T.
  - encode pocket via PocketCode.pocketCodeNCI(pocket_pdb, lig_pos=...)
  - critical_anchor is set to zeros (B, 500) as per the synthetic
    __main__ test in transformer_v1_res_fac2.py — the encoder still learns.
  - contact_idx is the argmax of the contact channel, or 0 if empty.

All tensors are CPU.

We DO NOT bring in a sophisticated FSMILES fragment-grammar tokenizer (the
repo never exposes one; FragmolUtil only decodes). Per-atom char-level
tokenization is sufficient for L1 gradient-flow + overfit smoke, and matches
the way L2 v3 hand-builds its acryl prefix.
"""
from __future__ import annotations
import os, re, sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem

RDLogger.DisableLog("rdApp.*")

# Both these are guaranteed importable by the time L1 train calls us (the
# train script has already inserted external/Lingo3DMol into sys.path).
from util.fragmol_frag_zyh import FragmolUtil
from util.pocket_code_all import PocketCode


# ---------------------------------------------------------------------------
# FSMILES vocab IDs (cache once)
# ---------------------------------------------------------------------------
_FU = FragmolUtil()
V = _FU.vocab_c2i_v1_decode_new

# Sanity check key tokens exist
for _t in ('start_0', 'sep_0', 'pad_0', 'end_0',
          'C_0', 'c_0', 'N_0', 'n_0', 'O_0', 'o_0', 'S_0', 's_0',
          'F_0', 'Cl_0', 'Br_0', '[nH]_0',
          '=_0', '#_0', '-_0', '(_0', ')_0',
          '1_0', '2_0', '3_0', '4_0', '5_0', '6_0',
          '[_0', ']_0', '[*]_0'):
    assert _t in V, f"vocab missing token: {_t}"

PAD_ID   = V['pad_0']
START_ID = V['start_0']
END_ID   = V['end_0']
SEP_ID   = V['sep_0']

# Atoms that map to a single "atom" token (the only ones with a coord).
# Multi-char atom symbols must be matched BEFORE single-char ones in the regex.
ATOM_SYMBOLS = ('Cl', 'Br', '[nH]', 'C', 'c', 'N', 'n', 'O', 'o', 'S', 's', 'F')

# Regex matching one FSMILES character token at a time. Order matters: longer
# atom symbols come first so 'Cl' is not split into 'C' + 'l'.
_ATOM_PATTERN = (
    r'Cl|Br|\[nH\]|'                # multi-char aliphatic / aromatic
    r'[CcNnOoSsF]|'                  # single-char aliphatic / aromatic / halogen
    r'\[\*\]|'                       # open-bond marker
    r'='   r'|'                      # double bond
    r'#'   r'|'                      # triple bond
    r'-'   r'|'                      # single bond explicit
    r'/'   r'|'                      # cis/trans
    r'\\' r'|'                       # cis/trans
    r'\('  r'|'                      # branch open
    r'\)'  r'|'                      # branch close
    r'[1-6]'                          # ring closure digits 1..6
)
TOKEN_RE = re.compile(_ATOM_PATTERN)

# Map a regex hit -> FSMILES vocab key (suffix `_0`)
def _tok_key(s: str) -> str:
    return s + '_0'

# Quick lookup: which regex hits correspond to atoms (need a coord)
_ATOM_SET = set(ATOM_SYMBOLS)


# ---------------------------------------------------------------------------
# SMILES tokenizer + per-token coords
# ---------------------------------------------------------------------------
def _tokenize_smiles(smiles: str, atom_xyz_grid: np.ndarray):
    """Tokenize SMILES with a regex; align each atom token to its grid coord.

    `atom_xyz_grid` is (n_atoms, 3) — voxel grid coords, in the same order
    as RDKit's atom indices for `Chem.MolFromSmiles(smiles)` followed by
    `Chem.RemoveHs` (no canonicalisation). For the smoke we use the SMILES
    as written; the conformer atom order matches the SMILES atom order.

    Returns
    -------
    tokens : list[int]   FSMILES token ids (no start/sep yet)
    coords : list[list[int]]   per-token (x,y,z) grid, non-atom tokens carry
             the previous atom's coord.
    n_atoms_used : int   number of atom tokens actually consumed.
    """
    tokens = []
    coords = []
    atom_idx = 0
    last_coord = atom_xyz_grid[0].tolist() if len(atom_xyz_grid) else [0, 0, 0]

    pos = 0
    while pos < len(smiles):
        m = TOKEN_RE.match(smiles, pos)
        if m is None:
            # Skip unparseable char (we don't tokenize @, [, ] beyond atoms;
            # smoke ligands are mostly acrylamide derivatives without stereo).
            pos += 1
            continue
        s = m.group(0)
        key = _tok_key(s)
        if key not in V:
            # Unknown token — skip, log later if needed.
            pos = m.end()
            continue
        tokens.append(V[key])
        if s in _ATOM_SET:
            if atom_idx < len(atom_xyz_grid):
                last_coord = atom_xyz_grid[atom_idx].tolist()
            atom_idx += 1
        coords.append(list(last_coord))
        pos = m.end()
    return tokens, coords, atom_idx


def _embed_smiles_3d(smiles: str, seed: int = 42):
    """ETKDG embed + MMFF optimise. Returns (n_atoms, 3) xyz (Å, RDKit order)
    or None on failure."""
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None, None
    molH = Chem.AddHs(mol)
    params = AllChem.ETKDGv3()
    params.randomSeed = seed
    if AllChem.EmbedMolecule(molH, params) != 0:
        return None, None
    try:
        AllChem.MMFFOptimizeMolecule(molH, maxIters=200)
    except Exception:
        # MMFF failure is OK if conformer exists — keep going.
        pass
    mol3d = Chem.RemoveHs(molH)
    conf = mol3d.GetConformer()
    xyz = np.array(
        [list(conf.GetAtomPosition(i)) for i in range(mol3d.GetNumAtoms())],
        dtype=np.float64,
    )
    return mol3d, xyz


def _xyz_to_grid(xyz: np.ndarray, center: np.ndarray) -> np.ndarray:
    """Map Å xyz -> voxel grid via L2 v3 formula."""
    grid = (xyz - center) / 0.1 + 119.5
    return np.rint(grid).astype(np.int64)


# ---------------------------------------------------------------------------
# Pocket encoder
# ---------------------------------------------------------------------------
_PC = PocketCode()


def _encode_pocket(pocket_pdb: str, lig_xyz: np.ndarray | None = None):
    """Run pocketCodeNCI on the pocket. Returns dict of pocket tensors."""
    type_, residue, mask, new_coords, center, contact, contact_scaffold = \
        _PC.pocketCodeNCI(pocket_pdb, center=None, pocket_contact=None,
                          lig_pos=lig_xyz)
    return {
        'type': np.array(type_,    dtype=np.int64),
        'residue': np.array(residue, dtype=np.int64),
        'mask': np.array(mask,     dtype=np.float32),
        'coords': np.array(new_coords, dtype=np.int64),
        'center': np.array(center, dtype=np.float64),
        'contact': np.array(contact, dtype=np.float32),
        'contact_scaffold': np.array(contact_scaffold, dtype=np.float32),
    }


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def build_smoke_batch(data_csv: str, batch_size: int, pocket_pdb: str,
                      T: int = 60):
    """Assemble a single training batch on CPU.

    Returns
    -------
    dict matching `TransformerModel.forward_train` kwargs.
    """
    df = pd.read_csv(data_csv)
    assert 'smiles' in df.columns, f"CSV must have a `smiles` column, got: {df.columns}"

    # Encode pocket ONCE; replicate for the batch.
    # We don't yet have ligand xyz so pass None — contact_scaffold will be
    # all-zero, which is fine for gradient-flow smoke.
    pocket = _encode_pocket(pocket_pdb, lig_xyz=None)
    pocket_center = pocket['center']
    print(f"[smoke] pocket: type/res/mask/coords/contact loaded; "
          f"center={pocket_center.tolist()}")
    print(f"[smoke] non-padding pocket atoms: {int(pocket['mask'].sum())}")

    # Collect `batch_size` ligands; skip on tokenization or embed failure.
    accepted = []   # list of dict(tokens, coords)
    n_tried, n_skipped = 0, 0
    for smi in df['smiles'].astype(str).tolist():
        if len(accepted) >= batch_size:
            break
        n_tried += 1
        mol3d, xyz = _embed_smiles_3d(smi)
        if xyz is None:
            n_skipped += 1
            continue
        # ETKDG embeds the ligand in its own (arbitrary) frame, near origin.
        # Translate so the ligand centroid sits at the pocket centroid (the
        # voxel grid is centred there). This is sufficient for smoke
        # gradient-flow — we are NOT testing geometric realism.
        ligand_centroid = xyz.mean(axis=0)
        xyz_translated = xyz - ligand_centroid + pocket_center
        grid_xyz = _xyz_to_grid(xyz_translated, pocket_center)
        # Grid range check: if out of [0, 240), skip — the model will OOB the embedding.
        if grid_xyz.min() < 0 or grid_xyz.max() >= 240:
            print(f"[smoke] skip (grid OOB [0,240)): smi={smi[:40]}... "
                  f"range=[{grid_xyz.min()},{grid_xyz.max()}]")
            n_skipped += 1
            continue
        # Use canonical SMILES from the embedded mol so atom order matches xyz.
        canon = Chem.MolToSmiles(mol3d, canonical=False)
        toks, per_tok_coords, n_atoms_used = _tokenize_smiles(canon, grid_xyz)
        # Prepend start_0 (coord = first atom's coord), append sep_0 (carries last).
        toks   = [START_ID] + toks + [SEP_ID]
        # start coord = first atom grid (model uses contact_idx but smoke doesn't care)
        first_coord = grid_xyz[0].tolist() if len(grid_xyz) else [0, 0, 0]
        last_coord  = per_tok_coords[-1] if per_tok_coords else first_coord
        per_tok_coords = [list(first_coord)] + per_tok_coords + [list(last_coord)]

        if len(toks) > T:
            print(f"[smoke] skip (too long {len(toks)}>{T}): smi={smi[:40]}...")
            n_skipped += 1
            continue
        accepted.append({'tokens': toks, 'coords': per_tok_coords, 'smi': canon})

    if len(accepted) < batch_size:
        raise RuntimeError(
            f"Could not assemble batch of {batch_size}; got only {len(accepted)} "
            f"after {n_tried} tries (skipped {n_skipped})."
        )

    print(f"[smoke] accepted {len(accepted)}/{n_tried} ligands "
          f"(skipped {n_skipped}). Token-seq lengths: "
          f"{[len(a['tokens']) for a in accepted]}")
    for i, a in enumerate(accepted):
        print(f"  lig {i}: T={len(a['tokens'])}  smi={a['smi']}")

    B = len(accepted)
    # Pad on the right
    target_token  = np.full((B, T), PAD_ID, dtype=np.int64)
    target_coords = np.zeros((B, T, 3), dtype=np.int64)
    target_mask   = np.zeros((B, T), dtype=np.float32)
    for i, a in enumerate(accepted):
        L = len(a['tokens'])
        target_token[i, :L]      = np.asarray(a['tokens'], dtype=np.int64)
        target_coords[i, :L, :] = np.asarray(a['coords'], dtype=np.int64)
        target_mask[i, :L]       = 1.0

    # Pocket tensors — replicate across batch.
    coords_p   = np.broadcast_to(pocket['coords'][None],   (B, 500, 3)).copy()
    type_p     = np.broadcast_to(pocket['type'][None],     (B, 500)).copy()
    residue_p  = np.broadcast_to(pocket['residue'][None],  (B, 500)).copy()
    mask_p     = np.broadcast_to(pocket['mask'][None],     (B, 500)).copy()

    # critical_anchor: zeros (same as synthetic __main__).
    critical_anchor = np.zeros((B, 500), dtype=np.int64)

    # contact_idx: pick the highest-contact pocket atom, else 0.
    contact_arr = pocket['contact']
    if contact_arr.sum() > 0:
        cidx = int(np.argmax(contact_arr))
    else:
        cidx = 0
    contact_idx = np.full((B,), cidx, dtype=np.int64)

    batch = {
        'coords':           torch.from_numpy(coords_p).long(),
        'type':             torch.from_numpy(type_p).long(),
        'residue':          torch.from_numpy(residue_p).long(),
        'critical_anchor':  torch.from_numpy(critical_anchor).long(),
        'src_mask':         torch.from_numpy(mask_p).float(),
        'target_token':     torch.from_numpy(target_token).long(),
        'target_coords':    torch.from_numpy(target_coords).long(),
        'target_mask':      torch.from_numpy(target_mask).float(),
        'contact_idx':      torch.from_numpy(contact_idx).long(),
    }

    # Final shape sanity
    expected_shapes = {
        'coords':          (B, 500, 3),
        'type':            (B, 500),
        'residue':         (B, 500),
        'critical_anchor': (B, 500),
        'src_mask':        (B, 500),
        'target_token':    (B, T),
        'target_coords':   (B, T, 3),
        'target_mask':     (B, T),
        'contact_idx':     (B,),
    }
    for k, expected in expected_shapes.items():
        got = tuple(batch[k].shape)
        assert got == expected, f"{k}: expected {expected}, got {got}"

    print(f"[smoke] batch built: B={B} T={T} contact_idx={cidx}")
    return batch


# ---------------------------------------------------------------------------
# CLI: print a built batch (for manual inspection).
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import argparse
    p = argparse.ArgumentParser()
    p.add_argument('--data_csv', required=True)
    p.add_argument('--pocket_pdb', required=True)
    p.add_argument('--batch_size', type=int, default=4)
    p.add_argument('--seq_len', type=int, default=60)
    args = p.parse_args()
    b = build_smoke_batch(args.data_csv, args.batch_size,
                          pocket_pdb=args.pocket_pdb, T=args.seq_len)
    for k, v in b.items():
        print(f"  {k}: shape={tuple(v.shape)} dtype={v.dtype} "
              f"min={v.min().item()} max={v.max().item()}")
