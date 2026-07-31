"""Build a PocketFlow-compatible Ligand dict pre-seeded with the acrylamide warhead.

Replaces `Ligand.empty_dict()` in `main_generate.py:54`. Warhead positions are
read from a 5-atom SDF (Cβ=Cα-C(=O)-N) e.g. `anchordiff/pockets/zap70_cys346/warhead_at_cys.sdf`.

The seed atoms are ordered:
  idx 0  Cβ   (Michael-attacked carbon — focus for autoregressive growth)
  idx 1  Cα
  idx 2  Ccarb
  idx 3  O   (carbonyl)
  idx 4  N   (amide)

Bonds: 0-1 DOUBLE, 1-2 SINGLE, 2-3 DOUBLE, 2-4 SINGLE.

Returns a dict with the same keys as `Ligand.empty_dict()` but populated. The
critical bit is to point the autoregressive focus at idx 0 so growth extends
FROM Cβ outward.
"""
from __future__ import annotations
import numpy as np
from pathlib import Path
from rdkit import Chem

# PocketFlow's atom_type_map for the ZINC ckpt
ATOM_TYPE_MAP = [6, 7, 8, 9, 15, 16, 17, 35, 53]

# Bond type codes used by PocketFlow:
#   1 = SINGLE, 2 = DOUBLE, 3 = TRIPLE  (4 = AROMATIC unused at gen-time)
BOND_TYPE_MAP = {
    Chem.rdchem.BondType.SINGLE: 1,
    Chem.rdchem.BondType.DOUBLE: 2,
    Chem.rdchem.BondType.TRIPLE: 3,
    Chem.rdchem.BondType.AROMATIC: 4,
}

# Element used for atom_feature one-hot (ATOM_FAMILIES has 8 entries; warhead
# atoms are sp2/sp3 C, N, O — all "Acceptor"/"Donor" tagged but for the seed
# we leave the feat_mat blank since the model conditions on element + bonds.)
ATOM_FAMILY_DIM = 8


def _easydict():
    """PocketFlow uses easydict.EasyDict for ligand dicts."""
    try:
        from easydict import EasyDict
        return EasyDict
    except ImportError:
        # fall back: a plain attribute-dict
        class _D(dict):
            def __getattr__(self, k): return self[k]
            def __setattr__(self, k, v): self[k] = v
        return _D


def load_acrylamide_seed_from_sdf(sdf_path: str) -> dict:
    """Read 5-atom acrylamide SDF and return a Ligand-shaped dict.

    SDF must contain exactly 5 atoms in the canonical order (Cβ, Cα, Ccarb, O, N).
    """
    suppl = Chem.SDMolSupplier(sdf_path, removeHs=True, sanitize=False)
    mol = None
    for m in suppl:
        if m is not None:
            mol = m
            break
    assert mol is not None, f"could not load mol from {sdf_path}"

    # Kekulize so we get explicit DOUBLE bond on Cβ=Cα and C=O
    try:
        Chem.SanitizeMol(mol)
    except Exception:
        pass
    Chem.Kekulize(mol, clearAromaticFlags=True)

    conf = mol.GetConformer()
    elements = []
    pos = []
    for a in mol.GetAtoms():
        elements.append(a.GetAtomicNum())
        p = conf.GetAtomPosition(a.GetIdx())
        pos.append([p.x, p.y, p.z])
    elements = np.asarray(elements, dtype=np.int64)
    pos = np.asarray(pos, dtype=np.float32)

    # Build bond_index/bond_type arrays in PocketFlow's symmetric format
    edge_index = []
    edge_type = []
    for b in mol.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        bt = BOND_TYPE_MAP[b.GetBondType()]
        edge_index.append([i, j])
        edge_index.append([j, i])
        edge_type.extend([bt, bt])
    edge_index = np.asarray(edge_index, dtype=np.int64)
    # sort by first endpoint (PocketFlow expects this)
    if edge_index.size:
        perm = edge_index[:, 0].argsort()
        edge_index = edge_index[perm].T
        edge_type = np.asarray(edge_type, dtype=np.int64)[perm]
    else:
        edge_index = np.empty([2, 0], dtype=np.int64)
        edge_type = np.empty(0, dtype=np.int64)

    atom_mass = np.array([a.GetMass() for a in mol.GetAtoms()], dtype=np.float32)
    center_of_mass = (pos * atom_mass[:, None]).sum(0) / atom_mass.sum()

    Dict = _easydict()
    out = Dict()
    out.element = elements
    out.pos = pos
    out.bond_index = edge_index
    out.bond_type = edge_type
    out.center_of_mass = center_of_mass.astype(np.float32)
    out.atom_feature = np.zeros((len(elements), ATOM_FAMILY_DIM), dtype=np.float32)
    out.ring_info = {}
    out.filename = str(sdf_path)
    # convenience: index of the Cβ atom = the focus seed atom
    # detection: Cβ is the C that has the C=C bond (DOUBLE between two carbons)
    cb_idx = None
    for b in mol.GetBonds():
        if b.GetBondType() == Chem.rdchem.BondType.DOUBLE:
            a, b_ = b.GetBeginAtom(), b.GetEndAtom()
            if a.GetAtomicNum() == 6 and b_.GetAtomicNum() == 6:
                # the Cβ is whichever has fewer heavy-atom neighbors (i.e.
                # the terminal CH2 of the acrylamide)
                cb_idx = a.GetIdx() if a.GetDegree() < b_.GetDegree() else b_.GetIdx()
                break
    if cb_idx is None:
        cb_idx = 0
    out.cb_seed_idx = int(cb_idx)
    return out


def build_seed_or_empty(sdf_path: str | None) -> tuple[dict, int | None]:
    """Helper used by inpaint.py / sample.py.

    Returns (ligand_dict, cb_seed_idx). If sdf_path is None or missing, returns
    an empty ligand dict (= vanilla PocketFlow).
    """
    if sdf_path is None or not Path(sdf_path).exists():
        # fall back to PocketFlow's empty ligand
        from pocket_flow.utils import Ligand  # type: ignore
        return Ligand.empty_dict(), None
    d = load_acrylamide_seed_from_sdf(sdf_path)
    return d, d.cb_seed_idx


if __name__ == "__main__":
    import sys
    p = sys.argv[1] if len(sys.argv) > 1 else None
    d, cb = build_seed_or_empty(p)
    print(f"loaded warhead seed: n_atoms={len(d.element)}  cb_seed_idx={cb}")
    print(f"  elements: {d.element.tolist()}")
    print(f"  pos[0]:   {d.pos[0].tolist()}")
    print(f"  bonds:    {d.bond_index.shape[1]//2} undirected")
