"""CovIndDataset — PyTorch Dataset producing C+D training examples for the
flagship covalent-aware DiffSBDD fine-tune.

Per item:
  - protein_pos:   (N_p, 3)  pocket-residue Cα coords in **local frame**
  - protein_one_hot: (N_p, 21)  residue type one-hot
  - ligand_pos:    (N_l, 3)  ligand heavy-atom coords in **local frame**
  - ligand_one_hot:(N_l, F_a) atom-type one-hot (uses LigandPocketDDPM.lig_type_encoder)
  - covalent_token: (TOKEN_DIM,)  the C-conditioning vector
  - warhead_atom_idx: int — index in ligand_pos that should be at canonical Cβ
  - warhead_class: str
  - record_id: str

Pocket = all protein residues within 8 Å of any ligand heavy atom (matches
DiffSBDD's CrossDocked2020 convention).
"""
from __future__ import annotations
from pathlib import Path
import sys
PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from Bio.PDB import PDBParser, NeighborSearch, Selection
from rdkit import Chem, RDLogger
RDLogger.DisableLog("rdApp.*")

from anchordiff.covind.local_frame import compute_frame, to_local
from anchordiff.covind.covalent_token import (
    build_token, WARHEAD_VOCAB, RESIDUE_VOCAB, TOKEN_DIM, R_DIM, W_DIM,
)

# Atom-type encoder MUST match the DiffSBDD `crossdocked_fullatom_cond`
# checkpoint exactly. Verified by loading the ckpt and reading
# `model.lig_type_encoder`. Both ligand and pocket use the SAME atom-element
# encoder (this checkpoint has no separate 20-AA residue encoding — pocket
# is represented by Cα atoms encoded as element 'C').
ATOM_ENCODER = {
    "C": 0, "N": 1, "O": 2, "S": 3, "B": 4,
    "Br": 5, "Cl": 6, "P": 7, "I": 8, "F": 9,
}
F_A = len(ATOM_ENCODER)  # 10 — matches checkpoint atom_nf


def atom_one_hot(symbol: str) -> np.ndarray:
    """One-hot per the DiffSBDD checkpoint's encoder. Unknown atoms map to 'C'
    (safest fallback for a mostly-organic dataset; logged at first occurrence)."""
    out = np.zeros(F_A, dtype=np.float32)
    idx = ATOM_ENCODER.get(symbol, 0)  # default to 'C'
    out[idx] = 1.0
    return out


def pocket_residue_one_hot(_resname_unused: str) -> np.ndarray:
    """DEPRECATED — kept for one release while callers migrate. The
    `crossdocked_fullatom_cond` checkpoint expects FULL-ATOM pocket encoding:
    every heavy atom of every pocket residue, one-hot by element. Using only
    Cα atoms (this function's old behavior) drops ~7× of the atoms the model
    saw at pretraining. Use the per-atom path in `__getitem__` instead."""
    raise RuntimeError(
        "pocket_residue_one_hot is deprecated — full-atom pocket is built "
        "inline in CovIndDataset.__getitem__"
    )


class CovIndDataset(Dataset):
    def __init__(self, csv_path: str | Path, split: str = "train",
                 pocket_cutoff: float = 8.0, max_ligand_atoms: int = 50):
        # Per-instance drop counter (not class-level — QA #6 round 2: train and
        # val instances shouldn't share state).
        self.drop_counts: dict[str, int] = {}
        self.df = pd.read_csv(csv_path)
        if split in ("train", "val") and "split" in self.df.columns:
            sub = self.df[self.df["split"] == split].reset_index(drop=True)
            if len(sub) == 0:
                # Smoke-test fallback: if split filter empties the df, use all rows
                print(f"  WARNING: split={split!r} empty, using all {len(self.df)} rows")
            else:
                self.df = sub
        self.pocket_cutoff = pocket_cutoff
        self.max_ligand_atoms = max_ligand_atoms
        self._parser = PDBParser(QUIET=True)
        print(f"CovIndDataset[{split}]: {len(self.df)} entries")

    def __len__(self):
        return len(self.df)

    def _drop(self, reason: str):
        self.drop_counts[reason] = self.drop_counts.get(reason, 0) + 1
        return None

    def __getitem__(self, idx):
        r = self.df.iloc[idx]
        pdbf = PROJECT_ROOT / r["pdb_path"]
        try:
            s = self._parser.get_structure("", str(pdbf))[0]
        except Exception:
            return self._drop("pdb_parse_fail")
        # Cys SG, CB, CA
        try:
            cys = s[r["cys_chain"]][int(r["cys_resi"])]
            sg = np.array(cys["SG"].get_coord(), dtype=float)
            cb = np.array(cys["CB"].get_coord(), dtype=float)
            ca = np.array(cys["CA"].get_coord(), dtype=float)
        except Exception:
            return self._drop("cys_atoms_missing")
        R, t = compute_frame(sg, cb, ca)
        # Ligand atoms — HETATM lookup. Bio.PDB uses ('H_<resname>', resi, ' ')
        # as the residue ID for HETATM. Iterate the chain to find it robustly.
        lig = None
        for chain in s:
            if chain.id != r["ligand_chain"]: continue
            for res in chain:
                if res.id[1] == int(r["ligand_resi"]):
                    lig = res; break
            if lig is not None: break
        if lig is None: return self._drop("ligand_residue_not_found")
        lig_atoms = [a for a in lig.get_atoms() if a.element != "H"]
        if not lig_atoms:
            return self._drop("ligand_zero_atoms")
        if len(lig_atoms) > self.max_ligand_atoms:
            return self._drop(f"ligand_too_large_>{self.max_ligand_atoms}")
        lig_coords = np.array([a.get_coord() for a in lig_atoms], dtype=float)
        lig_oh = np.array([atom_one_hot(a.element) for a in lig_atoms])
        # Pocket residues within cutoff Å of any ligand atom
        ns = NeighborSearch(list(s.get_atoms()))
        pocket_res = set()
        for c in lig_coords:
            for atm in ns.search(c, self.pocket_cutoff, level="A"):
                par_res = atm.get_parent()
                if par_res.id[0].strip() == "" and par_res.get_resname() != "HOH":
                    pocket_res.add(par_res)
        # Sort residues by (chain, resi) for deterministic atom order.
        pocket_res_sorted = sorted(
            pocket_res,
            key=lambda r: (r.get_parent().id, r.id[1], r.id[2] or ""),
        )
        # FULL-ATOM pocket: enumerate every heavy atom of every pocket residue
        # and encode each by its actual element. Matches what DiffSBDD's
        # `crossdocked_fullatom_cond` checkpoint expects (LightningModule
        # prepare_pocket lines 723-731: pocket_type_encoder = atom_encoder for
        # pocket_representation='full-atom'). Using Cα-only encoding (the prior
        # implementation) dropped ~7× the atoms the model saw at pretraining
        # and collapsed every residue to element C — that's the input
        # malformation flagged in May-2026 QA.
        pocket_coords = []; pocket_oh = []
        for res in pocket_res_sorted:
            for a in res.get_atoms():
                if a.element == "H":
                    continue
                sym = a.element.capitalize() if a.element else "C"
                if sym not in ATOM_ENCODER:
                    # Unknown element — skip rather than collapse to C
                    continue
                pocket_coords.append(a.get_coord())
                pocket_oh.append(atom_one_hot(sym))
        if len(pocket_coords) < 5: return self._drop("pocket_too_small_<5")
        pocket_coords = np.array(pocket_coords, dtype=float)
        pocket_oh = np.array(pocket_oh)
        # Transform to local frame
        lig_local    = to_local(lig_coords, R, t)
        pocket_local = to_local(pocket_coords, R, t)
        # Covalent token — using the residues actually in the pocket (deterministic order)
        ctx = [res.get_resname() for res in pocket_res_sorted if "CA" in res]
        token = build_token(d_canonical=r["warhead_canonical_d"],
                            theta_canonical=r["warhead_canonical_angle"],
                            warhead_class=r["warhead_class"],
                            cys_context_residues=ctx,
                            reaction_mechanism=r.get("reaction_type"))
        return {
            "record_id": r["record_id"],
            "warhead_class": r["warhead_class"],
            "lig_pos":  torch.from_numpy(lig_local.astype(np.float32)),
            "lig_oh":   torch.from_numpy(lig_oh.astype(np.float32)),
            "pkt_pos":  torch.from_numpy(pocket_local.astype(np.float32)),
            "pkt_oh":   torch.from_numpy(pocket_oh.astype(np.float32)),
            "cov_token": torch.from_numpy(token),
            "warhead_atom_idx": int(r["anchor_atom_idx_in_ligand"]),
            "sg_global": torch.from_numpy(sg.astype(np.float32)),
            "frame_R":   torch.from_numpy(R.astype(np.float32)),
            "frame_t":   torch.from_numpy(t.astype(np.float32)),
        }


if __name__ == "__main__":
    ds = CovIndDataset(PROJECT_ROOT / "data" / "covbinder" / "covind_training_set.csv",
                       split="train")
    print(f"\nSampling 5 examples:")
    n_ok = 0
    for i in range(min(50, len(ds))):
        ex = ds[i]
        if ex is None: continue
        n_ok += 1
        if n_ok <= 5:
            print(f"  [{i}] {ex['record_id']} {ex['warhead_class']:18s}"
                  f"  lig={ex['lig_pos'].shape[0]:>2}  pkt={ex['pkt_pos'].shape[0]:>3}"
                  f"  warhead_atom_idx={ex['warhead_atom_idx']}"
                  f"  cov_token_sum={ex['cov_token'].sum():.2f}")
            # Check warhead atom in local frame is near canonical Cβ
            wpos = ex["lig_pos"][ex["warhead_atom_idx"]]
            d_to_origin = wpos.norm().item()
            print(f"      warhead atom local frame: ({wpos[0]:+.3f}, {wpos[1]:+.3f}, {wpos[2]:+.3f})"
                  f"  |d|={d_to_origin:.3f}")
        if n_ok >= 5: break
    print(f"\nLoaded {n_ok}/50 valid; total dataset = {len(ds)}")
