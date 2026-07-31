"""C-arm v2.5: covalent-token conditioning with warhead Morgan FP.

What's different from v2 (covalent_token.py):
  - Replaces 18-d warhead one-hot with a 256-d Morgan FP of the warhead's
    canonical SMILES. Captures bond patterns / ring topology that class
    collapse hides (e.g., acrylamide vs vinyl sulfone both share C=C → EWG;
    one-hot loses this similarity, Morgan FP preserves it).
  - Keeps 20-d residue counts (will be replaced by ESM-2 pocket emb in v2.6).
  - Keeps 12-d mechanism one-hot.

Token layout (TOKEN_DIM_V2_5 = 290):
  [0]            : log10(d_canonical)
  [1]            : theta_canonical / 180
  [2..258]       : warhead canonical-SMILES Morgan FP (256 bits, r=2)
  [258..278]     : pocket residue counts                (20)
  [278..290]     : reaction-mechanism one-hot            (12)
"""
from __future__ import annotations
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")

from anchordiff.covind.covalent_token import (
    WARHEAD_VOCAB, MECHANISM_VOCAB, RESIDUE_VOCAB,
    R_DIM, M_DIM,
)

# Morgan FP of the canonical CORE SMILES of each warhead class. The model
# gets the chemistry of the warhead (bonds, rings, EWG patterns) instead of
# an arbitrary one-hot index. Same string at train + inference time.
WARHEAD_CANONICAL_SMILES = {
    "Michael Acceptor":       "C=CC(=O)N",          # acrylamide core
    "Halohydrocarbon":        "ClCC(=O)N",          # chloroacetamide
    "Vinyl Sulfone":          "C=CS(=O)(=O)C",      # vinyl sulfone
    "Vinylsulfone":           "C=CS(=O)(=O)C",
    "Beta Lactam":            "O=C1CCN1",           # β-lactam
    "Epoxide":                "C1CO1",              # oxirane
    "Disulfide":              "CSSC",               # disulfide
    "Aldehyde":               "C=O",                # carbonyl C
    "Aldehydic carbonyl":     "C=O",
    "Carbonyl":               "C=O",
    "Nitrile":                "C#N",                # nitrile
    "Sulfonyl Fluorine":      "FS(=O)(=O)C",        # sulfonyl fluoride
    "Sulfonic acid":          "OS(=O)(=O)C",
    "Thiol":                  "SC",                 # thiol (disulfide partner)
    "Ester":                  "O=C(O)C",
    "Diazomethyl Carbonyl":   "[N+]=[N-]C(=O)C",    # diazomethyl ketone
    "Thiosulfonate":          "S(=O)(=O)SC",
    "Aziridine":              "C1CN1",              # aziridine
}

WARHEAD_FP_BITS = 256

TOKEN_DIM_V2_5 = 2 + WARHEAD_FP_BITS + R_DIM + M_DIM  # 2 + 256 + 20 + 12 = 290


def _morgan_fp(smi: str, n_bits: int = WARHEAD_FP_BITS, radius: int = 2) -> np.ndarray:
    """Compute Morgan fingerprint as a 0/1 float vector."""
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr


# Pre-compute Morgan FPs for all warhead classes (cheap, done once at import)
WARHEAD_FP_TABLE: dict[str, np.ndarray] = {
    wc: _morgan_fp(smi) for wc, smi in WARHEAD_CANONICAL_SMILES.items()
}


def build_token_v2_5(
    d_canonical: float,
    theta_canonical: float,
    warhead_class: str,
    cys_context_residues: list[str],
    reaction_mechanism: str | None = None,
) -> np.ndarray:
    """Build the (TOKEN_DIM_V2_5,)-dim covalent conditioning token.

    Args:
      d_canonical: canonical SG-anchor bond length (Å)
      theta_canonical: canonical bond angle (degrees)
      warhead_class: one of WARHEAD_VOCAB; unknown → all-zero FP slice
      cys_context_residues: list of 3-letter residue names within pocket cutoff
      reaction_mechanism: one of MECHANISM_VOCAB; None/unknown → OTHER bit
    """
    out = np.zeros(TOKEN_DIM_V2_5, dtype=np.float32)
    out[0] = float(np.log10(max(d_canonical, 1e-3)))
    out[1] = float(theta_canonical) / 180.0
    # Warhead Morgan FP (256-d)
    fp = WARHEAD_FP_TABLE.get(warhead_class)
    if fp is not None:
        out[2:2 + WARHEAD_FP_BITS] = fp
    # Residue counts (20-d)
    rstart = 2 + WARHEAD_FP_BITS
    for resn in cys_context_residues:
        resn3 = resn.upper().strip()
        if resn3 in RESIDUE_VOCAB:
            out[rstart + RESIDUE_VOCAB.index(resn3)] += 1.0
    # Mechanism (12-d)
    mstart = 2 + WARHEAD_FP_BITS + R_DIM
    if reaction_mechanism in MECHANISM_VOCAB:
        out[mstart + MECHANISM_VOCAB.index(reaction_mechanism)] = 1.0
    else:
        out[mstart + MECHANISM_VOCAB.index("OTHER")] = 1.0
    return out


if __name__ == "__main__":
    t = build_token_v2_5(
        1.85, 107.0, "Michael Acceptor",
        ["LEU", "GLY", "CYS", "GLY", "ASN", "PHE", "GLY"],
        reaction_mechanism="Michael Addition",
    )
    print(f"TOKEN_DIM_V2_5 = {TOKEN_DIM_V2_5}")
    print(f"  d log10:   {t[0]:.3f}")
    print(f"  theta/180: {t[1]:.3f}")
    print(f"  warhead FP nonzero bits: {int(t[2:2+WARHEAD_FP_BITS].sum())}")
    print(f"  residue counts: " + " ".join(
        f"{RESIDUE_VOCAB[i]}={int(t[2+WARHEAD_FP_BITS+i])}"
        for i in range(R_DIM) if t[2+WARHEAD_FP_BITS+i] > 0))
    mi = int(np.argmax(t[2+WARHEAD_FP_BITS+R_DIM:]))
    print(f"  mechanism: idx={mi} → {MECHANISM_VOCAB[mi]}")

    # Sanity: different warhead classes give different FPs
    fps = {wc: _morgan_fp(smi) for wc, smi in list(WARHEAD_CANONICAL_SMILES.items())[:5]}
    print(f"\n  Pairwise Tanimoto similarity (sanity check):")
    keys = list(fps.keys())
    for i, k1 in enumerate(keys):
        for k2 in keys[i+1:]:
            inter = float((fps[k1] * fps[k2]).sum())
            union = float((fps[k1] + fps[k2] - fps[k1]*fps[k2]).sum())
            tc = inter / max(union, 1e-9)
            print(f"    {k1:25s} vs {k2:25s}  Tc={tc:.3f}")
