"""C: covalent token conditioning.

Token layout (TOKEN_DIM = 52, updated 2026-05-13):
  [0]            : log10(d_canonical)
  [1]            : theta_canonical / 180
  [2..2+W]       : warhead-class one-hot                (W = 18, expanded from 12)
  [2+W .. +R]    : Cys-context pocket-residue one-hot   (R = 20)
  [2+W+R .. +M]  : reaction-mechanism one-hot           (M = 12, NEW)

What changed:
- Dropped the 3 always-zero SG-position dims (waste).
- Expanded WARHEAD_VOCAB from 12 → 18: added Sulfonic acid, Thiol, Ester,
  Diazomethyl Carbonyl, Thiosulfonate, Aziridine. These cover ~230 CYS-
  targeting entries we were silently dropping (CovInDB2 QA, 2026-05-13).
- Added MECHANISM_VOCAB derived from CovInDB2 `Reaction` column. Two
  Michael Acceptors with different substituents can have different reaction
  subtypes (e.g., classic Michael vs aza-Michael) — class-only conditioning
  conflates them.
"""
from __future__ import annotations
import numpy as np

# Warhead-class vocabulary — keep order stable; new classes appended at end
# so existing one-hot indices stay valid (matters only for old ckpts; not for
# a fresh train).
WARHEAD_VOCAB = [
    "Michael Acceptor", "Halohydrocarbon", "Vinyl Sulfone", "Vinylsulfone",
    "Beta Lactam", "Epoxide", "Disulfide", "Aldehyde", "Aldehydic carbonyl",
    "Carbonyl", "Nitrile", "Sulfonyl Fluorine",
    # Added 2026-05-13 from CovInDB2 audit
    "Sulfonic acid", "Thiol", "Ester", "Diazomethyl Carbonyl",
    "Thiosulfonate", "Aziridine",
]
W_DIM = len(WARHEAD_VOCAB)  # 18

# Per-row reaction mechanism (from CovInDB2 `Reaction` column). The 12 most
# frequent values cover >97% of CYS-targeting entries; everything else maps
# to "OTHER". Same canonical order across train and inference.
MECHANISM_VOCAB = [
    "Nucleophilic Addition", "Nucleophilic Substitution", "Michael Addition",
    "Beta Lactam Addition", "Boronic Acid Addition", "Epoxide Opening",
    "Imine Condensation", "Disulfide Formation", "Ring-opening reaction",
    "Sulfonylation", "Phosphonate Substitution", "OTHER",
]
M_DIM = len(MECHANISM_VOCAB)  # 12

RESIDUE_VOCAB = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
R_DIM = len(RESIDUE_VOCAB)  # 20

TOKEN_DIM = 2 + W_DIM + R_DIM + M_DIM  # 2 + 18 + 20 + 12 = 52


def build_token(d_canonical: float, theta_canonical: float,
                warhead_class: str, cys_context_residues: list[str],
                reaction_mechanism: str | None = None) -> np.ndarray:
    """Build the (TOKEN_DIM,)-dim covalent conditioning token.

    Args:
      d_canonical: canonical SG-anchor bond length (Å)
      theta_canonical: canonical bond angle (degrees)
      warhead_class: one of WARHEAD_VOCAB; unknown → all-zero one-hot
      cys_context_residues: list of 3-letter residue names within pocket cutoff
      reaction_mechanism: one of MECHANISM_VOCAB (CovInDB2 `Reaction` value);
        None or unknown → OTHER bit set
    """
    out = np.zeros(TOKEN_DIM, dtype=np.float32)
    out[0] = float(np.log10(max(d_canonical, 1e-3)))
    out[1] = float(theta_canonical) / 180.0
    if warhead_class in WARHEAD_VOCAB:
        out[2 + WARHEAD_VOCAB.index(warhead_class)] = 1.0
    rstart = 2 + W_DIM
    for resn in cys_context_residues:
        resn3 = resn.upper().strip()
        if resn3 in RESIDUE_VOCAB:
            out[rstart + RESIDUE_VOCAB.index(resn3)] += 1.0
    mstart = 2 + W_DIM + R_DIM
    if reaction_mechanism in MECHANISM_VOCAB:
        out[mstart + MECHANISM_VOCAB.index(reaction_mechanism)] = 1.0
    else:
        out[mstart + MECHANISM_VOCAB.index("OTHER")] = 1.0
    return out


if __name__ == "__main__":
    t = build_token(1.85, 107.0, "Michael Acceptor",
                    ["LEU", "GLY", "CYS", "GLY", "ASN", "PHE", "GLY"],
                    reaction_mechanism="Michael Addition")
    print(f"TOKEN_DIM = {TOKEN_DIM}  (W={W_DIM} R={R_DIM} M={M_DIM})")
    print(f"  d log10:   {t[0]:.3f}")
    print(f"  theta/180: {t[1]:.3f}")
    wi = int(np.argmax(t[2:2 + W_DIM]))
    print(f"  warhead:   idx={wi} → {WARHEAD_VOCAB[wi]}")
    mi = int(np.argmax(t[2 + W_DIM + R_DIM:]))
    print(f"  mechanism: idx={mi} → {MECHANISM_VOCAB[mi]}")
    print(f"  residues:  " + " ".join(
        f"{RESIDUE_VOCAB[i]}={int(t[2 + W_DIM + i])}"
        for i in range(R_DIM) if t[2 + W_DIM + i] > 0))
