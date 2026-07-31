"""290-d covalent conditioning token + small adapter.

Ported in spirit from anchordiff/covind/{covalent_token_v2_5.py, cov_adapter_v2_5.py}.

Token layout (TOKEN_DIM = 290):
  [0]            log10(d_canonical)            -- SG-to-anchor target distance
  [1]            theta_canonical / 180         -- canonical bond angle
  [2:258]        warhead Morgan FP (256 bits, r=2)
  [258:278]      pocket residue counts (20)
  [278:290]      reaction mechanism one-hot (12)

For PocketFlow integration we do NOT inject into pocket one-hot directly
(PocketFlow's protein feature dim differs from DiffSBDD's). Instead the
adapter outputs a bias that is added to the encoder's ligand-atom embedding
input: this is a scalar feature so SE(3) equivariance is preserved.
"""
from __future__ import annotations
import numpy as np
import torch
import torch.nn as nn
from rdkit import Chem
from rdkit.Chem import AllChem
from rdkit import RDLogger

RDLogger.DisableLog("rdApp.*")

# ---- vocab (subset of anchordiff/covind) ----
RESIDUE_VOCAB = [
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLN", "GLU", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
]
R_DIM = len(RESIDUE_VOCAB)  # 20

MECHANISM_VOCAB = [
    "Michael Addition", "Nucleophilic Substitution", "Acylation",
    "Sulfonylation", "Schiff Base", "Disulfide Exchange",
    "Hemithioacetal", "Cyanation", "Ring Opening", "Reverse Michael",
    "Addition", "OTHER",
]
M_DIM = len(MECHANISM_VOCAB)  # 12

WARHEAD_CANONICAL_SMILES = {
    "Michael Acceptor":      "C=CC(=O)N",
    "Halohydrocarbon":       "ClCC(=O)N",
    "Vinyl Sulfone":         "C=CS(=O)(=O)C",
    "Vinylsulfone":          "C=CS(=O)(=O)C",
    "Beta Lactam":           "O=C1CCN1",
    "Epoxide":               "C1CO1",
    "Disulfide":             "CSSC",
    "Aldehyde":              "C=O",
    "Aldehydic carbonyl":    "C=O",
    "Carbonyl":              "C=O",
    "Nitrile":               "C#N",
    "Sulfonyl Fluorine":     "FS(=O)(=O)C",
    "Sulfonic acid":         "OS(=O)(=O)C",
    "Thiol":                 "SC",
    "Ester":                 "O=C(O)C",
    "Diazomethyl Carbonyl":  "[N+]=[N-]C(=O)C",
    "Thiosulfonate":         "S(=O)(=O)SC",
    "Aziridine":             "C1CN1",
}
WARHEAD_FP_BITS = 256
TOKEN_DIM = 2 + WARHEAD_FP_BITS + R_DIM + M_DIM  # 290


def _morgan_fp(smi: str, n_bits: int = WARHEAD_FP_BITS, radius: int = 2) -> np.ndarray:
    mol = Chem.MolFromSmiles(smi)
    if mol is None:
        return np.zeros(n_bits, dtype=np.float32)
    bv = AllChem.GetMorganFingerprintAsBitVect(mol, radius, nBits=n_bits)
    arr = np.zeros(n_bits, dtype=np.float32)
    from rdkit.DataStructs import ConvertToNumpyArray
    ConvertToNumpyArray(bv, arr)
    return arr


_FP_TABLE = {wc: _morgan_fp(smi) for wc, smi in WARHEAD_CANONICAL_SMILES.items()}


def build_token(
    d_canonical: float = 1.85,
    theta_canonical: float = 107.0,
    warhead_class: str = "Michael Acceptor",
    pocket_residues: list[str] | None = None,
    reaction_mechanism: str = "Michael Addition",
) -> np.ndarray:
    out = np.zeros(TOKEN_DIM, dtype=np.float32)
    out[0] = float(np.log10(max(d_canonical, 1e-3)))
    out[1] = float(theta_canonical) / 180.0
    fp = _FP_TABLE.get(warhead_class)
    if fp is not None:
        out[2:2 + WARHEAD_FP_BITS] = fp
    rstart = 2 + WARHEAD_FP_BITS
    for resn in (pocket_residues or []):
        resn3 = resn.upper().strip()
        if resn3 in RESIDUE_VOCAB:
            out[rstart + RESIDUE_VOCAB.index(resn3)] += 1.0
    mstart = rstart + R_DIM
    if reaction_mechanism in MECHANISM_VOCAB:
        out[mstart + MECHANISM_VOCAB.index(reaction_mechanism)] = 1.0
    else:
        out[mstart + MECHANISM_VOCAB.index("OTHER")] = 1.0
    return out


class CovTokenAdapter(nn.Module):
    """Maps the 290-d cov token to a per-feature bias added to ligand atom embed input.

    Args:
        feat_dim: width of the ligand atom feature vector (PocketFlow uses
                  ligand_atom_feature_dim from config; usually 15 for the ZINC
                  ckpt). We add a learnable bias to the FIRST atom (the seed
                  warhead Cβ) at every forward — kept SE(3)-safe because bias
                  modifies scalar features only.
    """

    def __init__(self, feat_dim: int, token_dim: int = TOKEN_DIM,
                 hidden_dim: int = 64, init_scale: float = 0.5,
                 token_dropout_p: float = 0.1):
        super().__init__()
        self.token_dim = token_dim
        self.feat_dim = feat_dim
        self.token_dropout_p = token_dropout_p
        self.mlp = nn.Sequential(
            nn.Linear(token_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feat_dim),
        )
        # log_scale init=log(init_scale) — exp keeps scale positive
        self.log_scale = nn.Parameter(
            torch.tensor(float(np.log(max(init_scale, 1e-3))))
        )
        for m in self.mlp:
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.01)
                nn.init.zeros_(m.bias)

    @property
    def scale(self) -> torch.Tensor:
        return torch.exp(self.log_scale)

    def forward(self, token: torch.Tensor) -> torch.Tensor:
        if self.training and self.token_dropout_p > 0:
            mask_shape = (1,) if token.dim() == 1 else (token.shape[0], 1)
            keep = torch.bernoulli(
                torch.full(mask_shape, 1.0 - self.token_dropout_p, device=token.device)
            )
            token = token * keep / (1.0 - self.token_dropout_p)
        h = self.mlp(token)
        return self.scale * torch.tanh(h)


if __name__ == "__main__":
    tok = build_token(
        1.85, 107.0, "Michael Acceptor",
        ["LEU", "GLY", "CYS", "GLY", "ASN", "PHE", "GLY"],
        "Michael Addition",
    )
    print(f"TOKEN_DIM = {TOKEN_DIM}")
    print(f"  d log10  = {tok[0]:.3f}")
    print(f"  theta/180= {tok[1]:.3f}")
    print(f"  FP nonzero bits = {int(tok[2:2+WARHEAD_FP_BITS].sum())}")
    print(f"  mech idx = {int(np.argmax(tok[2+WARHEAD_FP_BITS+R_DIM:]))}")

    a = CovTokenAdapter(feat_dim=15)
    print(f"  adapter params = {sum(p.numel() for p in a.parameters())}")
    a.eval()
    out = a(torch.from_numpy(tok))
    print(f"  bias shape={tuple(out.shape)}  |mean|={out.abs().mean().item():.4f}")
