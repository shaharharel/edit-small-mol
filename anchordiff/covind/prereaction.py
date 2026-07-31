"""Pre-reaction structure derivation — C+D v3 (E3 in experiment queue).

CovInDB 2.0 stores **post-bond covalent adducts** (Cys-S–C single bond ≈ 1.85 Å,
sp³ at the attacked carbon, original Michael acceptor C=C gone). The kinetics
that determine k_inact / K_i happen at the **pre-reaction (Michaelis) complex**:
warhead C=C still intact, ligand sits at van-der-Waals distance ~3.5 Å from Sγ.

This module reverses the adduct geometry per mechanism class to produce
prereactive training targets that match the actual rate-limiting step.

Supported reverse transforms (clean reverses possible):
  - **Addition**       Michael Acceptor, Vinyl Sulfone, Diazomethyl Carbonyl
                       — break S–Cβ, restore Cβ=Cα, translate outward 2 Å
  - **Hemithioacetal** Aldehyde, Carbonyl, Aldehydic carbonyl
                       — break S–C, restore C=O (and remove the new C-O-H),
                         translate outward 2 Å
  - **Thioimidate**    Nitrile
                       — break S–C, restore C≡N (the C=N → C≡N reverse),
                         translate outward 2 Å

NOT supported (leaving group is lost in adduct; cannot reconstruct):
  - SN2: Halohydrocarbon, Sulfonyl Fluoride, Aziridine
  - Disulfide: leaving thiol cannot be reconstructed
  - Sulfonylation, transesterification, phosphonate

Status: STUB — algorithm + data structures defined; concrete RDKit-mol
manipulation per mechanism is staged below as `_reverse_*` functions.
Each carries a short reference for the geometric transform.

Usage:
    from anchordiff.covind.prereaction import derive_prereactive
    pre_mol, pre_sg_xyz, pre_anchor_xyz, status = derive_prereactive(
        post_mol, sg_xyz, anchor_atom_idx, warhead_class, mechanism,
    )
    if status == "ok":
        # use pre_mol with new geometry for training
        ...
"""
from __future__ import annotations
from dataclasses import dataclass
from typing import Optional
import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem, BondType
from rdkit import RDLogger
RDLogger.DisableLog("rdApp.*")


# ── Configuration ────────────────────────────────────────────────────────────

# Translation distance to apply along Sγ→Cβ direction to go from post-bond
# (1.85 Å) to prereactive (~3.5 Å van der Waals approach).
PREREACTIVE_TRANSLATE_DELTA = 1.7  # Å — gives final d(Sγ-Cβ) ≈ 3.55 Å

# Bond length for restored double/triple bonds (RDKit conformer regeneration
# will handle this in the embed step; we set the target here for reference).
TARGET_BOND_LENGTHS = {
    "C=C": 1.34,
    "C=O": 1.21,
    "C#N": 1.16,
}


# Map mechanism → reverse-transform handler. The actual functions are stubs
# below; this dispatch table is the public-facing API.
MECHANISM_TO_REVERSER = {
    # Addition (C=C restored)
    "Michael Addition":                "reverse_michael",
    "Nucleophilic Addition":           "reverse_michael",  # close enough
    # Hemithioacetal (C=O restored)
    "hemithioacetal":                  "reverse_hemithioacetal",  # legacy lowercase
    # Thioimidate (C≡N restored)
    "thioimidate":                     "reverse_thioimidate",
}

# Warhead class → expected mechanism (when CovInDB2's Reaction column is null
# or generic). Used as a fallback so curation isn't lossy.
WARHEAD_TO_MECHANISM_FALLBACK = {
    "Michael Acceptor":      "Michael Addition",
    "Vinyl Sulfone":         "Michael Addition",
    "Vinylsulfone":          "Michael Addition",
    "Diazomethyl Carbonyl":  "Michael Addition",
    "Aldehyde":              "hemithioacetal",
    "Aldehydic carbonyl":    "hemithioacetal",
    "Carbonyl":              "hemithioacetal",
    "Nitrile":               "thioimidate",
}

# Mechanisms we explicitly skip (cannot reconstruct geometry).
UNSUPPORTED_MECHANISMS = {
    "Nucleophilic Substitution",   # SN2 — leaving group lost
    "Beta Lactam Addition",         # ring closure — geometrically nontrivial
    "Boronic Acid Addition",        # not enough records to bother yet
    "Epoxide Opening",              # ring closure — geometrically nontrivial
    "Imine Condensation",           # rare in CovInDB2
    "Disulfide Formation",          # leaving thiol lost
    "Ring-opening reaction",        # ambiguous
    "Sulfonylation",                # leaving group lost
    "Phosphonate Substitution",     # leaving group lost
    "OTHER",                        # default catch-all
}


# ── Output type ──────────────────────────────────────────────────────────────

@dataclass
class PrereactiveResult:
    """Carries the prereactive geometry + provenance.

    mol: RDKit mol with restored pre-reaction bond pattern (no Sγ-C covalent
         bond; warhead's original unsaturation restored)
    sg_xyz: (3,) — unchanged Cys Sγ position (same as input)
    anchor_xyz: (3,) — new position of the attacked carbon in prereactive
                geometry (sits ~3.5 Å from Sγ along the original approach
                direction)
    status: "ok" | "skipped:<reason>" | "error:<reason>"
    reverser: name of the reverse-transform applied (or None)
    """
    mol: Optional[Chem.Mol]
    sg_xyz: Optional[np.ndarray]
    anchor_xyz: Optional[np.ndarray]
    status: str
    reverser: Optional[str] = None


# ── Public API ───────────────────────────────────────────────────────────────

def derive_prereactive(
    post_mol: Chem.Mol,
    sg_xyz: np.ndarray,
    anchor_atom_idx: int,
    warhead_class: str,
    mechanism: Optional[str] = None,
) -> PrereactiveResult:
    """Reverse a post-bond covalent adduct to its prereactive Michaelis complex.

    Args:
        post_mol: RDKit mol of the LIGAND ONLY (the protein Cys is encoded
                  separately via sg_xyz). Should have 3D coords (conformer 0).
        sg_xyz: (3,) Cys Sγ position in the same frame as post_mol coordinates
        anchor_atom_idx: 0-based atom index in post_mol of the carbon that
                         was attacked (Cβ for Michael, C(=O) for aldehyde,
                         C(#N) for nitrile, etc.). Same value our `curate.py`
                         emits per record.
        warhead_class: one of WARHEAD_VOCAB strings
        mechanism: one of MECHANISM_VOCAB strings; if None or "OTHER",
                   we fall back to WARHEAD_TO_MECHANISM_FALLBACK.

    Returns:
        PrereactiveResult — see dataclass.
    """
    mech = mechanism or WARHEAD_TO_MECHANISM_FALLBACK.get(warhead_class)

    # Skip explicitly unsupported mechanisms (leaving group lost, etc.)
    if mech is None or mech in UNSUPPORTED_MECHANISMS:
        return PrereactiveResult(
            mol=None, sg_xyz=None, anchor_xyz=None,
            status=f"skipped:unsupported_mechanism({mech!r}, warhead={warhead_class!r})",
        )

    reverser_name = MECHANISM_TO_REVERSER.get(mech)
    if reverser_name is None:
        return PrereactiveResult(
            mol=None, sg_xyz=None, anchor_xyz=None,
            status=f"skipped:no_reverser_for_mechanism({mech!r})",
        )

    # Dispatch to the per-mechanism reverse transform
    try:
        if reverser_name == "reverse_michael":
            mol_pre, anchor_pre = _reverse_michael(post_mol, sg_xyz, anchor_atom_idx)
        elif reverser_name == "reverse_hemithioacetal":
            mol_pre, anchor_pre = _reverse_hemithioacetal(post_mol, sg_xyz, anchor_atom_idx)
        elif reverser_name == "reverse_thioimidate":
            mol_pre, anchor_pre = _reverse_thioimidate(post_mol, sg_xyz, anchor_atom_idx)
        else:
            return PrereactiveResult(
                mol=None, sg_xyz=None, anchor_xyz=None,
                status=f"error:unknown_reverser({reverser_name!r})",
            )
    except Exception as e:
        return PrereactiveResult(
            mol=None, sg_xyz=None, anchor_xyz=None,
            status=f"error:{type(e).__name__}:{str(e)[:80]}",
        )

    if mol_pre is None:
        return PrereactiveResult(
            mol=None, sg_xyz=None, anchor_xyz=None,
            status=f"error:reverser_returned_none({reverser_name!r})",
            reverser=reverser_name,
        )

    return PrereactiveResult(
        mol=mol_pre, sg_xyz=np.asarray(sg_xyz, dtype=float),
        anchor_xyz=anchor_pre, status="ok", reverser=reverser_name,
    )


# ── Per-mechanism reverse transforms ─────────────────────────────────────────
# Each function takes (post_mol, sg_xyz, anchor_atom_idx) and returns
# (prereactive_mol, new_anchor_xyz). The post_mol's coordinates and bond
# structure are NOT mutated — a fresh RWMol is built and returned.

def _reverse_michael(
    post_mol: Chem.Mol, sg_xyz: np.ndarray, anchor_atom_idx: int,
) -> tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
    """Reverse a Michael adduct.

    Pre-state (in CovInDB2):  Cys-S–Cβ–Cα(–EWG)  with single bonds, sp³ Cβ
    Post-state we restore:    Cys-S ⋅⋅⋅ Cβ=Cα(–EWG)  with C=C double, sp² Cβ

    Steps:
      1. Find the C-C bond between anchor (Cβ) and its sp³ neighbor that's
         most likely the original Cα — heuristic: the carbon neighbor that
         is itself bonded to an EWG (carbonyl C=O, SO2, CN, etc.).
      2. Change that C-C bond from SINGLE to DOUBLE.
      3. Reduce explicit-H count on both atoms by 1 (sp³→sp²).
      4. Translate all ligand atoms outward along (Cβ - Sγ) direction by
         PREREACTIVE_TRANSLATE_DELTA Å. New Sγ-Cβ distance ≈ 3.55 Å.

    Returns (prereactive_mol, new_anchor_xyz). new_anchor_xyz is the new
    position of the anchor atom (Cβ) in the prereactive frame.
    """
    rw = Chem.RWMol(post_mol)
    anchor = rw.GetAtomWithIdx(anchor_atom_idx)
    if anchor.GetSymbol() != "C":
        return None, None

    # Find Cα candidate: a C neighbor of Cβ that's bonded to an EWG
    cβ_neighbors = [n for n in anchor.GetNeighbors() if n.GetSymbol() == "C"]
    cα_idx = _find_cα_for_michael(rw, anchor_atom_idx, cβ_neighbors)
    if cα_idx is None:
        return None, None

    # Restore C=C double bond
    bond = rw.GetBondBetweenAtoms(anchor_atom_idx, cα_idx)
    if bond is None or bond.GetBondType() != BondType.SINGLE:
        return None, None
    bond.SetBondType(BondType.DOUBLE)

    # sp³→sp² on both Cβ and Cα: drop one H each
    _drop_one_H(rw, anchor_atom_idx)
    _drop_one_H(rw, cα_idx)

    pre_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(pre_mol)
    except Chem.MolSanitizeException:
        return None, None

    # Translate coords outward along (Cβ - Sγ)
    new_anchor_xyz = _translate_outward(pre_mol, anchor_atom_idx, sg_xyz)
    return pre_mol, new_anchor_xyz


def _reverse_hemithioacetal(
    post_mol: Chem.Mol, sg_xyz: np.ndarray, anchor_atom_idx: int,
) -> tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
    """Reverse a hemithioacetal adduct (aldehyde/ketone + Cys).

    Pre-state: Cys-S–C(OH)–R  with C sp³ (gem-diol-like center)
    Post we restore: Cys-S ⋅⋅⋅ C(=O)–R  with C sp², C=O double bond,
                     drop the C–O–H proton (now a free aldehyde/ketone)

    Steps:
      1. Find the O neighbor of anchor (carbonyl carbon)
      2. Convert C-O single bond → C=O double bond
      3. Drop the explicit H on the O (was the hydroxyl H)
      4. Drop one H on the C (sp³→sp²)
      5. Translate ligand outward
    """
    rw = Chem.RWMol(post_mol)
    anchor = rw.GetAtomWithIdx(anchor_atom_idx)
    if anchor.GetSymbol() != "C":
        return None, None

    # Find the O neighbor that's currently single-bonded with an OH
    o_idx = None
    for n in anchor.GetNeighbors():
        if n.GetSymbol() == "O":
            bond = rw.GetBondBetweenAtoms(anchor_atom_idx, n.GetIdx())
            if bond.GetBondType() == BondType.SINGLE:
                o_idx = n.GetIdx()
                break
    if o_idx is None:
        return None, None

    # C-O single → C=O double
    bond = rw.GetBondBetweenAtoms(anchor_atom_idx, o_idx)
    bond.SetBondType(BondType.DOUBLE)

    # Drop the OH proton from O, and one H from anchor (sp³→sp²)
    _drop_one_H(rw, o_idx)
    _drop_one_H(rw, anchor_atom_idx)

    pre_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(pre_mol)
    except Chem.MolSanitizeException:
        return None, None

    new_anchor_xyz = _translate_outward(pre_mol, anchor_atom_idx, sg_xyz)
    return pre_mol, new_anchor_xyz


def _reverse_thioimidate(
    post_mol: Chem.Mol, sg_xyz: np.ndarray, anchor_atom_idx: int,
) -> tuple[Optional[Chem.Mol], Optional[np.ndarray]]:
    """Reverse a thioimidate adduct (nitrile + Cys).

    Pre-state: Cys-S–C(=N–H)–R   with C sp², C=N double bond
    Post we restore: Cys-S ⋅⋅⋅ C(#N)–R  with C sp, C≡N triple bond,
                     drop the imine proton

    Steps:
      1. Find the N neighbor of anchor (imine nitrogen)
      2. Convert C=N → C≡N
      3. Drop the N-H proton
      4. Adjust H counts as needed
      5. Translate ligand outward
    """
    rw = Chem.RWMol(post_mol)
    anchor = rw.GetAtomWithIdx(anchor_atom_idx)
    if anchor.GetSymbol() != "C":
        return None, None

    # Find the N neighbor with C=N double bond
    n_idx = None
    for nbr in anchor.GetNeighbors():
        if nbr.GetSymbol() == "N":
            bond = rw.GetBondBetweenAtoms(anchor_atom_idx, nbr.GetIdx())
            if bond.GetBondType() in (BondType.DOUBLE, BondType.SINGLE):
                n_idx = nbr.GetIdx()
                break
    if n_idx is None:
        return None, None

    # Promote to triple bond
    bond = rw.GetBondBetweenAtoms(anchor_atom_idx, n_idx)
    bond.SetBondType(BondType.TRIPLE)

    # Drop the N-H (imine proton)
    _drop_one_H(rw, n_idx)

    pre_mol = rw.GetMol()
    try:
        Chem.SanitizeMol(pre_mol)
    except Chem.MolSanitizeException:
        return None, None

    new_anchor_xyz = _translate_outward(pre_mol, anchor_atom_idx, sg_xyz)
    return pre_mol, new_anchor_xyz


# ── Helpers ──────────────────────────────────────────────────────────────────

def _drop_one_H(rw: Chem.RWMol, heavy_idx: int) -> bool:
    """Drop one H from a heavy atom. Handles both implicit (NumExplicitHs)
    and explicit (separate H atom) representations. Returns True if a H was
    dropped, False if none was available.

    Why: post-bond mols sometimes come in with implicit Hs (no AddHs), and
    sometimes with explicit Hs (after AddHs). The reverser needs to handle both.
    """
    a = rw.GetAtomWithIdx(heavy_idx)
    # First: try explicit-H attribute
    h = a.GetNumExplicitHs()
    if h > 0:
        a.SetNumExplicitHs(h - 1)
        return True
    # Second: try removing a bonded H atom
    for nbr in a.GetNeighbors():
        if nbr.GetSymbol() == "H":
            rw.RemoveAtom(nbr.GetIdx())
            return True
    # No H available → caller decides if this is a problem
    return False


def _find_cα_for_michael(
    rw: Chem.RWMol, cβ_idx: int, cβ_neighbors: list[Chem.Atom],
) -> Optional[int]:
    """Pick the C-neighbor of Cβ that's most plausibly Cα — the one bonded
    to an EWG (carbonyl, sulfonyl, nitrile, nitro)."""
    EWG_PATTERNS = [
        # SMARTS that, when matched on a C neighbor of Cβ, indicates EWG
        ("[CX3]=[OX1]", "carbonyl"),         # C=O (amide, ester, ketone)
        ("[SX4](=O)=O", "sulfonyl"),         # SO2
        ("[CX2]#[NX1]", "nitrile"),          # C≡N
        ("[NX3+](=O)[O-]", "nitro"),         # NO2
    ]
    pre_mol = rw.GetMol()
    best_idx = None
    best_score = -1
    for n in cβ_neighbors:
        if n.GetIdx() == cβ_idx:
            continue
        # Score: 1 point per EWG attached to this neighbor
        score = 0
        for smarts, _label in EWG_PATTERNS:
            patt = Chem.MolFromSmarts(smarts)
            if patt is None: continue
            for match in pre_mol.GetSubstructMatches(patt):
                if n.GetIdx() in match:
                    score += 1
                    break
        if score > best_score:
            best_score = score
            best_idx = n.GetIdx()
    return best_idx if best_score > 0 else (cβ_neighbors[0].GetIdx() if cβ_neighbors else None)


def _translate_outward(
    pre_mol: Chem.Mol, anchor_atom_idx: int, sg_xyz: np.ndarray,
) -> np.ndarray:
    """Translate ALL atoms of pre_mol by PREREACTIVE_TRANSLATE_DELTA Å along
    the (anchor - Sγ) direction. Returns the new anchor position.

    The molecule's internal geometry (bond lengths, angles) is preserved —
    we're doing a rigid translation, not a re-embedding. The C=C / C=O / C#N
    bonds will end up at the wrong bond length (since we just promoted them
    without re-optimizing), but this is left as a downstream cleanup
    (RDKit MMFFOptimizeMolecule on the prereactive copy).
    """
    if pre_mol.GetNumConformers() == 0:
        return None
    conf = pre_mol.GetConformer()
    anchor_xyz = np.array(conf.GetAtomPosition(anchor_atom_idx))
    direction = anchor_xyz - np.asarray(sg_xyz, dtype=float)
    norm = float(np.linalg.norm(direction))
    if norm < 1e-6:
        return None
    direction = direction / norm
    shift = direction * PREREACTIVE_TRANSLATE_DELTA
    for i in range(pre_mol.GetNumAtoms()):
        p = np.array(conf.GetAtomPosition(i))
        new_p = p + shift
        conf.SetAtomPosition(i, new_p.tolist())
    new_anchor_xyz = np.array(conf.GetAtomPosition(anchor_atom_idx))
    return new_anchor_xyz


# ── Smoke test ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    # Smoke test on a synthetic Michael adduct:
    #   Cys-S–CH2–CH2–C(=O)–NH2  (post-bond acrylamide+Cys)
    # Expected pre-reaction:
    #   Cys-S ⋅⋅⋅ CH2=CH–C(=O)–NH2  (acrylamide intact, separated 3.5 Å)
    smi_post = "SCCC(=O)N"
    mol = Chem.MolFromSmiles(smi_post)
    mol = Chem.AddHs(mol)
    # Embed 3D
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    # Pretend S is at fixed position; anchor is the CH2 bonded to S
    sg_xyz = np.array(mol.GetConformer().GetAtomPosition(0))
    anchor_idx = 1  # the C bonded to S
    print(f"=== Smoke test: Michael reverse ===")
    print(f"Post-bond SMILES: {smi_post}")
    print(f"Sγ at: {sg_xyz}")
    print(f"Anchor (Cβ) at: {np.array(mol.GetConformer().GetAtomPosition(anchor_idx))}")
    print(f"d(Sγ-Cβ) before: {np.linalg.norm(np.array(mol.GetConformer().GetAtomPosition(anchor_idx)) - sg_xyz):.3f} Å")

    result = derive_prereactive(
        mol, sg_xyz, anchor_idx,
        warhead_class="Michael Acceptor",
        mechanism="Michael Addition",
    )
    print(f"\nResult status: {result.status}")
    print(f"Reverser: {result.reverser}")
    if result.mol is not None:
        smi_pre = Chem.MolToSmiles(Chem.RemoveHs(result.mol))
        print(f"Pre-reaction SMILES: {smi_pre}")
        print(f"New anchor at: {result.anchor_xyz}")
        d_new = np.linalg.norm(result.anchor_xyz - result.sg_xyz)
        print(f"d(Sγ-Cβ) after: {d_new:.3f} Å  (expected ≈ {1.85 + PREREACTIVE_TRANSLATE_DELTA:.2f})")

    print("\n=== Smoke test: Aldehyde reverse (hemithioacetal) ===")
    smi_post = "SC(O)c1ccccc1"  # PhCH(OH)-S-Cys (hemithioacetal post-bond)
    mol = Chem.MolFromSmiles(smi_post)
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol, randomSeed=42)
    AllChem.MMFFOptimizeMolecule(mol)
    sg_xyz = np.array(mol.GetConformer().GetAtomPosition(0))
    anchor_idx = 1
    result = derive_prereactive(
        mol, sg_xyz, anchor_idx,
        warhead_class="Aldehyde", mechanism="hemithioacetal",
    )
    print(f"Status: {result.status}")
    if result.mol is not None:
        print(f"Pre-reaction: {Chem.MolToSmiles(Chem.RemoveHs(result.mol))}")
        print(f"d after: {np.linalg.norm(result.anchor_xyz - result.sg_xyz):.3f} Å")

    print("\n=== Smoke test: Unsupported (SN2) — should skip ===")
    mol = Chem.MolFromSmiles("SCCl")  # Cys-S-CH2-Cl pretend post-bond is C-S
    mol = Chem.AddHs(mol)
    AllChem.EmbedMolecule(mol)
    result = derive_prereactive(
        mol, np.zeros(3), 1,
        warhead_class="Halohydrocarbon", mechanism="Nucleophilic Substitution",
    )
    print(f"Status: {result.status}  (expected: skipped:unsupported_mechanism)")
