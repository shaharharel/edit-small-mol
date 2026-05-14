"""Fix warhead bond orders in DiffSBDD inpaint outputs.

DiffSBDD's --fix_atoms preserves atom POSITIONS but its OpenBabel-based bond
perception often flips single/double bond assignments in the warhead, which
breaks the [CH2]=[CH]C(=O)N SMARTS match. The atom POSITIONS are correct.

This script reads inpaint.sdf, finds atoms 0-4 (which are always the fixed
warhead in inpaint outputs), and forces the canonical acrylamide bonds:
  0 = C_β  (CH2)
  1 = C_α  (CH)
  2 = C_carb (carbonyl C)
  3 = O carbonyl
  4 = N amide

Canonical bond pattern:
  0=1 DOUBLE, 1-2 SINGLE, 2=3 DOUBLE, 2-4 SINGLE.

Plus C_β must be CH2 (2 implicit H), C_α must be CH (1 implicit H), N must
have N-H valence to remain neutral, etc.

Writes a parallel <name>_fixed.sdf with the corrected bond orders. After this,
[CH2]=[CH]C(=O)N SMARTS matches and constraint projection / scoring work.
"""
from __future__ import annotations
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rdkit import Chem
from rdkit.Chem import AllChem


def fix_warhead_bonds(mol: Chem.Mol) -> Chem.Mol | None:
    """Return a new mol with warhead bond orders corrected. None if not fixable.

    Strategy:
      1. Verify atoms 0-4 are the canonical warhead element pattern (C, C, C, O, N).
      2. Note all non-warhead atoms that DiffSBDD's bond perceiver attached to
         the warhead-internal atoms {0, 1, 2, 3} — these are SCAFFOLD atoms that
         the perceiver wrongly bonded to {C_β, C_α, C_carb} instead of to N (the
         amide nitrogen, atom 4). Save them as candidates to re-attach to N.
      3. Delete all bonds touching atoms {0, 1, 2, 3}.
      4. Add canonical warhead bonds (0=1, 1-2, 2=3, 2-4).
      5. Re-attach orphan scaffold atoms to N: pick the closest non-warhead atom
         (within 2.5 Å of N) and add a SINGLE bond. This restores the N-scaffold
         bond that perception got wrong.
      6. Sanitize.
    """
    if mol is None:
        return None
    if mol.GetNumAtoms() < 6:
        return None

    elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(5)]
    if elems != ["C", "C", "C", "O", "N"]:
        return None

    rwm = Chem.RWMol(mol)

    # Step 2: collect scaffold atoms that bonded to warhead-internal atoms
    misbonded_scaffold_atoms = set()
    for b in rwm.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in (0, 1, 2, 3) and j >= 5:
            misbonded_scaffold_atoms.add(j)
        elif j in (0, 1, 2, 3) and i >= 5:
            misbonded_scaffold_atoms.add(i)

    # Step 3: remove all bonds involving atoms {0, 1, 2, 3}
    bonds_to_remove = []
    for b in rwm.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        if i in (0, 1, 2, 3) or j in (0, 1, 2, 3):
            bonds_to_remove.append((i, j))
    for i, j in bonds_to_remove:
        if rwm.GetBondBetweenAtoms(i, j) is not None:
            rwm.RemoveBond(i, j)

    # Step 4: add canonical warhead bonds — guard against duplicates because
    # bonds touching atom 4 (N) are NOT removed in step 3, so an existing 2-4
    # amide bond survives and an unconditional AddBond would raise + silently
    # drop the molecule (QA C3 finding 2026-05-10).
    def _add_if_absent(a, b, bt):
        if rwm.GetBondBetweenAtoms(a, b) is None:
            rwm.AddBond(a, b, bt)
        else:
            rwm.GetBondBetweenAtoms(a, b).SetBondType(bt)
    _add_if_absent(0, 1, Chem.BondType.DOUBLE)
    _add_if_absent(1, 2, Chem.BondType.SINGLE)
    _add_if_absent(2, 3, Chem.BondType.DOUBLE)
    _add_if_absent(2, 4, Chem.BondType.SINGLE)

    # Step 5: re-attach scaffold atom to N if N has no scaffold bond
    n_has_scaffold = any(b.GetOtherAtom(rwm.GetAtomWithIdx(4)).GetIdx() >= 5
                         for b in rwm.GetAtomWithIdx(4).GetBonds())
    if not n_has_scaffold:
        # Find closest non-warhead atom to N within 2.5 Å
        conf = rwm.GetConformer()
        n_pos = np.array(list(conf.GetAtomPosition(4)))
        candidates = []
        for k in range(5, rwm.GetNumAtoms()):
            d = np.linalg.norm(np.array(list(conf.GetAtomPosition(k))) - n_pos)
            if d < 2.5:
                candidates.append((d, k))
        candidates.sort()
        # Prefer atoms originally misbonded to warhead-internal carbons (more
        # likely to be the "intended" scaffold attachment point)
        candidates_misbonded = [c for c in candidates if c[1] in misbonded_scaffold_atoms]
        chosen = candidates_misbonded[0] if candidates_misbonded else (candidates[0] if candidates else None)
        if chosen is not None:
            rwm.AddBond(4, chosen[1], Chem.BondType.SINGLE)

    # Reset valence accounting on warhead atoms
    for i in range(5):
        a = rwm.GetAtomWithIdx(i)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)

    new_mol = rwm.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except Exception:
        return None

    # Keep only the fragment containing the warhead (atom 0). Drop dangling
    # fragments that DiffSBDD's bond perception failed to connect.
    frags = Chem.GetMolFrags(new_mol, asMols=False)
    if len(frags) > 1:
        warhead_frag = next((f for f in frags if 0 in f), None)
        if warhead_frag is None or len(warhead_frag) < 6:
            return None
        atoms_to_keep = set(warhead_frag)
        atoms_to_remove = sorted(
            [i for i in range(new_mol.GetNumAtoms()) if i not in atoms_to_keep],
            reverse=True,
        )
        rwm = Chem.RWMol(new_mol)
        for i in atoms_to_remove:
            rwm.RemoveAtom(i)
        new_mol = rwm.GetMol()
        try:
            Chem.SanitizeMol(new_mol)
        except Exception:
            return None

    return new_mol


def fix_sdf(sdf_in: Path, sdf_out: Path) -> dict:
    suppl = Chem.SDMolSupplier(str(sdf_in), sanitize=False)
    writer = Chem.SDWriter(str(sdf_out))
    n_in = n_fixed = n_warhead_match = 0
    smarts = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")
    for mol in suppl:
        if mol is None:
            continue
        n_in += 1
        new_mol = fix_warhead_bonds(mol)
        if new_mol is None:
            continue
        n_fixed += 1
        if new_mol.HasSubstructMatch(smarts):
            n_warhead_match += 1
        writer.write(new_mol)
    writer.close()
    return {"in": n_in, "fixed": n_fixed, "warhead_match": n_warhead_match}


def main():
    base = PROJECT_ROOT / "anchordiff_results" / "day1"
    for tgt in ["zap70_cys346", "btk_cys481"]:
        sdf_in = base / tgt / "inpaint.sdf"
        sdf_out = base / tgt / "inpaint_fixed.sdf"
        if not sdf_in.exists():
            continue
        print(f"=== {tgt} ===")
        stats = fix_sdf(sdf_in, sdf_out)
        print(f"  {stats}")
        print(f"  warhead match rate post-fix: {100*stats['warhead_match']/max(stats['in'],1):.1f}%")


if __name__ == "__main__":
    main()
