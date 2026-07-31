"""Fix warhead bond orders in DiffSBDD inpaint outputs — v6 surgical approach.

History:
  v1-v5: variants of "delete ALL bonds touching atoms {0,1,2,3}, rebuild
    canonical, reattach scaffold to N". This destroyed correctly-bonded
    scaffold-to-N connections that OB got right (~1.20 N-scaffold bonds/mol
    on average), and even when scaffolds were correctly bonded ONLY to N
    they were preserved fine — but for the ZAP70-B2 cohort the bulk of
    spurious bonds were scaffold→Cα/Ccarb/O. Removing those (correctly)
    AND the warhead-internal bonds (wrongly) shattered the scaffold body
    into multiple fragments. The script then kept only the warhead-
    connected fragment → MW collapse 422→196.

  v6 (this version, 2026-05-15 02:30 — QA-driven rewrite):
    SURGICAL approach. Categorize bonds, touch only what needs touching:
    1. Canonical warhead bonds (0=1, 1-2, 2=3, 2-4): set/add to correct type
    2. Other warhead-internal bonds (e.g., 0-3, 1-4): REMOVE (spurious)
    3. Scaffold-to-warhead-non-N bonds (atom in {0,1,2,3} + atom ≥5): REMOVE,
       track scaffold atom for possible re-attachment to N
    4. Scaffold-to-N bonds (atom=4, atom≥5): LEAVE — OB got these right
    5. Pure scaffold bonds: LEAVE
    6. Drop bare warhead-only fragments (no scaffold attached) — handled by
       sanitize + dust-drop

    This preserves the bulk of the scaffold body's internal bonds AND any
    correct N-scaffold attachment, so the scaffold doesn't shatter.

Acrylamide canonical layout (atoms 0-4):
  0 = C_β  (CH2)
  1 = C_α  (CH)
  2 = C_carb (carbonyl C)
  3 = O carbonyl
  4 = N amide

Canonical bonds: 0=1 DOUBLE, 1-2 SINGLE, 2=3 DOUBLE, 2-4 SINGLE.
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem, Descriptors
RDLogger.DisableLog("rdApp.*")


CANONICAL_BONDS = {
    (0, 1): Chem.BondType.DOUBLE,
    (1, 2): Chem.BondType.SINGLE,
    (2, 3): Chem.BondType.DOUBLE,
    (2, 4): Chem.BondType.SINGLE,
}
CANONICAL_ACRYL = Chem.MolFromSmarts("[CH2]=[CH]C(=O)N")

# Minimum fragment size to keep as "real chemistry" (vs OB dust).
MIN_RELEVANT_FRAGMENT_SIZE = 3


def _conf_pos(mol: Chem.Mol, idx: int) -> np.ndarray:
    return np.array(list(mol.GetConformer().GetAtomPosition(idx)))


def _strip_explicit_hs_on_warhead(rwm: Chem.RWMol) -> None:
    """Remove explicit-H atoms bonded to atoms {0..4} so SanitizeMol can
    re-perceive implicit Hs based on the corrected bond pattern."""
    h_to_remove = []
    for warhead_idx in range(5):
        if warhead_idx >= rwm.GetNumAtoms(): break
        for nbr in rwm.GetAtomWithIdx(warhead_idx).GetNeighbors():
            if nbr.GetAtomicNum() == 1:
                h_to_remove.append(nbr.GetIdx())
    for idx in sorted(set(h_to_remove), reverse=True):
        rwm.RemoveAtom(idx)


def fix_warhead_bonds(mol: Chem.Mol, debug_drops: dict | None = None) -> Chem.Mol | None:
    """v6 surgical fix.

    Steps:
      1. Verify atoms 0-4 are canonical warhead elements.
      2. Strip explicit-H atoms bonded to warhead heavies.
      3. Pre-fix shortcut: if mol already matches canonical SMARTS, return.
      4. Categorize bonds:
         - canonical warhead pairs → SET type (correct or already set)
         - other warhead-internal bonds → REMOVE
         - scaffold-to-warhead-non-N bonds → REMOVE (track for reattach)
         - scaffold-to-N bonds → LEAVE
         - pure scaffold bonds → LEAVE
      5. Reset valence on warhead atoms.
      6. If N has no scaffold neighbor, re-attach the nearest tracked
         misbonded scaffold atom to N (no distance cutoff).
      7. Sanitize.
      8. Drop dust fragments (<3 atoms).
    """
    def _bump_drop(reason: str):
        if debug_drops is not None:
            debug_drops[reason] = debug_drops.get(reason, 0) + 1

    if mol is None:
        _bump_drop("input_None"); return None
    if mol.GetNumAtoms() < 6:
        _bump_drop("too_few_atoms"); return None
    if mol.GetNumConformers() == 0:
        _bump_drop("no_conformer"); return None
    elems = [mol.GetAtomWithIdx(i).GetSymbol() for i in range(5)]
    if elems != ["C", "C", "C", "O", "N"]:
        _bump_drop("non_canonical_warhead_elements"); return None

    # Step 3: pre-fix shortcut — if already correct, return unchanged
    if mol.HasSubstructMatch(CANONICAL_ACRYL):
        _bump_drop("already_correct_no_fix_needed")
        m = Chem.Mol(mol)
        try:
            Chem.SanitizeMol(m)
            return m
        except Exception:
            _bump_drop("already_correct_but_sanitize_fail")
            return None

    rwm = Chem.RWMol(mol)
    _strip_explicit_hs_on_warhead(rwm)

    # Step 4: categorize + act per bond
    # KEY CHANGE (v7→v8): scaffold↔warhead-non-N bonds:
    #   - Cb (atom 0) and Cα (atom 1): keep AT MOST 1 scaffold bond at SINGLE.
    #     This allows α/β-substituted acrylamides (real chemistry) while
    #     preventing valence overflow.
    #   - Ccarb (atom 2): already has O+N bonds; adding scaffold → valence 5. DELETE.
    #   - O (atom 3): C=O double bond; scaffold bond makes O 3-coord. DELETE.
    # If multiple scaffold bonds on Cb or Cα, keep the ONE to the geometrically
    # closest scaffold atom (heuristic — could prefer largest fragment).
    misbonded_scaffold_atoms: set[int] = set()
    bonds_to_remove: list[tuple[int, int]] = []
    bonds_to_keep_single: list[tuple[int, int]] = []
    # Collect scaffold bonds per warhead-non-N atom for the "pick 1" logic
    scaffold_bonds_by_warhead = {0: [], 1: [], 2: [], 3: []}

    for b in rwm.GetBonds():
        i, j = b.GetBeginAtomIdx(), b.GetEndAtomIdx()
        pair = tuple(sorted([i, j]))

        # Skip H bonds (handled by step 2 already)
        if rwm.GetAtomWithIdx(i).GetAtomicNum() == 1 or rwm.GetAtomWithIdx(j).GetAtomicNum() == 1:
            continue

        if pair in CANONICAL_BONDS:
            b.SetBondType(CANONICAL_BONDS[pair])
            continue

        # Both ends in warhead but NOT a canonical pair (e.g., 0-3, 0-4, 1-3, 1-4, 3-4)
        if i < 5 and j < 5:
            bonds_to_remove.append(pair)
            continue

        # One end in {0,1,2,3} and the other is scaffold (≥5)
        if (i in (0, 1, 2, 3) and j >= 5) or (j in (0, 1, 2, 3) and i >= 5):
            warhead_idx = i if i < 5 else j
            scaffold_idx = j if i < 5 else i
            scaffold_bonds_by_warhead[warhead_idx].append((scaffold_idx, pair))
            continue

    # Decide what to do with scaffold↔warhead-non-N bonds
    for warhead_idx, bonds in scaffold_bonds_by_warhead.items():
        if not bonds:
            continue
        if warhead_idx in (2, 3):
            # Ccarb / O: DELETE all scaffold bonds (valence overflow)
            for _, pair in bonds:
                bonds_to_remove.append(pair)
            continue
        # Cb / Cα: keep AT MOST 1 scaffold bond at SINGLE (the geometrically closest)
        warhead_pos = _conf_pos(rwm, warhead_idx)
        bonds_with_dist = []
        for scaffold_idx, pair in bonds:
            d = float(np.linalg.norm(_conf_pos(rwm, scaffold_idx) - warhead_pos))
            bonds_with_dist.append((d, scaffold_idx, pair))
        bonds_with_dist.sort()  # ascending distance
        # Keep the closest one, delete the rest
        keep_d, keep_scaffold, keep_pair = bonds_with_dist[0]
        bonds_to_keep_single.append(keep_pair)
        misbonded_scaffold_atoms.add(keep_scaffold)
        for _, scaffold_idx, pair in bonds_with_dist[1:]:
            bonds_to_remove.append(pair)

    # Apply removals
    for i, j in bonds_to_remove:
        if rwm.GetBondBetweenAtoms(i, j) is not None:
            rwm.RemoveBond(i, j)

    # Apply downgrades (the chosen Cb/Cα-scaffold bonds → SINGLE)
    for i, j in bonds_to_keep_single:
        b = rwm.GetBondBetweenAtoms(i, j)
        if b is not None:
            b.SetBondType(Chem.BondType.SINGLE)

    # Step 4b: add canonical bonds that don't yet exist
    for (a, b), bt in CANONICAL_BONDS.items():
        if rwm.GetBondBetweenAtoms(a, b) is None:
            rwm.AddBond(a, b, bt)
        else:
            rwm.GetBondBetweenAtoms(a, b).SetBondType(bt)

    # Step 5: reset valence accounting on warhead atoms
    for i in range(5):
        a = rwm.GetAtomWithIdx(i)
        a.SetNoImplicit(False)
        a.SetNumExplicitHs(0)
        a.SetFormalCharge(0)
        a.SetIsAromatic(False)

    # Step 6: reattach a scaffold atom to N if N has no scaffold neighbor
    n_has_scaffold = any(
        b.GetOtherAtom(rwm.GetAtomWithIdx(4)).GetIdx() >= 5
        for b in rwm.GetAtomWithIdx(4).GetBonds()
    )
    if not n_has_scaffold:
        # Prefer atoms that were misbonded (they were spatially close to a
        # warhead carbon, so they're geometrically reasonable to bond to N).
        n_pos = _conf_pos(rwm, 4)
        cand = []
        candidates_pool = misbonded_scaffold_atoms if misbonded_scaffold_atoms else range(5, rwm.GetNumAtoms())
        for k in candidates_pool:
            d = float(np.linalg.norm(_conf_pos(rwm, k) - n_pos))
            cand.append((d, k))
        cand.sort()
        if cand:
            rwm.AddBond(4, cand[0][1], Chem.BondType.SINGLE)

    # Step 7: sanitize
    new_mol = rwm.GetMol()
    try:
        Chem.SanitizeMol(new_mol)
    except Exception:
        _bump_drop("sanitize_failed_post_fix")
        return None

    # Step 8: drop dust fragments (<3 atoms). Keep warhead-containing frag
    # + any other frag with ≥3 atoms (multi-fragment OK — caller decides).
    # If there's a SINGLE big fragment containing the warhead, perfect.
    frags = Chem.GetMolFrags(new_mol, asMols=False)
    if len(frags) > 1:
        warhead_frag = next((set(f) for f in frags if 0 in f), None)
        if warhead_frag is None:
            _bump_drop("warhead_lost"); return None
        atoms_to_keep = set(warhead_frag)
        for f in frags:
            if 0 in f: continue
            if len(f) >= MIN_RELEVANT_FRAGMENT_SIZE:
                atoms_to_keep.update(f)
        atoms_to_remove = sorted(
            [i for i in range(new_mol.GetNumAtoms()) if i not in atoms_to_keep],
            reverse=True,
        )
        if atoms_to_remove:
            rwm3 = Chem.RWMol(new_mol)
            for i in atoms_to_remove:
                rwm3.RemoveAtom(i)
            new_mol = rwm3.GetMol()
            try:
                Chem.SanitizeMol(new_mol)
            except Exception:
                _bump_drop("sanitize_failed_post_dust_remove")
                return None

    return new_mol


def fix_sdf(sdf_in: Path, sdf_out: Path, debug: bool = False) -> dict:
    suppl = Chem.SDMolSupplier(str(sdf_in), sanitize=False)
    writer = Chem.SDWriter(str(sdf_out))
    n_in = n_fixed = n_warhead_match = n_single_frag = 0
    mw_in_total = 0.0; mw_out_total = 0.0; mw_outs: list[float] = []
    drops: dict[str, int] = {}
    for mol in suppl:
        if mol is None:
            drops["sd_supplier_none"] = drops.get("sd_supplier_none", 0) + 1
            continue
        n_in += 1
        try: mw_in_total += Descriptors.MolWt(mol)
        except Exception: pass
        new_mol = fix_warhead_bonds(mol, debug_drops=drops)
        if new_mol is None:
            continue
        n_fixed += 1
        try:
            mw_out = Descriptors.MolWt(new_mol)
            mw_out_total += mw_out; mw_outs.append(mw_out)
        except Exception: pass
        if new_mol.HasSubstructMatch(CANONICAL_ACRYL):
            n_warhead_match += 1
        if '.' not in Chem.MolToSmiles(new_mol):
            n_single_frag += 1
        writer.write(new_mol)
    writer.close()
    stats = {
        "in": n_in, "fixed": n_fixed,
        "warhead_match": n_warhead_match,
        "single_frag": n_single_frag,
        "mw_in_avg": mw_in_total / max(n_in, 1),
        "mw_out_avg": mw_out_total / max(n_fixed, 1),
    }
    if mw_outs:
        stats["mw_out_median"] = float(np.median(mw_outs))
        stats["mw_out_min"] = float(min(mw_outs))
        stats["mw_out_max"] = float(max(mw_outs))
    if debug:
        stats["drops"] = drops
    return stats


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in", dest="sdf_in", help="single input SDF")
    ap.add_argument("--out", dest="sdf_out", help="single output SDF")
    ap.add_argument("--in-dir", dest="in_dir", help="legacy directory of inpaint.sdf inputs")
    ap.add_argument("--out-dir", dest="out_dir", help="parallel output directory")
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()
    if args.sdf_in:
        sdf_in = Path(args.sdf_in)
        sdf_out = Path(args.sdf_out) if args.sdf_out else sdf_in.with_name(sdf_in.stem + "_fixed.sdf")
        stats = fix_sdf(sdf_in, sdf_out, debug=args.debug)
        print(f"=== {sdf_in.name} → {sdf_out.name} ===")
        for k, v in stats.items(): print(f"  {k}: {v}")
        if stats["in"] > 0:
            print(f"  retention: {100*stats['fixed']/stats['in']:.1f}%  "
                  f"warhead: {100*stats['warhead_match']/stats['in']:.1f}%  "
                  f"single-frag: {100*stats['single_frag']/stats['in']:.1f}%")
    else:
        base = Path(args.in_dir) if args.in_dir else (PROJECT_ROOT / "anchordiff_results" / "day1")
        out_base = Path(args.out_dir) if args.out_dir else base
        for tgt in ["zap70_cys346", "btk_cys481"]:
            sdf_in = base / tgt / "inpaint.sdf"
            sdf_out = out_base / tgt / "inpaint_fixed.sdf"
            if not sdf_in.exists(): continue
            print(f"=== {tgt} ===")
            stats = fix_sdf(sdf_in, sdf_out, debug=args.debug)
            for k, v in stats.items(): print(f"  {k}: {v}")


if __name__ == "__main__":
    main()
