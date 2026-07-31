"""Cap an acrylamide warhead to its saturated propionamide analog.

This is the methodology of Cheng JCTC 2017 (EGFR) and Zhu 2015 (Schrödinger):
to compute the NON-COVALENT recognition free energy (the K_I proxy for a
covalent inhibitor), we replace the Michael acceptor `C=C-C(=O)-N` with the
saturated `C-C-C(=O)-N` and score MM-GBSA WITHOUT any covalent restraint.
The result is dG_recognition — what dG_bind would be if the bond never formed.

Usage as a library:
    from cap_warhead_to_analog import cap_acrylamide_to_propionamide
    capped_smiles, capped_pdb = cap_acrylamide_to_propionamide(smiles, pdb_path)
    if capped_smiles is None:
        # no acrylamide → caller skips this cofold
        ...

Inputs:
    smiles    — full ligand SMILES (must contain [CH2]=[CH]-[C](=O)-[N])
    pdb_path  — original Boltz cofold PDB (full complex; protein ATOM + ligand HETATM)

Outputs (or (None, None) if no acrylamide found):
    capped_smiles — saturated propionamide SMILES (Michael acceptor sat'd)
    capped_pdb    — new full-complex PDB with ligand C=C bonded as single, and
                    two new H atoms placed at sp3 geometry on the (Cβ, Cα) pair.
                    The covalent bond restraint to Cys-SG is NOT removed from
                    geometry — the scorer just won't add a CustomBondForce for it.

Atom-naming convention:
    Boltz PDB names ligand atoms as `<Element><canonical_rank_in_mol_with_Hs+1>`.
    We reproduce the same convention to find which C24/C25 are the β/α carbons.
    The manifest's `contact_breakdown.covalent[0]` already gives the Cβ name
    directly (e.g. "Cys346.SG—lig.C24"), but we don't require it: we discover
    the warhead in-place via RDKit substructure match on the bonded SMILES.
"""
from __future__ import annotations
import re
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
from rdkit import Chem, RDLogger
from rdkit.Chem import AllChem
RDLogger.DisableLog("rdApp.*")


ACRYLAMIDE_SMARTS = "[CH2]=[CH]-[C](=O)-[N]"


def _build_name_to_idx(mol_with_h) -> dict:
    """Reproduce Boltz's '<Element><canonical_rank+1>' naming on a mol-with-Hs."""
    can = list(AllChem.CanonicalRankAtoms(mol_with_h))
    name_to_idx = {}
    for i, atom in enumerate(mol_with_h.GetAtoms()):
        sym = atom.GetSymbol()
        if sym == "H":
            continue
        name_to_idx[f"{sym}{can[i] + 1}"] = i
    return name_to_idx


def _parse_pdb_ligand_lines(pdb_path: Path):
    """Return (protein_and_header_lines, ligand_hetatm_lines, ligand_resname)."""
    other_lines = []
    lig_by_res = {}
    for line in open(pdb_path):
        if line.startswith("HETATM"):
            resname = line[17:20].strip()
            if resname in {"HOH", "WAT", "NA", "K", "MG", "ZN", "CL", "CA", "BR"}:
                other_lines.append(line)
                continue
            lig_by_res.setdefault(resname, []).append(line)
        else:
            other_lines.append(line)
    if not lig_by_res:
        return other_lines, [], None
    lig_resname = max(lig_by_res, key=lambda k: len(lig_by_res[k]))
    return other_lines, lig_by_res[lig_resname], lig_resname


def _build_capped_mol_with_coords(smiles: str, ligand_lines: list):
    """Build a saturated propionamide mol with Boltz heavy-atom coords + AddHs(addCoords=True).

    Returns (mol_capped_with_h, capped_smiles, n_matched_heavies, n_total_heavies)
    or (None, None, 0, 0) if no acrylamide warhead is present in the SMILES.
    """
    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        return None, None, 0, 0

    pat = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    matches = parent.GetSubstructMatches(pat)
    if not matches:
        return None, None, 0, 0

    # First match: SMARTS [CH2]=[CH]-[C](=O)-[N]  ->  (Cβ, Cα, Cc=O, O, N)
    c_beta, c_alpha = matches[0][0], matches[0][1]

    # Saturate the Cβ=Cα double bond
    rw = Chem.RWMol(parent)
    bond = rw.GetBondBetweenAtoms(c_beta, c_alpha)
    if bond is None or bond.GetBondType() != Chem.BondType.DOUBLE:
        return None, None, 0, 0
    bond.SetBondType(Chem.BondType.SINGLE)
    capped = rw.GetMol()
    Chem.SanitizeMol(capped)
    capped_smiles = Chem.MolToSmiles(capped)

    # Parse Boltz coords from the (still acrylamide) ligand HETATM block.
    coords_by_name = {}
    for ln in ligand_lines:
        an = ln[12:16].strip()
        x = float(ln[30:38]); y = float(ln[38:46]); z = float(ln[46:54])
        coords_by_name[an] = (x, y, z)

    # Map Boltz atom names → RDKit indices using the PARENT (acrylamide) naming.
    parent_h = AllChem.AddHs(parent)
    name_to_idx_parent = _build_name_to_idx(parent_h)
    # parent_h indices align with parent indices for heavy atoms (AddHs appends Hs after)
    n_heavy_total = sum(1 for a in parent.GetAtoms() if a.GetAtomicNum() > 1)

    # Build capped + Hs, embed once for H seed, then plant heavy coords.
    capped_h = AllChem.AddHs(capped)
    # Heavy atom indices in `capped` and `capped_h` align (AddHs appends Hs).
    # Heavy atom indices in `parent` and `capped` align too (we only changed a bond order).
    # So we can copy coords by RDKit index directly from name_to_idx_parent.

    if AllChem.EmbedMolecule(capped_h, randomSeed=42) != 0:
        # Embedding failed — try a different seed
        if AllChem.EmbedMolecule(capped_h, randomSeed=1) != 0:
            return None, None, 0, 0

    conf = capped_h.GetConformer()
    matched = 0
    for name, xyz in coords_by_name.items():
        if name in name_to_idx_parent:
            idx = name_to_idx_parent[name]
            # idx is in parent indexing; same in capped + capped_h for heavies
            conf.SetAtomPosition(idx, xyz)
            matched += 1

    if matched < 5:
        return None, None, 0, 0

    # Re-add Hs with addCoords=True to get geometry-aware H placement on the
    # *now-sp3* Cβ and Cα. We do this by stripping current Hs (which have
    # arbitrary post-embed coords) then re-adding from heavies.
    capped_noh = Chem.RemoveHs(capped_h)
    capped_h_final = AllChem.AddHs(capped_noh, addCoords=True)

    # Constrained MMFF: fix every heavy, relax only Hs. Same trick as the
    # scorer's BUG-FIX #1.
    heavy_idx = [a.GetIdx() for a in capped_h_final.GetAtoms() if a.GetAtomicNum() > 1]
    conf_pre = capped_h_final.GetConformer()
    heavies_before = np.array(
        [[conf_pre.GetAtomPosition(i).x, conf_pre.GetAtomPosition(i).y,
          conf_pre.GetAtomPosition(i).z] for i in heavy_idx]
    )
    try:
        mmff_props = AllChem.MMFFGetMoleculeProperties(capped_h_final)
        if mmff_props is not None:
            ff = AllChem.MMFFGetMoleculeForceField(capped_h_final, mmff_props)
            if ff is not None:
                for hi in heavy_idx:
                    ff.AddFixedPoint(hi)
                ff.Minimize(maxIts=200)
    except Exception:
        # Fall back to AddHs(addCoords=True) geometry without minimization.
        pass

    conf_post = capped_h_final.GetConformer()
    heavies_after = np.array(
        [[conf_post.GetAtomPosition(i).x, conf_post.GetAtomPosition(i).y,
          conf_post.GetAtomPosition(i).z] for i in heavy_idx]
    )
    heavy_rmsd = float(np.sqrt(np.mean(np.sum((heavies_after - heavies_before) ** 2, axis=1))))
    if heavy_rmsd > 1e-3:
        # Hard fail — heavies must not drift
        return None, None, 0, 0

    return capped_h_final, capped_smiles, matched, n_heavy_total


def _write_capped_pdb(orig_pdb: Path, other_lines: list, lig_resname: str,
                      mol_capped_h, out_pdb: Path):
    """Write a new PDB with protein/header from orig + ligand from mol_capped_h.

    Ligand atoms are named '<Element><canonical_rank_in_capped_with_Hs+1>'
    to be consistent with Boltz convention (so the downstream scorer's
    SMILES-aware split_pdb can still match by name).
    """
    # Compute capped-naming once on the final capped+H mol
    name_to_idx = _build_name_to_idx(mol_capped_h)
    idx_to_name = {v: k for k, v in name_to_idx.items()}

    # Hydrogens just get sequential H-names within this ligand block.
    h_counter = 1

    # Use the ligand chain ID and residue index from the first HETATM line in original.
    # Pull from `other_lines` is wrong — ligand_lines are NOT in other_lines.
    # We need to re-read orig for the chain/resnum.
    chain_id = "B"
    res_seq = 1
    for line in open(orig_pdb):
        if line.startswith("HETATM"):
            rn = line[17:20].strip()
            if rn == lig_resname:
                chain_id = line[21]
                try:
                    res_seq = int(line[22:26])
                except Exception:
                    res_seq = 1
                break

    # Re-number HETATM atom serials starting from highest ATOM serial + 1
    max_serial = 0
    for ln in other_lines:
        if ln.startswith(("ATOM", "HETATM")):
            try:
                max_serial = max(max_serial, int(ln[6:11]))
            except Exception:
                pass
    serial = max_serial + 1

    conf = mol_capped_h.GetConformer()
    new_lig_lines = []
    for atom in mol_capped_h.GetAtoms():
        i = atom.GetIdx()
        sym = atom.GetSymbol()
        if sym == "H":
            name = f"H{h_counter}"
            h_counter += 1
        else:
            name = idx_to_name.get(i, f"{sym}{i+1}")
        p = conf.GetAtomPosition(i)
        # PDB strict columns: 1-6 record, 7-11 serial, 12 space, 13-16 atom
        # name (4 cols), 17 alt-loc, 18-20 resname, 21 space, 22 chain,
        # 23-26 res-seq, 27-30 spaces, 31-38 x, 39-46 y, 47-54 z, 55-60 occ,
        # 61-66 bfac, 67-76 spaces, 77-78 element.
        # For 1-2 char element + short index ("C24", "H1"), atom-name is
        # left-justified starting at col 14 (col 13 is a leading space).
        if len(name) <= 3:
            atom_name_field = f" {name:<3s}"
        else:
            atom_name_field = f"{name:<4s}"
        line = (
            f"HETATM{serial:5d} {atom_name_field} "
            f"{lig_resname:>3s} {chain_id}{res_seq:4d}    "
            f"{p.x:8.3f}{p.y:8.3f}{p.z:8.3f}  1.00  0.00          {sym:>2s}\n"
        )
        new_lig_lines.append(line)
        serial += 1

    # Write: all original non-ligand-HETATM lines + new ligand block + END
    # Strip any END/MASTER from other_lines first
    cleaned = [ln for ln in other_lines
               if not ln.startswith(("END", "MASTER", "CONECT"))]
    with open(out_pdb, "w") as f:
        f.writelines(cleaned)
        f.writelines(new_lig_lines)
        f.write("END\n")


def cap_acrylamide_to_propionamide(
    smiles: str, pdb_path: Path
) -> Tuple[Optional[str], Optional[Path]]:
    """Cap an acrylamide warhead to propionamide.

    Returns (capped_smiles, capped_pdb_path), or (None, None) if no
    acrylamide was found (or the operation failed).
    """
    pdb_path = Path(pdb_path)

    # Quick reject if no acrylamide in SMILES
    parent = Chem.MolFromSmiles(smiles)
    if parent is None:
        return None, None
    pat = Chem.MolFromSmarts(ACRYLAMIDE_SMARTS)
    if not parent.HasSubstructMatch(pat):
        return None, None

    other_lines, ligand_lines, lig_resname = _parse_pdb_ligand_lines(pdb_path)
    if not ligand_lines or lig_resname is None:
        return None, None

    mol_capped_h, capped_smiles, n_match, n_heavy = _build_capped_mol_with_coords(
        smiles, ligand_lines
    )
    if mol_capped_h is None:
        return None, None

    out_pdb = pdb_path.with_name(pdb_path.stem + "_capped.pdb")
    _write_capped_pdb(pdb_path, other_lines, lig_resname, mol_capped_h, out_pdb)
    return capped_smiles, out_pdb


def main():
    """Smoke test on a single cofold."""
    import sys
    pdb = Path("/Users/shaharharel/Documents/github/edit-small-mol/data/boltz_poses/"
               "boltz_results_top1000__zap70_cys346/predictions/"
               "003889_Amine_Replacements_3889/003889_Amine_Replacements_3889_model_0.pdb")
    smi = "C=CC(=O)N1Cc2cccc(C(=O)NC3=C(Cl)C(=O)c4[nH]ncc4C3=O)c2C1"
    capped_smi, capped_pdb = cap_acrylamide_to_propionamide(smi, pdb)
    print(f"original SMILES: {smi}")
    print(f"capped   SMILES: {capped_smi}")
    print(f"capped   PDB:    {capped_pdb}")
    assert capped_smi is not None and capped_smi.startswith("CCC(=O)"), capped_smi
    print("Smoke OK.")


if __name__ == "__main__":
    main()
