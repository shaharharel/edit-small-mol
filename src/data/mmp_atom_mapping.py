"""
Atom-level mapping extraction for MMP pairs.

Extracts structural information needed for the structured edit effect predictor:
- removed_atoms_A: Atom indices of leaving fragment in molecule A
- added_atoms_B: Atom indices of incoming fragment in molecule B
- attach_atoms_A: Attachment point atom indices in molecule A
- mapped_pairs: List of (atom_idx_A, atom_idx_B) for atoms in changed region

This module extends the MMP extraction to provide detailed atom-level information
for training graph-based models that need to know exactly which atoms changed.
"""

from typing import List, Tuple, Dict, Optional, Set
from rdkit import Chem
from rdkit.Chem import AllChem, rdFMCS
import logging
import pandas as pd

logger = logging.getLogger(__name__)


def extract_atom_mapping(
    smiles_a: str,
    smiles_b: str,
    edit_smiles: str,
    num_cuts: int
) -> Dict[str, any]:
    """
    Extract atom-level mapping information for an MMP pair.

    Args:
        smiles_a: SMILES for parent molecule A
        smiles_b: SMILES for edited molecule B
        edit_smiles: Reaction SMILES "fragment_out>>fragment_in"
        num_cuts: Number of cuts (attachment points)

    Returns:
        Dict containing:
        - removed_atoms_A: List[int] - Atom indices in A (leaving fragment)
        - added_atoms_B: List[int] - Atom indices in B (incoming fragment)
        - attach_atoms_A: List[int] - Attachment points in A
        - mapped_pairs: List[Tuple[int, int]] - (idx_A, idx_B) for mapped atoms

    Returns empty lists if extraction fails.
    """
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            logger.debug(f"Failed to parse molecules: {smiles_a}, {smiles_b}")
            return _empty_mapping()

        # Parse edit_smiles to get fragments
        if '>>' not in edit_smiles:
            logger.debug(f"Invalid edit_smiles format: {edit_smiles}")
            return _empty_mapping()

        fragment_out, fragment_in = edit_smiles.split('>>')

        # Get fragment molecules
        frag_out_mol = Chem.MolFromSmiles(fragment_out) if fragment_out else None
        frag_in_mol = Chem.MolFromSmiles(fragment_in) if fragment_in else None

        # Find removed atoms in A (match fragment_out in mol_a)
        removed_atoms_A = []
        if frag_out_mol is not None and frag_out_mol.GetNumAtoms() > 0:
            removed_atoms_A = _find_fragment_atoms(mol_a, frag_out_mol)

        # Find added atoms in B (match fragment_in in mol_b)
        added_atoms_B = []
        if frag_in_mol is not None and frag_in_mol.GetNumAtoms() > 0:
            added_atoms_B = _find_fragment_atoms(mol_b, frag_in_mol)

        # Find attachment points in A
        attach_atoms_A = _find_attachment_points(
            mol_a, removed_atoms_A, num_cuts
        )

        # Find mapped atom pairs using MCS
        mapped_pairs = _find_mapped_atoms(
            mol_a, mol_b, removed_atoms_A, added_atoms_B, attach_atoms_A
        )

        return {
            'removed_atoms_A': removed_atoms_A,
            'added_atoms_B': added_atoms_B,
            'attach_atoms_A': attach_atoms_A,
            'mapped_pairs': mapped_pairs
        }

    except Exception as e:
        logger.debug(f"Atom mapping extraction failed: {e}")
        return _empty_mapping()


def _empty_mapping() -> Dict[str, any]:
    """Return empty mapping structure."""
    return {
        'removed_atoms_A': [],
        'added_atoms_B': [],
        'attach_atoms_A': [],
        'mapped_pairs': []
    }


def _find_fragment_atoms(mol: Chem.Mol, fragment: Chem.Mol) -> List[int]:
    """
    Find which atoms in mol match the fragment using substructure matching.

    Args:
        mol: Parent molecule
        fragment: Fragment to find

    Returns:
        List of atom indices in mol that match fragment
    """
    if fragment is None or fragment.GetNumAtoms() == 0:
        return []

    # Use substructure matching
    matches = mol.GetSubstructMatches(fragment)

    if matches:
        # Return first match (assumes unique fragment location)
        return list(matches[0])

    # Try without stereochemistry if first attempt fails
    Chem.RemoveStereochemistry(fragment)
    matches = mol.GetSubstructMatches(fragment)

    if matches:
        return list(matches[0])

    return []


def _find_attachment_points(
    mol: Chem.Mol,
    removed_atoms: List[int],
    num_cuts: int
) -> List[int]:
    """
    Find attachment point atoms (where the edit connects to the core).

    Args:
        mol: Parent molecule
        removed_atoms: Atom indices of removed fragment
        num_cuts: Expected number of attachment points

    Returns:
        List of attachment atom indices in mol
    """
    if not removed_atoms:
        return []

    removed_set = set(removed_atoms)
    attach_atoms = []

    # Find atoms in removed fragment that have bonds to atoms outside
    for atom_idx in removed_atoms:
        atom = mol.GetAtomWithIdx(atom_idx)

        for neighbor in atom.GetNeighbors():
            neighbor_idx = neighbor.GetIdx()

            # If neighbor is NOT in removed fragment, this is an attachment point
            if neighbor_idx not in removed_set:
                # The attachment point is the neighbor (in the core), not the removed atom
                if neighbor_idx not in attach_atoms:
                    attach_atoms.append(neighbor_idx)

    return attach_atoms


def _find_mapped_atoms(
    mol_a: Chem.Mol,
    mol_b: Chem.Mol,
    removed_atoms_A: List[int],
    added_atoms_B: List[int],
    attach_atoms_A: List[int]
) -> List[Tuple[int, int]]:
    """
    Find atom mappings between A and B using MCS (Maximum Common Substructure).

    Maps atoms in the "changed region" around the edit site.
    This captures local structural differences for h_delta_local computation.

    Args:
        mol_a: Parent molecule A
        mol_b: Edited molecule B
        removed_atoms_A: Atoms being removed from A
        added_atoms_B: Atoms being added to B
        attach_atoms_A: Attachment points in A

    Returns:
        List of (atom_idx_A, atom_idx_B) tuples for mapped atoms
    """
    if len(removed_atoms_A) == 0 and len(added_atoms_B) == 0:
        return []

    try:
        # Use MCS to find atom mapping
        mcs_result = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=False,
            timeout=5  # 5 second timeout
        )

        if mcs_result.numAtoms == 0:
            return []

        # Get the common substructure
        mcs_mol = Chem.MolFromSmarts(mcs_result.smartsString)
        if mcs_mol is None:
            return []

        # Get matches in both molecules
        match_a = mol_a.GetSubstructMatch(mcs_mol)
        match_b = mol_b.GetSubstructMatch(mcs_mol)

        if not match_a or not match_b or len(match_a) != len(match_b):
            return []

        # Create mapping
        mapped_pairs = []
        removed_set = set(removed_atoms_A)
        added_set = set(added_atoms_B)
        attach_set = set(attach_atoms_A)

        for idx_in_mcs, (atom_idx_a, atom_idx_b) in enumerate(zip(match_a, match_b)):
            # Only include mappings in the "changed region":
            # - Near attachment points (within 1-2 hops)
            # - Or in the edited fragments themselves

            # Check if atom_a is near the edit
            is_near_edit_a = (
                atom_idx_a in removed_set or
                atom_idx_a in attach_set or
                _is_neighbor_of(mol_a, atom_idx_a, attach_set, max_hops=2)
            )

            # Check if atom_b is near the edit
            is_near_edit_b = (
                atom_idx_b in added_set or
                _is_neighbor_of(mol_b, atom_idx_b, attach_set, max_hops=2)
            )

            if is_near_edit_a or is_near_edit_b:
                mapped_pairs.append((atom_idx_a, atom_idx_b))

        return mapped_pairs

    except Exception as e:
        logger.debug(f"MCS mapping failed: {e}")
        return []


def _is_neighbor_of(
    mol: Chem.Mol,
    atom_idx: int,
    target_set: Set[int],
    max_hops: int = 2
) -> bool:
    """
    Check if atom is within max_hops of any atom in target_set.

    Args:
        mol: Molecule
        atom_idx: Atom index to check
        target_set: Set of target atom indices
        max_hops: Maximum number of bonds away

    Returns:
        True if atom is within max_hops of target_set
    """
    if not target_set:
        return False

    visited = {atom_idx}
    current_layer = {atom_idx}

    for hop in range(max_hops):
        next_layer = set()

        for idx in current_layer:
            atom = mol.GetAtomWithIdx(idx)

            for neighbor in atom.GetNeighbors():
                neighbor_idx = neighbor.GetIdx()

                # Found a target
                if neighbor_idx in target_set:
                    return True

                # Add to next layer if not visited
                if neighbor_idx not in visited:
                    next_layer.add(neighbor_idx)
                    visited.add(neighbor_idx)

        if not next_layer:
            break

        current_layer = next_layer

    return False


def serialize_mapping(mapping: Dict[str, any]) -> Dict[str, str]:
    """
    Serialize mapping to string format for CSV storage.

    Args:
        mapping: Dict with lists of atom indices and tuples

    Returns:
        Dict with string-serialized values
    """
    return {
        'removed_atoms_A': _serialize_list(mapping['removed_atoms_A']),
        'added_atoms_B': _serialize_list(mapping['added_atoms_B']),
        'attach_atoms_A': _serialize_list(mapping['attach_atoms_A']),
        'mapped_pairs': _serialize_pairs(mapping['mapped_pairs'])
    }


def deserialize_mapping(mapping_str: Dict[str, str]) -> Dict[str, any]:
    """
    Deserialize mapping from string format back to Python objects.

    Args:
        mapping_str: Dict with string-serialized values

    Returns:
        Dict with parsed lists and tuples
    """
    return {
        'removed_atoms_A': _deserialize_list(mapping_str['removed_atoms_A']),
        'added_atoms_B': _deserialize_list(mapping_str['added_atoms_B']),
        'attach_atoms_A': _deserialize_list(mapping_str['attach_atoms_A']),
        'mapped_pairs': _deserialize_pairs(mapping_str['mapped_pairs'])
    }


def _serialize_list(lst: List[int]) -> str:
    """Convert list of ints to string: [1,2,3] -> '1;2;3'"""
    if not lst:
        return ''
    return ';'.join(map(str, lst))


def _deserialize_list(s) -> List[int]:
    """Convert string to list of ints: '1;2;3' -> [1,2,3]"""
    # Handle various input types (str, int, float, nan)
    if pd.isna(s) or s == '' or s is None:
        return []
    # If it's already a number (pandas read as int), convert to single-element list
    if isinstance(s, (int, float)):
        return [int(s)]
    # Otherwise treat as string
    return [int(x) for x in str(s).split(';')]


def _serialize_pairs(pairs: List[Tuple[int, int]]) -> str:
    """Convert list of tuples to string: [(1,2),(3,4)] -> '1,2;3,4'"""
    if not pairs:
        return ''
    return ';'.join(f"{a},{b}" for a, b in pairs)


def _deserialize_pairs(s) -> List[Tuple[int, int]]:
    """Convert string to list of tuples: '1,2;3,4' -> [(1,2),(3,4)]"""
    # Handle various input types
    if pd.isna(s) or s == '' or s is None:
        return []
    # Convert to string if needed
    s = str(s)
    return [tuple(map(int, pair.split(','))) for pair in s.split(';')]
