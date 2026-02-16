"""Chemistry utilities using RDKit."""

import logging
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors
from rdkit.Chem.MolStandardize import rdMolStandardize

logger = logging.getLogger(__name__)


def smiles_to_mol(smiles: str) -> Optional[Chem.Mol]:
    """
    Convert SMILES to RDKit molecule object.

    Args:
        smiles: SMILES string

    Returns:
        RDKit Mol object or None if invalid
    """
    try:
        mol = Chem.MolFromSmiles(smiles)
        return mol
    except Exception as e:
        logger.warning(f"Failed to parse SMILES '{smiles}': {e}")
        return None


def mol_to_smiles(mol: Chem.Mol, canonical: bool = True) -> Optional[str]:
    """
    Convert RDKit molecule to SMILES.

    Args:
        mol: RDKit Mol object
        canonical: Whether to return canonical SMILES

    Returns:
        SMILES string or None if conversion fails
    """
    try:
        if canonical:
            return Chem.MolToSmiles(mol)
        else:
            return Chem.MolToSmiles(mol, canonical=False)
    except Exception as e:
        logger.warning(f"Failed to convert molecule to SMILES: {e}")
        return None


def standardize_smiles(smiles: str) -> Optional[str]:
    """
    Standardize a SMILES string.

    Performs:
    - Salt stripping (remove counterions)
    - Neutralization (convert to neutral form)
    - Canonical tautomer selection
    - Canonical SMILES generation

    Args:
        smiles: Input SMILES string

    Returns:
        Standardized canonical SMILES or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        # Remove salts/fragments - keep largest fragment
        remover = rdMolStandardize.FragmentRemover()
        mol = remover.remove(mol)

        # Neutralize charges
        uncharger = rdMolStandardize.Uncharger()
        mol = uncharger.uncharge(mol)

        # Select canonical tautomer
        enumerator = rdMolStandardize.TautomerEnumerator()
        mol = enumerator.Canonicalize(mol)

        # Return canonical SMILES
        return mol_to_smiles(mol, canonical=True)

    except Exception as e:
        logger.warning(f"Failed to standardize SMILES '{smiles}': {e}")
        return None


def compute_molecular_properties(smiles: str) -> Optional[dict]:
    """
    Compute common molecular properties.

    Args:
        smiles: SMILES string

    Returns:
        Dictionary of properties or None if invalid
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        return {
            'mw': Descriptors.MolWt(mol),
            'logp': Descriptors.MolLogP(mol),
            'tpsa': Descriptors.TPSA(mol),
            'hba': Descriptors.NumHAcceptors(mol),
            'hbd': Descriptors.NumHDonors(mol),
            'rotatable_bonds': Descriptors.NumRotatableBonds(mol),
            'aromatic_rings': Descriptors.NumAromaticRings(mol),
            'num_atoms': mol.GetNumAtoms()
        }
    except Exception as e:
        logger.warning(f"Failed to compute properties for '{smiles}': {e}")
        return None


def get_murcko_scaffold(smiles: str) -> Optional[str]:
    """
    Get Bemis-Murcko scaffold (core structure without side chains).

    This is useful for grouping compounds into series and
    controlling for scaffold effects in causal inference.

    Args:
        smiles: SMILES string

    Returns:
        Scaffold SMILES or None
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return None

    try:
        from rdkit.Chem.Scaffolds import MurckoScaffold
        scaffold = MurckoScaffold.GetScaffoldForMol(mol)
        return mol_to_smiles(scaffold)
    except Exception as e:
        logger.warning(f"Failed to compute scaffold for '{smiles}': {e}")
        return None


def apply_transformation(mol: Chem.Mol, from_smarts: str, to_smarts: str) -> List[Chem.Mol]:
    """
    Apply a transformation to a molecule.

    Finds all matches of from_smarts in mol and replaces with to_smarts.

    Args:
        mol: RDKit Mol object
        from_smarts: SMARTS pattern to find
        to_smarts: SMARTS pattern to replace with

    Returns:
        List of transformed molecules (may be multiple if pattern matches multiple times)
    """
    try:
        # Parse patterns
        from_pattern = Chem.MolFromSmarts(from_smarts)
        to_pattern = Chem.MolFromSmarts(to_smarts)

        if from_pattern is None or to_pattern is None:
            logger.warning(f"Invalid SMARTS patterns: {from_smarts} or {to_smarts}")
            return []

        # Find matches
        matches = mol.GetSubstructMatches(from_pattern)
        if not matches:
            return []

        # Apply transformation for each match
        results = []
        for match in matches:
            try:
                # Use RDKit's ReplaceSubstructs
                products = AllChem.ReplaceSubstructs(mol, from_pattern, to_pattern)
                results.extend(products)
            except Exception as e:
                logger.warning(f"Failed to apply transformation at match {match}: {e}")

        return results

    except Exception as e:
        logger.warning(f"Error in apply_transformation: {e}")
        return []


def fragment_molecule(smiles: str, max_cuts: int = 2) -> List[Tuple[str, str]]:
    """
    Fragment a molecule for MMP analysis.

    Args:
        smiles: SMILES string
        max_cuts: Maximum number of cuts (1-3)

    Returns:
        List of (core, fragment) tuples
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return []

    try:
        from rdkit.Chem import rdMMPA

        # Fragment the molecule
        fragments = rdMMPA.FragmentMol(mol, maxCuts=max_cuts, resultsAsMols=False)

        return fragments

    except Exception as e:
        logger.warning(f"Failed to fragment '{smiles}': {e}")
        return []


def is_valid_molecule(smiles: str,
                      max_mw: float = 1000,
                      min_atoms: int = 3,
                      max_atoms: int = 100) -> bool:
    """
    Check if a molecule passes basic validity filters.

    Args:
        smiles: SMILES string
        max_mw: Maximum molecular weight
        min_atoms: Minimum number of atoms
        max_atoms: Maximum number of atoms

    Returns:
        True if valid, False otherwise
    """
    mol = smiles_to_mol(smiles)
    if mol is None:
        return False

    try:
        # Check molecular weight
        mw = Descriptors.MolWt(mol)
        if mw > max_mw:
            return False

        # Check atom count
        num_atoms = mol.GetNumAtoms()
        if num_atoms < min_atoms or num_atoms > max_atoms:
            return False

        return True

    except Exception:
        return False


def parse_edit_smiles(edit_smiles: str) -> Tuple[str, str]:
    """
    Parse edit SMILES format into fragment A and fragment B.

    Args:
        edit_smiles: Reaction SMILES in format "fragment_a>>fragment_b"

    Returns:
        Tuple of (fragment_a_smiles, fragment_b_smiles)

    Raises:
        ValueError: If edit_smiles format is invalid

    """
    if '>>' not in edit_smiles:
        raise ValueError(f"Invalid edit SMILES format (missing '>>'): {edit_smiles}")

    parts = edit_smiles.split('>>')
    if len(parts) != 2:
        raise ValueError(f"Invalid edit SMILES format (expected 2 parts): {edit_smiles}")

    frag_a, frag_b = parts[0].strip(), parts[1].strip()

    if not frag_a or not frag_b:
        raise ValueError(f"Empty fragment in edit SMILES: {edit_smiles}")

    return frag_a, frag_b


def get_edit_name(mol_a_smiles: str, mol_b_smiles: str) -> str:
    """
    Generate a canonical name for a molecular edit.

    Args:
        mol_a_smiles: Molecule A SMILES
        mol_b_smiles: Molecule B SMILES

    Returns:
        Edit name in format "mol_a>>mol_b"
    """
    return f"{mol_a_smiles}>>{mol_b_smiles}"
