"""Chemistry utilities using RDKit."""

import logging
import math
import numpy as np
from typing import Optional, List, Tuple
from rdkit import Chem
from rdkit.Chem import AllChem, Descriptors, rdMolDescriptors, DataStructs
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


# ── Edit Feature Extraction ──────────────────────────────────────────────────

EDIT_FEAT_DIM = 28


def _mol_descriptors(mol: Chem.Mol) -> np.ndarray:
    """Compute 9 molecular descriptors for a molecule.

    Returns array of [logP, MW, TPSA, HBA, HBD, RotBonds, ArRings, AlipRings, Fsp3].
    """
    return np.array([
        Descriptors.MolLogP(mol),
        Descriptors.MolWt(mol),
        Descriptors.TPSA(mol),
        Descriptors.NumHAcceptors(mol),
        Descriptors.NumHDonors(mol),
        Descriptors.NumRotatableBonds(mol),
        Descriptors.NumAromaticRings(mol),
        Descriptors.NumAliphaticRings(mol),
        Descriptors.FractionCSP3(mol),
    ], dtype=np.float32)


def compute_edit_features(
    mol_a_smiles: str,
    mol_b_smiles: str,
    edit_smiles: str,
) -> np.ndarray:
    """Compute 28-dim edit feature vector for a matched molecular pair.

    Feature layout (28 dimensions total):
        [0:9]   Whole-molecule descriptor deltas (mol_b - mol_a):
                delta_logP, delta_MW, delta_TPSA, delta_HBA, delta_HBD,
                delta_RotBonds, delta_ArRings, delta_AlipRings, delta_Fsp3
        [9:18]  Fragment-level descriptor deltas (incoming - leaving):
                Same 9 descriptors computed on the edit fragments
        [18:24] Structural edit type features:
                Tanimoto similarity between fragments, n_attachment_points,
                attachment_is_aromatic, attachment_atom_type_encoded,
                adds_ring, removes_ring
        [24:28] Physicochemical change direction (sign indicators):
                sign(delta_TPSA), sign(delta_logP), sign(delta_MW),
                sign(delta_HBond_total) where HBond_total = HBA + HBD

    Args:
        mol_a_smiles: SMILES for the original molecule.
        mol_b_smiles: SMILES for the edited molecule.
        edit_smiles: Edit in reaction SMILES format "leaving_frag>>incoming_frag".

    Returns:
        Feature vector of shape (28,).
    """
    feats = np.zeros(EDIT_FEAT_DIM, dtype=np.float32)

    # Parse whole molecules
    mol_a = Chem.MolFromSmiles(mol_a_smiles) if mol_a_smiles else None
    mol_b = Chem.MolFromSmiles(mol_b_smiles) if mol_b_smiles else None

    # [0:9] Whole-molecule descriptor deltas
    if mol_a is not None and mol_b is not None:
        try:
            desc_a = _mol_descriptors(mol_a)
            desc_b = _mol_descriptors(mol_b)
            feats[0:9] = desc_b - desc_a
        except Exception:
            pass

    # Parse fragments from edit_smiles
    frag_leaving_mol = None
    frag_incoming_mol = None
    if edit_smiles and '>>' in edit_smiles:
        parts = edit_smiles.split('>>')
        if len(parts) == 2:
            leaving_smi, incoming_smi = parts[0].strip(), parts[1].strip()
            # Fragments may contain [*] attachment points; replace with [H] for descriptors
            leaving_clean = leaving_smi.replace('[*:1]', '[H]').replace('[*:2]', '[H]').replace('[*:3]', '[H]').replace('[*]', '[H]')
            incoming_clean = incoming_smi.replace('[*:1]', '[H]').replace('[*:2]', '[H]').replace('[*:3]', '[H]').replace('[*]', '[H]')
            frag_leaving_mol = Chem.MolFromSmiles(leaving_clean)
            frag_incoming_mol = Chem.MolFromSmiles(incoming_clean)

    # [9:18] Fragment-level descriptor deltas
    if frag_leaving_mol is not None and frag_incoming_mol is not None:
        try:
            desc_leaving = _mol_descriptors(frag_leaving_mol)
            desc_incoming = _mol_descriptors(frag_incoming_mol)
            feats[9:18] = desc_incoming - desc_leaving
        except Exception:
            pass

    # [18:24] Structural edit type features
    if frag_leaving_mol is not None and frag_incoming_mol is not None:
        try:
            # Tanimoto between fragment fingerprints
            fp_leaving = AllChem.GetMorganFingerprintAsBitVect(frag_leaving_mol, 2, nBits=1024)
            fp_incoming = AllChem.GetMorganFingerprintAsBitVect(frag_incoming_mol, 2, nBits=1024)
            feats[18] = DataStructs.TanimotoSimilarity(fp_leaving, fp_incoming)
        except Exception:
            feats[18] = 0.0

        # Count attachment points from original edit_smiles
        if edit_smiles and '>>' in edit_smiles:
            leaving_smi = edit_smiles.split('>>')[0].strip()
            n_attach = leaving_smi.count('[*')
            feats[19] = float(n_attach)

            # Check if attachment atom is aromatic (use mol_a and leaving fragment context)
            if mol_a is not None and n_attach > 0:
                try:
                    # Parse the leaving fragment with attachment points as query
                    frag_with_attach = Chem.MolFromSmiles(leaving_smi)
                    if frag_with_attach is not None:
                        # Check if any atom bonded to a dummy atom is aromatic
                        for atom in frag_with_attach.GetAtoms():
                            if atom.GetAtomicNum() == 0:  # dummy atom [*]
                                for neighbor in atom.GetNeighbors():
                                    if neighbor.GetIsAromatic():
                                        feats[20] = 1.0
                                        break
                except Exception:
                    pass

            # Attachment atom type: encode as atomic number / 10 for normalization
            if mol_a is not None and n_attach > 0:
                try:
                    frag_with_attach = Chem.MolFromSmiles(leaving_smi)
                    if frag_with_attach is not None:
                        for atom in frag_with_attach.GetAtoms():
                            if atom.GetAtomicNum() == 0:
                                for neighbor in atom.GetNeighbors():
                                    feats[21] = float(neighbor.GetAtomicNum()) / 10.0
                                    break
                                break
                except Exception:
                    pass

        # adds_ring: incoming fragment has ring but leaving does not
        try:
            leaving_has_ring = frag_leaving_mol.GetRingInfo().NumRings() > 0
            incoming_has_ring = frag_incoming_mol.GetRingInfo().NumRings() > 0
            feats[22] = 1.0 if (incoming_has_ring and not leaving_has_ring) else 0.0
            feats[23] = 1.0 if (leaving_has_ring and not incoming_has_ring) else 0.0
        except Exception:
            pass

    # [24:28] Physicochemical change direction (sign indicators)
    feats[24] = float(np.sign(feats[2]))   # sign(delta_TPSA)
    feats[25] = float(np.sign(feats[0]))   # sign(delta_logP)
    feats[26] = float(np.sign(feats[1]))   # sign(delta_MW)
    # sign(delta_HBond_total) where HBond_total = HBA + HBD
    delta_hbond_total = feats[3] + feats[4]  # delta_HBA + delta_HBD
    feats[27] = float(np.sign(delta_hbond_total))

    return feats


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
