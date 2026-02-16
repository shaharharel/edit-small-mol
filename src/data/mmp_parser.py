"""
MMP (Matched Molecular Pair) data parser.

Parses MMP structural information from CSV columns produced by mmpdb.
"""

from typing import Dict, List, Tuple, Optional
import numpy as np
import pandas as pd


def parse_atom_indices(indices_str: str) -> List[int]:
    """
    Parse semicolon-separated atom indices.

    Args:
        indices_str: String like "0;1;2;3" or empty string

    Returns:
        List of integer indices

    Examples:
        >>> parse_atom_indices("0;1;2;3")
        [0, 1, 2, 3]
        >>> parse_atom_indices("")
        []
        >>> parse_atom_indices("5")
        [5]
    """
    if pd.isna(indices_str) or indices_str == "" or indices_str is None:
        return []

    indices_str = str(indices_str).strip()
    if not indices_str:
        return []

    return [int(idx.strip()) for idx in indices_str.split(";") if idx.strip()]


def parse_mapped_pairs(pairs_str: str) -> List[Tuple[int, int]]:
    """
    Parse atom mapping pairs between molecule A and B.

    Args:
        pairs_str: String like "0,0;1,1;2,2" where each pair is "idx_A,idx_B"

    Returns:
        List of (idx_A, idx_B) tuples

    Examples:
        >>> parse_mapped_pairs("0,0;1,1;2,2")
        [(0, 0), (1, 1), (2, 2)]
        >>> parse_mapped_pairs("9,12;10,11")
        [(9, 12), (10, 11)]
        >>> parse_mapped_pairs("")
        []
    """
    if pd.isna(pairs_str) or pairs_str == "" or pairs_str is None:
        return []

    pairs_str = str(pairs_str).strip()
    if not pairs_str:
        return []

    pairs = []
    for pair in pairs_str.split(";"):
        pair = pair.strip()
        if "," in pair:
            parts = pair.split(",")
            if len(parts) == 2:
                try:
                    idx_a = int(parts[0].strip())
                    idx_b = int(parts[1].strip())
                    pairs.append((idx_a, idx_b))
                except ValueError:
                    continue

    return pairs


def parse_mmp_info(row: pd.Series) -> Dict:
    """
    Parse all MMP structural information from a DataFrame row.

    Expected columns:
        - removed_atoms_A: Indices of atoms in leaving fragment (e.g., "0;1;2;3")
        - added_atoms_B: Indices of atoms in incoming fragment (e.g., "13;14;15")
        - attach_atoms_A: Indices of attachment atoms in A (e.g., "9")
        - mapped_pairs: Atom mapping A→B (e.g., "9,12;10,11;11,10")

    Args:
        row: pandas Series with MMP columns

    Returns:
        Dict with parsed structural information:
        {
            'removed_atom_indices_A': List[int],
            'added_atom_indices_B': List[int],
            'attach_atom_indices_A': List[int],
            'mapped_atom_pairs': List[Tuple[int, int]]
        }
    """
    return {
        'removed_atom_indices_A': parse_atom_indices(row.get('removed_atoms_A', '')),
        'added_atom_indices_B': parse_atom_indices(row.get('added_atoms_B', '')),
        'attach_atom_indices_A': parse_atom_indices(row.get('attach_atoms_A', '')),
        'mapped_atom_pairs': parse_mapped_pairs(row.get('mapped_pairs', ''))
    }


def validate_mmp_data(df: pd.DataFrame) -> Tuple[bool, List[str]]:
    """
    Validate that DataFrame has required MMP columns.

    Args:
        df: DataFrame to validate

    Returns:
        Tuple of (is_valid, list_of_missing_columns)
    """
    required_columns = ['removed_atoms_A', 'added_atoms_B', 'attach_atoms_A', 'mapped_pairs']
    missing = [col for col in required_columns if col not in df.columns]

    return len(missing) == 0, missing


def parse_mmp_batch(df: pd.DataFrame) -> Dict[str, List]:
    """
    Parse MMP info for an entire DataFrame (batch).

    Args:
        df: DataFrame with MMP columns

    Returns:
        Dict with lists of parsed info for each sample
    """
    result = {
        'removed_atom_indices_A': [],
        'added_atom_indices_B': [],
        'attach_atom_indices_A': [],
        'mapped_atom_pairs': []
    }

    for idx in range(len(df)):
        row = df.iloc[idx]
        mmp_info = parse_mmp_info(row)

        result['removed_atom_indices_A'].append(mmp_info['removed_atom_indices_A'])
        result['added_atom_indices_B'].append(mmp_info['added_atom_indices_B'])
        result['attach_atom_indices_A'].append(mmp_info['attach_atom_indices_A'])
        result['mapped_atom_pairs'].append(mmp_info['mapped_atom_pairs'])

    return result
