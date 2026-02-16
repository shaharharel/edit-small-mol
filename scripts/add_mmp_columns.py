"""
Add MMP structural columns to pairs CSV for StructuredEditEffectPredictor.

This script computes:
- removed_atoms_A: Atom indices of the leaving fragment
- added_atoms_B: Atom indices of the incoming fragment
- attach_atoms_A: Atom indices where fragments attach
- mapped_pairs: Atom mapping between conserved regions

These columns enable fragment-based edit prediction with StructuredEditEffectPredictor.
"""

import pandas as pd
import numpy as np
from rdkit import Chem
from rdkit.Chem import rdFMCS
from typing import List, Tuple, Optional
from tqdm import tqdm


def find_mcs_mapping(mol_a: Chem.Mol, mol_b: Chem.Mol) -> Optional[List[Tuple[int, int]]]:
    """
    Find maximum common substructure and return atom mapping.

    Returns:
        List of (atom_idx_a, atom_idx_b) tuples for mapped atoms
    """
    try:
        # Find MCS
        mcs = rdFMCS.FindMCS(
            [mol_a, mol_b],
            atomCompare=rdFMCS.AtomCompare.CompareElements,
            bondCompare=rdFMCS.BondCompare.CompareOrder,
            ringMatchesRingOnly=True,
            completeRingsOnly=True,
            timeout=5
        )

        if not mcs.smartsString:
            return None

        mcs_mol = Chem.MolFromSmarts(mcs.smartsString)
        if mcs_mol is None:
            return None

        # Get matches
        match_a = mol_a.GetSubstructMatch(mcs_mol)
        match_b = mol_b.GetSubstructMatch(mcs_mol)

        if not match_a or not match_b:
            return None

        # Create mapping
        mapping = list(zip(match_a, match_b))
        return mapping

    except Exception as e:
        return None


def compute_mmp_features(
    smiles_a: str,
    smiles_b: str
) -> Tuple[List[int], List[int], List[int], List[Tuple[int, int]]]:
    """
    Compute MMP structural features for a molecule pair.

    Returns:
        - removed_atoms_A: Indices of atoms in leaving fragment
        - added_atoms_B: Indices of atoms in incoming fragment
        - attach_atoms_A: Indices of attachment points in A
        - mapped_pairs: Atom mapping between conserved regions
    """
    try:
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        if mol_a is None or mol_b is None:
            return [], [], [], []

        # Find atom mapping via MCS
        mapping = find_mcs_mapping(mol_a, mol_b)

        if mapping is None or len(mapping) == 0:
            # No common substructure - treat as complete replacement
            removed_atoms_A = list(range(mol_a.GetNumAtoms()))
            added_atoms_B = list(range(mol_b.GetNumAtoms()))
            attach_atoms_A = []
            mapped_pairs = []
            return removed_atoms_A, added_atoms_B, attach_atoms_A, mapped_pairs

        # Get mapped atom sets
        mapped_a = {a for a, b in mapping}
        mapped_b = {b for a, b in mapping}

        # Removed atoms are those in A but not in mapping
        removed_atoms_A = [i for i in range(mol_a.GetNumAtoms()) if i not in mapped_a]

        # Added atoms are those in B but not in mapping
        added_atoms_B = [i for i in range(mol_b.GetNumAtoms()) if i not in mapped_b]

        # Attachment atoms are mapped atoms connected to removed atoms
        attach_atoms_A = []
        for atom_idx in mapped_a:
            atom = mol_a.GetAtomWithIdx(atom_idx)
            for neighbor in atom.GetNeighbors():
                if neighbor.GetIdx() in removed_atoms_A:
                    attach_atoms_A.append(atom_idx)
                    break

        # Remove duplicates and sort
        attach_atoms_A = sorted(list(set(attach_atoms_A)))

        return removed_atoms_A, added_atoms_B, attach_atoms_A, mapping

    except Exception as e:
        print(f"Error processing pair: {e}")
        return [], [], [], []


def add_mmp_columns_to_csv(
    input_csv: str,
    output_csv: str,
    mol_a_col: str = 'mol_a',
    mol_b_col: str = 'mol_b'
):
    """
    Add MMP structural columns to pairs CSV.

    Args:
        input_csv: Path to input CSV with mol_a and mol_b SMILES
        output_csv: Path to output CSV with added MMP columns
        mol_a_col: Column name for molecule A SMILES
        mol_b_col: Column name for molecule B SMILES
    """
    print(f"\nLoading data from {input_csv}...")
    df = pd.read_csv(input_csv)

    print(f"Processing {len(df)} molecule pairs...")

    # Initialize columns
    df['removed_atoms_A'] = None
    df['added_atoms_B'] = None
    df['attach_atoms_A'] = None
    df['mapped_pairs'] = None

    # Process each pair
    for idx, row in tqdm(df.iterrows(), total=len(df), desc="Computing MMP features"):
        smiles_a = row[mol_a_col]
        smiles_b = row[mol_b_col]

        removed, added, attach, mapping = compute_mmp_features(smiles_a, smiles_b)

        # Store as string representations
        df.at[idx, 'removed_atoms_A'] = str(removed)
        df.at[idx, 'added_atoms_B'] = str(added)
        df.at[idx, 'attach_atoms_A'] = str(attach)
        df.at[idx, 'mapped_pairs'] = str(mapping)

    # Save
    print(f"\nSaving enhanced data to {output_csv}...")
    df.to_csv(output_csv, index=False)

    # Print statistics
    print(f"\n{'='*70}")
    print("MMP Feature Statistics:")
    print(f"{'='*70}")

    # Calculate avg sizes
    def parse_list(s):
        try:
            return eval(s)
        except:
            return []

    removed_sizes = [len(parse_list(x)) for x in df['removed_atoms_A']]
    added_sizes = [len(parse_list(x)) for x in df['added_atoms_B']]
    attach_sizes = [len(parse_list(x)) for x in df['attach_atoms_A']]
    mapping_sizes = [len(parse_list(x)) for x in df['mapped_pairs']]

    print(f"Removed atoms (avg):    {np.mean(removed_sizes):.1f} ± {np.std(removed_sizes):.1f}")
    print(f"Added atoms (avg):      {np.mean(added_sizes):.1f} ± {np.std(added_sizes):.1f}")
    print(f"Attachment points (avg): {np.mean(attach_sizes):.1f} ± {np.std(attach_sizes):.1f}")
    print(f"Mapped atoms (avg):     {np.mean(mapping_sizes):.1f} ± {np.std(mapping_sizes):.1f}")
    print(f"{'='*70}\n")

    print(f"✅ Enhanced CSV saved to: {output_csv}")


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Add MMP columns to pairs CSV')
    parser.add_argument('--input', type=str, required=True, help='Input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Output CSV file')
    parser.add_argument('--mol-a-col', type=str, default='mol_a', help='Column name for molecule A')
    parser.add_argument('--mol-b-col', type=str, default='mol_b', help='Column name for molecule B')

    args = parser.parse_args()

    add_mmp_columns_to_csv(
        input_csv=args.input,
        output_csv=args.output,
        mol_a_col=args.mol_a_col,
        mol_b_col=args.mol_b_col
    )
