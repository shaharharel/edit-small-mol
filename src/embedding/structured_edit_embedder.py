"""
Structured edit embeddings based on MMP fragment analysis.

Constructs rich edit representations from:
- Global molecule context
- Local attachment environment
- Fragment-level features (in/out substituents)
- RDKit descriptor deltas
- Optional atom-level local deltas

This embedder requires MMP-derived structural information.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem
from rdkit.Chem import Descriptors, rdMolDescriptors


class StructuredEditEmbedder(nn.Module):
    """
    Structured edit embedding layer that processes MMP-derived fragments.

    For each MMP pair (A, B) with edit e, constructs:

    Context features:
    - h_A_global: Global embedding of parent molecule A
    - h_env: Local environment (k-hop neighbors of attachment atoms)
    - h_T: Target-specific context (optional, for later)

    Edit features:
    - h_out: Leaving fragment embedding (substituent-out in A)
    - h_in: Incoming fragment embedding (substituent-in in B)
    - h_delta_local: Atom-level changes in edited region (optional)
    - v_delta_rdkit: RDKit descriptor differences

    Architecture:
        edit_emb = MLP_edit([h_out, h_in, h_delta_local, v_delta_rdkit])
        context_emb = [h_A_global, h_env, h_T]
        combined = concat(context_emb, edit_emb)

    Args:
        gnn_dim: Dimension of GNN atom embeddings (default: 300 for ChemProp)
        rdkit_descriptor_dim: Number of RDKit descriptors to use (default: auto)
        edit_mlp_dims: Hidden dimensions for edit MLP (default: [512, 256])
        dropout: Dropout probability (default: 0.1)
        k_hop_env: Number of hops for attachment environment (default: 2)
        use_local_delta: Whether to compute atom-level deltas in changed region (default: True)
        use_rdkit_fragment_descriptors: Whether to compute descriptors for fragments (default: True)

    Expected input format:
        Each training example should have:
        - mol_A, mol_B: RDKit Mol objects
        - H_A, H_B: Atom embeddings from GNN [n_atoms, gnn_dim]
        - h_A_global, h_B_global: Global embeddings [gnn_dim]
        - removed_atom_indices_A: Indices of atoms in leaving fragment
        - added_atom_indices_B: Indices of atoms in incoming fragment
        - attach_atom_indices_A: Indices of attachment atoms in A
        - mapped_atom_pairs: Optional[(idx_A, idx_B)] for local delta
    """

    def __init__(
        self,
        gnn_dim: int = 300,
        rdkit_descriptor_dim: Optional[int] = None,
        edit_mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.1,
        k_hop_env: int = 2,
        use_local_delta: bool = True,
        use_rdkit_fragment_descriptors: bool = True
    ):
        super().__init__()

        self.gnn_dim = gnn_dim
        self.k_hop_env = k_hop_env
        self.use_local_delta = use_local_delta
        self.use_rdkit_fragment_descriptors = use_rdkit_fragment_descriptors

        # RDKit descriptors to compute
        self.rdkit_descriptors = [
            ('MolLogP', Descriptors.MolLogP),
            ('MolWt', Descriptors.MolWt),
            ('NumHAcceptors', rdMolDescriptors.CalcNumHBA),
            ('NumHDonors', rdMolDescriptors.CalcNumHBD),
            ('TPSA', rdMolDescriptors.CalcTPSA),
            ('NumRotatableBonds', rdMolDescriptors.CalcNumRotatableBonds),
            ('NumAromaticRings', rdMolDescriptors.CalcNumAromaticRings),
            ('NumAliphaticRings', rdMolDescriptors.CalcNumAliphaticRings),
            ('FractionCsp3', rdMolDescriptors.CalcFractionCSP3),
        ]

        if rdkit_descriptor_dim is None:
            # Auto-compute: molecule descriptors + (optionally) fragment descriptors
            n_mol_descriptors = len(self.rdkit_descriptors)
            if self.use_rdkit_fragment_descriptors:
                # delta_desc (n) + (desc_in - desc_out) (n) = 2*n total
                rdkit_descriptor_dim = n_mol_descriptors * 2
            else:
                rdkit_descriptor_dim = n_mol_descriptors

        self.rdkit_descriptor_dim = rdkit_descriptor_dim

        # Compute edit embedding input dimension
        edit_input_dim = (
            gnn_dim +  # h_out
            gnn_dim +  # h_in
            (gnn_dim if use_local_delta else 0) +  # h_delta_local (optional)
            rdkit_descriptor_dim  # v_delta_rdkit
        )

        # Edit MLP
        if edit_mlp_dims is None:
            edit_mlp_dims = [512, 512, 300]

        edit_layers = []
        prev_dim = edit_input_dim
        for hidden_dim in edit_mlp_dims:
            edit_layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            prev_dim = hidden_dim

        self.edit_mlp = nn.Sequential(*edit_layers)
        self.edit_output_dim = prev_dim

        # Context dimension (what gets concatenated with edit embedding)
        # h_A_global + h_env (+ h_T later)
        self.context_dim = gnn_dim + gnn_dim  # Will add h_T later

        # Total output dimension
        self.output_dim = self.context_dim + self.edit_output_dim

    def compute_k_hop_neighbors(
        self,
        mol: Chem.Mol,
        seed_indices: List[int]
    ) -> List[int]:
        """
        Compute k-hop neighborhood of seed atoms.

        Args:
            mol: RDKit molecule
            seed_indices: Starting atom indices

        Returns:
            List of atom indices in k-hop neighborhood (including seeds)
        """
        env_indices = set(seed_indices)

        for _ in range(self.k_hop_env):
            new_indices = []
            for idx in env_indices:
                atom = mol.GetAtomWithIdx(idx)
                for neighbor in atom.GetNeighbors():
                    new_indices.append(neighbor.GetIdx())
            env_indices.update(new_indices)

        return sorted(list(env_indices))

    def compute_rdkit_descriptors(
        self,
        mol: Chem.Mol
    ) -> np.ndarray:
        """
        Compute RDKit descriptors for a molecule.

        Args:
            mol: RDKit molecule

        Returns:
            Descriptor vector as numpy array
        """
        descriptors = []
        for name, func in self.rdkit_descriptors:
            try:
                val = func(mol)
                # Handle None or inf values
                if val is None or np.isinf(val) or np.isnan(val):
                    val = 0.0
            except:
                val = 0.0
            descriptors.append(float(val))

        return np.array(descriptors, dtype=np.float32)

    def extract_fragment_mol(
        self,
        mol: Chem.Mol,
        atom_indices: List[int]
    ) -> Optional[Chem.Mol]:
        """
        Extract a fragment as a separate Mol object.

        Args:
            mol: Parent molecule
            atom_indices: Indices of atoms in fragment

        Returns:
            Fragment as RDKit Mol, or None if extraction fails
        """
        if not atom_indices:
            return None

        try:
            # Create a substructure by keeping only specified atoms
            # This is a simplified extraction - may need refinement
            emol = Chem.EditableMol(mol)
            atoms_to_remove = [i for i in range(mol.GetNumAtoms()) if i not in atom_indices]

            # Remove atoms in reverse order to maintain indices
            for idx in sorted(atoms_to_remove, reverse=True):
                emol.RemoveAtom(idx)

            frag_mol = emol.GetMol()
            Chem.SanitizeMol(frag_mol)
            return frag_mol
        except:
            return None

    def forward(
        self,
        # GNN embeddings
        H_A: torch.Tensor,  # [n_atoms_A, gnn_dim]
        H_B: torch.Tensor,  # [n_atoms_B, gnn_dim]
        h_A_global: torch.Tensor,  # [gnn_dim]
        h_B_global: torch.Tensor,  # [gnn_dim]

        # MMP structural info
        mol_A: Chem.Mol,
        mol_B: Chem.Mol,
        removed_atom_indices_A: List[int],
        added_atom_indices_B: List[int],
        attach_atom_indices_A: List[int],
        mapped_atom_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Construct structured edit embedding.

        Returns:
            Dictionary with:
            - 'edit_embedding': Final edit embedding [output_dim]
            - 'h_edit': Edit-specific embedding [edit_output_dim]
            - 'h_context': Context embedding [context_dim]
            - 'h_A_global': Global molecule A embedding [gnn_dim]
            - 'h_env': Local environment embedding [gnn_dim]
            - 'h_out': Leaving fragment embedding [gnn_dim]
            - 'h_in': Incoming fragment embedding [gnn_dim]
            - 'h_delta_local': Local atom-level delta [gnn_dim] (optional)
            - 'v_delta_rdkit': RDKit descriptor delta [rdkit_descriptor_dim]
        """
        device = H_A.device

        # Step 1: Attachment environment (h_env)
        env_indices = self.compute_k_hop_neighbors(mol_A, attach_atom_indices_A)
        env_vectors = H_A[env_indices]  # [n_env, gnn_dim]
        h_env = env_vectors.mean(dim=0)  # [gnn_dim]

        # Step 2: Fragment embeddings (h_out, h_in)
        if removed_atom_indices_A:
            out_vectors = H_A[removed_atom_indices_A]  # [n_out, gnn_dim]
            h_out = out_vectors.mean(dim=0)  # [gnn_dim]
        else:
            h_out = torch.zeros(self.gnn_dim, device=device)

        if added_atom_indices_B:
            in_vectors = H_B[added_atom_indices_B]  # [n_in, gnn_dim]
            h_in = in_vectors.mean(dim=0)  # [gnn_dim]
        else:
            h_in = torch.zeros(self.gnn_dim, device=device)

        # Step 3: Local delta (optional h_delta_local)
        h_delta_local = None
        if self.use_local_delta and mapped_atom_pairs:
            deltas = []
            for idx_A, idx_B in mapped_atom_pairs:
                if idx_A < H_A.shape[0] and idx_B < H_B.shape[0]:
                    delta_vec = H_B[idx_B] - H_A[idx_A]  # [gnn_dim]
                    deltas.append(delta_vec)

            if deltas:
                delta_tensor = torch.stack(deltas)  # [n_mapped, gnn_dim]
                h_delta_local = delta_tensor.mean(dim=0)  # [gnn_dim]

        if h_delta_local is None and self.use_local_delta:
            h_delta_local = torch.zeros(self.gnn_dim, device=device)

        # Step 4: RDKit Δ-descriptors
        desc_A = self.compute_rdkit_descriptors(mol_A)
        desc_B = self.compute_rdkit_descriptors(mol_B)
        delta_desc = desc_B - desc_A  # [n_descriptors]

        if self.use_rdkit_fragment_descriptors:
            # Compute descriptors for fragments
            frag_out = self.extract_fragment_mol(mol_A, removed_atom_indices_A)
            frag_in = self.extract_fragment_mol(mol_B, added_atom_indices_B)

            if frag_out is not None:
                desc_out = self.compute_rdkit_descriptors(frag_out)
            else:
                desc_out = np.zeros_like(desc_A)

            if frag_in is not None:
                desc_in = self.compute_rdkit_descriptors(frag_in)
            else:
                desc_in = np.zeros_like(desc_A)

            # Concatenate: molecule delta + fragment delta
            v_delta_rdkit = np.concatenate([delta_desc, desc_in - desc_out])
        else:
            v_delta_rdkit = delta_desc

        v_delta_rdkit = torch.from_numpy(v_delta_rdkit).float().to(device)

        # Step 5: Construct edit embedding input
        edit_input_components = [h_out, h_in]
        if self.use_local_delta:
            edit_input_components.append(h_delta_local)
        edit_input_components.append(v_delta_rdkit)

        edit_input = torch.cat(edit_input_components, dim=0)  # [edit_input_dim]

        # Process through edit MLP (add batch dimension for Linear layer)
        h_edit = self.edit_mlp(edit_input.unsqueeze(0)).squeeze(0)  # [edit_output_dim]

        # Step 6: Construct context embedding
        h_context = torch.cat([h_A_global, h_env], dim=0)  # [context_dim]
        # TODO: Add h_T (target-specific context) later

        # Step 7: Combine for final edit embedding
        edit_embedding = torch.cat([h_context, h_edit], dim=0)  # [output_dim]

        return {
            'edit_embedding': edit_embedding,
            'h_edit': h_edit,
            'h_context': h_context,
            'h_A_global': h_A_global,
            'h_env': h_env,
            'h_out': h_out,
            'h_in': h_in,
            'h_delta_local': h_delta_local,
            'v_delta_rdkit': v_delta_rdkit,
        }

    def forward_batch(
        self,
        # Batched GNN embeddings (list of variable-length tensors)
        H_A_list: List[torch.Tensor],  # List of [n_atoms_i, gnn_dim]
        H_B_list: List[torch.Tensor],  # List of [n_atoms_i, gnn_dim]
        h_A_global_batch: torch.Tensor,  # [batch_size, gnn_dim]
        h_B_global_batch: torch.Tensor,  # [batch_size, gnn_dim]

        # Batched MMP structural info
        mol_A_list: List[Chem.Mol],
        mol_B_list: List[Chem.Mol],
        removed_atoms_list: List[List[int]],
        added_atoms_list: List[List[int]],
        attach_atoms_list: List[List[int]],
        mapped_pairs_list: Optional[List[List[Tuple[int, int]]]] = None
    ) -> torch.Tensor:
        """
        Batched forward pass for training efficiency.

        Processes multiple samples in parallel where possible.

        Args:
            H_A_list: List of atom embedding tensors for molecule A
            H_B_list: List of atom embedding tensors for molecule B
            h_A_global_batch: Batched global embeddings for A [batch_size, gnn_dim]
            h_B_global_batch: Batched global embeddings for B [batch_size, gnn_dim]
            mol_A_list: List of RDKit Mol objects for A
            mol_B_list: List of RDKit Mol objects for B
            removed_atoms_list: List of removed atom indices per sample
            added_atoms_list: List of added atom indices per sample
            attach_atoms_list: List of attachment atom indices per sample
            mapped_pairs_list: Optional list of mapped atom pairs per sample

        Returns:
            Batched edit embeddings [batch_size, output_dim]
        """
        batch_size = len(H_A_list)
        device = h_A_global_batch.device

        if mapped_pairs_list is None:
            mapped_pairs_list = [None] * batch_size

        # Process each sample
        edit_embeddings = []
        for i in range(batch_size):
            result = self.forward(
                H_A=H_A_list[i],
                H_B=H_B_list[i],
                h_A_global=h_A_global_batch[i],
                h_B_global=h_B_global_batch[i],
                mol_A=mol_A_list[i],
                mol_B=mol_B_list[i],
                removed_atom_indices_A=removed_atoms_list[i],
                added_atom_indices_B=added_atoms_list[i],
                attach_atom_indices_A=attach_atoms_list[i],
                mapped_atom_pairs=mapped_pairs_list[i]
            )
            edit_embeddings.append(result['edit_embedding'])

        return torch.stack(edit_embeddings, dim=0)  # [batch_size, output_dim]

    def forward_from_smiles(
        self,
        mol_a_smiles: str,
        mol_b_smiles: str,
        embedder,  # ChemPropEmbedder with encode_with_atom_embeddings
        removed_atom_indices_A: List[int],
        added_atom_indices_B: List[int],
        attach_atom_indices_A: List[int],
        mapped_atom_pairs: Optional[List[Tuple[int, int]]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Convenience method: compute edit embedding from SMILES strings.

        This is useful for inference when you have SMILES and MMP info but
        not pre-computed embeddings.

        Args:
            mol_a_smiles: SMILES of molecule A
            mol_b_smiles: SMILES of molecule B
            embedder: ChemPropEmbedder instance (must support encode_with_atom_embeddings)
            removed_atom_indices_A: Indices of removed atoms in A
            added_atom_indices_B: Indices of added atoms in B
            attach_atom_indices_A: Indices of attachment atoms in A
            mapped_atom_pairs: Optional atom mapping between A and B

        Returns:
            Dictionary with edit embedding and components
        """
        # Get atom-level embeddings
        H_A_np, h_A_global_np = embedder.encode_with_atom_embeddings(mol_a_smiles)
        H_B_np, h_B_global_np = embedder.encode_with_atom_embeddings(mol_b_smiles)

        # Convert to tensors
        device = next(self.parameters()).device
        H_A = torch.from_numpy(H_A_np).float().to(device)
        H_B = torch.from_numpy(H_B_np).float().to(device)
        h_A_global = torch.from_numpy(h_A_global_np).float().to(device)
        h_B_global = torch.from_numpy(h_B_global_np).float().to(device)

        # Get RDKit Mol objects
        mol_A = Chem.MolFromSmiles(mol_a_smiles)
        mol_B = Chem.MolFromSmiles(mol_b_smiles)

        return self.forward(
            H_A=H_A,
            H_B=H_B,
            h_A_global=h_A_global,
            h_B_global=h_B_global,
            mol_A=mol_A,
            mol_B=mol_B,
            removed_atom_indices_A=removed_atom_indices_A,
            added_atom_indices_B=added_atom_indices_B,
            attach_atom_indices_A=attach_atom_indices_A,
            mapped_atom_pairs=mapped_atom_pairs
        )

    def freeze(self):
        """Freeze all parameters (stop learning)."""
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze(self):
        """Unfreeze all parameters (resume learning)."""
        for param in self.parameters():
            param.requires_grad = True
