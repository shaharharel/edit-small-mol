"""
Graphormer structured edit embedder with node-level embeddings.

Extracts local edit representations by:
1. Getting node-level embeddings from Graphormer's final transformer layer
2. Using atom indices to extract edit site embeddings
3. Applying same k-hop environment extraction as D-MPNN

Key features:
- Direct node-level embeddings from graph transformer
- Spatial encoding preserves distance information
- Compatible with local environment extraction pattern
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem

from .structured_edit_base import StructuredEditEmbedderBase


class GraphormerStructuredEditEmbedder(StructuredEditEmbedderBase):
    """
    Graphormer-based structured edit embedder.

    Uses node-level embeddings from Graphormer's final transformer layer
    before pooling, then applies local environment extraction like D-MPNN.

    Args:
        model_name: Graphormer model name (default: 'graphormer-base')
        device: Device to run on ('cuda' or 'cpu')
        k_hop_env: Number of hops for local environment (default: 2)
        trainable: Whether to enable gradient updates (default: False)
    """

    DEFAULT_MODELS = {
        'graphormer-base': 'clefourrier/graphormer-base-pcqm4mv2',
        'graphormer': 'clefourrier/graphormer-base-pcqm4mv2',
    }

    def __init__(
        self,
        model_name: str = 'graphormer-base',
        device: Optional[str] = None,
        k_hop_env: int = 2,
        trainable: bool = False
    ):
        # Resolve model name
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model and get embedding dim BEFORE calling super().__init__
        model, embedding_dim = self._load_graphormer(model_name, device, trainable)

        # Initialize base class FIRST
        super().__init__(
            embedding_dim=embedding_dim,
            k_hop_env=k_hop_env,
            trainable=trainable
        )

        # Now assign attributes
        self.model_name = model_name
        self.device = device
        self._embedding_dim = embedding_dim
        self.model = model

    @staticmethod
    def _load_graphormer(model_name: str, device: str, trainable: bool):
        """Load Graphormer model and return (model, embedding_dim)."""
        try:
            from transformers import GraphormerModel
        except ImportError:
            raise ImportError(
                "transformers library required for Graphormer. "
                "Install with: pip install transformers"
            )

        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading Graphormer structured embedder ({trainable_str})...")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            model = GraphormerModel.from_pretrained(model_name).to(device)

        embedding_dim = model.config.hidden_size

        # Control trainability
        if trainable:
            model.train()
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        print(f"  → Embedding dimension: {embedding_dim}")
        return model, embedding_dim

    def _smiles_to_graphormer_input(self, smiles: str) -> Tuple[Dict, Chem.Mol]:
        """
        Convert SMILES to Graphormer input format.

        Returns dict of tensors and the RDKit mol.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            raise ValueError(f"Invalid SMILES: {smiles}")

        num_atoms = mol.GetNumAtoms()

        # Compute degree (in-degree = out-degree for molecular graphs)
        degrees = []
        for atom in mol.GetAtoms():
            degrees.append(atom.GetDegree())

        in_degree = torch.tensor([degrees], dtype=torch.long, device=self.device)
        out_degree = torch.tensor([degrees], dtype=torch.long, device=self.device)

        # Compute shortest path distances for spatial encoding
        from rdkit.Chem import AllChem, GetDistanceMatrix
        dist_matrix = GetDistanceMatrix(mol)
        spatial_pos = torch.tensor(dist_matrix, dtype=torch.long, device=self.device).unsqueeze(0)

        # Node features (atomic numbers, padded to max_node=512)
        input_nodes = torch.zeros(1, 512, dtype=torch.long, device=self.device)
        for i, atom in enumerate(mol.GetAtoms()):
            input_nodes[0, i] = atom.GetAtomicNum()

        # Edge information
        # For simplicity, we create attention bias based on connectivity
        attn_bias = torch.zeros(1, num_atoms, num_atoms, device=self.device)

        # Padding mask
        padding_mask = torch.ones(1, 512, dtype=torch.bool, device=self.device)
        padding_mask[0, :num_atoms] = False

        return {
            'input_nodes': input_nodes,
            'in_degree': in_degree,
            'out_degree': out_degree,
            'spatial_pos': spatial_pos,
            'attn_bias': attn_bias,
            'num_atoms': num_atoms
        }, mol

    def get_node_embeddings(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor, Chem.Mol]:
        """
        Get node-level embeddings from Graphormer.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of:
            - node_embeddings: [n_atoms, embedding_dim]
            - global_embedding: [embedding_dim]
            - mol: RDKit molecule
        """
        try:
            inputs, mol = self._smiles_to_graphormer_input(smiles)
        except ValueError:
            # Return zeros for invalid molecules
            return (
                torch.zeros(1, self._embedding_dim, device=self.device),
                torch.zeros(self._embedding_dim, device=self.device),
                None
            )

        num_atoms = inputs.pop('num_atoms')

        # Forward pass
        if self.trainable:
            outputs = self.model(**inputs, output_hidden_states=True)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, output_hidden_states=True)

        # Get node embeddings from last hidden state
        # Shape: [1, seq_len, hidden_size]
        if hasattr(outputs, 'last_hidden_state'):
            hidden_states = outputs.last_hidden_state
        else:
            hidden_states = outputs[0]

        # Extract only actual atom embeddings (not padding)
        node_embeddings = hidden_states[0, :num_atoms]  # [n_atoms, embedding_dim]

        # Global embedding via mean pooling
        global_embedding = node_embeddings.mean(dim=0)  # [embedding_dim]

        return node_embeddings, global_embedding, mol

    def get_atom_embeddings(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level embeddings for a molecule.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of:
            - atom_embeddings: [n_atoms, embedding_dim]
            - global_embedding: [embedding_dim]
        """
        node_embs, global_emb, _ = self.get_node_embeddings(smiles)
        return node_embs, global_emb

    def get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Optional[Chem.Mol] = None
    ) -> Optional[Dict[int, List[int]]]:
        """
        Graphormer uses node-level embeddings, so no token mapping needed.

        Returns None to indicate direct atom indexing.
        """
        return None

    def forward(
        self,
        smiles_a: str,
        smiles_b: str,
        removed_atom_indices_a: List[int],
        added_atom_indices_b: List[int],
        attach_atom_indices_a: List[int],
        attach_atom_indices_b: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Compute structured edit embedding using Graphormer node embeddings.

        Uses the same extraction logic as D-MPNN but with Graphormer's
        transformer-based node representations.
        """
        # Get node embeddings
        H_a, h_a_global, mol_a = self.get_node_embeddings(smiles_a)
        H_b, h_b_global, mol_b = self.get_node_embeddings(smiles_b)

        zero = torch.zeros(self._embedding_dim, device=self.device)

        if mol_a is None or mol_b is None:
            return {
                'h_a_global': h_a_global if mol_a else zero,
                'h_b_global': h_b_global if mol_b else zero,
                'h_env_a': zero,
                'h_out': zero,
                'h_in': zero,
            }

        # Local environment in A
        h_env_a = self.get_local_environment_embedding(H_a, mol_a, attach_atom_indices_a)

        # Fragment embeddings
        h_out = self.get_fragment_embedding(H_a, removed_atom_indices_a)
        h_in = self.get_fragment_embedding(H_b, added_atom_indices_b)

        result = {
            'h_a_global': h_a_global,
            'h_b_global': h_b_global,
            'h_env_a': h_env_a,
            'h_out': h_out,
            'h_in': h_in,
        }

        # Optional: environment in B
        if attach_atom_indices_b:
            h_env_b = self.get_local_environment_embedding(H_b, mol_b, attach_atom_indices_b)
            result['h_env_b'] = h_env_b

        return result

    def freeze(self):
        """Freeze Graphormer parameters."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.trainable = False
        print("Graphormer frozen")

    def unfreeze(self):
        """Unfreeze Graphormer parameters."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self.trainable = True
        print("Graphormer unfrozen")

    def get_encoder_parameters(self) -> List:
        """Get trainable encoder parameters."""
        if self.trainable:
            return list(self.model.parameters())
        return []

    @property
    def name(self) -> str:
        """Return embedder name."""
        model_short = self.model_name.split('/')[-1]
        trainable_str = "_trainable" if self.trainable else "_frozen"
        return f"graphormer_structured_{model_short}{trainable_str}"


# Convenience constructor
def graphormer_structured_embedder(
    device: Optional[str] = None,
    trainable: bool = False,
    k_hop: int = 2
) -> GraphormerStructuredEditEmbedder:
    """
    Create Graphormer structured edit embedder.

    Args:
        device: Device to run on
        trainable: Whether to enable gradient updates
        k_hop: Number of hops for local environment

    Returns:
        GraphormerStructuredEditEmbedder instance
    """
    return GraphormerStructuredEditEmbedder(
        model_name='graphormer-base',
        device=device,
        k_hop_env=k_hop,
        trainable=trainable
    )
