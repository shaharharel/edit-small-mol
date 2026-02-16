"""
GraphMVP structured edit embedder.

Uses the GraphMVP GNN model to extract atom-level embeddings for
structured edit representation. GraphMVP provides high-quality molecular
representations pretrained with 3D geometry.

Key features:
- Atom-level GNN embeddings (300-dim by default)
- Direct atom index access (no token-to-atom mapping needed)
- Supports both GraphMVP_C (contrastive) and GraphMVP_G (generative) variants
- Trainable/frozen modes for end-to-end learning

Pretrained weights from: https://huggingface.co/chao1224/MoleculeSTM

References:
- GraphMVP: Pre-training Molecular Graph Representation with 3D Geometry (ICLR 2022)
- https://github.com/chao1224/GraphMVP
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing, global_mean_pool, global_add_pool, global_max_pool
from torch_geometric.utils import add_self_loops
from torch_geometric.data import Data, Batch
from typing import Dict, List, Tuple, Optional, Union
from pathlib import Path
import numpy as np

from rdkit import Chem

from .structured_edit_base import StructuredEditEmbedderBase


# =============================================================================
# OGB-style Atom/Bond Feature Dimensions
# =============================================================================

def get_atom_feature_dims():
    """Return dimensions for each atom feature (OGB-style)."""
    return [119, 4, 12, 12, 10, 6, 6, 2, 2]

def get_bond_feature_dims():
    """Return dimensions for each bond feature (OGB-style)."""
    return [5, 6, 2]


# =============================================================================
# Atom and Bond Encoders (OGB-style)
# =============================================================================

class AtomEncoder(nn.Module):
    """Encodes atom features into a single embedding vector."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.atom_embedding_list = nn.ModuleList()

        for dim in get_atom_feature_dims():
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.atom_embedding_list.append(emb)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Atom features [num_atoms, num_atom_features]
        Returns:
            Atom embeddings [num_atoms, emb_dim]
        """
        x_embedding = 0
        for i, emb in enumerate(self.atom_embedding_list):
            x_embedding += emb(x[:, i])
        return x_embedding


class BondEncoder(nn.Module):
    """Encodes bond features into a single embedding vector."""

    def __init__(self, emb_dim: int):
        super().__init__()
        self.bond_embedding_list = nn.ModuleList()

        for dim in get_bond_feature_dims():
            emb = nn.Embedding(dim, emb_dim)
            nn.init.xavier_uniform_(emb.weight.data)
            self.bond_embedding_list.append(emb)

    def forward(self, edge_attr: torch.Tensor) -> torch.Tensor:
        """
        Args:
            edge_attr: Bond features [num_edges, num_bond_features]
        Returns:
            Bond embeddings [num_edges, emb_dim]
        """
        bond_embedding = 0
        for i, emb in enumerate(self.bond_embedding_list):
            bond_embedding += emb(edge_attr[:, i])
        return bond_embedding


# =============================================================================
# GIN Convolution Layer
# =============================================================================

class GINConv(MessagePassing):
    """Graph Isomorphism Network convolution layer."""

    def __init__(self, emb_dim: int):
        super().__init__(aggr='add')

        self.mlp = nn.Sequential(
            nn.Linear(emb_dim, 2 * emb_dim),
            nn.BatchNorm1d(2 * emb_dim),
            nn.ReLU(),
            nn.Linear(2 * emb_dim, emb_dim)
        )
        self.eps = nn.Parameter(torch.Tensor([0.0]))
        self.bond_encoder = BondEncoder(emb_dim)

    def forward(self, x, edge_index, edge_attr):
        edge_embedding = self.bond_encoder(edge_attr)
        out = self.mlp((1 + self.eps) * x + self.propagate(
            edge_index, x=x, edge_attr=edge_embedding
        ))
        return out

    def message(self, x_j, edge_attr):
        return F.relu(x_j + edge_attr)

    def update(self, aggr_out):
        return aggr_out


# =============================================================================
# GraphMVP GNN Model
# =============================================================================

class GNN_graphmvp(nn.Module):
    """
    GraphMVP GNN encoder.

    Architecture:
    - OGB-style atom/bond encoders
    - 5 GIN layers with batch normalization
    - Jumping Knowledge (JK) aggregation
    - Global pooling
    """

    def __init__(
        self,
        num_layer: int = 5,
        emb_dim: int = 300,
        JK: str = "last",
        drop_ratio: float = 0.0,
        gnn_type: str = "gin"
    ):
        super().__init__()

        self.num_layer = num_layer
        self.emb_dim = emb_dim
        self.drop_ratio = drop_ratio
        self.JK = JK

        if num_layer < 2:
            raise ValueError("Number of GNN layers must be at least 2")

        # Atom encoder
        self.atom_encoder = AtomEncoder(emb_dim)

        # GNN layers
        self.gnns = nn.ModuleList()
        self.batch_norms = nn.ModuleList()

        for _ in range(num_layer):
            if gnn_type == "gin":
                self.gnns.append(GINConv(emb_dim))
            else:
                raise ValueError(f"Unsupported gnn_type: {gnn_type}")
            self.batch_norms.append(nn.BatchNorm1d(emb_dim))

    def forward(self, data) -> torch.Tensor:
        """
        Forward pass returning node-level embeddings.

        Args:
            data: PyG Data object with x, edge_index, edge_attr

        Returns:
            Node embeddings [num_nodes, emb_dim]
        """
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # Encode atoms
        h = self.atom_encoder(x)

        # Apply GNN layers
        h_list = [h]
        for layer in range(self.num_layer):
            h = self.gnns[layer](h_list[layer], edge_index, edge_attr)
            h = self.batch_norms[layer](h)

            if layer == self.num_layer - 1:
                h = F.dropout(h, self.drop_ratio, training=self.training)
            else:
                h = F.dropout(F.relu(h), self.drop_ratio, training=self.training)

            h_list.append(h)

        # Jumping Knowledge aggregation
        if self.JK == "last":
            node_representation = h_list[-1]
        elif self.JK == "sum":
            node_representation = sum(h_list[1:])
        elif self.JK == "concat":
            node_representation = torch.cat(h_list[1:], dim=1)
        else:
            raise ValueError(f"Unsupported JK: {self.JK}")

        return node_representation


# =============================================================================
# Molecule Featurizer (OGB-style)
# =============================================================================

class MoleculeFeaturizer:
    """
    Converts SMILES to PyG Data with OGB-style features.

    Atom features (9 total):
    - Atomic number (0-118)
    - Chirality type (0-3)
    - Degree (0-10)
    - Formal charge (-5 to +5 mapped to 0-11)
    - Num H (0-8)
    - Num radical electrons (0-4)
    - Hybridization (0-4)
    - Is aromatic (0-1)
    - Is in ring (0-1)

    Bond features (3 total):
    - Bond type (single=0, double=1, triple=2, aromatic=3, misc=4)
    - Stereo (0-5)
    - Is conjugated (0-1)
    """

    def __init__(self):
        self.allowable_features = {
            'possible_atomic_num_list': list(range(1, 119)) + ['misc'],
            'possible_chirality_list': [
                Chem.rdchem.ChiralType.CHI_UNSPECIFIED,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CW,
                Chem.rdchem.ChiralType.CHI_TETRAHEDRAL_CCW,
                Chem.rdchem.ChiralType.CHI_OTHER
            ],
            'possible_degree_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 'misc'],
            'possible_formal_charge_list': [-5, -4, -3, -2, -1, 0, 1, 2, 3, 4, 5, 'misc'],
            'possible_numH_list': [0, 1, 2, 3, 4, 5, 6, 7, 8, 'misc'],
            'possible_number_radical_e_list': [0, 1, 2, 3, 4, 'misc'],
            'possible_hybridization_list': [
                Chem.rdchem.HybridizationType.SP,
                Chem.rdchem.HybridizationType.SP2,
                Chem.rdchem.HybridizationType.SP3,
                Chem.rdchem.HybridizationType.SP3D,
                Chem.rdchem.HybridizationType.SP3D2,
                'misc'
            ],
            'possible_is_aromatic_list': [False, True],
            'possible_is_in_ring_list': [False, True],
            'possible_bond_type_list': [
                Chem.rdchem.BondType.SINGLE,
                Chem.rdchem.BondType.DOUBLE,
                Chem.rdchem.BondType.TRIPLE,
                Chem.rdchem.BondType.AROMATIC,
                'misc'
            ],
            'possible_bond_stereo_list': [
                Chem.rdchem.BondStereo.STEREONONE,
                Chem.rdchem.BondStereo.STEREOZ,
                Chem.rdchem.BondStereo.STEREOE,
                Chem.rdchem.BondStereo.STEREOCIS,
                Chem.rdchem.BondStereo.STEREOTRANS,
                Chem.rdchem.BondStereo.STEREOANY
            ],
            'possible_is_conjugated_list': [False, True]
        }

    def safe_index(self, lst, elem):
        """Return index of element in list, or last index if not found."""
        try:
            return lst.index(elem)
        except ValueError:
            return len(lst) - 1

    def smiles_to_graph(self, smiles: str) -> Optional[Data]:
        """
        Convert SMILES to PyG Data.

        Args:
            smiles: SMILES string

        Returns:
            PyG Data object or None if conversion fails
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None

        # Atom features
        atom_features_list = []
        for atom in mol.GetAtoms():
            atom_feature = [
                self.safe_index(self.allowable_features['possible_atomic_num_list'], atom.GetAtomicNum()),
                self.allowable_features['possible_chirality_list'].index(atom.GetChiralTag()),
                self.safe_index(self.allowable_features['possible_degree_list'], atom.GetTotalDegree()),
                self.safe_index(self.allowable_features['possible_formal_charge_list'], atom.GetFormalCharge()),
                self.safe_index(self.allowable_features['possible_numH_list'], atom.GetTotalNumHs()),
                self.safe_index(self.allowable_features['possible_number_radical_e_list'], atom.GetNumRadicalElectrons()),
                self.safe_index(self.allowable_features['possible_hybridization_list'], atom.GetHybridization()),
                self.allowable_features['possible_is_aromatic_list'].index(atom.GetIsAromatic()),
                self.allowable_features['possible_is_in_ring_list'].index(atom.IsInRing())
            ]
            atom_features_list.append(atom_feature)

        x = torch.tensor(atom_features_list, dtype=torch.long)

        # Edge features
        if mol.GetNumBonds() > 0:
            edges_list = []
            edge_features_list = []

            for bond in mol.GetBonds():
                i = bond.GetBeginAtomIdx()
                j = bond.GetEndAtomIdx()

                edge_feature = [
                    self.safe_index(self.allowable_features['possible_bond_type_list'], bond.GetBondType()),
                    self.allowable_features['possible_bond_stereo_list'].index(bond.GetStereo()),
                    self.allowable_features['possible_is_conjugated_list'].index(bond.GetIsConjugated())
                ]

                # Add both directions
                edges_list.append((i, j))
                edge_features_list.append(edge_feature)
                edges_list.append((j, i))
                edge_features_list.append(edge_feature)

            edge_index = torch.tensor(edges_list, dtype=torch.long).t().contiguous()
            edge_attr = torch.tensor(edge_features_list, dtype=torch.long)
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
            edge_attr = torch.empty((0, 3), dtype=torch.long)

        return Data(x=x, edge_index=edge_index, edge_attr=edge_attr)


# =============================================================================
# GraphMVP Structured Edit Embedder
# =============================================================================

class GraphMVPStructuredEditEmbedder(StructuredEditEmbedderBase):
    """
    GraphMVP-based structured edit embedder.

    Uses the GraphMVP GNN to extract atom-level embeddings for precise
    representation of molecular edits. Since GraphMVP operates on graphs,
    atom indices directly correspond to node indices (no token mapping needed).

    Args:
        variant: Model variant - 'GraphMVP_C' (contrastive) or 'GraphMVP_G' (generative)
        device: Device to run on ('cuda' or 'cpu')
        k_hop_env: Number of hops for local environment (default: 2)
        trainable: Whether to enable gradient updates (default: False)
        checkpoint_path: Optional custom checkpoint path
        num_layers: Number of GNN layers (default: 5)
        emb_dim: Embedding dimension (default: 300)
        drop_ratio: Dropout ratio (default: 0.0)
    """

    CHECKPOINT_DIR = Path(__file__).parent.parent.parent / "data/models/graphmvp/pretrained_GraphMVP"

    VARIANTS = {
        'GraphMVP_C': 'GraphMVP_C/model.pth',
        'GraphMVP_G': 'GraphMVP_G/model.pth',
        'graphmvp_c': 'GraphMVP_C/model.pth',
        'graphmvp_g': 'GraphMVP_G/model.pth',
        'graphmvp': 'GraphMVP_C/model.pth',  # Default to contrastive
    }

    def __init__(
        self,
        variant: str = 'GraphMVP_C',
        device: Optional[str] = None,
        k_hop_env: int = 2,
        trainable: bool = False,
        checkpoint_path: Optional[str] = None,
        num_layers: int = 5,
        emb_dim: int = 300,
        drop_ratio: float = 0.0
    ):
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Initialize base class
        super().__init__(
            embedding_dim=emb_dim,
            k_hop_env=k_hop_env,
            trainable=trainable
        )

        self.variant = variant
        self.device = device
        self._emb_dim = emb_dim

        # Create GNN model
        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading GraphMVP ({variant}, {trainable_str}) on {device}...")

        self.gnn = GNN_graphmvp(
            num_layer=num_layers,
            emb_dim=emb_dim,
            JK="last",
            drop_ratio=drop_ratio if trainable else 0.0,
            gnn_type="gin"
        ).to(device)

        # Load checkpoint
        if checkpoint_path is None:
            if variant.lower() not in self.VARIANTS:
                raise ValueError(f"Unknown variant: {variant}. Choose from: {list(self.VARIANTS.keys())}")
            checkpoint_path = self.CHECKPOINT_DIR / self.VARIANTS[variant.lower()]

        checkpoint_path = Path(checkpoint_path)
        if checkpoint_path.exists():
            print(f"  Loading weights from: {checkpoint_path}")
            state_dict = torch.load(checkpoint_path, map_location=device)
            self.gnn.load_state_dict(state_dict, strict=True)
            print(f"  Loaded {len(state_dict)} parameters")
        else:
            print(f"  WARNING: Checkpoint not found at {checkpoint_path}")
            print(f"  Using random initialization")
            print(f"  To download: from huggingface_hub import hf_hub_download")
            print(f"  hf_hub_download('chao1224/MoleculeSTM', 'pretrained_GraphMVP/{variant}/model.pth')")

        # Featurizer for SMILES -> graph conversion
        self.featurizer = MoleculeFeaturizer()

        # Global pooling
        self.pool = global_mean_pool

        # Control trainability
        if self.trainable:
            self.gnn.train()
        else:
            self.gnn.eval()
            for param in self.gnn.parameters():
                param.requires_grad = False

        print(f"  Embedding dim: {emb_dim}, Layers: {num_layers}")

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
        # Convert SMILES to graph
        graph = self.featurizer.smiles_to_graph(smiles)

        if graph is None:
            # Return zeros for invalid SMILES
            zero_atom = torch.zeros(1, self._emb_dim, device=self.device)
            zero_global = torch.zeros(self._emb_dim, device=self.device)
            return zero_atom, zero_global

        # Move to device
        graph = graph.to(self.device)
        graph.batch = torch.zeros(graph.x.size(0), dtype=torch.long, device=self.device)

        # Forward pass
        if self.trainable:
            node_embeddings = self.gnn(graph)
        else:
            with torch.no_grad():
                node_embeddings = self.gnn(graph)

        # Global pooling
        global_embedding = self.pool(node_embeddings, graph.batch).squeeze(0)

        return node_embeddings, global_embedding

    def get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Optional[Chem.Mol] = None
    ) -> Optional[Dict[int, List[int]]]:
        """
        GraphMVP is graph-based, so token-to-atom mapping is not needed.
        Atom indices directly correspond to node indices.

        Returns:
            None (not applicable for graph-based models)
        """
        return None

    def embed_batch(
        self,
        smiles_list: List[str]
    ) -> Tuple[List[torch.Tensor], torch.Tensor]:
        """
        Efficient batch embedding of multiple molecules.

        Args:
            smiles_list: List of SMILES strings

        Returns:
            Tuple of:
            - List of atom embeddings (one per molecule)
            - Global embeddings [batch_size, embedding_dim]
        """
        # Convert all SMILES to graphs
        graphs = []
        valid_indices = []

        for i, smiles in enumerate(smiles_list):
            graph = self.featurizer.smiles_to_graph(smiles)
            if graph is not None:
                graphs.append(graph)
                valid_indices.append(i)

        if not graphs:
            # All invalid
            atom_embs = [torch.zeros(1, self._emb_dim, device=self.device) for _ in smiles_list]
            global_embs = torch.zeros(len(smiles_list), self._emb_dim, device=self.device)
            return atom_embs, global_embs

        # Batch graphs
        batch = Batch.from_data_list(graphs).to(self.device)

        # Forward pass
        if self.trainable:
            node_embeddings = self.gnn(batch)
        else:
            with torch.no_grad():
                node_embeddings = self.gnn(batch)

        # Global pooling
        global_embeddings = self.pool(node_embeddings, batch.batch)

        # Split node embeddings by molecule
        atom_embs_valid = []
        ptr = 0
        for i in range(len(graphs)):
            n_atoms = (batch.batch == i).sum().item()
            atom_embs_valid.append(node_embeddings[ptr:ptr + n_atoms])
            ptr += n_atoms

        # Reconstruct full lists (with zeros for invalid)
        atom_embs = []
        global_embs = torch.zeros(len(smiles_list), self._emb_dim, device=self.device)

        valid_idx = 0
        for i in range(len(smiles_list)):
            if i in valid_indices:
                atom_embs.append(atom_embs_valid[valid_idx])
                global_embs[i] = global_embeddings[valid_idx]
                valid_idx += 1
            else:
                atom_embs.append(torch.zeros(1, self._emb_dim, device=self.device))

        return atom_embs, global_embs

    def freeze(self):
        """Freeze GNN parameters."""
        for param in self.gnn.parameters():
            param.requires_grad = False
        self.gnn.eval()
        self.trainable = False

    def unfreeze(self):
        """Unfreeze GNN parameters for training."""
        for param in self.gnn.parameters():
            param.requires_grad = True
        self.gnn.train()
        self.trainable = True

    @property
    def name(self) -> str:
        """Return the name of this embedder."""
        return f"GraphMVP-{self.variant}"

    def __repr__(self) -> str:
        trainable_str = "trainable" if self.trainable else "frozen"
        return f"GraphMVPStructuredEditEmbedder(variant={self.variant}, dim={self._emb_dim}, {trainable_str})"


# =============================================================================
# Convenience functions
# =============================================================================

def download_graphmvp_checkpoints(target_dir: Optional[str] = None):
    """
    Download GraphMVP pretrained checkpoints from HuggingFace.

    Args:
        target_dir: Directory to save checkpoints (default: data/models/graphmvp)
    """
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("Please install huggingface_hub: pip install huggingface_hub")
        return

    if target_dir is None:
        target_dir = GraphMVPStructuredEditEmbedder.CHECKPOINT_DIR

    target_dir = Path(target_dir)
    target_dir.mkdir(parents=True, exist_ok=True)

    print(f"Downloading GraphMVP checkpoints to {target_dir}")

    for variant in ['GraphMVP_C', 'GraphMVP_G']:
        print(f"  Downloading {variant}...")
        path = hf_hub_download(
            repo_id='chao1224/MoleculeSTM',
            filename=f'pretrained_GraphMVP/{variant}/model.pth',
            local_dir=str(target_dir.parent),
            local_dir_use_symlinks=False
        )
        print(f"    Saved to: {path}")

    print("Done!")


if __name__ == "__main__":
    # Test the embedder
    print("Testing GraphMVP Structured Edit Embedder")
    print("=" * 60)

    # Create embedder
    embedder = GraphMVPStructuredEditEmbedder(
        variant='GraphMVP_C',
        device='cpu',
        trainable=False
    )

    # Test single molecule
    smiles = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    print(f"\nTest molecule: {smiles}")

    atom_embs, global_emb = embedder.get_atom_embeddings(smiles)
    print(f"Atom embeddings shape: {atom_embs.shape}")
    print(f"Global embedding shape: {global_emb.shape}")

    # Test structured edit
    smiles_a = "CC(=O)Oc1ccccc1C(=O)O"  # Aspirin
    smiles_b = "CC(=O)Oc1ccc(Cl)cc1C(=O)O"  # Chloro-aspirin

    print(f"\nEdit: {smiles_a} -> {smiles_b}")
    result = embedder.forward(
        smiles_a=smiles_a,
        smiles_b=smiles_b,
        removed_atom_indices_a=[],
        added_atom_indices_b=[10],  # Cl atom
        attach_atom_indices_a=[5],  # Ring carbon
    )

    print("Edit representation shapes:")
    for key, val in result.items():
        print(f"  {key}: {val.shape}")

    print("\nDone!")
