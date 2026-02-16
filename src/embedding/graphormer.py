"""
Graphormer-based molecule embeddings.

Microsoft's Graphormer is a graph transformer that uses:
- Centrality encoding (node importance)
- Spatial encoding (shortest path distances)
- Edge encoding (edge features in attention)

This implementation provides two backends:
1. molfeat: Simple interface via datamol's molfeat library
2. transformers: Direct HuggingFace transformers integration

Supports both frozen (inference-only) and trainable modes for end-to-end learning.

References:
- Graphormer: https://arxiv.org/abs/2106.05234
- Graphormer for molecules: https://arxiv.org/abs/2203.04810
- HuggingFace: https://huggingface.co/clefourrier/graphormer-base-pcqm4mv2
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
from .base import MoleculeEmbedder


class GraphormerEmbedder(nn.Module, MoleculeEmbedder):
    """
    Graphormer-based molecule embedder using graph transformers.

    Uses Microsoft's Graphormer architecture pretrained on molecular data.
    Supports both frozen and trainable modes for end-to-end learning.

    Args:
        model_name: Model variant:
            - 'graphormer-base': Base model (47M params) from PCQM4Mv2
            - 'graphormer-small': Smaller variant for faster inference
            - Or any HuggingFace model path
        backend: Backend to use ('molfeat' or 'transformers')
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for encoding multiple molecules
        trainable: Whether transformer parameters should be trainable (default: False)
    """

    DEFAULT_MODELS = {
        'graphormer-base': 'clefourrier/graphormer-base-pcqm4mv2',
        'graphormer': 'clefourrier/graphormer-base-pcqm4mv2',  # Alias
    }

    def __init__(
        self,
        model_name: str = 'graphormer-base',
        backend: str = 'transformers',
        device: Optional[str] = None,
        batch_size: int = 32,
        trainable: bool = False
    ):
        super().__init__()

        # Resolve model name
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]

        self.model_name = model_name
        self.backend = backend
        self.batch_size = batch_size
        self.trainable = trainable

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize based on backend
        if backend == 'molfeat':
            self._init_molfeat()
        elif backend == 'transformers':
            self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {backend}. Use 'molfeat' or 'transformers'")

    def _init_molfeat(self):
        """Initialize using molfeat library."""
        try:
            from molfeat.trans.pretrained import GraphormerTransformer
        except ImportError:
            raise ImportError(
                "molfeat library required for Graphormer. "
                "Install with: pip install molfeat[transformers]"
            )

        trainable_str = "trainable" if self.trainable else "frozen"
        print(f"Loading Graphormer via molfeat ({trainable_str})...")

        self.transformer = GraphormerTransformer(
            kind=self.model_name,
            dtype=torch.float32
        )

        # molfeat doesn't expose the model directly for training
        # For trainable mode, we need to use the transformers backend
        if self.trainable:
            print("  ⚠ Warning: molfeat backend doesn't support trainable mode. "
                  "Use backend='transformers' for end-to-end training.")
            self.trainable = False

        self._embedding_dim = self.transformer.featurizer.hidden_size
        print(f"  → Embedding dimension: {self._embedding_dim}")

    def _init_transformers(self):
        """Initialize using HuggingFace transformers."""
        try:
            from transformers import GraphormerModel, GraphormerConfig
        except ImportError:
            raise ImportError(
                "transformers library required for Graphormer. "
                "Install with: pip install transformers"
            )

        try:
            from transformers import AutoFeatureExtractor
            self.feature_extractor = AutoFeatureExtractor.from_pretrained(self.model_name)
        except Exception:
            # Fall back to manual feature extraction
            self.feature_extractor = None

        trainable_str = "trainable" if self.trainable else "frozen"
        print(f"Loading Graphormer from {self.model_name} ({trainable_str})...")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.model = GraphormerModel.from_pretrained(self.model_name).to(self.device)

        # Control trainability
        if self.trainable:
            self.model.train()
            print(f"  → Transformer parameters are TRAINABLE")
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"  → Transformer parameters are FROZEN")

        self._embedding_dim = self.model.config.hidden_size
        print(f"  → Embedding dimension: {self._embedding_dim}")

    def _smiles_to_graph_data(self, smiles_list: List[str]) -> dict:
        """Convert SMILES to Graphormer input format."""
        try:
            from rdkit import Chem
        except ImportError:
            raise ImportError("RDKit required for Graphormer. Install with: pip install rdkit")

        # Use feature extractor if available
        if hasattr(self, 'feature_extractor') and self.feature_extractor is not None:
            try:
                return self.feature_extractor(smiles_list, return_tensors="pt", padding=True)
            except Exception:
                pass

        # Manual conversion using RDKit
        # This is a simplified version - Graphormer needs specific graph features
        from torch_geometric.data import Data, Batch

        data_list = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Get atoms and bonds
            num_atoms = mol.GetNumAtoms()

            # Node features (atomic number)
            x = torch.tensor([[atom.GetAtomicNum()] for atom in mol.GetAtoms()], dtype=torch.long)

            # Edge index (bonds)
            edge_index = []
            for bond in mol.GetBonds():
                i, j = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
                edge_index.extend([[i, j], [j, i]])

            if edge_index:
                edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
            else:
                edge_index = torch.zeros((2, 0), dtype=torch.long)

            data_list.append(Data(x=x, edge_index=edge_index, num_nodes=num_atoms))

        # Batch the data
        batch = Batch.from_data_list(data_list)
        return batch

    def encode(self, smiles: Union[str, List[str]]) -> np.ndarray:
        """
        Encode molecule(s) to embedding vector(s).

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Embedding vector(s) as numpy array
        """
        if isinstance(smiles, str):
            smiles = [smiles]
            return_single = True
        else:
            if isinstance(smiles, np.ndarray):
                smiles = smiles.tolist()
            return_single = False

        if self.backend == 'molfeat':
            embeddings = self.transformer(smiles)
            if isinstance(embeddings, torch.Tensor):
                embeddings = embeddings.cpu().numpy()
        else:
            # Batch encode with progress bar
            from tqdm.auto import tqdm

            all_embeddings = []
            n_batches = (len(smiles) + self.batch_size - 1) // self.batch_size

            with tqdm(total=len(smiles), desc="Encoding molecules", unit="mol") as pbar:
                for i in range(0, len(smiles), self.batch_size):
                    batch = smiles[i:i + self.batch_size]
                    batch_emb = self._encode_batch_transformers(batch)
                    all_embeddings.append(batch_emb)
                    pbar.update(len(batch))

            embeddings = np.vstack(all_embeddings)

        if return_single:
            return embeddings[0]
        return embeddings

    def _encode_batch_transformers(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch using transformers backend."""
        # Convert SMILES to graph data
        inputs = self._smiles_to_graph_data(smiles_list)

        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)

        with torch.no_grad():
            if isinstance(inputs, dict):
                outputs = self.model(**inputs)
            else:
                # Handle torch_geometric batch
                outputs = self.model(
                    input_nodes=inputs.x,
                    input_edges=inputs.edge_index,
                    attn_bias=None,
                    in_degree=None,
                    out_degree=None,
                    spatial_pos=None,
                    attn_edge_type=None
                )

        # Pool to get molecule-level embeddings
        # Use the graph-level output or mean pool node embeddings
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            embeddings = outputs[0].mean(dim=1)

        return embeddings.cpu().numpy()

    def encode_trainable(self, smiles: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode molecule(s) with gradient tracking for end-to-end training.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            torch.Tensor with gradient tracking
        """
        if self.backend == 'molfeat':
            raise NotImplementedError(
                "molfeat backend doesn't support trainable mode. "
                "Use backend='transformers' for end-to-end training."
            )

        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = list(smiles)

        # Convert SMILES to graph data
        inputs = self._smiles_to_graph_data(smiles_list)

        if isinstance(inputs, dict):
            inputs = {k: v.to(self.device) for k, v in inputs.items()}
        else:
            inputs = inputs.to(self.device)

        # Forward pass WITHOUT no_grad
        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

        if isinstance(inputs, dict):
            outputs = self.model(**inputs)
        else:
            outputs = self.model(
                input_nodes=inputs.x,
                input_edges=inputs.edge_index,
                attn_bias=None,
                in_degree=None,
                out_degree=None,
                spatial_pos=None,
                attn_edge_type=None
            )

        # Pool to get molecule-level embeddings
        if hasattr(outputs, 'last_hidden_state'):
            embeddings = outputs.last_hidden_state.mean(dim=1)
        else:
            embeddings = outputs[0].mean(dim=1)

        return embeddings

    def freeze(self):
        """Freeze transformer parameters."""
        if self.backend == 'transformers':
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        self.trainable = False
        print("Graphormer transformer frozen")

    def unfreeze(self):
        """Unfreeze transformer parameters."""
        if self.backend != 'transformers':
            raise NotImplementedError("Only transformers backend supports unfreeze")
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self.trainable = True
        print("Graphormer transformer unfrozen")

    def get_encoder_parameters(self):
        """Get trainable encoder parameters for optimizer."""
        if self.backend == 'transformers' and self.trainable:
            return list(self.model.parameters())
        return []

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        model_short = self.model_name.split('/')[-1]
        trainable_suffix = "_trainable" if self.trainable else "_frozen"
        return f"graphormer_{model_short}{trainable_suffix}"


# Convenience constructor
def graphormer_embedder(
    device: Optional[str] = None,
    trainable: bool = False,
    backend: str = 'transformers'
) -> GraphormerEmbedder:
    """
    Create Graphormer embedder.

    Args:
        device: Device to run on
        trainable: Whether to enable gradient updates
        backend: 'transformers' (recommended) or 'molfeat'

    Returns:
        GraphormerEmbedder instance
    """
    return GraphormerEmbedder(
        model_name='graphormer-base',
        backend=backend,
        device=device,
        trainable=trainable
    )
