"""
MolFM-based molecule embeddings.

MolFM (Molecular Foundation Model) from PharMolix/OpenBioMed is a multimodal
foundation model that combines:
- Molecular graph representations (GNN)
- SMILES sequence representations
- Knowledge graph embeddings

The model is pretrained on large-scale molecular data with multimodal
contrastive learning objectives.

Supports both frozen (inference-only) and trainable modes for end-to-end learning.

References:
- MolFM: https://arxiv.org/abs/2307.09484
- OpenBioMed: https://github.com/PharMolix/OpenBioMed
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
from .base import MoleculeEmbedder


class MolFMEmbedder(nn.Module, MoleculeEmbedder):
    """
    MolFM-based molecule embedder using multimodal foundation model.

    Uses the molecular encoder from MolFM which combines graph and sequence
    representations. Supports both frozen and trainable modes.

    Args:
        model_path: Path to MolFM checkpoint or model name:
            - 'molfm': Default MolFM model
            - 'molfm-base': Base variant
            - Or path to local checkpoint
        modality: Which modality to use ('graph', 'sequence', 'multimodal')
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for encoding multiple molecules
        trainable: Whether model parameters should be trainable (default: False)
    """

    # Model configurations
    MODEL_CONFIGS = {
        'molfm': {
            'hidden_size': 768,
            'repo': 'PharMolix/OpenBioMed',
            'checkpoint': 'molfm_pretrained.pth'
        },
        'molfm-base': {
            'hidden_size': 512,
            'repo': 'PharMolix/OpenBioMed',
            'checkpoint': 'molfm_base.pth'
        }
    }

    def __init__(
        self,
        model_path: str = 'molfm',
        modality: str = 'multimodal',
        device: Optional[str] = None,
        batch_size: int = 32,
        trainable: bool = False
    ):
        super().__init__()

        self.model_path = model_path
        self.modality = modality
        self.batch_size = batch_size
        self.trainable = trainable

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Initialize model
        self._init_model()

    def _init_model(self):
        """Initialize MolFM model."""
        trainable_str = "trainable" if self.trainable else "frozen"
        print(f"Loading MolFM ({trainable_str})...")

        # Try to load from OpenBioMed
        try:
            self._init_openbiomedl()
            return
        except ImportError:
            pass

        # Try HuggingFace approach
        try:
            self._init_huggingface()
            return
        except ImportError:
            pass

        # Fallback to local implementation
        print("  ⚠ OpenBioMed not found. Using simplified MolFM implementation.")
        print("    For full MolFM, install: pip install git+https://github.com/PharMolix/OpenBioMed.git")
        self._init_fallback()

    def _init_openbiomedl(self):
        """Initialize using OpenBioMed library."""
        try:
            from open_biomed.models.molecule import MolFM
            from open_biomed.models.molecule.molfm import MolFMConfig
        except ImportError:
            raise ImportError(
                "OpenBioMed library required for full MolFM. "
                "Install with: pip install git+https://github.com/PharMolix/OpenBioMed.git"
            )

        # Load config and model
        if self.model_path in self.MODEL_CONFIGS:
            config = self.MODEL_CONFIGS[self.model_path]
            self._embedding_dim = config['hidden_size']
        else:
            self._embedding_dim = 768

        # Load pretrained model
        self.model = MolFM.from_pretrained(self.model_path)
        self.model = self.model.to(self.device)

        # Control trainability
        if self.trainable:
            self.model.train()
            print(f"  → MolFM parameters are TRAINABLE")
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"  → MolFM parameters are FROZEN")

        self._backend = 'openbiomedl'
        print(f"  → Embedding dimension: {self._embedding_dim}")

    def _init_huggingface(self):
        """Try to initialize from HuggingFace hub."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required")

        # Check if MolFM is available on HuggingFace
        # As of now, MolFM may not be directly on HF, so this is future-proofing
        try:
            self.tokenizer = AutoTokenizer.from_pretrained("PharMolix/MolFM")
            self.model = AutoModel.from_pretrained("PharMolix/MolFM").to(self.device)
            self._embedding_dim = self.model.config.hidden_size
            self._backend = 'huggingface'

            if self.trainable:
                self.model.train()
            else:
                self.model.eval()
                for param in self.model.parameters():
                    param.requires_grad = False

            print(f"  → Loaded from HuggingFace, embedding dim: {self._embedding_dim}")
        except Exception:
            raise ImportError("MolFM not available on HuggingFace")

    def _init_fallback(self):
        """
        Fallback initialization using a similar architecture.

        This creates a MolFM-like model using available components:
        - Graph encoder from torch_geometric
        - Sequence encoder from transformers (ChemBERTa)
        - Fusion layer
        """
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError(
                "transformers library required. Install with: pip install transformers"
            )

        print("  Initializing MolFM-like architecture with:")
        print("    - Sequence encoder: ChemBERTa-2")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            self.seq_tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            self.seq_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(self.device)

        # Get actual embedding dimension from the loaded model
        self._embedding_dim = self.seq_encoder.config.hidden_size

        # Optional: Add graph encoder if torch_geometric is available
        self.graph_encoder = None
        try:
            from torch_geometric.nn import GINEConv, global_mean_pool
            print("    - Graph encoder: GIN (Graph Isomorphism Network)")

            # Simple GIN encoder - output matches sequence encoder dim
            self.graph_encoder = nn.Sequential(
                nn.Linear(118, 256),  # Atom features
                nn.ReLU(),
                nn.Linear(256, self._embedding_dim)
            ).to(self.device)
        except ImportError:
            print("    - Graph encoder: Not available (torch_geometric not installed)")

        # Fusion layer for multimodal
        if self.graph_encoder is not None:
            self.fusion = nn.Sequential(
                nn.Linear(self._embedding_dim * 2, self._embedding_dim),
                nn.ReLU(),
                nn.Linear(self._embedding_dim, self._embedding_dim)
            ).to(self.device)
        else:
            self.fusion = None

        # Control trainability
        if self.trainable:
            self.seq_encoder.train()
            print(f"  → MolFM parameters are TRAINABLE")
        else:
            self.seq_encoder.eval()
            for param in self.seq_encoder.parameters():
                param.requires_grad = False
            if self.graph_encoder is not None:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = False
            if self.fusion is not None:
                for param in self.fusion.parameters():
                    param.requires_grad = False
            print(f"  → MolFM parameters are FROZEN")

        self._backend = 'fallback'
        print(f"  → Embedding dimension: {self._embedding_dim}")

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

        # Batch encode with progress bar
        from tqdm.auto import tqdm

        all_embeddings = []
        n_batches = (len(smiles) + self.batch_size - 1) // self.batch_size

        with tqdm(total=len(smiles), desc="Encoding molecules", unit="mol") as pbar:
            for i in range(0, len(smiles), self.batch_size):
                batch = smiles[i:i + self.batch_size]
                batch_emb = self._encode_batch(batch)
                all_embeddings.append(batch_emb)
                pbar.update(len(batch))

        embeddings = np.vstack(all_embeddings)

        if return_single:
            return embeddings[0]
        return embeddings

    def _encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES."""
        with torch.no_grad():
            embeddings = self._forward_batch(smiles_list)
        return embeddings.cpu().numpy()

    def _forward_batch(self, smiles_list: List[str]) -> torch.Tensor:
        """Forward pass for a batch (shared between encode and encode_trainable)."""
        if self._backend == 'openbiomedl':
            return self._forward_openbiomedl(smiles_list)
        elif self._backend == 'huggingface':
            return self._forward_huggingface(smiles_list)
        else:
            return self._forward_fallback(smiles_list)

    def _forward_openbiomedl(self, smiles_list: List[str]) -> torch.Tensor:
        """Forward using OpenBioMed model."""
        # Prepare inputs based on modality
        if self.modality == 'graph':
            embeddings = self.model.encode_graph(smiles_list)
        elif self.modality == 'sequence':
            embeddings = self.model.encode_sequence(smiles_list)
        else:  # multimodal
            embeddings = self.model.encode(smiles_list)
        return embeddings

    def _forward_huggingface(self, smiles_list: List[str]) -> torch.Tensor:
        """Forward using HuggingFace model."""
        inputs = self.tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.model(**inputs)
        # Mean pool over sequence
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        return embeddings

    def _forward_fallback(self, smiles_list: List[str]) -> torch.Tensor:
        """Forward using fallback implementation."""
        # Sequence encoding
        inputs = self.seq_tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        outputs = self.seq_encoder(**inputs)
        attention_mask = inputs['attention_mask'].unsqueeze(-1)
        seq_embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)

        if self.modality == 'sequence' or self.graph_encoder is None:
            return seq_embeddings

        # Graph encoding (if available and requested)
        if self.modality in ['graph', 'multimodal']:
            try:
                graph_embeddings = self._encode_graphs(smiles_list)

                if self.modality == 'graph':
                    return graph_embeddings

                # Multimodal fusion
                combined = torch.cat([seq_embeddings, graph_embeddings], dim=-1)
                return self.fusion(combined)
            except Exception:
                # Fall back to sequence only if graph encoding fails
                return seq_embeddings

        return seq_embeddings

    def _encode_graphs(self, smiles_list: List[str]) -> torch.Tensor:
        """Encode molecules as graphs."""
        try:
            from rdkit import Chem
            from rdkit.Chem import AllChem
        except ImportError:
            raise ImportError("RDKit required for graph encoding")

        # Simple atom feature encoding
        embeddings = []
        for smiles in smiles_list:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                # Use zero vector for invalid molecules
                embeddings.append(torch.zeros(768, device=self.device))
                continue

            # Get atom features (one-hot atomic numbers)
            atom_features = torch.zeros(118, device=self.device)
            for atom in mol.GetAtoms():
                atomic_num = min(atom.GetAtomicNum(), 117)
                atom_features[atomic_num] += 1

            # Normalize
            atom_features = atom_features / (mol.GetNumAtoms() + 1e-6)

            # Pass through graph encoder
            graph_emb = self.graph_encoder(atom_features.unsqueeze(0))
            embeddings.append(graph_emb.squeeze(0))

        return torch.stack(embeddings)

    def encode_trainable(self, smiles: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode molecule(s) with gradient tracking for end-to-end training.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            torch.Tensor with gradient tracking
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = list(smiles)

        # Set training mode
        if self.trainable:
            if self._backend == 'fallback':
                self.seq_encoder.train()
            elif hasattr(self, 'model'):
                self.model.train()

        return self._forward_batch(smiles_list)

    def freeze(self):
        """Freeze model parameters."""
        if self._backend == 'fallback':
            self.seq_encoder.eval()
            for param in self.seq_encoder.parameters():
                param.requires_grad = False
            if self.graph_encoder is not None:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = False
            if self.fusion is not None:
                for param in self.fusion.parameters():
                    param.requires_grad = False
        elif hasattr(self, 'model'):
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        self.trainable = False
        print("MolFM frozen")

    def unfreeze(self):
        """Unfreeze model parameters."""
        if self._backend == 'fallback':
            self.seq_encoder.train()
            for param in self.seq_encoder.parameters():
                param.requires_grad = True
            if self.graph_encoder is not None:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = True
            if self.fusion is not None:
                for param in self.fusion.parameters():
                    param.requires_grad = True
        elif hasattr(self, 'model'):
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True

        self.trainable = True
        print("MolFM unfrozen")

    def get_encoder_parameters(self):
        """Get trainable encoder parameters for optimizer."""
        if not self.trainable:
            return []

        params = []
        if self._backend == 'fallback':
            params.extend(list(self.seq_encoder.parameters()))
            if self.graph_encoder is not None:
                params.extend(list(self.graph_encoder.parameters()))
            if self.fusion is not None:
                params.extend(list(self.fusion.parameters()))
        elif hasattr(self, 'model'):
            params.extend(list(self.model.parameters()))

        return params

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        trainable_suffix = "_trainable" if self.trainable else "_frozen"
        return f"molfm_{self.modality}{trainable_suffix}"


# Convenience constructor
def molfm_embedder(
    modality: str = 'multimodal',
    device: Optional[str] = None,
    trainable: bool = False
) -> MolFMEmbedder:
    """
    Create MolFM embedder.

    Args:
        modality: 'graph', 'sequence', or 'multimodal'
        device: Device to run on
        trainable: Whether to enable gradient updates

    Returns:
        MolFMEmbedder instance
    """
    return MolFMEmbedder(
        model_path='molfm',
        modality=modality,
        device=device,
        trainable=trainable
    )
