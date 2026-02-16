"""
ChemBERTa-based molecule embeddings.

Uses transformer models pre-trained on SMILES:
- ChemBERTa (original: seyonec/ChemBERTa-zinc-base-v1)
- ChemBERTa-2 77M MLM (DeepChem/ChemBERTa-77M-MLM) - recommended
- ChemBERTa-2 77M MTR (DeepChem/ChemBERTa-77M-MTR)
- MolBERT

ChemBERTa-2 is trained on 77M compounds from PubChem using either:
- MLM (Masked Language Modeling) - better for general embeddings
- MTR (Multi-Task Regression) - includes property prediction pretraining

Supports both frozen (inference-only) and trainable modes for end-to-end learning.

References:
- ChemBERTa-2: https://arxiv.org/abs/2209.01712
- HuggingFace: https://huggingface.co/DeepChem/ChemBERTa-77M-MLM
"""

import numpy as np
import torch
import torch.nn as nn
from transformers import AutoTokenizer, AutoModel
from typing import Union, List, Optional
from .base import MoleculeEmbedder


class ChemBERTaEmbedder(nn.Module, MoleculeEmbedder):
    """
    ChemBERTa/ChemBERTa-2 based molecule embedder.

    Uses transformer models trained on SMILES strings.
    Supports both frozen and trainable modes for end-to-end learning.

    Args:
        model_name: Model name or HuggingFace path. Options:
            - 'chemberta2-mlm' (recommended): ChemBERTa-2 77M MLM
            - 'chemberta2-mtr': ChemBERTa-2 77M MTR
            - 'chemberta': Original ChemBERTa-zinc-base-v1
            - 'chemberta-large': Original ChemBERTa-zinc-large-v1
            - Or any HuggingFace model path
        pooling: Pooling strategy ('mean', 'cls', 'max')
        device: Device to run on ('cuda' or 'cpu')
        batch_size: Batch size for encoding multiple molecules
        trainable: Whether transformer parameters should be trainable (default: False)
                  If True, gradients will backpropagate through the transformer.
                  If False, transformer is frozen (inference only).
    """

    DEFAULT_MODELS = {
        # ChemBERTa-2 models (recommended - 77M params, trained on 77M compounds)
        'chemberta2-mlm': 'DeepChem/ChemBERTa-77M-MLM',
        'chemberta2-mtr': 'DeepChem/ChemBERTa-77M-MTR',
        'chemberta2': 'DeepChem/ChemBERTa-77M-MLM',  # Alias for MLM (recommended)
        # Original ChemBERTa models
        'chemberta': 'seyonec/ChemBERTa-zinc-base-v1',
        'chemberta-large': 'seyonec/ChemBERTa-zinc-large-v1',
        # Other models
        'molbert': 'Danhup/MolBERT',
    }

    def __init__(
        self,
        model_name: str = 'chemberta',
        pooling: str = 'mean',
        device: Optional[str] = None,
        batch_size: int = 32,
        trainable: bool = False
    ):
        super().__init__()  # Initialize nn.Module

        # Resolve model name
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]

        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size
        self.trainable = trainable

        # Auto-detect device
        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Load model and tokenizer
        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading {model_name} ({trainable_str})...")
        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            self.model = AutoModel.from_pretrained(model_name).to(self.device)

        # Control trainability
        if self.trainable:
            self.model.train()
            print(f"  → Transformer parameters are TRAINABLE (gradients will backpropagate)")
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
            print(f"  → Transformer parameters are FROZEN (no gradient updates)")

        # Cache embedding dimension
        self._embedding_dim = self.model.config.hidden_size

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
            # Convert numpy array to list if needed
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
        else:
            return embeddings

    def _encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES."""
        # Tokenize
        inputs = self.tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Forward pass (no gradients)
        with torch.no_grad():
            outputs = self.model(**inputs)

        # Pool hidden states
        if self.pooling == 'mean':
            # Mean pool over sequence length (ignoring padding)
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.pooling == 'cls':
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'max':
            # Max pool over sequence length
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")

        return embeddings.cpu().numpy()

    def _pool_embeddings(self, outputs, attention_mask) -> torch.Tensor:
        """Apply pooling strategy to transformer outputs."""
        if self.pooling == 'mean':
            # Mean pool over sequence length (ignoring padding)
            mask = attention_mask.unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * mask).sum(1) / mask.sum(1)
        elif self.pooling == 'cls':
            # Use [CLS] token embedding
            embeddings = outputs.last_hidden_state[:, 0, :]
        elif self.pooling == 'max':
            # Max pool over sequence length
            embeddings = outputs.last_hidden_state.max(dim=1)[0]
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")
        return embeddings

    def encode_trainable(self, smiles: Union[str, List[str]]) -> torch.Tensor:
        """
        Encode molecule(s) to embedding tensors WITH gradient tracking.

        This method is designed for end-to-end training where gradients need to
        flow back through the transformer. Unlike encode(), this:
        - Returns PyTorch tensors (not numpy arrays)
        - Does NOT use torch.no_grad()
        - Keeps the model in train() mode (if trainable=True)

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            torch.Tensor of shape [batch_size, embedding_dim] with gradient tracking
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = list(smiles)

        # Tokenize
        inputs = self.tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Forward pass - NO torch.no_grad() to allow gradient flow
        if self.trainable:
            self.model.train()
        else:
            self.model.eval()

        outputs = self.model(**inputs)
        embeddings = self._pool_embeddings(outputs, inputs['attention_mask'])

        return embeddings

    def freeze(self):
        """
        Freeze transformer parameters (stop gradient updates).
        """
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.trainable = False
        print("ChemBERTa transformer frozen (no gradient updates)")

    def unfreeze(self):
        """
        Unfreeze transformer parameters (enable gradient updates).
        """
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self.trainable = True
        print("ChemBERTa transformer unfrozen (gradients will backpropagate)")

    def get_encoder_parameters(self):
        """
        Get trainable encoder parameters for optimizer.

        Returns:
            List of parameters that should be optimized with encoder learning rate.
        """
        if self.trainable:
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
        return f"chemberta_{model_short}_{self.pooling}{trainable_suffix}"


# Convenience constructors
def chemberta_embedder(pooling: str = 'mean', device: Optional[str] = None) -> ChemBERTaEmbedder:
    """
    Create ChemBERTa base embedder.

    Args:
        pooling: Pooling strategy ('mean', 'cls', 'max')
        device: Device to run on
    """
    return ChemBERTaEmbedder(model_name='chemberta', pooling=pooling, device=device)


def chemberta_large_embedder(pooling: str = 'mean', device: Optional[str] = None) -> ChemBERTaEmbedder:
    """Create ChemBERTa large embedder (better quality, slower)."""
    return ChemBERTaEmbedder(model_name='chemberta-large', pooling=pooling, device=device)


def chemberta2_embedder(
    pooling: str = 'mean',
    device: Optional[str] = None,
    trainable: bool = False,
    variant: str = 'mlm'
) -> ChemBERTaEmbedder:
    """
    Create ChemBERTa-2 embedder (recommended - 77M params, trained on 77M compounds).

    Args:
        pooling: Pooling strategy ('mean', 'cls', 'max')
        device: Device to run on
        trainable: Whether to enable gradient updates
        variant: Model variant - 'mlm' (Masked LM, recommended) or 'mtr' (Multi-Task Regression)

    Returns:
        ChemBERTaEmbedder configured for ChemBERTa-2
    """
    model_name = f'chemberta2-{variant}'
    return ChemBERTaEmbedder(
        model_name=model_name,
        pooling=pooling,
        device=device,
        trainable=trainable
    )
