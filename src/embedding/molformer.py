"""
MoLFormer-XL molecule embeddings.

Uses IBM's MoLFormer-XL transformer with linear attention,
pretrained on 1.1B molecules (ZINC + PubChem).

Linear attention gives O(n) complexity — faster on CPU than standard transformers.

References:
- MoLFormer: https://arxiv.org/abs/2106.09553
- HuggingFace: https://huggingface.co/ibm/MoLFormer-XL-both-10pct

Compatibility note:
    The HuggingFace-hosted MoLFormer code (configuration_molformer.py and
    modeling_molformer.py) was written for transformers ~4.30 and uses APIs
    removed in transformers >= 4.40 / 5.0:
      1. `from transformers.onnx import OnnxConfig` -- module removed,
         replaced by the `optimum` package.
      2. `from transformers.pytorch_utils import find_pruneable_heads_and_indices`
         -- function removed in transformers 5.0.
      3. `PreTrainedModel.get_head_mask` -- method removed in transformers 5.0.
    We monkey-patch all three before loading to avoid errors.
"""

import sys
import types
import numpy as np
import torch
import torch.nn as nn
from typing import Union, List, Optional
from .base import MoleculeEmbedder


def _patch_transformers_for_molformer():
    """Install compatibility shims for MoLFormer's remote code.

    The IBM-hosted model files import two things that no longer exist in
    transformers >= 5.0.  We create minimal stubs so the import succeeds
    without downgrading the library.

    Patches applied (idempotent):
      1. ``transformers.onnx.OnnxConfig`` -- empty base class stub.
      2. ``transformers.pytorch_utils.find_pruneable_heads_and_indices``
         -- re-implementation of the removed helper (simple set math).
      3. ``PreTrainedModel.get_head_mask`` -- re-implementation of the
         removed method that converts head_mask to per-layer format.
    """
    # --- Patch 1: transformers.onnx.OnnxConfig ---
    if 'transformers.onnx' not in sys.modules:
        class _OnnxConfig:
            """Minimal stub replacing the removed transformers.onnx.OnnxConfig."""
            pass

        onnx_mod = types.ModuleType('transformers.onnx')
        onnx_mod.OnnxConfig = _OnnxConfig
        sys.modules['transformers.onnx'] = onnx_mod

    # --- Patch 2: find_pruneable_heads_and_indices ---
    from transformers import pytorch_utils
    if not hasattr(pytorch_utils, 'find_pruneable_heads_and_indices'):
        def find_pruneable_heads_and_indices(heads, n_heads, head_size, already_pruned_heads):
            """Find heads and their indices taking ``already_pruned_heads`` into account.

            This is a re-implementation of the function removed in transformers 5.0.
            """
            mask = torch.ones(n_heads, head_size)
            heads = set(heads) - already_pruned_heads
            for head in heads:
                head = head - sum(1 if h < head else 0 for h in already_pruned_heads)
                mask[head] = 0
            index = torch.arange(len(mask.view(-1)))[mask.view(-1).bool()]
            return heads, index

        pytorch_utils.find_pruneable_heads_and_indices = find_pruneable_heads_and_indices

    # --- Patch 3: PreTrainedModel.get_head_mask ---
    from transformers.modeling_utils import PreTrainedModel
    if not hasattr(PreTrainedModel, 'get_head_mask'):
        @staticmethod
        def get_head_mask(head_mask, num_hidden_layers, is_attention_chunked=False):
            """Prepare the head mask if needed.

            Re-implementation of the method removed in transformers 5.0.

            Args:
                head_mask: Optional mask with shape ``[num_heads]`` or
                    ``[num_hidden_layers x num_heads]``.
                num_hidden_layers: Number of hidden layers in the model.
                is_attention_chunked: Unused, kept for API compat.

            Returns:
                List of ``None`` (no masking) or a list of per-layer mask
                tensors of shape ``[1, num_heads, 1, 1]``.
            """
            if head_mask is not None:
                if head_mask.dim() == 1:
                    head_mask = head_mask.unsqueeze(0).unsqueeze(0).unsqueeze(-1).unsqueeze(-1)
                    head_mask = head_mask.expand(num_hidden_layers, -1, -1, -1, -1)
                elif head_mask.dim() == 2:
                    head_mask = head_mask.unsqueeze(1).unsqueeze(-1).unsqueeze(-1)
                # head_mask shape: [num_hidden_layers x batch x num_heads x seq_length x seq_length]
                # or we just make it a list of None
                head_mask = [head_mask[i] for i in range(num_hidden_layers)]
            else:
                head_mask = [None] * num_hidden_layers
            return head_mask

        PreTrainedModel.get_head_mask = get_head_mask


class MoLFormerEmbedder(nn.Module, MoleculeEmbedder):
    """
    MoLFormer-XL molecule embedder.

    Uses linear-attention transformer pretrained on 1.1B molecules.
    Produces 768-dim embeddings from SMILES strings.

    Args:
        model_name: HuggingFace model path
        pooling: Pooling strategy ('mean', 'cls')
        device: Device to run on ('cpu', 'cuda', 'mps')
        batch_size: Batch size for encoding
    """

    DEFAULT_MODEL = 'ibm/MoLFormer-XL-both-10pct'

    def __init__(
        self,
        model_name: Optional[str] = None,
        pooling: str = 'mean',
        device: Optional[str] = None,
        batch_size: int = 32,
    ):
        super().__init__()

        model_name = model_name or self.DEFAULT_MODEL
        self.model_name = model_name
        self.pooling = pooling
        self.batch_size = batch_size

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        # Apply compatibility patches before loading the model
        _patch_transformers_for_molformer()

        print(f"Loading MoLFormer ({model_name})...")
        import warnings
        from transformers import AutoTokenizer, AutoModel
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=DeprecationWarning)
            warnings.filterwarnings("ignore", category=FutureWarning)
            self.tokenizer = AutoTokenizer.from_pretrained(
                model_name, trust_remote_code=True
            )
            self.model = AutoModel.from_pretrained(
                model_name, trust_remote_code=True
            ).to(self.device)

        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False

        # Fix rotary embeddings: inv_freq is a non-persistent buffer that isn't
        # saved in the checkpoint, so it's uninitialized after loading.
        # Recompute it from the class defaults.
        self._fix_rotary_embeddings()

        self._embedding_dim = self.model.config.hidden_size
        print(f"  → MoLFormer loaded: {self._embedding_dim}-dim embeddings, frozen")

    def _fix_rotary_embeddings(self):
        """Recompute rotary embedding buffers that were lost during loading.

        MoLFormer registers inv_freq with persistent=False, so it's not saved
        in the state dict. After loading, inv_freq is all zeros, producing NaN
        in cos/sin caches. We recompute inv_freq and rebuild the cache.
        """
        for layer in self.model.encoder.layer:
            rot = layer.attention.self.rotary_embeddings
            dim = rot.dim
            base = rot.base
            inv_freq = 1.0 / (base ** (torch.arange(0, dim, 2).float() / dim))
            rot.inv_freq = inv_freq.to(rot.inv_freq.device)
            rot._set_cos_sin_cache(
                seq_len=rot.max_seq_len_cached,
                device=rot.inv_freq.device,
                dtype=torch.get_default_dtype(),
            )

    def encode(self, smiles: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """Encode molecule(s) to embedding vector(s)."""
        if isinstance(smiles, str):
            smiles = [smiles]
            return_single = True
        else:
            if isinstance(smiles, np.ndarray):
                smiles = smiles.tolist()
            return_single = False

        from tqdm.auto import tqdm

        all_embeddings = []
        with tqdm(total=len(smiles), desc="MoLFormer encoding", unit="mol",
                  disable=not show_progress) as pbar:
            for i in range(0, len(smiles), self.batch_size):
                batch = smiles[i:i + self.batch_size]
                batch_emb = self._encode_batch(batch)
                all_embeddings.append(batch_emb)
                pbar.update(len(batch))

        embeddings = np.vstack(all_embeddings)
        return embeddings[0] if return_single else embeddings

    def _encode_batch(self, smiles_list: List[str]) -> np.ndarray:
        """Encode a batch of SMILES."""
        inputs = self.tokenizer(
            smiles_list,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)

        with torch.no_grad():
            outputs = self.model(**inputs)

        if self.pooling == 'mean':
            attention_mask = inputs['attention_mask'].unsqueeze(-1)
            embeddings = (outputs.last_hidden_state * attention_mask).sum(1) / attention_mask.sum(1)
        elif self.pooling == 'cls':
            embeddings = outputs.last_hidden_state[:, 0, :]
        else:
            raise ValueError(f"Invalid pooling: {self.pooling}")

        return embeddings.cpu().numpy()

    @property
    def embedding_dim(self) -> int:
        return self._embedding_dim

    @property
    def name(self) -> str:
        return f"molformer_{self.pooling}"
