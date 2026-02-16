"""
ChemBERTa-2 structured edit embedder with token-level attention.

Extracts local edit representations by:
1. Tokenizing SMILES and mapping tokens to atom indices
2. Using attention to focus on tokens corresponding to edit sites
3. Computing weighted embeddings for fragments and local environment

Key features:
- Token-to-atom mapping via SMILES character analysis
- Attention-weighted pooling for edit site tokens
- Support for substructure tokenization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem
from transformers import AutoTokenizer, AutoModel
import re

from .structured_edit_base import StructuredEditEmbedderBase


class ChemBERTaStructuredEditEmbedder(StructuredEditEmbedderBase):
    """
    ChemBERTa-2 based structured edit embedder.

    Uses token-level embeddings and attention to extract local representations
    around edit sites. Maps SMILES tokens to atom indices for precise extraction.

    Args:
        model_name: ChemBERTa model name (default: 'chemberta2-mlm')
        device: Device to run on ('cuda' or 'cpu')
        k_hop_env: Number of hops for local environment (default: 2)
        trainable: Whether to enable gradient updates (default: False)
        use_attention_weights: Use attention for token aggregation (default: True)
    """

    DEFAULT_MODELS = {
        'chemberta2-mlm': 'DeepChem/ChemBERTa-77M-MLM',
        'chemberta2-mtr': 'DeepChem/ChemBERTa-77M-MTR',
        'chemberta2': 'DeepChem/ChemBERTa-77M-MLM',
        'chemberta': 'seyonec/ChemBERTa-zinc-base-v1',
    }

    def __init__(
        self,
        model_name: str = 'chemberta2-mlm',
        device: Optional[str] = None,
        k_hop_env: int = 2,
        trainable: bool = False,
        use_attention_weights: bool = True
    ):
        # Resolve model name
        if model_name in self.DEFAULT_MODELS:
            model_name = self.DEFAULT_MODELS[model_name]

        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model and tokenizer
        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading ChemBERTa structured embedder ({trainable_str})...")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = AutoTokenizer.from_pretrained(model_name)
            model = AutoModel.from_pretrained(model_name).to(device)

        embedding_dim = model.config.hidden_size

        # Initialize base class FIRST (before any nn.Module attribute assignments)
        super().__init__(
            embedding_dim=embedding_dim,
            k_hop_env=k_hop_env,
            trainable=trainable
        )

        # Now assign attributes
        self.model_name = model_name
        self.use_attention_weights = use_attention_weights
        self.device = device
        self.tokenizer = tokenizer
        self.model = model

        # Control trainability
        if self.trainable:
            self.model.train()
        else:
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False

        # Attention layer for weighted pooling (learnable even when model is frozen)
        self.attention_layer = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim // 4),
            nn.Tanh(),
            nn.Linear(embedding_dim // 4, 1)
        ).to(device)

        print(f"  → Embedding dimension: {embedding_dim}")

    def get_token_embeddings(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor, List[str]]:
        """
        Get token-level embeddings for a SMILES string.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of:
            - token_embeddings: [n_tokens, embedding_dim]
            - global_embedding: [embedding_dim] (mean pooled)
            - tokens: List of token strings
        """
        # Tokenize
        inputs = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512
        ).to(self.device)

        # Get tokens for mapping
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Forward pass
        if self.trainable:
            outputs = self.model(**inputs, output_attentions=True)
        else:
            with torch.no_grad():
                outputs = self.model(**inputs, output_attentions=True)

        # Token embeddings (excluding [CLS] and [SEP])
        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]

        # Global embedding via mean pooling
        attention_mask = inputs['attention_mask'][0].unsqueeze(-1)
        global_embedding = (token_embeddings * attention_mask).sum(0) / attention_mask.sum()

        return token_embeddings, global_embedding, tokens

    def get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Optional[Chem.Mol] = None
    ) -> Optional[Dict[int, List[int]]]:
        """
        Map SMILES tokens to atom indices.

        This is non-trivial because:
        - Tokens may span multiple characters
        - Some characters are not atoms (bonds, brackets, etc.)
        - Tokenizer may use subword units

        Strategy:
        1. Track character position in SMILES
        2. Map characters to atoms using RDKit atom mapping
        3. Aggregate for each token

        Args:
            smiles: SMILES string
            mol: Optional pre-parsed RDKit mol

        Returns:
            Dict mapping token_idx → list of atom indices
        """
        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

        # Get tokens
        inputs = self.tokenizer(smiles, return_tensors='pt', add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Build character to atom mapping
        # We need to parse SMILES and track which character corresponds to which atom
        char_to_atom = self._build_char_to_atom_map(smiles, mol)

        # Build token to atom mapping
        token_to_atoms = {}
        char_pos = 0

        for tok_idx, token in enumerate(tokens):
            # Skip special tokens
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                continue

            # Clean token (remove ## prefix for subword tokens)
            clean_token = token.replace('##', '').replace('Ġ', '')

            # Find atoms for this token
            atoms_for_token = set()
            for i, char in enumerate(clean_token):
                if char_pos + i in char_to_atom:
                    atoms_for_token.update(char_to_atom[char_pos + i])

            if atoms_for_token:
                token_to_atoms[tok_idx] = sorted(list(atoms_for_token))

            char_pos += len(clean_token)

        return token_to_atoms

    def _build_char_to_atom_map(
        self,
        smiles: str,
        mol: Chem.Mol
    ) -> Dict[int, List[int]]:
        """
        Build mapping from SMILES character positions to atom indices.

        Uses atom map numbers or heuristic matching.
        """
        # Simple approach: match atom symbols to positions
        char_to_atom = {}

        # Pattern to find atom symbols in SMILES
        # Matches: C, N, O, S, P, F, Cl, Br, I, [anything in brackets]
        atom_pattern = r'(\[.*?\]|Br|Cl|[BCNOSPFI])'

        atom_idx = 0
        for match in re.finditer(atom_pattern, smiles):
            start, end = match.span()
            for pos in range(start, end):
                if pos not in char_to_atom:
                    char_to_atom[pos] = []
                char_to_atom[pos].append(atom_idx)
            atom_idx += 1

            # Don't exceed actual atom count
            if atom_idx >= mol.GetNumAtoms():
                break

        return char_to_atom

    def get_atom_embeddings(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level embeddings by aggregating token embeddings.

        For sequence models, we map tokens to atoms and aggregate.

        Args:
            smiles: SMILES string

        Returns:
            Tuple of:
            - atom_embeddings: [n_atoms, embedding_dim]
            - global_embedding: [embedding_dim]
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            # Return zeros for invalid molecules
            return (
                torch.zeros(1, self.embedding_dim, device=self.device),
                torch.zeros(self.embedding_dim, device=self.device)
            )

        n_atoms = mol.GetNumAtoms()

        # Get token embeddings
        token_embs, global_emb, tokens = self.get_token_embeddings(smiles)

        # Get token to atom mapping
        token_to_atoms = self.get_token_to_atom_mapping(smiles, mol)

        if token_to_atoms is None or not token_to_atoms:
            # Fallback: use global embedding for all atoms
            return (
                global_emb.unsqueeze(0).expand(n_atoms, -1),
                global_emb
            )

        # Aggregate token embeddings per atom
        atom_embeddings = torch.zeros(n_atoms, self.embedding_dim, device=self.device)
        atom_counts = torch.zeros(n_atoms, device=self.device)

        for tok_idx, atom_indices in token_to_atoms.items():
            if tok_idx < len(token_embs):
                for atom_idx in atom_indices:
                    if atom_idx < n_atoms:
                        atom_embeddings[atom_idx] += token_embs[tok_idx]
                        atom_counts[atom_idx] += 1

        # Average (avoid division by zero)
        atom_counts = atom_counts.clamp(min=1)
        atom_embeddings = atom_embeddings / atom_counts.unsqueeze(-1)

        # Fill zeros with global embedding
        zero_mask = (atom_counts == 1) & (atom_embeddings.sum(-1) == 0)
        atom_embeddings[zero_mask] = global_emb

        return atom_embeddings, global_emb

    def get_attention_weighted_embedding(
        self,
        token_embeddings: torch.Tensor,
        token_indices: List[int],
        attention_mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Compute attention-weighted embedding over selected tokens.

        Args:
            token_embeddings: [n_tokens, embedding_dim]
            token_indices: Indices of tokens to attend over
            attention_mask: Optional mask [n_tokens]

        Returns:
            Weighted embedding [embedding_dim]
        """
        if not token_indices:
            return torch.zeros(self.embedding_dim, device=self.device)

        # Filter valid indices
        valid_indices = [i for i in token_indices if i < len(token_embeddings)]
        if not valid_indices:
            return torch.zeros(self.embedding_dim, device=self.device)

        # Extract relevant embeddings
        selected_embs = token_embeddings[valid_indices]  # [n_selected, embedding_dim]

        if not self.use_attention_weights or len(valid_indices) == 1:
            return selected_embs.mean(dim=0)

        # Compute attention scores
        scores = self.attention_layer(selected_embs).squeeze(-1)  # [n_selected]
        weights = torch.softmax(scores, dim=0)  # [n_selected]

        # Weighted sum
        return (selected_embs * weights.unsqueeze(-1)).sum(dim=0)

    def forward_with_attention(
        self,
        smiles_a: str,
        smiles_b: str,
        removed_atom_indices_a: List[int],
        added_atom_indices_b: List[int],
        attach_atom_indices_a: List[int],
        attach_atom_indices_b: Optional[List[int]] = None
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using attention-weighted token embeddings.

        This provides finer-grained control over which tokens contribute
        to fragment and environment embeddings.

        Returns same dict as forward() but uses attention mechanism.
        """
        # Get token embeddings
        tok_embs_a, global_a, tokens_a = self.get_token_embeddings(smiles_a)
        tok_embs_b, global_b, tokens_b = self.get_token_embeddings(smiles_b)

        # Get token to atom mappings
        mol_a = Chem.MolFromSmiles(smiles_a)
        mol_b = Chem.MolFromSmiles(smiles_b)

        tok_to_atom_a = self.get_token_to_atom_mapping(smiles_a, mol_a)
        tok_to_atom_b = self.get_token_to_atom_mapping(smiles_b, mol_b)

        # Build reverse mapping: atom → tokens
        atom_to_tok_a = self._invert_mapping(tok_to_atom_a)
        atom_to_tok_b = self._invert_mapping(tok_to_atom_b)

        # Get tokens for removed atoms
        removed_tokens = []
        for atom_idx in removed_atom_indices_a:
            if atom_idx in atom_to_tok_a:
                removed_tokens.extend(atom_to_tok_a[atom_idx])
        removed_tokens = list(set(removed_tokens))

        # Get tokens for added atoms
        added_tokens = []
        for atom_idx in added_atom_indices_b:
            if atom_idx in atom_to_tok_b:
                added_tokens.extend(atom_to_tok_b[atom_idx])
        added_tokens = list(set(added_tokens))

        # Get tokens for attachment environment
        if mol_a is not None:
            env_indices_a = self.compute_k_hop_neighbors(mol_a, attach_atom_indices_a)
        else:
            env_indices_a = attach_atom_indices_a

        env_tokens_a = []
        for atom_idx in env_indices_a:
            if atom_idx in atom_to_tok_a:
                env_tokens_a.extend(atom_to_tok_a[atom_idx])
        env_tokens_a = list(set(env_tokens_a))

        # Compute embeddings with attention
        h_out = self.get_attention_weighted_embedding(tok_embs_a, removed_tokens)
        h_in = self.get_attention_weighted_embedding(tok_embs_b, added_tokens)
        h_env_a = self.get_attention_weighted_embedding(tok_embs_a, env_tokens_a)

        result = {
            'h_a_global': global_a,
            'h_b_global': global_b,
            'h_env_a': h_env_a,
            'h_out': h_out,
            'h_in': h_in,
        }

        # Optional: environment in B
        if attach_atom_indices_b and mol_b is not None:
            env_indices_b = self.compute_k_hop_neighbors(mol_b, attach_atom_indices_b)
            atom_to_tok_b_inv = self._invert_mapping(tok_to_atom_b)
            env_tokens_b = []
            for atom_idx in env_indices_b:
                if atom_idx in atom_to_tok_b_inv:
                    env_tokens_b.extend(atom_to_tok_b_inv[atom_idx])
            env_tokens_b = list(set(env_tokens_b))
            result['h_env_b'] = self.get_attention_weighted_embedding(tok_embs_b, env_tokens_b)

        return result

    def _invert_mapping(
        self,
        token_to_atoms: Optional[Dict[int, List[int]]]
    ) -> Dict[int, List[int]]:
        """Invert token→atoms mapping to atom→tokens."""
        if token_to_atoms is None:
            return {}

        atom_to_tokens = {}
        for tok_idx, atom_indices in token_to_atoms.items():
            for atom_idx in atom_indices:
                if atom_idx not in atom_to_tokens:
                    atom_to_tokens[atom_idx] = []
                atom_to_tokens[atom_idx].append(tok_idx)

        return atom_to_tokens

    def freeze(self):
        """Freeze transformer parameters (attention layer stays trainable)."""
        self.model.eval()
        for param in self.model.parameters():
            param.requires_grad = False
        self.trainable = False
        print("ChemBERTa transformer frozen")

    def unfreeze(self):
        """Unfreeze transformer parameters."""
        self.model.train()
        for param in self.model.parameters():
            param.requires_grad = True
        self.trainable = True
        print("ChemBERTa transformer unfrozen")

    def get_encoder_parameters(self) -> List:
        """Get trainable encoder parameters."""
        params = list(self.attention_layer.parameters())
        if self.trainable:
            params.extend(list(self.model.parameters()))
        return params

    @property
    def name(self) -> str:
        """Return embedder name."""
        model_short = self.model_name.split('/')[-1]
        trainable_str = "_trainable" if self.trainable else "_frozen"
        return f"chemberta_structured_{model_short}{trainable_str}"


# Convenience constructor
def chemberta2_structured_embedder(
    device: Optional[str] = None,
    trainable: bool = False,
    variant: str = 'mlm'
) -> ChemBERTaStructuredEditEmbedder:
    """
    Create ChemBERTa-2 structured edit embedder.

    Args:
        device: Device to run on
        trainable: Whether to enable gradient updates
        variant: Model variant ('mlm' or 'mtr')

    Returns:
        ChemBERTaStructuredEditEmbedder instance
    """
    model_name = f'chemberta2-{variant}'
    return ChemBERTaStructuredEditEmbedder(
        model_name=model_name,
        device=device,
        trainable=trainable
    )
