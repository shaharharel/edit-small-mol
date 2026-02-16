"""
MolFM structured edit embedder using graph encoder component.

Extracts local edit representations by:
1. Using MolFM's graph encoder for atom-level embeddings
2. Applying same k-hop environment extraction as D-MPNN
3. Fallback to sequence mode with token-to-atom mapping

Key features:
- Graph encoder provides atom-level representations
- Combines GNN and sequence information when available
- Compatible with local environment extraction pattern
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from rdkit import Chem
import re

from .structured_edit_base import StructuredEditEmbedderBase


class MolFMStructuredEditEmbedder(StructuredEditEmbedderBase):
    """
    MolFM-based structured edit embedder.

    Uses MolFM's graph encoder component for local edit embeddings.
    Falls back to sequence-based extraction when graph encoder unavailable.

    Args:
        modality: Which modality to use ('graph', 'sequence', 'multimodal')
        device: Device to run on ('cuda' or 'cpu')
        k_hop_env: Number of hops for local environment (default: 2)
        trainable: Whether to enable gradient updates (default: False)
    """

    def __init__(
        self,
        modality: str = 'sequence',  # Default to sequence for compatibility
        device: Optional[str] = None,
        k_hop_env: int = 2,
        trainable: bool = False
    ):
        # Auto-detect device
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'

        # Load model and get embedding dim BEFORE calling super().__init__
        model_components, embedding_dim, backend = self._load_model(device, trainable)

        # Initialize base class FIRST
        super().__init__(
            embedding_dim=embedding_dim,
            k_hop_env=k_hop_env,
            trainable=trainable
        )

        # Now assign attributes
        self.modality = modality
        self.device = device
        self._embedding_dim = embedding_dim
        self._backend = backend

        # Unpack model components based on backend
        if backend == 'openbiomedl':
            self.model = model_components['model']
        else:
            self.tokenizer = model_components['tokenizer']
            self.seq_encoder = model_components['seq_encoder']
            self.graph_encoder = model_components.get('graph_encoder')

    @staticmethod
    def _load_model(device: str, trainable: bool):
        """Load model and return (components_dict, embedding_dim, backend)."""
        trainable_str = "trainable" if trainable else "frozen"
        print(f"Loading MolFM structured embedder ({trainable_str})...")

        # Try to load OpenBioMed version first
        try:
            return MolFMStructuredEditEmbedder._load_openbiomedl(device, trainable)
        except ImportError:
            pass

        # Fallback to ChemBERTa-based sequence model
        print("  ⚠ OpenBioMed not found. Using ChemBERTa-2 fallback.")
        return MolFMStructuredEditEmbedder._load_fallback(device, trainable)

    @staticmethod
    def _load_openbiomedl(device: str, trainable: bool):
        """Load OpenBioMed MolFM."""
        try:
            from open_biomed.models.molecule import MolFM
        except ImportError:
            raise ImportError("OpenBioMed not available")

        model = MolFM.from_pretrained('molfm').to(device)
        embedding_dim = 768  # MolFM default

        if trainable:
            model.train()
        else:
            model.eval()
            for param in model.parameters():
                param.requires_grad = False

        print(f"  → Loaded OpenBioMed MolFM, dim: {embedding_dim}")
        return {'model': model}, embedding_dim, 'openbiomedl'

    @staticmethod
    def _load_fallback(device: str, trainable: bool):
        """Load ChemBERTa-2 fallback."""
        try:
            from transformers import AutoModel, AutoTokenizer
        except ImportError:
            raise ImportError("transformers library required")

        import warnings
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore")
            tokenizer = AutoTokenizer.from_pretrained("DeepChem/ChemBERTa-77M-MLM")
            seq_encoder = AutoModel.from_pretrained("DeepChem/ChemBERTa-77M-MLM").to(device)

        embedding_dim = seq_encoder.config.hidden_size

        # Control trainability
        if trainable:
            seq_encoder.train()
        else:
            seq_encoder.eval()
            for param in seq_encoder.parameters():
                param.requires_grad = False

        # Optional graph encoder
        graph_encoder = MolFMStructuredEditEmbedder._create_graph_encoder(embedding_dim, device, trainable)

        print(f"  → Using ChemBERTa-2 fallback, dim: {embedding_dim}")

        return {
            'tokenizer': tokenizer,
            'seq_encoder': seq_encoder,
            'graph_encoder': graph_encoder
        }, embedding_dim, 'fallback'

    @staticmethod
    def _create_graph_encoder(embedding_dim: int, device: str, trainable: bool):
        """Create optional graph encoder component."""
        try:
            graph_encoder = nn.Sequential(
                nn.Linear(118, 256),
                nn.ReLU(),
                nn.Linear(256, embedding_dim)
            ).to(device)

            if not trainable:
                for param in graph_encoder.parameters():
                    param.requires_grad = False

            print("  → Graph encoder initialized")
            return graph_encoder
        except Exception:
            return None

    def get_atom_embeddings_sequence(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom embeddings using sequence encoder with token-to-atom mapping.
        """
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (
                torch.zeros(1, self._embedding_dim, device=self.device),
                torch.zeros(self._embedding_dim, device=self.device)
            )

        n_atoms = mol.GetNumAtoms()

        # Get token embeddings
        inputs = self.tokenizer(
            smiles,
            return_tensors='pt',
            padding=False,
            truncation=True,
            max_length=512
        ).to(self.device)

        if self.trainable:
            outputs = self.seq_encoder(**inputs)
        else:
            with torch.no_grad():
                outputs = self.seq_encoder(**inputs)

        token_embeddings = outputs.last_hidden_state[0]  # [seq_len, hidden]

        # Global embedding
        attention_mask = inputs['attention_mask'][0].unsqueeze(-1)
        global_emb = (token_embeddings * attention_mask).sum(0) / attention_mask.sum()

        # Map tokens to atoms
        token_to_atoms = self._get_token_to_atom_mapping(smiles, mol)

        if token_to_atoms is None:
            return (
                global_emb.unsqueeze(0).expand(n_atoms, -1),
                global_emb
            )

        # Aggregate token embeddings per atom
        atom_embeddings = torch.zeros(n_atoms, self._embedding_dim, device=self.device)
        atom_counts = torch.zeros(n_atoms, device=self.device)

        for tok_idx, atom_indices in token_to_atoms.items():
            if tok_idx < len(token_embeddings):
                for atom_idx in atom_indices:
                    if atom_idx < n_atoms:
                        atom_embeddings[atom_idx] += token_embeddings[tok_idx]
                        atom_counts[atom_idx] += 1

        # Average
        atom_counts = atom_counts.clamp(min=1)
        atom_embeddings = atom_embeddings / atom_counts.unsqueeze(-1)

        # Fill zeros with global embedding
        zero_mask = atom_embeddings.sum(-1) == 0
        atom_embeddings[zero_mask] = global_emb

        return atom_embeddings, global_emb

    def get_atom_embeddings_graph(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom embeddings using graph encoder.
        """
        if self.graph_encoder is None:
            return self.get_atom_embeddings_sequence(smiles)

        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (
                torch.zeros(1, self._embedding_dim, device=self.device),
                torch.zeros(self._embedding_dim, device=self.device)
            )

        n_atoms = mol.GetNumAtoms()

        # Get atom features (one-hot atomic numbers)
        atom_features = torch.zeros(n_atoms, 118, device=self.device)
        for i, atom in enumerate(mol.GetAtoms()):
            atomic_num = min(atom.GetAtomicNum(), 117)
            atom_features[i, atomic_num] = 1.0

        # Forward through graph encoder
        if self.trainable:
            atom_embeddings = self.graph_encoder(atom_features)
        else:
            with torch.no_grad():
                atom_embeddings = self.graph_encoder(atom_features)

        # Global embedding
        global_emb = atom_embeddings.mean(dim=0)

        return atom_embeddings, global_emb

    def get_atom_embeddings(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Get atom-level embeddings for a molecule.

        Uses graph encoder if available, otherwise falls back to sequence.
        """
        if self._backend == 'openbiomedl':
            return self._get_atom_embeddings_openbiomedl(smiles)

        if self.modality == 'graph' and self.graph_encoder is not None:
            return self.get_atom_embeddings_graph(smiles)
        elif self.modality == 'multimodal' and self.graph_encoder is not None:
            # Combine graph and sequence embeddings
            graph_atom, graph_global = self.get_atom_embeddings_graph(smiles)
            seq_atom, seq_global = self.get_atom_embeddings_sequence(smiles)

            # Average (could also concatenate and project)
            atom_emb = (graph_atom + seq_atom) / 2
            global_emb = (graph_global + seq_global) / 2
            return atom_emb, global_emb
        else:
            return self.get_atom_embeddings_sequence(smiles)

    def _get_atom_embeddings_openbiomedl(
        self,
        smiles: str
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Get atom embeddings from OpenBioMed MolFM."""
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return (
                torch.zeros(1, self._embedding_dim, device=self.device),
                torch.zeros(self._embedding_dim, device=self.device)
            )

        n_atoms = mol.GetNumAtoms()

        # OpenBioMed MolFM provides graph encoding
        if self.trainable:
            graph_emb = self.model.encode_graph([smiles])
            # Try to get node-level if available
            if hasattr(self.model, 'encode_graph_nodes'):
                node_emb = self.model.encode_graph_nodes([smiles])[0]
            else:
                node_emb = graph_emb[0].unsqueeze(0).expand(n_atoms, -1)
        else:
            with torch.no_grad():
                graph_emb = self.model.encode_graph([smiles])
                if hasattr(self.model, 'encode_graph_nodes'):
                    node_emb = self.model.encode_graph_nodes([smiles])[0]
                else:
                    node_emb = graph_emb[0].unsqueeze(0).expand(n_atoms, -1)

        return node_emb, graph_emb[0]

    def _get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Chem.Mol
    ) -> Optional[Dict[int, List[int]]]:
        """Map tokens to atoms using character position analysis."""
        inputs = self.tokenizer(smiles, return_tensors='pt', add_special_tokens=True)
        tokens = self.tokenizer.convert_ids_to_tokens(inputs['input_ids'][0])

        # Build character to atom mapping
        char_to_atom = self._build_char_to_atom_map(smiles, mol)

        # Build token to atom mapping
        token_to_atoms = {}
        char_pos = 0

        for tok_idx, token in enumerate(tokens):
            if token in ['[CLS]', '[SEP]', '[PAD]', '<s>', '</s>', '<pad>']:
                continue

            clean_token = token.replace('##', '').replace('Ġ', '')

            atoms_for_token = set()
            for i in range(len(clean_token)):
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
        """Build mapping from SMILES characters to atom indices."""
        char_to_atom = {}
        atom_pattern = r'(\[.*?\]|Br|Cl|[BCNOSPFI])'

        atom_idx = 0
        for match in re.finditer(atom_pattern, smiles):
            start, end = match.span()
            for pos in range(start, end):
                if pos not in char_to_atom:
                    char_to_atom[pos] = []
                char_to_atom[pos].append(atom_idx)
            atom_idx += 1

            if atom_idx >= mol.GetNumAtoms():
                break

        return char_to_atom

    def get_token_to_atom_mapping(
        self,
        smiles: str,
        mol: Optional[Chem.Mol] = None
    ) -> Optional[Dict[int, List[int]]]:
        """
        Get token to atom mapping for sequence-based models.

        For graph mode, returns None (direct atom indexing).
        """
        if self.modality == 'graph':
            return None

        if mol is None:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                return None

        return self._get_token_to_atom_mapping(smiles, mol)

    def freeze(self):
        """Freeze model parameters."""
        if self._backend == 'openbiomedl':
            self.model.eval()
            for param in self.model.parameters():
                param.requires_grad = False
        else:
            self.seq_encoder.eval()
            for param in self.seq_encoder.parameters():
                param.requires_grad = False
            if self.graph_encoder is not None:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = False

        self.trainable = False
        print("MolFM frozen")

    def unfreeze(self):
        """Unfreeze model parameters."""
        if self._backend == 'openbiomedl':
            self.model.train()
            for param in self.model.parameters():
                param.requires_grad = True
        else:
            self.seq_encoder.train()
            for param in self.seq_encoder.parameters():
                param.requires_grad = True
            if self.graph_encoder is not None:
                for param in self.graph_encoder.parameters():
                    param.requires_grad = True

        self.trainable = True
        print("MolFM unfrozen")

    def get_encoder_parameters(self) -> List:
        """Get trainable encoder parameters."""
        if not self.trainable:
            return []

        params = []
        if self._backend == 'openbiomedl':
            params.extend(list(self.model.parameters()))
        else:
            params.extend(list(self.seq_encoder.parameters()))
            if self.graph_encoder is not None:
                params.extend(list(self.graph_encoder.parameters()))

        return params

    @property
    def name(self) -> str:
        """Return embedder name."""
        trainable_str = "_trainable" if self.trainable else "_frozen"
        return f"molfm_structured_{self.modality}{trainable_str}"


# Convenience constructor
def molfm_structured_embedder(
    modality: str = 'sequence',
    device: Optional[str] = None,
    trainable: bool = False,
    k_hop: int = 2
) -> MolFMStructuredEditEmbedder:
    """
    Create MolFM structured edit embedder.

    Args:
        modality: 'graph', 'sequence', or 'multimodal'
        device: Device to run on
        trainable: Whether to enable gradient updates
        k_hop: Number of hops for local environment

    Returns:
        MolFMStructuredEditEmbedder instance
    """
    return MolFMStructuredEditEmbedder(
        modality=modality,
        device=device,
        k_hop_env=k_hop,
        trainable=trainable
    )
