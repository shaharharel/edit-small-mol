"""
ChemProp (D-MPNN graph neural network) molecule embeddings.

Uses the official ChemProp v2 library to generate molecular representations.
Compatible with ChemProp v2.x API.
"""

import numpy as np
import torch.nn as nn
from typing import Union, List, Optional, Dict
from .base import MoleculeEmbedder


class ChemPropEmbedder(nn.Module, MoleculeEmbedder):
    """
    ChemProp D-MPNN molecule embedder (v2.x compatible).

    Uses ChemProp v2 library for molecular representations. Without a trained model,
    uses Morgan fingerprints from ChemProp's featurizer.

    Installation:
        pip install chemprop

    Args:
        model_path: Path to trained ChemProp v2 model checkpoint (optional)
                   If None, uses featurization based on `featurizer_type`
        batch_size: Batch size for encoding
        featurizer_type: Type of featurizer to use when no model provided:
                        - 'morgan': Morgan binary fingerprints (default, fast)
                        - 'rdkit2d': RDKit 2D descriptors (217 features, interpretable)
                        - 'graph': D-MPNN graph neural network (300-dim, learned structure)
        morgan_radius: Morgan fingerprint radius (ChemProp default: 2)
        morgan_length: Morgan fingerprint length (ChemProp default: 2048)
        include_chirality: Include chirality in Morgan fingerprints (default: True)
        device: Device for graph neural network ('cpu' or 'cuda')
        trainable: Whether GNN parameters should be trainable (default: False)
                  Only applies to featurizer_type='graph'. If True, gradients will
                  backpropagate through the GNN during training. If False, GNN is frozen.

    """

    def __init__(
        self,
        model_path: Optional[str] = None,
        batch_size: int = 50,
        featurizer_type: str = 'morgan',
        morgan_radius: int = 2,
        morgan_length: int = 2048,  # ChemProp default
        include_chirality: bool = True,
        device: str = 'cpu',  # D-MPNN embedder runs on CPU (embeddings are cached anyway)
        trainable: bool = False  # Whether to make GNN parameters trainable (only for graph featurizer)
    ):
        super().__init__()  # Initialize nn.Module
        self.model_path = model_path
        self.batch_size = batch_size
        self.featurizer_type = featurizer_type
        self.morgan_radius = morgan_radius
        self.morgan_length = morgan_length
        self.include_chirality = include_chirality
        self.device = device
        self.trainable = trainable

        # Try to import ChemProp v2
        try:
            import chemprop
            self.chemprop = chemprop
            self._chemprop_available = True
        except ImportError:
            raise ImportError(
                "ChemProp is not installed. Install with: pip install chemprop"
            )

        # Initialize
        if model_path:
            self._init_with_model()
        else:
            self._init_featurization_only()

    def _init_with_model(self):
        """Initialize with a trained ChemProp v2 model."""
        from chemprop.models import load_model

        print(f"Loading ChemProp v2 model from {self.model_path}")
        self.model = load_model(self.model_path)
        self.model.eval()

        # Get embedding dimension from model
        # In v2, we need to inspect the model architecture
        self._embedding_dim = 300  # Default, will be updated after first encode
        self._use_model = True
        self.featurizer = None

    def _init_featurization_only(self):
        """Initialize featurization without trained model."""
        self.model = None
        self._use_model = False

        if self.featurizer_type == 'morgan':
            from chemprop.featurizers import MorganBinaryFeaturizer

            print(f"Using ChemProp v2 Morgan fingerprints (r={self.morgan_radius}, "
                  f"len={self.morgan_length}, chirality={self.include_chirality})")

            self.featurizer = MorganBinaryFeaturizer(
                radius=self.morgan_radius,
                length=self.morgan_length,
                include_chirality=self.include_chirality
            )
            self._embedding_dim = self.morgan_length

        elif self.featurizer_type == 'rdkit2d':
            from chemprop.featurizers import RDKit2DFeaturizer

            print("Using ChemProp v2 RDKit 2D descriptors (217 features)")

            self.featurizer = RDKit2DFeaturizer()
            self._embedding_dim = 217  # RDKit2D has 217 features

        elif self.featurizer_type == 'graph':
            from chemprop.featurizers import SimpleMoleculeMolGraphFeaturizer
            from chemprop.nn import BondMessagePassing, MeanAggregation
            import torch

            trainable_str = "trainable" if self.trainable else "frozen"
            print(f"Using ChemProp v2 D-MPNN graph embeddings (300-dim, {trainable_str}) on {self.device}")

            # Create graph featurizer
            self.featurizer = SimpleMoleculeMolGraphFeaturizer()
            d_v, d_e = self.featurizer.shape  # (72, 14)

            # Create message passing network (randomly initialized)
            dropout = 0.1 if self.trainable else 0.0  # Add dropout if training
            self.message_passing = BondMessagePassing(
                d_v=d_v,
                d_e=d_e,
                d_h=300,  # hidden dimension
                depth=3,   # 3 message passing layers
                dropout=dropout
            )
            self.aggregation = MeanAggregation()

            # Move to device
            self.message_passing = self.message_passing.to(self.device)
            self.aggregation = self.aggregation.to(self.device)

            # Control trainability
            if self.trainable:
                # Set to training mode - parameters are trainable
                self.message_passing.train()
                print("  → GNN parameters are TRAINABLE (gradients will backpropagate)")
            else:
                # Set to eval mode (frozen - no training)
                self.message_passing.eval()
                for param in self.message_passing.parameters():
                    param.requires_grad = False
                print("  → GNN parameters are FROZEN (no gradient updates)")

            self._embedding_dim = 300

        else:
            raise ValueError(
                f"Unknown featurizer_type: {self.featurizer_type}. "
                "Supported: 'morgan', 'rdkit2d', 'graph'"
            )

    def encode(self, smiles: Union[str, List[str]], show_progress: bool = False) -> np.ndarray:
        """
        Encode molecule(s) to embedding vectors using ChemProp v2.

        Args:
            smiles: Single SMILES string or list of SMILES
            show_progress: If True, show tqdm progress bar (useful for large batches)

        Returns:
            Embedding vector(s) as numpy array
        """
        if isinstance(smiles, str):
            smiles_list = [smiles]
            return_single = True
        else:
            # Convert numpy array to list if needed
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = smiles
            return_single = False

        # Encode using appropriate method
        if self._use_model:
            embeddings = self._encode_with_model(smiles_list, show_progress=show_progress)
        else:
            embeddings = self._encode_with_featurization(smiles_list, show_progress=show_progress)

        if return_single:
            return embeddings[0]
        else:
            return embeddings

    def _encode_with_model(self, smiles_list: List[str], show_progress: bool = False) -> np.ndarray:
        """Encode using trained ChemProp v2 model to extract learned representations."""
        import torch
        from chemprop.data import MoleculeDatapoint
        from tqdm import tqdm

        # Create datapoints
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]

        # Extract embeddings
        self.model.eval()
        embeddings = []

        n_batches = (len(datapoints) + self.batch_size - 1) // self.batch_size
        batch_iter = range(0, len(datapoints), self.batch_size)
        if show_progress:
            batch_iter = tqdm(batch_iter, total=n_batches, desc="Encoding (model)")

        with torch.no_grad():
            for i in batch_iter:
                batch_dps = datapoints[i:i + self.batch_size]

                # Get model predictions (this uses the encoder internally)
                # For embeddings, we'd need to access the encoder directly
                # This depends on the specific model architecture
                try:
                    batch_embs = self.model.encode([dp.mol for dp in batch_dps])
                    embeddings.append(batch_embs.cpu().numpy())
                except AttributeError:
                    # Fallback: use model forward and extract hidden layer
                    raise NotImplementedError(
                        "Model-based encoding requires a model with encode() method. "
                        "For now, use featurization mode (no model_path)."
                    )

        return np.vstack(embeddings)

    def _encode_with_featurization(self, smiles_list: List[str], show_progress: bool = False) -> np.ndarray:
        """
        Encode using ChemProp v2 featurization (Morgan, RDKit2D, or graph-based).

        Args:
            smiles_list: List of SMILES strings to encode
            show_progress: If True, show tqdm progress bar
        """
        from chemprop.data import MoleculeDatapoint
        from tqdm import tqdm

        # Create datapoints
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]

        # Graph-based encoding requires special handling
        if self.featurizer_type == 'graph':
            return self._encode_with_graph_mpnn(datapoints, show_progress=show_progress)

        # Standard featurization (Morgan or RDKit2D)
        embeddings = []
        feat_iter = datapoints
        if show_progress:
            feat_iter = tqdm(datapoints, desc=f"Encoding ({self.featurizer_type})")

        for dp in feat_iter:
            try:
                # Use the featurizer on the RDKit mol object
                feat = self.featurizer(dp.mol)
                embeddings.append(feat)
            except Exception as e:
                # If featurization fails, use zero vector
                print(f"Warning: Could not featurize SMILES {dp.smiles}, using zeros: {e}")
                embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))

        return np.array(embeddings, dtype=np.float32)

    def _encode_with_graph_mpnn(self, datapoints: List, show_progress: bool = False) -> np.ndarray:
        """
        Encode using D-MPNN graph neural network.

        Uses message passing on molecular graphs to get learned representations.

        Args:
            datapoints: List of MoleculeDatapoint objects
            show_progress: If True, show tqdm progress bar
        """
        import torch
        from chemprop.data import BatchMolGraph
        from tqdm import tqdm

        # Build molecular graphs
        mol_graphs = []
        graph_iter = tqdm(datapoints, desc="Building graphs", disable=not show_progress)
        for dp in graph_iter:
            try:
                mol_graph = self.featurizer(dp.mol)
                mol_graphs.append(mol_graph)
            except Exception as e:
                print(f"Warning: Could not build graph for {dp.smiles}: {e}")
                # Use None as placeholder
                mol_graphs.append(None)

        # Process in batches
        embeddings = []
        self.message_passing.eval()

        n_batches = (len(mol_graphs) + self.batch_size - 1) // self.batch_size
        with torch.no_grad():
            batch_iter = range(0, len(mol_graphs), self.batch_size)
            if show_progress:
                batch_iter = tqdm(batch_iter, total=n_batches, desc="Encoding batches")
            for i in batch_iter:
                batch_graphs = mol_graphs[i:i + self.batch_size]

                # Filter out None graphs
                valid_graphs = [g for g in batch_graphs if g is not None]
                if not valid_graphs:
                    # All graphs failed, use zeros
                    embeddings.extend([
                        np.zeros(self._embedding_dim, dtype=np.float32)
                        for _ in batch_graphs
                    ])
                    continue

                # Batch graphs
                try:
                    batch_graph = BatchMolGraph(valid_graphs)

                    # Debug: check if BatchMolGraph was created properly
                    if batch_graph is None:
                        print(f"Warning: BatchMolGraph creation returned None")
                        embeddings.extend([
                            np.zeros(self._embedding_dim, dtype=np.float32)
                            for _ in batch_graphs
                        ])
                        continue

                except Exception as e:
                    print(f"Error creating BatchMolGraph: {e}")
                    import traceback
                    traceback.print_exc()
                    # If batching fails, use zeros for all
                    embeddings.extend([
                        np.zeros(self._embedding_dim, dtype=np.float32)
                        for _ in batch_graphs
                    ])
                    continue

                # Forward through message passing
                # Move batch_graph to the same device as the model
                batch_graph = batch_graph.to(self.device)
                h = self.message_passing(batch_graph)

                # Aggregate to molecule-level embeddings
                mol_embeddings = self.aggregation(h, batch_graph.batch)

                # Convert to numpy
                batch_embeddings = mol_embeddings.cpu().numpy()

                # Handle failed graphs
                valid_idx = 0
                for g in batch_graphs:
                    if g is None:
                        embeddings.append(np.zeros(self._embedding_dim, dtype=np.float32))
                    else:
                        embeddings.append(batch_embeddings[valid_idx])
                        valid_idx += 1

        return np.array(embeddings, dtype=np.float32)

    @property
    def embedding_dim(self) -> int:
        """Return the dimensionality of embeddings."""
        return self._embedding_dim

    @property
    def name(self) -> str:
        """Return the name of this embedding method."""
        if self._use_model:
            return f"chemprop_v2_model_{self._embedding_dim}"
        elif self.featurizer_type == 'morgan':
            chiral = "_chiral" if self.include_chirality else ""
            return f"chemprop_v2_morgan_r{self.morgan_radius}_l{self.morgan_length}{chiral}"
        elif self.featurizer_type == 'rdkit2d':
            return "chemprop_v2_rdkit2d_217"
        elif self.featurizer_type == 'graph':
            trainable_suffix = "_trainable" if self.trainable else "_frozen"
            return f"chemprop_v2_dmpnn_300{trainable_suffix}"
        else:
            return f"chemprop_v2_{self.featurizer_type}_{self._embedding_dim}"

    def freeze_gnn(self):
        """
        Freeze GNN parameters (stop gradient updates).
        Only applicable when featurizer_type='graph'.
        """
        if self.featurizer_type == 'graph' and hasattr(self, 'message_passing'):
            self.message_passing.eval()
            for param in self.message_passing.parameters():
                param.requires_grad = False
            self.trainable = False
            print("ChemProp GNN frozen (no gradient updates)")
        else:
            print("Warning: freeze_gnn() only applies to featurizer_type='graph'")

    def unfreeze_gnn(self):
        """
        Unfreeze GNN parameters (enable gradient updates).
        Only applicable when featurizer_type='graph'.
        """
        if self.featurizer_type == 'graph' and hasattr(self, 'message_passing'):
            self.message_passing.train()
            for param in self.message_passing.parameters():
                param.requires_grad = True
            self.trainable = True
            print("ChemProp GNN unfrozen (gradients will backpropagate)")
        else:
            print("Warning: unfreeze_gnn() only applies to featurizer_type='graph'")

    # Aliases for generalized encoder interface
    def freeze(self):
        """Freeze encoder parameters. Alias for freeze_gnn()."""
        self.freeze_gnn()

    def unfreeze(self):
        """Unfreeze encoder parameters. Alias for unfreeze_gnn()."""
        self.unfreeze_gnn()

    def get_encoder_parameters(self):
        """
        Get trainable encoder parameters for optimizer.

        Returns:
            List of parameters that should be optimized with encoder learning rate.
        """
        if self.featurizer_type == 'graph' and hasattr(self, 'message_passing') and self.trainable:
            return list(self.message_passing.parameters())
        return []

    def encode_with_atom_embeddings(
        self,
        smiles: Union[str, List[str]]
    ) -> Union[tuple, List[tuple]]:
        """
        Encode molecule(s) returning both atom-level and molecule-level embeddings.

        Only supported for featurizer_type='graph'. For other types, returns
        molecule embedding duplicated for each atom.

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            For single SMILES: (atom_embeddings, mol_embedding)
                - atom_embeddings: np.ndarray [n_atoms, embedding_dim]
                - mol_embedding: np.ndarray [embedding_dim]
            For list of SMILES: List of (atom_embeddings, mol_embedding) tuples
        """
        import torch
        from rdkit import Chem

        if isinstance(smiles, str):
            smiles_list = [smiles]
            return_single = True
        else:
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = smiles
            return_single = False

        if self.featurizer_type != 'graph':
            # For non-graph featurizers, return molecule embedding for each atom
            results = []
            mol_embeddings = self.encode(smiles_list)
            for i, smi in enumerate(smiles_list):
                mol = Chem.MolFromSmiles(smi)
                n_atoms = mol.GetNumAtoms() if mol else 1
                # Repeat molecule embedding for each atom (not ideal but compatible)
                atom_emb = np.tile(mol_embeddings[i], (n_atoms, 1))
                results.append((atom_emb, mol_embeddings[i]))

            if return_single:
                return results[0]
            return results

        # Graph-based encoding with atom-level embeddings
        from chemprop.data import MoleculeDatapoint, BatchMolGraph

        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]
        results = []

        # Process one at a time to get individual atom embeddings
        self.message_passing.eval()

        with torch.no_grad():
            for dp in datapoints:
                try:
                    mol_graph = self.featurizer(dp.mol)
                    if mol_graph is None:
                        raise ValueError("Failed to create mol graph")

                    # Create single-molecule batch and move to device
                    batch_graph = BatchMolGraph([mol_graph])
                    batch_graph = batch_graph.to(self.device)

                    # Get atom-level embeddings from message passing
                    # h has shape [n_atoms_in_batch, hidden_dim]
                    h = self.message_passing(batch_graph)

                    # Get number of atoms in this molecule
                    n_atoms = dp.mol.GetNumAtoms()

                    # Extract atom embeddings for this molecule
                    # Note: h includes all atoms, we take only actual atoms (not virtual)
                    atom_embeddings = h[:n_atoms].cpu().numpy()

                    # Aggregate for molecule embedding
                    mol_embedding = self.aggregation(h, batch_graph.batch).cpu().numpy()[0]

                    results.append((atom_embeddings.astype(np.float32),
                                   mol_embedding.astype(np.float32)))

                except Exception as e:
                    print(f"Warning: Could not encode {dp.smiles} with atom embeddings: {e}")
                    # Fallback: zeros
                    mol = Chem.MolFromSmiles(dp.smiles) if dp else None
                    n_atoms = mol.GetNumAtoms() if mol else 1
                    results.append((
                        np.zeros((n_atoms, self._embedding_dim), dtype=np.float32),
                        np.zeros(self._embedding_dim, dtype=np.float32)
                    ))

        if return_single:
            return results[0]
        return results

    def encode_trainable(
        self,
        smiles: Union[str, List[str]]
    ) -> "torch.Tensor":
        """
        Encode molecule(s) to embedding tensors WITH gradient tracking.

        This method is designed for end-to-end training where gradients need to
        flow back through the GNN. Unlike encode(), this:
        - Returns PyTorch tensors (not numpy arrays)
        - Does NOT use torch.no_grad()
        - Keeps the model in train() mode (if trainable=True)

        Only supported for featurizer_type='graph'. For other types, falls back
        to encode() wrapped as a tensor (no gradients possible).

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            torch.Tensor of shape [batch_size, embedding_dim] with gradient tracking
        """
        import torch
        from chemprop.data import MoleculeDatapoint, BatchMolGraph

        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = list(smiles)

        # For non-graph featurizers, fall back to numpy encode (no gradient support)
        if self.featurizer_type != 'graph':
            embeddings = self.encode(smiles_list)
            return torch.tensor(embeddings, dtype=torch.float32, device=self.device)

        # Build molecular graphs
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]
        mol_graphs = []
        valid_indices = []

        for i, dp in enumerate(datapoints):
            try:
                mol_graph = self.featurizer(dp.mol)
                if mol_graph is not None:
                    mol_graphs.append(mol_graph)
                    valid_indices.append(i)
            except Exception as e:
                print(f"Warning: Could not build graph for {dp.smiles}: {e}")

        if not mol_graphs:
            # All failed - return zeros
            return torch.zeros(len(smiles_list), self._embedding_dim,
                             dtype=torch.float32, device=self.device)

        # Batch all graphs and move to device
        batch_graph = BatchMolGraph(mol_graphs)
        batch_graph = batch_graph.to(self.device)

        # Forward through message passing (NO torch.no_grad!)
        # Keep train mode if trainable, otherwise eval
        if self.trainable:
            self.message_passing.train()
        else:
            self.message_passing.eval()

        h = self.message_passing(batch_graph)
        mol_embeddings = self.aggregation(h, batch_graph.batch)

        # Handle failed graphs by inserting zeros
        if len(valid_indices) < len(smiles_list):
            full_embeddings = torch.zeros(len(smiles_list), self._embedding_dim,
                                         dtype=torch.float32, device=self.device)
            for new_idx, orig_idx in enumerate(valid_indices):
                full_embeddings[orig_idx] = mol_embeddings[new_idx]
            return full_embeddings

        return mol_embeddings

    def encode_trainable_with_atom_embeddings(
        self,
        smiles: Union[str, List[str]]
    ) -> Dict[str, Union["torch.Tensor", List["torch.Tensor"]]]:
        """
        Encode molecule(s) returning both atom-level and molecule-level embeddings
        WITH gradient tracking for end-to-end training.

        This is the trainable version of encode_with_atom_embeddings().

        Only supported for featurizer_type='graph'. For other types, falls back
        to non-trainable encode_with_atom_embeddings().

        Args:
            smiles: Single SMILES string or list of SMILES

        Returns:
            Dictionary with:
                - 'atom_embeddings': List of tensors [n_atoms_i, embedding_dim] per molecule
                - 'mol_embeddings': Tensor [batch_size, embedding_dim]
        """
        import torch
        from chemprop.data import MoleculeDatapoint, BatchMolGraph
        from rdkit import Chem

        if isinstance(smiles, str):
            smiles_list = [smiles]
        else:
            if isinstance(smiles, np.ndarray):
                smiles_list = smiles.tolist()
            else:
                smiles_list = list(smiles)

        # For non-graph featurizers, fall back to numpy version wrapped as tensors
        if self.featurizer_type != 'graph':
            results = self.encode_with_atom_embeddings(smiles_list)
            atom_embs = [torch.tensor(r[0], dtype=torch.float32, device=self.device) for r in results]
            mol_embs = torch.tensor(np.stack([r[1] for r in results]), dtype=torch.float32, device=self.device)
            return {'atom_embeddings': atom_embs, 'mol_embeddings': mol_embs}

        # Build molecular graphs
        datapoints = [MoleculeDatapoint.from_smi(s) for s in smiles_list]
        mol_graphs = []
        valid_indices = []
        n_atoms_list = []

        for i, dp in enumerate(datapoints):
            try:
                mol_graph = self.featurizer(dp.mol)
                if mol_graph is not None:
                    mol_graphs.append(mol_graph)
                    valid_indices.append(i)
                    n_atoms_list.append(dp.mol.GetNumAtoms())
            except Exception as e:
                print(f"Warning: Could not build graph for {dp.smiles}: {e}")

        if not mol_graphs:
            # All failed - return zeros
            atom_embs = [torch.zeros(1, self._embedding_dim, dtype=torch.float32, device=self.device)
                        for _ in smiles_list]
            mol_embs = torch.zeros(len(smiles_list), self._embedding_dim,
                                  dtype=torch.float32, device=self.device)
            return {'atom_embeddings': atom_embs, 'mol_embeddings': mol_embs}

        # Batch all graphs and move to device
        batch_graph = BatchMolGraph(mol_graphs)
        batch_graph = batch_graph.to(self.device)

        # Forward through message passing (NO torch.no_grad!)
        if self.trainable:
            self.message_passing.train()
        else:
            self.message_passing.eval()

        # Get atom-level embeddings
        h = self.message_passing(batch_graph)
        mol_embeddings = self.aggregation(h, batch_graph.batch)

        # Split atom embeddings by molecule
        atom_embeddings_list = []
        atom_offset = 0
        for n_atoms in n_atoms_list:
            atom_emb = h[atom_offset:atom_offset + n_atoms]
            atom_embeddings_list.append(atom_emb)
            atom_offset += n_atoms

        # Handle failed graphs by inserting zeros
        if len(valid_indices) < len(smiles_list):
            full_atom_embs = []
            full_mol_embs = torch.zeros(len(smiles_list), self._embedding_dim,
                                       dtype=torch.float32, device=self.device)
            valid_iter = iter(zip(valid_indices, atom_embeddings_list))
            next_valid = next(valid_iter, None)

            for i, smi in enumerate(smiles_list):
                if next_valid and i == next_valid[0]:
                    full_atom_embs.append(next_valid[1])
                    full_mol_embs[i] = mol_embeddings[valid_indices.index(i)]
                    next_valid = next(valid_iter, None)
                else:
                    mol = Chem.MolFromSmiles(smi)
                    n_atoms = mol.GetNumAtoms() if mol else 1
                    full_atom_embs.append(torch.zeros(n_atoms, self._embedding_dim,
                                                     dtype=torch.float32, device=self.device))

            return {'atom_embeddings': full_atom_embs, 'mol_embeddings': full_mol_embs}

        return {'atom_embeddings': atom_embeddings_list, 'mol_embeddings': mol_embeddings}
