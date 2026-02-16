"""
Structured edit effect predictor - Complete implementation with training/prediction.

This version includes the full training loop and prediction functionality.
"""

import torch
import torch.nn as nn
import pytorch_lightning as pl
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from torch.utils.data import DataLoader, Dataset
from rdkit import Chem
from tqdm import tqdm

from src.embedding.structured_edit_embedder import StructuredEditEmbedder


# Enable Tensor Cores
torch.set_float32_matmul_precision('high')


class StructuredEditDataset(Dataset):
    """
    Dataset for structured edit prediction with MMP features.

    Stores all necessary data for fragment-based edit embedding:
    - Molecule RDKit objects
    - GNN embeddings (atom-level + global)
    - MMP structural indices
    - Labels
    """

    def __init__(
        self,
        mols_A: List[Chem.Mol],
        mols_B: List[Chem.Mol],
        H_A_list: List[torch.Tensor],  # Atom embeddings
        H_B_list: List[torch.Tensor],
        h_A_global_list: List[torch.Tensor],  # Global embeddings
        h_B_global_list: List[torch.Tensor],
        removed_atoms_A_list: List[List[int]],
        added_atoms_B_list: List[List[int]],
        attach_atoms_A_list: List[List[int]],
        mapped_pairs_list: Optional[List[List[Tuple[int, int]]]],
        delta_y: torch.Tensor
    ):
        self.mols_A = mols_A
        self.mols_B = mols_B
        self.H_A_list = H_A_list
        self.H_B_list = H_B_list
        self.h_A_global_list = h_A_global_list
        self.h_B_global_list = h_B_global_list
        self.removed_atoms_A_list = removed_atoms_A_list
        self.added_atoms_B_list = added_atoms_B_list
        self.attach_atoms_A_list = attach_atoms_A_list
        self.mapped_pairs_list = mapped_pairs_list if mapped_pairs_list else [None] * len(mols_A)
        self.delta_y = delta_y

    def __len__(self):
        return len(self.mols_A)

    def __getitem__(self, idx):
        return {
            'mol_A': self.mols_A[idx],
            'mol_B': self.mols_B[idx],
            'H_A': self.H_A_list[idx],
            'H_B': self.H_B_list[idx],
            'h_A_global': self.h_A_global_list[idx],
            'h_B_global': self.h_B_global_list[idx],
            'removed_atoms_A': self.removed_atoms_A_list[idx],
            'added_atoms_B': self.added_atoms_B_list[idx],
            'attach_atoms_A': self.attach_atoms_A_list[idx],
            'mapped_pairs': self.mapped_pairs_list[idx],
            'delta_y': self.delta_y[idx]
        }


class StructuredEditEffectMLP(pl.LightningModule):
    """
    PyTorch Lightning module for structured edit effect prediction.

    Architecture:
        GNN embeddings → StructuredEditEmbedder → MLP → Δproperty
    """

    def __init__(
        self,
        gnn_dim: int = 300,
        edit_mlp_dims: Optional[List[int]] = None,
        delta_mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        gnn_learning_rate: Optional[float] = None,
        mol_embedder: Optional[nn.Module] = None,
        n_tasks: int = 1,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        k_hop_env: int = 2,
        use_local_delta: bool = True,
        use_rdkit_fragment_descriptors: bool = True
    ):
        super().__init__()
        self.save_hyperparameters(ignore=['mol_embedder'])

        self.gnn_dim = gnn_dim
        self.learning_rate = learning_rate
        self.gnn_learning_rate = gnn_learning_rate if gnn_learning_rate is not None else 1e-5
        self.n_tasks = n_tasks
        self.mol_embedder = mol_embedder

        # Task names
        if task_names is None:
            self.task_names = ['delta_property'] if n_tasks == 1 else [f'task_{i}' for i in range(n_tasks)]
        else:
            if len(task_names) != n_tasks:
                raise ValueError(f"Number of task names ({len(task_names)}) must match n_tasks ({n_tasks})")
            self.task_names = task_names

        # Task weights
        self.task_weights = task_weights if task_weights else {name: 1.0 for name in self.task_names}

        # Structured edit embedder
        self.structured_edit_embedder = StructuredEditEmbedder(
            gnn_dim=gnn_dim,
            edit_mlp_dims=edit_mlp_dims,
            dropout=dropout,
            k_hop_env=k_hop_env,
            use_local_delta=use_local_delta,
            use_rdkit_fragment_descriptors=use_rdkit_fragment_descriptors
        )

        # Delta prediction MLP
        if delta_mlp_dims is None:
            delta_mlp_dims = [512, 256, 128]

        # Build delta predictor
        if n_tasks == 1:
            # Single-task
            layers = []
            prev_dim = self.structured_edit_embedder.output_dim
            for hidden_dim in delta_mlp_dims:
                layers.extend([
                    nn.Linear(prev_dim, hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(dropout)
                ])
                prev_dim = hidden_dim
            layers.append(nn.Linear(prev_dim, 1))
            self.delta_predictor = nn.Sequential(*layers)
        else:
            # Multi-task
            from src.models.architectures.multi_head import MultiTaskNetwork
            self.multi_task_network = MultiTaskNetwork(
                input_dim=self.structured_edit_embedder.output_dim,
                backbone_hidden_dims=delta_mlp_dims,
                shared_dim=delta_mlp_dims[-1],  # Use last hidden dim as shared_dim
                task_names=self.task_names,
                dropout=dropout
            )

    def forward(self, batch):
        """
        Forward pass for a batch.

        Args:
            batch: Dict from StructuredEditDataset

        Returns:
            Predictions [batch_size, n_tasks]
        """
        # Get device from model parameters
        device = next(self.structured_edit_embedder.parameters()).device

        # Process each example in batch through structured embedder
        edit_embeddings = []
        for i in range(len(batch['mol_A'])):
            # Move tensors to device
            H_A = batch['H_A'][i].to(device)
            H_B = batch['H_B'][i].to(device)
            h_A_global = batch['h_A_global'][i].to(device)
            h_B_global = batch['h_B_global'][i].to(device)

            edit_features = self.structured_edit_embedder(
                H_A=H_A,
                H_B=H_B,
                h_A_global=h_A_global,
                h_B_global=h_B_global,
                mol_A=batch['mol_A'][i],
                mol_B=batch['mol_B'][i],
                removed_atom_indices_A=batch['removed_atoms_A'][i],
                added_atom_indices_B=batch['added_atoms_B'][i],
                attach_atom_indices_A=batch['attach_atoms_A'][i],
                mapped_atom_pairs=batch['mapped_pairs'][i]
            )
            edit_embeddings.append(edit_features['edit_embedding'])

        edit_embeddings = torch.stack(edit_embeddings)

        # Predict delta
        if self.n_tasks == 1:
            predictions = self.delta_predictor(edit_embeddings)
        else:
            predictions = self.multi_task_network(edit_embeddings)

        return predictions

    def training_step(self, batch, batch_idx):
        """Training step."""
        predictions = self.forward(batch)
        targets = batch['delta_y']

        # Compute loss
        if self.n_tasks == 1:
            loss = nn.functional.mse_loss(predictions.squeeze(), targets.squeeze())
        else:
            # Multi-task weighted loss
            # predictions is a dict {task_name: tensor}
            task_losses = []
            for i, task_name in enumerate(self.task_names):
                weight = self.task_weights[task_name]
                task_loss = nn.functional.mse_loss(predictions[task_name], targets[:, i])
                task_losses.append(weight * task_loss)
                self.log(f'train_loss_{task_name}', task_loss, prog_bar=False)

            loss = sum(task_losses) / len(task_losses)

        self.log('train_loss', loss, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Validation step."""
        predictions = self.forward(batch)
        targets = batch['delta_y']

        # Compute loss
        if self.n_tasks == 1:
            loss = nn.functional.mse_loss(predictions.squeeze(), targets.squeeze())
        else:
            # Multi-task weighted loss
            # predictions is a dict {task_name: tensor}
            task_losses = []
            for i, task_name in enumerate(self.task_names):
                weight = self.task_weights[task_name]
                task_loss = nn.functional.mse_loss(predictions[task_name], targets[:, i])
                task_losses.append(weight * task_loss)
                self.log(f'val_loss_{task_name}', task_loss, prog_bar=False)

            loss = sum(task_losses) / len(task_losses)

        self.log('val_loss', loss, prog_bar=True)
        return loss

    def configure_optimizers(self):
        """Configure optimizer with separate learning rates for GNN and MLP."""
        if self.mol_embedder is not None and hasattr(self.mol_embedder, 'trainable') and self.mol_embedder.trainable:
            param_groups = []

            # GNN parameters
            gnn_params_set = set()
            if hasattr(self.mol_embedder, 'message_passing'):
                gnn_params = [p for p in self.mol_embedder.message_passing.parameters() if p.requires_grad]
                if gnn_params:
                    gnn_params_set = set(id(p) for p in gnn_params)
                    gnn_param_count = sum(p.numel() for p in gnn_params)
                    param_groups.append({'params': gnn_params, 'lr': self.gnn_learning_rate, 'name': 'gnn'})
                    print(f"\n{'='*70}")
                    print(f"OPTIMIZER SETUP:")
                    print(f"  → GNN: {len(gnn_params)} tensors, {gnn_param_count:,} params (lr={self.gnn_learning_rate})")

            # MLP parameters (exclude GNN params)
            mlp_params = [p for p in self.parameters() if p.requires_grad and id(p) not in gnn_params_set]
            mlp_param_count = sum(p.numel() for p in mlp_params)
            param_groups.append({'params': mlp_params, 'lr': self.learning_rate, 'name': 'mlp'})
            print(f"  → MLP: {len(mlp_params)} tensors, {mlp_param_count:,} params (lr={self.learning_rate})")
            print(f"  → TOTAL: {gnn_param_count + mlp_param_count:,} trainable parameters" if gnn_params_set else f"  → TOTAL: {mlp_param_count:,} parameters")
            print(f"{'='*70}\n")

            optimizer = torch.optim.Adam(param_groups)
        else:
            optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )

        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'frequency': 1,
                'strict': False  # Don't fail if val_loss is not available
            }
        }


class StructuredEditEffectPredictor:
    """
    High-level API for structured edit effect prediction with MMP features.

    Handles:
    1. GNN encoding (atom embeddings + global)
    2. Fragment-based edit embedding
    3. Training and prediction
    """

    def __init__(
        self,
        mol_embedder,
        gnn_dim: int = 300,
        edit_mlp_dims: Optional[List[int]] = None,
        delta_mlp_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        gnn_learning_rate: Optional[float] = None,
        batch_size: int = 32,
        max_epochs: int = 100,
        device: Optional[str] = None,
        task_names: Optional[List[str]] = None,
        task_weights: Optional[dict] = None,
        k_hop_env: int = 2,
        use_local_delta: bool = True,
        use_rdkit_fragment_descriptors: bool = True
    ):
        self.mol_embedder = mol_embedder
        self.gnn_dim = gnn_dim
        self.edit_mlp_dims = edit_mlp_dims
        self.delta_mlp_dims = delta_mlp_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.gnn_learning_rate = gnn_learning_rate if gnn_learning_rate is not None else 1e-5
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.task_names = task_names
        self.task_weights = task_weights
        self.k_hop_env = k_hop_env
        self.use_local_delta = use_local_delta
        self.use_rdkit_fragment_descriptors = use_rdkit_fragment_descriptors

        if device is None:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        else:
            self.device = device

        self.model = None
        self.trainer = None
        self.n_tasks = len(task_names) if task_names is not None else 1

    def _compute_gnn_embeddings(
        self,
        smiles_list: List[str],
        desc: str = "Computing GNN embeddings"
    ) -> Tuple[List[torch.Tensor], List[torch.Tensor]]:
        """
        Compute GNN embeddings for a list of SMILES.

        Returns:
            atom_embeddings_list: List of [n_atoms, gnn_dim] tensors
            global_embeddings_list: List of [gnn_dim] tensors
        """
        # Check if embedder supports atom-level embeddings
        if not hasattr(self.mol_embedder, 'message_passing'):
            raise ValueError(
                "mol_embedder must support atom-level embeddings (ChemProp graph mode). "
                f"Current embedder: {type(self.mol_embedder).__name__}"
            )

        atom_embeddings_list = []
        global_embeddings_list = []

        print(f"{desc} for {len(smiles_list)} molecules...")

        for smiles in tqdm(smiles_list, desc=desc):
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f"Invalid SMILES: {smiles}")

            # Get atom embeddings from GNN
            # This requires accessing internal ChemProp encoding
            # For now, placeholder - needs integration with ChemProp's encode_with_graph method
            from chemprop.data import MoleculeDatapoint
            from chemprop.data import BatchMolGraph

            dp = MoleculeDatapoint.from_smi(smiles)
            mol_graph = self.mol_embedder.featurizer(dp.mol)
            batch_graph = BatchMolGraph([mol_graph])

            # Forward through message passing
            with torch.no_grad():
                h = self.mol_embedder.message_passing(batch_graph)  # [n_atoms, gnn_dim]
                h_global = self.mol_embedder.aggregation(h, batch_graph.batch)  # [1, gnn_dim]

            atom_embeddings_list.append(h.cpu())
            global_embeddings_list.append(h_global.squeeze(0).cpu())

        return atom_embeddings_list, global_embeddings_list

    def fit(
        self,
        smiles_A: List[str],
        smiles_B: List[str],
        removed_atoms_A: List[List[int]],
        added_atoms_B: List[List[int]],
        attach_atoms_A: List[List[int]],
        delta_y: np.ndarray,
        mapped_pairs: Optional[List[List[Tuple[int, int]]]] = None,
        smiles_A_val: Optional[List[str]] = None,
        smiles_B_val: Optional[List[str]] = None,
        removed_atoms_A_val: Optional[List[List[int]]] = None,
        added_atoms_B_val: Optional[List[List[int]]] = None,
        attach_atoms_A_val: Optional[List[List[int]]] = None,
        delta_y_val: Optional[np.ndarray] = None,
        mapped_pairs_val: Optional[List[List[Tuple[int, int]]]] = None,
        verbose: bool = True
    ):
        """
        Train the model on MMP data.

        Args:
            smiles_A: SMILES for parent molecules
            smiles_B: SMILES for edited molecules
            removed_atoms_A: Indices of leaving fragment atoms
            added_atoms_B: Indices of incoming fragment atoms
            attach_atoms_A: Indices of attachment atoms
            delta_y: Property changes [n_samples, n_tasks] or [n_samples,]
            mapped_pairs: Optional atom mappings
            *_val: Validation data
            verbose: Show progress
        """
        print(f"\n{'='*70}")
        print("Structured Edit Effect Predictor - Training")
        print(f"{'='*70}\n")

        # 1. Compute GNN embeddings
        H_A_list, h_A_global_list = self._compute_gnn_embeddings(smiles_A, "Training embeddings")
        H_B_list, h_B_global_list = self._compute_gnn_embeddings(smiles_B, "Training embeddings")

        # 2. Create RDKit Mol objects
        mols_A = [Chem.MolFromSmiles(s) for s in smiles_A]
        mols_B = [Chem.MolFromSmiles(s) for s in smiles_B]

        # 3. Prepare labels
        delta_y = np.array(delta_y, dtype=np.float32)
        if delta_y.ndim == 1:
            delta_y = delta_y.reshape(-1, 1 if self.n_tasks == 1 else self.n_tasks)
        delta_y_tensor = torch.from_numpy(delta_y)

        # 4. Create dataset
        train_dataset = StructuredEditDataset(
            mols_A=mols_A,
            mols_B=mols_B,
            H_A_list=H_A_list,
            H_B_list=H_B_list,
            h_A_global_list=h_A_global_list,
            h_B_global_list=h_B_global_list,
            removed_atoms_A_list=removed_atoms_A,
            added_atoms_B_list=added_atoms_B,
            attach_atoms_A_list=attach_atoms_A,
            mapped_pairs_list=mapped_pairs,
            delta_y=delta_y_tensor
        )

        # 5. Create validation dataset
        val_dataset = None
        if smiles_A_val is not None:
            H_A_val_list, h_A_global_val_list = self._compute_gnn_embeddings(smiles_A_val, "Validation embeddings")
            H_B_val_list, h_B_global_val_list = self._compute_gnn_embeddings(smiles_B_val, "Validation embeddings")
            mols_A_val = [Chem.MolFromSmiles(s) for s in smiles_A_val]
            mols_B_val = [Chem.MolFromSmiles(s) for s in smiles_B_val]
            delta_y_val_tensor = torch.from_numpy(np.array(delta_y_val, dtype=np.float32).reshape(-1, delta_y_tensor.shape[1]))

            val_dataset = StructuredEditDataset(
                mols_A=mols_A_val,
                mols_B=mols_B_val,
                H_A_list=H_A_val_list,
                H_B_list=H_B_val_list,
                h_A_global_list=h_A_global_val_list,
                h_B_global_list=h_B_global_val_list,
                removed_atoms_A_list=removed_atoms_A_val,
                added_atoms_B_list=added_atoms_B_val,
                attach_atoms_A_list=attach_atoms_A_val,
                mapped_pairs_list=mapped_pairs_val,
                delta_y=delta_y_val_tensor
            )

        # 6. Create DataLoaders with custom collate function
        def collate_fn(batch):
            """Custom collate function that preserves lists of variable-length data."""
            # Stack delta_y tensors
            delta_y_batch = torch.stack([item['delta_y'] for item in batch])

            return {
                'mol_A': [item['mol_A'] for item in batch],
                'mol_B': [item['mol_B'] for item in batch],
                'H_A': [item['H_A'] for item in batch],
                'H_B': [item['H_B'] for item in batch],
                'h_A_global': [item['h_A_global'] for item in batch],
                'h_B_global': [item['h_B_global'] for item in batch],
                'removed_atoms_A': [item['removed_atoms_A'] for item in batch],
                'added_atoms_B': [item['added_atoms_B'] for item in batch],
                'attach_atoms_A': [item['attach_atoms_A'] for item in batch],
                'mapped_pairs': [item['mapped_pairs'] for item in batch],
                'delta_y': delta_y_batch
            }

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=collate_fn)
        val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn) if val_dataset else None

        # 7. Create model
        self.model = StructuredEditEffectMLP(
            gnn_dim=self.gnn_dim,
            edit_mlp_dims=self.edit_mlp_dims,
            delta_mlp_dims=self.delta_mlp_dims,
            dropout=self.dropout,
            learning_rate=self.learning_rate,
            gnn_learning_rate=self.gnn_learning_rate,
            mol_embedder=self.mol_embedder,
            n_tasks=self.n_tasks,
            task_names=self.task_names,
            task_weights=self.task_weights,
            k_hop_env=self.k_hop_env,
            use_local_delta=self.use_local_delta,
            use_rdkit_fragment_descriptors=self.use_rdkit_fragment_descriptors
        )

        # 8. Create trainer
        self.trainer = pl.Trainer(
            max_epochs=self.max_epochs,
            accelerator='auto',
            devices=1,
            enable_progress_bar=verbose,
            enable_model_summary=True
        )

        # 9. Train
        print(f"\nTraining for up to {self.max_epochs} epochs...")
        self.trainer.fit(self.model, train_loader, val_loader)

        print(f"\n{'='*70}")
        print("Training completed!")
        print(f"{'='*70}\n")

    def predict(
        self,
        smiles_A: List[str],
        smiles_B: List[str],
        removed_atoms_A: List[List[int]],
        added_atoms_B: List[List[int]],
        attach_atoms_A: List[List[int]],
        mapped_pairs: Optional[List[List[Tuple[int, int]]]] = None
    ) -> np.ndarray:
        """
        Predict on test data.

        Returns:
            Predictions [n_samples, n_tasks] or [n_samples,]
        """
        if self.model is None:
            raise ValueError("Model not trained. Call fit() first.")

        # Compute embeddings
        H_A_list, h_A_global_list = self._compute_gnn_embeddings(smiles_A, "Test embeddings")
        H_B_list, h_B_global_list = self._compute_gnn_embeddings(smiles_B, "Test embeddings")

        # Create Mol objects
        mols_A = [Chem.MolFromSmiles(s) for s in smiles_A]
        mols_B = [Chem.MolFromSmiles(s) for s in smiles_B]

        # Dummy labels
        dummy_labels = torch.zeros(len(smiles_A), self.n_tasks if self.n_tasks > 1 else 1)

        # Create dataset
        test_dataset = StructuredEditDataset(
            mols_A=mols_A,
            mols_B=mols_B,
            H_A_list=H_A_list,
            H_B_list=H_B_list,
            h_A_global_list=h_A_global_list,
            h_B_global_list=h_B_global_list,
            removed_atoms_A_list=removed_atoms_A,
            added_atoms_B_list=added_atoms_B,
            attach_atoms_A_list=attach_atoms_A,
            mapped_pairs_list=mapped_pairs,
            delta_y=dummy_labels
        )

        def collate_fn(batch):
            """Custom collate function that preserves lists of variable-length data."""
            delta_y_batch = torch.stack([item['delta_y'] for item in batch])
            return {
                'mol_A': [item['mol_A'] for item in batch],
                'mol_B': [item['mol_B'] for item in batch],
                'H_A': [item['H_A'] for item in batch],
                'H_B': [item['H_B'] for item in batch],
                'h_A_global': [item['h_A_global'] for item in batch],
                'h_B_global': [item['h_B_global'] for item in batch],
                'removed_atoms_A': [item['removed_atoms_A'] for item in batch],
                'added_atoms_B': [item['added_atoms_B'] for item in batch],
                'attach_atoms_A': [item['attach_atoms_A'] for item in batch],
                'mapped_pairs': [item['mapped_pairs'] for item in batch],
                'delta_y': delta_y_batch
            }

        test_loader = DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=collate_fn)

        # Predict
        self.model.eval()
        predictions = []
        with torch.no_grad():
            for batch in tqdm(test_loader, desc="Predicting"):
                preds = self.model.forward(batch)

                # Handle multi-task dict output
                if self.n_tasks > 1 and isinstance(preds, dict):
                    # Stack predictions for all tasks: {task_name: [batch_size]} -> [batch_size, n_tasks]
                    preds_stacked = torch.stack([preds[task_name] for task_name in self.task_names], dim=1)
                    predictions.append(preds_stacked.cpu())
                else:
                    predictions.append(preds.cpu())

        predictions = torch.cat(predictions, dim=0).numpy()

        if self.n_tasks == 1:
            predictions = predictions.squeeze()

        return predictions
