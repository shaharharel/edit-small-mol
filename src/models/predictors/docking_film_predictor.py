"""
FiLM-conditioned delta predictors augmented with docking features.

Three architectures that incorporate 3D protein-ligand binding information
alongside 2D molecular fingerprint differences:

1. DockingFiLMDeltaMLP — Concatenates docking features with Morgan diff
   before encoding into FiLM conditioning signal.

2. DockingDualStreamFiLM — Gated fusion of two streams:
   Stream 1 (2D): MLP on Morgan diff (chemical change)
   Stream 2 (3D): MLP on docking features (binding change)
   Learned gate selects when to use 2D vs 3D information.

3. HierarchicalFiLM — Two-level FiLM conditioning:
   Level 1: Standard FiLM blocks conditioned on Morgan diff
   Level 2: Additional FiLM blocks conditioned on docking features
   Allows the network to first adapt to the chemical edit, then refine
   based on binding context.
"""

import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as scipy_stats

from .film_delta_predictor import FiLMLayer, FiLMBlock

# Enable Tensor Cores for GPU
torch.set_float32_matmul_precision('high')


# ═══════════════════════════════════════════════════════════════════════════
# Architecture 1: DockingFiLMDeltaMLP
# ═══════════════════════════════════════════════════════════════════════════

class DockingFiLMDeltaMLP(nn.Module):
    """
    FiLM-conditioned MLP that incorporates docking features into the
    delta conditioning signal.

    Like FiLMDeltaMLP, but the delta_encoder takes [emb_b - emb_a, extra_feats]
    concatenated, where extra_feats are per-pair docking feature differences.

    Args:
        input_dim: Embedding dimension (same for A and B)
        extra_dim: Dimension of extra (docking) features per pair
        hidden_dims: List of hidden layer dimensions
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        extra_dim: int,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.extra_dim = extra_dim

        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim
            for _ in range(3):
                current_dim = current_dim // 2
                if current_dim < 64:
                    break
                hidden_dims.append(current_dim)
            if not hidden_dims:
                hidden_dims = [max(input_dim // 2, 64)]

        self.hidden_dims = hidden_dims

        # Delta encoder: takes [morgan_diff, extra_feats]
        delta_input_dim = input_dim + extra_dim
        delta_hidden = max(input_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(delta_input_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # FiLM blocks
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            block = FiLMBlock(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                cond_dim=delta_hidden,
                dropout=dropout,
            )
            self.blocks.append(block)
            prev_dim = hidden_dim

        self.output = nn.Linear(prev_dim, 1)

    def forward_single(self, x: torch.Tensor, delta_cond: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, delta_cond)
        return self.output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        extra_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass: predict delta from embeddings and docking features.

        Args:
            emb_a: Reference embeddings [batch, input_dim]
            emb_b: Variant embeddings [batch, input_dim]
            extra_feats: Per-pair extra features [batch, extra_dim]

        Returns:
            Predicted delta [batch]
        """
        delta = emb_b - emb_a
        delta_aug = torch.cat([delta, extra_feats], dim=-1)
        delta_cond = self.delta_encoder(delta_aug)

        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Architecture 2: DockingDualStreamFiLM
# ═══════════════════════════════════════════════════════════════════════════

class DockingDualStreamFiLM(nn.Module):
    """
    Two-stream FiLM with learned gating between 2D and 3D signals.

    Stream 1 (2D): MLP on Morgan FP difference (chemical change)
    Stream 2 (3D): MLP on docking feature differences (binding change)
    Gate: sigmoid(Linear(cat([s1, s2]))) learns when to use 2D vs 3D

    Args:
        input_dim: Embedding dimension (Morgan FP)
        dock_dim: Dimension of docking features per pair
        hidden_dims: Hidden dims for FiLM blocks
        stream_hidden: Hidden dim for each stream encoder
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        dock_dim: int,
        hidden_dims: Optional[List[int]] = None,
        stream_hidden: int = 128,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dock_dim = dock_dim

        if hidden_dims is None:
            hidden_dims = []
            current_dim = input_dim
            for _ in range(3):
                current_dim = current_dim // 2
                if current_dim < 64:
                    break
                hidden_dims.append(current_dim)
            if not hidden_dims:
                hidden_dims = [max(input_dim // 2, 64)]

        self.hidden_dims = hidden_dims

        # Stream 1: 2D chemical change (Morgan diff)
        self.stream_2d = nn.Sequential(
            nn.Linear(input_dim, stream_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Stream 2: 3D binding change (docking features)
        self.stream_3d = nn.Sequential(
            nn.Linear(dock_dim, stream_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Gating mechanism: learns when to trust 2D vs 3D
        self.gate = nn.Sequential(
            nn.Linear(stream_hidden * 2, stream_hidden),
            nn.Sigmoid(),
        )

        # Conditioning dimension after gated fusion
        cond_dim = stream_hidden

        # FiLM blocks
        self.blocks = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            block = FiLMBlock(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                cond_dim=cond_dim,
                dropout=dropout,
            )
            self.blocks.append(block)
            prev_dim = hidden_dim

        self.output = nn.Linear(prev_dim, 1)

    def forward_single(self, x: torch.Tensor, cond: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, cond)
        return self.output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        dock_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with gated 2D+3D fusion.

        Args:
            emb_a: Reference embeddings [batch, input_dim]
            emb_b: Variant embeddings [batch, input_dim]
            dock_feats: Per-pair docking feature differences [batch, dock_dim]

        Returns:
            Predicted delta [batch]
        """
        delta = emb_b - emb_a

        # Encode each stream
        s1 = self.stream_2d(delta)         # [batch, stream_hidden]
        s2 = self.stream_3d(dock_feats)    # [batch, stream_hidden]

        # Gated fusion
        gate_input = torch.cat([s1, s2], dim=-1)
        g = self.gate(gate_input)  # [batch, stream_hidden], values in [0, 1]
        fused = g * s1 + (1 - g) * s2  # [batch, stream_hidden]

        pred_a = self.forward_single(emb_a, fused)
        pred_b = self.forward_single(emb_b, fused)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Architecture 3: HierarchicalFiLM
# ═══════════════════════════════════════════════════════════════════════════

class HierarchicalFiLM(nn.Module):
    """
    Two-level FiLM conditioning: first adapt to chemical edit, then refine
    based on docking/binding context.

    Level 1: Standard FiLM blocks conditioned on Morgan diff
    Level 2: Additional FiLM blocks conditioned on docking features

    Args:
        input_dim: Embedding dimension
        dock_dim: Dimension of docking features per pair
        hidden_dims_l1: Hidden dims for level 1 (chemical edit)
        hidden_dims_l2: Hidden dims for level 2 (docking refinement)
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        dock_dim: int,
        hidden_dims_l1: Optional[List[int]] = None,
        hidden_dims_l2: Optional[List[int]] = None,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.dock_dim = dock_dim

        if hidden_dims_l1 is None:
            hidden_dims_l1 = [max(input_dim // 2, 128), max(input_dim // 4, 64)]
        if hidden_dims_l2 is None:
            hidden_dims_l2 = [max(hidden_dims_l1[-1] // 2, 64)]

        self.hidden_dims_l1 = hidden_dims_l1
        self.hidden_dims_l2 = hidden_dims_l2

        # Level 1 conditioning: Morgan diff
        delta_hidden = max(input_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(input_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Level 2 conditioning: docking features
        dock_hidden = max(dock_dim * 2, 32)
        self.dock_encoder = nn.Sequential(
            nn.Linear(dock_dim, dock_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Level 1 FiLM blocks (chemical edit conditioning)
        self.blocks_l1 = nn.ModuleList()
        prev_dim = input_dim
        for hidden_dim in hidden_dims_l1:
            block = FiLMBlock(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                cond_dim=delta_hidden,
                dropout=dropout,
            )
            self.blocks_l1.append(block)
            prev_dim = hidden_dim

        # Level 2 FiLM blocks (docking refinement)
        self.blocks_l2 = nn.ModuleList()
        for hidden_dim in hidden_dims_l2:
            block = FiLMBlock(
                input_dim=prev_dim,
                hidden_dim=hidden_dim,
                cond_dim=dock_hidden,
                dropout=dropout,
            )
            self.blocks_l2.append(block)
            prev_dim = hidden_dim

        self.output = nn.Linear(prev_dim, 1)

    def forward_single(
        self,
        x: torch.Tensor,
        delta_cond: torch.Tensor,
        dock_cond: torch.Tensor,
    ) -> torch.Tensor:
        h = x
        # Level 1: chemical edit
        for block in self.blocks_l1:
            h = block(h, delta_cond)
        # Level 2: docking refinement
        for block in self.blocks_l2:
            h = block(h, dock_cond)
        return self.output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        dock_feats: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass with hierarchical conditioning.

        Args:
            emb_a: Reference embeddings [batch, input_dim]
            emb_b: Variant embeddings [batch, input_dim]
            dock_feats: Per-pair docking feature differences [batch, dock_dim]

        Returns:
            Predicted delta [batch]
        """
        delta = emb_b - emb_a
        delta_cond = self.delta_encoder(delta)
        dock_cond = self.dock_encoder(dock_feats)

        pred_a = self.forward_single(emb_a, delta_cond, dock_cond)
        pred_b = self.forward_single(emb_b, delta_cond, dock_cond)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# DockingFiLMPredictor — High-level wrapper with training loop
# ═══════════════════════════════════════════════════════════════════════════

class DockingFiLMPredictor:
    """
    High-level wrapper for docking-augmented FiLM predictors.

    Handles training loop, early stopping, device management, and
    prediction for any of the three docking-aware architectures.

    Args:
        arch: Architecture name ('docking_film', 'dual_stream', 'hierarchical')
        extra_dim: Dimension of per-pair docking/extra features
        hidden_dims: Hidden layer dimensions (meaning depends on arch)
        dropout: Dropout probability
        learning_rate: Learning rate
        batch_size: Batch size
        max_epochs: Maximum training epochs
        patience: Early stopping patience
        device: 'cuda', 'mps', 'cpu', or None (auto-detect)
        stream_hidden: Hidden dim for dual-stream encoder (dual_stream only)
    """

    def __init__(
        self,
        arch: str = "docking_film",
        extra_dim: int = 3,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
        device: Optional[str] = None,
        stream_hidden: int = 128,
    ):
        self.arch = arch
        self.extra_dim = extra_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.stream_hidden = stream_hidden

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model = None
        self.input_dim = None

    def _build_model(self, input_dim: int) -> nn.Module:
        """Instantiate the appropriate architecture."""
        if self.arch == "docking_film":
            return DockingFiLMDeltaMLP(
                input_dim=input_dim,
                extra_dim=self.extra_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
        elif self.arch == "dual_stream":
            return DockingDualStreamFiLM(
                input_dim=input_dim,
                dock_dim=self.extra_dim,
                hidden_dims=self.hidden_dims,
                stream_hidden=self.stream_hidden,
                dropout=self.dropout,
            )
        elif self.arch == "hierarchical":
            return HierarchicalFiLM(
                input_dim=input_dim,
                dock_dim=self.extra_dim,
                hidden_dims_l1=self.hidden_dims,
                dropout=self.dropout,
            )
        else:
            raise ValueError(f"Unknown architecture: {self.arch}")

    def fit(
        self,
        emb_a_train: Union[np.ndarray, torch.Tensor],
        emb_b_train: Union[np.ndarray, torch.Tensor],
        dock_feats_train: Union[np.ndarray, torch.Tensor],
        delta_train: Union[np.ndarray, torch.Tensor],
        emb_a_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        emb_b_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        dock_feats_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        delta_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        verbose: bool = True,
        antisymmetric_aug: bool = False,
        antisym_reg_weight: float = 0.0,
    ) -> Dict[str, List[float]]:
        """
        Train the docking-augmented FiLM predictor.

        Args:
            emb_a_train: Reference embeddings [N, D]
            emb_b_train: Variant embeddings [N, D]
            dock_feats_train: Per-pair docking features [N, extra_dim]
            delta_train: Delta values [N]
            emb_a_val, emb_b_val, dock_feats_val, delta_val: Optional validation data
            verbose: Show training progress
            antisymmetric_aug: If True, augment training data with reversed
                pairs (emb_b, emb_a, -dock_feats, -delta) to enforce
                f(A->B) = -f(B->A). Doubles the training set.
            antisym_reg_weight: Weight for antisymmetry regularization loss
                L_sym = MSE(pred_forward + pred_reverse, 0). Only applied
                when > 0 during training (independent of antisymmetric_aug).

        Returns:
            Training history dict with 'train_loss' and 'val_loss' lists
        """
        emb_a_train = self._to_tensor(emb_a_train)
        emb_b_train = self._to_tensor(emb_b_train)
        dock_feats_train = self._to_tensor(dock_feats_train)
        delta_train = self._to_tensor(delta_train)

        # Antisymmetric data augmentation: append reversed pairs
        if antisymmetric_aug:
            emb_a_aug = torch.cat([emb_a_train, emb_b_train], dim=0)
            emb_b_aug = torch.cat([emb_b_train, emb_a_train], dim=0)
            dock_feats_train = torch.cat([dock_feats_train, -dock_feats_train], dim=0)
            delta_train = torch.cat([delta_train, -delta_train], dim=0)
            emb_a_train = emb_a_aug
            emb_b_train = emb_b_aug

        self.input_dim = emb_a_train.shape[1]
        self.model = self._build_model(self.input_dim)
        self.model = self.model.to(self.device)

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n  DockingFiLM ({self.arch}):")
            print(f"    Input dim: {self.input_dim}, Extra dim: {self.extra_dim}")
            print(f"    Training samples: {len(delta_train)}"
                  f"{' (incl. antisym aug)' if antisymmetric_aug else ''}")
            print(f"    Parameters: {n_params:,}")
            if antisym_reg_weight > 0:
                print(f"    Antisym reg weight: {antisym_reg_weight}")

        # Dataloaders
        train_dataset = TensorDataset(
            emb_a_train, emb_b_train, dock_feats_train, delta_train
        )
        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True
        )

        val_loader = None
        if all(v is not None for v in [emb_a_val, emb_b_val, dock_feats_val, delta_val]):
            emb_a_val = self._to_tensor(emb_a_val)
            emb_b_val = self._to_tensor(emb_b_val)
            dock_feats_val = self._to_tensor(dock_feats_val)
            delta_val = self._to_tensor(delta_val)
            val_dataset = TensorDataset(emb_a_val, emb_b_val, dock_feats_val, delta_val)
            val_loader = DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False)

        # Training
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode='min', factor=0.5, patience=10
        )
        criterion = nn.MSELoss()

        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            train_losses = []
            for batch_a, batch_b, batch_dock, batch_y in train_loader:
                batch_a = batch_a.to(self.device)
                batch_b = batch_b.to(self.device)
                batch_dock = batch_dock.to(self.device)
                batch_y = batch_y.to(self.device)

                optimizer.zero_grad()
                pred = self.model(batch_a, batch_b, batch_dock)
                loss = criterion(pred, batch_y)

                # Antisymmetry regularization: penalize pred_fwd + pred_rev != 0
                if antisym_reg_weight > 0:
                    pred_rev = self.model(batch_b, batch_a, -batch_dock)
                    sym_loss = torch.mean((pred + pred_rev) ** 2)
                    loss = loss + antisym_reg_weight * sym_loss

                loss.backward()
                optimizer.step()
                train_losses.append(loss.item())

            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch_a, batch_b, batch_dock, batch_y in val_loader:
                        batch_a = batch_a.to(self.device)
                        batch_b = batch_b.to(self.device)
                        batch_dock = batch_dock.to(self.device)
                        batch_y = batch_y.to(self.device)
                        pred = self.model(batch_a, batch_b, batch_dock)
                        loss = criterion(pred, batch_y)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"    Early stopping at epoch {epoch + 1}")
                    break
            else:
                history['val_loss'].append(train_loss)

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        if verbose:
            final_train = history['train_loss'][-1]
            final_val = history['val_loss'][-1] if history['val_loss'] else None
            msg = f"    Training complete: train_loss={final_train:.4f}"
            if final_val is not None:
                msg += f", val_loss={final_val:.4f}"
            print(msg)

        return history

    def predict(
        self,
        emb_a: Union[np.ndarray, torch.Tensor],
        emb_b: Union[np.ndarray, torch.Tensor],
        dock_feats: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        """
        Predict delta values.

        Args:
            emb_a: Reference embeddings [N, D]
            emb_b: Variant embeddings [N, D]
            dock_feats: Per-pair docking features [N, extra_dim]

        Returns:
            Predicted delta values [N]
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        emb_a = self._to_tensor(emb_a).to(self.device)
        emb_b = self._to_tensor(emb_b).to(self.device)
        dock_feats = self._to_tensor(dock_feats).to(self.device)

        self.model.eval()
        with torch.no_grad():
            pred = self.model(emb_a, emb_b, dock_feats).cpu().numpy()
        return pred

    def evaluate(
        self,
        emb_a: Union[np.ndarray, torch.Tensor],
        emb_b: Union[np.ndarray, torch.Tensor],
        dock_feats: Union[np.ndarray, torch.Tensor],
        delta_true: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        """Evaluate and return metrics, y_true, y_pred."""
        delta_true = self._to_numpy(delta_true)
        delta_pred = self.predict(emb_a, emb_b, dock_feats)

        mae = np.mean(np.abs(delta_pred - delta_true))
        mse = np.mean((delta_pred - delta_true) ** 2)
        rmse = np.sqrt(mse)

        if np.std(delta_pred) < 1e-6 or np.std(delta_true) < 1e-6:
            pearson = 0.0
            spearman = 0.0
        else:
            pearson, _ = scipy_stats.pearsonr(delta_pred, delta_true)
            spearman, _ = scipy_stats.spearmanr(delta_pred, delta_true)

        ss_res = np.sum((delta_true - delta_pred) ** 2)
        ss_tot = np.sum((delta_true - np.mean(delta_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = {
            'mae': float(mae),
            'mse': float(mse),
            'rmse': float(rmse),
            'pearson': float(pearson),
            'spearman': float(spearman),
            'r2': float(r2),
        }
        return metrics, delta_true, delta_pred

    def _to_tensor(self, x: Union[np.ndarray, torch.Tensor]) -> torch.Tensor:
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

    def _to_numpy(self, x: Union[np.ndarray, torch.Tensor]) -> np.ndarray:
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    @property
    def name(self) -> str:
        return f"DockingFiLM_{self.arch}"
