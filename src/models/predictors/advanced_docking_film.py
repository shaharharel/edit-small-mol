"""
Advanced docking-aware FiLM architectures for Iteration 2.

Informed by Iteration 1 results:
  - Simple 3-dim Vina features (score, inter, intra) beat all complex features
  - Pretraining adds consistent improvement
  - 280 molecules too small for high-capacity models

New architectures:
  1. ResidualCorrectionFiLM — FiLMDelta base + learned residual from docking
  2. MultiTaskDockingFiLM — Joint delta + Vina score prediction (auxiliary loss)
  3. FeatureGatedFiLM — Per-feature learned gates on docking features
  4. EnsemblePredictor — Snapshot/multi-seed ensemble

Key ideas:
  - Engineered features: differences, ratios, Z-scores of docking features
  - Residual learning: let base model handle chemistry, correction handles docking
  - Multi-task: auxiliary Vina prediction regularizes representations
  - Ensemble: average over snapshots or seeds for robustness
"""

import copy
import numpy as np
import torch
import torch.nn as nn
from typing import Dict, List, Optional, Tuple, Union
from torch.utils.data import DataLoader, TensorDataset
from scipy import stats as scipy_stats

from .film_delta_predictor import FiLMLayer, FiLMBlock, FiLMDeltaMLP


# ═══════════════════════════════════════════════════════════════════════════
# Feature Engineering
# ═══════════════════════════════════════════════════════════════════════════

def engineer_docking_features(
    vina_per_mol: np.ndarray,
    pairs_df,
    interact_per_mol: Optional[np.ndarray] = None,
    feature_set: str = "vina_engineered",
) -> np.ndarray:
    """Engineer per-pair docking features from per-molecule features.

    Feature sets:
        'vina_diff': Simple vina_b - vina_a (3d) — baseline
        'vina_engineered': Differences + absolute means + ratios (9d)
        'vina_selected': Differences + selected interaction features
        'full_engineered': All engineered features

    Args:
        vina_per_mol: [N_mols, 3] Vina features (score, inter, intra)
        pairs_df: DataFrame with idx_a, idx_b columns
        interact_per_mol: Optional [N_mols, 17] interaction features
        feature_set: Which feature set to use

    Returns:
        pair_feats: [N_pairs, D] engineered feature array
    """
    idx_a = pairs_df["idx_a"].values
    idx_b = pairs_df["idx_b"].values

    feats_a = vina_per_mol[idx_a]  # [N_pairs, 3]
    feats_b = vina_per_mol[idx_b]  # [N_pairs, 3]

    if feature_set == "vina_diff":
        return (feats_b - feats_a).astype(np.float32)

    # Differences
    diff = feats_b - feats_a  # [N_pairs, 3]
    # Absolute means (context)
    mean_abs = (np.abs(feats_a) + np.abs(feats_b)) / 2  # [N_pairs, 3]
    # Ratios (clipped to avoid inf)
    denom = np.clip(np.abs(feats_a), 1e-3, None)
    ratio = feats_b / denom  # [N_pairs, 3]
    ratio = np.clip(ratio, -10, 10)

    if feature_set == "vina_engineered":
        return np.hstack([diff, mean_abs, ratio]).astype(np.float32)

    if feature_set in ("vina_selected", "full_engineered") and interact_per_mol is not None:
        int_a = interact_per_mol[idx_a]
        int_b = interact_per_mol[idx_b]
        int_diff = int_b - int_a

        if feature_set == "vina_selected":
            # Select the most informative interaction features:
            # n_contact_residues(0), n_hbonds_total(3), n_hydrophobic(4),
            # burial_fraction(11), n_ligand_atoms_buried(12)
            selected_idx = [0, 3, 4, 11, 12]
            int_selected = int_diff[:, selected_idx]
            return np.hstack([diff, mean_abs, int_selected]).astype(np.float32)

        # full_engineered: all
        return np.hstack([diff, mean_abs, ratio, int_diff]).astype(np.float32)

    # Fallback
    return np.hstack([diff, mean_abs, ratio]).astype(np.float32)


# ═══════════════════════════════════════════════════════════════════════════
# Architecture 1: ResidualCorrectionFiLM
# ═══════════════════════════════════════════════════════════════════════════

class ResidualCorrectionFiLM(nn.Module):
    """FiLMDelta base + learned residual correction from docking features.

    The base FiLMDelta model handles the chemistry (Morgan diff conditioning).
    A small residual MLP learns a correction from docking features.

    final_pred = base_pred + alpha * residual_correction(dock_diff)

    where alpha is a learned scalar initialized to 0 (so the model starts
    as pure FiLMDelta and gradually learns to use docking).

    Args:
        input_dim: Embedding dimension (Morgan FP)
        dock_dim: Dimension of docking features per pair
        hidden_dims: Hidden dims for base FiLM blocks
        residual_hidden: Hidden dim for residual correction MLP
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        dock_dim: int,
        hidden_dims: Optional[List[int]] = None,
        residual_hidden: int = 32,
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

        # Base FiLMDelta model (no docking)
        delta_hidden = max(input_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(input_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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

        self.base_output = nn.Linear(prev_dim, 1)

        # Residual correction from docking features
        self.residual_mlp = nn.Sequential(
            nn.Linear(dock_dim, residual_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(residual_hidden, residual_hidden),
            nn.ReLU(),
            nn.Linear(residual_hidden, 1),
        )

        # Learned mixing scalar (initialized to 0 = pure FiLMDelta start)
        self.alpha = nn.Parameter(torch.tensor(0.0))

    def forward_base(self, x: torch.Tensor, delta_cond: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, delta_cond)
        return self.base_output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        dock_feats: torch.Tensor,
    ) -> torch.Tensor:
        delta = emb_b - emb_a
        delta_cond = self.delta_encoder(delta)

        base_a = self.forward_base(emb_a, delta_cond)
        base_b = self.forward_base(emb_b, delta_cond)
        base_pred = base_b - base_a

        # Residual correction from docking
        residual = self.residual_mlp(dock_feats).squeeze(-1)

        return base_pred + torch.sigmoid(self.alpha) * residual


# ═══════════════════════════════════════════════════════════════════════════
# Architecture 2: MultiTaskDockingFiLM
# ═══════════════════════════════════════════════════════════════════════════

class MultiTaskDockingFiLM(nn.Module):
    """FiLM with auxiliary task: predict docking scores alongside delta.

    The main task is delta prediction (as usual).
    Auxiliary task: predict the Vina score for mol_a and mol_b from their
    representations. This regularizes the embeddings to be "docking-aware".

    Args:
        input_dim: Embedding dimension
        dock_dim: Dimension of docking features per pair
        hidden_dims: Hidden dims for FiLM blocks
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        dock_dim: int,
        hidden_dims: Optional[List[int]] = None,
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

        # Delta encoder: takes [morgan_diff, dock_diff]
        delta_input_dim = input_dim + dock_dim
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

        # Main output: delta prediction
        self.delta_output = nn.Linear(prev_dim, 1)

        # Auxiliary output: Vina score prediction from embedding
        self.vina_head = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, 1),
        )

    def forward_single(self, x: torch.Tensor, delta_cond: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, delta_cond)
        return self.delta_output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        dock_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Forward pass returning delta prediction only."""
        delta = emb_b - emb_a
        delta_aug = torch.cat([delta, dock_feats], dim=-1)
        delta_cond = self.delta_encoder(delta_aug)

        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)
        return pred_b - pred_a

    def forward_multitask(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        dock_feats: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass returning (delta_pred, vina_pred_a, vina_pred_b)."""
        delta = emb_b - emb_a
        delta_aug = torch.cat([delta, dock_feats], dim=-1)
        delta_cond = self.delta_encoder(delta_aug)

        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)
        delta_pred = pred_b - pred_a

        vina_a = self.vina_head(emb_a).squeeze(-1)
        vina_b = self.vina_head(emb_b).squeeze(-1)

        return delta_pred, vina_a, vina_b


# ═══════════════════════════════════════════════════════════════════════════
# Architecture 3: FeatureGatedFiLM
# ═══════════════════════════════════════════════════════════════════════════

class FeatureGatedFiLM(nn.Module):
    """FiLM with per-feature learned gates on docking input.

    Each docking feature dimension gets a learned gate (sigmoid)
    that controls how much it contributes. This is a form of
    automatic feature selection that can zero out uninformative features.

    Args:
        input_dim: Embedding dimension
        dock_dim: Dimension of docking features
        hidden_dims: Hidden dims for FiLM blocks
        dropout: Dropout probability
    """

    def __init__(
        self,
        input_dim: int,
        dock_dim: int,
        hidden_dims: Optional[List[int]] = None,
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

        # Per-feature gate: learned importance per docking feature
        # Initialized to 0 (sigmoid(0) = 0.5, balanced start)
        self.feature_gate_logits = nn.Parameter(torch.zeros(dock_dim))

        # Delta encoder: [morgan_diff, gated_dock]
        delta_input_dim = input_dim + dock_dim
        delta_hidden = max(input_dim // 2, 64)
        self.delta_encoder = nn.Sequential(
            nn.Linear(delta_input_dim, delta_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

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

    def get_feature_importances(self) -> np.ndarray:
        """Return learned feature gate values (0 to 1)."""
        return torch.sigmoid(self.feature_gate_logits).detach().cpu().numpy()

    def forward_single(self, x: torch.Tensor, delta_cond: torch.Tensor) -> torch.Tensor:
        h = x
        for block in self.blocks:
            h = block(h, delta_cond)
        return self.output(h).squeeze(-1)

    def forward(
        self,
        emb_a: torch.Tensor,
        emb_b: torch.Tensor,
        dock_feats: torch.Tensor,
    ) -> torch.Tensor:
        delta = emb_b - emb_a

        # Apply per-feature gates
        gates = torch.sigmoid(self.feature_gate_logits)
        gated_dock = dock_feats * gates.unsqueeze(0)

        delta_aug = torch.cat([delta, gated_dock], dim=-1)
        delta_cond = self.delta_encoder(delta_aug)

        pred_a = self.forward_single(emb_a, delta_cond)
        pred_b = self.forward_single(emb_b, delta_cond)
        return pred_b - pred_a


# ═══════════════════════════════════════════════════════════════════════════
# Advanced Predictor Wrapper
# ═══════════════════════════════════════════════════════════════════════════

class AdvancedDockingFiLMPredictor:
    """High-level wrapper for advanced docking-aware FiLM architectures.

    Supports:
    - ResidualCorrectionFiLM ('residual')
    - MultiTaskDockingFiLM ('multitask')
    - FeatureGatedFiLM ('feature_gated')

    With additional features:
    - Cosine annealing LR schedule
    - Gradient clipping
    - Weight decay
    - Snapshot ensembling (save predictions at multiple epochs, average)
    """

    def __init__(
        self,
        arch: str = "residual",
        extra_dim: int = 3,
        hidden_dims: Optional[List[int]] = None,
        dropout: float = 0.2,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 100,
        patience: int = 15,
        device: Optional[str] = None,
        grad_clip: float = 1.0,
        aux_weight: float = 0.1,
        snapshot_ensemble: bool = False,
        n_snapshots: int = 5,
    ):
        self.arch = arch
        self.extra_dim = extra_dim
        self.hidden_dims = hidden_dims
        self.dropout = dropout
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.grad_clip = grad_clip
        self.aux_weight = aux_weight
        self.snapshot_ensemble = snapshot_ensemble
        self.n_snapshots = n_snapshots

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_built():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model = None
        self.snapshot_models = []  # For snapshot ensemble
        self.input_dim = None

    def _build_model(self, input_dim: int) -> nn.Module:
        if self.arch == "residual":
            return ResidualCorrectionFiLM(
                input_dim=input_dim,
                dock_dim=self.extra_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
        elif self.arch == "multitask":
            return MultiTaskDockingFiLM(
                input_dim=input_dim,
                dock_dim=self.extra_dim,
                hidden_dims=self.hidden_dims,
                dropout=self.dropout,
            )
        elif self.arch == "feature_gated":
            return FeatureGatedFiLM(
                input_dim=input_dim,
                dock_dim=self.extra_dim,
                hidden_dims=self.hidden_dims,
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
        vina_a_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
        vina_b_train: Optional[Union[np.ndarray, torch.Tensor]] = None,
        vina_a_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        vina_b_val: Optional[Union[np.ndarray, torch.Tensor]] = None,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Train the predictor."""
        emb_a_train = self._to_tensor(emb_a_train)
        emb_b_train = self._to_tensor(emb_b_train)
        dock_feats_train = self._to_tensor(dock_feats_train)
        delta_train = self._to_tensor(delta_train)

        self.input_dim = emb_a_train.shape[1]
        self.model = self._build_model(self.input_dim).to(self.device)

        if verbose:
            n_params = sum(p.numel() for p in self.model.parameters())
            print(f"\n  AdvancedDockingFiLM ({self.arch}):")
            print(f"    Input: {self.input_dim}d, Dock: {self.extra_dim}d")
            print(f"    Samples: {len(delta_train)}, Params: {n_params:,}")

        # Build dataloaders
        train_tensors = [emb_a_train, emb_b_train, dock_feats_train, delta_train]
        if self.arch == "multitask" and vina_a_train is not None:
            train_tensors.extend([
                self._to_tensor(vina_a_train),
                self._to_tensor(vina_b_train),
            ])
        train_loader = DataLoader(
            TensorDataset(*train_tensors),
            batch_size=self.batch_size, shuffle=True,
        )

        val_loader = None
        if all(v is not None for v in [emb_a_val, emb_b_val, dock_feats_val, delta_val]):
            val_tensors = [
                self._to_tensor(emb_a_val),
                self._to_tensor(emb_b_val),
                self._to_tensor(dock_feats_val),
                self._to_tensor(delta_val),
            ]
            if self.arch == "multitask" and vina_a_val is not None:
                val_tensors.extend([
                    self._to_tensor(vina_a_val),
                    self._to_tensor(vina_b_val),
                ])
            val_loader = DataLoader(
                TensorDataset(*val_tensors),
                batch_size=self.batch_size, shuffle=False,
            )

        # Optimizer with weight decay
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        # Cosine annealing with warm restarts
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=20, T_mult=2, eta_min=1e-5,
        )

        criterion = nn.MSELoss()
        history = {'train_loss': [], 'val_loss': []}
        best_val_loss = float('inf')
        best_state = None
        patience_counter = 0

        # Snapshot tracking
        snapshot_interval = max(1, self.max_epochs // self.n_snapshots)
        self.snapshot_models = []

        for epoch in range(self.max_epochs):
            # Train
            self.model.train()
            train_losses = []

            for batch in train_loader:
                batch = [t.to(self.device) for t in batch]
                batch_a, batch_b, batch_dock, batch_y = batch[:4]

                optimizer.zero_grad()

                if self.arch == "multitask" and len(batch) > 4:
                    batch_vina_a, batch_vina_b = batch[4], batch[5]
                    delta_pred, vina_pred_a, vina_pred_b = self.model.forward_multitask(
                        batch_a, batch_b, batch_dock
                    )
                    main_loss = criterion(delta_pred, batch_y)
                    aux_loss = (
                        criterion(vina_pred_a, batch_vina_a) +
                        criterion(vina_pred_b, batch_vina_b)
                    ) / 2
                    loss = main_loss + self.aux_weight * aux_loss
                else:
                    pred = self.model(batch_a, batch_b, batch_dock)
                    loss = criterion(pred, batch_y)

                loss.backward()

                # Gradient clipping
                if self.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(
                        self.model.parameters(), self.grad_clip
                    )

                optimizer.step()
                train_losses.append(loss.item())

            scheduler.step()
            train_loss = np.mean(train_losses)
            history['train_loss'].append(train_loss)

            # Validate
            if val_loader is not None:
                self.model.eval()
                val_losses = []
                with torch.no_grad():
                    for batch in val_loader:
                        batch = [t.to(self.device) for t in batch]
                        batch_a, batch_b, batch_dock, batch_y = batch[:4]
                        pred = self.model(batch_a, batch_b, batch_dock)
                        loss = criterion(pred, batch_y)
                        val_losses.append(loss.item())

                val_loss = np.mean(val_losses)
                history['val_loss'].append(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    best_state = {
                        k: v.cpu().clone() for k, v in self.model.state_dict().items()
                    }
                    patience_counter = 0
                else:
                    patience_counter += 1

                # Snapshot ensemble: save model at intervals
                if self.snapshot_ensemble and (epoch + 1) % snapshot_interval == 0:
                    snap = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                    self.snapshot_models.append(snap)

                if patience_counter >= self.patience:
                    if verbose:
                        print(f"    Early stop at epoch {epoch + 1}")
                    break
            else:
                history['val_loss'].append(train_loss)

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(self.device)

        if verbose:
            print(f"    Final: train={history['train_loss'][-1]:.4f}, "
                  f"val={history['val_loss'][-1]:.4f}")
            if self.arch == "feature_gated" and hasattr(self.model, 'get_feature_importances'):
                imps = self.model.get_feature_importances()
                print(f"    Feature gates: {np.round(imps, 3)}")

        return history

    def predict(
        self,
        emb_a: Union[np.ndarray, torch.Tensor],
        emb_b: Union[np.ndarray, torch.Tensor],
        dock_feats: Union[np.ndarray, torch.Tensor],
    ) -> np.ndarray:
        if self.model is None:
            raise RuntimeError("Model not trained!")

        emb_a = self._to_tensor(emb_a).to(self.device)
        emb_b = self._to_tensor(emb_b).to(self.device)
        dock_feats = self._to_tensor(dock_feats).to(self.device)

        if self.snapshot_ensemble and self.snapshot_models:
            # Average predictions across snapshots + best model
            all_preds = []
            for snap_state in self.snapshot_models:
                self.model.load_state_dict(snap_state)
                self.model = self.model.to(self.device)
                self.model.eval()
                with torch.no_grad():
                    p = self.model(emb_a, emb_b, dock_feats).cpu().numpy()
                all_preds.append(p)
            return np.mean(all_preds, axis=0)
        else:
            self.model.eval()
            with torch.no_grad():
                return self.model(emb_a, emb_b, dock_feats).cpu().numpy()

    def evaluate(
        self,
        emb_a: Union[np.ndarray, torch.Tensor],
        emb_b: Union[np.ndarray, torch.Tensor],
        dock_feats: Union[np.ndarray, torch.Tensor],
        delta_true: Union[np.ndarray, torch.Tensor],
    ) -> Tuple[Dict[str, float], np.ndarray, np.ndarray]:
        delta_true = self._to_numpy(delta_true)
        delta_pred = self.predict(emb_a, emb_b, dock_feats)

        mae = np.mean(np.abs(delta_pred - delta_true))
        if np.std(delta_pred) < 1e-6 or np.std(delta_true) < 1e-6:
            pearson = spearman = 0.0
        else:
            pearson, _ = scipy_stats.pearsonr(delta_pred, delta_true)
            spearman, _ = scipy_stats.spearmanr(delta_pred, delta_true)

        ss_res = np.sum((delta_true - delta_pred) ** 2)
        ss_tot = np.sum((delta_true - np.mean(delta_true)) ** 2)
        r2 = 1 - (ss_res / ss_tot) if ss_tot > 0 else 0.0

        metrics = {
            'mae': float(mae),
            'spearman': float(spearman),
            'pearson': float(pearson),
            'r2': float(r2),
        }
        return metrics, delta_true, delta_pred

    def _to_tensor(self, x):
        if isinstance(x, np.ndarray):
            return torch.from_numpy(x).float()
        return x.float()

    def _to_numpy(self, x):
        if isinstance(x, torch.Tensor):
            return x.cpu().numpy()
        return x

    @property
    def name(self) -> str:
        return f"AdvDockFiLM_{self.arch}"


# ═══════════════════════════════════════════════════════════════════════════
# Ensemble Predictor
# ═══════════════════════════════════════════════════════════════════════════

class EnsemblePredictor:
    """Ensemble of multiple trained predictors.

    Averages predictions from multiple models. Can combine different
    architectures or different seeds of the same architecture.
    """

    def __init__(self, predictors: List, weights: Optional[List[float]] = None):
        self.predictors = predictors
        if weights is None:
            self.weights = [1.0 / len(predictors)] * len(predictors)
        else:
            total = sum(weights)
            self.weights = [w / total for w in weights]

    def predict(self, emb_a, emb_b, dock_feats=None):
        preds = []
        for pred in self.predictors:
            if dock_feats is not None and hasattr(pred, 'extra_dim'):
                p = pred.predict(emb_a, emb_b, dock_feats)
            elif dock_feats is not None:
                try:
                    p = pred.predict(emb_a, emb_b, dock_feats)
                except TypeError:
                    p = pred.predict(emb_a, emb_b)
            else:
                p = pred.predict(emb_a, emb_b)
            preds.append(p)

        return sum(w * p for w, p in zip(self.weights, preds))
