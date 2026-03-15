"""
Attention-based delta predictors for edit effect prediction.

These architectures use cross-attention mechanisms to integrate edit signals
with molecular context for property change prediction. They provide strong
generalization to unseen molecular contexts.

Two promoted architectures:

1. **GatedCrossAttn**: Cross-attention with learned sigmoid gating.
   The sigmoid gate prevents attention from overwhelming the edit signal,
   which is critical for generalization to unseen molecular contexts.

2. **AttnThenFiLM**: Hybrid 1-attention + 2-FiLM architecture.
   Attention provides coarse context integration,
   FiLM provides fine-grained element-wise modulation.

Architecture overview:
    - Edit signal: z = LinearProj(emb_edited - emb_original)              [B, H]
    - Context: z_ctx = LinearProj(emb_original)                            [B, H]
    - Edit features: ef = structural edit features (28-dim)                [B, F]
    - Context tokens: [ef, z_ctx] → Linear → reshape → [B, N_tokens, H]
    - Integration: cross-attention (GatedCrossAttn) or attention+FiLM
    - Head: [z, z_ctx] → MLP → Δproperty

Forward signature: forward(wt, mt, ef) → predicted delta [B]
    where wt, mt are [B, D] embeddings and ef is [B, EFD] edit features.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Union, Dict, Tuple
from scipy.stats import spearmanr

from src.data.utils.chemistry import compute_edit_features, EDIT_FEAT_DIM


# ── Building Blocks ──────────────────────────────────────────────────────────


class ResidualCrossAttnLayer(nn.Module):
    """Single residual cross-attention layer.

    Query (z) attends to multiple context tokens, followed by feedforward.
    Implements standard pre-norm transformer block adapted for single-vector
    queries attending to multiple context keys.

    Args:
        hidden_dim: Hidden dimension.
        n_heads: Number of attention heads.
        dropout: Dropout probability.
    """
    def __init__(self, hidden_dim: int = 256, n_heads: int = 4, dropout: float = 0.1):
        super().__init__()
        self.cross_attn = nn.MultiheadAttention(
            hidden_dim, n_heads, dropout=dropout, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.ff = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim * 2, hidden_dim))
        self.norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, z: torch.Tensor, ctx_tokens: torch.Tensor) -> torch.Tensor:
        """
        Args:
            z: Edit representation [B, H].
            ctx_tokens: Context tokens [B, N, H].

        Returns:
            Updated edit representation [B, H].
        """
        q = z.unsqueeze(1)  # [B, 1, H]
        attn_out, _ = self.cross_attn(q, ctx_tokens, ctx_tokens)  # [B, 1, H]
        z = self.norm1(z + attn_out.squeeze(1))
        z = self.norm2(z + self.ff(z))
        return z


# ── GatedCrossAttn ───────────────────────────────────────────────────────────


class GatedCrossAttnMLP(nn.Module):
    """Cross-attention with learned sigmoid gating for edit effect prediction.

    z = z + sigmoid(gate(z)) * CrossAttn(z, ctx_tokens)

    The sigmoid gate controls how much the attention output influences the
    edit representation at each layer. This prevents attention from
    overwhelming the edit signal — critical for generalization to unseen
    molecular contexts.

    Args:
        input_dim: Embedding dimension (default: 1024).
        mut_feat_dim: Edit feature dimension (default: 28).
        hidden_dim: Hidden dimension (default: 256).
        n_layers: Number of gated cross-attention layers (default: 3).
        n_tokens: Number of context aspect tokens (default: 4).
        n_heads: Number of attention heads (default: 4).
        dropout: Dropout probability for head (default: 0.2).
        attn_dropout: Dropout probability for attention (default: 0.1).
    """
    def __init__(
        self,
        input_dim: int = 1024,
        mut_feat_dim: int = EDIT_FEAT_DIM,
        hidden_dim: int = 256,
        n_layers: int = 3,
        n_tokens: int = 4,
        n_heads: int = 4,
        dropout: float = 0.2,
        attn_dropout: float = 0.1,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mut_feat_dim = mut_feat_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.sub_enc = nn.Linear(input_dim, hidden_dim)
        self.ctx_enc = nn.Linear(input_dim, hidden_dim)

        # Context → aspect tokens
        self.ctx_to_tokens = nn.Sequential(
            nn.Linear(mut_feat_dim + hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, n_tokens * hidden_dim))
        self.n_tokens = n_tokens

        # Per-layer: cross-attn + gate + norm + ff
        self.attns = nn.ModuleList()
        self.gates = nn.ModuleList()
        self.norms1 = nn.ModuleList()
        self.ffs = nn.ModuleList()
        self.norms2 = nn.ModuleList()
        for _ in range(n_layers):
            self.attns.append(nn.MultiheadAttention(
                hidden_dim, n_heads, dropout=attn_dropout, batch_first=True))
            self.gates.append(nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.Sigmoid()))
            self.norms1.append(nn.LayerNorm(hidden_dim))
            self.ffs.append(nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2), nn.GELU(),
                nn.Dropout(attn_dropout), nn.Linear(hidden_dim * 2, hidden_dim)))
            self.norms2.append(nn.LayerNorm(hidden_dim))

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        wt: torch.Tensor,
        mt: torch.Tensor,
        mf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            wt: Wildtype embeddings [B, D].
            mt: Mutant embeddings [B, D].
            mf: Mutation features [B, MFD].

        Returns:
            Predicted delta [B].
        """
        z = self.sub_enc(mt - wt)
        z_ctx = self.ctx_enc(wt)
        ctx_input = torch.cat([mf, z_ctx], -1)
        ctx_tokens = self.ctx_to_tokens(ctx_input).reshape(-1, self.n_tokens, self.hidden_dim)

        for attn, gate, norm1, ff, norm2 in zip(
                self.attns, self.gates, self.norms1, self.ffs, self.norms2):
            q = z.unsqueeze(1)
            attn_out, _ = attn(q, ctx_tokens, ctx_tokens)
            attn_out = attn_out.squeeze(1)
            g = gate(z)
            z = norm1(z + g * attn_out)
            z = norm2(z + ff(z))

        feat = torch.cat([z, z_ctx], -1)
        return self.head(feat).squeeze(-1)

    def get_edit_embedding(
        self,
        wt: torch.Tensor,
        mt: torch.Tensor,
        mf: torch.Tensor,
    ) -> torch.Tensor:
        """Extract the intermediate edit embedding (before prediction head).

        Useful for visualization, clustering, and linear probe analysis.

        Returns:
            Edit embedding [B, H*2] (concatenation of z and z_ctx).
        """
        z = self.sub_enc(mt - wt)
        z_ctx = self.ctx_enc(wt)
        ctx_input = torch.cat([mf, z_ctx], -1)
        ctx_tokens = self.ctx_to_tokens(ctx_input).reshape(-1, self.n_tokens, self.hidden_dim)

        for attn, gate, norm1, ff, norm2 in zip(
                self.attns, self.gates, self.norms1, self.ffs, self.norms2):
            q = z.unsqueeze(1)
            attn_out, _ = attn(q, ctx_tokens, ctx_tokens)
            attn_out = attn_out.squeeze(1)
            g = gate(z)
            z = norm1(z + g * attn_out)
            z = norm2(z + ff(z))

        return torch.cat([z, z_ctx], -1)


# ── AttnThenFiLM ─────────────────────────────────────────────────────────────


class AttnThenFiLMMLP(nn.Module):
    """Hybrid architecture: 1 cross-attention layer followed by 2 FiLM layers.

    Cross-attention provides coarse context integration (which aspects of the
    molecular context matter for this edit), then FiLM provides fine-grained
    element-wise modulation of the edit representation.

    Args:
        input_dim: Embedding dimension (default: 1024).
        mut_feat_dim: Edit feature dimension (default: 28).
        hidden_dim: Hidden dimension (default: 256).
        n_tokens: Number of context aspect tokens (default: 4).
        n_heads: Number of attention heads (default: 4).
        n_film_layers: Number of FiLM layers after attention (default: 2).
        dropout: Dropout probability for head (default: 0.2).
    """
    def __init__(
        self,
        input_dim: int = 1024,
        mut_feat_dim: int = EDIT_FEAT_DIM,
        hidden_dim: int = 256,
        n_tokens: int = 4,
        n_heads: int = 4,
        n_film_layers: int = 2,
        dropout: float = 0.2,
    ):
        super().__init__()
        self.input_dim = input_dim
        self.mut_feat_dim = mut_feat_dim
        self.hidden_dim = hidden_dim

        # Encoders
        self.sub_enc = nn.Linear(input_dim, hidden_dim)
        self.ctx_enc = nn.Linear(input_dim, hidden_dim)

        # Attention layer
        self.ctx_to_tokens = nn.Sequential(
            nn.Linear(mut_feat_dim + hidden_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, n_tokens * hidden_dim))
        self.n_tokens = n_tokens
        self.attn_layer = ResidualCrossAttnLayer(hidden_dim, n_heads)

        # FiLM layers
        self.film_layers = nn.ModuleList()
        self.film_norms = nn.ModuleList()
        for _ in range(n_film_layers):
            self.film_layers.append(nn.Sequential(
                nn.Linear(mut_feat_dim + hidden_dim, hidden_dim), nn.GELU(),
                nn.Linear(hidden_dim, hidden_dim * 2)))
            self.film_norms.append(nn.LayerNorm(hidden_dim))

        # Prediction head
        self.head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim), nn.GELU(),
            nn.Dropout(dropout), nn.Linear(hidden_dim, 1))

    def forward(
        self,
        wt: torch.Tensor,
        mt: torch.Tensor,
        mf: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            wt: Wildtype embeddings [B, D].
            mt: Mutant embeddings [B, D].
            mf: Mutation features [B, MFD].

        Returns:
            Predicted delta [B].
        """
        z = self.sub_enc(mt - wt)
        z_ctx = self.ctx_enc(wt)

        # Cross-attention (coarse context integration)
        ctx_input = torch.cat([mf, z_ctx], -1)
        ctx_tokens = self.ctx_to_tokens(ctx_input).reshape(-1, self.n_tokens, self.hidden_dim)
        z = self.attn_layer(z, ctx_tokens)

        # FiLM (fine-grained modulation)
        cond = torch.cat([mf, z_ctx], -1)
        for film_layer, norm in zip(self.film_layers, self.film_norms):
            film = film_layer(cond)
            gamma, beta = film.chunk(2, dim=-1)
            z = z + F.gelu(norm(gamma * z + beta))

        feat = torch.cat([z, z_ctx], -1)
        return self.head(feat).squeeze(-1)

    def get_edit_embedding(
        self,
        wt: torch.Tensor,
        mt: torch.Tensor,
        mf: torch.Tensor,
    ) -> torch.Tensor:
        """Extract intermediate edit embedding (before prediction head).

        Returns:
            Edit embedding [B, H*2] (concatenation of z and z_ctx).
        """
        z = self.sub_enc(mt - wt)
        z_ctx = self.ctx_enc(wt)

        ctx_input = torch.cat([mf, z_ctx], -1)
        ctx_tokens = self.ctx_to_tokens(ctx_input).reshape(-1, self.n_tokens, self.hidden_dim)
        z = self.attn_layer(z, ctx_tokens)

        cond = torch.cat([mf, z_ctx], -1)
        for film_layer, norm in zip(self.film_layers, self.film_norms):
            film = film_layer(cond)
            gamma, beta = film.chunk(2, dim=-1)
            z = z + F.gelu(norm(gamma * z + beta))

        return torch.cat([z, z_ctx], -1)


# ── Edit Feature Extraction ─────────────────────────────────────────────────

# Backward compatibility alias
MUT_FEAT_DIM = EDIT_FEAT_DIM


def compute_edit_features_tensor(
    edit_info: dict,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Compute edit feature tensor from an edit info dict.

    Wraps compute_edit_features() from chemistry utils and returns a torch tensor.

    Args:
        edit_info: Dict with keys 'mol_a', 'mol_b', 'edit_smiles'.
            Falls back to empty strings for missing keys.
        device: Target device.

    Returns:
        Feature tensor [EDIT_FEAT_DIM] on the specified device.
    """
    mol_a = edit_info.get('mol_a', '')
    mol_b = edit_info.get('mol_b', '')
    edit_smiles = edit_info.get('edit_smiles', '')

    feats = compute_edit_features(mol_a, mol_b, edit_smiles)
    return torch.tensor(feats, device=device, dtype=torch.float32)


# Backward compatibility: keep compute_mutation_features as an alias
def compute_mutation_features(
    mutations_or_edit_info,
    device: torch.device = torch.device('cpu'),
) -> torch.Tensor:
    """Backward-compatible wrapper that delegates to compute_edit_features_tensor.

    Accepts either the new edit_info dict format (with 'mol_a', 'mol_b', 'edit_smiles')
    or returns zeros for old-style mutation list input.
    """
    if isinstance(mutations_or_edit_info, dict):
        return compute_edit_features_tensor(mutations_or_edit_info, device)
    # Old-style mutation list: return zeros since amino acid features are meaningless
    return torch.zeros(EDIT_FEAT_DIM, device=device, dtype=torch.float32)


# ── High-Level Predictor ─────────────────────────────────────────────────────


def get_cosine_schedule(optimizer, num_epochs: int, warmup_epochs: int = 5):
    """Cosine annealing LR schedule with linear warmup."""
    def lr_lambda(epoch):
        if epoch < warmup_epochs:
            return epoch / warmup_epochs
        progress = (epoch - warmup_epochs) / (num_epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


class AttentionDeltaPredictor:
    """High-level predictor wrapping GatedCrossAttn or AttnThenFiLM.

    Handles training, evaluation, checkpointing, and edit embedding extraction.

    Data format: Each sample is a dict with keys:
        - 'wt': Original molecule embedding (numpy array or list).
        - 'mt': Edited molecule embedding (numpy array or list).
        - 'mol_a': SMILES of original molecule.
        - 'mol_b': SMILES of edited molecule.
        - 'edit_smiles': Edit in "leaving_frag>>incoming_frag" format.
        - 'z_delta': Target property delta (for training/eval).

    Example:
        >>> predictor = AttentionDeltaPredictor(arch='gated_cross_attn')
        >>> predictor.fit(train_data, val_data)
        >>> preds = predictor.predict(test_data)
        >>> embeddings = predictor.get_embeddings(test_data)

    Args:
        arch: Architecture name ('gated_cross_attn' or 'attn_then_film').
        input_dim: Embedding dimension.
        mut_feat_dim: Edit feature dimension (default: 28).
        hidden_dim: Hidden dimension.
        learning_rate: Learning rate.
        weight_decay: Weight decay.
        batch_size: Batch size.
        max_epochs: Maximum training epochs.
        patience: Early stopping patience.
        use_cosine: Use cosine annealing LR schedule.
        warmup_epochs: Number of warmup epochs for cosine schedule.
        device: Device string or None for auto-detect.
        **model_kwargs: Additional kwargs passed to model constructor.
    """

    ARCHITECTURES = {
        'gated_cross_attn': GatedCrossAttnMLP,
        'attn_then_film': AttnThenFiLMMLP,
    }

    def __init__(
        self,
        arch: str = 'gated_cross_attn',
        input_dim: int = 1024,
        mut_feat_dim: int = EDIT_FEAT_DIM,
        hidden_dim: int = 256,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-4,
        batch_size: int = 64,
        max_epochs: int = 80,
        patience: int = 15,
        use_cosine: bool = True,
        warmup_epochs: int = 5,
        device: Optional[str] = None,
        **model_kwargs,
    ):
        if arch not in self.ARCHITECTURES:
            raise ValueError(f"Unknown arch '{arch}'. Choose from {list(self.ARCHITECTURES.keys())}")

        self.arch = arch
        self.input_dim = input_dim
        self.mut_feat_dim = mut_feat_dim
        self.hidden_dim = hidden_dim
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.batch_size = batch_size
        self.max_epochs = max_epochs
        self.patience = patience
        self.use_cosine = use_cosine
        self.warmup_epochs = warmup_epochs
        self.model_kwargs = model_kwargs

        if device is None:
            if torch.cuda.is_available():
                self.device = 'cuda'
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                self.device = 'mps'
            else:
                self.device = 'cpu'
        else:
            self.device = device

        self.model = None

    def _create_model(self) -> nn.Module:
        cls = self.ARCHITECTURES[self.arch]
        return cls(
            input_dim=self.input_dim,
            mut_feat_dim=self.mut_feat_dim,
            hidden_dim=self.hidden_dim,
            **self.model_kwargs,
        )

    def fit(
        self,
        train_data: list,
        val_data: list,
        verbose: bool = True,
    ) -> float:
        """Train the model and return best validation Spearman ρ.

        Args:
            train_data: List of dicts with keys 'wt', 'mt', 'mol_a', 'mol_b', 'edit_smiles', 'z_delta'.
            val_data: List of dicts with same keys.
            verbose: Print progress.

        Returns:
            Best validation Spearman ρ.
        """
        dev = self.device
        self.model = self._create_model().to(dev)
        opt = torch.optim.AdamW(
            self.model.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = get_cosine_schedule(opt, self.max_epochs, self.warmup_epochs) if self.use_cosine else None
        crit = nn.MSELoss()

        n_params = sum(p.numel() for p in self.model.parameters())
        if verbose:
            print(f"  {self.arch}: {n_params:,} params, device={dev}")

        best_rho = -2.0
        best_state = None
        no_imp = 0

        for ep in range(self.max_epochs):
            # Train
            self.model.train()
            np.random.shuffle(train_data)
            for i in range(0, len(train_data), self.batch_size):
                batch = train_data[i:i + self.batch_size]
                wt = torch.stack([torch.tensor(d['wt'], dtype=torch.float32) for d in batch]).to(dev)
                mt = torch.stack([torch.tensor(d['mt'], dtype=torch.float32) for d in batch]).to(dev)
                mf = torch.stack([compute_edit_features_tensor(d, torch.device(dev)) for d in batch])
                tgt = torch.tensor([d['z_delta'] for d in batch], dtype=torch.float32).to(dev)

                opt.zero_grad()
                out = self.model(wt, mt, mf)
                loss = crit(out, tgt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                opt.step()

            if scheduler:
                scheduler.step()

            # Evaluate
            rho = self._eval_spearman(val_data)
            if rho > best_rho:
                best_rho = rho
                best_state = {k: v.cpu().clone() for k, v in self.model.state_dict().items()}
                no_imp = 0
            else:
                no_imp += 1
                if no_imp >= self.patience:
                    break

        # Restore best
        if best_state is not None:
            self.model.load_state_dict(best_state)
            self.model = self.model.to(dev)

        return best_rho

    def _eval_spearman(self, data: list) -> float:
        """Compute Spearman ρ on data."""
        self.model.eval()
        dev = self.device
        preds, tgts = [], []

        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                wt = torch.stack([torch.tensor(d['wt'], dtype=torch.float32) for d in batch]).to(dev)
                mt = torch.stack([torch.tensor(d['mt'], dtype=torch.float32) for d in batch]).to(dev)
                mf = torch.stack([compute_edit_features_tensor(d, torch.device(dev)) for d in batch])
                out = self.model(wt, mt, mf)
                preds.extend(out.cpu().tolist())
                tgts.extend([d['z_delta'] for d in batch])

        rho, _ = spearmanr(preds, tgts)
        return rho if not np.isnan(rho) else -2.0

    def predict(self, data: list) -> np.ndarray:
        """Predict delta values.

        Args:
            data: List of dicts with keys 'wt', 'mt', 'mol_a', 'mol_b', 'edit_smiles'.

        Returns:
            Predictions array [N].
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        self.model.eval()
        dev = self.device
        preds = []

        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                wt = torch.stack([torch.tensor(d['wt'], dtype=torch.float32) for d in batch]).to(dev)
                mt = torch.stack([torch.tensor(d['mt'], dtype=torch.float32) for d in batch]).to(dev)
                mf = torch.stack([compute_edit_features_tensor(d, torch.device(dev)) for d in batch])
                out = self.model(wt, mt, mf)
                preds.extend(out.cpu().tolist())

        return np.array(preds)

    def get_embeddings(self, data: list) -> np.ndarray:
        """Extract intermediate edit embeddings.

        Args:
            data: List of dicts with keys 'wt', 'mt', 'mol_a', 'mol_b', 'edit_smiles'.

        Returns:
            Embeddings array [N, H*2].
        """
        if self.model is None:
            raise RuntimeError("Model not trained yet!")

        self.model.eval()
        dev = self.device
        embs = []

        with torch.no_grad():
            for i in range(0, len(data), self.batch_size):
                batch = data[i:i + self.batch_size]
                wt = torch.stack([torch.tensor(d['wt'], dtype=torch.float32) for d in batch]).to(dev)
                mt = torch.stack([torch.tensor(d['mt'], dtype=torch.float32) for d in batch]).to(dev)
                mf = torch.stack([compute_edit_features_tensor(d, torch.device(dev)) for d in batch])
                emb = self.model.get_edit_embedding(wt, mt, mf)
                embs.append(emb.cpu().numpy())

        return np.concatenate(embs, axis=0)

    def evaluate(
        self,
        data: list,
    ) -> Dict[str, float]:
        """Evaluate model on data.

        Args:
            data: List of dicts with keys 'wt', 'mt', 'mol_a', 'mol_b', 'edit_smiles', 'z_delta'.

        Returns:
            Dict with spearman, mse, mae metrics.
        """
        preds = self.predict(data)
        tgts = np.array([d['z_delta'] for d in data])

        rho, p_val = spearmanr(preds, tgts)
        mse = np.mean((preds - tgts) ** 2)
        mae = np.mean(np.abs(preds - tgts))

        return {
            'spearman': float(rho) if not np.isnan(rho) else 0.0,
            'spearman_pval': float(p_val) if not np.isnan(p_val) else 1.0,
            'mse': float(mse),
            'mae': float(mae),
        }

    def save_checkpoint(self, path: str):
        """Save model checkpoint."""
        if self.model is None:
            raise ValueError("Model not trained yet.")

        torch.save({
            'model_state_dict': self.model.state_dict(),
            'arch': self.arch,
            'hyperparameters': {
                'input_dim': self.input_dim,
                'mut_feat_dim': self.mut_feat_dim,
                'hidden_dim': self.hidden_dim,
                'learning_rate': self.learning_rate,
                'weight_decay': self.weight_decay,
                **self.model_kwargs,
            },
        }, path)

    @classmethod
    def load_checkpoint(
        cls,
        path: str,
        device: Optional[str] = None,
    ) -> 'AttentionDeltaPredictor':
        """Load model from checkpoint."""
        checkpoint = torch.load(path, map_location='cpu', weights_only=False)
        hparams = checkpoint['hyperparameters']

        predictor = cls(
            arch=checkpoint['arch'],
            input_dim=hparams['input_dim'],
            mut_feat_dim=hparams['mut_feat_dim'],
            hidden_dim=hparams['hidden_dim'],
            learning_rate=hparams['learning_rate'],
            weight_decay=hparams['weight_decay'],
            device=device,
        )

        predictor.model = predictor._create_model()
        predictor.model.load_state_dict(checkpoint['model_state_dict'])
        predictor.model.to(predictor.device)
        predictor.model.eval()

        return predictor

    @property
    def name(self) -> str:
        return self.arch
