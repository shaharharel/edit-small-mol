import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Dict
from src.models import PropertyPredictor, EditEffectPredictor, StructuredEditEffectPredictor
from src.models.predictors import TrainableEditEffectPredictor

# TrainablePropertyPredictor does not yet exist; guard to prevent ImportError
try:
    from src.models.predictors import TrainablePropertyPredictor
except ImportError:
    TrainablePropertyPredictor = None
from src.models.predictors.attention_delta_predictor import AttentionDeltaPredictor
from src.embedding import ChemBERTaEmbedder, ChemPropEmbedder, EditEmbedder
from src.embedding.trainable_edit_embedder import ConcatenationEditEmbedder
from src.embedding.edit_embedder import ConcatEditEmbedder

# Optional imports for new embedders
try:
    from src.embedding import GraphormerEmbedder
except ImportError:
    GraphormerEmbedder = None

try:
    from src.embedding import MolFMEmbedder
except ImportError:
    MolFMEmbedder = None


def is_embedder_trainable(embedder) -> bool:
    """Check if an embedder supports trainable mode."""
    return hasattr(embedder, 'trainable') and embedder.trainable


def create_embedder(embedder_type: str, trainable_encoder: bool = False, encoder_device: str = 'auto'):
    """
    Create molecule embedder with optional trainable encoder.

    Args:
        embedder_type: Type of embedder ('chemberta', 'chemprop', 'chemprop_dmpnn', etc.)
        trainable_encoder: Whether to make encoder parameters trainable (for GNN or transformer)
        encoder_device: Device for encoder ('cpu', 'cuda', or 'auto' for auto-detect)
    """
    # Auto-detect device if requested
    if encoder_device == 'auto':
        import torch
        if torch.cuda.is_available():
            encoder_device = 'cuda'
        elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
            encoder_device = 'mps'
        else:
            encoder_device = 'cpu'
        print(f"Auto-detected device: {encoder_device}")

    if embedder_type == 'chemberta':
        return ChemBERTaEmbedder(trainable=trainable_encoder, device=encoder_device)
    elif embedder_type == 'chemberta2' or embedder_type == 'chemberta2-mlm':
        # ChemBERTa-2 77M MLM (recommended)
        return ChemBERTaEmbedder(
            model_name='chemberta2-mlm',
            trainable=trainable_encoder,
            device=encoder_device
        )
    elif embedder_type == 'chemberta2-mtr':
        # ChemBERTa-2 77M MTR (with property prediction pretraining)
        return ChemBERTaEmbedder(
            model_name='chemberta2-mtr',
            trainable=trainable_encoder,
            device=encoder_device
        )
    elif embedder_type == 'chemprop':
        # Default: Morgan fingerprints (CPU-based, 2048-dim)
        return ChemPropEmbedder()
    elif embedder_type == 'chemprop_dmpnn':
        # D-MPNN graph neural network (GPU-capable, 300-dim)
        return ChemPropEmbedder(
            featurizer_type='graph',
            trainable=trainable_encoder,
            device=encoder_device
        )
    elif embedder_type == 'chemprop_morgan':
        # Explicit Morgan fingerprints (same as 'chemprop' default)
        return ChemPropEmbedder(featurizer_type='morgan')
    elif embedder_type == 'chemprop_rdkit':
        # RDKit 2D descriptors (CPU-based, 217-dim)
        return ChemPropEmbedder(featurizer_type='rdkit2d')
    elif embedder_type == 'graphormer':
        # Microsoft Graphormer (graph transformer)
        if GraphormerEmbedder is None:
            raise ImportError(
                "Graphormer requires transformers library. "
                "Install with: pip install transformers"
            )
        return GraphormerEmbedder(
            trainable=trainable_encoder,
            device=encoder_device,
            backend='transformers'
        )
    elif embedder_type == 'molfm':
        # MolFM multimodal foundation model
        if MolFMEmbedder is None:
            raise ImportError(
                "MolFM requires transformers library. "
                "Install with: pip install transformers"
            )
        return MolFMEmbedder(
            trainable=trainable_encoder,
            device=encoder_device,
            modality='multimodal'
        )
    elif embedder_type == 'molfm-sequence':
        # MolFM sequence-only (uses SMILES)
        if MolFMEmbedder is None:
            raise ImportError("MolFM requires transformers library.")
        return MolFMEmbedder(
            trainable=trainable_encoder,
            device=encoder_device,
            modality='sequence'
        )
    elif embedder_type == 'molfm-graph':
        # MolFM graph-only
        if MolFMEmbedder is None:
            raise ImportError("MolFM requires transformers library.")
        return MolFMEmbedder(
            trainable=trainable_encoder,
            device=encoder_device,
            modality='graph'
        )
    else:
        raise ValueError(f"Unknown embedder type: {embedder_type}")


def create_models(config, train_data: Dict, embedder=None) -> Dict:
    """
    Create models for each method in config.

    Args:
        config: ExperimentConfig with methods list
        train_data: Dict of task data
        embedder: Optional default embedder. If None, each method must specify embedder_type.

    Returns:
        Dict of models keyed by method name
    """
    models = {}
    task_names = list(train_data.keys())

    for method_config in config.methods:
        method_name = method_config['name']
        method_type = method_config['type']

        # Each method can specify its own embedder_type and trainable_encoder
        # Fall back to experiment-level defaults if not specified
        # Support both new (trainable_encoder) and legacy (trainable_gnn) parameter names
        if 'embedder_type' in method_config:
            trainable = method_config.get('trainable_encoder',
                        method_config.get('trainable_gnn',
                        getattr(config, 'trainable_encoder',
                        getattr(config, 'trainable_gnn', False))))
            device = method_config.get('encoder_device',
                     method_config.get('gnn_device',
                     getattr(config, 'encoder_device',
                     getattr(config, 'gnn_device', 'auto'))))
            method_embedder = create_embedder(
                embedder_type=method_config['embedder_type'],
                trainable_encoder=trainable,
                encoder_device=device
            )
        elif embedder is not None:
            method_embedder = embedder
        else:
            raise ValueError(
                f"Method '{method_name}' does not have embedder_type specified and no default embedder provided"
            )

        if method_type == 'baseline_property':
            # Use trainable predictor if embedder supports it
            if is_embedder_trainable(method_embedder):
                # Support both new (encoder_lr) and legacy (gnn_lr) parameter names
                encoder_lr = method_config.get('encoder_lr', method_config.get('gnn_lr', 1e-5))
                model = TrainablePropertyPredictor(
                    embedder=method_embedder,
                    task_names=task_names,
                    hidden_dims=method_config.get('hidden_dims'),
                    head_hidden_dims=method_config.get('head_hidden_dims'),
                    dropout=method_config.get('dropout', 0.2),
                    mlp_learning_rate=method_config.get('lr', 0.001),
                    encoder_learning_rate=encoder_lr,
                    batch_size=method_config.get('batch_size', 32),
                    max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50))
                )
                models[method_name] = {
                    'type': 'trainable_baseline_property',
                    'embedder': method_embedder,
                    'model': model,
                    'config': method_config
                }
            else:
                model = PropertyPredictor(
                    embedder=method_embedder,
                    task_names=task_names,
                    hidden_dims=method_config.get('hidden_dims'),
                    head_hidden_dims=method_config.get('head_hidden_dims'),
                    dropout=method_config.get('dropout', 0.2),
                    learning_rate=method_config.get('lr', 0.001),
                    batch_size=method_config.get('batch_size', 32),
                    max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50))
                )
                models[method_name] = {
                    'type': 'baseline_property',
                    'embedder': method_embedder,
                    'model': model,
                    'config': method_config
                }

        elif method_type == 'edit_framework':
            # Use trainable predictor if embedder supports it
            if is_embedder_trainable(method_embedder):
                # Support both new (encoder_lr) and legacy (gnn_lr) parameter names
                encoder_lr = method_config.get('encoder_lr', method_config.get('gnn_lr', 1e-5))
                model = TrainableEditEffectPredictor(
                    embedder=method_embedder,
                    task_names=task_names,
                    hidden_dims=method_config.get('hidden_dims'),
                    head_hidden_dims=method_config.get('head_hidden_dims'),
                    dropout=method_config.get('dropout', 0.2),
                    mlp_learning_rate=method_config.get('lr', 0.001),
                    encoder_learning_rate=encoder_lr,
                    batch_size=method_config.get('batch_size', 32),
                    max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50)),
                    use_edit_fragments=method_config.get('use_edit_fragments', False),
                    trainable_edit_hidden_dims=method_config.get('trainable_edit_dims', [512, 256])
                )
                models[method_name] = {
                    'type': 'trainable_edit_framework',
                    'mol_embedder': method_embedder,
                    'model': model,
                    'config': method_config
                }
            else:
                edit_embedder = EditEmbedder(method_embedder)
                # Support both new (encoder_lr) and legacy (gnn_lr) parameter names
                encoder_lr = method_config.get('encoder_lr', method_config.get('gnn_lr', 1e-5))

                model = EditEffectPredictor(
                    mol_embedder=method_embedder,
                    edit_embedder=edit_embedder,
                    task_names=task_names,
                    hidden_dims=method_config.get('hidden_dims'),
                    head_hidden_dims=method_config.get('head_hidden_dims'),
                    dropout=method_config.get('dropout', 0.2),
                    learning_rate=method_config.get('lr', 0.001),
                    gnn_learning_rate=encoder_lr,  # EditEffectPredictor still uses gnn_learning_rate
                    batch_size=method_config.get('batch_size', 32),
                    max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50)),
                    trainable_edit_embeddings=method_config.get('trainable_edit_embeddings', True),
                    trainable_edit_hidden_dims=method_config.get('trainable_edit_dims', [512, 256]),
                    trainable_edit_use_fragments=method_config.get('use_edit_fragments', False)
                )

                models[method_name] = {
                    'type': 'edit_framework',
                    'mol_embedder': method_embedder,
                    'edit_embedder': edit_embedder,
                    'model': model,
                    'config': method_config
                }

        elif method_type == 'edit_framework_structured':
            # StructuredEditEffectPredictor - uses MMP structural information
            # Requires: chemprop_dmpnn embedder (for atom-level embeddings)
            if not hasattr(method_embedder, 'message_passing'):
                raise ValueError(
                    "edit_framework_structured requires a graph-based embedder "
                    "(use embedder_type='chemprop_dmpnn')"
                )
            # Support both new (encoder_lr) and legacy (gnn_lr) parameter names
            encoder_lr = method_config.get('encoder_lr', method_config.get('gnn_lr', 1e-5))

            model = StructuredEditEffectPredictor(
                mol_embedder=method_embedder,
                gnn_dim=method_embedder.embedding_dim,
                edit_mlp_dims=method_config.get('edit_mlp_dims', [512, 512, 300]),
                delta_mlp_dims=method_config.get('delta_mlp_dims', method_config.get('hidden_dims', [512, 256, 128])),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 0.001),
                encoder_learning_rate=encoder_lr,
                batch_size=method_config.get('batch_size', 32),
                max_epochs=method_config.get('max_epochs', method_config.get('epochs', 50)),
                task_names=task_names,
                task_weights=method_config.get('task_weights'),
                k_hop_env=method_config.get('k_hop_env', 2),
                use_local_delta=method_config.get('use_local_delta', True),
                use_rdkit_fragment_descriptors=method_config.get('use_rdkit_descriptors', True)
            )

            models[method_name] = {
                'type': 'edit_framework_structured',
                'mol_embedder': method_embedder,
                'model': model,
                'config': method_config
            }

        elif method_type == 'attention_delta':
            # AttentionDeltaPredictor: GatedCrossAttn or AttnThenFiLM
            # Uses explicit edit embedding z = sub_enc(mt - wt) with cross-attention
            # conditioned on 28-dim chemical edit features + molecular context.
            # This is our primary edit-aware model.
            #
            # Required config keys:
            #   arch: 'gated_cross_attn' (default) or 'attn_then_film'
            #   embedder_type: e.g. 'chemberta', 'chemprop_dmpnn'
            # Optional:
            #   hidden_dim: int (default 256)
            #   n_layers: int for gated_cross_attn (default 3)
            #   n_tokens: int (default 4)
            #   n_heads: int (default 4)
            #   n_film_layers: int for attn_then_film (default 2)
            #   dropout: float (default 0.2)
            #   attn_dropout: float (default 0.1)
            #   lr: float (default 3e-4)
            #   weight_decay: float (default 1e-4)
            #   batch_size: int (default 64)
            #   max_epochs / epochs: int (default 100)
            #   patience: int (default 15)
            #   use_cosine: bool (default True)
            #   warmup_epochs: int (default 5)
            arch = method_config.get('arch', 'gated_cross_attn')
            input_dim = method_embedder.embedding_dim

            model_kwargs = {}
            if arch == 'gated_cross_attn':
                model_kwargs['n_layers'] = method_config.get('n_layers', 3)
                model_kwargs['attn_dropout'] = method_config.get('attn_dropout', 0.1)
            elif arch == 'attn_then_film':
                model_kwargs['n_film_layers'] = method_config.get('n_film_layers', 2)

            model = AttentionDeltaPredictor(
                arch=arch,
                input_dim=input_dim,
                hidden_dim=method_config.get('hidden_dim', 256),
                n_tokens=method_config.get('n_tokens', 4),
                n_heads=method_config.get('n_heads', 4),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 3e-4),
                weight_decay=method_config.get('weight_decay', 1e-4),
                batch_size=method_config.get('batch_size', 64),
                max_epochs=method_config.get('max_epochs', method_config.get('epochs', 100)),
                patience=method_config.get('patience', 15),
                use_cosine=method_config.get('use_cosine', True),
                warmup_epochs=method_config.get('warmup_epochs', 5),
                **model_kwargs,
            )

            models[method_name] = {
                'type': 'attention_delta',
                'mol_embedder': method_embedder,
                'model': model,
                'config': method_config
            }

        elif method_type == 'deepdelta':
            # DeepDelta-style baseline: concat([emb_a, emb_b]) → MLP → Δ
            # Mirrors the DeepDelta paper approach of pairwise concatenation.
            # Uses ConcatEditEmbedder: encodes (mol_a, mol_b) via learned MLP on concat.
            # Fair comparison: same molecule embedder as our attention_delta model.
            #
            # Required config keys:
            #   embedder_type: e.g. 'chemberta', 'chemprop_dmpnn'
            # Optional:
            #   concat_hidden_dims: list[int] for inner concat MLP (default [mol_dim, mol_dim//2])
            #   hidden_dims: list[int] for outer prediction MLP (default auto)
            #   dropout: float (default 0.2)
            #   lr: float (default 1e-3)
            #   batch_size: int (default 32)
            #   max_epochs / epochs: int (default 100)
            mol_dim = method_embedder.embedding_dim

            # Inner MLP that maps concat([emb_a, emb_b]) → edit_vector of dim mol_dim
            concat_nn = ConcatenationEditEmbedder(
                mol_dim=mol_dim,
                edit_dim=mol_dim,
                hidden_dims=method_config.get('concat_hidden_dims', [mol_dim, mol_dim // 2]),
                dropout=method_config.get('dropout', 0.1),
            )

            # Wrapper with encode_from_smiles() API compatible with EditEffectPredictor
            concat_edit_embedder = ConcatEditEmbedder(
                molecule_embedder=method_embedder,
                concat_module=concat_nn,
            )

            model = EditEffectPredictor(
                mol_embedder=method_embedder,
                edit_embedder=concat_edit_embedder,
                task_names=task_names,
                hidden_dims=method_config.get('hidden_dims'),
                dropout=method_config.get('dropout', 0.2),
                learning_rate=method_config.get('lr', 1e-3),
                batch_size=method_config.get('batch_size', 32),
                max_epochs=method_config.get('max_epochs', method_config.get('epochs', 100)),
                trainable_edit_embeddings=False,
            )

            models[method_name] = {
                'type': 'deepdelta',
                'mol_embedder': method_embedder,
                'edit_embedder': concat_edit_embedder,
                'model': model,
                'config': method_config
            }

    return models
