"""High-level prediction models and wrappers."""

from .film_delta_predictor import FiLMDeltaPredictor, FiLMDeltaMLP, FiLMLayer, FiLMBlock
from .attention_delta_predictor import (
    AttentionDeltaPredictor, GatedCrossAttnMLP, AttnThenFiLMMLP,
    ResidualCrossAttnLayer, compute_edit_features_tensor,
    compute_mutation_features, EDIT_FEAT_DIM, MUT_FEAT_DIM,
)

__all__ = [
    # FiLM-conditioned predictors
    'FiLMDeltaPredictor',
    'FiLMDeltaMLP',
    'FiLMLayer',
    'FiLMBlock',
    # Attention-based predictors
    'AttentionDeltaPredictor',
    'GatedCrossAttnMLP',
    'AttnThenFiLMMLP',
    'ResidualCrossAttnLayer',
    'compute_edit_features_tensor',
    'compute_mutation_features',
    'EDIT_FEAT_DIM',
    'MUT_FEAT_DIM',
]

# Optional imports
try:
    from .edit_aware_film_predictor import (
        DrfpFiLMDeltaMLP, DualStreamFiLMDeltaMLP,
        FragAnchoredFiLMDeltaMLP, MultiModalEditFiLMDeltaMLP,
        EditHypernetFiLMDeltaMLP,
    )
    __all__.extend([
        'DrfpFiLMDeltaMLP', 'DualStreamFiLMDeltaMLP',
        'FragAnchoredFiLMDeltaMLP', 'MultiModalEditFiLMDeltaMLP',
        'EditHypernetFiLMDeltaMLP',
    ])
except ImportError:
    pass

try:
    from .docking_film_predictor import DockingFiLMDeltaMLP
    __all__.append('DockingFiLMDeltaMLP')
except ImportError:
    pass

try:
    from .advanced_docking_film import (
        ResidualCorrectionFiLMDeltaMLP, MultiTaskFiLMDeltaMLP,
    )
    __all__.extend(['ResidualCorrectionFiLMDeltaMLP', 'MultiTaskFiLMDeltaMLP'])
except ImportError:
    pass
