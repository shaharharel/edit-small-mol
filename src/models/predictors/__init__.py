"""High-level prediction models and wrappers."""

from .property_predictor import PropertyPredictor, PropertyPredictorMLP
from .edit_effect_predictor import EditEffectPredictor, EditEffectMLP
from .structured_edit_effect_predictor_v2 import StructuredEditEffectPredictor, StructuredEditEffectMLP
from .trainable_edit_effect_predictor import TrainableEditEffectPredictor, TrainableEditEffectMLP
from .baseline_property_predictor import BaselinePropertyPredictor, BaselinePropertyMLP
from .film_delta_predictor import FiLMDeltaPredictor, FiLMDeltaMLP, FiLMLayer, FiLMBlock
from .attention_delta_predictor import (
    AttentionDeltaPredictor, GatedCrossAttnMLP, AttnThenFiLMMLP,
    ResidualCrossAttnLayer, compute_mutation_features, MUT_FEAT_DIM,
)

__all__ = [
    # Pre-computed embedding predictors
    'PropertyPredictor',
    'PropertyPredictorMLP',
    'EditEffectPredictor',
    'EditEffectMLP',
    'StructuredEditEffectPredictor',
    'StructuredEditEffectMLP',
    # End-to-end trainable predictors
    'TrainableEditEffectPredictor',
    'TrainableEditEffectMLP',
    # Baseline (non-edit) predictor
    'BaselinePropertyPredictor',
    'BaselinePropertyMLP',
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
    'compute_mutation_features',
    'MUT_FEAT_DIM',
]
