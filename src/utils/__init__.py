"""
Utility modules for small molecule edit effect prediction.

Includes splitting strategies, metrics, embedding caching, and logging.
"""

from .logging import setup_logger
from .embedding_cache import (
    EmbeddingCache,
    get_or_compute_embeddings_for_pairs,
    get_or_compute_embeddings_for_molecules,
    compute_all_embeddings_once,
    compute_all_embeddings_with_fragments,
    map_embeddings_to_pairs,
    map_embeddings_to_molecules,
    map_fragment_embeddings_to_pairs
)
from .splits import (
    MolecularSplitter,
    RandomSplitter,
    ScaffoldSplitter,
    TargetSplitter,
    ButinaSplitter,
    PropertyStratifiedSplitter,
    TemporalSplitter,
    FewShotTargetSplitter,
    CoreSplitter,
    AssaySplitter,
    get_splitter
)
from .metrics import (
    RegressionMetrics,
    MultiTaskMetrics,
    RankingMetrics,
    ChemistryMetrics,
    print_metrics_summary
)

__all__ = [
    'setup_logger',
    'EmbeddingCache',
    'get_or_compute_embeddings_for_pairs',
    'get_or_compute_embeddings_for_molecules',
    'compute_all_embeddings_once',
    'compute_all_embeddings_with_fragments',
    'map_embeddings_to_pairs',
    'map_embeddings_to_molecules',
    'map_fragment_embeddings_to_pairs',
    'MolecularSplitter',
    'RandomSplitter',
    'ScaffoldSplitter',
    'TargetSplitter',
    'ButinaSplitter',
    'PropertyStratifiedSplitter',
    'TemporalSplitter',
    'FewShotTargetSplitter',
    'CoreSplitter',
    'AssaySplitter',
    'get_splitter',
    'RegressionMetrics',
    'MultiTaskMetrics',
    'RankingMetrics',
    'ChemistryMetrics',
    'print_metrics_summary',
]
