"""
Small molecule embedding methods.
"""

from .base import MoleculeEmbedder
from .fingerprints import FingerprintEmbedder
from .edit_embedder import EditEmbedder
from .trainable_edit_embedder import TrainableEditEmbedder, ConcatenationEditEmbedder
from .structured_edit_embedder import StructuredEditEmbedder

__all__ = [
    'MoleculeEmbedder',
    'FingerprintEmbedder',
    'EditEmbedder',
    'TrainableEditEmbedder',
    'ConcatenationEditEmbedder',
    'StructuredEditEmbedder',
]

# Optional imports (require additional dependencies)
try:
    from .chemberta import ChemBERTaEmbedder
    __all__.append('ChemBERTaEmbedder')
except ImportError:
    ChemBERTaEmbedder = None

try:
    from .chemprop import ChemPropEmbedder
    __all__.append('ChemPropEmbedder')
except ImportError:
    ChemPropEmbedder = None

try:
    from .graphormer import GraphormerEmbedder
    __all__.append('GraphormerEmbedder')
except ImportError:
    GraphormerEmbedder = None

try:
    from .molfm import MolFMEmbedder
    __all__.append('MolFMEmbedder')
except ImportError:
    MolFMEmbedder = None

# Structured edit embedders (unified interface for local edit representations)
try:
    from .structured_edit_base import StructuredEditEmbedderBase
    __all__.append('StructuredEditEmbedderBase')
except ImportError:
    StructuredEditEmbedderBase = None

try:
    from .chemberta_structured import ChemBERTaStructuredEditEmbedder, chemberta2_structured_embedder
    __all__.extend(['ChemBERTaStructuredEditEmbedder', 'chemberta2_structured_embedder'])
except ImportError:
    ChemBERTaStructuredEditEmbedder = None
    chemberta2_structured_embedder = None

try:
    from .graphormer_structured import GraphormerStructuredEditEmbedder, graphormer_structured_embedder
    __all__.extend(['GraphormerStructuredEditEmbedder', 'graphormer_structured_embedder'])
except ImportError:
    GraphormerStructuredEditEmbedder = None
    graphormer_structured_embedder = None

try:
    from .molfm_structured import MolFMStructuredEditEmbedder, molfm_structured_embedder
    __all__.extend(['MolFMStructuredEditEmbedder', 'molfm_structured_embedder'])
except ImportError:
    MolFMStructuredEditEmbedder = None
    molfm_structured_embedder = None

try:
    from .graphmvp_structured import GraphMVPStructuredEditEmbedder, download_graphmvp_checkpoints
    __all__.extend(['GraphMVPStructuredEditEmbedder', 'download_graphmvp_checkpoints'])
except ImportError:
    GraphMVPStructuredEditEmbedder = None
    download_graphmvp_checkpoints = None

try:
    from .unimol import UniMolEmbedder, create_unimol_embedder
    __all__.extend(['UniMolEmbedder', 'create_unimol_embedder'])
except ImportError:
    UniMolEmbedder = None
    create_unimol_embedder = None

try:
    from .unimol_structured import UniMolStructuredEditEmbedder, unimol_structured_embedder
    __all__.extend(['UniMolStructuredEditEmbedder', 'unimol_structured_embedder'])
except ImportError:
    UniMolStructuredEditEmbedder = None
    unimol_structured_embedder = None
