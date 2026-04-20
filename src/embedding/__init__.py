"""
Small molecule embedding methods.
"""

from .base import MoleculeEmbedder
from .fingerprints import FingerprintEmbedder
from .edit_embedder import EditEmbedder
from .trainable_edit_embedder import TrainableEditEmbedder, ConcatenationEditEmbedder

__all__ = [
    'MoleculeEmbedder',
    'FingerprintEmbedder',
    'EditEmbedder',
    'TrainableEditEmbedder',
    'ConcatenationEditEmbedder',
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

try:
    from .molformer import MoLFormerEmbedder
    __all__.append('MoLFormerEmbedder')
except ImportError:
    MoLFormerEmbedder = None

try:
    from .unimol import UniMolEmbedder, create_unimol_embedder
    __all__.extend(['UniMolEmbedder', 'create_unimol_embedder'])
except ImportError:
    UniMolEmbedder = None
    create_unimol_embedder = None
