"""
MMPDB-based MMP extraction scripts.

This module provides tools to extract Matched Molecular Pairs using the mmpdb package,
which is optimized for large-scale MMP generation.
"""

from .extract_pairs_mmpdb import MMPDBExtractor

__all__ = ['MMPDBExtractor']
