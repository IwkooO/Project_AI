"""
CBM library package.

This package is the canonical home for Phase 1 (concept prediction) and Phase 2
(geolocation) code. Scripts under `scripts/` should be thin wrappers around
functions in this package.
"""

from .phase1.model import Phase1CBMTopKMil
from .phase2.model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead

__all__ = ["Phase1CBMTopKMil", "ConceptEmbeddingAdapter", "Stage2CrossAttentionGeoHead"]


