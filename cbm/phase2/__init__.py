"""Phase 2: geolocation."""

from .model import ConceptEmbeddingAdapter, Stage2CrossAttentionGeoHead
from .data import Stage2Dataset, collate_fn_stage2
from .geocells import fit_semantic_geocells, assign_geocells, compute_offsets, latlng_to_xyz
from .metrics import haversine_km, threshold_accuracies_km, cell_accuracy, xyz_to_latlng

__all__ = [
    "ConceptEmbeddingAdapter",
    "Stage2CrossAttentionGeoHead",
    "Stage2Dataset",
    "collate_fn_stage2",
    "fit_semantic_geocells",
    "assign_geocells",
    "compute_offsets",
    "latlng_to_xyz",
    "haversine_km",
    "threshold_accuracies_km",
    "cell_accuracy",
    "xyz_to_latlng",
]


