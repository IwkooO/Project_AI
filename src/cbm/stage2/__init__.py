"""
Stage 2 geolocation package.

Stage 2 consumes cached StreetCLIP embeddings + a frozen Phase1 concept model
(here: `query_topk_256`) to produce geolocation predictions.
"""

from src.cbm.stage2.dataset import Stage2Dataset, collate_fn_stage2
from src.cbm.stage2.geocells import (
    assign_geocells,
    compute_offsets,
    fit_semantic_geocells,
    latlng_to_xyz,
)
from src.cbm.stage2.metrics import (
    cell_accuracy,
    haversine_km,
    threshold_accuracies_km,
    xyz_to_latlng,
)
from src.cbm.stage2.models import (
    ConceptEmbeddingAdapter,
    Stage2CrossAttentionGeoHead,
)

__all__ = [
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
    "ConceptEmbeddingAdapter",
    "Stage2CrossAttentionGeoHead",
]


