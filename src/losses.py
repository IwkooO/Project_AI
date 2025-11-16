"""
Loss functions for StreetCLIP CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.evaluation import normalized_latlng_to_sphere


@dataclass
class LossWeights:
    concept: float = 1.0
    distance: float = 1.0
    country: float = 0.5


def concept_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def country_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def coordinate_loss(
    pred_coords: torch.Tensor,
    target_coords: torch.Tensor,
    loss_type: str = "mse",
) -> torch.Tensor:
    mask = ~torch.isnan(target_coords).any(dim=1)
    if mask.sum() == 0:
        return torch.zeros(1, device=pred_coords.device, dtype=pred_coords.dtype).squeeze()
    loss_type = loss_type.lower()

    if loss_type == "sphere":
        pred_sphere = F.normalize(pred_coords[mask], p=2, dim=1)
        target_sphere = normalized_latlng_to_sphere(target_coords[mask])
        cosine_distance = 1.0 - torch.sum(pred_sphere * target_sphere, dim=1)
        return cosine_distance.mean()

    if loss_type == "mse":
        return F.mse_loss(pred_coords[mask], target_coords[mask])

    raise ValueError(f"Unsupported coordinate loss type '{loss_type}'")


def combined_loss(
    concept_logits: torch.Tensor,
    country_logits: torch.Tensor,
    predicted_coords: torch.Tensor,
    concept_targets: torch.Tensor,
    country_targets: torch.Tensor,
    coordinate_targets: torch.Tensor,
    weights: Optional[LossWeights] = None,
    coordinate_loss_type: str = "mse",
) -> torch.Tensor:
    weights = weights or LossWeights()

    losses = {}
    losses["concept"] = concept_loss(concept_logits, concept_targets) * weights.concept
    losses["country"] = country_loss(country_logits, country_targets) * weights.country
    losses["distance"] = (
        coordinate_loss(predicted_coords, coordinate_targets, coordinate_loss_type)
        * weights.distance
    )

    total = sum(losses.values())
    return total, losses




