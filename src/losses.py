"""
Loss functions for StreetCLIP CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class LossWeights:
    concept: float = 1.0
    distance: float = 1.0
    country: float = 0.5


def concept_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def country_loss(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    return F.cross_entropy(logits, targets)


def coordinate_loss(pred_coords: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(target_coords).any(dim=1)
    if mask.sum() == 0:
        return torch.zeros(1, device=pred_coords.device, dtype=pred_coords.dtype).squeeze()
    return F.mse_loss(pred_coords[mask], target_coords[mask])


def combined_loss(
    concept_logits: torch.Tensor,
    country_logits: torch.Tensor,
    predicted_coords: torch.Tensor,
    concept_targets: torch.Tensor,
    country_targets: torch.Tensor,
    coordinate_targets: torch.Tensor,
    weights: Optional[LossWeights] = None,
) -> torch.Tensor:
    weights = weights or LossWeights()

    losses = {}
    losses["concept"] = concept_loss(concept_logits, concept_targets) * weights.concept
    losses["country"] = country_loss(country_logits, country_targets) * weights.country
    losses["distance"] = coordinate_loss(predicted_coords, coordinate_targets) * weights.distance

    total = sum(losses.values())
    return total, losses




