"""
Loss functions for StreetCLIP CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F

from src.evaluation import normalized_latlng_to_sphere, haversine_distance


@dataclass
class LossWeights:
    concept: float = 1.0
    distance: float = 1.0
    country: float = 1.0
    contrastive: float = 0.1
    divergence: float = 0.1


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

    if loss_type == "haversine":
        distances = haversine_distance(pred_coords[mask], target_coords[mask])
        if distances.numel() == 0:
            return torch.zeros(
                1, device=pred_coords.device, dtype=pred_coords.dtype
            ).squeeze()
        return distances.mean()

    raise ValueError(f"Unsupported coordinate loss type '{loss_type}'")


def image_gps_contrastive_loss(
    image_features: torch.Tensor,
    location_features: torch.Tensor,
    temperature: float = 0.07,
) -> torch.Tensor:
    """
    InfoNCE loss between image and location features.
    Assumes one-to-one correspondence between batch elements.
    """
    # Normalize features
    image_features = F.normalize(image_features, p=2, dim=1)
    location_features = F.normalize(location_features, p=2, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(image_features, location_features.T) / temperature
    
    # Targets are diagonal (0, 1, 2, ...)
    batch_size = logits.shape[0]
    targets = torch.arange(batch_size, device=logits.device)
    
    # Symmetric loss (Image -> GPS and GPS -> Image)
    loss_i2g = F.cross_entropy(logits, targets)
    loss_g2i = F.cross_entropy(logits.T, targets)
    
    return (loss_i2g + loss_g2i) / 2


def concept_space_divergence_loss(
    image_concept_logits: torch.Tensor,
    location_concept_logits: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    KL Divergence between image concept distribution and location concept distribution.
    """
    # Use log_softmax for input and softmax for target for KLDivLoss
    # We want to align both distributions. Symmetric KL is often used.
    
    img_log_probs = F.log_softmax(image_concept_logits / temperature, dim=1)
    loc_log_probs = F.log_softmax(location_concept_logits / temperature, dim=1)
    
    img_probs = F.softmax(image_concept_logits / temperature, dim=1)
    loc_probs = F.softmax(location_concept_logits / temperature, dim=1)
    
    # KL(Image || Location)
    kl_i2l = F.kl_div(img_log_probs, loc_probs, reduction="batchmean")
    
    # KL(Location || Image)
    kl_l2i = F.kl_div(loc_log_probs, img_probs, reduction="batchmean")
    
    return (kl_i2l + kl_l2i) / 2


def combined_loss(
    concept_logits: torch.Tensor,
    country_logits: torch.Tensor,
    predicted_coords: torch.Tensor,
    concept_targets: torch.Tensor,
    country_targets: torch.Tensor,
    coordinate_targets: torch.Tensor,
    # New arguments for alignment losses
    image_features: Optional[torch.Tensor] = None,
    location_features: Optional[torch.Tensor] = None,
    location_concept_logits: Optional[torch.Tensor] = None,
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

    # Alignment losses
    if (
        weights.contrastive > 0 
        and image_features is not None 
        and location_features is not None
    ):
        losses["contrastive"] = (
            image_gps_contrastive_loss(image_features, location_features)
            * weights.contrastive
        )
        
    if (
        weights.divergence > 0
        and location_concept_logits is not None
    ):
        losses["divergence"] = (
            concept_space_divergence_loss(concept_logits, location_concept_logits)
            * weights.divergence
    )

    total = sum(losses.values())
    return total, losses
