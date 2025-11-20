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
    distillation: float = 0.0


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

    if loss_type == "vmf":
        return mixture_vmf_loss(pred_coords[mask], target_coords[mask])

    raise ValueError(f"Unsupported coordinate loss type '{loss_type}'")


def log_vmf_prob(mu: torch.Tensor, kappa: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    """
    Log probability of vMF distribution in 3D.
    mu: (B, K, 3) or (B, 3)
    kappa: (B, K, 1) or (B, 1)
    x: (B, 1, 3) or (B, 3)
    Returns: (B, K) or (B,)
    """
    # Broadcast shapes
    if x.dim() == 2 and mu.dim() == 3:
        x = x.unsqueeze(1)
    
    # dot product
    dot_prod = torch.sum(mu * x, dim=-1)
    
    # kappa
    k = kappa.squeeze(-1)
    
    # Stable log(sinh(k)) calculation
    # For large k (> 20), log(sinh(k)) approx k - log(2)
    # For small k, log(sinh(k))
    
    log_sinh = torch.zeros_like(k)
    large_k_mask = k > 20.0
    
    log_sinh[large_k_mask] = k[large_k_mask] - 0.69314718  # log(2)
    log_sinh[~large_k_mask] = torch.log(torch.sinh(k[~large_k_mask]) + 1e-6)
    
    # log(C(k)) = log(k) - log(4*pi) - log(sinh(k))
    # log(4*pi) approx 2.53102425
    log_C = torch.log(k + 1e-6) - 2.53102425 - log_sinh
    
    return log_C + k * dot_prod


def mixture_vmf_loss(pred_params: torch.Tensor, target_coords: torch.Tensor) -> torch.Tensor:
    """
    Negative Log Likelihood for Mixture of von Mises-Fisher distributions.
    
    Args:
        pred_params: (B, 5*K) flattened parameters.
                     Layout per mixture k: [mu_x, mu_y, mu_z, kappa, logit_pi]
        target_coords: (B, 3) unit vectors.
    """
    B = pred_params.shape[0]
    K = pred_params.shape[1] // 5
    
    params = pred_params.view(B, K, 5)
    
    # Extract components
    mu_raw = params[:, :, :3]
    kappa_raw = params[:, :, 3:4]
    pi_logits = params[:, :, 4]
    
    # Normalize mu
    mu = F.normalize(mu_raw, p=2, dim=2)
    
    # Positivity for kappa (softplus) + epsilon to avoid 0
    kappa = F.softplus(kappa_raw) + 1e-6
    
    # Log softmax for mixing weights
    log_pi = F.log_softmax(pi_logits, dim=1)
    
    # Calculate log prob for each component
    # target_coords: (B, 3) -> broadcast to (B, 1, 3) inside log_vmf_prob
    log_probs_k = log_vmf_prob(mu, kappa, target_coords) # (B, K)
    
    # Mixture log prob: log(sum(pi_k * prob_k)) = log(sum(exp(log_pi + log_prob_k)))
    # Use logsumexp for stability
    log_prob = torch.logsumexp(log_pi + log_probs_k, dim=1)
    
    return -log_prob.mean()


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


def distillation_loss(
    student_features: torch.Tensor,
    teacher_features: torch.Tensor,
    temperature: float = 1.0
) -> torch.Tensor:
    """
    Distillation loss (KL Divergence) between student and teacher features.
    Assuming features are logits or normalized embeddings.
    """
    # If features are embeddings, we might want cosine distance or MSE
    # If they are logits, KL div.
    
    # For GeoCLIP embeddings (normalized), MSE or Cosine is often used.
    # Cosine Embedding Loss: 1 - cos(x, y)
    
    # Check if normalized
    if torch.abs(student_features.norm(p=2, dim=1).mean() - 1.0) < 1e-2:
        # Normalized embeddings -> Cosine distance
        # loss = 1 - cosine_similarity
        cos_sim = F.cosine_similarity(student_features, teacher_features, dim=1)
        return (1.0 - cos_sim).mean()
    
    # Otherwise MSE
    return F.mse_loss(student_features, teacher_features)


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
    teacher_features: Optional[torch.Tensor] = None,
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

    if (
        weights.distillation > 0
        and teacher_features is not None
        and image_features is not None
    ):
        losses["distillation"] = (
            distillation_loss(image_features, teacher_features)
            * weights.distillation
        )

    total = sum(losses.values())
    return total, losses
