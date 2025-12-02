"""
Loss functions for StreetCLIP CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn.functional as F

from src.evaluation import normalized_latlng_to_sphere, haversine_distance


@dataclass
class LossWeights:
    concept: float = 1.0
    distance: float = 1.0
    country: float = 1.0
    contrastive: float = 1.0
    divergence: float = 1.0


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
        
        # Normalize: Divide by 1000.0 so that 1.0 loss ~= 1000km error
        # This balances the magnitude with CrossEntropy (~0.5 - 5.0)
        normalized_distances = distances / 1000.0
        return normalized_distances.mean()

    raise ValueError(f"Unsupported coordinate loss type '{loss_type}'")


def contrastive_alignment_loss(
    z_img: torch.Tensor,
    z_loc: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Computes the symmetric contrastive loss (InfoNCE) between image and location concept vectors.
    
    Args:
        z_img: Image concept activations [batch, k]
        z_loc: Location concept activations [batch, k]
        temperature: Scaling factor for logits
    
    Returns:
        Scalar loss
    """
    # Normalize features to use cosine similarity? 
    # The PDF formula uses dot product: exp(z_img . z_loc / tau)
    # If z are unnormalized, dot product is fine but magnitude matters.
    # CLIP usually normalizes. Let's normalize to be safe and match CLIP-style behavior.
    z_img_norm = F.normalize(z_img, p=2, dim=1)
    z_loc_norm = F.normalize(z_loc, p=2, dim=1)
    
    logits = torch.matmul(z_img_norm, z_loc_norm.t()) / temperature
    labels = torch.arange(len(z_img), device=z_img.device)
    
    loss_i2l = F.cross_entropy(logits, labels)
    loss_l2i = F.cross_entropy(logits.t(), labels)
    
    return (loss_i2l + loss_l2i) / 2.0


def concept_divergence_loss(
    z_img: torch.Tensor,
    z_loc: torch.Tensor,
    sigma: float = 1.0
) -> torch.Tensor:
    """
    Computes the Concept Space Divergence Loss using Gaussian Kernel MMD.
    Formula (6) in the PDF.
    
    L_concept = -1/N^2 * sum(log K(z_img_i, z_img_j)) 
                -1/N^2 * sum(log K(z_loc_i, z_loc_j))
                + 2/N^2 * sum(log K(z_img_i, z_loc_j))
                
    Actually, the PDF formula (6) is slightly weird:
    L_concept = 1/N^2 sum[ log K(z_img, z_img) + log K(z_loc, z_loc) - 2 log K(z_img, z_loc) ]
    
    This looks like minimizing: E[log K(x,x)] + E[log K(y,y)] - 2 E[log K(x,y)]
    Which relates to Cauchy-Schwarz Divergence or Information Potential.
    
    K(x, y) = exp(-||x - y||^2 / 2*sigma^2)
    log K(x, y) = -||x - y||^2 / 2*sigma^2
    
    If we take log K directly, it simplifies to just L2 distance terms?
    Let's look closer at the formula.
    The formula in the PDF is:
    L_concept = 1/N^2 sum_{i,j} [ log K(...) + log K(...) - 2 log K(...) ]
    
    Since log(exp(-d^2/2s^2)) = -d^2/2s^2, this effectively minimizes:
    - (Mean Intra-class Distances) + 2 * (Mean Inter-class Distances)
    Wait, minimizing (-Distance) means Maximizing Distance?
    
    Minimizing L_concept = Minimize [ -Dist(Img,Img) - Dist(Loc,Loc) + 2*Dist(Img,Loc) ]
    = Maximize [ Dist(Img,Img) + Dist(Loc,Loc) ] - Minimize [ 2*Dist(Img,Loc) ]
    = Maximize Spread within modalities AND Minimize Distance between modalities.
    
    This matches "distributional alignment" + "modality-invariant representation".
    
    Args:
        z_img: Image concept activations [batch, k]
        z_loc: Location concept activations [batch, k]
        sigma: Kernel bandwidth
        
    Returns:
        Scalar loss
    """
    # We will implement based on the Euclidean distance interpretation of the log-Gaussian kernel
    # for numerical stability and efficiency.
    
    def compute_pairwise_sq_distances(x, y):
        """Compute squared Euclidean distances between all pairs of rows in x and y."""
        # x: [N, D], y: [M, D]
        # dist_sq[i, j] = ||x[i] - y[j]||^2
        # = ||x[i]||^2 + ||y[j]||^2 - 2 <x[i], y[j]>
        x_norm = (x**2).sum(1).view(-1, 1)
        y_norm = (y**2).sum(1).view(1, -1)
        dist_sq = x_norm + y_norm - 2.0 * torch.mm(x, y.t())
        return torch.clamp(dist_sq, min=0.0)

    N = z_img.size(0)
    scale = 1.0 / (2 * sigma**2)
    
    dist_img_img = compute_pairwise_sq_distances(z_img, z_img)
    dist_loc_loc = compute_pairwise_sq_distances(z_loc, z_loc)
    dist_img_loc = compute_pairwise_sq_distances(z_img, z_loc)
    
    # Term 1: log K(z_img, z_img) = -dist_sq / 2sigma^2
    term1 = -scale * dist_img_img.mean()
    
    # Term 2: log K(z_loc, z_loc)
    term2 = -scale * dist_loc_loc.mean()
    
    # Term 3: -2 * log K(z_img, z_loc) = -2 * (-dist_sq / ...) = + 2 * scale * dist
    term3 = 2 * scale * dist_img_loc.mean() # Note the + sign because of -(-...)
    
    # But wait, the formula (6) has -2 log K(...)
    # So: sum ( ... - 2 log K )
    # = sum ( -d_ii - d_jj - 2(-d_ij) )
    # = sum ( -d_ii - d_jj + 2d_ij )
    
    # Let's re-read closely:
    # L = 1/N^2 sum [ log K(img,img) + log K(loc,loc) - 2 log K(img,loc) ]
    # log K = -d^2
    # L ~ -d(img,img) - d(loc,loc) + 2d(img,loc)
    
    # So minimizing L means:
    # 1. Minimizing 2d(img,loc) -> Make image and location close (Alignment)
    # 2. Minimizing -d(img,img) -> Maximizing d(img,img) -> Spread out images (Uniformity/Diversity)
    # 3. Minimizing -d(loc,loc) -> Maximizing d(loc,loc) -> Spread out locations
    
    return term1 + term2 + term3


def clip_contrastive_loss(
    emb_a: torch.Tensor,
    emb_b: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Standard CLIP-style symmetric contrastive loss (InfoNCE).
    
    Used for:
    - Concept-Text Alignment: concept_emb ↔ text_emb (note embeddings)
    - Can also be used for Concept-GPS alignment in simple cases
    
    Args:
        emb_a: First embedding [batch, dim] (e.g., concept embeddings)
        emb_b: Second embedding [batch, dim] (e.g., text embeddings)
        temperature: Temperature scaling factor
        
    Returns:
        Scalar loss (symmetric cross-entropy)
    """
    # L2 normalize embeddings
    emb_a_norm = F.normalize(emb_a, p=2, dim=1)
    emb_b_norm = F.normalize(emb_b, p=2, dim=1)
    
    # Compute similarity matrix
    logits = torch.matmul(emb_a_norm, emb_b_norm.t()) / temperature
    
    # Labels: diagonal entries are positive pairs
    labels = torch.arange(len(emb_a), device=emb_a.device)
    
    # Symmetric loss
    loss_a2b = F.cross_entropy(logits, labels)
    loss_b2a = F.cross_entropy(logits.t(), labels)
    
    return (loss_a2b + loss_b2a) / 2.0


def geocell_contrastive_loss(
    concept_emb: torch.Tensor,
    gps_emb: torch.Tensor,
    cell_labels: torch.Tensor,
    temperature: float = 0.07,
    eps: float = 1e-8
) -> torch.Tensor:
    """
    Geocell-aware contrastive loss for Concept-GPS alignment.
    
    Positive pairs: samples in the SAME geocell
    Negative pairs: samples in DIFFERENT geocells
    
    This encourages concept embeddings to align with GPS embeddings
    of nearby locations (same geocell), while pushing apart embeddings
    from distant locations (different geocells).
    
    Args:
        concept_emb: Concept embeddings [batch, dim]
        gps_emb: GPS embeddings [batch, dim]
        cell_labels: Geocell labels for each sample [batch]
        temperature: Temperature scaling factor
        eps: Small constant for numerical stability
        
    Returns:
        Scalar loss
    """
    batch_size = concept_emb.size(0)
    
    # L2 normalize embeddings
    concept_norm = F.normalize(concept_emb, p=2, dim=1)
    gps_norm = F.normalize(gps_emb, p=2, dim=1)
    
    # Compute similarity matrix: [batch, batch]
    sim = torch.matmul(concept_norm, gps_norm.t()) / temperature
    
    # Create positive mask: samples in the SAME geocell
    # pos_mask[i, j] = 1 if cell_labels[i] == cell_labels[j]
    pos_mask = (cell_labels.unsqueeze(0) == cell_labels.unsqueeze(1)).float()
    
    # For numerical stability, we use log-sum-exp trick
    # For each row i, we want:
    #   -log( sum_{j in positive} exp(sim[i,j]) / sum_{k} exp(sim[i,k]) )
    
    # Compute log partition function (log of sum over all)
    log_sum_exp_all = torch.logsumexp(sim, dim=1)  # [batch]
    
    # Compute log of sum over positives
    # Mask out negatives with large negative value before logsumexp
    neg_inf_mask = (1 - pos_mask) * (-1e9)
    sim_pos_only = sim + neg_inf_mask
    log_sum_exp_pos = torch.logsumexp(sim_pos_only, dim=1)  # [batch]
    
    # Loss: -log(positive_sum / total_sum) = log_sum_exp_all - log_sum_exp_pos
    loss_c2g = (log_sum_exp_all - log_sum_exp_pos).mean()
    
    # Symmetric: GPS to Concept direction
    sim_t = sim.t()
    log_sum_exp_all_t = torch.logsumexp(sim_t, dim=1)
    sim_pos_only_t = sim_t + neg_inf_mask.t()
    log_sum_exp_pos_t = torch.logsumexp(sim_pos_only_t, dim=1)
    loss_g2c = (log_sum_exp_all_t - log_sum_exp_pos_t).mean()
    
    return (loss_c2g + loss_g2c) / 2.0


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
