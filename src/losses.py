"""
Loss functions for StreetCLIP CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, Dict

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.evaluation import normalized_latlng_to_sphere, haversine_distance


# ============================================================================
# SEMANTIC SIMILARITY UTILITIES
# ============================================================================

def compute_concept_similarity_matrix(
    prototypes: torch.Tensor,
    margin: float = 0.0,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Compute pairwise cosine similarity matrix between concept prototypes.
    
    Args:
        prototypes: L2-normalized text prototype embeddings [num_concepts, dim]
        margin: Minimum similarity threshold (values below are clipped to 0)
        temperature: Temperature for sharpening/softening similarities
        
    Returns:
        Similarity matrix [num_concepts, num_concepts] with values in [0, 1]
    """
    # Ensure normalized
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
    # Cosine similarity: [num_concepts, num_concepts]
    sim_matrix = torch.matmul(prototypes_norm, prototypes_norm.T)
    
    # Apply margin clipping (values below margin become 0)
    if margin > 0:
        sim_matrix = torch.clamp(sim_matrix - margin, min=0) / (1 - margin + 1e-8)
    
    # Temperature scaling (higher temp = softer, lower = sharper)
    if temperature != 1.0:
        sim_matrix = sim_matrix ** (1.0 / temperature)
    
    return sim_matrix


class SemanticSoftCrossEntropy(nn.Module):
    """
    Semantic Soft Cross-Entropy Loss for concept classification.
    
    Instead of only penalizing wrong predictions, this loss:
    1. Gives partial credit for predicting semantically similar concepts
    2. Creates soft targets based on prototype similarity
    3. Blends hard CE loss with semantic similarity loss
    
    This helps prevent overfitting by allowing the model to learn that
    "Beach - tropical" and "Beach - temperate" are semantically close,
    rather than treating them as completely unrelated classes.
    
    Loss = (1 - λ) * HardCE(pred, target) + λ * SoftCE(pred, soft_targets)
    
    Where soft_targets[i] ∝ sim(prototype[target], prototype[i])
    """
    
    def __init__(
        self,
        similarity_matrix: torch.Tensor,
        lambda_soft: float = 0.2,
        temperature: float = 2.0,
        normalize_soft_targets: bool = True,
    ):
        """
        Args:
            similarity_matrix: Precomputed [num_concepts, num_concepts] similarity matrix
            lambda_soft: Weight for soft target loss (0 = pure hard CE, 1 = pure soft)
            temperature: Temperature for softening the soft target distribution
            normalize_soft_targets: Whether to normalize soft targets to sum to 1
        """
        super().__init__()
        self.register_buffer('similarity_matrix', similarity_matrix)
        self.lambda_soft = lambda_soft
        self.temperature = temperature
        self.normalize_soft_targets = normalize_soft_targets
    
    def forward(
        self, 
        logits: torch.Tensor, 
        targets: torch.Tensor,
        return_components: bool = False,
    ) -> torch.Tensor:
        """
        Args:
            logits: Predicted logits [batch, num_concepts]
            targets: Ground truth concept indices [batch]
            return_components: If True, return dict with loss components
            
        Returns:
            Combined loss (scalar) or dict with components if return_components=True
        """
        batch_size = logits.size(0)
        num_classes = logits.size(1)
        device = logits.device
        
        # 1. Hard cross-entropy loss
        hard_loss = F.cross_entropy(logits, targets)
        
        # 2. Build soft targets from similarity matrix
        # For each sample, soft_targets[i] = similarity(gt_concept, concept_i)
        # Shape: [batch, num_classes]
        soft_targets = self.similarity_matrix[targets]  # [batch, num_classes]
        
        # Apply temperature to soften/sharpen the distribution
        if self.temperature != 1.0:
            soft_targets = soft_targets ** (1.0 / self.temperature)
        
        # Normalize to probability distribution
        if self.normalize_soft_targets:
            soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
        
        # 3. Soft cross-entropy: CE between predictions and soft targets
        log_probs = F.log_softmax(logits, dim=1)
        soft_loss = -(soft_targets * log_probs).sum(dim=1).mean()
        
        # 4. Combine losses
        combined_loss = (1 - self.lambda_soft) * hard_loss + self.lambda_soft * soft_loss
        
        if return_components:
            return {
                'loss': combined_loss,
                'hard_loss': hard_loss,
                'soft_loss': soft_loss,
            }
        return combined_loss
    
    def update_similarity_matrix(self, new_matrix: torch.Tensor):
        """Update the similarity matrix (e.g., if prototypes change during training)."""
        self.similarity_matrix = new_matrix.to(self.similarity_matrix.device)


def semantic_soft_cross_entropy(
    logits: torch.Tensor,
    targets: torch.Tensor,
    similarity_matrix: torch.Tensor,
    lambda_soft: float = 0.2,
    temperature: float = 2.0,
) -> torch.Tensor:
    """
    Functional version of SemanticSoftCrossEntropy.
    
    Args:
        logits: Predicted logits [batch, num_concepts]
        targets: Ground truth concept indices [batch]
        similarity_matrix: Precomputed [num_concepts, num_concepts] similarity matrix
        lambda_soft: Weight for soft target loss
        temperature: Temperature for soft targets
        
    Returns:
        Combined loss (scalar)
    """
    # Hard cross-entropy
    hard_loss = F.cross_entropy(logits, targets)
    
    # Build soft targets
    soft_targets = similarity_matrix[targets]  # [batch, num_classes]
    if temperature != 1.0:
        soft_targets = soft_targets ** (1.0 / temperature)
    soft_targets = soft_targets / soft_targets.sum(dim=1, keepdim=True).clamp(min=1e-8)
    
    # Soft cross-entropy
    log_probs = F.log_softmax(logits, dim=1)
    soft_loss = -(soft_targets * log_probs).sum(dim=1).mean()
    
    # Combine
    return (1 - lambda_soft) * hard_loss + lambda_soft * soft_loss


def compute_semantic_close_accuracy(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    similarity_matrix: torch.Tensor,
    threshold: float = 0.7,
) -> Tuple[float, float]:
    """
    Compute both hard accuracy and semantic-close accuracy.
    
    Semantic-close accuracy counts a prediction as correct if:
    - It matches exactly (hard correct), OR
    - It's within top-k most similar concepts to the ground truth
    - The similarity between predicted and target concept >= threshold
    
    Args:
        predictions: Predicted class indices [batch]
        targets: Ground truth class indices [batch]
        similarity_matrix: [num_concepts, num_concepts] similarity matrix
        threshold: Minimum similarity to count as "close" prediction
        
    Returns:
        Tuple of (hard_accuracy, semantic_close_accuracy)
    """
    batch_size = predictions.size(0)
    
    # Hard accuracy
    hard_correct = (predictions == targets).float()
    hard_acc = hard_correct.mean().item()
    
    # Semantic-close accuracy
    # Get similarity between predicted and target concepts
    pred_target_sim = similarity_matrix[targets, predictions]  # [batch]
    semantic_close = (pred_target_sim >= threshold).float()
    
    # A prediction is "semantically close" if it's either exactly correct OR similar enough
    close_correct = torch.max(hard_correct, semantic_close)
    semantic_close_acc = close_correct.mean().item()
    
    return hard_acc, semantic_close_acc


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


class FocalLoss(torch.nn.Module):
    """
    Focal Loss for handling class imbalance and hard examples.
    
    FL(p_t) = -α_t * (1 - p_t)^γ * log(p_t)
    
    When γ=0, equivalent to cross-entropy. Higher γ focuses more on hard examples.
    
    Args:
        gamma: Focusing parameter (default 2.0). Higher values focus more on hard examples.
        alpha: Class weights tensor [num_classes] or None. If None, no class weighting.
        reduction: 'mean', 'sum', or 'none'
        label_smoothing: Label smoothing factor (default 0.0)
    """
    
    def __init__(
        self,
        gamma: float = 2.0,
        alpha: Optional[torch.Tensor] = None,
        reduction: str = 'mean',
        label_smoothing: float = 0.0
    ):
        super().__init__()
        self.gamma = gamma
        self.reduction = reduction
        self.label_smoothing = label_smoothing
        # Register alpha as buffer so it moves with model to device
        if alpha is not None:
            self.register_buffer('alpha', alpha)
        else:
            self.alpha = None
    
    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        """
        Args:
            logits: Predictions [batch, num_classes]
            targets: Ground truth labels [batch] (class indices)
            
        Returns:
            Focal loss (scalar if reduction='mean' or 'sum')
        """
        num_classes = logits.size(1)
        
        # Apply label smoothing to create soft targets
        if self.label_smoothing > 0:
            # One-hot encode targets
            targets_one_hot = F.one_hot(targets, num_classes).float()
            # Smooth: (1 - ε) * one_hot + ε / num_classes
            targets_smooth = (1 - self.label_smoothing) * targets_one_hot + self.label_smoothing / num_classes
        else:
            targets_smooth = None
        
        # Compute softmax probabilities
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)
        
        if targets_smooth is not None:
            # Soft targets: compute weighted sum
            # p_t for each class weighted by target probability
            p_t = (probs * targets_smooth).sum(dim=1)
            log_p_t = (log_probs * targets_smooth).sum(dim=1)
        else:
            # Hard targets: gather probability of correct class
            p_t = probs.gather(1, targets.unsqueeze(1)).squeeze(1)
            log_p_t = log_probs.gather(1, targets.unsqueeze(1)).squeeze(1)
        
        # Focal weight: (1 - p_t)^γ
        focal_weight = (1 - p_t) ** self.gamma
        
        # Focal loss: -α * (1 - p_t)^γ * log(p_t)
        focal_loss = -focal_weight * log_p_t
        
        # Apply class weights (alpha)
        if self.alpha is not None:
            alpha_t = self.alpha.gather(0, targets)
            focal_loss = alpha_t * focal_loss
        
        # Reduction
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


def concept_prototype_contrastive_loss(
    embeddings: torch.Tensor,
    prototypes: torch.Tensor,
    labels: torch.Tensor,
    temperature: float = 0.07
) -> torch.Tensor:
    """
    Contrastive loss that pulls embeddings toward correct prototype and pushes away from others.
    
    This is essentially a softmax cross-entropy over prototype similarities, but framed
    as contrastive learning to leverage the semantic structure of prototypes.
    
    loss = -log(exp(sim(emb, T[label])/τ) / Σ_j exp(sim(emb, T[j])/τ))
    
    Args:
        embeddings: Image/concept embeddings [batch, dim]
        prototypes: Text prototype embeddings [num_classes, dim]
        labels: Ground truth class indices [batch]
        temperature: Temperature for softmax scaling
        
    Returns:
        Scalar loss
    """
    # L2 normalize embeddings and prototypes
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    prototypes_norm = F.normalize(prototypes, p=2, dim=1)
    
    # Compute similarity matrix: [batch, num_classes]
    sim = torch.matmul(embeddings_norm, prototypes_norm.t()) / temperature
    
    # Cross-entropy loss with labels as targets
    loss = F.cross_entropy(sim, labels)
    
    return loss


def hierarchical_consistency_loss(
    meta_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    meta_to_parent_idx: torch.Tensor,
    temperature: float = 1.0,
) -> torch.Tensor:
    """
    Hierarchical consistency loss that enforces agreement between meta and parent predictions.
    
    Key insight: If the model predicts meta concept X with high probability, it should also
    predict the parent concept that X belongs to. This loss penalizes inconsistent predictions.
    
    For each sample:
    1. Compute expected parent distribution from meta predictions: P(parent_j) = Σ_{i ∈ children(j)} P(meta_i)
    2. Compute KL divergence between this expected parent distribution and actual parent predictions
    
    Args:
        meta_logits: Meta concept logits [batch, num_metas]
        parent_logits: Parent concept logits [batch, num_parents]
        meta_to_parent_idx: Mapping from meta index to parent index [num_metas]
        temperature: Temperature for softening distributions (higher = softer)
        
    Returns:
        Scalar loss (KL divergence)
    """
    batch_size = meta_logits.size(0)
    num_metas = meta_logits.size(1)
    num_parents = parent_logits.size(1)
    device = meta_logits.device
    
    # Get meta probabilities (softened by temperature)
    meta_probs = F.softmax(meta_logits / temperature, dim=1)  # [batch, num_metas]
    
    # Compute expected parent distribution from meta predictions
    # For each parent j: P(parent_j | meta) = Σ_{i: parent(i)=j} P(meta_i)
    # Create one-hot mapping: [num_metas, num_parents] where M[i,j] = 1 if meta i belongs to parent j
    meta_to_parent_onehot = F.one_hot(meta_to_parent_idx, num_classes=num_parents).float()  # [num_metas, num_parents]
    
    # Expected parent distribution: [batch, num_parents]
    expected_parent_probs = torch.matmul(meta_probs, meta_to_parent_onehot)  # [batch, num_parents]
    
    # Get actual parent probabilities
    actual_parent_probs = F.softmax(parent_logits / temperature, dim=1)  # [batch, num_parents]
    
    # KL divergence: KL(expected || actual) = Σ expected * log(expected / actual)
    # Use log_softmax for numerical stability
    actual_parent_log_probs = F.log_softmax(parent_logits / temperature, dim=1)
    
    # Add small epsilon to avoid log(0)
    eps = 1e-8
    expected_parent_probs = expected_parent_probs + eps
    expected_parent_probs = expected_parent_probs / expected_parent_probs.sum(dim=1, keepdim=True)
    
    # KL divergence
    kl_div = F.kl_div(actual_parent_log_probs, expected_parent_probs, reduction='batchmean')
    
    return kl_div


def parent_guided_meta_loss(
    meta_logits: torch.Tensor,
    parent_logits: torch.Tensor,
    meta_labels: torch.Tensor,
    parent_labels: torch.Tensor,
    meta_to_parent_idx: torch.Tensor,
    hard_mask: bool = False,
    soft_temperature: float = 2.0,
) -> torch.Tensor:
    """
    Parent-guided meta classification loss.
    
    Instead of predicting over all meta classes equally, this loss:
    1. Uses parent predictions to create a soft mask over meta concepts
    2. Upweights meta concepts belonging to the predicted/correct parent
    3. Downweights meta concepts from other parents
    
    This effectively reduces the problem from 1-of-N to a hierarchical decision.
    
    Args:
        meta_logits: Meta concept logits [batch, num_metas]
        parent_logits: Parent concept logits [batch, num_parents]
        meta_labels: Ground truth meta labels [batch]
        parent_labels: Ground truth parent labels [batch]
        meta_to_parent_idx: Mapping from meta index to parent index [num_metas]
        hard_mask: If True, use hard gating (only concepts from predicted parent)
                   If False, use soft gating based on parent probabilities
        soft_temperature: Temperature for soft gating (lower = sharper)
        
    Returns:
        Scalar loss
    """
    batch_size = meta_logits.size(0)
    num_metas = meta_logits.size(1)
    num_parents = parent_logits.size(1)
    device = meta_logits.device
    
    # Create mapping matrix: [num_metas, num_parents]
    meta_to_parent_onehot = F.one_hot(meta_to_parent_idx, num_classes=num_parents).float()
    
    if hard_mask:
        # Hard gating: use ground truth parent to mask
        # Only allow meta concepts that belong to the correct parent
        parent_mask = meta_to_parent_onehot[:, parent_labels].T  # [batch, num_metas]
        # Apply mask: set logits of non-matching metas to -inf
        masked_logits = meta_logits + (1 - parent_mask) * (-1e9)
    else:
        # Soft gating: weight meta logits by parent probabilities
        parent_probs = F.softmax(parent_logits / soft_temperature, dim=1)  # [batch, num_parents]
        
        # For each meta concept, get the probability of its parent
        # meta_parent_probs[b, i] = P(parent of meta i | image b)
        meta_parent_probs = torch.matmul(meta_to_parent_onehot, parent_probs.T).T  # [batch, num_metas]
        
        # Use these as soft weights (log-space for numerical stability)
        # Higher parent prob -> higher weight for that meta concept
        log_weights = torch.log(meta_parent_probs + 1e-8)
        masked_logits = meta_logits + log_weights
    
    # Standard cross-entropy on masked/weighted logits
    loss = F.cross_entropy(masked_logits, meta_labels)
    
    return loss


def inter_parent_contrastive_loss(
    embeddings: torch.Tensor,
    parent_labels: torch.Tensor,
    temperature: float = 0.1,
) -> torch.Tensor:
    """
    Contrastive loss that pulls together samples from the same parent concept
    and pushes apart samples from different parents.
    
    This encourages the model to learn parent-level structure in the embedding space.
    
    Args:
        embeddings: Concept embeddings [batch, dim]
        parent_labels: Parent concept labels [batch]
        temperature: Temperature for contrastive loss
        
    Returns:
        Scalar loss
    """
    batch_size = embeddings.size(0)
    device = embeddings.device
    
    # Normalize embeddings
    embeddings_norm = F.normalize(embeddings, p=2, dim=1)
    
    # Compute similarity matrix
    sim = torch.matmul(embeddings_norm, embeddings_norm.T) / temperature  # [batch, batch]
    
    # Create positive mask: same parent
    pos_mask = (parent_labels.unsqueeze(0) == parent_labels.unsqueeze(1)).float()
    
    # Remove self-similarity from positives
    pos_mask.fill_diagonal_(0)
    
    # Check if there are any positive pairs
    num_positives = pos_mask.sum(dim=1)
    
    # For samples with no positive pairs, skip them
    valid_mask = num_positives > 0
    if not valid_mask.any():
        return torch.tensor(0.0, device=device)
    
    # Compute log-sum-exp over all (for denominator)
    # Mask out self-similarity with large negative value
    self_mask = torch.eye(batch_size, device=device)
    sim_masked = sim - self_mask * 1e9
    log_sum_exp_all = torch.logsumexp(sim_masked, dim=1)
    
    # Compute log-sum-exp over positives only
    neg_inf_mask = (1 - pos_mask) * (-1e9)
    sim_pos_only = sim + neg_inf_mask
    log_sum_exp_pos = torch.logsumexp(sim_pos_only, dim=1)
    
    # Loss: -log(pos_sum / all_sum) for valid samples
    loss_per_sample = log_sum_exp_all - log_sum_exp_pos
    loss = loss_per_sample[valid_mask].mean()
    
    return loss


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
