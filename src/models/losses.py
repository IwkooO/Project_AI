"""
Loss functions for Concept Head training with PU Learning.

Implements:
    - nnPU Loss: Non-negative Positive-Unlabeled learning loss
    - Parameter Drift Loss: Regularization to keep u_k close to v_k
    - Behavioral Drift Loss: Regularization to keep predictions close to zero-shot
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class nnPULoss(nn.Module):
    """
    Non-negative Positive-Unlabeled (nnPU) Learning Loss.
    
    For each concept k, computes:
        R_uPU(k) = π_k · BCE_pos(P_k) + BCE_neg(U_k) - π_k · BCE_neg(P_k)
        R_nnPU(k) = max(R_uPU(k), 0)
    
    Where:
        - π_k: Class prior for concept k
        - P_k: Positive samples (labeled with concept k)
        - U_k: Unlabeled samples (not labeled with concept k)
        - BCE_pos: -log(p_k(x)) - penalizes low probability on positives
        - BCE_neg: -log(1 - p_k(x)) - penalizes high probability on negatives
    
    The correction term (- π_k · BCE_neg(P_k)) accounts for positives 
    that appear in the unlabeled set.
    """
    
    def __init__(
        self,
        priors: torch.Tensor,
        beta: float = 0.0,
        gamma: float = 1.0,
        clip_weights: Optional[torch.Tensor] = None,
        weight_threshold_percentile: float = 10.0
    ):
        """
        Args:
            priors: Class priors π_k ∈ R^K (one per concept)
            beta: Non-negative risk threshold (default 0.0 = strict nnPU)
            gamma: Weight for negative term in correction (default 1.0)
            clip_weights: Optional CLIP-based weights for unlabeled samples [N, K]
            weight_threshold_percentile: Percentile for CLIP weight thresholding
        """
        super().__init__()
        self.register_buffer('priors', priors)
        self.beta = beta
        self.gamma = gamma
        self.clip_weights = clip_weights
        self.weight_threshold_percentile = weight_threshold_percentile
        
        # Small epsilon to avoid log(0)
        self.eps = 1e-7
    
    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute nnPU loss for a batch.
        
        Args:
            probs: Predicted concept probabilities ∈ R^(B×K)
            labels: Single positive label per sample ∈ R^B (values in {0, ..., K-1})
            sample_weights: Optional weights for CLIP prior weighting [B, K]
        
        Returns:
            loss: Total nnPU loss (scalar)
            metrics: Dictionary with per-term losses for logging
        """
        B, K = probs.shape
        device = probs.device
        
        # Clamp probabilities to avoid numerical issues
        probs = torch.clamp(probs, self.eps, 1 - self.eps)
        
        # Create one-hot encoding for positive labels
        # pos_mask[b, k] = 1 if sample b is labeled with concept k
        pos_mask = F.one_hot(labels, num_classes=K).float()  # [B, K]
        
        # Unlabeled mask: all concepts except the labeled one
        unlabeled_mask = 1.0 - pos_mask  # [B, K]
        
        # BCE losses (per sample, per concept)
        bce_pos = -torch.log(probs)  # -log(p_k(x))
        bce_neg = -torch.log(1 - probs)  # -log(1 - p_k(x))
        
        # Apply sample weights for CLIP prior weighting if provided
        if sample_weights is not None:
            # Downweight unlabeled samples with high CLIP similarity
            bce_neg_weighted = bce_neg * sample_weights
        else:
            bce_neg_weighted = bce_neg
        
        # Compute loss terms per concept
        total_loss = torch.tensor(0.0, device=device)
        losses_per_concept = []
        
        for k in range(K):
            pi_k = self.priors[k]
            
            # Get masks for this concept
            pos_k = pos_mask[:, k]  # [B] - 1 for positive samples
            unlabeled_k = unlabeled_mask[:, k]  # [B] - 1 for unlabeled samples
            
            # Count positives and unlabeled
            n_pos = pos_k.sum()
            n_unlabeled = unlabeled_k.sum()
            
            # Term 1: π_k · BCE_pos(P_k)
            if n_pos > 0:
                pos_loss = (pos_k * bce_pos[:, k]).sum() / n_pos
                term1 = pi_k * pos_loss
            else:
                term1 = torch.tensor(0.0, device=device)
            
            # Term 2: BCE_neg(U_k) - weighted
            if n_unlabeled > 0:
                if sample_weights is not None:
                    weights_k = sample_weights[:, k] * unlabeled_k
                    weight_sum = weights_k.sum()
                    if weight_sum > 0:
                        term2 = (weights_k * bce_neg[:, k]).sum() / weight_sum
                    else:
                        term2 = (unlabeled_k * bce_neg[:, k]).sum() / n_unlabeled
                else:
                    term2 = (unlabeled_k * bce_neg[:, k]).sum() / n_unlabeled
            else:
                term2 = torch.tensor(0.0, device=device)
            
            # Term 3: - π_k · BCE_neg(P_k) (correction term)
            if n_pos > 0:
                neg_on_pos = (pos_k * bce_neg[:, k]).sum() / n_pos
                term3 = -pi_k * self.gamma * neg_on_pos
            else:
                term3 = torch.tensor(0.0, device=device)
            
            # Unbiased PU risk
            r_upu = term1 + term2 + term3
            
            # Non-negative version
            r_nnpu = torch.max(r_upu, torch.tensor(self.beta, device=device))
            
            losses_per_concept.append(r_nnpu)
            total_loss = total_loss + r_nnpu
        
        # Average over concepts
        avg_loss = total_loss / K
        
        metrics = {
            'nnpu_loss': avg_loss.item(),
            'losses_per_concept': [l.item() for l in losses_per_concept]
        }
        
        return avg_loss, metrics


class ParameterDriftLoss(nn.Module):
    """
    Parameter drift regularization loss.
    
    Keeps learned parameters close to LaBo anchors:
        L_drift_param = λ_drift · Σ_k ||u_k - v_k||²
    """
    
    def __init__(self, lambda_drift: float = 0.01):
        """
        Args:
            lambda_drift: Weight for drift regularization (default 0.01)
        """
        super().__init__()
        self.lambda_drift = lambda_drift
    
    def forward(self, concept_head) -> torch.Tensor:
        """
        Compute parameter drift loss.
        
        Args:
            concept_head: SpatialConceptHead module
        
        Returns:
            loss: Drift loss (scalar)
        """
        drift = concept_head.get_parameter_drift()
        
        # Also include attention drift if trainable
        attention_drift = concept_head.get_attention_drift()
        
        total_drift = drift + attention_drift
        return self.lambda_drift * total_drift


class BehavioralDriftLoss(nn.Module):
    """
    Behavioral drift regularization loss.
    
    Keeps predictions close to zero-shot CLIP predictions on an anchor set:
        L_drift_behav = λ_behav · Σ_k Σ_x∈A (p_k(x) - p_k^zero(x))²
    
    Where p_k^zero(x) = σ(z_k(x) / τ_zero) is the zero-shot probability.
    """
    
    def __init__(
        self,
        lambda_behav: float = 0.001,
        tau_zero: float = 1.0
    ):
        """
        Args:
            lambda_behav: Weight for behavioral drift (default 0.001)
            tau_zero: Temperature for zero-shot probability (default 1.0)
        """
        super().__init__()
        self.lambda_behav = lambda_behav
        self.tau_zero = tau_zero
    
    def forward(
        self,
        probs: torch.Tensor,
        zero_shot_scores: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute behavioral drift loss.
        
        Args:
            probs: Predicted concept probabilities ∈ R^(B×K)
            zero_shot_scores: Zero-shot CLIP scores z_k(x) ∈ R^(B×K)
        
        Returns:
            loss: Behavioral drift loss (scalar)
        """
        # Compute zero-shot probabilities
        zero_shot_probs = torch.sigmoid(zero_shot_scores / self.tau_zero)
        
        # MSE between predicted and zero-shot probabilities
        drift = ((probs - zero_shot_probs) ** 2).mean()
        
        return self.lambda_behav * drift


class ConceptLoss(nn.Module):
    """
    Combined loss for concept head training.
    
    L_concepts = Σ_k R_nnPU(k) + L_drift_param + L_drift_behav
    """
    
    def __init__(
        self,
        priors: torch.Tensor,
        lambda_drift: float = 0.01,
        lambda_behav: float = 0.0,
        tau_zero: float = 1.0,
        nnpu_beta: float = 0.0,
        nnpu_gamma: float = 1.0
    ):
        """
        Args:
            priors: Class priors π_k ∈ R^K
            lambda_drift: Weight for parameter drift loss
            lambda_behav: Weight for behavioral drift loss (0 to disable)
            tau_zero: Temperature for zero-shot probability
            nnpu_beta: Non-negative risk threshold
            nnpu_gamma: Weight for correction term
        """
        super().__init__()
        
        self.nnpu_loss = nnPULoss(
            priors=priors,
            beta=nnpu_beta,
            gamma=nnpu_gamma
        )
        self.drift_loss = ParameterDriftLoss(lambda_drift=lambda_drift)
        self.behav_loss = BehavioralDriftLoss(
            lambda_behav=lambda_behav,
            tau_zero=tau_zero
        ) if lambda_behav > 0 else None
    
    def forward(
        self,
        probs: torch.Tensor,
        labels: torch.Tensor,
        concept_head,
        zero_shot_scores: Optional[torch.Tensor] = None,
        sample_weights: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined concept loss.
        
        Args:
            probs: Predicted concept probabilities ∈ R^(B×K)
            labels: Single positive label per sample ∈ R^B
            concept_head: SpatialConceptHead module
            zero_shot_scores: (optional) Zero-shot scores for behavioral drift
            sample_weights: (optional) CLIP weights for unlabeled samples
        
        Returns:
            loss: Total loss (scalar)
            metrics: Dictionary with component losses
        """
        # nnPU loss
        nnpu, nnpu_metrics = self.nnpu_loss(probs, labels, sample_weights)
        
        # Parameter drift loss
        drift = self.drift_loss(concept_head)
        
        # Behavioral drift loss (if enabled)
        behav = torch.tensor(0.0, device=probs.device)
        if self.behav_loss is not None and zero_shot_scores is not None:
            behav = self.behav_loss(probs, zero_shot_scores)
        
        # Total loss
        total = nnpu + drift + behav
        
        metrics = {
            'total_loss': total.item(),
            'nnpu_loss': nnpu.item(),
            'drift_loss': drift.item(),
            'behav_loss': behav.item() if isinstance(behav, torch.Tensor) else 0.0,
            **nnpu_metrics
        }
        
        return total, metrics


def compute_clip_sample_weights(
    zero_shot_scores: torch.Tensor,
    labels: torch.Tensor,
    low_percentile: float = 10.0
) -> torch.Tensor:
    """
    Compute sample weights for CLIP prior weighting.
    
    For unlabeled samples:
        - w_k(x) = 1 if z_k(x) < low percentile threshold (likely negative)
        - w_k(x) = small value otherwise (downweight high similarity)
    
    Args:
        zero_shot_scores: Zero-shot scores z_k(x) ∈ R^(B×K)
        labels: Single positive label per sample ∈ R^B
        low_percentile: Percentile threshold for "likely negative"
    
    Returns:
        weights: Sample weights ∈ R^(B×K)
    """
    B, K = zero_shot_scores.shape
    device = zero_shot_scores.device
    
    # Create one-hot for positive labels
    pos_mask = F.one_hot(labels, num_classes=K).float()  # [B, K]
    unlabeled_mask = 1.0 - pos_mask
    
    # Compute thresholds per concept
    weights = torch.ones_like(zero_shot_scores)
    
    for k in range(K):
        scores_k = zero_shot_scores[:, k]
        threshold = torch.quantile(scores_k, low_percentile / 100.0)
        
        # Samples below threshold get weight 1 (likely negative)
        # Samples above threshold get reduced weight
        high_sim_mask = (scores_k >= threshold).float()
        
        # Linear decay: weight decreases as score increases
        max_score = scores_k.max()
        if max_score > threshold:
            # Normalize scores above threshold to [0, 1]
            normalized = (scores_k - threshold) / (max_score - threshold + 1e-8)
            # Weight: 1 at threshold, 0.1 at max
            weight_k = 1.0 - 0.9 * normalized * high_sim_mask
        else:
            weight_k = torch.ones_like(scores_k)
        
        # Only apply weights to unlabeled samples
        weights[:, k] = weight_k * unlabeled_mask[:, k] + pos_mask[:, k]
    
    return weights

