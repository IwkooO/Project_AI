"""
Concept Head with LaBo initialization.

This module implements concept heads that map StreetCLIP embeddings to 
concept probabilities using LaBo-initialized linear classifiers.

Supports two modes:
    1. Global (default): Uses pooled image embedding z(x) directly
       - Simple: logits = z(x) @ U.T
       - Standard LaBo approach
    
    2. Spatial: Uses patch tokens with attention pooling
       - Attention scores: s_k,p^attn = v_k^T · t_p
       - Pooled feature: h_k(x) = Σ_p α_k,p(x) · t_p
       - Provides spatial interpretability via attention heatmaps
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


class ConceptHead(nn.Module):
    """
    Concept Head with LaBo initialization.
    
    Supports both global (default) and spatial modes.
    Global mode uses pooled embeddings directly.
    Spatial mode uses attention over patches for interpretability.
    """
    
    def __init__(
        self,
        concept_embeddings: torch.Tensor,
        mode: str = "global",
        num_patches: int = 196,
        hidden_dim: int = 768,
        temperature: float = 1.0,
        trainable_attention: bool = False
    ):
        """
        Args:
            concept_embeddings: Text embeddings V ∈ R^(K×768) from StreetCLIP text encoder
            mode: "global" (default) or "spatial"
            num_patches: Number of patches P (default 196 for 14x14), only used in spatial mode
            hidden_dim: Embedding dimension (default 768)
            temperature: Temperature τ_attn for attention softmax (only for spatial mode)
            trainable_attention: If True, make attention queries trainable (only for spatial mode)
        """
        super().__init__()
        
        assert mode in ["global", "spatial"], f"mode must be 'global' or 'spatial', got {mode}"
        
        self.mode = mode
        self.num_concepts = concept_embeddings.shape[0]
        self.num_patches = num_patches
        self.hidden_dim = hidden_dim
        self.temperature = temperature
        
        # Output projection vectors (u_k) - LaBo initialized from v_k
        # Shape: [K, 768]
        self.output_vectors = nn.Parameter(concept_embeddings.clone())
        
        # Store original embeddings for drift regularization
        self.register_buffer('labo_anchors', concept_embeddings.clone())
        
        # Spatial mode: attention query vectors
        if mode == "spatial":
            if trainable_attention:
                self.attention_queries = nn.Parameter(concept_embeddings.clone())
            else:
                self.register_buffer('attention_queries', concept_embeddings.clone())
    
    def forward(
        self,
        embeddings: torch.Tensor,
        patch_tokens: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass: compute concept probabilities.
        
        Args:
            embeddings: Pooled image embeddings z(x) ∈ R^(B×768)
            patch_tokens: (optional) Patch tokens T(x) ∈ R^(B×P×768), required for spatial mode
            return_attention: If True, also return attention weights (spatial mode only)
        
        Returns:
            probs: Concept probabilities ĉ(x) ∈ R^(B×K)
            attention: (optional) Attention weights α ∈ R^(B×K×P), None for global mode
        """
        if self.mode == "global":
            return self._forward_global(embeddings)
        else:
            if patch_tokens is None:
                raise ValueError("patch_tokens required for spatial mode")
            return self._forward_spatial(patch_tokens, return_attention)
    
    def _forward_global(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, None]:
        """
        Global mode: simple dot product between pooled embedding and concept vectors.
        
        logits = z(x) @ U.T
        """
        # Compute logits: [B, 768] @ [768, K] = [B, K]
        logits = torch.matmul(embeddings, self.output_vectors.T)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        return probs, None
    
    def _forward_spatial(
        self,
        patch_tokens: torch.Tensor,
        return_attention: bool = False
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Spatial mode: attention-weighted pooling over patches.
        
        1. Compute attention: α_k,p = softmax(v_k^T · t_p / τ)
        2. Pool: h_k = Σ_p α_k,p · t_p
        3. Logit: s_k = u_k^T · h_k
        """
        B, P, D = patch_tokens.shape
        K = self.num_concepts
        
        # Compute attention scores: [B, P, D] @ [D, K] = [B, P, K]
        attn_scores = torch.matmul(patch_tokens, self.attention_queries.T)
        
        # Apply temperature and softmax over patches
        attn_scores = attn_scores / self.temperature
        attn_weights = F.softmax(attn_scores, dim=1)  # [B, P, K]
        
        # Compute concept-specific pooled features
        # [B, K, P] @ [B, P, D] = [B, K, D]
        attn_weights_t = attn_weights.transpose(1, 2)
        pooled_features = torch.bmm(attn_weights_t, patch_tokens)
        
        # Compute logits using output vectors
        logits = (pooled_features * self.output_vectors.unsqueeze(0)).sum(dim=-1)
        
        # Apply sigmoid to get probabilities
        probs = torch.sigmoid(logits)
        
        if return_attention:
            return probs, attn_weights.transpose(1, 2)  # [B, K, P]
        return probs, None
    
    def get_attention_heatmaps(
        self,
        patch_tokens: torch.Tensor,
        grid_size: int = 14
    ) -> torch.Tensor:
        """
        Get attention heatmaps reshaped to spatial grid.
        Only available in spatial mode.
        
        Args:
            patch_tokens: T(x) ∈ R^(B×P×768)
            grid_size: Spatial grid size (default 14 for 14x14)
        
        Returns:
            heatmaps: Attention maps ∈ R^(B×K×H×W)
        """
        if self.mode != "spatial":
            raise ValueError("Attention heatmaps only available in spatial mode")
        
        B = patch_tokens.shape[0]
        K = self.num_concepts
        
        _, attn_weights = self._forward_spatial(patch_tokens, return_attention=True)
        
        # Reshape to spatial grid
        heatmaps = attn_weights.view(B, K, grid_size, grid_size)
        
        return heatmaps
    
    def get_parameter_drift(self) -> torch.Tensor:
        """
        Compute parameter drift from LaBo anchors.
        
        Returns:
            drift: ||u_k - v_k||^2 summed over all concepts
        """
        drift = ((self.output_vectors - self.labo_anchors) ** 2).sum()
        return drift
    
    def get_attention_drift(self) -> torch.Tensor:
        """
        Compute attention query drift from LaBo anchors (if trainable, spatial mode only).
        
        Returns:
            drift: ||v_k^trained - v_k^original||^2 summed over all concepts
        """
        if self.mode == "spatial" and hasattr(self, 'attention_queries'):
            if isinstance(self.attention_queries, nn.Parameter):
                drift = ((self.attention_queries - self.labo_anchors) ** 2).sum()
                return drift
        return torch.tensor(0.0, device=self.labo_anchors.device)


# Alias for backward compatibility
SpatialConceptHead = ConceptHead


class ConceptBottleneckModel(nn.Module):
    """
    Full Concept Bottleneck Model combining concept head with optional geo head.
    
    This is a convenience wrapper that can be extended for Phase 2.
    """
    
    def __init__(
        self,
        concept_head: ConceptHead,
        geo_head: Optional[nn.Module] = None
    ):
        super().__init__()
        self.concept_head = concept_head
        self.geo_head = geo_head
    
    def forward(
        self,
        embeddings: torch.Tensor,
        patch_tokens: Optional[torch.Tensor] = None,
        return_attention: bool = False
    ):
        """
        Forward pass through concept head and optional geo head.
        
        Args:
            embeddings: Pooled image embeddings z(x) ∈ R^(B×768)
            patch_tokens: (optional) Patch tokens for spatial mode
            return_attention: If True, return attention weights
        
        Returns:
            concept_probs: ĉ(x) ∈ R^(B×K)
            geo_pred: (optional) Location prediction if geo_head is present
            attention: (optional) Attention weights
        """
        concept_probs, attention = self.concept_head(
            embeddings, patch_tokens, return_attention
        )
        
        geo_pred = None
        if self.geo_head is not None:
            geo_pred = self.geo_head(concept_probs)
        
        return concept_probs, geo_pred, attention
