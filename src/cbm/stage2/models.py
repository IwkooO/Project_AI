"""
Stage 2 geolocation models.

Stage 2 consumes:
- Cached StreetCLIP embeddings (pooled + patch tokens)
- Frozen Phase1 concept model (query_topk_256) logits
- Concept embeddings derived from Phase1 logits

Outputs:
- Cell classification (semantic geocell)
- Offset regression (fine coordinate prediction)
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptEmbeddingAdapter(nn.Module):
    """
    Converts Phase1 concept logits to concept embeddings.
    
    Given Phase1 logits z ∈ [B, K]:
    - p = softmax(z)
    - concept_emb = p @ E where E ∈ [K, D] is concept vector bank
    
    E can come from:
    - Phase1 checkpoint (if stored)
    - CLIP text embeddings (fallback)
    """
    
    def __init__(
        self,
        concept_vectors: torch.Tensor,  # [K, D]
        temperature: float = 1.0,
    ):
        """
        Args:
            concept_vectors: [K, D] concept vector bank E
            temperature: Temperature for softmax (default 1.0)
        """
        super().__init__()
        self.register_buffer("concept_vectors", concept_vectors)  # [K, D]
        self.temperature = float(temperature)
    
    def forward(self, phase1_logits: torch.Tensor) -> torch.Tensor:
        """
        Args:
            phase1_logits: [B, K] concept logits from Phase1
        
        Returns:
            concept_emb: [B, D] concept embeddings
        """
        # Softmax over concepts
        p = F.softmax(phase1_logits / self.temperature, dim=-1)  # [B, K]
        
        # Weighted combination: p @ E
        concept_emb = torch.matmul(p, self.concept_vectors)  # [B, D]
        
        return concept_emb


class Stage2CrossAttentionGeoHead(nn.Module):
    """
    Cross-attention geolocation head for Stage 2.
    
    Architecture:
    - Cross-attention between concept embeddings and patch tokens
    - Ablation modes: 'both', 'concept_only', 'image_only'
    - Outputs: cell logits + offset regression
    """
    
    def __init__(
        self,
        concept_dim: int,
        patch_dim: int,
        num_cells: int,
        hidden_dim: int = 512,
        num_heads: int = 8,
        num_layers: int = 2,
        dropout: float = 0.1,
        mode: str = "both",
        pooled_dim: int | None = None,
    ):
        """
        Args:
            concept_dim: Dimension of concept embeddings
            patch_dim: Dimension of patch tokens
            num_cells: Number of semantic geocells
            hidden_dim: Hidden dimension for cross-attention
            num_heads: Number of attention heads
            num_layers: Number of cross-attention layers
            dropout: Dropout rate
            mode: 'both' (concept + image), 'concept_only', 'image_only'
            pooled_dim: Dimension of pooled embeddings (CLS token). If None, will use patch_dim.
        """
        super().__init__()
        
        if mode not in ["both", "concept_only", "image_only"]:
            raise ValueError(f"mode must be one of ['both', 'concept_only', 'image_only'], got {mode}")
        
        self.mode = mode
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        
        # Use pooled_dim if provided, otherwise fall back to patch_dim
        if pooled_dim is None:
            pooled_dim = patch_dim
        
        # Projections to hidden_dim
        if mode in ["both", "concept_only"]:
            self.concept_proj = nn.Linear(concept_dim, hidden_dim)
        
        if mode in ["both", "image_only"]:
            self.pooled_proj = nn.Linear(pooled_dim, hidden_dim)
        
        # Fusion layers
        if mode == "both":
            # Gated residual fusion:
            #   h = LN(h_img + sigmoid(g([h_img; h_concept])) * h_concept)
            #
            # This makes concepts "can't-hurt" by allowing the model to ignore them
            # when they are noisy, while still enabling additive improvements.
            self.image_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.concept_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gate_hidden = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            self.gate_out = nn.Linear(hidden_dim, hidden_dim)
            nn.init.constant_(self.gate_out.bias, -2.0)  # start mostly-closed (sigmoid ~ 0.12)
            self.fuse_norm = nn.LayerNorm(hidden_dim)
        elif mode == "concept_only":
            # Just use concept embeddings directly
            self.concept_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        elif mode == "image_only":
            # Use CLS token (pooled embeddings) directly
            self.pooled_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )
        
        # Output heads
        # Cell classification
        self.cell_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cells),
        )
        
        # Offset regression (3D Cartesian: x, y, z)
        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),  # [x, y, z]
        )
    
    def forward(
        self,
        concept_emb: torch.Tensor,  # [B, concept_dim]
        patch_tokens: torch.Tensor | None = None,  # [B, P, patch_dim]
        pooled_emb: torch.Tensor | None = None,  # [B, pooled_dim] CLS token
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            concept_emb: [B, concept_dim] concept embeddings
            patch_tokens: [B, P, patch_dim] patch tokens (optional, used for Phase1 only)
            pooled_emb: [B, pooled_dim] pooled embeddings (CLS token, used for 'both' and 'image_only' modes)
        
        Returns:
            (cell_logits, offset_pred)
              - cell_logits: [B, num_cells]
              - offset_pred: [B, 3] (x, y, z in 3D Cartesian)
        """
        if self.mode == "both":
            if pooled_emb is None:
                raise ValueError("pooled_emb required for mode='both'")
            
            # Project to hidden_dim
            concept_h = self.concept_proj(concept_emb)  # [B, hidden_dim]
            img_h = self.pooled_proj(pooled_emb)  # [B, hidden_dim]

            # Adapt each branch
            img_h = self.image_adapter(img_h)  # [B, hidden_dim]
            concept_h = self.concept_adapter(concept_h)  # [B, hidden_dim]

            # Gate the concept residual
            gate_in = torch.cat([img_h, concept_h], dim=1)  # [B, 2*hidden_dim]
            gate = torch.sigmoid(self.gate_out(self.gate_hidden(gate_in)))  # [B, hidden_dim]

            hidden = self.fuse_norm(img_h + gate * concept_h)  # [B, hidden_dim]
        
        elif self.mode == "concept_only":
            concept_h = self.concept_proj(concept_emb)  # [B, hidden_dim]
            hidden = self.concept_mlp(concept_h)  # [B, hidden_dim]
        
        elif self.mode == "image_only":
            if pooled_emb is None:
                raise ValueError("pooled_emb required for mode='image_only'")
            # Use CLS token directly
            pooled_h = self.pooled_proj(pooled_emb)  # [B, hidden_dim]
            hidden = self.pooled_mlp(pooled_h)  # [B, hidden_dim]
        
        # Outputs
        cell_logits = self.cell_head(hidden)  # [B, num_cells]
        offset_pred = self.offset_head(hidden)  # [B, 3]
        
        return cell_logits, offset_pred
