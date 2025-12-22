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
        """
        super().__init__()
        
        if mode not in ["both", "concept_only", "image_only"]:
            raise ValueError(f"mode must be one of ['both', 'concept_only', 'image_only'], got {mode}")
        
        self.mode = mode
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim
        
        # Projections to hidden_dim
        if mode in ["both", "concept_only"]:
            self.concept_proj = nn.Linear(concept_dim, hidden_dim)
        
        if mode in ["both", "image_only"]:
            self.patch_proj = nn.Linear(patch_dim, hidden_dim)
        
        # Cross-attention layers
        if mode == "both":
            # Cross-attention: concept queries attend to patch keys/values
            # Use MultiheadAttention for cross-attention
            self.cross_attn_layers = nn.ModuleList([
                nn.MultiheadAttention(
                    embed_dim=hidden_dim,
                    num_heads=num_heads,
                    dropout=dropout,
                    batch_first=True,
                )
                for _ in range(num_layers)
            ])
            self.cross_attn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
            self.cross_attn_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 4),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 4, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ])
            self.cross_attn_ffn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim)
                for _ in range(num_layers)
            ])
        elif mode == "concept_only":
            # Just use concept embeddings directly
            self.concept_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        elif mode == "image_only":
            # Just use patch tokens (mean pooled)
            self.patch_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
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
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            concept_emb: [B, concept_dim] concept embeddings
            patch_tokens: [B, P, patch_dim] patch tokens (optional)
        
        Returns:
            (cell_logits, offset_pred)
              - cell_logits: [B, num_cells]
              - offset_pred: [B, 3] (x, y, z in 3D Cartesian)
        """
        if self.mode == "both":
            if patch_tokens is None:
                raise ValueError("patch_tokens required for mode='both'")
            
            # Project to hidden_dim
            concept_h = self.concept_proj(concept_emb)  # [B, hidden_dim]
            patch_h = self.patch_proj(patch_tokens)  # [B, P, hidden_dim]
            
            # Cross-attention: concept queries attend to patch keys/values
            # Concept as query: [B, 1, hidden_dim]
            query = concept_h.unsqueeze(1)  # [B, 1, hidden_dim]
            
            # Patches as key/value: [B, P, hidden_dim]
            key_value = patch_h  # [B, P, hidden_dim]
            
            # Apply cross-attention layers
            hidden = query  # [B, 1, hidden_dim]
            for i, (attn, norm, ffn, ffn_norm) in enumerate(zip(
                self.cross_attn_layers,
                self.cross_attn_norms,
                self.cross_attn_ffns,
                self.cross_attn_ffn_norms,
            )):
                # Cross-attention: query attends to key_value
                attn_out, _ = attn(hidden, key_value, key_value)  # [B, 1, hidden_dim]
                hidden = norm(hidden + attn_out)  # Residual + norm
                
                # FFN
                ffn_out = ffn(hidden)
                hidden = ffn_norm(hidden + ffn_out)  # Residual + norm
            
            hidden = hidden.squeeze(1)  # [B, hidden_dim]
        
        elif self.mode == "concept_only":
            concept_h = self.concept_proj(concept_emb)  # [B, hidden_dim]
            hidden = self.concept_mlp(concept_h)  # [B, hidden_dim]
        
        elif self.mode == "image_only":
            if patch_tokens is None:
                raise ValueError("patch_tokens required for mode='image_only'")
            patch_h = self.patch_proj(patch_tokens)  # [B, P, hidden_dim]
            patch_mean = patch_h.mean(dim=1)  # [B, hidden_dim]
            hidden = self.patch_mlp(patch_mean)  # [B, hidden_dim]
        
        # Outputs
        cell_logits = self.cell_head(hidden)  # [B, num_cells]
        offset_pred = self.offset_head(hidden)  # [B, 3]
        
        return cell_logits, offset_pred
