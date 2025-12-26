"""
Phase 2 geolocation models.
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
    """

    def __init__(
        self,
        concept_vectors: torch.Tensor,  # [K, D]
        temperature: float = 1.0,
    ):
        super().__init__()
        self.register_buffer("concept_vectors", concept_vectors)  # [K, D]
        self.temperature = float(temperature)

    def forward(self, phase1_logits: torch.Tensor) -> torch.Tensor:
        p = F.softmax(phase1_logits / self.temperature, dim=-1)  # [B, K]
        return torch.matmul(p, self.concept_vectors)  # [B, D]


class Stage2CrossAttentionGeoHead(nn.Module):
    """
    Geolocation head for Phase 2.

    For "both" mode:
    - Uses cross-attention: concept queries attend to image patch tokens
    - Allows concepts to focus on relevant spatial regions in the image
    - Complementary fusion: learnable weighted combination of image and concept features
    - Each modality can specialize based on its strengths
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
        super().__init__()

        if mode not in ["both", "concept_only", "image_only"]:
            raise ValueError(f"mode must be one of ['both', 'concept_only', 'image_only'], got {mode}")

        self.mode = mode
        self.num_cells = num_cells
        self.hidden_dim = hidden_dim

        if pooled_dim is None:
            pooled_dim = patch_dim

        if mode in ["both", "concept_only"]:
            self.concept_proj = nn.Linear(concept_dim, hidden_dim)

        if mode in ["both", "image_only"]:
            self.pooled_proj = nn.Linear(pooled_dim, hidden_dim)

        if mode == "both":
            # Image pathway: process pooled embeddings
            self.image_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            
            # Concept pathway: process concept embeddings
            self.concept_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            
            # Patch token projection for cross-attention
            self.patch_proj = nn.Linear(patch_dim, hidden_dim)
            
            # Cross-attention layers: concept queries attend to image patches
            # This allows concepts to focus on relevant spatial regions
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
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            self.cross_attn_ffns = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(hidden_dim, hidden_dim * 2),
                    nn.GELU(),
                    nn.Dropout(dropout),
                    nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.Dropout(dropout),
                )
                for _ in range(num_layers)
            ])
            self.cross_attn_ffn_norms = nn.ModuleList([
                nn.LayerNorm(hidden_dim) for _ in range(num_layers)
            ])
            
            # Complementary fusion: learnable weighted combination
            # Each modality gets its own weight, allowing specialization
            self.fusion_weights = nn.Parameter(torch.ones(2) * 0.5)  # [img_weight, concept_weight]
            self.fusion_norm = nn.LayerNorm(hidden_dim)
        elif mode == "concept_only":
            self.concept_mlp = nn.Sequential(
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
            )
        else:  # image_only
            self.pooled_mlp = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.Dropout(dropout),
            )

        self.cell_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cells),
        )

        self.offset_head = nn.Sequential(
            nn.LayerNorm(hidden_dim),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )

    def forward(
        self,
        concept_emb: torch.Tensor,  # [B, concept_dim]
        patch_tokens: torch.Tensor | None = None,  # [B, P, patch_dim] - now used!
        pooled_emb: torch.Tensor | None = None,  # [B, pooled_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gate = None
        if self.mode == "both":
            if pooled_emb is None:
                raise ValueError("pooled_emb required for mode='both'")
            if patch_tokens is None:
                raise ValueError("patch_tokens required for mode='both' (for cross-attention)")

            # Process image features (pooled global representation)
            img_h = self.pooled_proj(pooled_emb)  # [B, hidden_dim]
            img_h = self.image_adapter(img_h)  # [B, hidden_dim]
            
            # Process concept features
            concept_h = self.concept_proj(concept_emb)  # [B, hidden_dim]
            concept_h = self.concept_adapter(concept_h)  # [B, hidden_dim]
            
            # Project patch tokens for cross-attention
            patch_h = self.patch_proj(patch_tokens)  # [B, P, hidden_dim]
            
            # Cross-attention: concept queries attend to image patches
            # This allows concepts to focus on relevant spatial regions
            # Concept features are queries, patch tokens are keys/values
            for attn, norm, ffn, ffn_norm in zip(
                self.cross_attn_layers,
                self.cross_attn_norms,
                self.cross_attn_ffns,
                self.cross_attn_ffn_norms,
            ):
                # Use concept features as queries to attend to image patches
                concept_query = concept_h.unsqueeze(1)  # [B, 1, hidden_dim]
                
                # Cross-attention: concept queries attend to image patches
                concept_attended, _ = attn(
                    query=concept_query,
                    key=patch_h,
                    value=patch_h,
                )  # [B, 1, hidden_dim]
                
                # Residual connection: add attended features to original concept features
                concept_h = norm(concept_h + concept_attended.squeeze(1))  # [B, hidden_dim]
                
                # FFN with residual
                concept_h = ffn_norm(concept_h + ffn(concept_h))  # [B, hidden_dim]
            
            # Complementary fusion: learnable weighted combination
            # Normalize weights to sum to 1 (softmax-like but allows both to contribute)
            fusion_weights = torch.softmax(self.fusion_weights, dim=0)  # [2]
            img_weight, concept_weight = fusion_weights[0], fusion_weights[1]
            
            # Combine: each modality contributes based on learned weights
            # This allows the model to learn when to rely more on image vs concepts
            hidden = self.fusion_norm(
                img_weight * img_h + concept_weight * concept_h
            )  # [B, hidden_dim]
            
            # Store gate values for monitoring (concept weight)
            gate = torch.full((hidden.shape[0], self.hidden_dim), concept_weight.item(), 
                            device=hidden.device, dtype=hidden.dtype)  # [B, hidden_dim]
        elif self.mode == "concept_only":
            concept_h = self.concept_proj(concept_emb)
            hidden = self.concept_mlp(concept_h)
        else:  # image_only
            if pooled_emb is None:
                raise ValueError("pooled_emb required for mode='image_only'")
            pooled_h = self.pooled_proj(pooled_emb)
            hidden = self.pooled_mlp(pooled_h)

        cell_logits = self.cell_head(hidden)
        offset_pred = self.offset_head(hidden)
        return cell_logits, offset_pred, gate


