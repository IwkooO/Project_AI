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

    NOTE: Despite the name, the current implementation fuses concept embeddings
    with pooled image embeddings; it does not use patch_tokens.
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
            nn.init.constant_(self.gate_out.bias, -2.0)
            self.fuse_norm = nn.LayerNorm(hidden_dim)
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
        patch_tokens: torch.Tensor | None = None,  # unused (kept for API compatibility)
        pooled_emb: torch.Tensor | None = None,  # [B, pooled_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        gate = None
        if self.mode == "both":
            if pooled_emb is None:
                raise ValueError("pooled_emb required for mode='both'")

            concept_h = self.concept_proj(concept_emb)
            img_h = self.pooled_proj(pooled_emb)

            img_h = self.image_adapter(img_h)
            concept_h = self.concept_adapter(concept_h)

            gate_in = torch.cat([img_h, concept_h], dim=1)
            gate = torch.sigmoid(self.gate_out(self.gate_hidden(gate_in)))
            hidden = self.fuse_norm(img_h + gate * concept_h)
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


