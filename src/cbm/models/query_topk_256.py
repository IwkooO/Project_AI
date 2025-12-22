"""
Hard top-K concept model (scratch concept space) for concept prediction.

This is a lightweight alternative to the StreetCLIP-initialized query model:
- Projects raw cached patch tokens -> concept_dim via a small MLP (LN -> Linear -> GELU -> Dropout)
- Learns concept embeddings/queries from scratch (no concept-name / CLIP text init)
- Does NOT use StreetCLIP's visual_projection weights
- Aggregates logits with hard top-K + temperature-scaled LogSumExp (MIL-style)

API-compatible outputs: (logits, hidden, attn, None)
"""
 
from __future__ import annotations
 
import torch
import torch.nn as nn
import torch.nn.functional as F
 
from .query_sparse import PatchMixer
 
 
class ConceptHeadQueryTopK(nn.Module):
    """
    Concept head:
      (1) Project patch tokens -> concept_dim concept space
      (2) Patch mixing in concept space (Transformer encoder)
      (3) Per-(concept, patch) evidence via learned concept queries (K x concept_dim)
      (4) Hard top-K + LogSumExp pooling to produce concept logits
 
    Note: This model is intentionally *not* text-initialized and does not use
    StreetCLIP's visual_projection. It learns the concept space from scratch.
    """
 
    def __init__(
        self,
        patch_dim: int,
        num_concepts: int,
        concept_dim: int = 256,
        dropout: float = 0.3,
        mil_topk: int = 8,
        mil_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        mix_local_kernel_size: int | None = None,
    ):
        super().__init__()
        if mil_topk <= 0:
            raise ValueError(f"mil_topk must be > 0, got {mil_topk}")
        if mil_tau <= 0:
            raise ValueError(f"mil_tau must be > 0, got {mil_tau}")
 
        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.mil_topk = int(mil_topk)
        self.mil_tau = float(mil_tau)
 
        # Two-stage projection: process at full res, then compress
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, patch_dim), # 1. Process at full res (768 -> 768)
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(patch_dim, concept_dim, bias=False), # 2. Compress (768 -> 256)
            nn.GELU(), # Optional, usually good for embeddings
        )
        # self.patch_proj = nn.Sequential(
        #     nn.LayerNorm(patch_dim),
        #     nn.Linear(patch_dim, concept_dim, bias=False),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        # )
 
        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
            local_kernel_size=mix_local_kernel_size,
        )
 
        # One learned concept query per concept: [K, concept_dim]
        self.query = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.query)
 
        # Optional per-concept bias on evidence scores
        self.bias = nn.Parameter(torch.zeros(num_concepts))
 
    def forward(self, patches: torch.Tensor):
        """
        Args:
            patches: [B, P, patch_dim] cached patch tokens
 
        Returns:
            (logits, hidden, attn, None)
              - logits: [B, K]
              - hidden: [B, concept_dim] (mean pooled contextual patches)
              - attn:   [B, K, P] (masked softmax over top-K evidence only)
        """
        if patches is None:
            raise ValueError(f"{self.__class__.__name__} requires patch tokens (patches).")
 
        x = self.patch_proj(patches)   # [B, P, concept_dim]
        x = self.patch_mixer(x)        # [B, P, concept_dim]
 
        # Evidence scores: [B, K, P]
        scores = torch.einsum("bpd,kd->bkp", x, self.query) + self.bias.view(1, -1, 1)
 
        # Hard top-K MIL pooling -> logits
        k = min(self.mil_topk, scores.size(-1))
        topk_vals, topk_idx = scores.topk(k=k, dim=-1)  # [B, K, k], [B, K, k]
        logits = self.mil_tau * torch.logsumexp(topk_vals / self.mil_tau, dim=-1)  # [B, K]
 
        # Attention maps: softmax over top-K evidence only (clean maps)
        attn_logits = torch.full_like(scores, float("-inf"))
        attn_logits.scatter_(dim=-1, index=topk_idx, src=scores.gather(dim=-1, index=topk_idx))
        attn = F.softmax(attn_logits / self.mil_tau, dim=-1)  # [B, K, P]
 
        hidden = x.mean(dim=1)  # [B, concept_dim]
        return logits, hidden, attn, None
 
 
class CBM_QueryTopK(nn.Module):
    """
    Phase-1 concept model wrapper around ConceptHeadQueryTopK.
 
    Supports:
    - cached mode: forward(patches) where patches are [B, P, patch_dim]
    - trainable backbone mode (optional): forward(images) if vision_encoder is provided
    """
 
    def __init__(
        self,
        num_concepts: int,
        patch_dim: int = 1024,
        concept_dim: int = 256,
        dropout: float = 0.3,
        mil_topk: int = 8,
        mil_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        mix_local_kernel_size: int | None = None,
        vision_encoder: nn.Module | None = None,
        expected_num_patches: int | None = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.expected_num_patches = expected_num_patches
        self.use_trainable_backbone = vision_encoder is not None
 
        self.concept_head = ConceptHeadQueryTopK(
            patch_dim=patch_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            dropout=dropout,
            mil_topk=mil_topk,
            mil_tau=mil_tau,
            mix_depth=mix_depth,
            mix_heads=mix_heads,
            mix_mlp_ratio=mix_mlp_ratio,
            mix_dropout=mix_dropout,
            mix_local_kernel_size=mix_local_kernel_size,
        )
 
    def forward(self, patches_or_images: torch.Tensor):
        if self.use_trainable_backbone:
            if self.vision_encoder is None:
                raise RuntimeError("vision_encoder is None but use_trainable_backbone=True")
 
            outputs = self.vision_encoder(patches_or_images)
            hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
 
            seq_len = int(hidden_states.shape[1])
            if self.expected_num_patches is not None:
                if seq_len == self.expected_num_patches:
                    patches = hidden_states
                elif seq_len > self.expected_num_patches:
                    num_extra = seq_len - self.expected_num_patches
                    patches = hidden_states[:, num_extra:, :]
                else:
                    raise RuntimeError(
                        f"Unexpected sequence length {seq_len} < expected {self.expected_num_patches}"
                    )
            else:
                patches = hidden_states
        else:
            patches = patches_or_images
 
        return self.concept_head(patches)


# Backwards-compatible aliases (in case anything referenced the old names).
ConceptHeadQueryTopK256 = ConceptHeadQueryTopK
CBM_QueryTopK256 = CBM_QueryTopK
