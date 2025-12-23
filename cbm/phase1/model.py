"""
Phase 1 CBM: hard Top-K MIL pooling concept predictor (scratch concept space).
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from cbm.utils.neighborhood_attention import get_local_attention_bias


def _extract_patch_tokens(
    hidden_states: torch.Tensor,
    expected_num_patches: int | None = None,
) -> torch.Tensor:
    """
    Extract patch tokens from a vision encoder last_hidden_state.

    Args:
        hidden_states: [B, seq_len, D]
        expected_num_patches: If provided, drop leading special tokens so the
            returned tensor has exactly expected_num_patches tokens.

    Returns:
        patches: [B, P, D]
    """
    if hidden_states.dim() != 3:
        raise ValueError(f"hidden_states must be [B, seq_len, D], got {tuple(hidden_states.shape)}")

    seq_len = int(hidden_states.shape[1])
    if expected_num_patches is not None:
        exp = int(expected_num_patches)
        if seq_len == exp:
            return hidden_states
        if seq_len > exp:
            return hidden_states[:, (seq_len - exp) :, :]
        raise RuntimeError(f"seq_len={seq_len} < expected_num_patches={exp}")

    # Fallback: if seq_len is a square, assume already patches.
    side = int(math.isqrt(seq_len))
    if side * side == seq_len:
        return hidden_states

    # Try dropping 1 leading token (CLS) if that yields a square.
    side2 = int(math.isqrt(max(0, seq_len - 1)))
    if side2 * side2 == (seq_len - 1):
        return hidden_states[:, 1:, :]

    return hidden_states


class PatchMixer(nn.Module):
    """
    Lightweight patch mixing block (Transformer encoder).
    """

    def __init__(
        self,
        dim: int,
        depth: int = 1,
        num_heads: int = 4,
        mlp_ratio: float = 4.0,
        dropout: float = 0.1,
        local_kernel_size: int | None = None,
    ):
        super().__init__()

        if local_kernel_size is not None:
            k = int(local_kernel_size)
            if k <= 0 or (k % 2 == 0):
                raise ValueError(f"local_kernel_size must be odd and > 0, got {local_kernel_size}")
            self.local_kernel_size = k
        else:
            self.local_kernel_size = None

        self.depth = int(depth)
        if self.depth > 0:
            ff_dim = int(dim * mlp_ratio)
            layer = nn.TransformerEncoderLayer(
                d_model=dim,
                nhead=num_heads,
                dim_feedforward=ff_dim,
                dropout=dropout,
                batch_first=True,
                activation="gelu",
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(layer, num_layers=self.depth)
        else:
            self.encoder = None

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, dim]
        if self.depth == 0:
            return x
        if self.local_kernel_size is None:
            assert self.encoder is not None
            return self.encoder(x)

        p = int(x.shape[1])
        attn_bias = get_local_attention_bias(
            p,
            self.local_kernel_size,
            device=x.device,
            dtype=x.dtype,
        )
        assert self.encoder is not None
        return self.encoder(x, mask=attn_bias)


class ConceptHeadTopKMil(nn.Module):
    """
    Concept head:
      (1) Project patch tokens -> concept_dim concept space
      (2) Patch mixing in concept space
      (3) Per-(concept, patch) evidence via learned concept queries
      (4) Hard top-K + LogSumExp pooling to produce concept logits

    Returns: (logits, hidden, attn, None)
    """

    def __init__(
        self,
        *,
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
        # --- Optional STKIM regularization (training only) ---
        stk_mask_prob: float = 0.0,
        stk_k_mask: int = 1,
        stk_mask_fill: str = "min",  # "min" (recommended) or "zero"
        # --- Patch projection architecture ---
        proj_type: str = "simple",  # "simple" | "two_stage" | "bottleneck"
    ):
        super().__init__()
        if int(mil_topk) <= 0:
            raise ValueError(f"mil_topk must be > 0, got {mil_topk}")
        if float(mil_tau) <= 0:
            raise ValueError(f"mil_tau must be > 0, got {mil_tau}")
        if not (0.0 <= float(stk_mask_prob) <= 1.0):
            raise ValueError(f"stk_mask_prob must be in [0,1], got {stk_mask_prob}")
        if int(stk_k_mask) < 0:
            raise ValueError(f"stk_k_mask must be >= 0, got {stk_k_mask}")
        stk_mask_fill = str(stk_mask_fill).lower().strip()
        if stk_mask_fill not in {"min", "zero"}:
            raise ValueError(f"stk_mask_fill must be 'min' or 'zero', got {stk_mask_fill!r}")
        proj_type = str(proj_type).lower().strip()
        if proj_type not in {"simple", "two_stage", "bottleneck"}:
            raise ValueError(f"proj_type must be one of ['simple', 'two_stage', 'bottleneck'], got {proj_type!r}")

        self.num_concepts = int(num_concepts)
        self.concept_dim = int(concept_dim)
        self.mil_topk = int(mil_topk)
        self.mil_tau = float(mil_tau)
        self.stk_mask_prob = float(stk_mask_prob)
        self.stk_k_mask = int(stk_k_mask)
        self.stk_mask_fill = stk_mask_fill

        # Patch projection architecture selection
        if proj_type == "simple":
            self.patch_proj = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, concept_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        elif proj_type == "two_stage":
            self.patch_proj = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, patch_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(patch_dim, concept_dim, bias=False),
                nn.GELU(),
            )
        else:  # bottleneck
            mid_dim = (patch_dim + concept_dim) // 2
            self.patch_proj = nn.Sequential(
                nn.LayerNorm(patch_dim),
                nn.Linear(patch_dim, mid_dim, bias=False),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(mid_dim, concept_dim, bias=False),
                nn.GELU(),
            )

        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
            local_kernel_size=mix_local_kernel_size,
        )

        self.query = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.query)
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, patches: torch.Tensor):
        if patches is None:
            raise ValueError(f"{self.__class__.__name__} requires patch tokens (patches).")
        if patches.dim() != 3:
            raise ValueError(f"patches must be [B,P,D], got {tuple(patches.shape)}")

        x = self.patch_proj(patches)  # [B, P, concept_dim]
        x = self.patch_mixer(x)  # [B, P, concept_dim]

        scores = torch.einsum("bpd,kd->bkp", x, self.query) + self.bias.view(1, -1, 1)

        # STKIM regularization: optionally mask top evidence patches per concept.
        if self.training and self.stk_mask_prob > 0.0 and self.stk_k_mask > 0:
            P = int(scores.size(-1))
            k_mask = min(self.stk_k_mask, max(0, P - 1))
            if k_mask > 0:
                top_mask_idx = scores.topk(k=k_mask, dim=-1).indices  # [B, K, k_mask]
                do_mask = (
                    torch.rand(scores.size(0), scores.size(1), 1, device=scores.device) < self.stk_mask_prob
                )  # [B, K, 1]

                fill_value = 0.0 if self.stk_mask_fill == "zero" else torch.finfo(scores.dtype).min

                orig_top = scores.gather(dim=-1, index=top_mask_idx)
                new_top = torch.where(
                    do_mask.expand_as(orig_top),
                    torch.full_like(orig_top, fill_value),
                    orig_top,
                )
                scores = scores.scatter(dim=-1, index=top_mask_idx, src=new_top)

        k = min(self.mil_topk, scores.size(-1))
        topk_vals, topk_idx = scores.topk(k=k, dim=-1)
        logits = self.mil_tau * torch.logsumexp(topk_vals / self.mil_tau, dim=-1)  # [B, K]

        attn_logits = torch.full_like(scores, float("-inf"))
        attn_logits.scatter_(dim=-1, index=topk_idx, src=scores.gather(dim=-1, index=topk_idx))
        attn = F.softmax(attn_logits / self.mil_tau, dim=-1)  # [B, K, P]

        hidden = x.mean(dim=1)  # [B, concept_dim]
        return logits, hidden, attn, None


class Phase1CBMTopKMil(nn.Module):
    """
    Phase-1 model wrapper around ConceptHeadTopKMil.

    Supports:
    - cached mode: forward(patches) where patches are [B, P, patch_dim]
    - trainable backbone mode (optional): forward(images) if vision_encoder is provided
    """

    def __init__(
        self,
        *,
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
        stk_mask_prob: float = 0.0,
        stk_k_mask: int = 1,
        stk_mask_fill: str = "min",
        proj_type: str = "simple",
        vision_encoder: nn.Module | None = None,
        expected_num_patches: int | None = None,
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.expected_num_patches = expected_num_patches
        self.use_trainable_backbone = vision_encoder is not None

        self.concept_head = ConceptHeadTopKMil(
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
            stk_mask_prob=stk_mask_prob,
            stk_k_mask=stk_k_mask,
            stk_mask_fill=stk_mask_fill,
            proj_type=proj_type,
        )

    def forward(self, patches_or_images: torch.Tensor):
        if self.use_trainable_backbone:
            if self.vision_encoder is None:
                raise RuntimeError("vision_encoder is None but use_trainable_backbone=True")
            outputs = self.vision_encoder(patches_or_images)
            patches = _extract_patch_tokens(outputs.last_hidden_state, self.expected_num_patches)
        else:
            patches = patches_or_images

        return self.concept_head(patches)


