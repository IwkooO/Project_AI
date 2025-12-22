"""
Query-based sparse attention model for concept prediction.

This module contains the ConceptHeadQuerySparse and CBM_QuerySparse classes,
which implement patch-only concept prediction with hard top-K selection.

Optional regularization:
  - Stochastic Top-K Instance Masking (STKIM): during training, randomly masks
    the highest-scoring patches (per concept) with probability p to prevent
    attention concentration and encourage learning from alternative evidence.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math

from .neighborhood_attention import get_local_attention_bias


class PatchMixer(nn.Module):
    """
    Lightweight patch mixing block (Transformer encoder).

    IMPORTANT: Mixing happens *before* concept evidence scoring, but logits and
    attention maps are still computed from per-(concept, patch) evidence scores,
    preserving faithfulness.
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
            # depth=0 => identity mixing
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
        # nn.TransformerEncoder forwards this as the per-layer self-attention mask.
        assert self.encoder is not None
        return self.encoder(x, mask=attn_bias)


class ConceptHeadQuerySparse(nn.Module):
    """
    Concept head that implements:
      (1) Concept-query cross-attention scores (query · contextualized patch)
      (2) Hard top-K selection (like MIL Mixed)
      (3) LogSumExp aggregation over top-K patches only
      (4) Optional local+context fusion: score = α_k * s_ctx + (1-α_k) * s_local

    This approach addresses gradient dilution by concentrating gradients on
    the top-K most relevant patches, matching MIL Mixed's successful strategy.
    
    Patch-only: This model only uses patch tokens, not pooled embeddings.
    """

    def __init__(
        self,
        patch_dim: int,
        num_concepts: int,
        concept_dim: int = 256,
        dropout: float = 0.3,
        attn_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        mix_local_kernel_size: int | None = None,
        use_local_scores: bool = True,
        vision_proj_init_weight: torch.Tensor | None = None,
        mil_topk: int = 8,  # Hard top-K selection (like MIL Mixed)
        # --- Optional STKIM regularization (training only) ---
        stk_mask_prob: float = 0.0,
        stk_k_mask: int = 1,
        stk_mask_fill: str = "min",  # "min" (recommended) or "zero"
    ):
        super().__init__()
        if attn_tau <= 0:
            raise ValueError(f"attn_tau must be > 0, got {attn_tau}")
        if not (0.0 <= float(stk_mask_prob) <= 1.0):
            raise ValueError(f"stk_mask_prob must be in [0,1], got {stk_mask_prob}")
        if int(stk_k_mask) < 0:
            raise ValueError(f"stk_k_mask must be >= 0, got {stk_k_mask}")
        stk_mask_fill = str(stk_mask_fill).lower().strip()
        if stk_mask_fill not in {"min", "zero"}:
            raise ValueError(f"stk_mask_fill must be 'min' or 'zero', got {stk_mask_fill!r}")

        self.num_concepts = num_concepts
        self.attn_tau = float(attn_tau)
        self.use_local_scores = bool(use_local_scores)
        self.concept_dim = int(concept_dim)
        self.mil_topk = int(mil_topk)
        self.stk_mask_prob = float(stk_mask_prob)
        self.stk_k_mask = int(stk_k_mask)
        self.stk_mask_fill = stk_mask_fill
        # Patch projection into concept space (matching MIL Mixed structure).
        #
        # Use Sequential with LayerNorm, Linear, GELU, Dropout for stability.
        # Initialize the Linear layer with CLIP visual_projection weights if provided.
        if vision_proj_init_weight is None:
            raise ValueError(
                "vision_proj_init_weight must be provided for this baseline "
                "(expected StreetCLIP visual_projection.weight)."
            )

        # Expect weight shaped [concept_dim, patch_dim] (PyTorch Linear weight layout).
        if vision_proj_init_weight.dim() != 2:
            raise ValueError(
                f"vision_proj_init_weight must be 2D [concept_dim, patch_dim], got {tuple(vision_proj_init_weight.shape)}"
            )
        if vision_proj_init_weight.shape[0] != concept_dim or vision_proj_init_weight.shape[1] != patch_dim:
            raise ValueError(
                "vision_proj_init_weight shape mismatch: "
                f"expected [{concept_dim}, {patch_dim}], got {tuple(vision_proj_init_weight.shape)}"
            )

        # Match MIL Mixed projection structure: LayerNorm -> Linear -> GELU -> Dropout
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, concept_dim, bias=False),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        # Initialize the Linear layer (index 1) with CLIP weights
        with torch.no_grad():
            self.patch_proj[1].weight.copy_(vision_proj_init_weight.to(dtype=self.patch_proj[1].weight.dtype))

        # Contextualize patches (patch self-attention)
        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
            local_kernel_size=mix_local_kernel_size,
        )

        # Concept queries (for contextual scores): [K, D] -> K is num_concepts and D is the concept_dim
        self.query = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.query)

        # Local prototypes (optional): [K, D]
        self.local_weight = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.local_weight)

        # Per-concept fusion gate α_k in [0,1] (initialized near 0.5)
        self.fuse_logit = nn.Parameter(torch.zeros(num_concepts))

        # Optional per-concept bias (applied to fused scores)
        self.bias = nn.Parameter(torch.zeros(num_concepts))

    def forward(self, patches: torch.Tensor):
        """
        Forward pass using patch tokens only.
        
        Args:
            patches: Patch tokens [B, P, patch_dim]
            
        Returns:
            tuple: (logits, hidden, attn, None)
                - logits: [B, K] concept logits
                - hidden: [B, concept_dim] hidden representation (for SupCon if enabled)
                - attn: [B, K, P] attention weights
                - None: placeholder for compatibility
        """
        if patches is None:
            raise ValueError("ConceptHeadQuerySparse requires patch tokens (patches).")

        # patches: [B, P, patch_dim]
        x_local = self.patch_proj(patches)   # [B, P, D]
        x_ctx = self.patch_mixer(x_local)    # [B, P, D]

        # Contextual query scores: s_ctx[b,k,p] = <q_k, x_ctx[b,p]>
        # Using einsum for clarity.
        # Removed score_scale to match MIL Mixed magnitude
        s_ctx = torch.einsum("bpd,kd->bkp", x_ctx, self.query)  # [B, K, P]

        if self.use_local_scores:
            s_local = torch.einsum("bpd,kd->bkp", x_local, self.local_weight)  # [B, K, P]
            alpha = torch.sigmoid(self.fuse_logit).view(1, -1, 1)  # [1, K, 1]
            scores = alpha * s_ctx + (1.0 - alpha) * s_local
        else:
            scores = s_ctx

        scores = scores + self.bias.view(1, -1, 1)

        # ---------------------------------------------------------
        # STKIM (Stochastic Top-K Instance Masking) regularization
        # ---------------------------------------------------------
        # During training, with probability p, mask the top-k_mask highest-scoring
        # patches *per concept* to discourage attention concentration.
        #
        # For sparse concepts, a conservative default is stk_k_mask=1 (mask top-1).
        #
        # Masking is applied to the evidence scores before top-K MIL pooling so
        # the pooling naturally shifts to "runner-up" patches.
        if self.training and self.stk_mask_prob > 0.0 and self.stk_k_mask > 0:
            P = int(scores.size(-1))
            # Ensure at least one patch remains unmasked.
            k_mask = min(self.stk_k_mask, max(0, P - 1))
            if k_mask > 0:
                # Indices of top-k_mask patches: [B, K, k_mask]
                top_mask_idx = scores.topk(k=k_mask, dim=-1).indices

                # Decide (per sample, per concept) whether to apply masking.
                # Shape: [B, K, 1] to broadcast across the k_mask positions.
                do_mask = (torch.rand(scores.size(0), scores.size(1), 1, device=scores.device) < self.stk_mask_prob)

                if self.stk_mask_fill == "zero":
                    fill_value = 0.0
                else:
                    # Use dtype min instead of -inf for broad dtype safety.
                    fill_value = torch.finfo(scores.dtype).min

                # Replace only the selected top positions when do_mask is true.
                # This blocks gradients through masked top patches (as intended).
                # Use non-inplace scatter to avoid gradient computation issues
                orig_top = scores.gather(dim=-1, index=top_mask_idx)
                new_top = torch.where(
                    do_mask.expand_as(orig_top),
                    torch.full_like(orig_top, fill_value),
                    orig_top,
                )
                # Create a new tensor with masked values (non-inplace)
                scores = scores.scatter(dim=-1, index=top_mask_idx, src=new_top)

        # ---------------------------------------------------------
        # HARD TOP-K SELECTION (like MIL Mixed)
        # ---------------------------------------------------------
        # This addresses the "gradient dilution problem" by:
        # 1. Using hard top-K selection (fixed K=8) instead of soft attention
        # 2. Using LogSumExp (smooth max) over top-K only
        # 3. This matches MIL Mixed's successful approach while preserving
        #    the contextual query mechanism
        # ---------------------------------------------------------

        # 1. Hard Top-K Selection
        k = min(self.mil_topk, scores.size(-1))
        topk_vals, topk_idx = scores.topk(k=k, dim=-1)  # [B, K, k], [B, K, k]
        
        # 2. LogSumExp Aggregation (only on top-K)
        logits = self.attn_tau * torch.logsumexp(topk_vals / self.attn_tau, dim=-1)  # [B, K]
        
        # 3. Attention for Visualization (matches MIL Mixed style)
        # Softmax over top-K only, others get -inf
        attn_logits = torch.full_like(scores, float("-inf"))
        attn_logits.scatter_(dim=-1, index=topk_idx, src=scores.gather(dim=-1, index=topk_idx))
        attn = F.softmax(attn_logits / self.attn_tau, dim=-1)  # [B, K, P]

        # Hidden representation (for SupCon if enabled elsewhere): mean pooled contextual patches
        hidden = x_ctx.mean(dim=1)

        return logits, hidden, attn, None


class CBM_QuerySparse(nn.Module):
    """
    CBM Phase 1 concept model (query-based sparse attention).
    
    Patch-only: This model only uses patch tokens, not pooled embeddings.
    
    Can operate in two modes:
    1. Cached mode: Takes pre-computed patch tokens as input (frozen backbone)
    2. Trainable backbone mode: Takes raw images, extracts patch tokens via vision encoder
    """

    def __init__(
        self,
        num_concepts: int,
        patch_dim: int = 1024,
        concept_dim: int = 256,
        dropout: float = 0.3,
        attn_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        mix_local_kernel_size: int | None = None,
        use_local_scores: bool = True,
        vision_proj_init_weight: torch.Tensor | None = None,
        vision_encoder: nn.Module | None = None,
        expected_num_patches: int | None = None,
        mil_topk: int = 8,
        stk_mask_prob: float = 0.0,
        stk_k_mask: int = 1,
        stk_mask_fill: str = "min",
    ):
        super().__init__()
        self.vision_encoder = vision_encoder
        self.expected_num_patches = expected_num_patches
        self.use_trainable_backbone = vision_encoder is not None
        
        self.concept_head = ConceptHeadQuerySparse(
            patch_dim=patch_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            dropout=dropout,
            attn_tau=attn_tau,
            mix_depth=mix_depth,
            mix_heads=mix_heads,
            mix_mlp_ratio=mix_mlp_ratio,
            mix_dropout=mix_dropout,
            mix_local_kernel_size=mix_local_kernel_size,
            use_local_scores=use_local_scores,
            vision_proj_init_weight=vision_proj_init_weight,
            mil_topk=mil_topk,
            stk_mask_prob=stk_mask_prob,
            stk_k_mask=stk_k_mask,
            stk_mask_fill=stk_mask_fill,
        )

    def forward(self, patches_or_images: torch.Tensor):
        """
        Forward pass.
        
        Args:
            patches_or_images: 
                - If trainable_backbone=False: Patch tokens [B, P, patch_dim]
                - If trainable_backbone=True: Raw images [B, C, H, W]
            
        Returns:
            tuple: (logits, hidden, attn, None)
                - logits: [B, K] concept logits
                - hidden: [B, concept_dim] hidden representation
                - attn: [B, K, P] attention weights
                - None: placeholder for compatibility
        """
        if self.use_trainable_backbone:
            # Extract patch tokens from images via vision encoder
            if self.vision_encoder is None:
                raise RuntimeError("vision_encoder is None but use_trainable_backbone=True")
            
            # Get hidden states from vision encoder
            outputs = self.vision_encoder(patches_or_images)
            hidden_states = outputs.last_hidden_state  # [B, seq_len, hidden_dim]
            
            # Extract patch tokens (drop CLS token if present)
            seq_len = hidden_states.shape[1]
            if self.expected_num_patches is not None:
                if seq_len == self.expected_num_patches:
                    patches = hidden_states
                elif seq_len > self.expected_num_patches:
                    # Drop leading special tokens (CLS, etc.)
                    num_extra = seq_len - self.expected_num_patches
                    patches = hidden_states[:, num_extra:, :]
                else:
                    raise RuntimeError(
                        f"Unexpected sequence length {seq_len} < expected {self.expected_num_patches}"
                    )
            else:
                # Fallback: assume first token is CLS if seq_len is not a perfect square
                side = int(seq_len ** 0.5)
                if side * side == seq_len:
                    patches = hidden_states
                else:
                    # Try dropping 1 token
                    side2 = int((seq_len - 1) ** 0.5)
                    if side2 * side2 == (seq_len - 1):
                        patches = hidden_states[:, 1:, :]
                    else:
                        patches = hidden_states  # Last resort: use all tokens
        else:
            # Use pre-computed patch tokens
            patches = patches_or_images
        
        return self.concept_head(patches)
