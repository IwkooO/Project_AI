"""
Query-based sparse attention model for concept prediction.

This module contains the ConceptHeadQuerySparse and CBM_QuerySparse classes,
which implement patch-only concept prediction with sparse attention.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def sparsemax(logits: torch.Tensor, dim: int = -1) -> torch.Tensor:
    """
    Sparsemax activation (Martins & Astudillo, 2016).

    Like softmax, maps logits -> probabilities that sum to 1, but can produce
    exact zeros (sparse distributions). Useful for sparse, interpretable attention.

    Intuition:
    - softmax = "exponentiate then normalize" -> always dense (never exact zeros)
    - sparsemax = "project onto the probability simplex" -> can hit the boundary
      of the simplex, producing exact zeros (sparsity).

    Mathematically, for each vector z (along `dim`), sparsemax computes:
      sparsemax(z) = argmin_p ||p - z||^2  subject to p >= 0 and sum(p) = 1
    i.e., Euclidean projection onto the simplex.
    """
    # 1) Numerical stability: shift logits by their maximum.
    #    This does NOT change the projection result, but avoids large magnitudes.
    z = logits - logits.max(dim=dim, keepdim=True).values

    # 2) Sort z in descending order (needed to find the active set / support).
    z_sorted, _ = torch.sort(z, dim=dim, descending=True)

    # 3) Prefix sums of sorted logits: z_cumsum[j] = sum_{i<=j} z_sorted[i]
    z_cumsum = z_sorted.cumsum(dim)

    # 4) Determine the support size k (how many entries will be > 0 after projection).
    #
    #    For sparsemax, the optimal threshold tau satisfies:
    #      p_i = max(z_i - tau, 0)
    #    and sum_i p_i = 1.
    #
    #    If we sort z in descending order, the support is the largest k such that:
    #      1 + k * z_sorted[k] > sum_{j<=k} z_sorted[j]
    #    (using 1-indexing for the inequality).
    #
    #    We'll compute this condition for all possible k and count how many hold.
    r = torch.arange(1, z_sorted.size(dim) + 1, device=z.device, dtype=z.dtype)
    view = [1] * z.dim()
    view[dim] = -1
    r = r.view(*view)

    # Boolean mask indicating which k satisfy the support condition.
    support = 1 + r * z_sorted > z_cumsum

    # Count how many entries are in the support; clamp to at least 1 to be safe.
    k = support.sum(dim=dim, keepdim=True).clamp(min=1)

    # 5) Compute the threshold tau:
    #      tau = (sum_{j<=k} z_sorted[j] - 1) / k
    #    We grab the cumulative sum at index (k-1) because of 0-based indexing.
    z_cumsum_k = z_cumsum.gather(dim, k - 1)
    tau = (z_cumsum_k - 1) / k.to(z.dtype)

    # 6) Compute projected probabilities:
    #      p = max(z - tau, 0)
    #    This yields exact zeros outside the support (sparse distribution).
    p = torch.clamp(z - tau, min=0.0)

    # 7) Numerical safety: enforce sum(p)=1 (tiny error can accumulate in fp16/bf16).
    p = p / (p.sum(dim=dim, keepdim=True) + 1e-12)
    return p


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
    ):
        super().__init__()
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
        self.encoder = nn.TransformerEncoder(layer, num_layers=depth)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, P, dim]
        return self.encoder(x)


class ConceptHeadQuerySparse(nn.Module):
    """
    Concept head that implements:
      (1) Concept-query cross-attention scores (query · contextualized patch)
      (2) Sparse attention (sparsemax or softmax)
      (3) Local+context fusion: score = α_k * s_ctx + (1-α_k) * s_local

    Faithfulness:
      - attention A[k,p] is computed directly from fused scores s[k,p]
      - logits are computed as sum_p A[k,p] * s[k,p]
        so attention directly determines the logit.
    
    Patch-only: This model only uses patch tokens, not pooled embeddings.
    """

    def __init__(
        self,
        patch_dim: int,
        num_concepts: int,
        concept_dim: int = 256,
        dropout: float = 0.3,
        attn_type: str = "sparsemax",  # "sparsemax" or "softmax"
        attn_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        use_local_scores: bool = True,
        vision_proj_init_weight: torch.Tensor | None = None,
    ):
        super().__init__()
        if attn_tau <= 0:
            raise ValueError(f"attn_tau must be > 0, got {attn_tau}")
        if attn_type not in ("sparsemax", "softmax"):
            raise ValueError(f"attn_type must be 'sparsemax' or 'softmax', got {attn_type}")

        self.num_concepts = num_concepts
        self.attn_type = attn_type
        self.attn_tau = float(attn_tau)
        self.use_local_scores = bool(use_local_scores)
        self.concept_dim = int(concept_dim)
        # Standard dot-product attention scaling (stabilizes score magnitudes).
        self.score_scale = 1.0 / math.sqrt(max(1, self.concept_dim))

        # Patch projection into concept space (CLIP-style trainable projection).
        #
        # Baseline behavior: we REQUIRE `vision_proj_init_weight` and use it to initialize
        # a trainable linear map patch_dim -> concept_dim (typically StreetCLIP's
        # visual_projection.weight). No alternative projection path here.
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

        proj = nn.Linear(patch_dim, concept_dim, bias=False)
        with torch.no_grad():
            proj.weight.copy_(vision_proj_init_weight.to(dtype=proj.weight.dtype))

        # Keep this projection trainable: it is a normal nn.Linear parameter.
        self.patch_proj = proj

        # Contextualize patches (patch self-attention)
        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
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

    def _attn(self, score_bKp: torch.Tensor) -> torch.Tensor:
        # score_bKp: [B, K, P]
        scaled = score_bKp / self.attn_tau
        if self.attn_type == "softmax":
            return F.softmax(scaled, dim=-1)
        return sparsemax(scaled, dim=-1)

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
        s_ctx = torch.einsum("bpd,kd->bkp", x_ctx, self.query) * self.score_scale  # [B, K, P]

        if self.use_local_scores:
            s_local = torch.einsum("bpd,kd->bkp", x_local, self.local_weight) * self.score_scale  # [B, K, P]
            alpha = torch.sigmoid(self.fuse_logit).view(1, -1, 1)  # [1, K, 1]
            scores = alpha * s_ctx + (1.0 - alpha) * s_local
        else:
            scores = s_ctx

        scores = scores + self.bias.view(1, -1, 1)

        # Attention and logits (faithful)
        attn = self._attn(scores)  # [B, K, P]
        logits = (attn * scores).sum(dim=-1)  # [B, K]

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
        attn_type: str = "sparsemax",
        attn_tau: float = 0.25,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
        use_local_scores: bool = True,
        vision_proj_init_weight: torch.Tensor | None = None,
        vision_encoder: nn.Module | None = None,
        expected_num_patches: int | None = None,
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
            attn_type=attn_type,
            attn_tau=attn_tau,
            mix_depth=mix_depth,
            mix_heads=mix_heads,
            mix_mlp_ratio=mix_mlp_ratio,
            mix_dropout=mix_dropout,
            use_local_scores=use_local_scores,
            vision_proj_init_weight=vision_proj_init_weight,
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
