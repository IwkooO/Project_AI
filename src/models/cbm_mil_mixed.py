import torch
import torch.nn as nn
import torch.nn.functional as F


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


class ConceptHeadMILMixed(nn.Module):
    """
    Faithful-by-construction concept head with patch mixing.

    Pipeline:
      patches -> patch_proj -> patch_mixer -> evidence scores s[k,p]
      logits = MIL top-k logsumexp(s[k,p])
      attn = softmax over patches derived from s[k,p] (masked to top-k)

    The attention map is derived from the same evidence that produces logits.
    """

    def __init__(
        self,
        patch_dim: int,
        num_concepts: int,
        concept_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        mil_topk: int = 8,
        mil_tau: float = 0.1,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
    ):
        super().__init__()
        if mil_topk <= 0:
            raise ValueError(f"mil_topk must be > 0, got {mil_topk}")
        if mil_tau <= 0:
            raise ValueError(f"mil_tau must be > 0, got {mil_tau}")

        self.num_concepts = num_concepts
        self.mil_topk = mil_topk
        self.mil_tau = mil_tau

        # Project patch tokens into compact concept space (capacity control)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, concept_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # Patch mixing in concept space (cheaper + less overfitting than mixing in patch_dim)
        self.patch_mixer = PatchMixer(
            dim=concept_dim,
            depth=mix_depth,
            num_heads=mix_heads,
            mlp_ratio=mix_mlp_ratio,
            dropout=(dropout if mix_dropout is None else mix_dropout),
        )

        # One prototype per concept in concept space (scores patches)
        self.concept_weight = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.concept_weight)
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))

        # Global representation for SupCon (not used for logits)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

    def forward(self, pooled: torch.Tensor, patches: torch.Tensor):
        # pooled is ignored for logits; kept for API compatibility
        if patches is None:
            raise ValueError(
                "ConceptHeadMILMixed requires patch tokens. "
                "Enable --load-patch-tokens and ensure cache contains patch tokens."
            )

        # patches: [B, P, patch_dim]
        x = self.patch_proj(patches)  # [B, P, concept_dim]
        x = self.patch_mixer(x)       # [B, P, concept_dim]

        # Evidence scores: [B, P, K] then transpose to [B, K, P]
        scores_bpK = torch.matmul(x, self.concept_weight.t()) + self.concept_bias  # [B, P, K]
        scores_bKp = scores_bpK.transpose(1, 2)  # [B, K, P]

        # Top-k MIL pooling -> logits
        k = min(self.mil_topk, scores_bKp.size(-1))
        topk_vals = scores_bKp.topk(k, dim=-1).values  # [B, K, k]
        logits = self.mil_tau * torch.logsumexp(topk_vals / self.mil_tau, dim=-1)  # [B, K]

        # Attention maps: softmax over top-k evidence only (sparse, clean maps)
        topk_idx = scores_bKp.topk(k, dim=-1).indices  # [B, K, k]
        attn_logits = torch.full_like(scores_bKp, float("-inf"))
        attn_logits.scatter_(dim=-1, index=topk_idx, src=scores_bKp.gather(dim=-1, index=topk_idx))
        attn = F.softmax(attn_logits / self.mil_tau, dim=-1)  # [B, K, P]

        # Global representation for SupCon
        pooled_global = x.mean(dim=1)  # [B, concept_dim]
        hidden = self.global_proj(pooled_global)  # [B, hidden_dim]

        return logits, hidden, attn, None


class CBM_MIL_Mixed(nn.Module):
    """
    CBM Phase 1 concept model:
    - Faithful MIL head
    - Optional patch mixing (Transformer encoder)
    """

    def __init__(
        self,
        num_concepts: int,
        patch_dim: int = 1024,
        concept_dim: int = 256,
        hidden_dim: int = 512,
        dropout: float = 0.3,
        mil_topk: int = 8,
        mil_tau: float = 0.1,
        mix_depth: int = 1,
        mix_heads: int = 4,
        mix_mlp_ratio: float = 4.0,
        mix_dropout: float | None = None,
    ):
        super().__init__()
        self.concept_head = ConceptHeadMILMixed(
            patch_dim=patch_dim,
            num_concepts=num_concepts,
            concept_dim=concept_dim,
            hidden_dim=hidden_dim,
            dropout=dropout,
            mil_topk=mil_topk,
            mil_tau=mil_tau,
            mix_depth=mix_depth,
            mix_heads=mix_heads,
            mix_mlp_ratio=mix_mlp_ratio,
            mix_dropout=mix_dropout,
        )

    def forward(self, pooled: torch.Tensor, patches: torch.Tensor):
        return self.concept_head(pooled, patches)

