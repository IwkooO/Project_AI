
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConceptHead(nn.Module):
    """
    Faithful-by-construction concept head.

    Given patch tokens, we compute per-(concept, patch) evidence scores.
    The same scores are used to:
    - pool into concept logits (MIL/top-k pooling)
    - generate attention maps (normalized evidence over patches)

    This avoids the common pitfall where "attention" is only loosely related to
    the actual decision path.
    """
    def __init__(
        self,
        input_dim=768,
        patch_dim=1024,
        num_concepts=100,
        hidden_dim=512,
        dropout=0.3,
        concept_dim=256,
        mil_topk=8,
        mil_tau=0.1,
    ):
        super().__init__()

        if mil_topk <= 0:
            raise ValueError(f"mil_topk must be > 0, got {mil_topk}")
        if mil_tau <= 0:
            raise ValueError(f"mil_tau must be > 0, got {mil_tau}")

        self.num_concepts = num_concepts
        self.mil_topk = mil_topk
        self.mil_tau = mil_tau

        # Project patch tokens into a compact concept space (reduces memorization on fixed embeddings)
        self.patch_proj = nn.Sequential(
            nn.LayerNorm(patch_dim),
            nn.Linear(patch_dim, concept_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )

        # One prototype per concept in concept space (used to score patches)
        self.concept_weight = nn.Parameter(torch.empty(num_concepts, concept_dim))
        nn.init.xavier_uniform_(self.concept_weight)
        self.concept_bias = nn.Parameter(torch.zeros(num_concepts))

        # A global representation for SupCon (does not drive logits; logits come from patch evidence)
        self.global_proj = nn.Sequential(
            nn.LayerNorm(concept_dim),
            nn.Linear(concept_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
    
    def forward(self, x, patches=None):
        # Patch tokens are required for faithful spatial maps.
        if patches is None:
            raise ValueError(
                "ConceptHead requires patch tokens to compute concept evidence maps. "
                "Pass `patches` (e.g., enable --load-patch-tokens and ensure cache contains patch tokens)."
            )

        # patches: [B, P, patch_dim]
        patches_c = self.patch_proj(patches)  # [B, P, concept_dim]

        # Evidence scores per concept per patch:
        # scores_bpK = patches_c @ W^T + b
        scores_bpK = torch.matmul(patches_c, self.concept_weight.t()) + self.concept_bias  # [B, P, K]
        scores_bKp = scores_bpK.transpose(1, 2)  # [B, K, P]

        # Top-k MIL pooling -> logits
        k = min(self.mil_topk, scores_bKp.size(-1))
        topk_vals = scores_bKp.topk(k, dim=-1).values  # [B, K, k]
        logits = self.mil_tau * torch.logsumexp(topk_vals / self.mil_tau, dim=-1)  # [B, K]

        # Attention maps: normalize evidence over patches, but only over top-k patches (sparser, cleaner maps)
        # This is faithful because it's derived from the same evidence scores.
        topk_idx = scores_bKp.topk(k, dim=-1).indices  # [B, K, k]
        attn_logits = torch.full_like(scores_bKp, float("-inf"))
        attn_logits.scatter_(dim=-1, index=topk_idx, src=scores_bKp.gather(dim=-1, index=topk_idx))
        attn_weights = F.softmax(attn_logits / self.mil_tau, dim=-1)  # [B, K, P]

        # Global representation for SupCon (mean pooled projected patches)
        pooled_global = patches_c.mean(dim=1)  # [B, concept_dim]
        hidden = self.global_proj(pooled_global)  # [B, hidden_dim]

        return logits, hidden, attn_weights, None


class CBM(nn.Module):
    """
    Concept Bottleneck Model - Phase 1: Concept Prediction Only.
    
    Focuses solely on predicting visual concepts from images.
    Geo prediction components removed for Phase 1 training.
    """
    def __init__(
        self,
        num_concepts,
        input_dim=768,
        patch_dim=1024,
        hidden_dim=512,
        dropout=0.3,
        concept_dim=256,
        mil_topk=8,
        mil_tau=0.1,
    ):
        super().__init__()
        self.concept_head = ConceptHead(
            input_dim,
            patch_dim,
            num_concepts,
            hidden_dim,
            dropout,
            concept_dim=concept_dim,
            mil_topk=mil_topk,
            mil_tau=mil_tau,
        )
        
    def forward(self, z, patches=None):
        """
        Forward pass for concept prediction.
        
        Args:
            z: Global pooled embedding [B, input_dim]
            patches: Optional spatial patch tokens [B, P, patch_dim]
        
        Returns:
            c_logits: Concept logits [B, num_concepts]
            c_hidden: Hidden representation [B, hidden_dim]
            attn_weights: Attention weights [B, num_concepts, P] or None
            pooled_ctx: Pooled context from patches [B, num_concepts, input_dim] or None
        """
        c_logits, c_hidden, attn_weights, pooled_ctx = self.concept_head(z, patches)
        return c_logits, c_hidden, attn_weights, pooled_ctx


