
import torch
import torch.nn as nn
import torch.nn.functional as F


class PatchEncoder(nn.Module):
    """
    Shallow transformer encoder over patch tokens to refine spatial features.
    Keeps compute modest but lets the model mix local cues.
    """
    def __init__(self, dim=1024, depth=2, num_heads=4, mlp_ratio=4.0, dropout=0.1):
        super().__init__()
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=dim,
            nhead=num_heads,
            dim_feedforward=int(dim * mlp_ratio),
            dropout=dropout,
            batch_first=True,
            activation="gelu",
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=depth)

    def forward(self, x):
        # x: [B, P, dim]
        return self.encoder(x)


class ConceptHead(nn.Module):
    """
    Predicts concept probabilities from image embeddings.
    Uses patch encoder + multi-head pooling; can use concept-specific queries for disentangling.
    Returns attention weights for interpretability.
    """
    def __init__(
        self,
        input_dim=768,
        patch_dim=1024,
        num_concepts=100,
        hidden_dim=512,
        dropout=0.3,
        patch_depth=2,
        patch_heads=4,
        num_pool_heads=4,
    ):
        super().__init__()

        self.use_concept_queries = True  # always use concept-specific queries

        # Patch encoder to refine spatial tokens
        self.patch_encoder = PatchEncoder(
            dim=patch_dim,
            depth=patch_depth,
            num_heads=patch_heads,
            dropout=dropout,
        )

        # One query per concept
        num_queries = num_concepts
        self.pool_queries = nn.Parameter(torch.randn(num_queries, patch_dim))
        nn.init.xavier_uniform_(self.pool_queries)

        # Multi-head attention to pool patches
        self.mha_pool = nn.MultiheadAttention(
            embed_dim=patch_dim,
            num_heads=num_pool_heads,
            batch_first=True,
        )
        self.pool_proj = nn.Linear(patch_dim, input_dim)

        # Per-concept projection
        self.concept_query_proj = nn.Parameter(torch.randn(num_concepts, input_dim))
        nn.init.xavier_uniform_(self.concept_query_proj)
        self.concept_query_bias = nn.Parameter(torch.zeros(num_concepts))

        # MLP head
        self.pre = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.gate = nn.Linear(input_dim, hidden_dim)
        self.second = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        )
        self.output_layer = nn.Linear(hidden_dim, num_concepts)
    
    def forward(self, x, patches=None):
        attn_weights = None

        # If patches provided, encode and pool
        if patches is not None:
            # patches: [B, P, patch_dim]
            patches_enc = self.patch_encoder(patches)

            # Prepare queries
            B = patches_enc.size(0)
            queries = self.pool_queries.unsqueeze(0).expand(B, -1, -1)  # [B, Q, D]

            # Multi-head attention pooling
            # attn_out: [B, Q, D], attn_weights: [B, Q, P]
            attn_out, attn_weights = self.mha_pool(queries, patches_enc, patches_enc)

            # Project pooled tokens to match pooled embedding dim
            pooled_ctx = self.pool_proj(attn_out)  # [B, Q, input_dim]

            # One query per concept -> logits directly
            # concept_query_proj: [K, input_dim], pooled_ctx: [B, K, input_dim]
            logits_direct = (pooled_ctx * self.concept_query_proj).sum(dim=-1) + self.concept_query_bias
            pooled_for_mlp = pooled_ctx  # keep for hidden representation aggregation

            # Patch-only mode: ignore pooled/CLS embedding `x`.
            # Build the per-image representation purely from patch-derived context.
            x = pooled_ctx.mean(dim=1)
        else:
            pooled_for_mlp = None
            logits_direct = None

        # MLP head
        h = self.pre(x)
        gated = torch.sigmoid(self.gate(x))
        h = h + gated * h
        h2 = self.second(h)
        hidden = h + h2  # residual

        # If concept queries used, combine direct logits with MLP output
        logits_mlp = self.output_layer(hidden)
        if logits_direct is not None:
            logits = logits_direct + logits_mlp
        else:
            logits = logits_mlp

        return logits, hidden, attn_weights, pooled_for_mlp


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
        patch_depth=2,
        patch_heads=4,
        num_pool_heads=4,
    ):
        super().__init__()
        self.concept_head = ConceptHead(
            input_dim,
            patch_dim,
            num_concepts,
            hidden_dim,
            dropout,
            patch_depth,
            patch_heads,
            num_pool_heads,
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


