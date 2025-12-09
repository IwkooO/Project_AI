
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

            # Combine global pooled (x) with spatial context (mean over queries)
            x = x + pooled_ctx.mean(dim=1)
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


class GeoHead(nn.Module):
    """
    Predicts S2 cell and offset from image embeddings + concept probabilities.
    
    Coarse: [z, c_probs] -> MLP -> M cells
    Fine:   [z, c_probs] -> MLP -> M * 2 offsets
    """
    def __init__(self, input_dim=768, num_concepts=100, num_cells=1000, hidden_dim=512, dropout=0.3):
        super().__init__()
        
        # Input is image embedding + concept probabilities
        combined_dim = input_dim + num_concepts
        
        # Shared feature extractor
        self.feature_net = nn.Sequential(
            nn.Linear(combined_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout)
        )
        
        # Coarse Head (Cell Classification)
        self.coarse_head = nn.Linear(hidden_dim, num_cells)
        
        # Fine Head (Offset Regression)
        # Predicts (d_lat, d_lng) for each cell
        self.fine_head = nn.Linear(hidden_dim, num_cells * 2)
        
        self.num_cells = num_cells

    def forward(self, z, c_probs):
        # z: [B, 768]
        # c_probs: [B, K]
        
        x = torch.cat([z, c_probs], dim=1)
        features = self.feature_net(x)
        
        # Coarse logits: [B, M]
        cell_logits = self.coarse_head(features)
        
        # Fine offsets: [B, M, 2]
        offsets = self.fine_head(features).view(-1, self.num_cells, 2)
        
        return cell_logits, offsets


class RelevanceGate(nn.Module):
    """
    Learns a per-concept gate conditioned on concept logits/probs and pooled embedding.
    Encourages concepts that help geo prediction to be emphasized.
    """
    def __init__(self, num_concepts, input_dim=768, hidden_dim=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(num_concepts + input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, num_concepts),
            nn.Sigmoid(),
        )

    def forward(self, c_logits, z):
        c_probs = torch.softmax(c_logits, dim=1)
        gate = self.net(torch.cat([c_probs, z], dim=1))
        gated = c_probs * gate
        return gated, gate


class CBM(nn.Module):
    """
    Concept Bottleneck Model for Geolocation.
    Phase 1: Train ConceptHead
    Phase 2: Freeze ConceptHead, Train GeoHead
    """
    def __init__(
        self,
        num_concepts,
        num_cells,
        input_dim=768,
        patch_dim=1024,
        hidden_dim=512,
        dropout=0.3,
        patch_depth=2,
        patch_heads=4,
        num_pool_heads=4,
        gate_hidden=256,
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
        self.geo_head = GeoHead(input_dim, num_concepts, num_cells, hidden_dim, dropout)
        self.relevance_gate = RelevanceGate(num_concepts, input_dim=input_dim, hidden_dim=gate_hidden)
        
    def forward(self, z, patches=None):
        # 1. Concept Prediction
        c_logits, c_hidden, attn_weights, pooled_ctx = self.concept_head(z, patches)
        c_probs_raw = torch.softmax(c_logits, dim=1)

        # 1b. Relevance gate to emphasize geo-useful concepts
        c_probs_gated, gate_values = self.relevance_gate(c_logits, z)
        
        # 2. Geo Prediction (using soft concept probs)
        # Detach c_probs if we want to stop gradients from geo loss to concept head (Phase 2 style)
        # But usually we can train joint or freeze concept head via optimizer
        cell_logits, offsets = self.geo_head(z, c_probs_gated)
        
        return c_logits, cell_logits, offsets, c_hidden, attn_weights, gate_values, c_probs_raw, c_probs_gated


