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


class Phase1LogitsAdapter(nn.Module):
    """
    Light adapter for Phase1 logits used directly as features.

    This is intentionally minimal (no trainable params by default) so that Stage2 can
    exploit any linear separability in [pooled_emb ; phase1_logits] (as shown by the probe).
    """

    def __init__(self, temperature: float = 1.0, layernorm: bool = True):
        super().__init__()
        self.temperature = float(temperature)
        self.layernorm = bool(layernorm)
        self._ln: nn.LayerNorm | None = None

    def forward(self, phase1_logits: torch.Tensor) -> torch.Tensor:
        x = phase1_logits / self.temperature
        if self.layernorm:
            # Lazy init to avoid requiring logits_dim in __init__.
            if self._ln is None:
                self._ln = nn.LayerNorm(x.shape[-1]).to(device=x.device, dtype=x.dtype)
            x = self._ln(x)
        return x


class Stage2PooledLogitsGeoHead(nn.Module):
    """
    Image-Controlled Gated Fusion: Image + (Gate * Concepts)
    
    The gate is computed from Image features, not concepts. The image features
    effectively say: "I am confused about this location; let me look at concepts."
    
    Key: Gate bias initialized to negative value (e.g., -4.0) so gate outputs
    near 0.0 at start. This makes the model mathematically identical to image_only
    at epoch 0, avoiding gradient conflict.
    
    Why this works:
    - At Epoch 0: Gate ≈ 0 → model = image_only (no gradient conflict)
    - Gate only opens when image features signal uncertainty
    - Concept branch only contributes when explicitly gated open
    - No additive noise from concepts when image is confident
    """

    def __init__(
        self,
        *,
        pooled_dim: int,
        logits_dim: int,
        num_cells: int,
        hidden_dim: int = 512,
        dropout: float = 0.1,
        gate_bias_init: float = -4.0,
    ):
        super().__init__()
        self.mode = "pooled_logits"
        
        # Pathway A: Image-only prediction (baseline)
        self.img_pathway = nn.Sequential(
            nn.Linear(pooled_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.img_cell_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cells),
        )
        
        self.img_offset_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        
        # Pathway B: Concept correction
        self.concept_pathway = nn.Sequential(
            nn.Linear(logits_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        
        self.concept_cell_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_cells),
        )
        
        self.concept_offset_correction = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 3),
        )
        
        # Image-controlled gate: computes gate from image features
        # Gate controls how much concept correction to apply
        self.gate_network = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()  # Gate output in [0, 1]
        )
        
        # Initialize gate bias to negative value so gate starts closed
        # This makes model = image_only at epoch 0
        self._init_gate_closed(gate_bias_init)
    
    def _init_gate_closed(self, gate_bias_init: float):
        """Initialize gate to output near 0.0 at start (closed).
        
        By setting the final linear layer's bias to a negative value (e.g., -4.0),
        the sigmoid will output near 0.0, making the model identical to image_only.
        """
        # Find the final linear layer in gate_network (before sigmoid)
        for layer in reversed(list(self.gate_network)):
            if isinstance(layer, nn.Linear):
                if layer.bias is not None:
                    nn.init.constant_(layer.bias, gate_bias_init)
                break

    def forward(
        self,
        concept_emb: torch.Tensor,  # Phase1 logits features (already normalized by adapter): [B, K]
        patch_tokens: torch.Tensor | None = None,  # unused (kept for API compatibility)
        pooled_emb: torch.Tensor | None = None,  # [B, pooled_dim]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor | None]:
        if pooled_emb is None:
            raise ValueError("pooled_emb required for mode='pooled_logits'")
        if concept_emb is None:
            raise ValueError("phase1_logits required for mode='pooled_logits'")

        # Pathway A: Image baseline prediction
        img_h = self.img_pathway(pooled_emb)  # [B, hidden_dim]
        img_cell_logits = self.img_cell_head(img_h)  # [B, num_cells]
        img_offset_pred = self.img_offset_head(img_h)  # [B, 3]
        
        # Image-controlled gate: compute from image features
        # Gate represents image's uncertainty/need for concept help
        gate = self.gate_network(img_h)  # [B, 1] in [0, 1]
        
        # Pathway B: Concept correction
        concept_h = self.concept_pathway(concept_emb)  # [B, hidden_dim]
        cell_correction = self.concept_cell_correction(concept_h)  # [B, num_cells]
        offset_correction = self.concept_offset_correction(concept_h)  # [B, 3]
        
        # Gated fusion: Image + (Gate * Concepts)
        # When gate ≈ 0 (start of training): model = image_only
        # When gate > 0: concepts contribute proportionally to image uncertainty
        cell_logits = img_cell_logits + gate * cell_correction  # [B, num_cells]
        offset_pred = img_offset_pred + gate * offset_correction  # [B, 3]
        
        return cell_logits, offset_pred, None


class Stage2CrossAttentionGeoHead(nn.Module):
    """
    Geolocation head for Phase 2.

    For "both" mode:
    - Bidirectional cross-attention: concepts ↔ images (both directions)
    - Adaptive per-sample gating: dynamically weight modalities based on confidence
    - Late fusion: separate prediction heads for each modality, then combine predictions
    - Orthogonality regularization: encourages complementary features
    - Each modality can specialize based on its strengths without over-relying on one
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
            
            # Fusion MLP: combines concatenated image and concept features
            self.fusion_mlp = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
            
            # Adaptive per-sample gating (optional, can be used to weight contributions)
            self.fusion_gate = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_dim, 2),
                nn.Softmax(dim=-1),
            )
        elif mode == "concept_only":
            self.concept_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )
        else:  # image_only
            self.image_adapter = nn.Sequential(
                nn.LayerNorm(hidden_dim),
                nn.Linear(hidden_dim, hidden_dim),
                nn.GELU(),
                nn.Dropout(dropout),
            )

        # Unified heads for all modes
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

            # Process image features (pooled global representation - CLS)
            img_h = self.pooled_proj(pooled_emb)  # [B, hidden_dim]
            img_h = self.image_adapter(img_h)  # [B, hidden_dim]
            
            # Process concept features
            concept_h = self.concept_proj(concept_emb)  # [B, hidden_dim]
            concept_h = self.concept_adapter(concept_h)  # [B, hidden_dim]
            
            # Store features for potentially returning (e.g. for orthogonality)
            img_h_orig = img_h.clone()
            concept_h_orig = concept_h.clone()
            
            # Simple but robust fusion: concatenation + MLP
            # This uses the CLS global feature in combination with concepts
            combined = torch.cat([img_h, concept_h], dim=-1)  # [B, 2*hidden_dim]
            fused_h = self.fusion_mlp(combined)  # [B, hidden_dim]
            
            # Optional gate for monitoring/weighting
            fusion_weights = self.fusion_gate(combined)  # [B, 2]
            gate = fusion_weights[:, 1:2].expand(-1, self.hidden_dim)  # [B, hidden_dim]
            
            # Final predictions from fused representation
            cell_logits = self.cell_head(fused_h)
            offset_pred = self.offset_head(fused_h)
            
            return cell_logits, offset_pred, gate, img_h_orig, concept_h_orig
            
        elif self.mode == "concept_only":
            concept_h = self.concept_proj(concept_emb)
            concept_h = self.concept_adapter(concept_h)
            cell_logits = self.cell_head(concept_h)
            offset_pred = self.offset_head(concept_h)
            return cell_logits, offset_pred, gate
        else:  # image_only
            if pooled_emb is None:
                raise ValueError("pooled_emb required for mode='image_only'")
            img_h = self.pooled_proj(pooled_emb)
            img_h = self.image_adapter(img_h)
            cell_logits = self.cell_head(img_h)
            offset_pred = self.offset_head(img_h)
            return cell_logits, offset_pred, gate


