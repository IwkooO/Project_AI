"""
Concept-Aware Global Image-GPS Alignment Model.

Strict Concept Bottleneck Model (CBM) architecture where ALL downstream tasks
operate on concept embeddings, not raw image features.
"""

from __future__ import annotations

from typing import Optional, Iterable, List, Dict

import torch
from torch import nn
import torch.nn.functional as F
from geoclip import LocationEncoder


# ============================================================================
# TEXT PROTOTYPE UTILITIES
# ============================================================================

# Default templates for concept text encoding
DEFAULT_CONCEPT_TEMPLATES = [
    "A street view showing {}",
    "A photo of {} in the scene",
    "An area characterized by {}",
    "{} visible from the road",
    "A location with {}",
]

DEFAULT_PARENT_TEMPLATES = [
    "A {} area",
    "A scene showing {} features",
    "An environment with {} characteristics",
    "{} landscape",
    "A {} region",
]


@torch.no_grad()
def build_text_prototypes(
    concept_names: List[str],
    text_encoder: nn.Module,
    concept_descriptions: Optional[Dict[str, str]] = None,
    templates: Optional[List[str]] = None,
    device: torch.device = None,
    max_description_length: int = 200,
) -> torch.Tensor:
    """
    Build text prototypes for concepts using StreetCLIP text encoder.
    
    For each concept, encodes either:
    1. The cleaned note description (if available and non-empty)
    2. Template-filled prompts using the concept name
    
    Then averages across all encodings and L2-normalizes.
    
    Args:
        concept_names: List of concept names (meta_name or parent_concept)
        text_encoder: StreetCLIP text encoder with get_text_features() method
        concept_descriptions: Optional dict mapping concept_name -> description text
        templates: List of template strings with {} placeholder for concept name
        device: Device to place prototypes on
        max_description_length: Truncate descriptions longer than this (CLIP has 77 token limit)
        
    Returns:
        Tensor of shape [num_concepts, embedding_dim] with L2-normalized prototypes
    """
    if templates is None:
        templates = DEFAULT_CONCEPT_TEMPLATES
    
    if device is None:
        device = next(text_encoder.parameters()).device
    
    text_encoder.eval()
    prototypes = []
    
    for concept_name in concept_names:
        embeddings = []
        
        # Option 1: Use description if available
        if concept_descriptions and concept_name in concept_descriptions:
            desc = concept_descriptions[concept_name]
            if desc and len(desc.strip()) > 0:
                # Truncate if too long (CLIP has 77 token limit, ~4 chars per token)
                if len(desc) > max_description_length:
                    desc = desc[:max_description_length] + "..."
                
                try:
                    emb = text_encoder.get_text_features(desc)
                    embeddings.append(emb)
                except Exception:
                    pass  # Fall back to templates
        
        # Option 2: Use templates (always add for robustness)
        for template in templates:
            prompt = template.format(concept_name)
            try:
                emb = text_encoder.get_text_features(prompt)
                embeddings.append(emb)
            except Exception:
                continue
        
        if len(embeddings) == 0:
            # Fallback: just encode the concept name directly
            emb = text_encoder.get_text_features(concept_name)
            embeddings.append(emb)
        
        # Stack and average
        stacked = torch.cat(embeddings, dim=0)  # [num_prompts, dim]
        avg_emb = stacked.mean(dim=0, keepdim=True)  # [1, dim]
        prototypes.append(avg_emb)
    
    # Stack all prototypes: [num_concepts, dim]
    prototypes_tensor = torch.cat(prototypes, dim=0)
    
    # L2 normalize
    prototypes_tensor = F.normalize(prototypes_tensor, p=2, dim=1)
    
    return prototypes_tensor.to(device)


def build_meta_to_parent_idx(
    meta_to_parent: Dict[str, str],
    concept_to_idx: Dict[str, int],
    parent_to_idx: Dict[str, int],
) -> torch.Tensor:
    """
    Build a tensor that maps meta_name index to parent_concept index.
    
    Args:
        meta_to_parent: Dict mapping meta_name -> parent_concept
        concept_to_idx: Dict mapping meta_name -> index
        parent_to_idx: Dict mapping parent_concept -> index
        
    Returns:
        Tensor of shape [num_concepts] where tensor[concept_idx] = parent_idx
    """
    num_concepts = len(concept_to_idx)
    mapping = torch.zeros(num_concepts, dtype=torch.long)
    
    for meta_name, meta_idx in concept_to_idx.items():
        parent_concept = meta_to_parent.get(meta_name, 'unknown')
        parent_idx = parent_to_idx.get(parent_concept, 0)
        mapping[meta_idx] = parent_idx
    
    return mapping


# ============================================================================
# STAGE 1 CONCEPT MODEL (Text-Prototype Based)
# ============================================================================

class Stage1ConceptModel(nn.Module):
    """
    Stage 1 Concept Model with text-anchored prototype-based classification.
    
    Instead of MLP-based concept head, uses cosine similarity between
    image embeddings and text prototypes, with learnable refinements:
    - Learnable prototype residuals (fine-tune text prototypes)
    - Per-concept bias (calibrate confidence per class)
    - Learnable temperature/logit scale
    
    Supports hierarchical supervision with both meta-level (fine-grained)
    and parent-level (coarse) concept predictions.
    """
    
    def __init__(
        self,
        image_encoder: nn.Module,
        T_meta: torch.Tensor,
        T_parent: torch.Tensor,
        meta_to_parent_idx: torch.Tensor,
        streetclip_dim: int = 768,
        concept_emb_dim: int = 512,
        init_logit_scale: float = 14.0,  # ~1/0.07 temperature
        learnable_prototypes: bool = True,
        prototype_residual_scale: float = 0.01,
    ):
        """
        Args:
            image_encoder: Pretrained StreetCLIP image encoder (will be frozen)
            T_meta: Text prototypes for meta_name concepts [num_metas, dim]
            T_parent: Text prototypes for parent concepts [num_parents, dim]
            meta_to_parent_idx: Tensor mapping meta_idx -> parent_idx [num_metas]
            streetclip_dim: Dimension of StreetCLIP features (768)
            concept_emb_dim: Dimension of concept embedding space (512)
            init_logit_scale: Initial value for learnable logit scale
            learnable_prototypes: Whether to learn residuals on top of prototypes
            prototype_residual_scale: Scale for initializing prototype residuals
        """
        super().__init__()
        
        self.streetclip_dim = streetclip_dim
        self.concept_emb_dim = concept_emb_dim
        self.num_metas = T_meta.shape[0]
        self.num_parents = T_parent.shape[0]
        
        # ========== Frozen Image Encoder ==========
        self.image_encoder = image_encoder
        self.image_encoder.eval()
        for p in self.image_encoder.parameters():
            p.requires_grad = False
        
        # ========== Text Prototypes (frozen base + learnable residuals) ==========
        self.register_buffer("T_meta_base", T_meta)  # [num_metas, dim]
        self.register_buffer("T_parent_base", T_parent)  # [num_parents, dim]
        self.register_buffer("meta_to_parent_idx", meta_to_parent_idx)  # [num_metas]
        
        # Learnable prototype residuals
        if learnable_prototypes:
            self.meta_residuals = nn.Parameter(
                torch.randn_like(T_meta) * prototype_residual_scale
            )
            self.parent_residuals = nn.Parameter(
                torch.randn_like(T_parent) * prototype_residual_scale
            )
        else:
            self.register_buffer("meta_residuals", torch.zeros_like(T_meta))
            self.register_buffer("parent_residuals", torch.zeros_like(T_parent))
        
        # ========== Concept Bottleneck Projection ==========
        # Projects from StreetCLIP dim (768) to concept embedding dim (512)
        self.concept_bottleneck = nn.Sequential(
            nn.Linear(streetclip_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, concept_emb_dim),
            nn.LayerNorm(concept_emb_dim),
        )
        
        # ========== Text Prototype Projection ==========
        # Projects text prototypes from StreetCLIP dim to concept embedding dim
        self.prototype_projection = nn.Linear(streetclip_dim, concept_emb_dim, bias=False)
        
        # ========== Learnable Logit Scales and Biases ==========
        self.logit_scale_meta = nn.Parameter(torch.tensor(init_logit_scale))
        self.logit_scale_parent = nn.Parameter(torch.tensor(init_logit_scale))
        
        self.meta_bias = nn.Parameter(torch.zeros(self.num_metas))
        self.parent_bias = nn.Parameter(torch.zeros(self.num_parents))
        
        # Initialize bottleneck weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layers with Xavier uniform initialization."""
        for layer in self.concept_bottleneck:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        # Initialize prototype projection
        nn.init.xavier_uniform_(self.prototype_projection.weight)
    
    @property
    def T_meta(self) -> torch.Tensor:
        """Get effective meta prototypes (base + residuals, projected and normalized)."""
        T = self.T_meta_base + self.meta_residuals
        T = self.prototype_projection(T)  # Project to concept_emb_dim
        return F.normalize(T, p=2, dim=1)
    
    @property
    def T_parent(self) -> torch.Tensor:
        """Get effective parent prototypes (base + residuals, projected and normalized)."""
        T = self.T_parent_base + self.parent_residuals
        T = self.prototype_projection(T)  # Project to concept_emb_dim
        return F.normalize(T, p=2, dim=1)
    
    def forward(
        self,
        images: torch.Tensor,
        meta_labels: Optional[torch.Tensor] = None,
        parent_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass through Stage 1 concept model.
        
        Args:
            images: Image tensor [batch, 3, H, W]
            meta_labels: Ground truth meta concept indices [batch]
            parent_labels: Ground truth parent concept indices [batch]
            
        Returns:
            Dict containing:
                - concept_emb: Concept embeddings [batch, concept_emb_dim]
                - meta_logits: Meta concept logits [batch, num_metas]
                - parent_logits: Parent concept logits [batch, num_parents]
                - meta_probs: Meta concept probabilities [batch, num_metas]
                - parent_probs: Parent concept probabilities [batch, num_parents]
        """
        # 1. Extract image features (frozen encoder)
        with torch.no_grad():
            x_img = self.image_encoder(images)  # [batch, 768]
        
        # 2. Project through concept bottleneck
        concept_emb = self.concept_bottleneck(x_img)  # [batch, 512]
        concept_emb_norm = F.normalize(concept_emb, p=2, dim=1)
        
        # 3. Compute meta concept logits via cosine similarity
        # logits = scale * (emb @ T.T) + bias
        meta_logits = self.logit_scale_meta * (concept_emb_norm @ self.T_meta.T) + self.meta_bias
        meta_probs = F.softmax(meta_logits, dim=-1)
        
        # 4. Compute parent concept logits
        parent_logits = self.logit_scale_parent * (concept_emb_norm @ self.T_parent.T) + self.parent_bias
        parent_probs = F.softmax(parent_logits, dim=-1)
        
        return {
            "concept_emb": concept_emb,
            "meta_logits": meta_logits,
            "parent_logits": parent_logits,
            "meta_probs": meta_probs,
            "parent_probs": parent_probs,
        }
    
    def forward_from_features(
        self,
        image_features: torch.Tensor,
        meta_labels: Optional[torch.Tensor] = None,
        parent_labels: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass using precomputed image features (skips image encoder).
        
        Use this for faster training when image encoder is frozen.
        
        Args:
            image_features: Precomputed image features [batch, 768]
            meta_labels: Ground truth meta concept indices [batch]
            parent_labels: Ground truth parent concept indices [batch]
            
        Returns:
            Same dict as forward()
        """
        # Project through concept bottleneck
        concept_emb = self.concept_bottleneck(image_features)  # [batch, 512]
        concept_emb_norm = F.normalize(concept_emb, p=2, dim=1)
        
        # Compute meta concept logits
        meta_logits = self.logit_scale_meta * (concept_emb_norm @ self.T_meta.T) + self.meta_bias
        meta_probs = F.softmax(meta_logits, dim=-1)
        
        # Compute parent concept logits
        parent_logits = self.logit_scale_parent * (concept_emb_norm @ self.T_parent.T) + self.parent_bias
        parent_probs = F.softmax(parent_logits, dim=-1)
        
        return {
            "concept_emb": concept_emb,
            "meta_logits": meta_logits,
            "parent_logits": parent_logits,
            "meta_probs": meta_probs,
            "parent_probs": parent_probs,
        }
    
    def get_trainable_params(self) -> Iterable[nn.Parameter]:
        """
        Return parameters to train in Stage 1.
        
        Includes:
        - concept_bottleneck (projection layer)
        - prototype_projection (text prototype projection)
        - meta_residuals (prototype refinements)
        - parent_residuals (prototype refinements)
        - logit_scale_meta, logit_scale_parent
        - meta_bias, parent_bias
        """
        params = list(self.concept_bottleneck.parameters())
        params.extend(self.prototype_projection.parameters())
        params.append(self.logit_scale_meta)
        params.append(self.logit_scale_parent)
        params.append(self.meta_bias)
        params.append(self.parent_bias)
        
        # Only include residuals if they're learnable (nn.Parameter)
        if isinstance(self.meta_residuals, nn.Parameter):
            params.append(self.meta_residuals)
        if isinstance(self.parent_residuals, nn.Parameter):
            params.append(self.parent_residuals)
        
        return params
    
    def get_prototype_regularization_loss(self, lambda_reg: float = 0.001) -> torch.Tensor:
        """
        Compute L2 regularization loss on prototype residuals.
        
        Prevents prototypes from drifting too far from CLIP's semantic space.
        
        Args:
            lambda_reg: Regularization coefficient
            
        Returns:
            Scalar regularization loss
        """
        reg_loss = lambda_reg * (
            self.meta_residuals.pow(2).sum() + 
            self.parent_residuals.pow(2).sum()
        )
        return reg_loss
    
    def get_intra_parent_consistency_loss(self, lambda_intra: float = 0.01) -> torch.Tensor:
        """
        Regularization that encourages meta concepts within the same parent 
        to have similar prototype residuals.
        
        This creates soft weight sharing within parent groups, helping the model
        learn parent-level structure before fine-grained distinctions.
        
        Args:
            lambda_intra: Regularization coefficient
            
        Returns:
            Scalar regularization loss
        """
        # Get effective meta prototypes with residuals
        meta_protos = self.T_meta  # [num_metas, dim], already projected and normalized
        
        # For each parent, compute mean of its children's prototypes
        num_parents = self.num_parents
        device = meta_protos.device
        
        # Count metas per parent
        parent_counts = torch.zeros(num_parents, device=device)
        parent_sums = torch.zeros(num_parents, meta_protos.size(1), device=device)
        
        for meta_idx in range(self.num_metas):
            parent_idx = self.meta_to_parent_idx[meta_idx].item()
            parent_counts[parent_idx] += 1
            parent_sums[parent_idx] += meta_protos[meta_idx]
        
        # Compute mean prototype per parent (avoid div by zero)
        parent_counts = parent_counts.clamp(min=1)
        parent_means = parent_sums / parent_counts.unsqueeze(1)  # [num_parents, dim]
        
        # Compute variance: how much each meta deviates from its parent's mean
        variance_loss = 0.0
        for meta_idx in range(self.num_metas):
            parent_idx = self.meta_to_parent_idx[meta_idx].item()
            parent_mean = parent_means[parent_idx]
            deviation = meta_protos[meta_idx] - parent_mean
            variance_loss += deviation.pow(2).sum()
        
        variance_loss = variance_loss / self.num_metas
        
        return lambda_intra * variance_loss


# ============================================================================
# ORIGINAL CONCEPT-AWARE GEO MODEL (for reference and Stage 2)
# ============================================================================

class ConceptAwareGeoModel(nn.Module):
    """
    Concept-Aware Global Image-GPS Alignment Model with strict CBM architecture.
    
    Pipeline:
        Image → Frozen StreetCLIP → Image Features (768d) → Concept Bottleneck → Concept Embeddings (512d)
                                                                    ↓
                                              ALL downstream heads (country, cell, offset)
                                                                    ↓
        GPS → LocationEncoder → GPS Embeddings (512d) ←── Contrastive Alignment
        Text → StreetCLIP Text → Text Embeddings (512d) ←── Contrastive Alignment
    
    Key constraint: All downstream predictions use ONLY concept embeddings, not raw image features.
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        num_concepts: int,
        num_countries: int,
        num_cells: int,
        streetclip_dim: int = 768,
        concept_emb_dim: int = 512,
        coord_output_dim: int = 2,
        text_encoder: Optional[nn.Module] = None,
    ):
        """
        Args:
            image_encoder: Pretrained StreetCLIPEncoder (frozen)
            num_concepts: Number of concepts (k) - used for auxiliary concept classification
            num_countries: Number of countries (c)
            num_cells: Number of semantic geocells
            streetclip_dim: Output dimension of StreetCLIP image encoder (768)
            concept_emb_dim: Dimension of concept embedding space (512, matches text encoder)
            coord_output_dim: Output dimension for coordinate head (2 for lat/lng, 3 for sphere)
            text_encoder: Frozen text encoder for text embedding (used externally)
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder
        self.num_concepts = num_concepts
        self.num_countries = num_countries
        self.num_cells = num_cells
        self.streetclip_dim = streetclip_dim
        self.concept_emb_dim = concept_emb_dim
        self.coord_output_dim = coord_output_dim
        
        # Location Encoder (GeoCLIP style) - outputs 512d
        self.location_encoder = LocationEncoder()
        
        # ========== Concept Bottleneck Layer ==========
        # Maps image features (768d) → concept embeddings (512d)
        # This is the ONLY path from images to downstream tasks
        self.concept_bottleneck = nn.Sequential(
            nn.Linear(streetclip_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, concept_emb_dim),
            nn.LayerNorm(concept_emb_dim),
        )
        
        # ========== Downstream Heads (ALL operate on concept_emb ONLY) ==========
        
        # Concept Classification Head (auxiliary, for interpretability)
        # Input: concept_emb (512d) → concept logits
        self.concept_head = nn.Sequential(
            nn.Linear(concept_emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_concepts)
        )
        
        # Country Classification Head
        # Input: concept_emb (512d) → country logits
        self.country_head = nn.Sequential(
            nn.Linear(concept_emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, num_countries)
        )
        
        # Semantic Geocell Classification Head (Coarse Location)
        # Input: concept_emb (512d) → cell logits
        self.cell_head = nn.Sequential(
            nn.Linear(concept_emb_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(512, num_cells)
        )
        
        # Offset Regression Head (Fine Location)
        # Input: concept_emb (512d) → coordinate offsets
        self.offset_head = nn.Sequential(
            nn.Linear(concept_emb_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.2),
            nn.Linear(256, coord_output_dim)
        )
        
        # Initialize weights
        self._init_weights()
    
    def _init_weights(self):
        """Initialize projection layers with Xavier uniform initialization."""
        for module in [self.concept_bottleneck, self.concept_head, 
                       self.country_head, self.cell_head, self.offset_head]:
            for layer in module:
                if isinstance(layer, nn.Linear):
                    nn.init.xavier_uniform_(layer.weight)
                    if layer.bias is not None:
                        nn.init.zeros_(layer.bias)

    def forward(self, images: torch.Tensor, gps_coords: Optional[torch.Tensor] = None):
        """
        Forward pass through the CBM.
        
        Args:
            images: Image tensor [batch, 3, H, W]
            gps_coords: GPS coordinates [batch, 2] (lat, lon) - for GPS embedding during training
            
        Returns:
            Dict containing:
                - concept_emb: Concept embeddings [batch, 512] - the bottleneck representation
                - concept_logits: Concept classification logits [batch, num_concepts]
                - country_logits: Country predictions [batch, num_countries]
                - cell_logits: Geocell predictions [batch, num_cells]
                - pred_offsets: Predicted coordinate offsets [batch, coord_dim]
                - gps_emb: GPS embeddings [batch, 512] (if gps_coords provided)
        """
        # 1. Image Encoder (frozen) → Image Features
        x_img = self.image_encoder(images)  # [batch, 768]
        
        # 2. Concept Bottleneck → Concept Embeddings
        concept_emb = self.concept_bottleneck(x_img)  # [batch, 512]
        
        # 3. All downstream heads operate ONLY on concept_emb
        concept_logits = self.concept_head(concept_emb)  # [batch, num_concepts]
        country_logits = self.country_head(concept_emb)  # [batch, num_countries]
        cell_logits = self.cell_head(concept_emb)  # [batch, num_cells]
        pred_offsets = self.offset_head(concept_emb)  # [batch, coord_dim]
        
        result = {
            "concept_emb": concept_emb,
            "concept_logits": concept_logits,
            "country_logits": country_logits,
            "cell_logits": cell_logits,
            "pred_offsets": pred_offsets,
        }

        # 4. GPS Encoding (for contrastive alignment during training)
        if gps_coords is not None:
            gps_emb = self.encode_gps(gps_coords)  # [batch, 512]
            result["gps_emb"] = gps_emb
            
        return result
    
    def encode_gps(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode GPS coordinates into the same embedding space as concepts.
        
        Args:
            gps_coords: GPS coordinates [batch, 2] (lat, lon in degrees)
            
        Returns:
            gps_emb: GPS embeddings [batch, 512]
        """
        # LocationEncoder outputs 512d by default
        gps_emb = self.location_encoder(gps_coords)  # [batch, 512]
        return gps_emb
    
    def get_concept_embedding(self, images: torch.Tensor) -> torch.Tensor:
        """
        Get concept embeddings for images (inference utility).
        
        Args:
            images: Image tensor [batch, 3, H, W]
            
        Returns:
            concept_emb: Concept embeddings [batch, 512]
        """
        x_img = self.image_encoder(images)
        concept_emb = self.concept_bottleneck(x_img)
        return concept_emb

    def forward_from_features(
        self, image_features: torch.Tensor, gps_coords: Optional[torch.Tensor] = None
    ):
        """
        Forward pass using PRECOMPUTED image features (skips image encoder).
        
        Use this for Stage 1 and Stage 2 training when image encoder is frozen,
        to avoid redundant computation of image embeddings each epoch.
        
        Args:
            image_features: Precomputed image features [batch, 768] from frozen encoder
            gps_coords: GPS coordinates [batch, 2] (lat, lon) - for GPS embedding during training
            
        Returns:
            Same dict as forward() method
        """
        # 1. Skip image encoder - use precomputed features directly
        # 2. Concept Bottleneck → Concept Embeddings
        concept_emb = self.concept_bottleneck(image_features)  # [batch, 512]
        
        # 3. All downstream heads operate ONLY on concept_emb
        concept_logits = self.concept_head(concept_emb)  # [batch, num_concepts]
        country_logits = self.country_head(concept_emb)  # [batch, num_countries]
        cell_logits = self.cell_head(concept_emb)  # [batch, num_cells]
        pred_offsets = self.offset_head(concept_emb)  # [batch, coord_dim]
        
        result = {
            "concept_emb": concept_emb,
            "concept_logits": concept_logits,
            "country_logits": country_logits,
            "cell_logits": cell_logits,
            "pred_offsets": pred_offsets,
        }

        # 4. GPS Encoding (for contrastive alignment during training)
        if gps_coords is not None:
            gps_emb = self.encode_gps(gps_coords)  # [batch, 512]
            result["gps_emb"] = gps_emb
            
        return result

    @torch.no_grad()
    def extract_image_features(self, images: torch.Tensor) -> torch.Tensor:
        """
        Extract image features using the frozen image encoder.
        
        Use this for precomputing features before Stage 1/2 training.
        
        Args:
            images: Image tensor [batch, 3, H, W]
            
        Returns:
            image_features: Image features [batch, 768]
        """
        self.image_encoder.eval()
        return self.image_encoder(images)

    @torch.no_grad()
    def extract_concept_embeddings(self, image_features: torch.Tensor) -> torch.Tensor:
        """
        Extract concept embeddings from precomputed image features.
        
        Use this for precomputing concept embeddings before Stage 2 training.
        
        Args:
            image_features: Precomputed image features [batch, 768]
            
        Returns:
            concept_emb: Concept embeddings [batch, 512]
        """
        return self.concept_bottleneck(image_features)

    def forward_from_concept_emb(self, concept_emb: torch.Tensor):
        """
        Forward pass using PRECOMPUTED concept embeddings (skips image encoder AND concept bottleneck).
        
        Use this for Stage 2 training when both image encoder and concept bottleneck are frozen,
        to avoid redundant computation each epoch.
        
        Args:
            concept_emb: Precomputed concept embeddings [batch, 512] from frozen bottleneck
            
        Returns:
            Dict containing only Stage 2 outputs:
                - cell_logits: Geocell predictions [batch, num_cells]
                - pred_offsets: Predicted coordinate offsets [batch, coord_dim]
        """
        # Only run Stage 2 heads on precomputed concept embeddings
        cell_logits = self.cell_head(concept_emb)  # [batch, num_cells]
        pred_offsets = self.offset_head(concept_emb)  # [batch, coord_dim]
        
        return {
            "cell_logits": cell_logits,
            "pred_offsets": pred_offsets,
        }

    def parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        """Return parameters that should be optimized (excludes frozen image encoder)."""
        return (
            list(self.concept_bottleneck.parameters()) +
            list(self.concept_head.parameters()) +
            list(self.country_head.parameters()) +
            list(self.cell_head.parameters()) +
            list(self.offset_head.parameters()) +
            list(self.location_encoder.parameters())
        )

    # ========== Stage-Specific Parameter Methods ==========
    
    def get_stage1_params(self) -> Iterable[nn.Parameter]:
        """
        Return parameters trainable in Stage 1 (concept bottleneck + global alignment).
        Includes: concept_bottleneck, concept_head, country_head, location_encoder
        """
        return (
            list(self.concept_bottleneck.parameters()) +
            list(self.concept_head.parameters()) +
            list(self.country_head.parameters()) +
            list(self.location_encoder.parameters())
        )
    
    def get_stage2_params(self) -> Iterable[nn.Parameter]:
        """
        Return parameters trainable in Stage 2 (geolocation head training).
        Includes: cell_head, offset_head
        """
        return (
            list(self.cell_head.parameters()) +
            list(self.offset_head.parameters())
        )
    
    def freeze_stage1(self):
        """Freeze all Stage 1 parameters after Stage 1 training."""
        for p in self.get_stage1_params():
            p.requires_grad = False
    
    def freeze_image_encoder(self):
        """Freeze the image encoder (call after Stage 0)."""
        self.image_encoder.freeze_encoder()
    
    def freeze_all_except_stage2(self):
        """Freeze everything except Stage 2 heads."""
        # Freeze image encoder
        self.freeze_image_encoder()
        # Freeze Stage 1
        self.freeze_stage1()
        # Ensure Stage 2 is unfrozen
        for p in self.get_stage2_params():
            p.requires_grad = True

    # ========== Inference Methods ==========
    
    @torch.no_grad()
    def predict_location(
        self,
        images: torch.Tensor,
        cell_centers: torch.Tensor,
    ) -> dict:
        """
        Predict location from images only (for GeoGuessr inference).
        
        This is the main inference method - no GPS coordinates needed.
        
        Args:
            images: Image tensor [batch, 3, H, W]
            cell_centers: Geocell center coordinates [num_cells, 3] in Cartesian (x, y, z)
            
        Returns:
            Dict containing:
                - pred_lat: Predicted latitudes [batch]
                - pred_lng: Predicted longitudes [batch]
                - pred_coords: Predicted coordinates [batch, 2] as (lat, lng)
                - pred_cell: Predicted cell indices [batch]
                - cell_probs: Cell classification probabilities [batch, num_cells]
                - country_probs: Country classification probabilities [batch, num_countries]
                - concept_probs: Concept classification probabilities [batch, num_concepts]
        """
        self.eval()
        
        # Run forward pass (no GPS coords for inference)
        outputs = self.forward(images, gps_coords=None)
        
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        concept_logits = outputs["concept_logits"]
        country_logits = outputs["country_logits"]
        
        # Get predicted cell
        cell_probs = F.softmax(cell_logits, dim=1)
        pred_cells = cell_logits.argmax(dim=1)
        
        # Get cell centers for predicted cells
        pred_cell_centers = cell_centers[pred_cells]  # [batch, 3]
        
        # Compute final coordinates
        if self.coord_output_dim == 3:
            # 3D Cartesian output: add offset and normalize to unit sphere
            pred_cart = pred_cell_centers + pred_offsets
            pred_cart = F.normalize(pred_cart, p=2, dim=1)
            # Convert to lat/lng
            pred_lat, pred_lng = self._cartesian_to_latlng(pred_cart)
        else:
            # 2D lat/lng offset output
            # Convert cell center from Cartesian to lat/lng
            c_x, c_y, c_z = pred_cell_centers[:, 0], pred_cell_centers[:, 1], pred_cell_centers[:, 2]
            c_lat = torch.rad2deg(torch.asin(torch.clamp(c_z, -1.0, 1.0)))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            # Add offset
            pred_lat = c_lat + pred_offsets[:, 0]
            pred_lng = c_lng + pred_offsets[:, 1]
            # Normalize longitude to [-180, 180]
            pred_lng = ((pred_lng + 180) % 360) - 180
        
        pred_coords = torch.stack([pred_lat, pred_lng], dim=1)
        
        return {
            "pred_lat": pred_lat,
            "pred_lng": pred_lng,
            "pred_coords": pred_coords,
            "pred_cell": pred_cells,
            "cell_probs": cell_probs,
            "country_probs": F.softmax(country_logits, dim=1),
            "concept_probs": F.softmax(concept_logits, dim=1),
        }
    
    @staticmethod
    def _cartesian_to_latlng(cart: torch.Tensor) -> tuple:
        """
        Convert Cartesian coordinates on unit sphere to lat/lng.
        
        Args:
            cart: Cartesian coordinates [batch, 3] (x, y, z)
            
        Returns:
            Tuple of (lat, lng) tensors in degrees
        """
        x, y, z = cart[:, 0], cart[:, 1], cart[:, 2]
        lat = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
        lng = torch.rad2deg(torch.atan2(y, x))
        return lat, lng
