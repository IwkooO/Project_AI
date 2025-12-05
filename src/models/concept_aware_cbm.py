"""
Concept-Aware Global Image-GPS Alignment Model.

Strict Concept Bottleneck Model (CBM) architecture where ALL downstream tasks
operate on concept embeddings, not raw image features.
"""

from __future__ import annotations

from typing import Optional, Iterable

import torch
from torch import nn
import torch.nn.functional as F
from geoclip import LocationEncoder


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
