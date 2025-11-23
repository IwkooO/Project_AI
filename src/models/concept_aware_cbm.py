"""
Concept-Aware Global Image-GPS Alignment Model.
"""

from __future__ import annotations

from typing import Optional, Iterable, Union, Tuple

import torch
from torch import nn
import torch.nn.functional as F
from geoclip import LocationEncoder


class ConceptAwareGeoModel(nn.Module):
    """
    Concept-Aware Global Image-GPS Alignment Model.
    
    Components:
    - Image Encoder: StreetCLIP
    - Location Encoder: GeoCLIP's LocationEncoder
    - Concept Alignment: Joint projection to concept space
    """

    def __init__(
        self,
        image_encoder: nn.Module,
        concept_features: torch.Tensor,
        num_concepts: int,
        num_countries: int,
        num_cells: int,
        streetclip_dim: int = 768,
        location_encoder_dim: int = 512,
        coord_output_dim: int = 2,
        text_encoder: Optional[nn.Module] = None,
    ):
        """
        Args:
            image_encoder: Pretrained StreetCLIPEncoder
            concept_features: Pre-computed concept text embeddings (E_concept) [k, d_streetclip]
            num_concepts: Number of concepts (k)
            num_countries: Number of countries (c)
            num_cells: Number of semantic geocells
            streetclip_dim: Output dimension of StreetCLIP encoder
            location_encoder_dim: Output dimension of LocationEncoder (default 512)
            coord_output_dim: Output dimension for coordinate head (2 for lat/lng, 3 for sphere)
            text_encoder: Frozen text encoder for semantic loss
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.text_encoder = text_encoder # Should be frozen/handled externally but good to have ref
        self.location_encoder = LocationEncoder()
        self.num_concepts = num_concepts
        self.num_countries = num_countries
        self.num_cells = num_cells
        self.streetclip_dim = streetclip_dim
        self.coord_output_dim = coord_output_dim
        
        # Ensure concept features are float32 and on the correct device (handled in forward/to)
        # We register E_concept as a buffer so it's saved with the model but not updated by optimizer
        self.register_buffer("concept_basis_init", concept_features.clone().detach())
        
        # Learnable offset delta (initialized to small random values or zeros)
        # Shape matches concept_features: [k, d_streetclip]
        self.delta = nn.Parameter(torch.zeros_like(concept_features))
        
        # Image Projector (f_img): Maps image embeddings to concept activations
        # Input: d_streetclip, Output: k (concepts)
        self.image_projector = nn.Sequential(
            nn.Linear(streetclip_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_concepts)
        )
        
        # Semantic Geocell Head (Coarse)
        # Input: [Concepts, z_img] -> Fused Dimension
        fused_dim = num_concepts + streetclip_dim
        self.cell_head = nn.Sequential(
            nn.Linear(fused_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(1024, num_cells)
        )
        
        # Offset Head (Fine)
        # Input: Fused Dimension -> 2 (lat, lng offset) or 3 (xyz)
        self.offset_head = nn.Sequential(
            nn.Linear(fused_dim, 512),
            nn.LayerNorm(512),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(512, coord_output_dim)
        )
        
        # Country Head: Maps Concept Activations to Country Logits (Auxiliary)
        self.country_head = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_countries)
        )
        
        # Location Adapter: Maps LocationEncoder output (512) to StreetCLIP dimension
        self.location_adapter = nn.Linear(location_encoder_dim, streetclip_dim)
        
    def get_concept_basis(self) -> torch.Tensor:
        """
        Compute the current concept basis B = E_concept + Delta.
        Returns B of shape [d_streetclip, k]
        """
        # E_concept is stored as [k, d], so we add delta [k, d] then transpose to get [d, k]
        basis = self.concept_basis_init + self.delta
        return basis.t()

    def forward(self, images: torch.Tensor, gps_coords: Optional[torch.Tensor] = None):
        """
        Args:
            images: Image tensor [batch, 3, H, W]
            gps_coords: GPS coordinates [batch, 2] (lat, lon) - Optional for inference
            
        Returns:
            z_img: Image concept activations [batch, k]
            z_loc: Location concept activations [batch, k] (if gps provided)
            country_logits: Country predictions [batch, c]
            cell_logits: Geocell predictions [batch, num_cells]
            pred_offsets: Predicted coordinate offsets [batch, coord_dim]
            fused_features: [batch, fused_dim]
        """
        # 1. Image Path (Always executed)
        # x_img: [batch, d_streetclip]
        x_img = self.image_encoder(images)
        # z_img: [batch, k]
        z_img = self.image_projector(x_img)
        
        # Fusion: [Concepts, StreetCLIP features]
        fused_features = torch.cat([z_img, x_img], dim=1)
        
        # Hierarchical Heads
        cell_logits = self.cell_head(fused_features)
        pred_offsets = self.offset_head(fused_features)
        
        # Country from Concepts (Interpretability check)
        country_logits = self.country_head(z_img)
        
        result = {
            "z_img": z_img,
            "country_logits": country_logits,
            "cell_logits": cell_logits,
            "pred_offsets": pred_offsets,
            "fused_features": fused_features
        }

        if gps_coords is not None:
            # 2. Location Path (Training mode)
            x_loc_raw = self.location_encoder(gps_coords)
            x_loc = self.location_adapter(x_loc_raw)
            B = self.get_concept_basis()
            z_loc = torch.matmul(x_loc, B)
            result["z_loc"] = z_loc
            
        return result
        
    def encode_location(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode GPS coordinates into concept space (for building gallery).
        """
        x_loc_raw = self.location_encoder(gps_coords)
        x_loc = self.location_adapter(x_loc_raw)
        B = self.get_concept_basis()
        z_loc = torch.matmul(x_loc, B)
        return z_loc
    
    def parameters_to_optimize(self) -> Iterable[nn.Parameter]:
        """Return parameters that should be optimized."""
        return (
            list(self.image_projector.parameters()) + 
            list(self.location_adapter.parameters()) + 
            list(self.location_encoder.parameters()) + 
            list(self.country_head.parameters()) +
            list(self.cell_head.parameters()) +
            list(self.offset_head.parameters()) +
            [self.delta]
        )
