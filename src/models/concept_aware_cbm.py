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
        streetclip_dim: int = 768,
        location_encoder_dim: int = 512,
        coord_output_dim: int = 2,
    ):
        """
        Args:
            image_encoder: Pretrained StreetCLIPEncoder
            concept_features: Pre-computed concept text embeddings (E_concept) [k, d_streetclip]
            num_concepts: Number of concepts (k)
            num_countries: Number of countries (c)
            streetclip_dim: Output dimension of StreetCLIP encoder
            location_encoder_dim: Output dimension of LocationEncoder (default 512)
            coord_output_dim: Output dimension for coordinate head (2 for lat/lng, 3 for sphere)
        """
        super().__init__()
        self.image_encoder = image_encoder
        self.location_encoder = LocationEncoder()
        self.num_concepts = num_concepts
        self.num_countries = num_countries
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
        # Added dropout for regularization to prevent overfitting
        self.image_projector = nn.Sequential(
            nn.Linear(streetclip_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),  # Dropout for regularization
            nn.Linear(256, num_concepts)
        )
        
        # Country Head: Maps concept activations to country logits
        # Input: k (concepts), Output: c (countries)
        self.country_head = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, num_countries)
        )
        
        # Coordinate Regression Head: Maps concept activations to coordinates
        # Input: k (concepts), Output: coord_output_dim (2 or 3)
        self.coord_head = nn.Sequential(
            nn.Linear(num_concepts, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(256, coord_output_dim)
        )

        # Location Adapter: Maps LocationEncoder output (512) to StreetCLIP dimension
        # This allows the location embedding x_loc to be compatible with the basis B
        self.location_adapter = nn.Linear(location_encoder_dim, streetclip_dim)
        
    def get_concept_basis(self) -> torch.Tensor:
        """
        Compute the current concept basis B = E_concept + Delta.
        Returns B of shape [d_streetclip, k]
        """
        # E_concept is stored as [k, d], so we add delta [k, d] then transpose to get [d, k]
        basis = self.concept_basis_init + self.delta
        return basis.t()

    def forward(self, images: torch.Tensor, gps_coords: Optional[torch.Tensor] = None) -> Union[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor, torch.Tensor]]:
        """
        Args:
            images: Image tensor [batch, 3, H, W]
            gps_coords: GPS coordinates [batch, 2] (lat, lon) - Optional for inference
            
        Returns:
            If gps_coords is provided (Training):
                z_img: Image concept activations [batch, k]
                z_loc: Location concept activations [batch, k]
                country_logits: Country predictions [batch, c]
                pred_coords: Predicted coordinates [batch, 2]
            If gps_coords is None (Inference):
                z_img: Image concept activations [batch, k]
                country_logits: Country predictions [batch, c]
                pred_coords: Predicted coordinates [batch, 2]
        """
        # 1. Image Path (Always executed)
        # x_img: [batch, d_streetclip]
        x_img = self.image_encoder(images)
        # z_img: [batch, k]
        z_img = self.image_projector(x_img)
        
        # Predict country from concept activations
        country_logits = self.country_head(z_img)
        
        # Predict coordinates from concept activations
        pred_coords = self.coord_head(z_img)
        
        # If predicting 3D sphere coordinates, normalize to unit sphere
        if self.coord_output_dim == 3:
            pred_coords = F.normalize(pred_coords, p=2, dim=1)
        
        # If no GPS coordinates provided, return image concept vector, country logits, and pred_coords (Inference mode)
        if gps_coords is None:
            return z_img, country_logits, pred_coords
            
        # 2. Location Path (Training mode)
        # x_loc_raw: [batch, 512]
        x_loc_raw = self.location_encoder(gps_coords)
        # x_loc: [batch, d_streetclip]
        x_loc = self.location_adapter(x_loc_raw)
        
        # 3. Projection to Concept Space via Basis B
        # B: [d_streetclip, k]
        B = self.get_concept_basis()
        
        # z_loc = x_loc @ B  -> [batch, d] @ [d, k] = [batch, k]
        z_loc = torch.matmul(x_loc, B)
        
        return z_img, z_loc, country_logits, pred_coords
        
    def encode_location(self, gps_coords: torch.Tensor) -> torch.Tensor:
        """
        Encode GPS coordinates into concept space (for building gallery).
        Args:
            gps_coords: [batch, 2]
        Returns:
            z_loc: [batch, k]
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
            list(self.coord_head.parameters()) +
            [self.delta]
        )
