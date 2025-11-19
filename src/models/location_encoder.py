"""
Location encoder using sinusoidal positional embeddings (GeoCLIP).
"""

from __future__ import annotations

import math
import torch
from torch import nn


class GeoCLIPLocationEncoder(nn.Module):
    """
    Location encoder based on GeoCLIP architecture.
    Uses sinusoidal positional embeddings (Fourier features) followed by an MLP.
    """

    def __init__(
        self,
        feature_dim: int = 768,
        hidden_dim: int = 512,
        num_frequencies: int = 32,
        sigma: float = 10.0,  # Scale for random frequencies
    ):
        super().__init__()
        self.feature_dim = feature_dim
        self.num_frequencies = num_frequencies
        self.sigma = sigma

        # Input is (sin, cos) for each frequency * 2 coordinates (lat, lng) = 4 * num_frequencies
        # But typically Fourier features are: [sin(2pi * f * x), cos(2pi * f * x), ...]
        # We map 2D coords -> High dim
        
        # GeoCLIP implementation style:
        # Random Fourier Features
        self.register_buffer(
            "frequencies", torch.randn(2, num_frequencies) * sigma
        )
        
        input_dim = 2 * num_frequencies  # sin and cos for each frequency projection
        
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, feature_dim),
        )

    def forward(self, coords: torch.Tensor) -> torch.Tensor:
        """
        Args:
            coords: Normalized coordinates in [-1, 1], shape (batch, 2)
                    (lat, lng) where lat in [-1, 1], lng in [-1, 1]
        
        Returns:
            Location embeddings: (batch, feature_dim)
        """
        # Check expected range (normalized)
        # coords are (lat_norm, lng_norm)
        
        # Project coordinates using random frequencies
        # (batch, 2) @ (2, num_frequencies) -> (batch, num_frequencies)
        projections = 2 * math.pi * coords @ self.frequencies
        
        # Fourier features: [sin(proj), cos(proj)]
        # shape: (batch, 2 * num_frequencies)
        features = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        
        # MLP
        return self.mlp(features)


