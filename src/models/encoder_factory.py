"""
Generic encoder factory for vision transformer models.
Supports CLIP-based models and any AutoModel-compatible vision models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers import CLIPImageProcessor, CLIPModel, AutoImageProcessor, AutoModel


@dataclass
class EncoderConfig:
    """Configuration for vision encoder."""

    model_name: str
    finetune: bool = False
    device: Optional[torch.device] = None


class VisionEncoder(nn.Module):
    """Generic wrapper for vision transformer encoders."""

    def __init__(self, config: EncoderConfig):
        super().__init__()
        self.config = config
        self.model_name = config.model_name.lower()
        
        # Load model - only CLIP needs special handling, everything else uses AutoModel
        if self._is_clip_model():
            self._load_clip_model()
        else:
            self._load_auto_model()
        
        if not self.config.finetune:
            self.freeze_encoder()
        
        if self.config.device is not None:
            self.model.to(self.config.device)
        
        # Extract feature dimension
        self.feature_dim = self._get_feature_dim()

    def _is_clip_model(self) -> bool:
        """Check if model is CLIP-based."""
        clip_indicators = ["clip", "streetclip"]
        return any(indicator in self.model_name for indicator in clip_indicators)

    def _load_clip_model(self):
        """Load CLIP-based model."""
        self.model = CLIPModel.from_pretrained(self.config.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config.model_name)
        self.encoder_type = "clip"

    def _load_auto_model(self):
        """Load any vision model using AutoModel."""
        try:
            self.model = AutoModel.from_pretrained(self.config.model_name)
            try:
                self.image_processor = AutoImageProcessor.from_pretrained(self.config.model_name)
            except Exception:
                # Some models might not have processors, use default
                self.image_processor = None
            self.encoder_type = "auto"
        except Exception as e:
            raise ValueError(
                f"Could not load model {self.config.model_name}. "
                f"Error: {e}. Make sure transformers is up to date: "
                f"pip install --upgrade transformers"
            )

    def _get_feature_dim(self) -> int:
        """Extract feature dimension from model."""
        if self.encoder_type == "clip":
            return self.model.vision_model.config.hidden_size
        else:
            # Generic AutoModel - try common attribute names
            if hasattr(self.model.config, "hidden_size"):
                return self.model.config.hidden_size
            elif hasattr(self.model.config, "embed_dim"):
                return self.model.config.embed_dim
            elif hasattr(self.model, "embeddings"):
                # Try to infer from embeddings
                if hasattr(self.model.embeddings, "patch_embeddings"):
                    if hasattr(self.model.embeddings.patch_embeddings, "projection"):
                        if hasattr(self.model.embeddings.patch_embeddings.projection, "out_channels"):
                            return self.model.embeddings.patch_embeddings.projection.out_channels
                        elif hasattr(self.model.embeddings.patch_embeddings.projection, "out_features"):
                            return self.model.embeddings.patch_embeddings.projection.out_features
            # Last resort: use dummy forward pass
            return self._infer_dim_from_forward()
    
    def _infer_dim_from_forward(self) -> int:
        """Infer feature dimension by running a dummy forward pass."""
        with torch.no_grad():
            # Try common input sizes
            for img_size in [224, 336, 518]:
                try:
                    dummy_input = torch.zeros(1, 3, img_size, img_size)
                    if self.config.device is not None:
                        dummy_input = dummy_input.to(self.config.device)
                    output = self.forward(dummy_input)
                    return output.shape[-1]
                except Exception:
                    continue
            
            # If all sizes fail, raise error
            raise ValueError(
                f"Could not infer feature dimension from model {self.config.model_name}. "
                f"Tried input sizes [224, 336, 518]. Please check model configuration."
            )

    def freeze_encoder(self):
        """Freeze all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        """Unfreeze all encoder parameters."""
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Preprocessed image tensor [batch, 3, H, W]
        Returns:
            CLS token features [batch, hidden_size]
        """
        if self.encoder_type == "clip":
            outputs = self.model.vision_model(pixel_values=pixel_values)
            cls_embeddings = outputs.last_hidden_state[:, 0]
            return cls_embeddings
        else:
            # Generic AutoModel - try common output patterns
            outputs = self.model(pixel_values=pixel_values)
            if hasattr(outputs, "last_hidden_state"):
                # Standard case: use CLS token (first token)
                cls_embeddings = outputs.last_hidden_state[:, 0]
            elif hasattr(outputs, "pooler_output"):
                # Some models have pooler_output
                cls_embeddings = outputs.pooler_output
            elif isinstance(outputs, torch.Tensor):
                # Some models return tensor directly
                cls_embeddings = outputs[:, 0] if len(outputs.shape) > 2 else outputs
            elif isinstance(outputs, (tuple, list)):
                # Some models return tuple (last_hidden_state, ...)
                cls_embeddings = outputs[0][:, 0] if len(outputs[0].shape) > 2 else outputs[0]
            elif isinstance(outputs, dict):
                # Try to get from dict
                if "last_hidden_state" in outputs:
                    cls_embeddings = outputs["last_hidden_state"][:, 0]
                elif "pooler_output" in outputs:
                    cls_embeddings = outputs["pooler_output"]
                else:
                    raise ValueError(f"Could not extract features from model output: {outputs.keys()}")
            else:
                raise ValueError(f"Unexpected output type from model: {type(outputs)}")
            return cls_embeddings

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Get image features without gradients."""
        return self.forward(pixel_values)


def create_encoder(model_name: str, finetune: bool = False, device: Optional[torch.device] = None) -> VisionEncoder:
    """
    Factory function to create a vision encoder.
    
    Args:
        model_name: HuggingFace model identifier (e.g., "geolocal/StreetCLIP", "facebook/dinov3-vit7b16-pretrain-lvd1689m")
        finetune: Whether to allow fine-tuning the encoder
        device: Device to load the model on
    
    Returns:
        VisionEncoder instance
    """
    config = EncoderConfig(
        model_name=model_name,
        finetune=finetune,
        device=device,
    )
    return VisionEncoder(config)


@dataclass
class CoarseRouterConfig:
    """Configuration for the coarse-to-fine routing mechanism."""
    
    num_coarse_cells: int = 64  # Total number of coarse regions (e.g. S2 cells or K-means clusters)
    hidden_dim: int = 512
    dropout: float = 0.1


class HierarchicalRouter(nn.Module):
    """
    Coarse-to-Fine Router.
    Predicts a coarse region first, then routes to a region-specific coordinate head.
    Also supports integrating location encoding logic directly.
    """
    
    def __init__(self, config: CoarseRouterConfig, input_dim: int):
        super().__init__()
        self.config = config
        
        # Coarse predictor (Router)
        # Predicts logits for each coarse cell
        self.router_head = nn.Sequential(
            nn.Linear(input_dim, config.hidden_dim),
            nn.LayerNorm(config.hidden_dim),
            nn.GELU(),
            nn.Dropout(config.dropout),
            nn.Linear(config.hidden_dim, config.num_coarse_cells)
        )
        
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        """
        Args:
            features: Input features (e.g. concept probabilities or embeddings) [B, D]
        Returns:
            coarse_logits: [B, num_coarse_cells]
        """
        return self.router_head(features)


class GeoCLIPLocationEncoder(nn.Module):
    """
    Location encoder based on GeoCLIP architecture.
    Uses sinusoidal positional embeddings (Fourier features) followed by an MLP.
    Migrated from location_encoder.py to centralize factory logic.
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
            coords: Coordinates, shape (batch, 2) or (batch, 3)
                    - If (batch, 2): Normalized coordinates in [-1, 1] (lat, lng)
                    - If (batch, 3): 3D Cartesian coordinates on unit sphere (x, y, z)
        
        Returns:
            Location embeddings: (batch, feature_dim)
        """
        # Handle 3D Cartesian coordinates (from vMF loss)
        if coords.shape[-1] == 3:
            # Convert 3D Cartesian to normalized lat/lng
            # Check for zero vectors (from NaN replacement) - use default location
            coords_norm = coords.norm(p=2, dim=-1, keepdim=True)
            zero_mask = coords_norm.squeeze(-1) < 1e-6
            
            # Normalize to ensure unit sphere, handling zero vectors
            coords_3d = coords.clone()
            # Replace zero vectors with default (equator, prime meridian) before normalizing
            default_vec = torch.tensor([1.0, 0.0, 0.0], device=coords.device, dtype=coords.dtype)
            coords_3d[zero_mask] = default_vec
            coords_3d = F.normalize(coords_3d, p=2, dim=-1)
            
            x, y, z = coords_3d[:, 0], coords_3d[:, 1], coords_3d[:, 2]
            z = torch.clamp(z, -1.0, 1.0)
            lat_rad = torch.asin(z)
            lng_rad = torch.atan2(y, x)
            # Convert to degrees then normalize
            lat_deg = torch.rad2deg(lat_rad)
            lng_deg = torch.rad2deg(lng_rad)
            lat_norm = lat_deg / 90.0
            lng_norm = lng_deg / 180.0
            coords_2d = torch.stack([lat_norm, lng_norm], dim=1)
        else:
            # Already 2D normalized coordinates
            coords_2d = coords
        
        # Project coordinates using random frequencies
        # (batch, 2) @ (2, num_frequencies) -> (batch, num_frequencies)
        projections = 2 * 3.14159265359 * coords_2d @ self.frequencies
        
        # Fourier features: [sin(proj), cos(proj)]
        features = torch.cat([torch.sin(projections), torch.cos(projections)], dim=-1)
        
        return self.mlp(features)

