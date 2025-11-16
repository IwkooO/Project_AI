"""Configuration defaults for StreetCLIP CBM geolocation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class TrainingStageConfig:
    concept_epochs: int = 5
    prediction_epochs: int = 5
    finetune_epochs: int = 0


@dataclass
class StreetCLIPCBMConfig:
    encoder_model: str = "geolocal/StreetCLIP"  # Generic encoder model name
    streetclip_model: Optional[str] = None  # Deprecated: use encoder_model instead
    image_size: int = 336
    batch_size: int = 16
    encoder_lr: float = 1e-5
    cbm_lr: float = 1e-3
    finetune_lr: float = 1e-5
    concept_weight: float = 1.0
    distance_weight: float = 1.0
    country_weight: float = 0.5
    coordinate_loss_type: str = "mse"
    sequential: bool = True
    country_filter: Optional[str] = None
    require_coordinates: bool = False
    stages: TrainingStageConfig = field(default_factory=TrainingStageConfig)
    
    def __post_init__(self):
        """Handle backward compatibility for streetclip_model."""
        if self.streetclip_model is not None:
            self.encoder_model = self.streetclip_model
        valid_loss_types = {"mse", "sphere"}
        if self.coordinate_loss_type not in valid_loss_types:
            raise ValueError(
                f"Invalid coordinate_loss_type '{self.coordinate_loss_type}'. "
                f"Expected one of {sorted(valid_loss_types)}."
            )


# Deprecated: Feature dimensions are now auto-detected from encoders
FEATURE_DIM_BY_MODEL: Dict[str, int] = {
    "geolocal/StreetCLIP": 1024,
    "openai/clip-vit-large-patch14-336": 1024,
}

DEFAULT_CONFIG = StreetCLIPCBMConfig()


