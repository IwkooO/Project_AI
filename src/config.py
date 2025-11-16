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
    streetclip_model: str = "geolocal/StreetCLIP"
    image_size: int = 336
    batch_size: int = 16
    encoder_lr: float = 1e-5
    cbm_lr: float = 1e-3
    finetune_lr: float = 1e-5
    concept_weight: float = 1.0
    distance_weight: float = 1.0
    country_weight: float = 0.5
    sequential: bool = True
    country_filter: Optional[str] = None
    require_coordinates: bool = False
    stages: TrainingStageConfig = field(default_factory=TrainingStageConfig)


FEATURE_DIM_BY_MODEL: Dict[str, int] = {
    "geolocal/StreetCLIP": 1024,
    "openai/clip-vit-large-patch14-336": 1024,
}

DEFAULT_CONFIG = StreetCLIPCBMConfig()


