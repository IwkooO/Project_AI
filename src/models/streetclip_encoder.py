"""
StreetCLIP encoder wrapper for CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel


@dataclass
class StreetCLIPConfig:
    """Configuration for StreetCLIP encoder."""

    model_name: str = "geolocal/StreetCLIP"
    finetune: bool = False
    device: Optional[torch.device] = None


class StreetCLIPEncoder(nn.Module):
    """Wrapper around a pretrained StreetCLIP vision encoder."""

    def __init__(self, config: Optional[StreetCLIPConfig] = None):
        super().__init__()
        self.config = config or StreetCLIPConfig()
        self.model = CLIPModel.from_pretrained(self.config.model_name)
        self.image_processor = CLIPImageProcessor.from_pretrained(self.config.model_name)

        if not self.config.finetune:
            self.freeze_encoder()

        if self.config.device is not None:
            self.model.to(self.config.device)

        self.feature_dim = self.model.vision_model.config.hidden_size

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Preprocessed CLIP pixel values [batch, 3, 336, 336]
        Returns:
            CLS token features [batch, hidden_size]
        """
        outputs = self.model.vision_model(pixel_values=pixel_values)
        cls_embeddings = outputs.last_hidden_state[:, 0]
        return cls_embeddings

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values)




