"""
StreetCLIP encoder wrapper for CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List

import torch
from torch import nn
from transformers import CLIPImageProcessor, CLIPModel, CLIPTokenizer


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
        try:
            self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name)
        except Exception:
            # Fallback for some models that might not have tokenizer config explicitly matching
            self.tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")

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

    @torch.no_grad()
    def encode_text(self, text_list: List[str], device: Optional[torch.device] = None) -> torch.Tensor:
        """
        Encode a list of text strings using the CLIP text encoder.
        
        Args:
            text_list: List of strings to encode
            device: Target device for computation (defaults to model device)
            
        Returns:
            Normalized text embeddings [num_texts, feature_dim]
        """
        target_device = device if device is not None else self.model.device
        
        inputs = self.tokenizer(
            text_list, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        ).to(target_device)
        
        text_features = self.model.get_text_features(**inputs)
        
        # Normalize features
        text_features = text_features / text_features.norm(p=2, dim=-1, keepdim=True)
        
        return text_features
