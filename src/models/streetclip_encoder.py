"""
StreetCLIP encoder wrapper for CBM geolocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, List, Union

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
        self.tokenizer = CLIPTokenizer.from_pretrained(self.config.model_name)

        if not self.config.finetune:
            self.freeze_encoder()

        if self.config.device is not None:
            self.model.to(self.config.device)

        # Use projection dim (shared embedding space) for contrastive learning
        self.feature_dim = self.model.config.projection_dim
        # Also store raw hidden size for CBM heads that may need it
        self.hidden_size = self.model.vision_model.config.hidden_size

    def freeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = False

    def unfreeze_encoder(self):
        for param in self.model.parameters():
            param.requires_grad = True

    def unfreeze_top_layers(self, num_layers: int = 2):
        """
        Unfreeze only the top N transformer layers of the vision encoder + projection.
        Used for Stage 0 domain contrastive pretraining.
        
        Args:
            num_layers: Number of top layers to unfreeze (default: 2)
        """
        # Freeze everything first
        self.freeze_encoder()
        # Unfreeze top N layers of vision_model.encoder.layers
        layers = self.model.vision_model.encoder.layers
        for layer in layers[-num_layers:]:
            for param in layer.parameters():
                param.requires_grad = True
        # Also unfreeze post_layernorm and visual_projection
        for param in self.model.vision_model.post_layernorm.parameters():
            param.requires_grad = True
        for param in self.model.visual_projection.parameters():
            param.requires_grad = True

    def unfreeze_text_encoder(self):
        """Unfreeze the text encoder for Stage 0 training."""
        for param in self.model.text_model.parameters():
            param.requires_grad = True
        for param in self.model.text_projection.parameters():
            param.requires_grad = True

    def freeze_text_encoder(self):
        """Freeze the text encoder after Stage 0."""
        for param in self.model.text_model.parameters():
            param.requires_grad = False
        for param in self.model.text_projection.parameters():
            param.requires_grad = False

    def get_trainable_params(self):
        """Return list of parameters with requires_grad=True."""
        return [p for p in self.model.parameters() if p.requires_grad]

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Args:
            pixel_values: Preprocessed CLIP pixel values [batch, 3, 336, 336]
        Returns:
            Projected image features [batch, projection_dim] (768 for CLIP ViT-L)
        """
        # Use get_image_features to get projected embeddings (same space as text)
        return self.model.get_image_features(pixel_values=pixel_values)

    def get_unprojected_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get raw CLS token features before projection (for CBM heads).
        
        Args:
            pixel_values: Preprocessed CLIP pixel values [batch, 3, 336, 336]
        Returns:
            Raw CLS token features [batch, hidden_size] (1024 for CLIP ViT-L)
        """
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state[:, 0]

    def get_patch_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get all patch token features (excluding CLS token).
        
        For ViT-L/14@336px: 336/14 = 24, so 24x24 = 576 patches.
        
        Args:
            pixel_values: Preprocessed CLIP pixel values [batch, 3, 336, 336]
        Returns:
            Patch tokens [batch, 576, hidden_size] (1024 for CLIP ViT-L)
        """
        outputs = self.model.vision_model(pixel_values=pixel_values)
        # last_hidden_state: [batch, 1 + num_patches, hidden_size]
        # First token is CLS, rest are patch tokens
        return outputs.last_hidden_state[:, 1:]

    def get_all_tokens(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """
        Get CLS + all patch tokens.
        
        Args:
            pixel_values: Preprocessed CLIP pixel values [batch, 3, 336, 336]
        Returns:
            All tokens [batch, 1 + num_patches, hidden_size] (577 tokens for ViT-L/14@336)
        """
        outputs = self.model.vision_model(pixel_values=pixel_values)
        return outputs.last_hidden_state

    def get_features_and_patches(self, pixel_values: torch.Tensor) -> tuple:
        """
        Get both projected CLS features and raw patch tokens in one forward pass.
        
        Args:
            pixel_values: Preprocessed CLIP pixel values [batch, 3, 336, 336]
        Returns:
            Tuple of:
                - image_features: Projected CLS features [batch, projection_dim] (768)
                - patch_tokens: Raw patch tokens [batch, 576, hidden_size] (1024)
        """
        vision_outputs = self.model.vision_model(pixel_values=pixel_values)
        # Get patch tokens (exclude CLS)
        patch_tokens = vision_outputs.last_hidden_state[:, 1:]
        # Get projected CLS (through post_layernorm and visual_projection)
        pooled_output = vision_outputs.last_hidden_state[:, 0]
        pooled_output = self.model.vision_model.post_layernorm(pooled_output)
        image_features = self.model.visual_projection(pooled_output)
        return image_features, patch_tokens

    @torch.no_grad()
    def get_image_features(self, pixel_values: torch.Tensor) -> torch.Tensor:
        return self.forward(pixel_values)

    @torch.no_grad()
    def get_text_features(self, text: Union[List[str], str]) -> torch.Tensor:
        """
        Get text features for a list of strings or a single string (inference mode).
        
        Args:
            text: List of strings or single string to encode
            
        Returns:
            Text features tensor [batch_size, hidden_size]
        """
        return self._encode_text(text)

    def get_text_features_trainable(self, text: Union[List[str], str]) -> torch.Tensor:
        """
        Get text features with gradient tracking (for Stage 0 training).
        
        Args:
            text: List of strings or single string to encode
            
        Returns:
            Text features tensor [batch_size, hidden_size]
        """
        return self._encode_text(text)

    def _encode_text(self, text: Union[List[str], str]) -> torch.Tensor:
        """Internal text encoding (shared by inference and training modes)."""
        if isinstance(text, str):
            text = [text]
            
        inputs = self.tokenizer(
            text, 
            padding=True, 
            truncation=True, 
            return_tensors="pt"
        )
        
        # Move inputs to the same device as the model
        model_device = next(self.model.parameters()).device
        inputs = {k: v.to(model_device) for k, v in inputs.items()}
            
        text_features = self.model.get_text_features(**inputs)
        return text_features
