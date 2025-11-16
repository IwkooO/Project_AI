"""
Generic encoder factory for vision transformer models.
Supports CLIP-based models and any AutoModel-compatible vision models.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch
from torch import nn
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

