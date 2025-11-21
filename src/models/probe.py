import torch
import torch.nn as nn
from transformers import AutoModel

class StreetClipProbe(nn.Module):
    """
    Linear Probe on top of frozen StreetCLIP backbone.
    """
    def __init__(self, num_classes: int, model_name: str = "geolocal/StreetCLIP", freeze_backbone: bool = True):
        """
        Args:
            num_classes: Number of output classes (concepts).
            model_name: HuggingFace model identifier.
            freeze_backbone: Whether to freeze the backbone parameters.
        """
        super().__init__()
        
        # Load backbone
        print(f"Loading backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Determine embedding dimension
        # CLIP models usually expose projection_dim or hidden_size
        if hasattr(self.backbone.config, "projection_dim"):
            self.embed_dim = self.backbone.config.projection_dim
        elif hasattr(self.backbone.config, "hidden_size"):
            self.embed_dim = self.backbone.config.hidden_size
        else:
            # Fallback, commonly 512 or 768
            self.embed_dim = 768 
            print(f"Warning: Could not determine embedding dimension from config. Assuming {self.embed_dim}.")

        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen.")
            
        # Linear Probe Head
        self.head = nn.Linear(self.embed_dim, num_classes)
        
    def forward(self, images):
        """
        Args:
            images: Tensor of shape (B, C, H, W)
            
        Returns:
            logits: Tensor of shape (B, num_classes)
        """
        # Get features from backbone
        # HuggingFace CLIP models return output object
        # We typically want the image embeddings (pooled output)
        
        # Note: huggingface CLIP expects pixel_values argument
        outputs = self.backbone.get_image_features(pixel_values=images)
        
        # Normalize features (optional, but standard for CLIP probes)
        # outputs = outputs / outputs.norm(p=2, dim=-1, keepdim=True)
        
        # Pass through linear head
        logits = self.head(outputs)
        
        return logits

