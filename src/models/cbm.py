import torch
import torch.nn as nn
from transformers import AutoModel

class GeoCBM(nn.Module):
    """
    Joint Concept Bottleneck Model for Geolocation.
    Strict Bottleneck: Country and Coordinates are predicted SOLELY from Concept predictions.
    """
    def __init__(self, 
                 num_concepts: int, 
                 num_countries: int, 
                 model_name: str = "geolocal/StreetCLIP", 
                 freeze_backbone: bool = True,
                 hidden_dim_coord: int = 512):
        """
        Args:
            num_concepts: Number of concept classes.
            num_countries: Number of country classes.
            model_name: HuggingFace model identifier.
            freeze_backbone: Whether to freeze the backbone parameters.
            hidden_dim_coord: Hidden dimension for the coordinate predictor MLP.
        """
        super().__init__()
        
        # Load backbone
        print(f"Loading backbone: {model_name}")
        self.backbone = AutoModel.from_pretrained(model_name)
        
        # Determine embedding dimension
        if hasattr(self.backbone.config, "projection_dim"):
            self.embed_dim = self.backbone.config.projection_dim
        elif hasattr(self.backbone.config, "hidden_size"):
            self.embed_dim = self.backbone.config.hidden_size
        else:
            self.embed_dim = 768 
            
        # Freeze backbone
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
            print("Backbone frozen.")

        # 1. Concept Head (The Bottleneck)
        # Maps Image Features -> Concept Logits
        self.concept_head = nn.Linear(self.embed_dim, num_concepts)
        
        # 2. Country Head (Downstream)
        # Maps Concept Logits -> Country Logits
        # Strict bottleneck: Input is num_concepts
        self.country_head = nn.Linear(num_concepts, num_countries)
        
        # 3. Coordinate Head (Downstream)
        # Maps Concept Logits -> Lat/Lon
        # MLP for better mapping capability from concepts to coords
        self.coord_head = nn.Sequential(
            nn.Linear(num_concepts, hidden_dim_coord),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(hidden_dim_coord, hidden_dim_coord // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim_coord // 2, 2) # Output: Lat, Lon (Degrees)
        )
        
    def forward(self, images, return_attentions: bool = False):
        """
        Args:
            images: Tensor of shape (B, C, H, W)
            return_attentions: If True, returns attention maps from the backbone.
            
        Returns:
            tuple: (concept_logits, country_logits, coord_preds)
            OR
            tuple: ((concept_logits, country_logits, coord_preds), attentions) if return_attentions=True
        """
        # 1. Backbone Forward
        # Use vision_model directly to support output_attentions
        # Some older versions or specific CLIP implementations don't support return_dict in forward
        # But they typically return a tuple if return_dict is not passed or None
        # We trust it to follow standard HF conventions or the config default
        
        vision_outputs = self.backbone.vision_model(
            pixel_values=images, 
            output_attentions=return_attentions
        )
        
        # Check if output is dict-like or tuple
        if hasattr(vision_outputs, 'pooler_output'):
            pooler_output = vision_outputs.pooler_output
            attentions = vision_outputs.attentions if return_attentions else None
        else:
            # Tuple return: (last_hidden_state, pooler_output, hidden_states, attentions)
            # But looking at CLIPVisionTransformer, it returns (last_hidden_state, pooler_output, attentions)
            # We need to be careful. 
            # Usually: output[0] = last_hidden_state, output[1] = pooler_output
            pooler_output = vision_outputs[1]
            attentions = vision_outputs.attentions if hasattr(vision_outputs, 'attentions') else (vision_outputs[-1] if return_attentions else None)
        
        # Apply projection if it exists (standard in CLIP)
        if hasattr(self.backbone, 'visual_projection'):
            image_features = self.backbone.visual_projection(pooler_output)
        else:
            image_features = pooler_output

        # 2. Concept Prediction (Bottleneck)
        concept_logits = self.concept_head(image_features)
        
        # 3. Downstream Predictions (Strict Bottleneck)
        # We use logits as input. 
        # Ideally, if concepts are "presence", we might want sigmoid(logits), 
        # but logits preserve more information for gradients.
        # Using raw logits is standard for "soft" CBMs.
        
        country_logits = self.country_head(concept_logits)
        coord_preds = self.coord_head(concept_logits)
        
        outputs = (concept_logits, country_logits, coord_preds)
        
        if return_attentions:
            return outputs, vision_outputs.attentions
            
        return outputs

    def load_probe_weights(self, probe_path: str):
        """
        Load weights from a pre-trained Concept Probe.
        Expects the probe to have 'backbone' and 'head' (mapped to concept_head).
        """
        print(f"Loading probe weights from {probe_path}")
        state_dict = torch.load(probe_path, map_location='cpu')
        
        # If state_dict contains 'model_state_dict' key (common practice), use it
        if 'model_state_dict' in state_dict:
            state_dict = state_dict['model_state_dict']
            
        # Filter and load backbone weights
        backbone_keys = {k: v for k, v in state_dict.items() if k.startswith('backbone.')}
        missing, unexpected = self.backbone.load_state_dict(
            {k.replace('backbone.', ''): v for k, v in backbone_keys.items()}, 
            strict=False
        )
        print(f"Backbone loaded. Missing: {len(missing)}, Unexpected: {len(unexpected)}")
        
        # Filter and load concept head weights
        # In Probe, it's called 'head'. In CBM, it's 'concept_head'.
        head_keys = {k: v for k, v in state_dict.items() if k.startswith('head.')}
        self.concept_head.load_state_dict(
            {k.replace('head.', ''): v for k, v in head_keys.items()}, 
            strict=True
        )
        print("Concept head loaded.")

