# #!/usr/bin/env python3
# """
# StreetCLIP Inference Module for GeoGuessr Bot.

# Uses HuggingFace StreetCLIP model to predict geolocation by matching
# image features against a grid of coordinate text prompts.
# """

# import torch
# import torch.nn.functional as F
# from PIL import Image
# from transformers import CLIPModel, CLIPProcessor
# from typing import Tuple, Optional


# def format_coord(lat: float, lng: float) -> str:
#     """Format coordinates as text prompt for StreetCLIP."""
#     ns = "N" if lat >= 0 else "S"
#     ew = "E" if lng >= 0 else "W"
#     return f"{abs(lat):.1f}°{ns}, {abs(lng):.1f}°{ew}"


# def build_grid(step_deg: float = 5.0) -> list:
#     """Build a grid of coordinate text prompts."""
#     coords = []
#     lat_vals = torch.arange(-90, 90 + 1e-3, step_deg)
#     lng_vals = torch.arange(-180, 180 + 1e-3, step_deg)
#     for lat in lat_vals:
#         for lng in lng_vals:
#             coords.append((float(lat), float(lng), format_coord(float(lat), float(lng))))
#     return coords


# class StreetCLIPInference:
#     """StreetCLIP inference for geolocation prediction."""
    
#     def __init__(self, device: torch.device, grid_step: float = 5.0):
#         """
#         Initialize StreetCLIP model and build coordinate grid.
        
#         Args:
#             device: Torch device (cuda/cpu)
#             grid_step: Grid step size in degrees (default 5.0)
#         """
#         self.device = device
#         self.model = CLIPModel.from_pretrained("geolocal/StreetCLIP").to(device)
#         self.processor = CLIPProcessor.from_pretrained("geolocal/StreetCLIP")
#         self.model.eval()
        
#         # Build coordinate grid
#         coords = build_grid(grid_step)
#         self.coord_tensor = torch.tensor(
#             [(c[0], c[1]) for c in coords], 
#             dtype=torch.float32, 
#             device=device
#         )
        
#         # Precompute text features for all coordinate prompts
#         text_inputs = self.processor(
#             text=[c[2] for c in coords],
#             return_tensors="pt",
#             padding=True,
#             truncation=True,
#         )
#         text_inputs = {k: v.to(device) for k, v in text_inputs.items()}
        
#         with torch.no_grad():
#             text_feats = self.model.get_text_features(**text_inputs)
#             self.text_feats = F.normalize(text_feats, dim=-1)
        
#         print(f"✅ StreetCLIP initialized with {len(coords)} coordinate candidates (grid_step={grid_step}°)")
    
#     @torch.no_grad()
#     def predict(self, image: Image.Image) -> Tuple[float, float]:
#         """
#         Predict latitude and longitude for an image.
        
#         Args:
#             image: PIL Image (RGB)
            
#         Returns:
#             Tuple of (latitude, longitude) in degrees
#         """
#         # Ensure image is RGB
#         if image.mode != "RGB":
#             image = image.convert("RGB")
        
#         # Process image with explicit channel format to handle edge cases
#         image_inputs = self.processor.image_processor(
#             images=image, 
#             return_tensors="pt",
#             input_data_format="channels_last"  # PIL images are HWC format
#         )
#         image_inputs = {k: v.to(self.device) for k, v in image_inputs.items()}
        
#         # Get image features
#         image_feats = self.model.get_image_features(**image_inputs)
#         image_feats = F.normalize(image_feats, dim=-1)
        
#         # Compute similarity with all coordinate text prompts
#         logits = image_feats @ self.text_feats.T  # [1, num_coords]
        
#         # Get best matching coordinate
#         top_idx = logits.argmax(dim=1).item()
#         pred_lat, pred_lng = self.coord_tensor[top_idx].cpu().tolist()
        
#         return (pred_lat, pred_lng)
    
#     def get_checkpoint_info(self) -> dict:
#         """Return checkpoint information for logging."""
#         return {
#             "stage1_checkpoint": "vanilla_huggingface_streetclip",
#             "stage2_checkpoint": "vanilla_huggingface_streetclip"
#         }

# Use GeoCLIP instead of StreetCLIP
import torch
import tempfile
import os
from pathlib import Path
from typing import Tuple
from PIL import Image
from geoclip import GeoCLIP


class StreetCLIPInference:
    """
    GeoCLIP inference for geolocation prediction.
    
    Note: Class name kept as StreetCLIPInference for API compatibility.
    """
    
    def __init__(self, device: torch.device, grid_step: float = 5.0):
        """
        Initialize GeoCLIP model.
        
        Args:
            device: Torch device (cuda/cpu) - GeoCLIP handles device internally
            grid_step: Not used for GeoCLIP, kept for API compatibility
        """
        self.device = device
        self.model = GeoCLIP()
        if torch.cuda.is_available() and device.type == "cuda":
            self.model = self.model.to(device)
        
        print(f"✅ GeoCLIP initialized (replacing StreetCLIP)")
    
    @torch.no_grad()
    def predict(self, image: Image.Image) -> Tuple[float, float]:
        """
        Predict latitude and longitude for an image.
        
        Args:
            image: PIL Image (RGB)
            
        Returns:
            Tuple of (latitude, longitude) in degrees
        """
        # Ensure image is RGB
        if image.mode != "RGB":
            image = image.convert("RGB")
        
        # Resize small images to avoid CLIP processor channel ambiguity issues
        # CLIP expects at least 224x224, so resize if smaller
        min_size = 224
        if image.width < min_size or image.height < min_size:
            # Resize to at least min_size while maintaining aspect ratio
            scale = max(min_size / image.width, min_size / image.height)
            new_width = max(min_size, int(image.width * scale))
            new_height = max(min_size, int(image.height * scale))
            image = image.resize((new_width, new_height), Image.Resampling.LANCZOS)
        
        # GeoCLIP expects an image path, so save to temporary file
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp_file:
            tmp_path = tmp_file.name
            image.save(tmp_path, 'PNG')
        
        try:
            # Get top prediction (top_k=1 for single best prediction)
            top_pred_gps, top_pred_prob = self.model.predict(tmp_path, top_k=1)
            
            # Extract lat, lng from first prediction
            if len(top_pred_gps) > 0:
                lat, lng = top_pred_gps[0]
                return (float(lat), float(lng))
            else:
                # Fallback: return (0, 0) if no prediction
                return (0.0, 0.0)
        finally:
            # Clean up temporary file
            if os.path.exists(tmp_path):
                os.unlink(tmp_path)
    
    def get_checkpoint_info(self) -> dict:
        """Return checkpoint information for logging."""
        return {
            "stage1_checkpoint": "vanilla_geoclip",
            "stage2_checkpoint": "vanilla_geoclip"
        }

