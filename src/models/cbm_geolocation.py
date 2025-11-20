"""
Concept Bottleneck Model for StreetCLIP-based geolocation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional, Tuple

import torch
from torch import nn
import torch.nn.functional as F


from src.models.encoder_factory import HierarchicalRouter, CoarseRouterConfig

class CBMGeolocationModel(nn.Module):
    """Concept bottleneck model with StreetCLIP encoder."""

    def __init__(
        self,
        encoder: nn.Module,
        num_concepts: int,
        num_countries: int,
        feature_dim: int = 768,
        coordinate_loss_type: str = "mse",
        coordinate_input: str = "probs",
        coordinate_feature_skip_dim: Optional[int] = 256,
        detach_concepts_for_prediction: bool = True,
        coordinate_residual_center: Optional[torch.Tensor] = None,
        coordinate_residual_bounds: Optional[torch.Tensor] = None,
        # New arguments for alignment
        location_encoder: Optional[nn.Module] = None,
        concept_text_features: Optional[torch.Tensor] = None,
        num_mixtures: int = 5,
        # Hierarchical routing config
        use_hierarchical_routing: bool = False,
        num_coarse_cells: int = 64,
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.location_encoder = location_encoder
        self.coordinate_loss_type = coordinate_loss_type.lower()
        self.num_mixtures = num_mixtures
        self.use_hierarchical_routing = use_hierarchical_routing
        if self.coordinate_loss_type not in {"mse", "sphere", "haversine", "vmf"}:
            raise ValueError(
                f"Unsupported coordinate_loss_type '{coordinate_loss_type}'. "
                "Expected 'mse', 'sphere', 'haversine', or 'vmf'."
            )

        if coordinate_input not in {"probs", "logits"}:
            raise ValueError("coordinate_input must be either 'probs' or 'logits'")

        self.coordinate_input = coordinate_input
        self.detach_concepts_for_prediction = detach_concepts_for_prediction

        # Register coordinate residual center and bounds as buffers
        self.register_buffer(
            "coordinate_residual_center",
            coordinate_residual_center.view(1, -1)
            if coordinate_residual_center is not None
            else None,
        )
        self.register_buffer(
            "coordinate_residual_bounds",
            coordinate_residual_bounds.view(1, -1)
            if coordinate_residual_bounds is not None
            else None,
        )

        # --- Concept Layer ---
        # If concept_text_features provided, use Concept-Aware Alignment architecture
        if concept_text_features is not None:
            # Get text feature dimension from concept bank
            text_feature_dim = concept_text_features.shape[1]
            
            # Register concept bank as a parameter (can be frozen or finetuned)
            self.register_parameter(
                "concept_bank", 
                nn.Parameter(concept_text_features.clone(), requires_grad=True)
            )
            
            # Adapter to project image features to text feature space
            # Image encoder outputs feature_dim (e.g., 1024), text encoder outputs text_feature_dim (e.g., 768)
            self.concept_adapter = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, text_feature_dim)  # Project to concept bank space (text_dim)
            )
            self.text_feature_dim = text_feature_dim
            self.use_concept_bank = True
        else:
            # Legacy MLP concept layer
            self.concept_layer = nn.Sequential(
                nn.Linear(feature_dim, 256),
                nn.LayerNorm(256),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(256, num_concepts),
            )
            self.use_concept_bank = False
            self.text_feature_dim = None

        # --- Location Adapter (if location encoder present) ---
        if self.location_encoder is not None:
            # Adapter to project location features to concept space
            # Location encoder outputs feature_dim (e.g., 1024)
            # If using concept bank, project to text_feature_dim (e.g., 768), otherwise keep feature_dim
            target_dim = self.text_feature_dim if self.use_concept_bank else feature_dim
            self.location_adapter = nn.Sequential(
                nn.Linear(feature_dim, feature_dim),
                nn.LayerNorm(feature_dim),
                nn.GELU(),
                nn.Linear(feature_dim, target_dim)  # Project to concept bank space or keep feature_dim
            )

        # --- Prediction Heads Setup (needed for hierarchical routing) ---
        coord_in_dim = num_concepts
        self.feature_skip = None
        if coordinate_feature_skip_dim is not None and coordinate_feature_skip_dim > 0:
            self.feature_skip = nn.Sequential(
                nn.Linear(feature_dim, coordinate_feature_skip_dim),
                nn.LayerNorm(coordinate_feature_skip_dim),
                nn.GELU(),
            )
            coord_in_dim += coordinate_feature_skip_dim

        if self.coordinate_loss_type == "sphere":
            coord_out_dim = 3
        elif self.coordinate_loss_type == "vmf":
            # 5 params per mixture: mu(3), kappa(1), pi(1)
            coord_out_dim = self.num_mixtures * 5
        else:
            coord_out_dim = 2

        # --- Hierarchical Routing (Optional) ---
        self.hierarchical_router = None
        self.fine_grained_heads = None

        if self.use_hierarchical_routing:
            # Router config
            router_config = CoarseRouterConfig(
                num_coarse_cells=num_coarse_cells,
                hidden_dim=512,
            )

            # Router takes concept features
            # If using concept bank, input dim is text_feature_dim. Otherwise num_concepts
            router_input_dim = self.text_feature_dim if self.use_concept_bank else num_concepts

            self.hierarchical_router = HierarchicalRouter(router_config, router_input_dim)

            # Fine-grained heads: One per coarse cell
            # Reuse coordinate head architecture but one per cell
            self.fine_grained_heads = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(coord_in_dim, 256),
                    nn.LayerNorm(256),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(256, 128),
                    nn.GELU(),
                    nn.Dropout(0.1),
                    nn.Linear(128, coord_out_dim),
                ) for _ in range(num_coarse_cells)
            ])

        # --- Prediction Heads ---
        self.country_head = nn.Sequential(
                nn.Linear(num_concepts, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_countries),
        )

        self.coordinate_head = nn.Sequential(
            nn.Linear(coord_in_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(128, coord_out_dim),
        )

    def forward(
        self, 
        images: torch.Tensor, 
        target_coords: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Returns:
            concept_logits, country_logits, coordinates, 
            image_features (adapted), location_features (adapted), location_concept_logits
        """
        # 1. Image Encoding
        raw_features = self.encoder(images)
        
        # 2. Concept Prediction
        if self.use_concept_bank:
            # Adapt image features
            image_features = self.concept_adapter(raw_features)
            # Compute logits via dot product with concept bank
            # image_features: [B, D], concept_bank: [C, D] -> [B, C]
            concept_logits = image_features @ self.concept_bank.T
            # Scale? CLIP uses scaling, but here we might rely on adapter
            # Adding a learnable scale or fixed scale is common. 
            # For now, let linear layers handle magnitude.
        else:
            image_features = raw_features
            concept_logits = self.concept_layer(raw_features)
            
        concept_probs = F.softmax(concept_logits, dim=1)
        
        # 3. Location Processing (Training only)
        location_features = None
        location_concept_logits = None
        
        if self.location_encoder is not None and target_coords is not None:
            # Encode locations
            # Check for NaNs in coords
            mask = ~torch.isnan(target_coords).any(dim=1)
            if mask.any():
                # We only compute for valid coords, but to keep shapes consistent
                # we might need to handle full batch.
                # If we have NaNs, we can just zero them out or skip loss later.
                # For simplicity, pass all, assume loss function handles masking.
                # Replace NaNs with 0 for forward pass (gradient will be masked in loss)
                safe_coords = torch.nan_to_num(target_coords, nan=0.0)
                
                raw_loc_features = self.location_encoder(safe_coords)
                location_features = self.location_adapter(raw_loc_features)
                
                # Location -> Concept logits
                if self.use_concept_bank:
                    location_concept_logits = location_features @ self.concept_bank.T
        
        # 4. Downstream Predictions (Country, Coords)
        country_input = concept_probs
        coord_input = concept_probs if self.coordinate_input == "probs" else concept_logits

        if self.detach_concepts_for_prediction:
            coord_input = coord_input.detach()

        if self.feature_skip is not None:
            projected_features = self.feature_skip(raw_features)
            coord_input = torch.cat([coord_input, projected_features], dim=1)

        country_logits = self.country_head(country_input)
        
        # Coordinate prediction
        if self.use_hierarchical_routing:
             # 1. Predict coarse cell logits
             # Input to router: we can use concept features or logits.
             # Using concept_logits or adapt features if available
             router_input = image_features if self.use_concept_bank else concept_logits
             coarse_logits = self.hierarchical_router(router_input) # [B, num_cells]
             
             # 2. Select best cell (hard routing for inference, soft for training?)
             # For simplicity: Soft routing (mixture of experts style) or just top-1
             # Let's do weighted sum of heads based on router probabilities (MoE style)
             router_probs = F.softmax(coarse_logits, dim=1) # [B, num_cells]
             
             # Run all heads? Expensive if num_cells is large (e.g. 64).
             # Optimization: Run only top-k heads or batched matrix mult.
             # Since heads are small MLPs, we can stack weights?
             # For 64 cells, running loop is slow.
             # Let's use top-1 for now for simplicity, or weighted avg of all.
             
             # Better approach for efficiency: 
             # Compute output for ALL heads in parallel using grouped conv or batched linear?
             # For this MVP, let's iterate. 64 is small enough.
             
             # But we need gradients for router.
             # Weighted sum:
             # coord_out = sum(prob_i * head_i(input))
             
             # To make it efficient:
             # Batch process? 
             # [B, D] -> [B, 1, D]
             # heads weights: [num_cells, D, H]
             # This suggests implementing fine heads as a single BatchedLinear layer.
             # But we used nn.Sequential.
             
             # Fallback: Loop (slow but correct)
             head_outputs = []
             for head in self.fine_grained_heads:
                 head_outputs.append(head(coord_input))
             
             # Stack: [B, num_cells, out_dim]
             head_outputs = torch.stack(head_outputs, dim=1)
             
             # Weight by router probs: [B, num_cells, 1]
             weighted_out = head_outputs * router_probs.unsqueeze(-1)
             
             # Sum: [B, out_dim]
             coord_logits = weighted_out.sum(dim=1)
             
        else:
            coord_logits = self.coordinate_head(coord_input)
        
        if self.coordinate_loss_type == "sphere":
            coordinates = F.normalize(coord_logits, p=2, dim=1)
        elif self.coordinate_loss_type == "vmf":
            coordinates = coord_logits
        else:
            coordinate_delta = torch.tanh(coord_logits)
            if (
                self.coordinate_residual_center is not None
                and self.coordinate_residual_bounds is not None
            ):
                coordinates = torch.clamp(
                    self.coordinate_residual_center
                    + coordinate_delta * self.coordinate_residual_bounds,
                    -1.0,
                    1.0,
                )
            else:
                coordinates = coordinate_delta
                
        return (
            concept_logits, 
            country_logits, 
            coordinates, 
            image_features if self.use_concept_bank else None,
            location_features,
            location_concept_logits
        )

    def coordinate_parameters(self) -> Iterable[nn.Parameter]:
        params = list(self.coordinate_head.parameters())
        if self.feature_skip:
            params += list(self.feature_skip.parameters())
        return params

    def parameters_for_stage(
        self,
        stage: str,
        train_prediction_head: bool = False,
        train_country_head: bool = False,
    ) -> Iterable[nn.Parameter]:
        """Return parameters to optimize for the given stage."""
        stage = stage.lower()
        if stage == "concept":
            # Concept stage now involves alignment
            params = []
            if self.use_concept_bank:
                params += list(self.concept_adapter.parameters())
                params += [self.concept_bank]
            else:
                params += list(self.concept_layer.parameters())
            
            params += list(p for p in self.encoder.parameters() if p.requires_grad)
            
            if self.location_encoder is not None:
                params += list(self.location_encoder.parameters())
                params += list(self.location_adapter.parameters())

            if train_country_head:
                params += list(self.country_head.parameters())
            if train_prediction_head:
                params += self.coordinate_parameters()
            return params
            
        if stage == "prediction":
            return list(self.country_head.parameters()) + self.coordinate_parameters()
        if stage == "finetune":
            return self.parameters()
        raise ValueError(f"Unknown stage {stage}")

    def set_stage(
        self,
        stage: str,
        finetune_encoder: bool = False,
        train_prediction_head: bool = False,
        train_country_head: bool = False,
    ):
        stage = stage.lower()

        # Reset grads
        for param in self.encoder.parameters():
            param.requires_grad = finetune_encoder

        def _set_requires_grad(modules, value: bool):
            if modules is None:
                return
            if isinstance(modules, (list, tuple)):
                for module in modules:
                    _set_requires_grad(module, value)
            elif isinstance(modules, nn.Parameter):
                modules.requires_grad = value
            else:
                for param in modules.parameters():
                    param.requires_grad = value

        if stage == "concept":
            if self.use_concept_bank:
                _set_requires_grad(self.concept_adapter, True)
                _set_requires_grad(self.concept_bank, True) # Allow fine-tuning concepts
            else:
                _set_requires_grad(self.concept_layer, True)
                
            _set_requires_grad(self.country_head, train_country_head)
            _set_requires_grad(self.coordinate_head, train_prediction_head)
            if self.feature_skip is not None:
                _set_requires_grad(self.feature_skip, train_prediction_head)
                
            # Enable location branch
            if self.location_encoder is not None:
                _set_requires_grad(self.location_encoder, True)
                _set_requires_grad(self.location_adapter, True)
            return

        if stage == "prediction":
            if self.use_concept_bank:
                _set_requires_grad(self.concept_adapter, False)
                _set_requires_grad(self.concept_bank, False)
            else:
                _set_requires_grad(self.concept_layer, False)
                
            # Disable location branch for prediction stage (only needed for concept/alignment training)
            if self.location_encoder is not None:
                _set_requires_grad(self.location_encoder, False)
                _set_requires_grad(self.location_adapter, False)

            _set_requires_grad(self.country_head, True)
            _set_requires_grad(self.coordinate_head, True)
            if self.feature_skip is not None:
                _set_requires_grad(self.feature_skip, True)
            return

        if stage == "finetune":
            _set_requires_grad(self, True)
            return

        raise ValueError(f"Unknown stage {stage}")

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True
