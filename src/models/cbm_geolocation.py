"""
Concept Bottleneck Model for StreetCLIP-based geolocation.
"""

from __future__ import annotations

from typing import Dict, Iterable, Optional

import torch
from torch import nn
import torch.nn.functional as F


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
    ):
        super().__init__()
        self.encoder = encoder
        self.feature_dim = feature_dim
        self.coordinate_loss_type = coordinate_loss_type.lower()
        if self.coordinate_loss_type not in {"mse", "sphere", "haversine"}:
            raise ValueError(
                f"Unsupported coordinate_loss_type '{coordinate_loss_type}'. "
                "Expected 'mse', 'sphere', or 'haversine'."
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

        # Define concept, country, and coordinate heads
        self.concept_layer = nn.Sequential(
            nn.Linear(feature_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, num_concepts),
        )
        self.country_head = nn.Sequential(
                nn.Linear(num_concepts, 128),
                nn.LayerNorm(128),
                nn.GELU(),
                nn.Dropout(0.1),
                nn.Linear(128, num_countries),
        )

        coord_in_dim = num_concepts
        self.feature_skip = None
        if coordinate_feature_skip_dim is not None and coordinate_feature_skip_dim > 0:
            self.feature_skip = nn.Sequential(
                nn.Linear(feature_dim, coordinate_feature_skip_dim),
                nn.LayerNorm(coordinate_feature_skip_dim),
                nn.GELU(),
            )
            coord_in_dim += coordinate_feature_skip_dim

        coord_out_dim = 3 if self.coordinate_loss_type == "sphere" else 2
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

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        concept_logits = self.concept_layer(features)
        concept_probs = F.softmax(concept_logits, dim=1)
        country_input = concept_probs
        coord_input = concept_probs if self.coordinate_input == "probs" else concept_logits

        if self.detach_concepts_for_prediction:
            coord_input = coord_input.detach()

        if self.feature_skip is not None:
            projected_features = self.feature_skip(features)
            coord_input = torch.cat([coord_input, projected_features], dim=1)

        country_logits = self.country_head(country_input)
        coord_logits = self.coordinate_head(coord_input)
        if self.coordinate_loss_type == "sphere":
            coordinates = F.normalize(coord_logits, p=2, dim=1)
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
        return concept_logits, country_logits, coordinates

    def coordinate_parameters(self) -> Iterable[nn.Parameter]:
        params = list(self.coordinate_head.parameters())
        if self.feature_skip is not None:
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
            params = list(self.concept_layer.parameters()) + list(
                p for p in self.encoder.parameters() if p.requires_grad
            )
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
            else:
                for param in modules.parameters():
                    param.requires_grad = value

        if stage == "concept":
            _set_requires_grad(self.concept_layer, True)
            _set_requires_grad(self.country_head, train_country_head)
            _set_requires_grad(self.coordinate_head, train_prediction_head)
            if self.feature_skip is not None:
                _set_requires_grad(self.feature_skip, train_prediction_head)
            return

        if stage == "prediction":
            _set_requires_grad(self.concept_layer, False)
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




