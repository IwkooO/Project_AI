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

        self.concept_layer = nn.Linear(feature_dim, num_concepts)
        self.country_head = nn.Linear(num_concepts, num_countries)

        coord_out_dim = 3 if self.coordinate_loss_type == "sphere" else 2
        self.coordinate_head = nn.Linear(num_concepts, coord_out_dim)

    def forward(self, images: torch.Tensor):
        features = self.encoder(images)
        concept_logits = self.concept_layer(features)
        concept_probs = F.softmax(concept_logits, dim=1)
        country_logits = self.country_head(concept_probs)
        coord_logits = self.coordinate_head(concept_probs)
        if self.coordinate_loss_type == "sphere":
            coordinates = F.normalize(coord_logits, p=2, dim=1)
        else:
            coordinates = torch.tanh(coord_logits)
        return concept_logits, country_logits, coordinates

    def parameters_for_stage(self, stage: str) -> Iterable[nn.Parameter]:
        """Return parameters to optimize for the given stage."""
        stage = stage.lower()
        if stage == "concept":
            return list(self.concept_layer.parameters()) + list(
                p for p in self.encoder.parameters() if p.requires_grad
            )
        if stage == "prediction":
            return list(self.country_head.parameters()) + list(self.coordinate_head.parameters())
        if stage == "finetune":
            return self.parameters()
        raise ValueError(f"Unknown stage {stage}")

    def set_stage(self, stage: str, finetune_encoder: bool = False):
        stage = stage.lower()

        # Reset grads
        for param in self.encoder.parameters():
            param.requires_grad = finetune_encoder

        if stage == "concept":
            for param in self.concept_layer.parameters():
                param.requires_grad = True
            for param in list(self.country_head.parameters()) + list(self.coordinate_head.parameters()):
                param.requires_grad = False
            return

        if stage == "prediction":
            for param in self.concept_layer.parameters():
                param.requires_grad = False
            for param in list(self.country_head.parameters()) + list(self.coordinate_head.parameters()):
                param.requires_grad = True
            return

        if stage == "finetune":
            for param in self.parameters():
                param.requires_grad = True
            return

        raise ValueError(f"Unknown stage {stage}")

    def freeze_all(self):
        for param in self.parameters():
            param.requires_grad = False

    def unfreeze_all(self):
        for param in self.parameters():
            param.requires_grad = True




