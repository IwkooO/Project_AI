"""Evaluation utilities for StreetCLIP CBM geolocation."""

from __future__ import annotations

import math
from typing import Dict

import torch

EARTH_RADIUS_KM = 6371.0


def denormalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Convert normalized coordinates back to degrees."""
    lat = coords[:, 0] * 90.0
    lng = coords[:, 1] * 180.0
    return torch.stack([lat, lng], dim=1)


def haversine_distance(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(true_coords).any(dim=1)
    if mask.sum() == 0:
        return torch.zeros(0, device=pred_coords.device)

    pred = denormalize_coordinates(pred_coords[mask])
    true = denormalize_coordinates(true_coords[mask])

    lat1 = torch.deg2rad(true[:, 0])
    lon1 = torch.deg2rad(true[:, 1])
    lat2 = torch.deg2rad(pred[:, 0])
    lon2 = torch.deg2rad(pred[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distances = EARTH_RADIUS_KM * c
    return distances


def accuracy_within_threshold(distances: torch.Tensor, threshold_km: float) -> float:
    if distances.numel() == 0:
        return 0.0
    return float((distances <= threshold_km).float().mean().item())


def compute_geolocation_metrics(
    concept_logits: torch.Tensor,
    country_logits: torch.Tensor,
    predicted_coords: torch.Tensor,
    concept_targets: torch.Tensor,
    country_targets: torch.Tensor,
    coordinate_targets: torch.Tensor,
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}

    with torch.no_grad():
        metrics["concept_accuracy"] = float(
            (concept_logits.argmax(dim=1) == concept_targets).float().mean().item()
        )
        metrics["country_accuracy"] = float(
            (country_logits.argmax(dim=1) == country_targets).float().mean().item()
        )

        mask = ~torch.isnan(coordinate_targets).any(dim=1)
        if mask.sum() > 0:
            mse = torch.mean((predicted_coords[mask] - coordinate_targets[mask]) ** 2)
            mae = torch.mean(torch.abs(predicted_coords[mask] - coordinate_targets[mask]))
            metrics["coord_mse"] = float(mse.item())
            metrics["coord_mae"] = float(mae.item())

            distances = haversine_distance(predicted_coords, coordinate_targets)
            if distances.numel() > 0:
                metrics["median_km"] = float(distances.median().item())
                for threshold in [1, 10, 100, 1000]:
                    metrics[f"acc@{threshold}km"] = accuracy_within_threshold(distances, threshold)
        else:
            metrics["coord_mse"] = math.nan
            metrics["coord_mae"] = math.nan
            metrics["median_km"] = math.nan

    return metrics



