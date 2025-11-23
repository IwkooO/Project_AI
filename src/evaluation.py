"""Evaluation utilities for StreetCLIP CBM geolocation."""

from __future__ import annotations

import math
from typing import Dict

import torch
import torch.nn.functional as F

EARTH_RADIUS_KM = 6371.0


def denormalize_coordinates(coords: torch.Tensor) -> torch.Tensor:
    """Convert normalized coordinates back to degrees.
    
    Args:
        coords: Normalized coordinates, shape (2,) for single pair or (N, 2) for batch.
                Can be torch.Tensor or numpy array.
    
    Returns:
        Denormalized coordinates in degrees, shape (2,) or (N, 2).
    """
    # Convert numpy to torch if needed
    if not isinstance(coords, torch.Tensor):
        coords = torch.from_numpy(coords) if hasattr(coords, '__array__') else torch.tensor(coords)
    
    # Handle 1D input (single coordinate pair)
    if coords.dim() == 1:
        lat = coords[0] * 90.0
        lng = coords[1] * 180.0
        return torch.stack([lat, lng])
    
    # Handle 2D input (batch)
    lat = coords[:, 0] * 90.0
    lng = coords[:, 1] * 180.0
    return torch.stack([lat, lng], dim=1)


def latlng_to_sphere(lat_lng_deg: torch.Tensor) -> torch.Tensor:
    """Convert latitude/longitude in degrees to 3D unit vectors."""
    lat_rad = torch.deg2rad(lat_lng_deg[:, 0])
    lng_rad = torch.deg2rad(lat_lng_deg[:, 1])
    cos_lat = torch.cos(lat_rad)
    x = cos_lat * torch.cos(lng_rad)
    y = cos_lat * torch.sin(lng_rad)
    z = torch.sin(lat_rad)
    xyz = torch.stack([x, y, z], dim=1)
    return F.normalize(xyz, p=2, dim=1)


def normalized_latlng_to_sphere(coords: torch.Tensor) -> torch.Tensor:
    """Convert normalized lat/lng in [-1, 1] to 3D unit vectors."""
    lat_lng_deg = denormalize_coordinates(coords)
    return latlng_to_sphere(lat_lng_deg)


def sphere_to_latlng(sphere_vecs: torch.Tensor) -> torch.Tensor:
    """Convert 3D unit vectors to latitude/longitude in degrees."""
    sphere_vecs = F.normalize(sphere_vecs, p=2, dim=1)
    x, y, z = sphere_vecs[:, 0], sphere_vecs[:, 1], sphere_vecs[:, 2]
    z = torch.clamp(z, -1.0, 1.0)
    lat = torch.rad2deg(torch.asin(z))
    lng = torch.rad2deg(torch.atan2(y, x))
    return torch.stack([lat, lng], dim=1)


def sphere_to_normalized_latlng(sphere_vecs: torch.Tensor) -> torch.Tensor:
    """Convert 3D unit vectors to normalized lat/lng coordinates."""
    lat_lng_deg = sphere_to_latlng(sphere_vecs)
    lat_norm = lat_lng_deg[:, 0] / 90.0
    lng_norm = lat_lng_deg[:, 1] / 180.0
    return torch.stack([lat_norm, lng_norm], dim=1)


def haversine_distance(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    mask = ~torch.isnan(true_coords).any(dim=1)
    if mask.sum() == 0:
        return torch.zeros(0, device=pred_coords.device)

    # Check if coordinates seem normalized (max value <= 1.0001)
    # This is a heuristic to determine if we need to denormalize
    is_normalized = pred_coords[mask].abs().max() <= 1.0001 and true_coords[mask].abs().max() <= 1.0001
    
    if is_normalized:
        pred = denormalize_coordinates(pred_coords[mask])
        true = denormalize_coordinates(true_coords[mask])
    else:
        pred = pred_coords[mask]
        true = true_coords[mask]

    lat1 = torch.deg2rad(true[:, 0])
    lon1 = torch.deg2rad(true[:, 1])
    lat2 = torch.deg2rad(pred[:, 0])
    lon2 = torch.deg2rad(pred[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    # Clamp a to [0, 1] to avoid numerical instability in sqrt/asin
    a = torch.clamp(a, min=0.0, max=1.0)
    c = 2 * torch.asin(torch.sqrt(a))
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
    coordinate_loss_type: str = "mse",
) -> Dict[str, float]:
    metrics: Dict[str, float] = {}
    coord_type = coordinate_loss_type.lower()

    if coord_type == "sphere":
        predicted_coords_for_metrics = sphere_to_normalized_latlng(predicted_coords)
    elif coord_type in {"mse", "haversine"}:
        predicted_coords_for_metrics = predicted_coords
    else:
        raise ValueError(f"Unsupported coordinate_loss_type '{coordinate_loss_type}'")

    with torch.no_grad():
        metrics["concept_accuracy"] = float(
            (concept_logits.argmax(dim=1) == concept_targets).float().mean().item()
        )
        metrics["country_accuracy"] = float(
            (country_logits.argmax(dim=1) == country_targets).float().mean().item()
        )

        mask = ~torch.isnan(coordinate_targets).any(dim=1)
        if mask.sum() > 0:
            mse = torch.mean(
                (predicted_coords_for_metrics[mask] - coordinate_targets[mask]) ** 2
            )
            mae = torch.mean(
                torch.abs(predicted_coords_for_metrics[mask] - coordinate_targets[mask])
            )
            metrics["coord_mse"] = float(mse.item())
            metrics["coord_mae"] = float(mae.item())

            distances = haversine_distance(
                predicted_coords_for_metrics, coordinate_targets
            )
            if distances.numel() > 0:
                metrics["median_km"] = float(distances.median().item())
                metrics["mean_km"] = float(distances.mean().item())
                metrics["p90_km"] = float(torch.quantile(distances, 0.9).item())
                for threshold in [1, 10, 100, 1000]:
                    metrics[f"acc@{threshold}km"] = accuracy_within_threshold(distances, threshold)

            # Check if normalized to correctly denormalize for bias metrics
            is_normalized = predicted_coords_for_metrics[mask].abs().max() <= 1.0001 and coordinate_targets[mask].abs().max() <= 1.0001
            
            if is_normalized:
                pred_deg = denormalize_coordinates(predicted_coords_for_metrics[mask])
                true_deg = denormalize_coordinates(coordinate_targets[mask])
            else:
                pred_deg = predicted_coords_for_metrics[mask]
                true_deg = coordinate_targets[mask]

            lat_bias = pred_deg[:, 0] - true_deg[:, 0]
            lng_bias = pred_deg[:, 1] - true_deg[:, 1]
            metrics["lat_bias_deg"] = float(lat_bias.mean().item())
            metrics["lng_bias_deg"] = float(lng_bias.mean().item())
            metrics["lat_std_deg"] = float(pred_deg[:, 0].std(unbiased=False).item())
            metrics["lng_std_deg"] = float(pred_deg[:, 1].std(unbiased=False).item())

            centroid = coordinate_targets[mask].mean(dim=0, keepdim=True)
            centroid_preds = centroid.expand_as(coordinate_targets[mask])
            centroid_distances = haversine_distance(
                centroid_preds, coordinate_targets[mask]
            )
            if centroid_distances.numel() > 0:
                metrics["centroid_median_km"] = float(centroid_distances.median().item())
                metrics["centroid_mean_km"] = float(centroid_distances.mean().item())
        else:
            metrics["coord_mse"] = math.nan
            metrics["coord_mae"] = math.nan
            metrics["median_km"] = math.nan

    return metrics

def compute_haversine_distance(pred_coords: torch.Tensor, true_coords: torch.Tensor) -> torch.Tensor:
    """Compute Haversine distance between two points on the Earth's surface.
    
    Args:
        pred_coords: Predicted coordinates, shape (2,) for single pair or (N, 2) for batch.
        true_coords: True coordinates, shape (2,) for single pair or (N, 2) for batch.
    
    Returns:
        Distance in kilometers.
    """
    def _to_tensor(coords):
        if isinstance(coords, torch.Tensor):
            tensor = coords
        elif hasattr(coords, "__array__"):
            tensor = torch.from_numpy(coords)
        else:
            tensor = torch.tensor(coords)
        return tensor.to(torch.float32)

    def _prepare_coords(coords_tensor: torch.Tensor):
        was_1d = coords_tensor.dim() == 1
        tensor = coords_tensor if not was_1d else coords_tensor.unsqueeze(0)
        needs_denorm = tensor.abs().max() <= 1.0001
        tensor = denormalize_coordinates(tensor) if needs_denorm else tensor
        return tensor, was_1d

    pred_tensor = _to_tensor(pred_coords)
    true_tensor = _to_tensor(true_coords)

    pred_deg, pred_was_1d = _prepare_coords(pred_tensor)
    true_deg, true_was_1d = _prepare_coords(true_tensor)

    lat1 = torch.deg2rad(true_deg[:, 0])
    lon1 = torch.deg2rad(true_deg[:, 1])
    lat2 = torch.deg2rad(pred_deg[:, 0])
    lon2 = torch.deg2rad(pred_deg[:, 1])

    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1) * torch.cos(lat2) * torch.sin(dlon / 2) ** 2
    c = 2 * torch.atan2(torch.sqrt(a), torch.sqrt(1 - a))
    distances = EARTH_RADIUS_KM * c

    if pred_was_1d and true_was_1d:
        return distances.squeeze(0)
    return distances



