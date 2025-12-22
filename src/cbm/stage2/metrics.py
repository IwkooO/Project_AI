"""
Geolocation metrics for Stage 2 evaluation.

Metrics:
- Haversine distance (km)
- Threshold accuracies (1km, 10km, 100km, 1000km, 2500km)
- Cell accuracy
"""

from __future__ import annotations

import numpy as np
import torch


def haversine_km(
    lat1: np.ndarray | torch.Tensor,
    lng1: np.ndarray | torch.Tensor,
    lat2: np.ndarray | torch.Tensor,
    lng2: np.ndarray | torch.Tensor,
) -> np.ndarray | torch.Tensor:
    """
    Compute Haversine distance between two sets of coordinates.
    
    Args:
        lat1, lng1: [N] first set of coordinates
        lat2, lng2: [N] second set of coordinates
    
    Returns:
        distances: [N] distances in kilometers
    """
    # Convert to radians
    if isinstance(lat1, torch.Tensor):
        lat1_rad = torch.deg2rad(lat1)
        lng1_rad = torch.deg2rad(lng1)
        lat2_rad = torch.deg2rad(lat2)
        lng2_rad = torch.deg2rad(lng2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlng / 2) ** 2
        c = 2 * torch.arcsin(torch.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        distances = R * c
        
        return distances
    else:
        lat1_rad = np.deg2rad(lat1)
        lng1_rad = np.deg2rad(lng1)
        lat2_rad = np.deg2rad(lat2)
        lng2_rad = np.deg2rad(lng2)
        
        # Haversine formula
        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad
        
        a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in km
        R = 6371.0
        distances = R * c
        
        return distances


def threshold_accuracies_km(
    pred_lat: np.ndarray | torch.Tensor,
    pred_lng: np.ndarray | torch.Tensor,
    true_lat: np.ndarray | torch.Tensor,
    true_lng: np.ndarray | torch.Tensor,
    thresholds: list[float] = [1.0, 10.0, 100.0, 1000.0, 2500.0],
) -> dict[str, float]:
    """
    Compute accuracy at various distance thresholds.
    
    Args:
        pred_lat, pred_lng: [N] predicted coordinates
        true_lat, true_lng: [N] ground truth coordinates
        thresholds: List of distance thresholds in km
    
    Returns:
        dict mapping threshold names to accuracies (0-1)
    """
    distances = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
    
    if isinstance(distances, torch.Tensor):
        distances = distances.cpu().numpy()
    
    results = {}
    for threshold in thresholds:
        acc = np.mean(distances <= threshold)
        results[f"acc_{threshold}km"] = float(acc)
    
    return results


def cell_accuracy(
    pred_cells: np.ndarray | torch.Tensor,
    true_cells: np.ndarray | torch.Tensor,
) -> float:
    """
    Compute cell classification accuracy.
    
    Args:
        pred_cells: [N] predicted cell indices
        true_cells: [N] ground truth cell indices
    
    Returns:
        accuracy: 0-1
    """
    if isinstance(pred_cells, torch.Tensor):
        pred_cells = pred_cells.cpu().numpy()
    if isinstance(true_cells, torch.Tensor):
        true_cells = true_cells.cpu().numpy()
    
    return float(np.mean(pred_cells == true_cells))


def xyz_to_latlng(xyz: np.ndarray | torch.Tensor) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    """
    Convert 3D Cartesian coordinates to lat/lng.
    
    Args:
        xyz: [N, 3] (x, y, z) coordinates on unit sphere
    
    Returns:
        (lat, lng): [N] each, in degrees
    """
    if isinstance(xyz, torch.Tensor):
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        lat = torch.rad2deg(torch.arcsin(torch.clamp(z, -1.0, 1.0)))
        lng = torch.rad2deg(torch.atan2(y, x))
        return lat, lng
    else:
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
        lng = np.rad2deg(np.arctan2(y, x))
        return lat, lng
