"""
Geolocation metrics for Phase 2 evaluation.
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
    if isinstance(lat1, torch.Tensor):
        lat1_rad = torch.deg2rad(lat1)
        lng1_rad = torch.deg2rad(lng1)
        lat2_rad = torch.deg2rad(lat2)
        lng2_rad = torch.deg2rad(lng2)

        dlat = lat2_rad - lat1_rad
        dlng = lng2_rad - lng1_rad

        a = torch.sin(dlat / 2) ** 2 + torch.cos(lat1_rad) * torch.cos(lat2_rad) * torch.sin(dlng / 2) ** 2
        c = 2 * torch.arcsin(torch.sqrt(a))
        return 6371.0 * c

    lat1_rad = np.deg2rad(lat1)
    lng1_rad = np.deg2rad(lng1)
    lat2_rad = np.deg2rad(lat2)
    lng2_rad = np.deg2rad(lng2)

    dlat = lat2_rad - lat1_rad
    dlng = lng2_rad - lng1_rad

    a = np.sin(dlat / 2) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlng / 2) ** 2
    c = 2 * np.arcsin(np.sqrt(a))
    return 6371.0 * c


def threshold_accuracies_km(
    pred_lat: np.ndarray | torch.Tensor,
    pred_lng: np.ndarray | torch.Tensor,
    true_lat: np.ndarray | torch.Tensor,
    true_lng: np.ndarray | torch.Tensor,
    thresholds: list[float] | None = None,
) -> dict[str, float]:
    if thresholds is None:
        thresholds = [1.0, 10.0, 100.0, 1000.0, 2500.0]

    distances = haversine_km(pred_lat, pred_lng, true_lat, true_lng)
    if isinstance(distances, torch.Tensor):
        distances = distances.cpu().numpy()

    return {f"acc_{thr}km": float(np.mean(distances <= thr)) for thr in thresholds}


def cell_accuracy(
    pred_cells: np.ndarray | torch.Tensor,
    true_cells: np.ndarray | torch.Tensor,
) -> float:
    if isinstance(pred_cells, torch.Tensor):
        pred_cells = pred_cells.cpu().numpy()
    if isinstance(true_cells, torch.Tensor):
        true_cells = true_cells.cpu().numpy()
    return float(np.mean(pred_cells == true_cells))


def xyz_to_latlng(xyz: np.ndarray | torch.Tensor) -> tuple[np.ndarray | torch.Tensor, np.ndarray | torch.Tensor]:
    if isinstance(xyz, torch.Tensor):
        # Normalize to unit sphere to ensure arcsin is valid and accurate
        xyz = F.normalize(xyz, p=2, dim=-1)
        x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
        lat = torch.rad2deg(torch.arcsin(torch.clamp(z, -1.0, 1.0)))
        lng = torch.rad2deg(torch.atan2(y, x))
        return lat, lng

    # For numpy
    norm = np.linalg.norm(xyz, axis=1, keepdims=True)
    xyz = xyz / (norm + 1e-8)
    x, y, z = xyz[:, 0], xyz[:, 1], xyz[:, 2]
    lat = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
    lng = np.rad2deg(np.arctan2(y, x))
    return lat, lng


