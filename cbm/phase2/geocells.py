"""
Semantic geocell fitting and assignment.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans


def latlng_to_xyz(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat)
    lng_rad = np.deg2rad(lng)
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def fit_semantic_geocells(
    train_coords: np.ndarray,  # [N_train, 2] (lat, lng)
    train_countries: Optional[np.ndarray] = None,  # [N_train] country labels
    num_cells: int = 1000,
    per_country: bool = False,
    random_state: int = 42,
) -> tuple[np.ndarray, dict]:
    train_xyz = latlng_to_xyz(train_coords[:, 0], train_coords[:, 1])  # [N_train, 3]

    if per_country and train_countries is not None:
        unique_countries = np.unique(train_countries)
        unique_countries = unique_countries[~pd.isna(unique_countries)]

        all_centers = []
        cell_to_country = {}
        num_cells_per_country = {}
        cell_idx = 0

        for country in unique_countries:
            mask = train_countries == country
            country_xyz = train_xyz[mask]

            if len(country_xyz) < num_cells:
                centers = country_xyz
                n_cells = len(centers)
            else:
                kmeans = KMeans(n_clusters=num_cells, random_state=random_state, n_init=10)
                kmeans.fit(country_xyz)
                centers = kmeans.cluster_centers_
                n_cells = num_cells

            all_centers.append(centers)

            for _ in range(n_cells):
                cell_to_country[cell_idx] = country
                cell_idx += 1

            num_cells_per_country[country] = n_cells

        centers_xyz = np.vstack(all_centers)
        metadata = {
            "cell_to_country": cell_to_country,
            "num_cells_per_country": num_cells_per_country,
            "per_country": True,
            "total_cells": len(centers_xyz),
        }
    else:
        if len(train_xyz) < num_cells:
            centers_xyz = train_xyz
        else:
            kmeans = KMeans(n_clusters=num_cells, random_state=random_state, n_init=10)
            kmeans.fit(train_xyz)
            centers_xyz = kmeans.cluster_centers_

        metadata = {"per_country": False, "total_cells": len(centers_xyz)}

    return centers_xyz, metadata


def assign_geocells(
    coords: np.ndarray,  # [N, 2] (lat, lng)
    countries: Optional[np.ndarray] = None,  # [N] country labels
    centers_xyz: np.ndarray | None = None,  # [num_cells, 3]
    metadata: Optional[dict] = None,
) -> np.ndarray:
    if centers_xyz is None:
        raise ValueError("centers_xyz must be provided")

    xyz = latlng_to_xyz(coords[:, 0], coords[:, 1])  # [N, 3]

    if metadata is not None and metadata.get("per_country", False) and countries is not None:
        cell_labels = np.zeros(len(coords), dtype=np.int64)
        cell_to_country = metadata["cell_to_country"]

        unique_countries = np.unique(countries)
        unique_countries = unique_countries[~pd.isna(unique_countries)]

        country_to_cell_indices: dict[str, list[int]] = {}
        for cell_idx, country in cell_to_country.items():
            country_to_cell_indices.setdefault(country, []).append(cell_idx)

        for country in unique_countries:
            mask = countries == country
            if not mask.any():
                continue

            country_xyz = xyz[mask]
            if country in country_to_cell_indices:
                country_cell_indices = np.array(country_to_cell_indices[country])
                country_centers = centers_xyz[country_cell_indices]
                distances = np.linalg.norm(country_xyz[:, None, :] - country_centers[None, :, :], axis=2)
                nearest = np.argmin(distances, axis=1)
                cell_labels[mask] = country_cell_indices[nearest]
            else:
                distances = np.linalg.norm(xyz[:, None, :] - centers_xyz[None, :, :], axis=2)
                cell_labels[mask] = np.argmin(distances, axis=1)
    else:
        distances = np.linalg.norm(xyz[:, None, :] - centers_xyz[None, :, :], axis=2)
        cell_labels = np.argmin(distances, axis=1)

    return cell_labels


def compute_offsets(
    coords: np.ndarray,  # [N, 2] (lat, lng)
    cell_labels: np.ndarray,  # [N]
    centers_xyz: np.ndarray,  # [num_cells, 3]
) -> np.ndarray:
    xyz = latlng_to_xyz(coords[:, 0], coords[:, 1])
    cell_centers = centers_xyz[cell_labels]
    return xyz - cell_centers


