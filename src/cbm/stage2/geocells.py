"""
Semantic geocell fitting and assignment.

Geocells are fitted on train split only (to prevent data leakage).
Val/test samples are assigned to nearest train-fitted cell center.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from typing import Optional
from sklearn.cluster import KMeans


def latlng_to_xyz(lat: np.ndarray, lng: np.ndarray) -> np.ndarray:
    """
    Convert lat/lng to 3D Cartesian coordinates (unit sphere).
    
    Args:
        lat: [N] latitude in degrees
        lng: [N] longitude in degrees
    
    Returns:
        xyz: [N, 3] (x, y, z) coordinates on unit sphere
    """
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
    """
    Fit semantic geocells on training data only.
    
    Args:
        train_coords: [N_train, 2] (lat, lng) training coordinates
        train_countries: Optional [N_train] country labels for per-country fitting
        num_cells: Total number of geocells
        per_country: If True, fit cells per country (num_cells per country)
        random_state: Random seed for KMeans
    
    Returns:
        (centers_xyz, metadata)
          - centers_xyz: [num_cells, 3] or [num_countries * num_cells, 3] cell centers in 3D Cartesian
          - metadata: dict with 'cell_to_country', 'num_cells_per_country', etc.
    """
    # Convert to 3D Cartesian
    train_xyz = latlng_to_xyz(train_coords[:, 0], train_coords[:, 1])  # [N_train, 3]
    
    if per_country and train_countries is not None:
        # Fit cells per country
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
                # Not enough samples, use all samples as centers
                centers = country_xyz
                n_cells = len(centers)
            else:
                # KMeans clustering
                kmeans = KMeans(n_clusters=num_cells, random_state=random_state, n_init=10)
                kmeans.fit(country_xyz)
                centers = kmeans.cluster_centers_
                n_cells = num_cells
            
            all_centers.append(centers)
            
            for i in range(n_cells):
                cell_to_country[cell_idx] = country
                cell_idx += 1
            
            num_cells_per_country[country] = n_cells
        
        centers_xyz = np.vstack(all_centers)  # [total_cells, 3]
        
        metadata = {
            'cell_to_country': cell_to_country,
            'num_cells_per_country': num_cells_per_country,
            'per_country': True,
            'total_cells': len(centers_xyz),
        }
    else:
        # Global fitting (all training data)
        if len(train_xyz) < num_cells:
            # Not enough samples, use all samples as centers
            centers_xyz = train_xyz
            num_cells = len(centers_xyz)
        else:
            # KMeans clustering
            kmeans = KMeans(n_clusters=num_cells, random_state=random_state, n_init=10)
            kmeans.fit(train_xyz)
            centers_xyz = kmeans.cluster_centers_
        
        metadata = {
            'per_country': False,
            'total_cells': len(centers_xyz),
        }
    
    print(f"Fitted {len(centers_xyz)} geocells")
    if per_country:
        print(f"  Per-country mode: {len(unique_countries)} countries")
    
    return centers_xyz, metadata


def assign_geocells(
    coords: np.ndarray,  # [N, 2] (lat, lng)
    countries: Optional[np.ndarray] = None,  # [N] country labels
    centers_xyz: np.ndarray = None,  # [num_cells, 3] cell centers
    metadata: Optional[dict] = None,  # Metadata from fit_semantic_geocells
) -> np.ndarray:
    """
    Assign coordinates to nearest geocell center.
    
    Args:
        coords: [N, 2] (lat, lng) coordinates to assign
        countries: Optional [N] country labels (used if per_country=True)
        centers_xyz: [num_cells, 3] cell centers in 3D Cartesian
        metadata: Metadata from fit_semantic_geocells
    
    Returns:
        cell_labels: [N] cell indices (0 to num_cells-1)
    """
    # Convert to 3D Cartesian
    xyz = latlng_to_xyz(coords[:, 0], coords[:, 1])  # [N, 3]
    
    if metadata is not None and metadata.get('per_country', False) and countries is not None:
        # Per-country assignment
        cell_labels = np.zeros(len(coords), dtype=np.int64)
        cell_to_country = metadata['cell_to_country']
        
        unique_countries = np.unique(countries)
        unique_countries = unique_countries[~pd.isna(unique_countries)]
        
        # Build country -> cell indices mapping
        country_to_cell_indices = {}
        for cell_idx, country in cell_to_country.items():
            if country not in country_to_cell_indices:
                country_to_cell_indices[country] = []
            country_to_cell_indices[country].append(cell_idx)
        
        # Assign per country
        for country in unique_countries:
            mask = countries == country
            if not mask.any():
                continue
            
            country_xyz = xyz[mask]  # [N_country, 3]
            
            if country in country_to_cell_indices:
                country_cell_indices = np.array(country_to_cell_indices[country])
                country_centers = centers_xyz[country_cell_indices]  # [num_cells_country, 3]
                
                # Compute distances: [N_country, num_cells_country]
                distances = np.linalg.norm(
                    country_xyz[:, None, :] - country_centers[None, :, :],
                    axis=2
                )
                
                # Assign to nearest
                nearest = np.argmin(distances, axis=1)  # [N_country]
                cell_labels[mask] = country_cell_indices[nearest]
            else:
                # Country not in training data, assign to nearest global cell
                distances = np.linalg.norm(
                    country_xyz[:, None, :] - centers_xyz[None, :, :],
                    axis=2
                )
                cell_labels[mask] = np.argmin(distances, axis=1)
    else:
        # Global assignment (all cells)
        # Compute distances: [N, num_cells]
        distances = np.linalg.norm(
            xyz[:, None, :] - centers_xyz[None, :, :],
            axis=2
        )
        
        # Assign to nearest
        cell_labels = np.argmin(distances, axis=1)  # [N]
    
    return cell_labels


def compute_offsets(
    coords: np.ndarray,  # [N, 2] (lat, lng)
    cell_labels: np.ndarray,  # [N] cell indices
    centers_xyz: np.ndarray,  # [num_cells, 3] cell centers
) -> np.ndarray:
    """
    Compute offset from cell center to actual coordinates.
    
    Args:
        coords: [N, 2] (lat, lng) actual coordinates
        cell_labels: [N] assigned cell indices
        centers_xyz: [num_cells, 3] cell centers in 3D Cartesian
    
    Returns:
        offsets: [N, 3] offset in 3D Cartesian (xyz - center_xyz)
    """
    xyz = latlng_to_xyz(coords[:, 0], coords[:, 1])  # [N, 3]
    cell_centers = centers_xyz[cell_labels]  # [N, 3]
    offsets = xyz - cell_centers  # [N, 3]
    return offsets
