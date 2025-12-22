"""
Visualization utilities for Stage 2 geolocation predictions.
"""

from __future__ import annotations

import torch
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional
from PIL import Image
from src.cbm.stage2.metrics import xyz_to_latlng


def visualize_attention_overlay(
    image_path: str,
    attention_weights: np.ndarray,  # [P] or [H, W] attention weights
    output_path: str | Path,
    patch_size: int = 14,
    image_size: int = 336,
):
    """
    Overlay attention weights on image.
    
    Args:
        image_path: Path to input image
        attention_weights: [P] or [H, W] attention weights (will be normalized)
        output_path: Path to save visualization
        patch_size: Patch size in pixels
        image_size: Image size in pixels
    """
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        img = img.resize((image_size, image_size))
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return
    
    # Normalize attention
    attention = attention_weights.copy()
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)
    
    # Reshape to spatial grid if needed
    grid_size = image_size // patch_size
    if attention.ndim == 1:
        if len(attention) == grid_size * grid_size:
            attention = attention.reshape(grid_size, grid_size)
        else:
            # Pad or interpolate
            attention = attention[:grid_size * grid_size]
            attention = attention.reshape(grid_size, grid_size)
    
    # Upsample to image size using simple repeat
    # attention is [grid_size, grid_size]
    scale = image_size // grid_size
    attention_upsampled = np.repeat(np.repeat(attention, scale, axis=0), scale, axis=1)
    # Crop if needed
    if attention_upsampled.shape[0] > image_size:
        attention_upsampled = attention_upsampled[:image_size, :image_size]
    
    # Create overlay
    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    
    # Overlay attention as heatmap
    im = ax.imshow(attention_upsampled, alpha=0.5, cmap='hot', interpolation='bilinear')
    ax.axis('off')
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_predictions_map(
    pred_lat: np.ndarray,
    pred_lng: np.ndarray,
    true_lat: np.ndarray,
    true_lng: np.ndarray,
    output_path: str | Path,
    max_samples: int = 1000,
):
    """
    Visualize predictions on a world map.
    
    Args:
        pred_lat, pred_lng: [N] predicted coordinates
        true_lat, true_lng: [N] ground truth coordinates
        output_path: Path to save visualization
        max_samples: Maximum number of samples to plot (for performance)
    """
    if len(pred_lat) > max_samples:
        indices = np.random.choice(len(pred_lat), max_samples, replace=False)
        pred_lat = pred_lat[indices]
        pred_lng = pred_lng[indices]
        true_lat = true_lat[indices]
        true_lng = true_lng[indices]
    
    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    
    # Plot world map background (simple)
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')
    ax.set_title('Geolocation Predictions')
    ax.grid(True, alpha=0.3)
    
    # Plot ground truth
    ax.scatter(true_lng, true_lat, c='blue', s=10, alpha=0.5, label='Ground Truth', marker='o')
    
    # Plot predictions
    ax.scatter(pred_lng, pred_lat, c='red', s=10, alpha=0.5, label='Predictions', marker='x')
    
    # Draw lines connecting predictions to ground truth
    for i in range(min(100, len(pred_lat))):  # Limit lines for readability
        ax.plot([true_lng[i], pred_lng[i]], [true_lat[i], pred_lat[i]], 
                'gray', alpha=0.3, linewidth=0.5)
    
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()


def dump_predictions(
    pano_ids: list[str],
    pred_lat: np.ndarray,
    pred_lng: np.ndarray,
    true_lat: np.ndarray,
    true_lng: np.ndarray,
    distances_km: np.ndarray,
    output_path: str | Path,
    top_k: int = 100,
):
    """
    Dump predictions to a text file (sorted by error).
    
    Args:
        pano_ids: List of pano IDs
        pred_lat, pred_lng: [N] predicted coordinates
        true_lat, true_lng: [N] ground truth coordinates
        distances_km: [N] prediction errors in km
        output_path: Path to save text file
        top_k: Number of worst predictions to show
    """
    # Sort by error (worst first)
    sorted_indices = np.argsort(distances_km)[::-1]
    
    with open(output_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write(f"Stage 2 Geolocation Predictions (Top {top_k} worst errors)\n")
        f.write("=" * 80 + "\n\n")
        
        for i, idx in enumerate(sorted_indices[:top_k]):
            f.write(f"Rank {i+1}: {pano_ids[idx]}\n")
            f.write(f"  True:  ({true_lat[idx]:.4f}, {true_lng[idx]:.4f})\n")
            f.write(f"  Pred:  ({pred_lat[idx]:.4f}, {pred_lng[idx]:.4f})\n")
            f.write(f"  Error: {distances_km[idx]:.2f} km\n")
            f.write("\n")
        
        f.write("=" * 80 + "\n")
        f.write("Statistics:\n")
        f.write(f"  Mean error: {np.mean(distances_km):.2f} km\n")
        f.write(f"  Median error: {np.median(distances_km):.2f} km\n")
        f.write(f"  Min error: {np.min(distances_km):.2f} km\n")
        f.write(f"  Max error: {np.max(distances_km):.2f} km\n")
        f.write("=" * 80 + "\n")


def visualize_geocell_centers(
    centers_xyz: np.ndarray,  # [num_cells, 3] cell centers in 3D Cartesian
    output_path: str | Path,
    title: str = "Geocell Centers",
):
    """
    Visualize geocell centers on a world map.
    
    Args:
        centers_xyz: [num_cells, 3] cell centers in 3D Cartesian coordinates
        output_path: Path to save visualization
        title: Title for the plot
    """
    # Convert 3D Cartesian to lat/lng
    centers_lat, centers_lng = xyz_to_latlng(centers_xyz)
    
    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    
    # Plot world map background
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'{title} ({len(centers_xyz)} cells)', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    # Plot geocell centers
    scatter = ax.scatter(
        centers_lng, 
        centers_lat, 
        c='red', 
        s=20, 
        alpha=0.6, 
        marker='o',
        edgecolors='darkred',
        linewidths=0.5,
        label=f'Geocell Centers (n={len(centers_xyz)})'
    )
    
    ax.legend(fontsize=10, loc='upper right')
    
    # Add some statistics as text
    stats_text = (
        f"Total cells: {len(centers_xyz)}\n"
        f"Lat range: [{centers_lat.min():.2f}°, {centers_lat.max():.2f}°]\n"
        f"Lng range: [{centers_lng.min():.2f}°, {centers_lng.max():.2f}°]"
    )
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, 
            fontsize=9, verticalalignment='top',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches='tight')
    plt.close()
    
    print(f"Saved geocell centers map to {output_path}")
