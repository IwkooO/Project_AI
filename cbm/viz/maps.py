"""
Visualization utilities for Phase 2 geolocation predictions.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from cbm.phase2.metrics import xyz_to_latlng


def visualize_attention_overlay(
    image_path: str,
    attention_weights: np.ndarray,  # [P] or [H, W]
    output_path: str | Path,
    patch_size: int = 14,
    image_size: int = 336,
):
    try:
        img = Image.open(image_path).convert("RGB")
        img = img.resize((image_size, image_size))
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return

    attention = attention_weights.copy()
    attention = (attention - attention.min()) / (attention.max() - attention.min() + 1e-8)

    grid_size = image_size // patch_size
    if attention.ndim == 1:
        attention = attention[: grid_size * grid_size].reshape(grid_size, grid_size)

    scale = image_size // grid_size
    attention_upsampled = np.repeat(np.repeat(attention, scale, axis=0), scale, axis=1)
    if attention_upsampled.shape[0] > image_size:
        attention_upsampled = attention_upsampled[:image_size, :image_size]

    fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    ax.imshow(img)
    im = ax.imshow(attention_upsampled, alpha=0.5, cmap="hot", interpolation="bilinear")
    ax.axis("off")
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def visualize_predictions_map(
    pred_lat: np.ndarray,
    pred_lng: np.ndarray,
    true_lat: np.ndarray,
    true_lng: np.ndarray,
    output_path: str | Path,
    max_samples: int = 1000,
):
    if len(pred_lat) > max_samples:
        indices = np.random.choice(len(pred_lat), max_samples, replace=False)
        pred_lat = pred_lat[indices]
        pred_lng = pred_lng[indices]
        true_lat = true_lat[indices]
        true_lng = true_lng[indices]

    fig, ax = plt.subplots(1, 1, figsize=(16, 8))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Geolocation Predictions")
    ax.grid(True, alpha=0.3)

    ax.scatter(true_lng, true_lat, c="blue", s=10, alpha=0.5, label="Ground Truth", marker="o")
    ax.scatter(pred_lng, pred_lat, c="red", s=10, alpha=0.5, label="Predictions", marker="x")

    for i in range(min(100, len(pred_lat))):
        ax.plot([true_lng[i], pred_lng[i]], [true_lat[i], pred_lat[i]], "gray", alpha=0.3, linewidth=0.5)

    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
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
    sorted_indices = np.argsort(distances_km)[::-1]

    with open(output_path, "w") as f:
        f.write("=" * 80 + "\n")
        f.write(f"Stage 2 Geolocation Predictions (Top {top_k} worst errors)\n")
        f.write("=" * 80 + "\n\n")

        for i, idx in enumerate(sorted_indices[:top_k]):
            f.write(f"Rank {i+1}: {pano_ids[idx]}\n")
            f.write(f"  True:  ({true_lat[idx]:.4f}, {true_lng[idx]:.4f})\n")
            f.write(f"  Pred:  ({pred_lat[idx]:.4f}, {pred_lng[idx]:.4f})\n")
            f.write(f"  Error: {distances_km[idx]:.2f} km\n\n")

        f.write("=" * 80 + "\n")
        f.write("Statistics:\n")
        f.write(f"  Mean error: {np.mean(distances_km):.2f} km\n")
        f.write(f"  Median error: {np.median(distances_km):.2f} km\n")
        f.write(f"  Min error: {np.min(distances_km):.2f} km\n")
        f.write(f"  Max error: {np.max(distances_km):.2f} km\n")
        f.write("=" * 80 + "\n")


def visualize_geocell_centers(
    centers_xyz: np.ndarray,  # [num_cells, 3]
    output_path: str | Path,
    title: str = "Geocell Centers",
):
    centers_lat, centers_lng = xyz_to_latlng(centers_xyz)

    fig, ax = plt.subplots(1, 1, figsize=(20, 10))
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xlabel("Longitude", fontsize=12)
    ax.set_ylabel("Latitude", fontsize=12)
    ax.set_title(f"{title} ({len(centers_xyz)} cells)", fontsize=14, fontweight="bold")
    ax.grid(True, alpha=0.3, linestyle="--")

    ax.scatter(
        centers_lng,
        centers_lat,
        c="red",
        s=20,
        alpha=0.6,
        marker="o",
        edgecolors="darkred",
        linewidths=0.5,
        label=f"Geocell Centers (n={len(centers_xyz)})",
    )

    ax.legend(fontsize=10, loc="upper right")
    stats_text = (
        f"Total cells: {len(centers_xyz)}\n"
        f"Lat range: [{centers_lat.min():.2f}°, {centers_lat.max():.2f}°]\n"
        f"Lng range: [{centers_lng.min():.2f}°, {centers_lng.max():.2f}°]"
    )
    ax.text(
        0.02,
        0.98,
        stats_text,
        transform=ax.transAxes,
        fontsize=9,
        verticalalignment="top",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.5),
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close()


