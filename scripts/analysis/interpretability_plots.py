#!/usr/bin/env python3
"""
Interpretability Visualization Script for Stage 2 Geolocation Models

Generates publication-quality, offline visualizations:
1. Geocells on world map (spherical Voronoi tessellation)
2. Prediction error maps (pointwise and cell-aggregated)
3. Ablation comparison plots (CDF, delta maps)

All visualizations are fully offline using Natural Earth data via geopandas.
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import LogNorm, Normalize, LinearSegmentedColormap
import geopandas as gpd
from shapely.geometry import Point, Polygon, LineString
from shapely.ops import unary_union

from src.dataset import get_transforms_from_processor
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from src.losses import haversine_distance
from scripts.training.train_stage2_cross_attention import (
    Stage2ImageDataset,
    stage2_collate_fn,
    load_stage1_checkpoint,
    load_image_encoder_weights_from_stage0_checkpoint,
    is_missing_or_none_path,
    compute_predicted_coords,
    compute_offset_targets,
    cell_center_to_latlng,
    latlng_to_cartesian,
    cartesian_to_latlng,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datetime pattern for run directories
DATETIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

# Ablation modes
ABLATION_MODES = ["both", "concept_only", "image_only"]

# Threshold accuracies for bar plots
THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
}


def find_latest_datetime_dir(parent_dir: Path, exclude_subdirs: Optional[List[str]] = None) -> Optional[Path]:
    """Find the subdirectory with the latest datetime timestamp."""
    if exclude_subdirs is None:
        exclude_subdirs = []
    
    datetime_dirs = []
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and subdir.name not in exclude_subdirs:
            if DATETIME_PATTERN.match(subdir.name):
                datetime_dirs.append(subdir)
    
    if not datetime_dirs:
        return None
    
    datetime_dirs.sort(key=lambda x: x.name, reverse=True)
    return datetime_dirs[0]


def find_latest_stage2_checkpoints(results_root: Path) -> List[Tuple[Path, str, str]]:
    """
    Find the LATEST Stage 2 checkpoint for each ablation mode and variant.
    
    Returns list of (checkpoint_path, variant, ablation_mode) tuples.
    """
    checkpoints = []
    results_root = Path(results_root)
    
    for mode in ABLATION_MODES:
        mode_dir = results_root / f"stage2_cross_attention_{mode}"
        if not mode_dir.exists():
            logger.warning(f"Mode directory not found: {mode_dir}")
            continue
        
        # Find LATEST finetuned checkpoint
        finetuned_dirs = [
            subdir for subdir in mode_dir.iterdir()
            if subdir.is_dir() and subdir.name != "vanilla_stage1" and DATETIME_PATTERN.match(subdir.name)
        ]
        if finetuned_dirs:
            finetuned_dirs.sort(key=lambda x: x.name, reverse=True)
            latest_finetuned = finetuned_dirs[0]
            ckpt_path = latest_finetuned / "checkpoints" / "best_model_stage2_xattn.pt"
            if ckpt_path.exists():
                checkpoints.append((ckpt_path, "finetuned", mode))
                logger.info(f"Found latest Stage 2 checkpoint: {ckpt_path} (variant=finetuned, mode={mode})")
        
        # Find LATEST vanilla checkpoint
        vanilla_dir = mode_dir / "vanilla_stage1"
        if vanilla_dir.exists():
            latest_vanilla_subdir = find_latest_datetime_dir(vanilla_dir)
            if latest_vanilla_subdir:
                ckpt_path = latest_vanilla_subdir / "checkpoints" / "best_model_stage2_xattn.pt"
                if ckpt_path.exists():
                    checkpoints.append((ckpt_path, "vanilla", mode))
                    logger.info(f"Found latest Stage 2 checkpoint: {ckpt_path} (variant=vanilla, mode={mode})")
    
    return checkpoints


def load_stage2_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    force_vanilla_encoder_for_patches: bool = False,
) -> tuple:
    """Load Stage 2 checkpoint."""
    logger.info(f"Loading Stage 2 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load Stage 1 checkpoint
    stage1_ckpt_path = Path(ckpt["stage1_checkpoint"])
    stage1_ckpt_data = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    encoder_config = StreetCLIPConfig(
        model_name=ckpt["encoder_model"],
        finetune=False,
        device=device,
    )
    image_encoder = StreetCLIPEncoder(encoder_config)

    if force_vanilla_encoder_for_patches:
        logger.info("force_vanilla_encoder_for_patches=True: using base encoder weights for patch extraction")
    else:
        embedded_enc = stage1_ckpt_data.get("image_encoder_state_dict")
        if embedded_enc is not None:
            logger.info("Using image_encoder_state_dict embedded in Stage1 checkpoint for patch extraction")
            image_encoder.load_state_dict(embedded_enc, strict=False)
        else:
            stage0_checkpoint = stage1_ckpt_data.get("stage0_checkpoint")
            if not is_missing_or_none_path(stage0_checkpoint):
                load_image_encoder_weights_from_stage0_checkpoint(Path(stage0_checkpoint), image_encoder)
            else:
                logger.info("Stage1 indicates vanilla lineage (no Stage0 checkpoint); using base encoder weights")

    image_encoder.model.eval()
    for param in image_encoder.model.parameters():
        param.requires_grad = False
    
    stage1_model, concept_info = load_stage1_checkpoint(
        stage1_ckpt_path,
        image_encoder,
        device,
    )
    
    # Create Stage 2 model
    model = Stage2CrossAttentionGeoHead(
        patch_dim=ckpt["patch_dim"],
        concept_emb_dim=ckpt["concept_dim"],
        num_cells=ckpt["num_cells"],
        coord_output_dim=ckpt["coord_output_dim"],
        num_heads=ckpt["num_heads"],
        ablation_mode=ckpt.get("ablation_mode", "both"),
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    
    cell_centers = ckpt["cell_centers"].to(device)
    
    return model, image_encoder, stage1_model, cell_centers, concept_info, ckpt


@torch.no_grad()
def run_inference(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Stage1ConceptModel,
    test_loader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    ablation_mode: str,
    patch_dim: int,
    concept_dim: int,
) -> pd.DataFrame:
    """Run inference and collect per-sample predictions."""
    model.eval()
    stage1_model.eval()
    
    all_predictions = []
    
    for batch in tqdm(test_loader, desc="Running inference"):
        images, coordinates, cell_labels, countries, image_paths = batch
        images = images.to(device)
        coordinates = coordinates.to(device)
        cell_labels = cell_labels.to(device)
        
        # Compute features + patch tokens
        if ablation_mode == "concept_only":
            img_features = image_encoder(images)
            concept_embs = stage1_model.concept_bottleneck(img_features.float())
            patch_tokens = torch.empty((images.size(0), 0, patch_dim), device=device, dtype=img_features.dtype)
        else:
            img_features, patch_tokens = image_encoder.get_features_and_patches(images)
            if ablation_mode == "image_only":
                concept_embs = torch.zeros((images.size(0), concept_dim), device=device, dtype=img_features.dtype)
            else:
                concept_embs = stage1_model.concept_bottleneck(img_features.float())
        
        # Forward pass
        outputs = model(concept_embs, patch_tokens, return_attention=False, return_gate=False)
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        
        # Get predictions
        pred_cells = cell_logits.argmax(dim=1)
        pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)
        
        # Compute errors
        dists = haversine_distance(pred_coords, coordinates)
        
        # Collect per-sample data
        for i in range(len(images)):
            gt_lat, gt_lng = coordinates[i, 0].item(), coordinates[i, 1].item()
            pred_lat, pred_lng = pred_coords[i, 0].item(), pred_coords[i, 1].item()
            error_km = dists[i].item()
            
            # Extract pano_id from image_path
            img_path = Path(image_paths[i])
            pano_id = img_path.stem if img_path.stem else f"unknown_{i}"
            
            all_predictions.append({
                "pano_id": pano_id,
                "image_path": str(image_paths[i]),
                "country": countries[i],
                "gt_lat": gt_lat,
                "gt_lng": gt_lng,
                "pred_lat": pred_lat,
                "pred_lng": pred_lng,
                "error_km": error_km,
                "pred_cell": pred_cells[i].item(),
                "gt_cell": cell_labels[i].item(),
            })
    
    return pd.DataFrame(all_predictions)


def load_world_basemap() -> gpd.GeoDataFrame:
    """Load Natural Earth world basemap (offline)."""
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        return world
    except Exception as e:
        logger.warning(f"Could not load Natural Earth dataset: {e}")
        logger.info("Creating minimal world basemap...")
        # Fallback: create a simple world outline
        world_bounds = Polygon([
            (-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)
        ])
        return gpd.GeoDataFrame([1], geometry=[world_bounds], crs="EPSG:4326")


def compute_spherical_voronoi(cell_centers: torch.Tensor) -> List[Polygon]:
    """
    Compute spherical Voronoi tessellation from cell centers.
    
    Returns list of Shapely Polygons representing Voronoi regions.
    """
    centers_np = cell_centers.cpu().numpy()
    
    # Ensure centers are on unit sphere
    norms = np.linalg.norm(centers_np, axis=1, keepdims=True)
    centers_np = centers_np / (norms + 1e-10)  # Avoid division by zero
    
    # Compute spherical Voronoi
    try:
        vor = SphericalVoronoi(centers_np, radius=1.0, center=np.array([0, 0, 0]))
        vor.sort_vertices_of_regions()
    except Exception as e:
        logger.warning(f"Spherical Voronoi computation failed: {e}")
        return []
    
    # Convert regions to lat/lng polygons
    polygons = []
    for i, region in enumerate(vor.regions):
        if len(region) < 3 or -1 in region:
            continue
        
        try:
            vertices_xyz = vor.vertices[region]
            # Convert to lat/lng
            z = vertices_xyz[:, 2]
            y = vertices_xyz[:, 1]
            x = vertices_xyz[:, 0]
            
            lats = np.rad2deg(np.arcsin(np.clip(z, -1.0, 1.0)))
            lngs = np.rad2deg(np.arctan2(y, x))
            
            # Handle dateline crossing by unwrapping
            lngs = np.unwrap(np.deg2rad(lngs), period=2*np.pi)
            lngs = np.rad2deg(lngs)
            
            # Clamp lats to valid range
            lats = np.clip(lats, -90, 90)
            
            # Create polygon
            coords = list(zip(lngs, lats))
            if len(coords) >= 3:
                poly = Polygon(coords)
                if poly.is_valid and not poly.is_empty:
                    polygons.append(poly)
        except Exception as e:
            logger.debug(f"Skipping invalid Voronoi region {i}: {e}")
            continue
    
    return polygons


def plot_geocells_voronoi(
    cell_centers: torch.Tensor,
    output_path: Path,
    world_basemap: Optional[gpd.GeoDataFrame] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (20, 10),
):
    """Plot geocells as spherical Voronoi tessellation on world map."""
    logger.info("Computing spherical Voronoi tessellation...")
    voronoi_polygons = compute_spherical_voronoi(cell_centers)
    
    logger.info(f"Computed {len(voronoi_polygons)} Voronoi regions")
    
    # Convert cell centers to lat/lng
    cell_lats, cell_lngs = cell_center_to_latlng(cell_centers)
    cell_lats = cell_lats.cpu().numpy()
    cell_lngs = cell_lngs.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot world basemap
    if world_basemap is not None:
        world_basemap.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)
    
    # Plot Voronoi regions
    voronoi_gdf = gpd.GeoDataFrame(range(len(voronoi_polygons)), geometry=voronoi_polygons, crs="EPSG:4326")
    voronoi_gdf.plot(ax=ax, edgecolor='steelblue', facecolor='none', linewidth=0.8, alpha=0.6)
    
    # Plot cell centers
    ax.scatter(
        cell_lngs, cell_lats,
        c='red', s=40, marker='*', edgecolors='darkred', linewidths=0.5,
        zorder=10, label=f'Cell Centers (N={len(cell_centers)})'
    )
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Semantic Geocells: Spherical Voronoi Tessellation', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='upper right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved geocell Voronoi map to {output_path}")


def plot_error_points(
    predictions_df: pd.DataFrame,
    output_path: Path,
    world_basemap: Optional[gpd.GeoDataFrame] = None,
    max_points: int = 10000,
    dpi: int = 300,
    figsize: Tuple[int, int] = (20, 10),
):
    """Plot GT points colored by prediction error."""
    df = predictions_df.copy()
    
    # Subsample if too many points
    if len(df) > max_points:
        df = df.sample(n=max_points, random_state=42)
        logger.info(f"Subsampled to {max_points} points for visualization")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot world basemap
    if world_basemap is not None:
        world_basemap.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)
    
    # Plot points colored by error (log scale)
    errors = df['error_km'].values
    scatter = ax.scatter(
        df['gt_lng'], df['gt_lat'],
        c=errors, s=8, alpha=0.6, cmap='viridis_r',
        norm=LogNorm(vmin=max(0.1, errors.min()), vmax=errors.max()),
        edgecolors='none'
    )
    
    # Add colorbar
    cbar = plt.colorbar(scatter, ax=ax, label='Prediction Error (km)', pad=0.02)
    cbar.ax.tick_params(labelsize=10)
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Prediction Errors: Ground Truth Locations', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error points map to {output_path}")


def plot_error_by_cell(
    predictions_df: pd.DataFrame,
    cell_centers: torch.Tensor,
    output_path: Path,
    world_basemap: Optional[gpd.GeoDataFrame] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (20, 10),
):
    """Plot Voronoi cells colored by mean prediction error."""
    # Aggregate errors by predicted cell
    cell_errors = predictions_df.groupby('pred_cell')['error_km'].agg(['mean', 'count']).reset_index()
    cell_errors.columns = ['cell_id', 'mean_error', 'sample_count']
    
    # Compute Voronoi tessellation
    logger.info("Computing Voronoi tessellation for error choropleth...")
    voronoi_polygons = compute_spherical_voronoi(cell_centers)
    
    # Create mapping from cell_id to error
    error_map = dict(zip(cell_errors['cell_id'], cell_errors['mean_error']))
    
    # Create GeoDataFrame with error values
    voronoi_with_errors = []
    for i, poly in enumerate(voronoi_polygons):
        if i < len(cell_centers):
            error = error_map.get(i, np.nan)
            voronoi_with_errors.append({'geometry': poly, 'mean_error': error, 'cell_id': i})
    
    if not voronoi_with_errors:
        logger.warning("No valid Voronoi polygons for error choropleth")
        return
    
    voronoi_gdf = gpd.GeoDataFrame(voronoi_with_errors, crs="EPSG:4326")
    voronoi_gdf = voronoi_gdf.dropna(subset=['mean_error'])
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot world basemap
    if world_basemap is not None:
        world_basemap.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)
    
    # Plot Voronoi cells colored by error
    voronoi_gdf.plot(
        ax=ax, column='mean_error', cmap='YlOrRd', edgecolor='black', linewidth=0.3,
        legend=True, legend_kwds={'label': 'Mean Error (km)', 'shrink': 0.8},
        missing_kwds={'color': 'lightgray', 'edgecolor': 'black', 'linewidth': 0.3}
    )
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title('Mean Prediction Error by Geocell', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error-by-cell choropleth to {output_path}")


def plot_error_cdf(
    predictions_by_checkpoint: Dict[str, pd.DataFrame],
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot CDF of distance errors for multiple checkpoints/ablations."""
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    for tag, df in predictions_by_checkpoint.items():
        errors = df['error_km'].values
        sorted_errors = np.sort(errors)
        cumulative = np.arange(1, len(sorted_errors) + 1) / len(sorted_errors)
        ax.plot(sorted_errors, cumulative, label=tag, linewidth=2, alpha=0.8)
    
    ax.set_xlabel('Distance Error (km)', fontsize=12)
    ax.set_ylabel('Cumulative Probability', fontsize=12)
    ax.set_title('Cumulative Distribution of Prediction Errors', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(True, alpha=0.3, linestyle='--')
    ax.legend(loc='lower right', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved error CDF to {output_path}")


def plot_ablation_delta(
    predictions_ref: pd.DataFrame,
    predictions_other: pd.DataFrame,
    cell_centers: torch.Tensor,
    ref_tag: str,
    other_tag: str,
    output_path: Path,
    world_basemap: Optional[gpd.GeoDataFrame] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (20, 10),
):
    """Plot cellwise delta error map between two ablations."""
    # Aggregate errors by cell for both
    ref_errors = predictions_ref.groupby('pred_cell')['error_km'].mean()
    other_errors = predictions_other.groupby('pred_cell')['error_km'].mean()
    
    # Compute delta (other - ref, positive = other is worse)
    all_cells = set(ref_errors.index) | set(other_errors.index)
    deltas = {}
    for cell_id in all_cells:
        ref_err = ref_errors.get(cell_id, np.nan)
        other_err = other_errors.get(cell_id, np.nan)
        if not (np.isnan(ref_err) or np.isnan(other_err)):
            deltas[cell_id] = other_err - ref_err
    
    if not deltas:
        logger.warning("No overlapping cells for delta computation")
        return
    
    # Compute Voronoi tessellation
    logger.info("Computing Voronoi tessellation for delta map...")
    voronoi_polygons = compute_spherical_voronoi(cell_centers)
    
    # Create GeoDataFrame with delta values
    voronoi_with_deltas = []
    for i, poly in enumerate(voronoi_polygons):
        if i < len(cell_centers) and i in deltas:
            voronoi_with_deltas.append({'geometry': poly, 'delta_error': deltas[i], 'cell_id': i})
    
    if not voronoi_with_deltas:
        logger.warning("No valid Voronoi polygons for delta map")
        return
    
    voronoi_gdf = gpd.GeoDataFrame(voronoi_with_deltas, crs="EPSG:4326")
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot world basemap
    if world_basemap is not None:
        world_basemap.plot(ax=ax, color='lightgray', edgecolor='white', linewidth=0.5, alpha=0.3)
    
    # Plot delta map (diverging colormap: blue = better, red = worse)
    vmin = voronoi_gdf['delta_error'].min()
    vmax = voronoi_gdf['delta_error'].max()
    vcenter = 0.0
    
    voronoi_gdf.plot(
        ax=ax, column='delta_error', cmap='RdBu_r', edgecolor='black', linewidth=0.3,
        vmin=vmin, vmax=vmax, center=vcenter,
        legend=True, legend_kwds={'label': f'Δ Error: {other_tag} - {ref_tag} (km)', 'shrink': 0.8}
    )
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude', fontsize=12)
    ax.set_ylabel('Latitude', fontsize=12)
    ax.set_title(f'Error Delta Map: {other_tag} vs {ref_tag}', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved ablation delta map to {output_path}")


def plot_threshold_accuracies(
    predictions_by_checkpoint: Dict[str, pd.DataFrame],
    output_path: Path,
    dpi: int = 300,
    figsize: Tuple[int, int] = (10, 6),
):
    """Plot threshold accuracies as bar chart."""
    tags = list(predictions_by_checkpoint.keys())
    thresholds = list(THRESHOLD_ACCURACIES.keys())
    threshold_values = [THRESHOLD_ACCURACIES[t] for t in thresholds]
    
    # Compute accuracies for each checkpoint
    accuracies = {}
    for tag, df in predictions_by_checkpoint.items():
        errors = df['error_km'].values
        accs = []
        for thresh_val in threshold_values:
            acc = np.mean(errors <= thresh_val)
            accs.append(acc)
        accuracies[tag] = accs
    
    # Create bar plot
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    x = np.arange(len(thresholds))
    width = 0.8 / len(tags)
    
    for i, tag in enumerate(tags):
        offset = (i - len(tags) / 2 + 0.5) * width
        ax.bar(x + offset, accuracies[tag], width, label=tag, alpha=0.8)
    
    ax.set_xlabel('Accuracy Threshold', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Threshold Accuracies by Ablation', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(thresholds, rotation=45, ha='right')
    ax.set_ylim([0, 1])
    ax.grid(True, alpha=0.3, axis='y', linestyle='--')
    ax.legend(loc='upper left', fontsize=10)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved threshold accuracies plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate interpretability plots for Stage 2 models")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 checkpoint (required if not using --batch_mode)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Auto-detect and process all latest Stage 2 checkpoints")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root directory containing results (for batch mode)")
    parser.add_argument("--csv_path", type=str, default=None,
                        help="Path to CSV dataset (not required with --skip_model_viz)")
    parser.add_argument("--splits_json", type=str, default=None,
                        help="Path to splits.json file (not required with --skip_model_viz)")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--geoguessr_id", type=str, default="6906237dc7731161a37282b2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots. Default: results/interpretability/<checkpoint_name>")
    parser.add_argument("--split", type=str, default="test", choices=["train", "val", "test"],
                        help="Which split to visualize (not required with --skip_model_viz)")
    parser.add_argument("--force_vanilla_encoder_for_patches", action="store_true", default=False)
    parser.add_argument("--skip_model_viz", action="store_true", default=False,
                        help="Skip checkpoint-based visualizations and only visualize geocells from cell_centers")
    
    args = parser.parse_args()
    
    # Check if --skip_model_viz requires dataset args
    if args.skip_model_viz:
        if args.csv_path is None or args.splits_json is None:
            parser.error("--csv_path and --splits_json are required for normal mode")
        logger.info("--skip_model_viz flag set: Will only visualize geocells from checkpoints (no model loading/inference)")
    else:
        if args.csv_path is None:
            parser.error("--csv_path is required when not using --skip_model_viz")
        if args.splits_json is None:
            parser.error("--splits_json is required when not using --skip_model_viz")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        if args.skip_model_viz:
            output_dir = Path(args.results_root) / "geocells"
        else:
            output_dir = Path(args.results_root) / "interpretability"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load world basemap once
    logger.info("Loading world basemap...")
    world_basemap = load_world_basemap()
    
    # ============ SKIP MODEL VIZ MODE (lightweight geocells only) ============
    if args.skip_model_viz:
        # Determine checkpoints to process
        if args.batch_mode:
            logger.info("Batch mode: Auto-detecting latest Stage 2 checkpoints...")
            results_root = Path(args.results_root)
            checkpoint_tuples = find_latest_stage2_checkpoints(results_root)
            
            if len(checkpoint_tuples) == 0:
                logger.error("No Stage 2 checkpoints found!")
                return
            
            logger.info(f"Found {len(checkpoint_tuples)} checkpoint(s) to process")
        else:
            if args.stage2_checkpoint is None:
                parser.error("--stage2_checkpoint is required when not using --batch_mode")
            checkpoint_tuples = [(Path(args.stage2_checkpoint), "checkpoint", "none")]
        
        # Process each checkpoint (only for cell_centers)
        for ckpt_path, variant, ablation_mode in checkpoint_tuples:
            tag = f"{variant}_{ablation_mode}" if variant != "checkpoint" else "checkpoint"
            logger.info(f"\n{'='*60}")
            logger.info(f"Processing: {ckpt_path}")
            logger.info(f"Tag: {tag}")
            logger.info(f"{'='*60}")
            
            # Load checkpoint (only for cell_centers)
            logger.info(f"Loading cell centers from {ckpt_path}")
            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            cell_centers = ckpt["cell_centers"]
            
            # Generate geocell visualization
            title_suffix = f"({tag})" if variant != "checkpoint" else ""
            output_path = output_dir / f"geocells_voronoi_{tag}.png"
            plot_geocells_voronoi(
                cell_centers, output_path,
                world_basemap=world_basemap,
                title_suffix=title_suffix
            )
        
        logger.info("\n" + "="*60)
        logger.info("Geocell visualization complete!")
        logger.info(f"Output directory: {output_dir}")
        logger.info("="*60)
        return
    
    # ============ NORMAL MODE: Full interpretability analysis ============
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load splits
    logger.info(f"Loading splits from {args.splits_json}")
    with open(args.splits_json, 'r') as f:
        splits_data = json.load(f)
    
    split_pano_ids = set(splits_data[f"{args.split}_pano_ids"])
    logger.info(f"{args.split.capitalize()} split: {len(split_pano_ids)} samples")
    
    # Load dataset from CSV
    logger.info(f"Loading dataset from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # Build dataset samples
    image_paths = []
    coordinates = []
    countries = []
    pano_ids = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        pano_id = row.get("pano_id") or row.get("panoId")
        if pd.isna(pano_id) or str(pano_id) not in split_pano_ids:
            continue
        
        if "image_path" in row and pd.notna(row["image_path"]):
            img_path = Path(row["image_path"])
        else:
            img_path = Path(args.data_root) / args.geoguessr_id / "export" / f"{pano_id}.jpg"
        
        if not img_path.exists():
            continue
        
        lat = row.get("latitude") or row.get("lat")
        lng = row.get("longitude") or row.get("lng")
        country = row.get("country", "unknown")
        
        if pd.isna(lat) or pd.isna(lng):
            continue
        
        image_paths.append(str(img_path))
        coordinates.append(torch.tensor([lat, lng], dtype=torch.float32))
        countries.append(country if not pd.isna(country) else "unknown")
        pano_ids.append(str(pano_id))
    
    if len(coordinates) == 0:
        raise RuntimeError(f"No {args.split} samples found!")
    
    coordinates = torch.stack(coordinates)
    logger.info(f"Dataset: {len(coordinates)} samples")
    
    # Determine checkpoints to process
    if args.batch_mode:
        logger.info("Batch mode: Auto-detecting latest Stage 2 checkpoints...")
        results_root = Path(args.results_root)
        checkpoint_tuples = find_latest_stage2_checkpoints(results_root)
        
        if len(checkpoint_tuples) == 0:
            logger.error("No Stage 2 checkpoints found!")
            return
        
        logger.info(f"Found {len(checkpoint_tuples)} checkpoint(s) to process")
    else:
        if args.stage2_checkpoint is None:
            parser.error("--stage2_checkpoint is required when not using --batch_mode")
        
        # Single checkpoint mode - need to determine variant and ablation_mode
        ckpt_path = Path(args.stage2_checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        variant = "vanilla" if is_missing_or_none_path(ckpt.get("stage0_checkpoint")) else "finetuned"
        ablation_mode = ckpt.get("ablation_mode", "both")
        checkpoint_tuples = [(ckpt_path, variant, ablation_mode)]
    
    # Process each checkpoint
    predictions_by_checkpoint = {}
    cell_centers_by_checkpoint = {}
    
    for ckpt_path, variant, ablation_mode in checkpoint_tuples:
        tag = f"{variant}_{ablation_mode}"
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ckpt_path}")
        logger.info(f"Tag: {tag}")
        logger.info(f"{'='*60}")
        
        # Setup output directory
        if args.output_dir:
            output_dir = Path(args.output_dir) / tag
        else:
            output_dir = Path(args.results_root) / "interpretability" / tag
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Load checkpoint
        model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
            ckpt_path, device, args.force_vanilla_encoder_for_patches
        )
        
        # Assign geocells to samples
        coords_np = coordinates.cpu().numpy()
        centers_np = cell_centers.cpu().numpy()
        
        lat_rad = np.deg2rad(coords_np[:, 0])
        lng_rad = np.deg2rad(coords_np[:, 1])
        x = np.cos(lat_rad) * np.cos(lng_rad)
        y = np.cos(lat_rad) * np.sin(lng_rad)
        z = np.sin(lat_rad)
        xyz = np.stack([x, y, z], axis=1)
        
        distances = np.linalg.norm(xyz[:, None, :] - centers_np[None, :, :], axis=2)
        cell_labels = torch.tensor(np.argmin(distances, axis=1), dtype=torch.long)
        
        # Create dataset and loader
        transforms = get_transforms_from_processor(image_encoder.image_processor)
        dataset = Stage2ImageDataset(
            image_paths=image_paths,
            coordinates=coordinates,
            cell_labels=cell_labels,
            countries=countries,
            transforms=transforms,
        )
        
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=stage2_collate_fn,
            pin_memory=True,
        )
        
        # Run inference
        predictions_df = run_inference(
            model, image_encoder, stage1_model, loader, device,
            cell_centers, ckpt["coord_output_dim"], ablation_mode,
            ckpt["patch_dim"], ckpt["concept_dim"]
        )
        
        # Save predictions
        preds_path = output_dir / f"predictions_{tag}.csv.gz"
        predictions_df.to_csv(preds_path, index=False, compression='gzip')
        logger.info(f"Saved predictions to {preds_path}")
        
        predictions_by_checkpoint[tag] = predictions_df
        cell_centers_by_checkpoint[tag] = cell_centers
        
        # Generate individual plots for this checkpoint
        logger.info("Generating plots...")
        
        # Geocell Voronoi map
        plot_geocells_voronoi(
            cell_centers, output_dir / f"geocells_voronoi_{tag}.png",
            world_basemap=world_basemap
        )
        
        # Error points map
        plot_error_points(
            predictions_df, output_dir / f"errors_points_{tag}.png",
            world_basemap=world_basemap
        )
        
        # Error by cell choropleth
        plot_error_by_cell(
            predictions_df, cell_centers, output_dir / f"errors_by_cell_{tag}.png",
            world_basemap=world_basemap
        )
        
        # Threshold accuracies
        plot_threshold_accuracies(
            {tag: predictions_df}, output_dir / f"threshold_accuracies_{tag}.png"
        )
        
        # Save worst-case samples
        worst_samples = predictions_df.nlargest(20, 'error_km')[['pano_id', 'image_path', 'error_km', 'gt_lat', 'gt_lng', 'pred_lat', 'pred_lng']]
        worst_path = output_dir / f"worst_samples_{tag}.csv"
        worst_samples.to_csv(worst_path, index=False)
        logger.info(f"Saved worst-case samples to {worst_path}")
    
    # Generate comparison plots if multiple checkpoints
    if len(predictions_by_checkpoint) > 1:
        logger.info("\nGenerating comparison plots...")
        comparison_dir = Path(args.output_dir) if args.output_dir else Path(args.results_root) / "interpretability"
        comparison_dir.mkdir(parents=True, exist_ok=True)
        
        # Error CDF comparison
        plot_error_cdf(
            predictions_by_checkpoint, comparison_dir / "error_cdf_comparison.png"
        )
        
        # Threshold accuracies comparison
        plot_threshold_accuracies(
            predictions_by_checkpoint, comparison_dir / "threshold_accuracies_comparison.png"
        )
        
        # Delta maps between ablations (if we have both variants)
        tags_list = list(predictions_by_checkpoint.keys())
        if len(tags_list) >= 2:
            # Compare first two (or specific pairs)
            ref_tag = tags_list[0]
            other_tag = tags_list[1]
            ref_cell_centers = cell_centers_by_checkpoint[ref_tag]
            
            plot_ablation_delta(
                predictions_by_checkpoint[ref_tag],
                predictions_by_checkpoint[other_tag],
                ref_cell_centers,
                ref_tag, other_tag,
                comparison_dir / f"ablation_delta_{other_tag}_vs_{ref_tag}.png",
                world_basemap=world_basemap
            )
    
    logger.info("\n" + "="*60)
    logger.info("Interpretability plots generation complete!")
    logger.info("="*60)


if __name__ == "__main__":
    main()

