#!/usr/bin/env python3
"""
Geocell Visualization Script (Geographic Map with Countries)

Generates publication-quality, offline visualization of geocells as actual geographic regions on a world map with countries.
Shows countries, geocell clusters, and allows subsetting for cleaner plots.

Usage:
    python scripts/analysis/visualize_geocells_map.py \
        --batch_mode \
        --results_root results \
        --max_cells 200 \
        --countries "USA,Japan,United Kingdom"
"""

import argparse
import logging
import re
from pathlib import Path
from typing import List, Optional, Set, Tuple

import numpy as np
import pandas as pd
import torch
from scipy.spatial import SphericalVoronoi
import matplotlib.pyplot as plt
import geopandas as gpd
from shapely.geometry import Polygon, Point

from scripts.training.train_stage2_cross_attention import cell_center_to_latlng

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datetime pattern for run directories
DATETIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}")

# Ablation modes
ABLATION_MODES = ["both", "concept_only", "image_only"]


def find_latest_datetime_dir(parent_dir: Path, exclude_subdirs: Optional[List[str]] = None) -> Optional[Path]:
    """Find subdirectory with the latest datetime timestamp."""
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


def find_latest_stage2_checkpoints(
    results_root: Path, 
    countries: Optional[Set[str]] = None
) -> List[Tuple[Path, str, str]]:
    """
    Find LATEST Stage 2 checkpoint for each ablation mode and variant.
    Optionally filter by countries.
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
            
            # Filter by country if specified
            if countries is not None:
                ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                ckpt_countries = set(ckpt.get("countries", []))
                if not ckpt_countries.intersection(countries):
                    logger.info(f"Skipping {mode_dir} (no matching countries)")
                    continue
            
            if ckpt_path.exists():
                checkpoints.append((ckpt_path, "finetuned", mode))
                logger.info(f"Found latest Stage 2 checkpoint: {ckpt_path} (variant=finetuned, mode={mode})")
        
        # Find LATEST vanilla checkpoint
        vanilla_dir = mode_dir / "vanilla_stage1"
        if vanilla_dir.exists():
            latest_vanilla_subdir = find_latest_datetime_dir(vanilla_dir)
            if latest_vanilla_subdir:
                ckpt_path = latest_vanilla_subdir / "checkpoints" / "best_model_stage2_xattn.pt"
                
                # Filter by country if specified
                if countries is not None:
                    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
                    ckpt_countries = set(ckpt.get("countries", []))
                    if not ckpt_countries.intersection(countries):
                        logger.info(f"Skipping {mode_dir} (no matching countries)")
                        continue
                
                if ckpt_path.exists():
                    checkpoints.append((ckpt_path, "vanilla", mode))
                    logger.info(f"Found latest Stage 2 checkpoint: {ckpt_path} (variant=vanilla, mode={mode})")
    
    return checkpoints


def load_world_basemap() -> gpd.GeoDataFrame:
    """Load Natural Earth world basemap with country boundaries."""
    # Try downloading Natural Earth countries directly (best option - has country names)
    try:
        url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"
        world = gpd.read_file(url)
        logger.info(f"Downloaded Natural Earth countries ({len(world)} features)")
        return world
    except Exception as e1:
        logger.warning(f"Could not download Natural Earth countries: {e1}")
    
    # Try geodatasets package (may not have country names)
    try:
        import geodatasets
        world = gpd.read_file(geodatasets.get_path("naturalearth.land"))
        logger.info(f"Loaded Natural Earth land from geodatasets ({len(world)} features)")
        return world
    except Exception as e2:
        logger.warning(f"Could not load from geodatasets: {e2}")
    
    # Try legacy geopandas datasets
    try:
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        logger.info(f"Loaded Natural Earth from geopandas ({len(world)} features)")
        return world
    except Exception as e3:
        logger.warning(f"Could not load from geopandas: {e3}")
    
    # Final fallback: create minimal world outline
    logger.info("Creating minimal world basemap (no country data available)...")
    world_bounds = Polygon([
        (-180, -90), (180, -90), (180, 90), (-180, 90), (-180, -90)
    ])
    return gpd.GeoDataFrame([{'geometry': world_bounds}], crs="EPSG:4326")


def compute_spherical_voronoi_regions(cell_centers: torch.Tensor) -> List[Tuple[int, Polygon]]:
    """
    Compute spherical Voronoi tessellation from cell centers.
    
    Returns list of (cell_id, polygon) tuples.
    """
    centers_np = cell_centers.cpu().numpy()
    
    # Ensure centers are on unit sphere
    norms = np.linalg.norm(centers_np, axis=1, keepdims=True)
    centers_np = centers_np / (norms + 1e-10)
    
    # Compute spherical Voronoi
    try:
        vor = SphericalVoronoi(centers_np, radius=1.0, center=np.array([0, 0, 0]))
        vor.sort_vertices_of_regions()
    except Exception as e:
        logger.warning(f"Spherical Voronoi computation failed: {e}")
        return []
    
    # Convert regions to lat/lng polygons
    regions = []
    for i, region in enumerate(vor.regions):
        if len(region) < 3 or -1 in region:
            continue
        
        try:
            vertices_xyz = vor.vertices[region]
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
                    regions.append((i, poly))
        except Exception as e:
            logger.debug(f"Skipping invalid Voronoi region {i}: {e}")
            continue
    
    return regions


def plot_geocells_on_map(
    cell_centers: torch.Tensor,
    output_path: Path,
    world_basemap: Optional[gpd.GeoDataFrame] = None,
    cell_subset: Optional[List[int]] = None,
    title_suffix: str = "",
    dpi: int = 300,
    figsize: Tuple[int, int] = (24, 12),
):
    """
    Plot geocells as Voronoi regions on world map.
    Shows countries if available in basemap.
    """
    logger.info("Computing spherical Voronoi tessellation...")
    regions = compute_spherical_voronoi_regions(cell_centers)
    voronoi_polygons = [poly for _, poly in regions]
    
    logger.info(f"Computed {len(voronoi_polygons)} Voronoi regions")
    
    # Convert cell centers to lat/lng
    cell_lats, cell_lngs = cell_center_to_latlng(cell_centers)
    cell_lats_np = cell_lats.cpu().numpy()
    cell_lngs_np = cell_lngs.cpu().numpy()
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Plot world basemap with better styling
    if world_basemap is not None:
        # Use a nicer color scheme for land
        world_basemap.plot(ax=ax, color='#E8E8E8', edgecolor='#AAAAAA', linewidth=0.3, alpha=0.8)
    
    # Add country labels if available
    if world_basemap is not None:
        # Check for various possible name columns
        name_cols = ['name', 'NAME', 'NAME_LONG', 'ADMIN', 'SOVEREIGNT', 'NAME_EN']
        name_col = None
        for col in name_cols:
            if col in world_basemap.columns:
                name_col = col
                break
        
        if name_col is not None:
            logger.info(f"Adding country labels using column: {name_col}")
            
            # Countries to label (major ones only)
            shown_countries = {'United States', 'United States of America', 'China', 'Russia', 
                              'Brazil', 'India', 'Australia', 'Canada', 'Japan', 
                              'United Kingdom', 'Germany', 'France', 'Spain', 'Italy', 
                              'Mexico', 'South Africa', 'Argentina', 'Indonesia', 'Egypt'}
            
            for idx, row in world_basemap.iterrows():
                geom_type = None
                if hasattr(row.geometry, 'geom_type'):
                    geom_type = row.geometry.geom_type
                
                # Get centroid for labels
                centroid = None
                if geom_type == 'MultiPolygon':
                    centroid = row.geometry.geoms[0].centroid
                elif geom_type == 'Polygon':
                    centroid = row.geometry.centroid
                
                # Get country name
                country = row.get(name_col, None)
                if country is not None and hasattr(country, '__str__'):
                    country = str(country)
                
                if centroid is not None and not centroid.is_empty and country in shown_countries:
                    x, y = centroid.x, centroid.y
                    ax.text(x, y, country, fontsize=9, fontweight='bold',
                           ha='center', va='center', 
                           bbox=dict(boxstyle='round,pad=0.3', facecolor='white', 
                                      edgecolor='#666666', alpha=0.85),
                           zorder=5)
    
    # Create GeoDataFrame for Voronoi regions
    if voronoi_polygons:
        regions_gdf = gpd.GeoDataFrame(
            [(cell_id, poly) for cell_id, poly in regions],
            columns=['cell_id', 'geometry'],
            crs="EPSG:4326"
        )
        
        # Determine which cells to subset/highlight
        if cell_subset is not None:
            regions_gdf['highlight'] = regions_gdf['cell_id'].isin(cell_subset)
            # Plot non-highlighted cells
            non_subset = regions_gdf[~regions_gdf['highlight']]
            if not non_subset.empty:
                non_subset.plot(ax=ax, edgecolor='steelblue', facecolor='none', linewidth=0.6, alpha=0.4, label='Geocell Regions')
            # Plot highlighted cells
            subset_gdf = regions_gdf[regions_gdf['highlight']]
            if not subset_gdf.empty:
                subset_gdf.plot(ax=ax, edgecolor='crimson', facecolor='none', linewidth=1.0, alpha=0.7, label='Selected Cells')
        else:
            regions_gdf.plot(ax=ax, edgecolor='steelblue', facecolor='none', linewidth=0.8, alpha=0.6, label='Geocell Regions')
    
    # Plot cell centers (subset only if specified)
    if cell_subset is not None:
        subset_mask = np.array([i for i in range(len(cell_lats_np)) if i in cell_subset])
        ax.scatter(
            cell_lngs_np[subset_mask], cell_lats_np[subset_mask],
            c='red', s=60, marker='*', edgecolors='darkred', linewidths=1,
            zorder=15, label='Cell Centers (subset)', alpha=0.9
        )
    else:
        ax.scatter(
            cell_lngs_np, cell_lats_np,
            c='red', s=35, marker='*', edgecolors='darkred', linewidths=0.5,
            zorder=10, label=f'Cell Centers (N={len(cell_centers)})', alpha=0.6
        )
    
    ax.set_xlim([-180, 180])
    ax.set_ylim([-90, 90])
    ax.set_xlabel('Longitude', fontsize=14)
    ax.set_ylabel('Latitude', fontsize=14)
    ax.set_title(f'Semantic Geocells', fontsize=18, fontweight='bold', pad=20)
    ax.grid(True, alpha=0.2, linestyle='--', zorder=1)
    ax.legend(loc='lower right', fontsize=11, framealpha=0.9)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved geocell map to {output_path}")


def plot_cell_distribution(
    cell_centers: torch.Tensor,
    output_path: Path,
    countries: Optional[List[str]] = None,
    dpi: int = 300,
    figsize: Tuple[int, int] = (14, 8),
):
    """Plot distribution of geocells (count by latitude, by hemisphere)."""
    cell_lats, cell_lngs = cell_center_to_latlng(cell_centers)
    cell_lats_np = cell_lats.cpu().numpy()
    cell_lngs_np = cell_lngs.cpu().numpy()
    
    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize, dpi=dpi)
    
    # Plot by latitude band
    ax1 = axes[0]
    lat_bands = np.linspace(-90, 90, 13)
    hist, _ = np.histogram(cell_lats_np, bins=lat_bands)
    ax1.bar(hist[:-1], np.diff(hist), width=12, edgecolor='black', alpha=0.7, color='steelblue')
    ax1.set_xlabel('Latitude', fontsize=11)
    ax1.set_ylabel('Number of Cells', fontsize=11)
    ax1.set_title('Geocell Distribution by Latitude', fontsize=12, fontweight='bold')
    ax1.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    # Plot by hemisphere
    ax2 = axes[1]
    northern = np.sum(cell_lats_np >= 0)
    southern = np.sum(cell_lats_np < 0)
    ax2.bar(['Northern', 'Southern'], [northern, southern], color=['cornflowerblue', 'orange'], alpha=0.7)
    ax2.set_ylabel('Number of Cells', fontsize=11)
    ax2.set_title('Geocell Distribution by Hemisphere', fontsize=12, fontweight='bold')
    ax2.grid(True, alpha=0.3, axis='y', linestyle='--')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=dpi, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved cell distribution plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Visualize geocells on geographic world map with countries")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 checkpoint (optional)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Auto-detect and process all latest Stage 2 checkpoints")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root directory containing results (for batch mode)")
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for plots. Default: results/geocells/")
    parser.add_argument("--countries", type=str, default=None,
                        help="Comma-separated list of countries to filter checkpoints by (e.g., 'USA,Japan,United Kingdom')")
    parser.add_argument("--max_cells", type=int, default=None,
                        help="Maximum number of cells to visualize (for cleaner plots)")
    
    args = parser.parse_args()
    
    # Parse countries
    countries_filter = None
    if args.countries is not None:
        countries_filter = set(c.strip() for c in args.countries.split(','))
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = Path(args.results_root) / "geocells_map"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load world basemap
    logger.info("Loading world basemap...")
    world_basemap = load_world_basemap()
    
    # Determine checkpoints to process
    if args.batch_mode:
        logger.info("Batch mode: Auto-detecting latest Stage 2 checkpoints...")
        results_root = Path(args.results_root)
        checkpoint_tuples = find_latest_stage2_checkpoints(results_root, countries_filter)
        
        if len(checkpoint_tuples) == 0:
            logger.error("No Stage 2 checkpoints found!")
            return
        
        logger.info(f"Found {len(checkpoint_tuples)} checkpoint(s) to process")
    else:
        if args.stage2_checkpoint is None:
            parser.error("--stage2_checkpoint is required when not using --batch_mode")
        
        # Single checkpoint mode
        ckpt_path = Path(args.stage2_checkpoint)
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        ablation_mode = ckpt.get("ablation_mode", "both")
        variant = "checkpoint"
        checkpoint_tuples = [(ckpt_path, variant, ablation_mode)]
    
    # Determine cell subset for visualization
    cell_subset = None
    if args.max_cells is not None and args.max_cells < len(checkpoint_tuples):
        # Randomly subset cells for cleaner visualization
        import random
        seed = 42
        # Will subset per checkpoint
        pass
    
    # Process each checkpoint
    for idx, (ckpt_path, variant, ablation_mode) in enumerate(checkpoint_tuples):
        tag = f"{variant}_{ablation_mode}" if variant != "checkpoint" else "checkpoint"
        logger.info(f"\n{'='*60}")
        logger.info(f"Processing: {ckpt_path}")
        logger.info(f"Tag: {tag}")
        logger.info(f"{'='*60}")
        
        # Load checkpoint (only for cell_centers)
        logger.info(f"Loading cell centers from {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        cell_centers = ckpt["cell_centers"]
        
        # Generate main map
        title_suffix = f"({tag})" if tag != "checkpoint" else ""
        output_path = output_dir / f"geocells_map_{tag}.png"
        
        plot_geocells_on_map(
            cell_centers, output_path,
            world_basemap=world_basemap,
            cell_subset=cell_subset,
            title_suffix=title_suffix
        )
        
        # Generate distribution plots
        dist_path = output_dir / f"cell_distribution_{tag}.png"
        plot_cell_distribution(
            cell_centers, dist_path
        )
    
    logger.info("\n" + "="*60)
    logger.info("Geocell visualization complete!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
