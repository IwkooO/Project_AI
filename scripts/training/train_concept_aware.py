#!/usr/bin/env python3
"""
Training script for Concept-Aware Global Image-GPS Alignment.
"""

import argparse
import logging
import os
import csv
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image as PILImage
from datetime import datetime
import pandas as pd
from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified,
    get_transforms_from_processor,
)
from src.iwo_dataset import CBMDataset
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.losses import (
    contrastive_alignment_loss,
    concept_divergence_loss,
    coordinate_loss,
)
from src.concepts.utils import extract_concepts_from_dataset
from src.evaluation import (
    denormalize_coordinates,
    haversine_distance,
    sphere_to_latlng,
    accuracy_within_threshold,
)
from sklearn.cluster import KMeans

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Constants ----------
ENCODER_MODEL_TO_NAME = {
    "geolocal/StreetCLIP": "streetclip",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3",
    "facebook/dinov2-base": "dinov2",
}


# ---------- Helper Functions ----------
def generate_semantic_geocells(dataset, min_samples_per_cell=500, output_dir=None):
    """
    Generate semantic geocells using Admin-like clustering.
    Algorithm: For each country, if samples > min_samples_per_cell, run K-Means to split it.
    Args:
        dataset: Dataset to generate geocells from
        min_samples_per_cell: Minimum number of samples per cell
        output_dir: Optional directory to save visualization files. If None, saves to current directory.
    Returns:
        - cell_centers: Tensor [N_cells, 3] (Cartesian)
        - sample_to_cell: Tensor [Total_Samples] mapping each sample index to cell ID
    """
    logger.info("Generating Semantic Geocells (Per-Country Clustering)...")

    # Collect all data
    all_coords = []  # (lat, lng)
    all_countries = []

    # Iterate dataset to gather metadata (this might be slow for huge datasets, but okay for 43k)
    # CBMDataset stores samples in .samples list
    if hasattr(dataset, "samples"):
        samples = dataset.samples
    elif hasattr(dataset, "data"):
        # Fallback for raw CBMDataset if samples not exposed (but it is in our code)
        samples = dataset.samples
    else:
        raise ValueError("Dataset format not recognized for cell generation")

    for s in samples:
        all_coords.append([s["lat"], s["lng"]])
        all_countries.append(s["country"])

    all_coords = np.array(all_coords)
    all_countries = np.array(all_countries)

    unique_countries = np.unique(all_countries)

    cell_centers_list = []
    sample_to_cell_map = np.zeros(len(samples), dtype=int)

    current_cell_id_offset = 0

    for country in tqdm(unique_countries, desc="Clustering Countries"):
        country_mask = all_countries == country
        country_indices = np.where(country_mask)[0]
        country_coords = all_coords[country_indices]

        n_samples = len(country_coords)

        if n_samples > min_samples_per_cell:
            # Determine K for this country
            # E.g., 1 cell per 500 samples
            k = max(1, n_samples // min_samples_per_cell)

            # Perform K-Means on 3D sphere coords to avoid pole issues?
            # For simplicity on local regions, Lat/Lng K-Means is usually 'okay' but Cartesian is better.
            # Let's convert to Cartesian for clustering
            lat_rad = np.deg2rad(country_coords[:, 0])
            lng_rad = np.deg2rad(country_coords[:, 1])
            x = np.cos(lat_rad) * np.cos(lng_rad)
            y = np.cos(lat_rad) * np.sin(lng_rad)
            z = np.sin(lat_rad)
            cart_coords = np.stack([x, y, z], axis=1)

            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cart_coords)

            # Cluster centers are in Cartesian, need to be stored
            centers = kmeans.cluster_centers_
            # Normalize centers to unit sphere
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)

            # Assign cell IDs
            local_labels = kmeans.labels_
            global_labels = local_labels + current_cell_id_offset

            sample_to_cell_map[country_indices] = global_labels

            # Add centers
            for center in centers:
                cell_centers_list.append(center)

            current_cell_id_offset += k

        else:
            # Country is too small, treat as single cell
            # Compute mean center
            lat_rad = np.deg2rad(country_coords[:, 0])
            lng_rad = np.deg2rad(country_coords[:, 1])
            x = np.cos(lat_rad) * np.cos(lng_rad)
            y = np.cos(lat_rad) * np.sin(lng_rad)
            z = np.sin(lat_rad)

            mean_x = np.mean(x)
            mean_y = np.mean(y)
            mean_z = np.mean(z)

            center = np.array([mean_x, mean_y, mean_z])
            center = center / np.linalg.norm(center)

            cell_centers_list.append(center)

            sample_to_cell_map[country_indices] = current_cell_id_offset
            current_cell_id_offset += 1

    cell_centers = torch.tensor(np.stack(cell_centers_list), dtype=torch.float32)
    sample_to_cell = torch.tensor(sample_to_cell_map, dtype=torch.long)

    logger.info(f"Generated {len(cell_centers)} Semantic Geocells.")

    # Visualize Geocells on a Geographic Map (Matplotlib only)
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Rectangle

        # Convert cell centers to lat/lng for plotting
        cx, cy, cz = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
        clat = np.rad2deg(np.arcsin(cz.numpy()))
        clng = np.rad2deg(np.arctan2(cy.numpy(), cx.numpy()))

        # Determine output path
        if output_dir is not None:
            output_path = Path(output_dir)
            output_path.mkdir(parents=True, exist_ok=True)
            png_path = output_path / "geocells_map.png"
        else:
            png_path = "geocells_map.png"

        # Create figure with world map background
        fig, ax = plt.subplots(figsize=(16, 9))
        
        # Subsample samples for visualization (max 10000 points)
        max_samples_viz = min(len(all_coords), 10000)
        if len(all_coords) > max_samples_viz:
            indices = np.random.choice(len(all_coords), size=max_samples_viz, replace=False)
            viz_lats = all_coords[indices, 0]
            viz_lngs = all_coords[indices, 1]
            viz_cells = sample_to_cell_map[indices]
        else:
            viz_lats = all_coords[:, 0]
            viz_lngs = all_coords[:, 1]
            viz_cells = sample_to_cell_map

        # Plot samples colored by cell ID
        scatter = ax.scatter(
            viz_lngs, viz_lats, c=viz_cells, s=2, alpha=0.5, 
            cmap='tab20', edgecolors='none'
        )

        # Plot cell centers as red stars
        ax.scatter(clng, clat, c='red', s=100, marker='*', 
                  edgecolors='black', linewidths=0.5, label='Cell Centers', zorder=10)

        # Add colorbar for cell IDs
        cbar = plt.colorbar(scatter, ax=ax, label='Cell ID')
        
        # Set world map bounds
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        
        # Add gridlines
        ax.grid(True, alpha=0.3, linestyle='--')
        ax.set_xlabel('Longitude', fontsize=12)
        ax.set_ylabel('Latitude', fontsize=12)
        ax.set_title(f'Semantic Geocells Distribution (K={len(cell_centers)} cells)', fontsize=14, fontweight='bold')
        ax.legend(loc='upper right', fontsize=10)

        # Add world map outline (simple rectangle)
        world_rect = Rectangle((-180, -90), 360, 180, 
                               fill=False, edgecolor='black', linewidth=1.5)
        ax.add_patch(world_rect)

        plt.tight_layout()
        plt.savefig(str(png_path), dpi=150, bbox_inches='tight')
        plt.close()
        logger.info(f"Saved geocell visualization to {png_path}")

    except Exception as e:
        logger.error(f"Failed to visualize geocells: {e}")
        import traceback
        logger.error(traceback.format_exc())

    return cell_centers, sample_to_cell


def collate_batch(batch):
    """
    Custom collate function to handle variable-length metadata fields.
    Metadata dict contains 'images' which is a list of variable length.
    """
    images = torch.stack([item[0] for item in batch])
    concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    coordinates = torch.stack([item[3] for item in batch])
    metadata = [item[4] for item in batch]  # Keep as list of dicts, don't collate

    return images, concept_indices, target_indices, coordinates, metadata


@torch.no_grad()
def visualize_predictions(
    model,
    val_loader,
    concept_names,
    idx_to_country,
    device,
    args,
    checkpoint_dir,
    epoch,
    cell_centers,
    num_samples=4,
):
    """
    Visualize top predicted concepts and in-batch location retrieval for validation samples.
    Creates a single combined chart for all samples, saves it to disk, and logs to wandb if enabled.
    Self-matches are masked out (diagonal of similarity matrix set to -inf) to ensure
    proper retrieval evaluation. Only uses validation set (unseen during training).
    """
    model.eval()
    logger.info(
        f"\n=== Visualizing Predictions (Top 5 Concepts & In-Batch Retrieval) ==="
    )

    # Create visualization directory
    viz_dir = checkpoint_dir / "visualizations" / f"epoch_{epoch}"
    viz_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch
    wandb_images = []

    for batch in val_loader:
        images, concept_indices, _, coords, metadata, cell_labels, note_embs = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_indices = concept_indices.to(device)

        # Forward pass
        outputs = model(images, coords)
        z_img = outputs["z_img"]
        z_loc = outputs.get("z_loc")
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]

        # Reconstruct predicted coordinates
        pred_cells = cell_logits.argmax(dim=1)
        batch_cell_centers = cell_centers[pred_cells] # [B, 3]

        if model.coord_output_dim == 3:
            # Prediction is in 3D Cartesian space
            pred_cart = batch_cell_centers + pred_offsets
            pred_cart = torch.nn.functional.normalize(pred_cart, p=2, dim=1)
            pred_coords = sphere_to_latlng(pred_cart) # [B, 2]
        else:
            # Prediction is 2D Lat/Lng offset
            c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
            c_lat = torch.rad2deg(torch.asin(c_z))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
            pred_coords = batch_cell_latlng + pred_offsets

        # In-Batch Retrieval Similarity
        z_img_norm = torch.nn.functional.normalize(z_img, p=2, dim=1)
        z_loc_norm = torch.nn.functional.normalize(z_loc, p=2, dim=1)
        similarity = torch.matmul(z_img_norm, z_loc_norm.t())

        # Mask out diagonal (self-matches) by setting to -inf
        # This prevents the model from matching an image with its own location
        mask = torch.eye(len(images), device=similarity.device, dtype=torch.bool)
        similarity_masked = similarity.masked_fill(mask, float("-inf"))

        # Create single figure for all samples
        # Height per sample = 4 inches, Width = 15 inches
        n_display = min(len(images), num_samples)

        for i in range(n_display):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))

            # Find best match (diagonal is masked, so no self-matches possible)
            best_loc_idx = similarity_masked[i].argmax().item()

            ax_img = axes[0]
            ax_bar = axes[1]

            # 1. Image & Concepts
            probs = z_img[i]
            top_scores, top_indices = torch.topk(probs, k=5)

            # Denormalize image for display
            # StreetCLIP uses CLIP mean/std. We need to inverse normalize.
            # Mean: (0.481, 0.457, 0.408), Std: (0.268, 0.261, 0.275)
            img_cpu = images[i].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img_disp = std * img_cpu + mean
            img_disp = np.clip(img_disp, 0, 1)

            # Top panel: Image + Retrieval Info
            ax_img.imshow(img_disp)
            ax_img.axis("off")

            # Retrieval Info
            gt_lat = metadata[i]["lat"]
            gt_lng = metadata[i]["lng"]
            gt_country = metadata[i]["country"]

            # Get predicted coordinates from the regression head
            pred_coords_raw = pred_coords[i].detach().cpu()

            # Handle 3D sphere coordinates -> 2D lat/lng
            if pred_coords_raw.shape[0] == 3:
                # sphere_to_latlng expects [N, 3], so unsqueeze
                pred_coords_deg = sphere_to_latlng(
                    pred_coords_raw.unsqueeze(0)
                ).squeeze(0)
                pred_lat = pred_coords_deg[0].item()
                pred_lng = pred_coords_deg[1].item()
            else:
                # 2D output (assumed raw degrees)
                pred_coords_np = pred_coords_raw.numpy()
                pred_lat = pred_coords_np[0]
                pred_lng = pred_coords_np[1]

            # Calculate Haversine distance
            # We need to use the same distance function as training/eval
            # Convert to tensors for haversine_distance
            gt_coord_tensor = coords[i].unsqueeze(0)

            # Ensure pred_coord_tensor is appropriate for haversine_distance
            if pred_coords_raw.shape[0] == 3:
                pred_coord_tensor = torch.tensor(
                    [pred_lat, pred_lng], device=device
                ).unsqueeze(0)
            else:
                pred_coord_tensor = pred_coords[i].unsqueeze(0)

            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()

            # Classification Prediction
            pred_country_idx = country_logits[i].argmax().item()
            pred_country_cls = idx_to_country[pred_country_idx]

            # GT Concept
            gt_concept_idx = concept_indices[i].item()
            gt_concept_name = concept_names[gt_concept_idx]

            # Top predicted concept
            top_concept_idx = top_indices[0].item()
            top_concept_name = concept_names[top_concept_idx]

            # Calculate probabilities for display
            country_prob = torch.softmax(country_logits[i], dim=0)[
                pred_country_idx
            ].item()
            gt_country_idx_val = -1
            # This is inefficient but safe way to find index by value if idx_to_country is dict
            for k, v in idx_to_country.items():
                if v == gt_country:
                    gt_country_idx_val = k
                    break
            gt_country_prob = 0.0
            if gt_country_idx_val != -1:
                gt_country_prob = torch.softmax(country_logits[i], dim=0)[
                    gt_country_idx_val
                ].item()

            # title = f"GT: {gt_country} ({gt_lat:.2f}, {gt_lng:.2f}) | Concept: {gt_concept_name}\n"
            # title += f"Pred: {pred_country_cls} ({country_prob:.2f}) | ({pred_lat:.2f}, {pred_lng:.2f})\n"
            # title += f"Error: {distance_km:.1f} km | Top Concept: {top_concept_name}"
            title = f"Epoch {epoch} | Image ID: {metadata[i]['pano_id']}\n"
            title += f"Country: Pred: {pred_country_cls} | True: {gt_country}\n"
            title += f"Coords: Pred({pred_lat:.3f}, {pred_lng:.3f}) | True({gt_lat:.3f}, {gt_lng:.3f})\n"
            title += f"Concept: GT: {gt_concept_name} | Pred: {top_concept_name}\n"
            title += f"Dist Error: {distance_km:.1f} km"

            ax_img.set_title(title, fontsize=10)

            # Bottom panel: Bar Chart of Concepts
            scores_np = top_scores.cpu().numpy()
            concepts_np = [concept_names[idx.item()] for idx in top_indices]
            top_indices_cpu = top_indices.cpu().numpy()

            # Color logic: Orange if GT, else SteelBlue
            bar_colors = [
                "orange" if idx == gt_concept_idx else "steelblue"
                for idx in top_indices_cpu
            ]

            y_pos = np.arange(len(concepts_np))

            ax_bar.barh(y_pos, scores_np, align="center", color=bar_colors)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(concepts_np)
            ax_bar.invert_yaxis()  # labels read top-to-bottom
            ax_bar.set_xlabel("Activation Score")

            # Check if GT concept is in top 5
            gt_in_top5 = gt_concept_idx in top_indices_cpu
            bar_title = "Top 5 Predicted Concepts"
            if not gt_in_top5:
                bar_title += f" | GT: {gt_concept_name}"
            ax_bar.set_title(bar_title)

            plt.tight_layout()

            # Save to disk as one image per sample
            save_path = viz_dir / f"sample_{i}.png"
            plt.savefig(save_path, dpi=150, bbox_inches="tight")

            if args.use_wandb:
                wandb_images.append(
                    wandb.Image(str(save_path), caption=f"Epoch {epoch} Sample {i}")
                )

            plt.close(fig)
            logger.info(f"Saved visualization for sample {i} to {save_path}")

        break  # Only one batch

    # Log all images to wandb in a single key for grid view
    if args.use_wandb and wandb_images:
        wandb.log({f"predictions/epoch_{epoch}": wandb_images})


@torch.no_grad()
def dump_diagnostics(
    model,
    dataloader,
    device,
    output_path: Path,
    concept_names: List[str],
    idx_to_country: Dict[int, str],
    cell_centers: torch.Tensor,
    max_samples: int = 64,
    log_to_wandb: bool = False,
    wandb_step: Optional[int] = None,
):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for batch in dataloader:
        if len(rows) >= max_samples:
            break

        images, concept_idx, target_idx, coords, metadata, cell_labels, note_embs = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        target_idx = target_idx.to(device)

        # Forward pass
        outputs = model(images, coords)
        z_img = outputs["z_img"]
        z_loc = outputs.get("z_loc")
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]

        # Reconstruct coordinates
        pred_cells = cell_logits.argmax(dim=1)
        batch_cell_centers = cell_centers[pred_cells]

        if model.coord_output_dim == 3:
            pred_cart = batch_cell_centers + pred_offsets
            pred_cart = torch.nn.functional.normalize(pred_cart, p=2, dim=1)
            pred_coords = sphere_to_latlng(pred_cart)
        else:
            c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
            c_lat = torch.rad2deg(torch.asin(c_z))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
            pred_coords = batch_cell_latlng + pred_offsets

        # Concept probabilities
        concept_probs = torch.softmax(z_img, dim=1)

        # Country probabilities
        country_probs = torch.softmax(country_logits, dim=1)

        for i in range(len(images)):
            if len(rows) >= max_samples:
                break

            # Concept info
            top_concept_idx = concept_probs[i].argmax().item()
            top_concept = concept_names[top_concept_idx]
            true_concept_idx_val = concept_idx[i].item()
            true_concept = concept_names[true_concept_idx_val]

            # Country info
            pred_country_idx = country_probs[i].argmax().item()
            pred_country_cls = idx_to_country[pred_country_idx]
            true_country_idx_val = target_idx[i].item()
            true_country_cls = idx_to_country[true_country_idx_val]

            # Location info (regression based)
            gt_lat = metadata[i]["lat"]
            gt_lng = metadata[i]["lng"]

            # Get coordinate prediction from regression head
            pred_coords_raw = pred_coords[i].detach().cpu()
            if pred_coords_raw.shape[0] == 3:
                pred_coords_deg = sphere_to_latlng(
                    pred_coords_raw.unsqueeze(0)
                ).squeeze(0)
                pred_lat = pred_coords_deg[0].item()
                pred_lng = pred_coords_deg[1].item()
            else:
                pred_coords_np = pred_coords_raw.numpy()
                pred_lat = pred_coords_np[0]
                pred_lng = pred_coords_np[1]

            # Calculate distance
            gt_coord_tensor = coords[i].unsqueeze(0)
            if pred_coords_raw.shape[0] == 3:
                pred_coord_tensor = torch.tensor(
                    [pred_lat, pred_lng], device=device
                ).unsqueeze(0)
            else:
                pred_coord_tensor = pred_coords[i].unsqueeze(0)

            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()

            # Build row dict - only include pano_id if it exists in metadata
            row = {
                "pred_lat": float(pred_lat),
                "pred_lng": float(pred_lng),
                "true_lat": float(gt_lat),
                "true_lng": float(gt_lng),
                "distance_km": distance_km,
                "top_concept": top_concept,
                "true_concept": true_concept,
                "pred_country_cls": pred_country_cls,
                "true_country_cls": true_country_cls,
                "concept_correct": bool(top_concept_idx == true_concept_idx_val),
                "country_correct": bool(pred_country_idx == true_country_idx_val),
            }

            # Only add pano_id if it exists in metadata
            if "pano_id" in metadata[i]:
                row["pano_id"] = metadata[i]["pano_id"]

            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    logger.info(f"Dumped {len(rows)} diagnostic samples to {output_path}")

    if log_to_wandb:
        table = wandb.Table(columns=fieldnames)
        for row in rows:
            table.add_data(*[row[col] for col in fieldnames])
        wandb.log({f"diagnostics/{output_path.stem}": table}, step=wandb_step)


def create_checkpoint_dir(
    encoder_model: str = None,
    country_filter: str = None,
    coordinate_loss_type: str = "haversine",
) -> Path:
    """Create checkpoint directory with format:
    results/concept-aware/<encoder_name>/<country>/<loss_type>/<timestamp>/
    where:
    - encoder_name: sanitized encoder model name (replace "/" with "-")
    - country: country name or "global" if no country filter
    - loss_type: coordinate loss type (haversine, mse, sphere, etc.)
    - timestamp: formatted date and time (YYYY-MM-DD_HH-MM-SS)

    Creates subdirectories: checkpoints/, logs/, visualizations/, diagnostics/
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Encoder name: sanitize model name (replace "/" with "-")
    if encoder_model:
        encoder_name = ENCODER_MODEL_TO_NAME.get(encoder_model, encoder_model)
        encoder_name = encoder_name.replace("/", "-")
    else:
        encoder_name = "unknown"

    # Country: use filter or "global" if None
    country = country_filter if country_filter else "global"

    # Loss type: use provided coordinate loss type
    loss_type = coordinate_loss_type.lower()

    # Create full directory structure
    timestamp_dir = (
        Path("results")
        / "concept-aware"
        / encoder_name
        / country
        / loss_type
        / timestamp
    )
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (timestamp_dir / "checkpoints").mkdir(exist_ok=True)
    (timestamp_dir / "logs").mkdir(exist_ok=True)
    (timestamp_dir / "visualizations").mkdir(exist_ok=True)
    (timestamp_dir / "diagnostics").mkdir(exist_ok=True)

    return timestamp_dir


# ---------- Main Training Function ----------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ---------- Setup Data & Image Encoder ----------
    logger.info("Initializing Dataset...")

    # Use StreetCLIP transforms
    base_encoder = StreetCLIPEncoder(StreetCLIPConfig(model_name=args.encoder_model))
    transforms = get_transforms_from_processor(base_encoder.image_processor)

    # Dataset to use
    # full_dataset = PanoramaCBMDataset(
    #         transform=transforms,
    #         require_coordinates=True,
    #         country=args.country_filter,
    #         use_normalized_coordinates=False,
    #         geoguessr_id=args.geoguessr_id,
    #         data_root=args.data_root
    #     )
    full_dataset = CBMDataset(
        dataframe=pd.read_csv("/scratch-shared/pnair/Project_AI/data/dataset-43k.csv"),
        transform=transforms,
        encoder_model=args.encoder_model,
        country=args.country_filter,
    )

    # Diagnostic: Check concept distribution
    all_concepts = [s["meta_name"] for s in full_dataset.samples]
    concept_counts = Counter(all_concepts)
    logger.info(
        f"Total samples: {len(full_dataset.samples)}, Concepts: {len(concept_counts)}"
    )
    logger.info(
        f"Samples per concept - Min: {min(concept_counts.values())}, Max: {max(concept_counts.values())}, Avg: {len(full_dataset.samples)/len(concept_counts):.1f}"
    )
    logger.info("Concept distribution (Top 10):")
    for name, count in concept_counts.most_common(10):
        logger.info(f"  {name}: {count}")

    # Use more balanced split: 70/20/10 to get more validation samples
    # With only 251 samples, 10% validation gives only ~21 samples which is too small
    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )

    # ---------- Compute Class Weights for Concept Imbalance ----------
    def compute_concept_weights(train_samples, concept_to_idx, device):
        """
        Compute class weights using inverse frequency weighting.
        Formula: weight[i] = total_samples / (num_concepts * count[i])
        This gives rare concepts higher weight, common concepts lower weight.
        
        This function works for ANY distribution:
        - If concept has 1 sample: weight = total_samples / num_concepts (highest)
        - If concept has many samples: weight approaches 0 (lowest)
        - Normalized so weights sum to num_concepts (maintains loss scale)
        """
        concept_counts = Counter(s['meta_name'] for s in train_samples)
        num_concepts = len(concept_to_idx)
        total_samples = len(train_samples)
        
        weights = torch.ones(num_concepts, device=device)
        for concept_name, idx in concept_to_idx.items():
            count = concept_counts.get(concept_name, 1)  # Avoid division by zero
            weights[idx] = total_samples / (num_concepts * count)
        
        # Normalize so weights sum to num_concepts (keeps loss scale similar)
        weights = weights * (num_concepts / weights.sum())
        
        return weights

    # Compute concept weights if enabled
    concept_weights = None
    if args.use_class_weights:
        concept_weights = compute_concept_weights(train_samples, full_dataset.concept_to_idx, device)
        logger.info(f"Computed concept weights - Min: {concept_weights.min():.4f}, "
                   f"Max: {concept_weights.max():.4f}, Mean: {concept_weights.mean():.4f}, "
                   f"Std: {concept_weights.std():.4f}")
    else:
        logger.info("Class weights disabled - using uniform weighting")

    # Create subset datasets (need to implement wrapper or just list sampling)
    # Re-using the logic from dataset.py main block
    from src.dataset import SubsetDataset

    train_dataset = SubsetDataset(full_dataset, train_samples)
    val_dataset = SubsetDataset(full_dataset, val_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Important for contrastive loss stability
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    test_dataset = SubsetDataset(full_dataset, test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    logger.info(
        f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}"
    )
    logger.info(
        f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}"
    )

    # Check train concept distribution
    train_concepts = [s["meta_name"] for s in train_samples]
    train_concept_counts = Counter(train_concepts)
    logger.info(
        f"Train set: {len(train_concepts)} samples across {len(train_concept_counts)} concepts"
    )

    # Precompute Note Embeddings for Training
    logger.info("Pre-computing Note Embeddings for Semantic Alignment...")
    all_notes = [s["note"] for s in full_dataset.samples]

    note_embeddings_list = []
    with torch.no_grad():
        for i in range(0, len(all_notes), 32):
            batch_notes = all_notes[i : i + 32]
            # Replace empty notes with " " to avoid tokenizer errors
            batch_notes = [n if n and n.strip() else " " for n in batch_notes]
            feats = base_encoder.get_text_features(text=batch_notes)
            note_embeddings_list.append(feats.cpu())
    all_note_embeddings = torch.cat(note_embeddings_list, dim=0)

    # Inject into samples
    for i, sample in enumerate(full_dataset.samples):
        sample["note_embedding"] = all_note_embeddings[i]

    # Re-define collate to handle cells and note embeddings
    def collate_batch_v2(batch):
        images = torch.stack([item[0] for item in batch])
        concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
        target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        coordinates = torch.stack([item[3] for item in batch])
        metadata = [item[4] for item in batch]

        cell_labels = []
        note_embs = []

        for m in metadata:
            pid = m["pano_id"]
            cell_labels.append(pano_to_cell[pid])
            note_embs.append(pano_to_note_emb[pid])

        cell_labels = torch.tensor(cell_labels, dtype=torch.long)
        note_embs = torch.stack(note_embs)

        return (
            images,
            concept_indices,
            target_indices,
            coordinates,
            metadata,
            cell_labels,
            note_embs,
        )

    # Build lookup maps
    pano_to_note_emb = {s["pano_id"]: s["note_embedding"] for s in full_dataset.samples}
    # pano_to_cell already built

    # Update DataLoaders to use v2 collate
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch_v2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch_v2,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch_v2,
    )

    logger.info(
        f"  Samples per concept - Min: {min(train_concept_counts.values())}, Max: {max(train_concept_counts.values())}, Avg: {len(train_concepts)/len(train_concept_counts):.1f}"
    )

    # ---------- Setup Output Directory ----------
    if args.output_dir is None:
        checkpoint_dir = create_checkpoint_dir(
            encoder_model=args.encoder_model,
            country_filter=args.country_filter,
            coordinate_loss_type=args.coordinate_loss_type,
        )
    else:
        checkpoint_dir = Path(args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Ensure subdirectories exist
        (checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
        (checkpoint_dir / "logs").mkdir(exist_ok=True)
        (checkpoint_dir / "visualizations").mkdir(exist_ok=True)
        (checkpoint_dir / "diagnostics").mkdir(exist_ok=True)

    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Setup file logging
    log_file = checkpoint_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # ---------- Initialize WandB ----------
    if args.use_wandb:
        # Create tags
        tags = [
            args.encoder_model,
            args.country_filter if args.country_filter else "global",
            "finetuned" if args.finetune_encoder else "frozen",
            "concept_aware",
            args.coordinate_loss_type,
        ]

        wandb.init(
            project="concept-aware-geolocation",
            name=f"concept-aware-geolocation-{args.country_filter if args.country_filter else 'global'}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=args,
            tags=tags,
            dir=str(checkpoint_dir),
        )

    # ---------- Semantic Geocell Generation ----------
    viz_dir = checkpoint_dir / "visualizations"
    cell_centers, sample_to_cell = generate_semantic_geocells(
        full_dataset, min_samples_per_cell=args.min_samples_per_cell, output_dir=viz_dir
    )

    # Log geocell map to wandb if enabled
    geocell_png_path = viz_dir / "geocells_map.png"
    if args.use_wandb and geocell_png_path.exists():
        wandb.log(
            {
                "geocells_map": wandb.Image(
                    str(geocell_png_path), caption="Semantic Geocells Distribution"
                )
            }
        )

    cell_centers = cell_centers.to(device)
    num_cells = len(cell_centers)

    # Assign cell labels to datasets
    # We need to be careful mapping back to train/val/test splits
    # SubsetDataset doesn't easily support adding new attributes, so we'll create a lookup tensor/dict
    # sample_to_cell is aligned with full_dataset.samples

    # We need to pass these cell labels into the batch.
    # Update collate_batch or wrapper to fetch cell_label?
    # Easiest way: Inject cell_label into the sample dict in full_dataset
    for i, sample in enumerate(full_dataset.samples):
        sample["cell_label"] = sample_to_cell[i].item()

    # Re-define collate_batch to handle cell_label
    def collate_batch_with_cells(batch):
        images = torch.stack([item[0] for item in batch])
        concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
        target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        coordinates = torch.stack([item[3] for item in batch])
        metadata = [item[4] for item in batch]

        # Extract cell labels from metadata (since we injected it into samples, it should be in metadata if dataset returns it?)
        # Wait, Dataset.__getitem__ returns metadata dict. Let's check if it includes everything from sample.
        # CBMDataset and PanoramaCBMDataset construct metadata dict explicitly. We need to update them or hack it here.
        # Since we modified full_dataset.samples in memory, let's check if __getitem__ uses that.
        # Yes, __getitem__ pulls from self.samples[idx].
        # BUT, __getitem__ constructs a NEW metadata dict explicitly selecting fields.
        # So 'cell_label' won't be in the returned metadata dict unless we patch __getitem__.

        # Patching __getitem__ on the fly is messy.
        # Better approach: The collate function receives the result of __getitem__.
        # We can look up cell_label using the sample index? No, indices are local to batch.

        # Alternative: Pass sample_to_cell tensor to the training loop and look up using global indices?
        # But we don't have global indices in the batch.

        # Let's inject 'cell_label' into the metadata dict returned by __getitem__
        # We can monkey-patch the dataset class or just wrap the dataset.

        # Let's rely on 'pano_id' to look up cell label if we build a map.
        # metadata contains 'pano_id'.
        cell_labels = []
        for m in metadata:
            # We can rely on the fact that we updated full_dataset.samples
            # We need a quick lookup pano_id -> cell_label
            pass

        # Actually, let's just modify the metadata dict in the batch since we have 'cell_label' in full_dataset.samples?
        # No, we can't access full_dataset from here easily without global scope.

        return images, concept_indices, target_indices, coordinates, metadata

    # Create pano_id -> cell_label map for O(1) lookup during training
    pano_to_cell = {s["pano_id"]: s["cell_label"] for s in full_dataset.samples}

    # ---------- Concept Extraction & Encoding ----------
    logger.info("Extracting and encoding concepts...")
    # We need concepts from the ENTIRE dataset to build the basis, not just training set
    concept_names, concept_map = extract_concepts_from_dataset(full_dataset)

    logger.info(f"Found {len(concept_names)} unique concepts.")

    # Verify concept alignment
    # concept_names should match keys in full_dataset.concept_to_idx (both sorted by name)
    dataset_concepts = sorted(full_dataset.concept_to_idx.keys())
    if concept_names != dataset_concepts:
        logger.error(
            "Concept mismatch between extract_concepts_from_dataset and dataset.concept_to_idx!"
        )
        logger.error(f"Extract: {concept_names[:5]}...")
        logger.error(f"Dataset: {dataset_concepts[:5]}...")
        raise ValueError("Concept alignment failed")

    # Further verification: check index mapping
    for i, name in enumerate(concept_names):
        idx = full_dataset.concept_to_idx[name]
        if idx != i:
            raise ValueError(f"Index mismatch for {name}: expected {i}, got {idx}")
    logger.info("Concept alignment verified.")

    # Encode concepts to get E_concept
    # Move encoder to device for inference - ensure model is explicitly moved
    base_encoder.model.to(device)
    base_encoder = base_encoder.to(device)  # Also move any other components
    # Verify model is on the correct device (compare device types, not exact strings)
    model_device = next(base_encoder.model.parameters()).device
    if model_device.type != device.type:
        raise RuntimeError(
            f"Model is on {model_device.type} but expected {device.type}"
        )
    logger.info(f"Base encoder device: {model_device}")

    # Batch encode concepts to avoid OOM if many concepts
    concept_texts = [concept_map[name] for name in concept_names]
    concept_embeddings = []

    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(concept_texts), batch_size):
            batch_texts = concept_texts[i : i + batch_size]
            feats = base_encoder.get_text_features(batch_texts)
            concept_embeddings.append(feats.cpu())

    E_concept = torch.cat(concept_embeddings, dim=0)  # [k, d]
    logger.info(f"Concept Basis shape: {E_concept.shape}")

    # ---------- Initialize Concept-Aware Model ----------
    logger.info("Initializing ConceptAwareGeoModel...")
    # Configure StreetCLIP for training (finetune=False usually for encoder)
    encoder_config = StreetCLIPConfig(
        model_name=args.encoder_model, finetune=args.finetune_encoder, device=device
    )
    # Re-create encoder to ensure clean state / config application
    image_encoder = StreetCLIPEncoder(encoder_config)
    actual_feature_dim = image_encoder.feature_dim
    logger.info(f"StreetCLIP vision encoder dimension: {actual_feature_dim}")

    # Project concept embeddings to match vision encoder dimension if needed
    if E_concept.shape[1] != actual_feature_dim:
        logger.warning(
            f"Dimension mismatch: Concept embeddings have dim {E_concept.shape[1]}, "
            f"but vision encoder has dim {actual_feature_dim}. "
            f"Projecting concept embeddings to match vision dimension..."
        )
        # Project concept embeddings to match vision dimension
        projection = nn.Linear(E_concept.shape[1], actual_feature_dim, bias=False).to(
            device
        )
        # Initialize with small random values to preserve semantic information
        nn.init.normal_(projection.weight, mean=0.0, std=0.02)

        with torch.no_grad():
            E_concept_projected = []
            for i in range(0, len(E_concept), batch_size):
                batch = E_concept[i : i + batch_size].to(device)
                projected = projection(batch)
                E_concept_projected.append(projected.cpu())
            E_concept = torch.cat(E_concept_projected, dim=0)
        logger.info(f"Projected Concept Basis shape: {E_concept.shape}")

    # Final verification
    assert (
        E_concept.shape[1] == actual_feature_dim
    ), f"Concept basis dimension {E_concept.shape[1]} must match vision encoder dimension {actual_feature_dim}"

    # Determine coordinate output dimension
    coord_output_dim = 3 if args.coordinate_loss_type == "sphere" else 2
    logger.info(f"Coordinate output dimension: {coord_output_dim}")

    model = ConceptAwareGeoModel(
        image_encoder=image_encoder,
        concept_features=E_concept,
        num_concepts=len(concept_names),
        num_countries=len(full_dataset.country_to_idx),
        num_cells=num_cells,  # Semantic Cells
        streetclip_dim=actual_feature_dim,
        location_encoder_dim=512,  # GeoCLIP default
        coord_output_dim=coord_output_dim,
        text_encoder=base_encoder,  # Pass frozen encoder for semantic alignment
    )
    model.to(device)

    # ---------- Optimizer & Scheduler ----------
    # Differential learning rates: Higher LR for concept head
    concept_lr = args.lr * args.concept_lr_multiplier
    other_lr = args.lr
    
    # Separate parameter groups
    concept_params = []
    other_params = []
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'image_projector' in name:
                concept_params.append(param)
            else:
                other_params.append(param)
    
    param_groups = [
        {'params': concept_params, 'lr': concept_lr},
        {'params': other_params, 'lr': other_lr}
    ]
    
    optimizer = torch.optim.AdamW(
        param_groups, weight_decay=args.weight_decay
    )
    logger.info(f"Using differential learning rates: concept_head={concept_lr:.2e}, others={other_lr:.2e}")

    # Initialize AMP scaler
    scaler = torch.amp.GradScaler('cuda', enabled=args.use_amp)
    logger.info(f"AMP enabled: {args.use_amp}")
    logger.info(f"Gradient accumulation steps: {args.gradient_accumulation_steps}")

    # Learning rate scheduler: Warmup + Cosine Annealing
    warmup_epochs = max(1, int(args.epochs * args.warmup_ratio))
    from torch.optim.lr_scheduler import LinearLR, CosineAnnealingLR, SequentialLR
    
    warmup_scheduler = LinearLR(
        optimizer, start_factor=0.1, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer, T_max=args.epochs - warmup_epochs, eta_min=args.lr * 0.01
    )
    scheduler = SequentialLR(
        optimizer, schedulers=[warmup_scheduler, cosine_scheduler], milestones=[warmup_epochs]
    )
    logger.info(f"Using warmup ({warmup_epochs} epochs) + cosine annealing scheduler")

    # ---------- Training Loop ----------
    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_contrastive = 0
        total_divergence = 0
        total_concept_loss = 0
        total_country_loss = 0
        total_coords_loss = 0

        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_concept_correct = 0
        total_concept_count = 0
        total_country_correct = 0
        total_country_count = 0

        for batch in pbar:
            # ---------- Forward Pass ----------
            # Updated unpack to include note_embeddings and cell_labels
            images, concept_idx, target_idx, coords, _, cell_labels, note_embs = batch

            images = images.to(device)
            coords = coords.to(device)
            concept_idx = concept_idx.to(device)
            target_idx = target_idx.to(device)
            cell_labels = cell_labels.to(device)
            note_embs = note_embs.to(device)

            # AMP Context
            with torch.amp.autocast('cuda', enabled=args.use_amp):
                # Model returns dict now
                outputs = model(images, coords)
                z_img = outputs["z_img"]
                z_loc = outputs.get(
                    "z_loc"
                )  # May be None if coords not passed (but they are)
                country_logits = outputs["country_logits"]
                cell_logits = outputs["cell_logits"]
                pred_offsets = outputs["pred_offsets"]

                # ---------- Compute Metrics ----------
                pred_concepts = z_img.argmax(dim=1)
                concept_correct = (pred_concepts == concept_idx).sum().item()
                total_concept_correct += concept_correct
                total_concept_count += len(concept_idx)

                pred_countries = country_logits.argmax(dim=1)
                country_correct = (pred_countries == target_idx).sum().item()
                total_country_correct += country_correct
                total_country_count += len(target_idx)

                # ---------- Compute Loss ----------
                # 1. Contrastive Loss (Image vs Location Concept alignment)
                loss_contrastive = contrastive_alignment_loss(
                    z_img, z_loc, temperature=args.temperature
                )

                # 2. Concept Divergence Loss
                loss_divergence = concept_divergence_loss(
                    z_img, z_loc, sigma=args.sigma
                )

                # 3. Concept Classification Loss
                loss_concept = nn.functional.cross_entropy(
                    z_img, concept_idx, 
                    weight=concept_weights if args.use_class_weights else None,
                    label_smoothing=args.label_smoothing
                )

                # 4. Country Classification Loss (Auxiliary)
                loss_country = nn.functional.cross_entropy(
                    country_logits, target_idx, label_smoothing=args.label_smoothing
                )

                # 5. Semantic Reconstruction Loss (Neuro-Symbolic)
                concept_probs = torch.softmax(z_img, dim=1)
                basis = model.get_concept_basis().t()  # [k, d]
                pred_note_embs = torch.matmul(concept_probs, basis)

                # Normalize for Cosine Distance (MSE on normalized vectors)
                # Fix tensor dimension mismatch for loss_semantic
                if pred_note_embs.shape[1] != note_embs.shape[1]:
                    # Naive fix: Project to matching dimension or slice/pad
                    if pred_note_embs.shape[1] > note_embs.shape[1]:
                        pred_note_shared = pred_note_embs[:, : note_embs.shape[1]]
                    else:
                        # Pad with zeros
                        padding = torch.zeros(
                            pred_note_embs.shape[0],
                            note_embs.shape[1] - pred_note_embs.shape[1],
                            device=device,
                        )
                        pred_note_shared = torch.cat([pred_note_embs, padding], dim=1)
                else:
                    pred_note_shared = pred_note_embs

                pred_note_norm = torch.nn.functional.normalize(
                    pred_note_shared, p=2, dim=1
                )
                target_note_norm = torch.nn.functional.normalize(note_embs, p=2, dim=1)
                loss_semantic = nn.functional.mse_loss(pred_note_norm, target_note_norm)

                # 6. Cell Classification Loss (Coarse Location)
                loss_cell = nn.functional.cross_entropy(
                    cell_logits, cell_labels, label_smoothing=args.label_smoothing
                )

                # 7. Offset Regression Loss (Fine Location)
                batch_cell_centers = cell_centers[cell_labels]  # [B, 3]

                # 3D vs 2D offsets
                if model.coord_output_dim == 3:
                    # Convert True Coords (Lat/Lng) to Cartesian
                    from src.dataset import latlon_to_cartesian

                    lat_rad = torch.deg2rad(coords[:, 0])
                    lng_rad = torch.deg2rad(coords[:, 1])
                    x = torch.cos(lat_rad) * torch.cos(lng_rad)
                    y = torch.cos(lat_rad) * torch.sin(lng_rad)
                    z = torch.sin(lat_rad)
                    true_cart = torch.stack([x, y, z], dim=1)  # [B, 3]

                    target_offsets = true_cart - batch_cell_centers
                    loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
                else:
                    # 2D Lat/Lng offsets
                    # Convert cell centers to Lat/Lng
                    c_x, c_y, c_z = (
                        batch_cell_centers[:, 0],
                        batch_cell_centers[:, 1],
                        batch_cell_centers[:, 2],
                    )
                    c_lat = torch.rad2deg(torch.asin(c_z))
                    c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
                    batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)

                    if args.coordinate_loss_type == "haversine":
                        # Haversine loss on (Cell + Pred_Offset) vs True
                        pred_latlng = batch_cell_latlng + pred_offsets
                        loss_offset = coordinate_loss(
                            pred_latlng, coords, loss_type="haversine"
                        )
                    else:
                        target_offsets = coords - batch_cell_latlng
                        # Handle wraparound
                        target_offsets[:, 1] = (target_offsets[:, 1] + 180) % 360 - 180
                        loss_offset = nn.functional.mse_loss(
                            pred_offsets, target_offsets
                        )

                # Total Loss
                loss = (
                    args.lambda_contrastive * loss_contrastive
                    + args.lambda_divergence * loss_divergence
                    + args.lambda_concept * loss_concept
                    + args.lambda_country * loss_country
                    + args.lambda_semantic * loss_semantic
                    + args.lambda_cell * loss_cell
                    + args.lambda_offset * loss_offset
                )

                # Scale loss for gradient accumulation
                loss = loss / args.gradient_accumulation_steps

            # ---------- Backward Pass ----------
            # Use scaler for backward
            scaler.scale(loss).backward()

            if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            # ---------- Batch Logging ----------
            total_loss += (
                loss.item() * args.gradient_accumulation_steps
            )  # Rescale for logging
            total_contrastive += loss_contrastive.item()
            total_divergence += loss_divergence.item()
            total_concept_loss += loss_concept.item()
            total_country_loss += loss_country.item()

            batch_concept_acc = concept_correct / len(concept_idx)
            batch_country_acc = country_correct / len(target_idx)

            pbar.set_postfix(
                {
                    "loss": f"{loss.item():.4f}",
                    "sem": f"{loss_semantic.item():.4f}",
                    "cell": f"{loss_cell.item():.4f}",
                    "off": f"{loss_offset.item():.4f}",
                    "concept_acc": f"{batch_concept_acc:.3f}",
                }
            )

            if args.use_wandb:
                wandb.log(
                    {
                        "batch_loss": loss.item(),
                        "batch_contrastive": loss_contrastive.item(),
                        "batch_divergence": loss_divergence.item(),
                        "batch_concept_loss": loss_concept.item(),
                        "batch_country_loss": loss_country.item(),
                        "batch_semantic_loss": loss_semantic.item(),
                        "batch_cell_loss": loss_cell.item(),
                        "batch_offset_loss": loss_offset.item(),
                        "batch_concept_accuracy": batch_concept_acc,
                        "batch_country_accuracy": batch_country_acc,
                    }
                )

        # ---------- Epoch Validation ----------
        avg_train_loss = total_loss / len(train_loader)
        avg_train_concept_loss = total_concept_loss / len(train_loader)
        train_concept_acc = (
            total_concept_correct / total_concept_count
            if total_concept_count > 0
            else 0.0
        )
        train_country_acc = (
            total_country_correct / total_country_count
            if total_country_count > 0
            else 0.0
        )

        logger.info(
            f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Concept Acc: {train_concept_acc:.4f}, Country Acc: {train_country_acc:.4f}"
        )

        if args.use_wandb:
            wandb.log(
                {
                    "train_loss": avg_train_loss,
                    "train_concept_loss": avg_train_concept_loss,
                    "train_concept_accuracy": train_concept_acc,
                    "train_country_accuracy": train_country_acc,
                }
            )

        # Need to update validate function signature to accept cell info or handle it
        val_metrics = validate(model, val_loader, device, args, cell_centers, concept_weights)
        val_concept_acc = val_metrics["concept_acc"]

        # ---------- Learning Rate Scheduling & Early Stopping ----------
        scheduler.step()  # Cosine scheduler doesn't need loss value
        if val_concept_acc > best_val_acc:
            best_val_acc = val_concept_acc
            patience_counter = 0
            # Save best model
            best_model_path = checkpoint_dir / "checkpoints" / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Concept Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if (
                args.early_stopping_patience > 0
                and patience_counter >= args.early_stopping_patience
            ):
                logger.info(
                    f"Early stopping triggered after {epoch+1} epochs. Best Val Acc: {best_val_acc:.4f}"
                )
                break

        # ---------- Visualization & Diagnostics ----------
        if (epoch + 1) % args.save_interval == 0:
            visualize_predictions(
                model,
                val_loader,
                concept_names,
                full_dataset.idx_to_country,
                device,
                args,
                checkpoint_dir,
                epoch + 1,
                cell_centers=cell_centers,
            )

            # Diagnostics
            diag_path = checkpoint_dir / "diagnostics" / f"epoch_{epoch+1}.csv"
            dump_diagnostics(
                model,
                val_loader,
                device,
                diag_path,
                concept_names,
                full_dataset.idx_to_country,
                cell_centers=cell_centers,
                log_to_wandb=args.use_wandb,
                wandb_step=epoch + 1,
            )

        # ---------- Save Checkpoint ----------
        if (epoch + 1) % args.save_interval == 0:
            save_path = (
                checkpoint_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
            )
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # ---------- Final Test Evaluation ----------
    logger.info("Evaluating on test set...")
    # Load best model if saved
    best_model_path = checkpoint_dir / "checkpoints" / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best model for testing.")

    test_metrics = validate(model, test_loader, device, args, cell_centers)
    logger.info(f"Test Metrics: {test_metrics}")

    if args.use_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})

        # Dump test diagnostics
        test_diag_path = checkpoint_dir / "diagnostics" / "test_diagnostics.csv"
        dump_diagnostics(
            model,
            test_loader,
            device,
            test_diag_path,
            concept_names,
            full_dataset.idx_to_country,
            cell_centers=cell_centers,
            max_samples=len(test_dataset),  # Dump all test samples
            log_to_wandb=True,
        )


# ---------- Validation Function ----------
@torch.no_grad()
def validate(model, val_loader, device, args, cell_centers, concept_weights=None):
    model.eval()
    total_loss = 0
    total_concept_correct = 0
    total_concept_count = 0
    total_country_correct = 0
    total_country_count = 0

    # New metrics
    total_cell_correct = 0

    # Collect all distances for threshold accuracy computation
    all_distances = []

    for batch in val_loader:
        images, concept_idx, target_idx, coords, _, cell_labels, note_embs = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        target_idx = target_idx.to(device)
        cell_labels = cell_labels.to(device)
        note_embs = note_embs.to(device)

        outputs = model(images, coords)
        z_img = outputs["z_img"]
        z_loc = outputs.get("z_loc")
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]

        # Compute concept accuracy
        pred_concepts = z_img.argmax(dim=1)
        concept_correct = (pred_concepts == concept_idx).sum().item()
        total_concept_correct += concept_correct
        total_concept_count += len(concept_idx)

        # Compute country accuracy
        pred_countries = country_logits.argmax(dim=1)
        country_correct = (pred_countries == target_idx).sum().item()
        total_country_correct += country_correct
        total_country_count += len(target_idx)

        # Compute cell accuracy
        pred_cells = cell_logits.argmax(dim=1)
        cell_correct = (pred_cells == cell_labels).sum().item()
        total_cell_correct += cell_correct

        # Compute Final Coordinate Prediction & Distance
        # Pred = Cell_Center[Pred_Cell] + Pred_Offset
        # We rely on predicted cell, not ground truth cell for validation metric!
        # 1. Get predicted cell centers
        pred_cell_centers = cell_centers[pred_cells]  # [B, 3]

        # 2. Get predicted Lat/Lng
        if model.coord_output_dim == 3:
            # Prediction is in 3D Cartesian space
            pred_cart = pred_cell_centers + pred_offsets
            # Convert to Lat/Lng for haversine
            # Normalize first to be on sphere? Or just assume noisy offset?
            # Ideally normalize.
            pred_cart = torch.nn.functional.normalize(pred_cart, p=2, dim=1)
            pred_latlng_deg = sphere_to_latlng(pred_cart)  # [B, 2]
            pred_coords = pred_latlng_deg
        else:
            # Prediction is 2D Lat/Lng offset
            # Convert cell center to Lat/Lng
            c_x, c_y, c_z = (
                pred_cell_centers[:, 0],
                pred_cell_centers[:, 1],
                pred_cell_centers[:, 2],
            )
            c_lat = torch.rad2deg(torch.asin(c_z))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            pred_cell_latlng = torch.stack([c_lat, c_lng], dim=1)

            pred_coords = pred_cell_latlng + pred_offsets
            # Normalize Lng to [-180, 180]? Haversine usually handles it but cleaner to normalize.

        # Compute distances for threshold accuracy
        batch_distances = haversine_distance(pred_coords, coords)
        if batch_distances.numel() > 0:
            all_distances.append(batch_distances.cpu())

        # Compute Loss (Validation)
        # ... (Mirror training loss logic but no backward)
        # For brevity, just summing main components for monitoring 'loss'
        loss_concept = nn.functional.cross_entropy(
            z_img, concept_idx, 
            weight=concept_weights if args.use_class_weights else None,
            label_smoothing=args.label_smoothing
        )
        loss_country = nn.functional.cross_entropy(
            country_logits, target_idx, label_smoothing=args.label_smoothing
        )
        loss_cell = nn.functional.cross_entropy(
            cell_logits, cell_labels, label_smoothing=args.label_smoothing
        )

        # Semantic loss
        concept_probs = torch.softmax(z_img, dim=1)
        basis = model.get_concept_basis().t()
        pred_note_embs = torch.matmul(concept_probs, basis)

        # Fix tensor dimension mismatch for validation loss_semantic
        if pred_note_embs.shape[1] != note_embs.shape[1]:
            if pred_note_embs.shape[1] > note_embs.shape[1]:
                pred_note_shared = pred_note_embs[:, : note_embs.shape[1]]
            else:
                padding = torch.zeros(
                    pred_note_embs.shape[0],
                    note_embs.shape[1] - pred_note_embs.shape[1],
                    device=device,
                )
                pred_note_shared = torch.cat([pred_note_embs, padding], dim=1)
        else:
            pred_note_shared = pred_note_embs

        pred_note_norm = torch.nn.functional.normalize(pred_note_shared, p=2, dim=1)
        target_note_norm = torch.nn.functional.normalize(note_embs, p=2, dim=1)
        loss_semantic = nn.functional.mse_loss(pred_note_norm, target_note_norm)

        loss = (
            args.lambda_concept * loss_concept
            + args.lambda_country * loss_country
            + args.lambda_cell * loss_cell
            + args.lambda_semantic * loss_semantic
        )

        total_loss += loss.item()

    # Compute threshold accuracies
    threshold_accuracies = {}
    if all_distances:
        all_distances_tensor = torch.cat(all_distances)
        thresholds = {
            "street": 1.0,  # 1 km
            "city": 25.0,  # 25 km
            "region": 200.0,  # 200 km
            "country": 750.0,  # 750 km
            "continent": 2500.0,  # 2500 km
        }
        for level, threshold_km in thresholds.items():
            acc = accuracy_within_threshold(all_distances_tensor, threshold_km)
            threshold_accuracies[f"acc_{level}"] = acc

        median_error = torch.median(all_distances_tensor).item()
        threshold_accuracies["median_error_km"] = median_error
    else:
        threshold_accuracies = {
            "acc_street": 0.0,
            "acc_city": 0.0,
            "acc_region": 0.0,
            "acc_country": 0.0,
            "acc_continent": 0.0,
            "median_error_km": 0.0,
        }

    avg_loss = total_loss / len(val_loader)
    val_concept_acc = (
        total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
    )
    val_country_acc = (
        total_country_correct / total_country_count if total_country_count > 0 else 0.0
    )
    val_cell_acc = (
        total_cell_correct / total_concept_count if total_concept_count > 0 else 0.0
    )  # Denom same as batch size

    # Log threshold accuracies
    log_msg = f"Validation Loss: {avg_loss:.4f}, Val Concept Acc: {val_concept_acc:.4f}, Val Country Acc: {val_country_acc:.4f}, Val Cell Acc: {val_cell_acc:.4f}"
    log_msg += f", Median Error: {threshold_accuracies['median_error_km']:.1f}km"
    logger.info(log_msg)
    logger.info(
        f"Thresholds: Street={threshold_accuracies['acc_street']:.3f}, City={threshold_accuracies['acc_city']:.3f}, Region={threshold_accuracies['acc_region']:.3f}"
    )

    if args.use_wandb:
        wandb.log(
            {
                "val_loss": avg_loss,
                "val_concept_accuracy": val_concept_acc,
                "val_country_accuracy": val_country_acc,
                "val_cell_accuracy": val_cell_acc,
                "val_median_error_km": threshold_accuracies["median_error_km"],
                "val_acc_street_1km": threshold_accuracies["acc_street"],
                "val_acc_city_25km": threshold_accuracies["acc_city"],
                "val_acc_region_200km": threshold_accuracies["acc_region"],
                "val_acc_country_750km": threshold_accuracies["acc_country"],
                "val_acc_continent_2500km": threshold_accuracies["acc_continent"],
            }
        )

    return {
        "loss": avg_loss,
        "concept_acc": val_concept_acc,
        "country_acc": val_country_acc,
        "cell_acc": val_cell_acc,
        **threshold_accuracies,
    }


# ---------- Main Entry Point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Concept-Aware CBM")

    # ---------- Dataset Arguments ----------
    parser.add_argument(
        "--geoguessr_id",
        type=str,
        default="6906237dc7731161a37282b2",
        help="Geoguessr ID",
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="Data root directory"
    )

    # ---------- Data & Model Arguments ----------
    parser.add_argument(
        "--encoder_model",
        type=str,
        default="geolocal/StreetCLIP",
        help="Image Encoder model to use",
    )
    parser.add_argument(
        "--finetune_encoder",
        action="store_true",
        help="Whether to finetune the encoder",
    )
    parser.add_argument(
        "--country_filter", type=str, default=None, help="Filter for country"
    )

    # ---------- Training Hyperparameters ----------
    parser.add_argument(
        "--batch_size", type=int, default=32, help="Batch size for training"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs to train"
    )
    parser.add_argument(
        "--lr", type=float, default=1e-4, help="Learning rate for optimizer"
    )
    parser.add_argument(
        "--concept_lr_multiplier",
        type=float,
        default=3.0,
        help="Multiplier for concept head learning rate (default: 3.0)",
    )
    parser.add_argument(
        "--warmup_ratio",
        type=float,
        default=0.1,
        help="Fraction of epochs for learning rate warmup (default: 0.1)",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=0.1, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=5,
        help="Number of epochs to wait before early stopping",
    )
    parser.add_argument(
        "--gradient_accumulation_steps",
        type=int,
        default=1,
        help="Number of update steps to accumulate before performing a backward/update pass",
    )
    parser.add_argument(
        "--use_amp",
        action="store_true",
        help="Whether to use Automatic Mixed Precision (AMP)",
    )

    parser.add_argument(
        "--min_samples_per_cell",
        type=int,
        default=500,
        help="Minimum number of samples to form a semantic geocell",
    )

    # ---------- Loss Hyperparameters ----------
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.07,
        help="Temperature for contrastive alignment loss",
    )
    parser.add_argument(
        "--lambda_semantic",
        type=float,
        default=1.0,
        help="Weight for semantic reconstruction loss",
    )
    parser.add_argument(
        "--lambda_divergence",
        type=float,
        default=0.1,
        help="Weight for concept divergence loss",
    )
    parser.add_argument(
        "--lambda_concept", type=float, default=10.0, help="Weight for concept loss"
    )
    parser.add_argument(
        "--lambda_contrastive", type=float, default=0.1, help="Weight for contrastive loss"
    )
    parser.add_argument(
        "--lambda_country",
        type=float,
        default=0.1,
        help="Weight for country loss (deprecated/auxiliary)",
    )
    parser.add_argument(
        "--lambda_cell",
        type=float,
        default=1.0,
        help="Weight for cell classification loss",
    )
    parser.add_argument(
        "--lambda_offset",
        type=float,
        default=10.0,
        help="Weight for offset regression loss",
    )
    parser.add_argument(
        "--sigma", type=float, default=1.0, help="Sigma for concept divergence loss"
    )
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument(
        "--use_class_weights",
        action="store_true",
        help="Use class weights for concept loss to handle imbalance (default: True)",
    )
    parser.add_argument(
        "--no_class_weights",
        dest="use_class_weights",
        action="store_false",
        help="Disable class weights for concept loss",
    )
    parser.set_defaults(use_class_weights=True)
    parser.add_argument(
        "--coordinate_loss_type",
        type=str,
        default="haversine",
        choices=["haversine", "mse", "sphere"],
        help="Type of coordinate loss: haversine, mse, or sphere",
    )

    # ---------- Miscellaneous Arguments ----------
    parser.add_argument(
        "--output_dir",
        type=str,
        default=None,
        help="Output directory (auto-generated if not provided)",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=5,
        help="When to save checkpoints and run visualizations and diagnostics",
    )
    parser.add_argument(
        "--use_wandb",
        action="store_true",
        default=True,
        help="Whether to use Weights & Biases for logging",
    )

    args = parser.parse_args()
    train(args)
