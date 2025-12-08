#!/usr/bin/env python3
"""
Training script for Concept-Aware Global Image-GPS Alignment.
With optional embedding precomputation for faster training.
"""

import argparse
import logging
import csv
from typing import Dict, List, Optional
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from datetime import datetime
from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified,
    get_transforms_from_processor,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.losses import (
    coordinate_loss,
    clip_contrastive_loss,
    geocell_contrastive_loss,
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

# Visualization and formatting constants
CLIP_MEAN = np.array([0.48145466, 0.4578275, 0.40821073])
CLIP_STD = np.array([0.26862954, 0.26130258, 0.27577711])
VIZ_DPI = 150
VIZ_FIGSIZE = (16, 12)
VIZ_TOP_K = 5
THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
    "continent": 2500.0,
}


# ---------- Helper Functions ----------
def save_checkpoint(
    model,
    checkpoint_path: Path,
    cell_centers: torch.Tensor,
    concept_names: List[str],
    country_to_idx: Dict[str, int],
    idx_to_country: Dict[int, str],
    encoder_model: str,
    extra_info: Optional[Dict] = None,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
):
    """Save model checkpoint with all metadata needed for inference and resumption."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cell_centers": cell_centers.cpu(),
        "num_cells": len(cell_centers),
        "concept_names": concept_names,
        "num_concepts": len(concept_names),
        "country_to_idx": country_to_idx,
        "idx_to_country": idx_to_country,
        "num_countries": len(country_to_idx),
        "coord_output_dim": model.coord_output_dim,
        "encoder_model": encoder_model,
    }
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if extra_info:
        checkpoint.update(extra_info)
    torch.save(checkpoint, checkpoint_path)


def load_checkpoint(checkpoint_path: Path, device: torch.device = None):
    """Load model checkpoint with all metadata."""
    checkpoint = torch.load(checkpoint_path, map_location=device)
    if device is not None and "cell_centers" in checkpoint:
        checkpoint["cell_centers"] = checkpoint["cell_centers"].to(device)
    return checkpoint


def format_coordinate(lat: float, lng: float, precision: int = 3) -> str:
    """Format coordinates for display."""
    return f"({lat:.{precision}f}, {lng:.{precision}f})"


def format_distance(km: float, precision: int = 1) -> str:
    """Format distance for display."""
    if km < 1:
        return f"{km*1000:.0f}m"
    elif km < 1000:
        return f"{km:.{precision}f}km"
    else:
        return f"{km/1000:.2f}Mm"


def get_stage_summary(stage: int, epoch: int, total_epochs: int) -> str:
    """Get formatted stage summary string."""
    return f"[Stage {stage}] Epoch {epoch}/{total_epochs}"


def log_metrics(metrics: Dict[str, float], prefix: str = "", stage: int = None):
    """Log metrics in a structured format."""
    stage_str = f"[Stage {stage}] " if stage is not None else ""
    prefix_str = f"{prefix}_" if prefix else ""

    main_metrics = []
    if "loss" in metrics:
        main_metrics.append(f"Loss: {metrics['loss']:.4f}")
    if "concept_acc" in metrics:
        main_metrics.append(f"Concept Acc: {metrics['concept_acc']:.3f}")
    if "country_acc" in metrics:
        main_metrics.append(f"Country Acc: {metrics['country_acc']:.3f}")
    if "cell_acc" in metrics:
        main_metrics.append(f"Cell Acc: {metrics['cell_acc']:.3f}")
    if "median_error_km" in metrics:
        main_metrics.append(f"Median Error: {format_distance(metrics['median_error_km'])}")

    if main_metrics:
        logger.info(f"{stage_str}{prefix_str}{' | '.join(main_metrics)}")

    if any(k.startswith("acc_") for k in metrics.keys()):
        thresholds = []
        for level in ["street", "city", "region", "country", "continent"]:
            key = f"acc_{level}"
            if key in metrics:
                thresholds.append(f"{level.capitalize()}: {metrics[key]:.3f}")
        if thresholds:
            logger.info(f"  Thresholds: {' | '.join(thresholds)}")


def generate_semantic_geocells(dataset, min_samples_per_cell=500, output_dir=None):
    """Generate semantic geocells using per-country K-Means clustering."""
    logger.info("Generating Semantic Geocells...")

    all_coords = []
    all_countries = []

    if hasattr(dataset, "samples"):
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

        lat_rad = np.deg2rad(country_coords[:, 0])
        lng_rad = np.deg2rad(country_coords[:, 1])
        x = np.cos(lat_rad) * np.cos(lng_rad)
        y = np.cos(lat_rad) * np.sin(lng_rad)
        z = np.sin(lat_rad)

        if n_samples > min_samples_per_cell:
            k = max(1, n_samples // min_samples_per_cell)
            cart_coords = np.stack([x, y, z], axis=1)
            kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
            kmeans.fit(cart_coords)
            centers = kmeans.cluster_centers_
            centers = centers / np.linalg.norm(centers, axis=1, keepdims=True)
            global_labels = kmeans.labels_ + current_cell_id_offset
            sample_to_cell_map[country_indices] = global_labels
            for center in centers:
                cell_centers_list.append(center)
            current_cell_id_offset += k
        else:
            center = np.array([np.mean(x), np.mean(y), np.mean(z)])
            center = center / np.linalg.norm(center)
            cell_centers_list.append(center)
            sample_to_cell_map[country_indices] = current_cell_id_offset
            current_cell_id_offset += 1

    cell_centers = torch.tensor(np.stack(cell_centers_list), dtype=torch.float32)
    sample_to_cell = torch.tensor(sample_to_cell_map, dtype=torch.long)
    logger.info(f"Generated {len(cell_centers)} Semantic Geocells.")

    # Visualization
    if output_dir:
        from matplotlib.patches import Rectangle

        cx, cy, cz = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
        clat = np.rad2deg(np.arcsin(cz.numpy()))
        clng = np.rad2deg(np.arctan2(cy.numpy(), cx.numpy()))

        png_path = Path(output_dir) / "geocells_map.png"
        Path(output_dir).mkdir(parents=True, exist_ok=True)

        fig, ax = plt.subplots(figsize=(16, 9))
        max_viz = min(len(all_coords), 10000)
        indices = (
            np.random.choice(len(all_coords), size=max_viz, replace=False)
            if len(all_coords) > max_viz
            else np.arange(len(all_coords))
        )
        scatter = ax.scatter(
            all_coords[indices, 1],
            all_coords[indices, 0],
            c=sample_to_cell_map[indices],
            s=2,
            alpha=0.5,
            cmap="tab20",
        )
        ax.scatter(clng, clat, c="red", s=100, marker="*", edgecolors="black", linewidths=0.5, label="Cell Centers", zorder=10)
        plt.colorbar(scatter, ax=ax, label="Cell ID")
        ax.set_xlim([-180, 180])
        ax.set_ylim([-90, 90])
        ax.grid(True, alpha=0.3, linestyle="--")
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")
        ax.set_title(f"Semantic Geocells (K={len(cell_centers)})")
        ax.add_patch(Rectangle((-180, -90), 360, 180, fill=False, edgecolor="black", linewidth=1.5))
        plt.savefig(str(png_path), dpi=VIZ_DPI, bbox_inches="tight")
        plt.close()
        logger.info(f"Saved geocell visualization to {png_path}")

    return cell_centers, sample_to_cell


def collate_batch(batch):
    """Custom collate for variable-length metadata."""
    images = torch.stack([item[0] for item in batch])
    concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    coordinates = torch.stack([item[3] for item in batch])
    metadata = [item[4] for item in batch]
    return images, concept_indices, target_indices, coordinates, metadata


# ---------- Precomputed Embeddings Dataset ----------
class PrecomputedEmbeddingsDataset(torch.utils.data.Dataset):
    """Dataset that wraps precomputed image embeddings for faster training."""

    def __init__(
        self,
        embeddings: torch.Tensor,
        concept_indices: torch.Tensor,
        country_indices: torch.Tensor,
        coordinates: torch.Tensor,
        cell_labels: torch.Tensor,
        metadata: List[Dict],
    ):
        self.embeddings = embeddings
        self.concept_indices = concept_indices
        self.country_indices = country_indices
        self.coordinates = coordinates
        self.cell_labels = cell_labels
        self.metadata = metadata

    def __len__(self):
        return len(self.embeddings)

    def __getitem__(self, idx):
        return (
            self.embeddings[idx],
            self.concept_indices[idx],
            self.country_indices[idx],
            self.coordinates[idx],
            self.metadata[idx],
            self.cell_labels[idx],
        )


def collate_precomputed_batch(batch):
    """Collate function for precomputed embeddings dataset."""
    embeddings = torch.stack([item[0] for item in batch])
    # Use torch.tensor to ensure correct dtype (matching original collate_batch behavior)
    concept_indices = torch.tensor([item[1].item() if isinstance(item[1], torch.Tensor) else item[1] for item in batch], dtype=torch.long)
    country_indices = torch.tensor([item[2].item() if isinstance(item[2], torch.Tensor) else item[2] for item in batch], dtype=torch.long)
    coordinates = torch.stack([item[3] for item in batch])
    metadata = [item[4] for item in batch]
    cell_labels = torch.tensor([item[5].item() if isinstance(item[5], torch.Tensor) else item[5] for item in batch], dtype=torch.long)
    return embeddings, concept_indices, country_indices, coordinates, metadata, cell_labels


@torch.no_grad()
def precompute_image_embeddings(
    model,
    dataloader,
    device,
    desc: str = "Precomputing embeddings",
) -> PrecomputedEmbeddingsDataset:
    """
    Precompute image embeddings for all samples using the frozen image encoder.
    
    Uses eval() mode for deterministic embeddings since:
    1. The encoder is frozen after Stage 0 - no dropout during actual training either
    2. BatchNorm should use running statistics for consistent embeddings
    3. Embeddings must be identical whether computed once or on-the-fly
    """
    model.eval()  # Use eval mode for consistent, deterministic embeddings
    
    all_embeddings = []
    all_concept_indices = []
    all_country_indices = []
    all_coordinates = []
    all_cell_labels = []
    all_metadata = []

    for batch in tqdm(dataloader, desc=desc):
        images, concept_idx, country_idx, coords, metadata, cell_labels = batch
        images = images.to(device)

        # Extract features using frozen image encoder (eval mode for deterministic output)
        features = model.image_encoder(images)  # [B, 768]

        all_embeddings.append(features.cpu())
        all_concept_indices.append(concept_idx)
        all_country_indices.append(country_idx)
        all_coordinates.append(coords)
        all_cell_labels.append(cell_labels)
        all_metadata.extend(metadata)

    # Concatenate all batches
    embeddings = torch.cat(all_embeddings, dim=0)
    concept_indices = torch.cat(all_concept_indices, dim=0)
    country_indices = torch.cat(all_country_indices, dim=0)
    coordinates = torch.cat(all_coordinates, dim=0)
    cell_labels = torch.cat(all_cell_labels, dim=0)

    logger.info(f"Precomputed {len(embeddings)} embeddings of shape {embeddings.shape}")

    return PrecomputedEmbeddingsDataset(
        embeddings=embeddings,
        concept_indices=concept_indices,
        country_indices=country_indices,
        coordinates=coordinates,
        cell_labels=cell_labels,
        metadata=all_metadata,
    )


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
    stage: int = None,
):
    """Simplified visualization: image with title + Top 5 concept predictions."""
    model.eval()
    viz_dir = checkpoint_dir / "visualizations" / f"epoch_{epoch}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    wandb_images = []

    for batch in val_loader:
        images, concept_indices, _, coords, metadata, cell_labels = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_indices = concept_indices.to(device)

        outputs = model(images, coords)
        concept_logits = outputs["concept_logits"]
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        concept_probs = torch.softmax(concept_logits, dim=1)
        country_probs = torch.softmax(country_logits, dim=1)

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
            pred_coords = torch.stack([c_lat, c_lng], dim=1) + pred_offsets

        n_display = min(len(images), num_samples)

        for i in range(n_display):
            fig, (ax_img, ax_concept) = plt.subplots(2, 1, figsize=(10, 8), gridspec_kw={"height_ratios": [2, 1]})

            img_cpu = images[i].cpu().permute(1, 2, 0).numpy()
            img_disp = np.clip(CLIP_STD * img_cpu + CLIP_MEAN, 0, 1)
            ax_img.imshow(img_disp)
            ax_img.axis("off")

            gt_lat, gt_lng, gt_country = metadata[i]["lat"], metadata[i]["lng"], metadata[i]["country"]
            pred_lat, pred_lng = pred_coords[i, 0].item(), pred_coords[i, 1].item()

            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = pred_coords[i].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()

            pred_country_idx = country_logits[i].argmax().item()
            pred_country = idx_to_country[pred_country_idx]
            country_confidence = country_probs[i, pred_country_idx].item()

            gt_concept_idx = concept_indices[i].item()
            gt_concept_name = concept_names[gt_concept_idx]
            top_concept_idx = concept_probs[i].argmax().item()
            top_concept_name = concept_names[top_concept_idx]
            concept_confidence = concept_probs[i, top_concept_idx].item()

            title_parts = [
                f"Epoch {epoch} | Stage {stage}" if stage is not None else f"Epoch {epoch}",
                f"ID: {metadata[i].get('pano_id', 'N/A')}",
                "",
                f"[Location]",
                f"  Pred: {format_coordinate(pred_lat, pred_lng)} ({format_distance(distance_km)})",
                f"  True: {format_coordinate(gt_lat, gt_lng)}",
                "",
                f"[Country]",
                f"  Pred: {pred_country} ({country_confidence:.2%})",
                f"  True: {gt_country}",
                "",
                f"[Concept]",
                f"  Pred: {top_concept_name} ({concept_confidence:.2%})",
                f"  True: {gt_concept_name}",
            ]
            ax_img.set_title("\n".join(title_parts), fontsize=9, family="monospace", loc="left")

            top_k = min(5, len(concept_names))
            top_scores, top_indices = torch.topk(concept_probs[i], k=top_k)
            top_indices_cpu = top_indices.cpu().numpy()

            bar_colors = ["#FF8C00" if idx == gt_concept_idx else "#4169E1" for idx in top_indices_cpu]

            y_pos = np.arange(top_k)
            ax_concept.barh(y_pos, top_scores.cpu().numpy(), color=bar_colors, alpha=0.8)
            ax_concept.set_yticks(y_pos)
            ax_concept.set_yticklabels([concept_names[idx.item()] for idx in top_indices], fontsize=10)
            ax_concept.invert_yaxis()
            ax_concept.set_xlabel("Probability", fontsize=10)
            ax_concept.set_title(f"Top {top_k} Concepts", fontsize=11, fontweight="bold", pad=10)
            ax_concept.grid(axis="x", alpha=0.3, linestyle="--")
            ax_concept.set_xlim(0, 1.0)

            plt.tight_layout()

            save_path = viz_dir / f"sample_{i}_epoch_{epoch}.png"
            plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")

            if args.use_wandb:
                caption = f"Epoch {epoch} | Stage {stage} | Sample {i} | Error: {format_distance(distance_km)}"
                wandb_images.append(wandb.Image(str(save_path), caption=caption))

            plt.close(fig)

        break

    if args.use_wandb and wandb_images:
        wandb_key = f"predictions/epoch_{epoch}_stage_{stage}" if stage is not None else f"predictions/epoch_{epoch}"
        wandb.log({wandb_key: wandb_images}, step=epoch)

    logger.info(f"Saved {n_display} visualization(s) to {viz_dir}")


@torch.no_grad()
def plot_error_distribution(model, dataloader, device, cell_centers, output_path: Path, stage: int = None):
    """Plot error distribution histogram."""
    model.eval()
    all_distances = []

    for batch in dataloader:
        images, _, _, coords, _, _ = batch
        images = images.to(device)
        coords = coords.to(device)

        outputs = model(images, coords)
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]

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
            pred_coords = torch.stack([c_lat, c_lng], dim=1) + pred_offsets

        distances = haversine_distance(pred_coords, coords)
        all_distances.extend(distances.cpu().tolist())

    if not all_distances:
        logger.warning("No distances collected for error distribution plot")
        return

    fig, ax = plt.subplots(figsize=(10, 6))
    all_distances_array = np.array(all_distances)
    ax.hist(all_distances_array, bins=50, edgecolor="black", alpha=0.7)
    median_dist = np.median(all_distances_array)
    mean_dist = np.mean(all_distances_array)
    ax.axvline(median_dist, color="red", linestyle="--", label=f"Median: {format_distance(median_dist)}")
    ax.axvline(mean_dist, color="green", linestyle="--", label=f"Mean: {format_distance(mean_dist)}")
    ax.set_xlabel("Distance Error (km)", fontsize=11)
    ax.set_ylabel("Frequency", fontsize=11)
    title = f"Error Distribution"
    if stage is not None:
        title += f" - Stage {stage}"
    ax.set_title(title, fontsize=12, fontweight="bold")
    ax.legend()
    ax.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close()

    logger.info(f"Saved error distribution to {output_path}")


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
    """Dump prediction diagnostics to CSV."""
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    for batch in dataloader:
        if len(rows) >= max_samples:
            break

        images, concept_idx, target_idx, coords, metadata, cell_labels = batch
        images, coords = images.to(device), coords.to(device)
        concept_idx, target_idx = concept_idx.to(device), target_idx.to(device)

        outputs = model(images, coords)
        concept_logits, country_logits = outputs["concept_logits"], outputs["country_logits"]
        cell_logits, pred_offsets = outputs["cell_logits"], outputs["pred_offsets"]

        pred_cells = cell_logits.argmax(dim=1)
        batch_cell_centers = cell_centers[pred_cells]

        if model.coord_output_dim == 3:
            pred_cart = torch.nn.functional.normalize(batch_cell_centers + pred_offsets, p=2, dim=1)
            pred_coords = sphere_to_latlng(pred_cart)
        else:
            c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
            pred_coords = torch.stack([torch.rad2deg(torch.asin(c_z)), torch.rad2deg(torch.atan2(c_y, c_x))], dim=1) + pred_offsets

        concept_probs = torch.softmax(concept_logits, dim=1)
        country_probs = torch.softmax(country_logits, dim=1)

        for i in range(len(images)):
            if len(rows) >= max_samples:
                break

            top_concept_idx = concept_probs[i].argmax().item()
            pred_country_idx = country_probs[i].argmax().item()
            pred_coords_raw = pred_coords[i].detach().cpu()

            if pred_coords_raw.shape[0] == 3:
                pred_coords_deg = sphere_to_latlng(pred_coords_raw.unsqueeze(0)).squeeze(0)
                pred_lat, pred_lng = pred_coords_deg[0].item(), pred_coords_deg[1].item()
            else:
                pred_lat, pred_lng = pred_coords_raw.numpy()

            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = (
                torch.tensor([pred_lat, pred_lng], device=device).unsqueeze(0)
                if pred_coords_raw.shape[0] == 3
                else pred_coords[i].unsqueeze(0)
            )
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()

            row = {
                "pred_lat": float(pred_lat),
                "pred_lng": float(pred_lng),
                "true_lat": float(metadata[i]["lat"]),
                "true_lng": float(metadata[i]["lng"]),
                "distance_km": distance_km,
                "top_concept": concept_names[top_concept_idx],
                "true_concept": concept_names[concept_idx[i].item()],
                "pred_country": idx_to_country[pred_country_idx],
                "true_country": idx_to_country[target_idx[i].item()],
                "concept_correct": bool(top_concept_idx == concept_idx[i].item()),
                "country_correct": bool(pred_country_idx == target_idx[i].item()),
            }
            if "pano_id" in metadata[i]:
                row["pano_id"] = metadata[i]["pano_id"]
            rows.append(row)

    if not rows:
        return

    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    logger.info(f"Dumped {len(rows)} diagnostic samples to {output_path}")

    if log_to_wandb:
        table = wandb.Table(columns=list(rows[0].keys()))
        for row in rows:
            table.add_data(*row.values())
        wandb.log({f"diagnostics/{output_path.stem}": table}, step=wandb_step)


def create_checkpoint_dir(
    encoder_model: str = None,
    country_filter: str = None,
    coordinate_loss_type: str = "haversine",
) -> Path:
    """Create checkpoint directory with timestamp structure."""
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    if encoder_model:
        encoder_name = ENCODER_MODEL_TO_NAME.get(encoder_model, encoder_model)
        encoder_name = encoder_name.replace("/", "-")
    else:
        encoder_name = "unknown"

    country = country_filter if country_filter else "global"
    loss_type = coordinate_loss_type.lower()

    timestamp_dir = Path("results") / "concept-aware-3-stage" / encoder_name / country / loss_type / timestamp
    timestamp_dir.mkdir(parents=True, exist_ok=True)

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

    base_encoder = StreetCLIPEncoder(StreetCLIPConfig(model_name=args.encoder_model))
    base_encoder.model.to(device)
    base_encoder = base_encoder.to(device)
    transforms = get_transforms_from_processor(base_encoder.image_processor)

    full_dataset = PanoramaCBMDataset(
        transform=transforms,
        require_coordinates=True,
        country=args.country_filter,
        use_normalized_coordinates=False,
        encoder_model=args.encoder_model,
        geoguessr_id=args.geoguessr_id,
        data_root=args.data_root,
        csv_path=args.csv_path,
    )

    all_concepts = [s["meta_name"] for s in full_dataset.samples]
    concept_counts = Counter(all_concepts)
    logger.info(f"Total samples: {len(full_dataset.samples)}, Concepts: {len(concept_counts)}")
    logger.info(f"Samples per concept - Min: {min(concept_counts.values())}, Max: {max(concept_counts.values())}, Avg: {len(full_dataset.samples)/len(concept_counts):.1f}")
    logger.info("Concept distribution (Top 10):")
    for name, count in concept_counts.most_common(10):
        logger.info(f"  {name}: {count}")

    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )

    # ---------- Compute Class Weights ----------
    def compute_concept_weights(train_samples, concept_to_idx, device):
        concept_counts = Counter(s["meta_name"] for s in train_samples)
        num_concepts = len(concept_to_idx)
        total_samples = len(train_samples)

        weights = torch.ones(num_concepts, device=device)
        for concept_name, idx in concept_to_idx.items():
            count = concept_counts.get(concept_name, 1)
            weights[idx] = total_samples / (num_concepts * count)

        weights = weights * (num_concepts / weights.sum())
        return weights

    concept_weights = None
    if args.use_class_weights:
        concept_weights = compute_concept_weights(train_samples, full_dataset.concept_to_idx, device)
        logger.info(f"Computed concept weights - Min: {concept_weights.min():.4f}, Max: {concept_weights.max():.4f}, Mean: {concept_weights.mean():.4f}")
    else:
        logger.info("Class weights disabled - using uniform weighting")

    from src.dataset import SubsetDataset

    train_dataset = SubsetDataset(full_dataset, train_samples)
    val_dataset = SubsetDataset(full_dataset, val_samples)
    test_dataset = SubsetDataset(full_dataset, test_samples)

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
        collate_fn=collate_batch,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch,
    )

    logger.info(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    logger.info(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")

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
        (checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
        (checkpoint_dir / "logs").mkdir(exist_ok=True)
        (checkpoint_dir / "visualizations").mkdir(exist_ok=True)
        (checkpoint_dir / "diagnostics").mkdir(exist_ok=True)

    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    log_file = checkpoint_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s"))
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # ---------- Initialize WandB ----------
    if args.use_wandb:
        tags = [
            args.encoder_model,
            args.country_filter if args.country_filter else "global",
            "three-stage",
            args.coordinate_loss_type,
            args.csv_path if args.csv_path else args.geoguessr_id,
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

    geocell_png_path = viz_dir / "geocells_map.png"
    if args.use_wandb and geocell_png_path.exists():
        wandb.log({"geocells_map": wandb.Image(str(geocell_png_path), caption="Semantic Geocells Distribution")})

    cell_centers = cell_centers.to(device)
    num_cells = len(cell_centers)

    for i, sample in enumerate(full_dataset.samples):
        sample["cell_label"] = sample_to_cell[i].item()

    pano_to_cell = {s["pano_id"]: s["cell_label"] for s in full_dataset.samples}

    def collate_batch_v2(batch):
        images = torch.stack([item[0] for item in batch])
        concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
        target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
        coordinates = torch.stack([item[3] for item in batch])
        metadata = [item[4] for item in batch]
        cell_labels = torch.tensor([pano_to_cell[m["pano_id"]] for m in metadata], dtype=torch.long)
        return images, concept_indices, target_indices, coordinates, metadata, cell_labels

    # Update DataLoaders with cell labels
    train_loader = DataLoader(
        train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True, drop_last=True, collate_fn=collate_batch_v2
    )
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_batch_v2)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_batch_v2)

    # ---------- Concept Extraction ----------
    logger.info("Extracting and encoding concepts...")
    concept_names, concept_map = extract_concepts_from_dataset(full_dataset)
    logger.info(f"Found {len(concept_names)} unique concepts.")

    dataset_concepts = sorted(full_dataset.concept_to_idx.keys())
    if concept_names != dataset_concepts:
        raise ValueError("Concept alignment failed")

    for i, name in enumerate(concept_names):
        if full_dataset.concept_to_idx[name] != i:
            raise ValueError(f"Index mismatch for {name}")
    logger.info("Concept alignment verified.")

    model_device = next(base_encoder.model.parameters()).device
    logger.info(f"Base encoder device: {model_device}")

    # ---------- Initialize Model ----------
    logger.info("Initializing ConceptAwareGeoModel (Strict CBM Architecture)...")
    encoder_config = StreetCLIPConfig(model_name=args.encoder_model, finetune=args.finetune_encoder, device=device)
    image_encoder = StreetCLIPEncoder(encoder_config)
    actual_feature_dim = image_encoder.feature_dim
    logger.info(f"StreetCLIP vision encoder dimension: {actual_feature_dim}")

    coord_output_dim = 3 if args.coordinate_loss_type == "sphere" else 2
    logger.info(f"Coordinate output dimension: {coord_output_dim}")

    model = ConceptAwareGeoModel(
        image_encoder=image_encoder,
        num_concepts=len(concept_names),
        num_countries=len(full_dataset.country_to_idx),
        num_cells=num_cells,
        streetclip_dim=actual_feature_dim,
        concept_emb_dim=512,
        coord_output_dim=coord_output_dim,
        text_encoder=base_encoder,
    )
    model.to(device)

    # ---------- Load Checkpoint if Resuming ----------
    resume_from_stage = None
    resume_from_epoch = None
    checkpoint_optimizer_state = None
    checkpoint_scheduler_state = None

    if args.resume_from_checkpoint:
        logger.info(f"Loading checkpoint from: {args.resume_from_checkpoint}")
        checkpoint = load_checkpoint(Path(args.resume_from_checkpoint), device=device)

        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded model state from checkpoint")

        checkpoint_cell_centers = checkpoint["cell_centers"]
        if checkpoint_cell_centers.shape != cell_centers.shape:
            logger.warning(f"Cell centers shape mismatch: checkpoint {checkpoint_cell_centers.shape} vs current {cell_centers.shape}")
            cell_centers = checkpoint_cell_centers
        else:
            logger.info("Cell centers match checkpoint")

        checkpoint_stage = None
        if "extra_info" in checkpoint and checkpoint["extra_info"] and "stage" in checkpoint["extra_info"]:
            checkpoint_stage = checkpoint["extra_info"]["stage"]
        else:
            checkpoint_path = Path(args.resume_from_checkpoint)
            if "stage2" in checkpoint_path.name.lower():
                checkpoint_stage = 2
            elif "stage1" in checkpoint_path.name.lower():
                checkpoint_stage = 1
            elif "stage0" in checkpoint_path.name.lower():
                checkpoint_stage = 0
            else:
                checkpoint_stage = 1

        checkpoint_epoch = checkpoint.get("epoch", None)

        if args.resume_from_epoch is not None:
            resume_from_stage = checkpoint_stage
            resume_from_epoch = args.resume_from_epoch - 1
            logger.info(f"Resuming from Stage {resume_from_stage}, Epoch {args.resume_from_epoch}")
        elif checkpoint_epoch is not None:
            stage_epochs = [args.stage0_epochs, args.stage1_epochs, args.stage2_epochs]
            if checkpoint_stage < len(stage_epochs):
                max_epochs = stage_epochs[checkpoint_stage]
                if checkpoint_epoch >= max_epochs - 1:
                    resume_from_stage = checkpoint_stage + 1
                    resume_from_epoch = None
                    logger.info(f"Checkpoint from completed Stage {checkpoint_stage}, starting Stage {resume_from_stage}")
                else:
                    resume_from_stage = checkpoint_stage
                    resume_from_epoch = checkpoint_epoch
                    logger.info(f"Resuming from Stage {resume_from_stage}, Epoch {checkpoint_epoch + 1}")
            else:
                resume_from_stage = checkpoint_stage + 1
                resume_from_epoch = None
        else:
            resume_from_stage = checkpoint_stage + 1
            resume_from_epoch = None
            logger.info(f"Checkpoint from Stage {checkpoint_stage}, starting Stage {resume_from_stage}")

        if "optimizer_state_dict" in checkpoint:
            checkpoint_optimizer_state = checkpoint["optimizer_state_dict"]
        if "scheduler_state_dict" in checkpoint:
            checkpoint_scheduler_state = checkpoint["scheduler_state_dict"]

    # ---------- Three-Stage Training Setup ----------
    from torch.optim.lr_scheduler import CosineAnnealingLR

    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)

    logger.info(f"=== THREE-STAGE TRAINING PIPELINE ===")
    logger.info(f"Stage 0: {args.stage0_epochs} epochs - Domain Contrastive Pretraining")
    logger.info(f"Stage 1: {args.stage1_epochs} epochs - Concept Bottleneck + Global Alignment")
    logger.info(f"Stage 2: {args.stage2_epochs} epochs - Geolocation Head Training")
    total_epochs = args.stage0_epochs + args.stage1_epochs + args.stage2_epochs
    logger.info(f"Total epochs: {total_epochs}")

    # Initialize global_epoch counter
    if resume_from_epoch is not None:
        if resume_from_stage == 0:
            global_epoch = resume_from_epoch
        elif resume_from_stage == 1:
            global_epoch = args.stage0_epochs + resume_from_epoch
        elif resume_from_stage == 2:
            global_epoch = args.stage0_epochs + args.stage1_epochs + resume_from_epoch
        else:
            global_epoch = 0
    elif resume_from_stage == 1:
        global_epoch = args.stage0_epochs
    elif resume_from_stage == 2:
        global_epoch = args.stage0_epochs + args.stage1_epochs
    else:
        global_epoch = 0

    # ========================================================================
    # STAGE 0: Domain Contrastive Pretraining
    # ========================================================================
    if resume_from_stage is None or resume_from_stage == 0:
        logger.info(f"\n{'='*74}")
        logger.info(f"STAGE 0: Domain Contrastive Pretraining")
        logger.info(f"{'='*60}")

        model.image_encoder.unfreeze_top_layers(args.unfreeze_layers)
        model.image_encoder.unfreeze_text_encoder()

        # Stage 0 params: encoder + concept_bottleneck (for img-gps alignment) + location_encoder
        stage0_params = list(model.image_encoder.get_trainable_params())
        stage0_params += list(model.concept_bottleneck.parameters())
        stage0_params += list(model.location_encoder.parameters())
        num_trainable = sum(p.numel() for p in stage0_params)
        optimizer = torch.optim.AdamW(stage0_params, lr=args.stage0_lr, weight_decay=args.stage0_weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.stage0_epochs, eta_min=args.stage0_lr * 0.01)

        if resume_from_stage == 0 and checkpoint_optimizer_state is not None:
            try:
                optimizer.load_state_dict(checkpoint_optimizer_state)
            except ValueError as e:
                logger.warning(f"Could not load optimizer state (param mismatch), starting fresh: {e}")
        if resume_from_stage == 0 and checkpoint_scheduler_state is not None:
            scheduler.load_state_dict(checkpoint_scheduler_state)

        logger.info(f"Stage 0 optimizer: {num_trainable:,} params, lr={args.stage0_lr:.2e}")

        best_val_loss = float("inf")
        patience_counter = 0

        start_epoch = resume_from_epoch if (resume_from_stage == 0 and resume_from_epoch is not None) else 0

        for epoch in range(start_epoch, args.stage0_epochs):
            model.train()
            total_loss = 0
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage0_epochs} [Stage 0]")

            for batch in pbar:
                images, concept_idx, target_idx, coords, metadata, cell_labels = batch
                images = images.to(device)
                coords = coords.to(device)
                notes = [m["note"] for m in metadata]

                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    img_features = model.image_encoder(images)  # [batch, 768]
                    text_features = model.image_encoder.get_text_features_trainable(notes)  # [batch, 768]
                    gps_features = model.encode_gps(coords)  # [batch, 512]
                    
                    # Image-Text contrastive (both 768-dim)
                    img_text_loss = clip_contrastive_loss(img_features, text_features, temperature=args.temperature)
                    
                    # Image-GPS contrastive: project img to 512-dim via concept_bottleneck
                    img_features_512 = model.concept_bottleneck(img_features)  # [batch, 512]
                    img_gps_loss = clip_contrastive_loss(img_features_512, gps_features, temperature=args.temperature)
                    
                    loss = img_text_loss + img_gps_loss
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * args.gradient_accumulation_steps
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "img_text_loss": f"{img_text_loss.item():.4f}", "img_gps_loss": f"{img_gps_loss.item():.4f}"})

                if args.use_wandb:
                    wandb.log({"batch_loss": loss.item(), "stage": 0, "img_text_loss": img_text_loss.item(), "img_gps_loss": img_gps_loss.item()})

            avg_train_loss = total_loss / len(train_loader)
            logger.info(f"Epoch {epoch+1} [Stage 0] Train Loss: {avg_train_loss:.4f}")

            if args.use_wandb:
                wandb.log({"train_loss": avg_train_loss, "epoch": global_epoch + 1, "stage": 0})

            model.eval()
            val_loss = 0
            with torch.no_grad():
                for batch in val_loader:
                    images, _, _, _, metadata, _ = batch
                    images = images.to(device)
                    notes = [m["note"] for m in metadata]
                    img_features = model.image_encoder(images)
                    text_features = model.image_encoder.get_text_features(notes)
                    loss = clip_contrastive_loss(img_features, text_features, temperature=args.temperature)
                    val_loss += loss.item()

            val_loss /= len(val_loader)
            logger.info(f"Epoch {epoch+1} [Stage 0] Val Loss: {val_loss:.4f}")

            if args.use_wandb:
                wandb.log({"val_loss": val_loss})

            scheduler.step()

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage0.pt"
                save_checkpoint(model, best_model_path, cell_centers, concept_names, full_dataset.country_to_idx, full_dataset.idx_to_country, args.encoder_model, {"stage": 0, "val_loss": val_loss}, optimizer, scheduler, epoch)
                logger.info(f"Saved best Stage 0 model (Val Loss: {val_loss:.4f})")
            else:
                patience_counter += 1
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    logger.info(f"Stage 0 early stopping at epoch {epoch+1}")
                    break

            global_epoch += 1

        logger.info("Freezing image encoder and text encoder after Stage 0...")
        model.image_encoder.freeze_encoder()
        model.image_encoder.freeze_text_encoder()
    else:
        logger.info(f"Skipping Stage 0 (resuming from Stage {resume_from_stage})")
        model.image_encoder.freeze_encoder()
        model.image_encoder.freeze_text_encoder()
        if global_epoch < args.stage0_epochs:
            global_epoch = args.stage0_epochs

    # ========================================================================
    # PRECOMPUTE IMAGE EMBEDDINGS (after Stage 0, before Stage 1)
    # ========================================================================
    use_precomputed = False
    train_loader_precomputed = None
    val_loader_precomputed = None
    
    if args.precompute_embeddings:
        logger.info(f"\n{'='*74}")
        logger.info("PRECOMPUTING IMAGE EMBEDDINGS (frozen encoder, training mode for dropout consistency)")
        logger.info(f"{'='*60}")
        
        # Create loaders without shuffling for precomputation
        train_loader_for_precompute = DataLoader(
            train_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=False, collate_fn=collate_batch_v2
        )
        val_loader_for_precompute = DataLoader(
            val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, collate_fn=collate_batch_v2
        )

        # Precompute embeddings with model in TRAINING mode (matching actual training behavior)
        train_precomputed = precompute_image_embeddings(model, train_loader_for_precompute, device, "Precomputing train embeddings")
        val_precomputed = precompute_image_embeddings(model, val_loader_for_precompute, device, "Precomputing val embeddings")

        # Create precomputed dataloaders
        train_loader_precomputed = DataLoader(
            train_precomputed, batch_size=args.batch_size, shuffle=True, num_workers=0, pin_memory=True, drop_last=True, collate_fn=collate_precomputed_batch
        )
        val_loader_precomputed = DataLoader(
            val_precomputed, batch_size=args.batch_size, shuffle=False, num_workers=0, collate_fn=collate_precomputed_batch
        )
        
        logger.info(f"Created precomputed dataloaders: Train={len(train_loader_precomputed)} batches, Val={len(val_loader_precomputed)} batches")
        use_precomputed = True

    # ========================================================================
    # STAGE 1: Concept Bottleneck + Global Alignment Training
    # ========================================================================
    if resume_from_stage is not None and resume_from_stage >= 2:
        logger.info(f"Skipping Stage 1 (resuming from Stage {resume_from_stage})")
        model.freeze_stage1()
        if global_epoch < args.stage0_epochs + args.stage1_epochs:
            global_epoch = args.stage0_epochs + args.stage1_epochs
    else:
        logger.info(f"\n{'='*74}")
        logger.info(f"STAGE 1: Concept Bottleneck + Global Alignment Training")
        logger.info(f"{'='*60}")

        stage1_params = model.get_stage1_params()
        stage1_num_trainable = sum(p.numel() for p in stage1_params)
        optimizer = torch.optim.AdamW(stage1_params, lr=args.stage1_lr, weight_decay=args.stage1_weight_decay)
        scheduler = CosineAnnealingLR(optimizer, T_max=args.stage1_epochs, eta_min=args.stage1_lr * 0.01)

        if resume_from_stage == 1 and checkpoint_optimizer_state is not None:
            try:
                optimizer.load_state_dict(checkpoint_optimizer_state)
            except ValueError as e:
                logger.warning(f"Could not load optimizer state (param mismatch), starting fresh: {e}")
        if resume_from_stage == 1 and checkpoint_scheduler_state is not None:
            scheduler.load_state_dict(checkpoint_scheduler_state)

        logger.info(f"Stage 1 optimizer: {stage1_num_trainable:,} params, lr={args.stage1_lr:.2e}")

        best_val_acc = 0.0
        patience_counter = 0

        start_epoch = resume_from_epoch if (resume_from_stage == 1 and resume_from_epoch is not None) else 0

        # Select dataloader
        current_train_loader = train_loader_precomputed if use_precomputed else train_loader
        current_val_loader = val_loader_precomputed if use_precomputed else val_loader

        for epoch in range(start_epoch, args.stage1_epochs):
            model.train()
            total_loss = 0
            total_concept_correct = 0
            total_concept_count = 0
            total_country_correct = 0
            total_country_count = 0
            
            pbar = tqdm(current_train_loader, desc=f"Epoch {epoch+1}/{args.stage1_epochs} [Stage 1]")

            for batch in pbar:
                if use_precomputed:
                    embeddings, concept_idx, target_idx, coords, _, cell_labels = batch
                    embeddings = embeddings.to(device)
                else:
                    images, concept_idx, target_idx, coords, _, cell_labels = batch
                    images = images.to(device)
                
                coords = coords.to(device)
                concept_idx = concept_idx.to(device)
                target_idx = target_idx.to(device)
                cell_labels = cell_labels.to(device)

                with torch.amp.autocast("cuda", enabled=args.use_amp):
                    if use_precomputed:
                        outputs = model.forward_from_features(embeddings, coords)
                    else:
                        outputs = model(images, coords)
                    
                    concept_emb = outputs["concept_emb"]
                    concept_logits = outputs["concept_logits"]
                    country_logits = outputs["country_logits"]
                    gps_emb = outputs["gps_emb"]

                    pred_concepts = concept_logits.argmax(dim=1)
                    concept_correct = (pred_concepts == concept_idx).sum().item()
                    total_concept_correct += concept_correct
                    total_concept_count += len(concept_idx)

                    pred_countries = country_logits.argmax(dim=1)
                    country_correct = (pred_countries == target_idx).sum().item()
                    total_country_correct += country_correct
                    total_country_count += len(target_idx)

                    loss_concept = nn.functional.cross_entropy(
                        concept_logits, concept_idx, weight=concept_weights if args.use_class_weights else None, label_smoothing=args.label_smoothing
                    )
                    loss_country = nn.functional.cross_entropy(country_logits, target_idx, label_smoothing=args.label_smoothing)

                    loss = args.lambda_concept * loss_concept + args.lambda_country * loss_country
                    loss = loss / args.gradient_accumulation_steps

                scaler.scale(loss).backward()
                if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()

                total_loss += loss.item() * args.gradient_accumulation_steps
                batch_concept_acc = concept_correct / len(concept_idx)
                pbar.set_postfix({"loss": f"{loss.item():.4f}", "concept": f"{loss_concept.item():.4f}", "country": f"{loss_country.item():.4f}", "concept_acc": f"{batch_concept_acc:.3f}"})

                if args.use_wandb:
                    wandb.log({"batch_loss": loss.item(), "stage": 1, "batch_concept_acc": batch_concept_acc})

            avg_train_loss = total_loss / len(current_train_loader)
            train_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
            train_country_acc = total_country_correct / total_country_count if total_country_count > 0 else 0.0

            log_metrics({"loss": avg_train_loss, "concept_acc": train_concept_acc, "country_acc": train_country_acc}, prefix="Train", stage=1)

            if args.use_wandb:
                wandb.log({"train_loss": avg_train_loss, "train_concept_acc": train_concept_acc, "epoch": global_epoch + 1, "stage": 1})

            logger.info(f"Validating Stage 1...")
            val_metrics = validate(model, current_val_loader, device, args, cell_centers, concept_weights, current_stage=1, use_precomputed=use_precomputed)

            scheduler.step()

            val_metric = val_metrics["concept_acc"]
            if val_metric > best_val_acc:
                best_val_acc = val_metric
                patience_counter = 0
                best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage1.pt"
                save_checkpoint(model, best_model_path, cell_centers, concept_names, full_dataset.country_to_idx, full_dataset.idx_to_country, args.encoder_model, {"stage": 1, "concept_acc": val_metric}, optimizer, scheduler, epoch)
                logger.info(f"Saved best Stage 1 model (Concept Acc: {val_metric:.4f})")
            else:
                patience_counter += 1
                if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                    logger.info(f"Stage 1 early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % args.save_interval == 0:
                visualize_predictions(model, val_loader, concept_names, full_dataset.idx_to_country, device, args, checkpoint_dir, global_epoch + 1, cell_centers, stage=1)
                diag_path = checkpoint_dir / "diagnostics" / f"stage1_epoch_{epoch+1}.csv"
                dump_diagnostics(model, val_loader, device, diag_path, concept_names, full_dataset.idx_to_country, cell_centers, log_to_wandb=args.use_wandb, wandb_step=global_epoch + 1)
                plot_error_distribution(model, val_loader, device, cell_centers, checkpoint_dir / "visualizations" / f"error_dist_stage1_epoch_{epoch+1}.png", stage=1)

            global_epoch += 1

        logger.info("Freezing Stage 1 parameters...")
        model.freeze_stage1()

    # ========================================================================
    # STAGE 2: Geolocation Head Training
    # ========================================================================
    logger.info(f"\n{'='*74}")
    logger.info(f"STAGE 2: Geolocation Head Training")
    logger.info(f"{'='*60}")

    stage2_params = model.get_stage2_params()
    stage2_num_trainable = sum(p.numel() for p in stage2_params)
    optimizer = torch.optim.AdamW(stage2_params, lr=args.stage2_lr, weight_decay=args.stage2_weight_decay)
    scheduler = CosineAnnealingLR(optimizer, T_max=args.stage2_epochs, eta_min=args.stage2_lr * 0.01)

    if resume_from_stage == 2 and checkpoint_optimizer_state is not None:
        try:
            optimizer.load_state_dict(checkpoint_optimizer_state)
        except ValueError as e:
            logger.warning(f"Could not load optimizer state (param mismatch), starting fresh: {e}")
    if resume_from_stage == 2 and checkpoint_scheduler_state is not None:
        scheduler.load_state_dict(checkpoint_scheduler_state)

    logger.info(f"Stage 2 optimizer: {stage2_num_trainable:,} params, lr={args.stage2_lr:.2e}")

    best_val_metric = float("inf")
    patience_counter = 0

    start_epoch = resume_from_epoch if (resume_from_stage == 2 and resume_from_epoch is not None) else 0

    # For Stage 2, always use original loaders (need full forward pass for validation metrics)
    for epoch in range(start_epoch, args.stage2_epochs):
        model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage2_epochs} [Stage 2]")

        for batch in pbar:
            images, concept_idx, target_idx, coords, _, cell_labels = batch
            images = images.to(device)
            coords = coords.to(device)
            cell_labels = cell_labels.to(device)

            with torch.amp.autocast("cuda", enabled=args.use_amp):
                outputs = model(images, coords)
                cell_logits = outputs["cell_logits"]
                pred_offsets = outputs["pred_offsets"]

                loss_cell = nn.functional.cross_entropy(cell_logits, cell_labels, label_smoothing=args.label_smoothing)

                batch_cell_centers = cell_centers[cell_labels]
                if model.coord_output_dim == 3:
                    lat_rad = torch.deg2rad(coords[:, 0])
                    lng_rad = torch.deg2rad(coords[:, 1])
                    x = torch.cos(lat_rad) * torch.cos(lng_rad)
                    y = torch.cos(lat_rad) * torch.sin(lng_rad)
                    z = torch.sin(lat_rad)
                    true_cart = torch.stack([x, y, z], dim=1)
                    target_offsets = true_cart - batch_cell_centers
                    loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
                else:
                    c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
                    c_lat = torch.rad2deg(torch.asin(c_z))
                    c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
                    batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
                    if args.coordinate_loss_type == "haversine":
                        pred_latlng = batch_cell_latlng + pred_offsets
                        loss_offset = coordinate_loss(pred_latlng, coords, loss_type="haversine")
                    else:
                        target_offsets = coords - batch_cell_latlng
                        target_offsets[:, 1] = (target_offsets[:, 1] + 180) % 360 - 180
                        loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)

                loss = args.lambda_cell * loss_cell + args.lambda_offset * loss_offset
                loss = loss / args.gradient_accumulation_steps

            scaler.scale(loss).backward()
            if (pbar.n + 1) % args.gradient_accumulation_steps == 0:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()

            total_loss += loss.item() * args.gradient_accumulation_steps
            pbar.set_postfix({"loss": f"{loss.item():.4f}", "cell": f"{loss_cell.item():.4f}", "offset": f"{loss_offset.item():.4f}"})

            if args.use_wandb:
                wandb.log({"batch_loss": loss.item(), "stage": 2})

        avg_train_loss = total_loss / len(train_loader)
        log_metrics({"loss": avg_train_loss}, prefix="Train", stage=2)

        if args.use_wandb:
            wandb.log({"train_loss": avg_train_loss, "epoch": global_epoch + 1, "stage": 2})

        logger.info(f"Validating Stage 2...")
        val_metrics = validate(model, val_loader, device, args, cell_centers, concept_weights, current_stage=2)

        scheduler.step()

        val_metric = val_metrics["median_error_km"]
        if val_metric < best_val_metric:
            best_val_metric = val_metric
            patience_counter = 0
            best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage2.pt"
            save_checkpoint(model, best_model_path, cell_centers, concept_names, full_dataset.country_to_idx, full_dataset.idx_to_country, args.encoder_model, {"stage": 2, "median_error_km": val_metric}, optimizer, scheduler, epoch)
            logger.info(f"Saved best Stage 2 model (Median Error: {val_metric:.1f}km)")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Stage 2 early stopping at epoch {epoch+1}")
                break

        if (epoch + 1) % args.save_interval == 0:
            visualize_predictions(model, val_loader, concept_names, full_dataset.idx_to_country, device, args, checkpoint_dir, global_epoch + 1, cell_centers, stage=2)
            diag_path = checkpoint_dir / "diagnostics" / f"stage2_epoch_{epoch+1}.csv"
            dump_diagnostics(model, val_loader, device, diag_path, concept_names, full_dataset.idx_to_country, cell_centers, log_to_wandb=args.use_wandb, wandb_step=global_epoch + 1)
            plot_error_distribution(model, val_loader, device, cell_centers, checkpoint_dir / "visualizations" / f"error_dist_stage2_epoch_{epoch+1}.png", stage=2)

        global_epoch += 1

    # ---------- Final Test Evaluation ----------
    logger.info(f"\n{'='*74}")
    logger.info("FINAL TEST EVALUATION")
    logger.info(f"{'='*60}")

    best_model_path = checkpoint_dir / "checkpoints" / "best_model_stage2.pt"
    if best_model_path.exists():
        checkpoint = load_checkpoint(best_model_path, device=device)
        model.load_state_dict(checkpoint["model_state_dict"])
        logger.info("Loaded best Stage 2 model for testing.")

    test_metrics = validate(model, test_loader, device, args, cell_centers, concept_weights, current_stage=2)
    logger.info(f"Test Metrics: {test_metrics}")

    if args.use_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        test_diag_path = checkpoint_dir / "diagnostics" / "test_diagnostics.csv"
        dump_diagnostics(model, test_loader, device, test_diag_path, concept_names, full_dataset.idx_to_country, cell_centers, max_samples=len(test_dataset), log_to_wandb=True)


# ---------- Validation Function ----------
@torch.no_grad()
def validate(model, val_loader, device, args, cell_centers, concept_weights=None, current_stage=1, use_precomputed=False):
    model.eval()
    total_loss = 0
    total_concept_correct = 0
    total_concept_count = 0
    total_country_correct = 0
    total_country_count = 0
    total_cell_correct = 0

    all_distances = []

    val_pbar = tqdm(val_loader, desc=f"Validating Stage {current_stage}")
    for batch in val_pbar:
        if use_precomputed:
            embeddings, concept_idx, target_idx, coords, _, cell_labels = batch
            embeddings = embeddings.to(device)
        else:
            images, concept_idx, target_idx, coords, _, cell_labels = batch
            images = images.to(device)
        
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        target_idx = target_idx.to(device)
        cell_labels = cell_labels.to(device)

        if use_precomputed:
            outputs = model.forward_from_features(embeddings, coords)
        else:
            outputs = model(images, coords)
        concept_emb = outputs["concept_emb"]
        concept_logits = outputs["concept_logits"]
        country_logits = outputs["country_logits"]
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        gps_emb = outputs["gps_emb"]

        pred_concepts = concept_logits.argmax(dim=1)
        concept_correct = (pred_concepts == concept_idx).sum().item()
        total_concept_correct += concept_correct
        total_concept_count += len(concept_idx)

        pred_countries = country_logits.argmax(dim=1)
        country_correct = (pred_countries == target_idx).sum().item()
        total_country_correct += country_correct
        total_country_count += len(target_idx)

        pred_cells = cell_logits.argmax(dim=1)
        cell_correct = (pred_cells == cell_labels).sum().item()
        total_cell_correct += cell_correct

        pred_cell_centers = cell_centers[pred_cells]

        if model.coord_output_dim == 3:
            pred_cart = pred_cell_centers + pred_offsets
            pred_cart = torch.nn.functional.normalize(pred_cart, p=2, dim=1)
            pred_latlng_deg = sphere_to_latlng(pred_cart)
            pred_coords = pred_latlng_deg
        else:
            c_x, c_y, c_z = pred_cell_centers[:, 0], pred_cell_centers[:, 1], pred_cell_centers[:, 2]
            c_lat = torch.rad2deg(torch.asin(c_z))
            c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
            pred_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
            pred_coords = pred_cell_latlng + pred_offsets

        batch_distances = haversine_distance(pred_coords, coords)
        if batch_distances.numel() > 0:
            all_distances.append(batch_distances.cpu())

        if current_stage == 1:
            loss_concept_gps = geocell_contrastive_loss(concept_emb, gps_emb, cell_labels, temperature=args.temperature)
            loss_concept = nn.functional.cross_entropy(concept_logits, concept_idx, weight=concept_weights if args.use_class_weights else None, label_smoothing=args.label_smoothing)
            loss_country = nn.functional.cross_entropy(country_logits, target_idx, label_smoothing=args.label_smoothing)
            loss = args.lambda_concept_gps * loss_concept_gps + args.lambda_concept * loss_concept + args.lambda_country * loss_country
        else:
            loss_cell = nn.functional.cross_entropy(cell_logits, cell_labels, label_smoothing=args.label_smoothing)
            batch_cell_centers = cell_centers[cell_labels]
            if model.coord_output_dim == 3:
                lat_rad = torch.deg2rad(coords[:, 0])
                lng_rad = torch.deg2rad(coords[:, 1])
                x = torch.cos(lat_rad) * torch.cos(lng_rad)
                y = torch.cos(lat_rad) * torch.sin(lng_rad)
                z = torch.sin(lat_rad)
                true_cart = torch.stack([x, y, z], dim=1)
                target_offsets = true_cart - batch_cell_centers
                loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
            else:
                c_x, c_y, c_z = batch_cell_centers[:, 0], batch_cell_centers[:, 1], batch_cell_centers[:, 2]
                c_lat = torch.rad2deg(torch.asin(c_z))
                c_lng = torch.rad2deg(torch.atan2(c_y, c_x))
                batch_cell_latlng = torch.stack([c_lat, c_lng], dim=1)
                if args.coordinate_loss_type == "haversine":
                    pred_latlng = batch_cell_latlng + pred_offsets
                    loss_offset = coordinate_loss(pred_latlng, coords, loss_type="haversine")
                else:
                    target_offsets = coords - batch_cell_latlng
                    target_offsets[:, 1] = (target_offsets[:, 1] + 180) % 360 - 180
                    loss_offset = nn.functional.mse_loss(pred_offsets, target_offsets)
            loss = args.lambda_cell * loss_cell + args.lambda_offset * loss_offset

        total_loss += loss.item()

    threshold_accuracies = {}
    if all_distances:
        all_distances_tensor = torch.cat(all_distances)
        for level, threshold_km in THRESHOLD_ACCURACIES.items():
            acc = accuracy_within_threshold(all_distances_tensor, threshold_km)
            threshold_accuracies[f"acc_{level}"] = acc
        median_error = torch.median(all_distances_tensor).item()
        threshold_accuracies["median_error_km"] = median_error
    else:
        threshold_accuracies = {"acc_street": 0.0, "acc_city": 0.0, "acc_region": 0.0, "acc_country": 0.0, "acc_continent": 0.0, "median_error_km": 0.0}

    avg_loss = total_loss / len(val_loader)
    val_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
    val_country_acc = total_country_correct / total_country_count if total_country_count > 0 else 0.0
    val_cell_acc = total_cell_correct / total_concept_count if total_concept_count > 0 else 0.0

    val_metrics = {"loss": avg_loss, "concept_acc": val_concept_acc, "country_acc": val_country_acc, "cell_acc": val_cell_acc, **threshold_accuracies}
    log_metrics(val_metrics, prefix="Val", stage=current_stage)

    if args.use_wandb:
        wandb.log({
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
        })

    return val_metrics


# ---------- Main Entry Point ----------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Concept-Aware CBM")

    # Dataset Arguments
    parser.add_argument("--geoguessr_id", type=str, default="6906237dc7731161a37282b2")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--csv_path", type=str, default=None)

    # Model Arguments
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--country_filter", type=str, default=None)

    # Training Arguments
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    parser.add_argument("--resume_from_epoch", type=int, default=None)

    # Stage 0
    parser.add_argument("--stage0_epochs", type=int, default=5)
    parser.add_argument("--stage0_lr", type=float, default=3e-5)
    parser.add_argument("--stage0_weight_decay", type=float, default=0.05)
    parser.add_argument("--unfreeze_layers", type=int, default=2)

    # Stage 1
    parser.add_argument("--stage1_epochs", type=int, default=15)
    parser.add_argument("--stage1_lr", type=float, default=1e-4)
    parser.add_argument("--stage1_weight_decay", type=float, default=0.01)

    # Stage 2
    parser.add_argument("--stage2_epochs", type=int, default=15)
    parser.add_argument("--stage2_lr", type=float, default=3e-4)
    parser.add_argument("--stage2_weight_decay", type=float, default=0.01)

    # General
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--use_amp", action="store_true")
    parser.add_argument("--min_samples_per_cell", type=int, default=500)

    # Loss
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_concept_gps", type=float, default=1.0)
    parser.add_argument("--lambda_concept", type=float, default=1.0)
    parser.add_argument("--lambda_country", type=float, default=0.5)
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_offset", type=float, default=5.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    parser.add_argument("--coordinate_loss_type", type=str, default="haversine", choices=["haversine", "mse", "sphere"])

    # Misc
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    parser.add_argument("--precompute_embeddings", action="store_true", default=False, help="Precompute image embeddings for faster Stage 1 training")

    args = parser.parse_args()
    train(args)
