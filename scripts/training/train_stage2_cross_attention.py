#!/usr/bin/env python3
"""
Stage 2 Cross-Attention Training Script: Interpretable Geolocation Prediction

This script implements Stage 2 training for geolocation prediction with interpretability:
- Cross-attention between concept embeddings (query) and image patch tokens (keys/values)
- Attention visualization showing which image patches drive predictions
- Semantic Geocell classification + coordinate offset regression
- Uses frozen Stage 1 Concept Bottleneck loaded from checkpoint

Architecture:
- concept_emb [B, 512] as query (computed on-the-fly via frozen Stage 1 bottleneck)
- patch_tokens [B, 576, 1024] projected to [B, 576, 512] as keys/values
- cross_attn output -> cell_head, offset_head

Trainable: patch_proj, cross_attn, cell_head, offset_head
Frozen: image_encoder, concept_bottleneck (from Stage 1)

Data Strategy:
- Loads images directly and computes all embeddings on-the-fly
- No precomputed embeddings needed - requires Stage 1 checkpoint
- TRAIN + VAL for training, TEST for final evaluation
"""

import argparse
import logging
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import Counter
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from matplotlib.colors import LinearSegmentedColormap
from PIL import Image

from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified,
    get_transforms_from_processor,
    SubsetDataset,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from src.losses import haversine_distance
from src.concepts.utils import extract_concepts_from_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ---------- Constants ----------
VIZ_DPI = 150
THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
    "continent": 2500.0,
}

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


# ---------- Helper Functions ----------
def sanitize_for_filename(text: str) -> str:
    """Sanitize text for safe filenames."""
    return re.sub(r"[^a-zA-Z0-9_-]", "_", str(text))


def split_metadata_batch(metadata, batch_size: int) -> List[Dict]:
    """Convert a collated metadata batch (dict of lists) into a list of dicts."""
    if isinstance(metadata, list):
        return metadata
    if not isinstance(metadata, dict):
        return [{} for _ in range(batch_size)]
    
    result = []
    for i in range(batch_size):
        entry = {}
        for k, v in metadata.items():
            try:
                entry[k] = v[i]
            except Exception:
                entry[k] = v
        result.append(entry)
    return result


def format_distance(km: float, precision: int = 1) -> str:
    """Format distance for display."""
    if km < 1:
        return f"{km*1000:.0f}m"
    elif km < 1000:
        return f"{km:.{precision}f}km"
    else:
        return f"{km/1000:.2f}Mm"


# ---------- Dataset ----------
class Stage2ImageDataset(Dataset):
    """
    Dataset that loads images directly for Stage 2 training.
    
    Computes on-the-fly:
    - patch_tokens: [576, 1024] from StreetCLIP ViT
    - concept_emb: [512] from frozen Stage 1 concept bottleneck
    
    This avoids storing large precomputed embeddings.
    """
    
    def __init__(
        self,
        image_paths: List[str],
        coordinates: torch.Tensor,  # [N, 2] lat/lng
        cell_labels: torch.Tensor,  # [N]
        countries: List[str],
        transforms=None,
    ):
        self.image_paths = image_paths
        self.coordinates = coordinates
        self.cell_labels = cell_labels
        self.countries = countries
        self.transforms = transforms
        
        assert len(image_paths) == len(coordinates) == len(cell_labels) == len(countries)
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        
        # Load and transform image
        try:
            pil_image = Image.open(image_path).convert("RGB")
            if self.transforms:
                img_tensor = self.transforms(pil_image)
            else:
                img_tensor = torch.zeros(3, 336, 336)  # Placeholder
        except Exception as e:
            logger.debug(f"Failed to load {image_path}: {e}")
            img_tensor = torch.zeros(3, 336, 336)  # Placeholder
        
        return (
            img_tensor,
            self.coordinates[idx],
            self.cell_labels[idx],
            self.countries[idx],
            image_path,
        )


def stage2_collate_fn(batch):
    """Collate function for Stage 2 dataset."""
    images = torch.stack([item[0] for item in batch])
    coords = torch.stack([item[1] for item in batch])
    cell_labels = torch.stack([item[2] for item in batch])
    countries = [item[3] for item in batch]
    image_paths = [item[4] for item in batch]
    return images, coords, cell_labels, countries, image_paths


def load_stage1_checkpoint(
    checkpoint_path: Path,
    image_encoder: StreetCLIPEncoder,
    device: torch.device,
) -> Tuple[Stage1ConceptModel, Dict]:
    """
    Load Stage 1 model from checkpoint.
    
    Args:
        checkpoint_path: Path to Stage 1 checkpoint
        image_encoder: Already-initialized image encoder
        device: Device to load model on
        
    Returns:
        Tuple of (Stage1ConceptModel, concept_info_dict)
        concept_info_dict contains: concept_names, parent_names, concept_to_idx, parent_to_idx
    """
    logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract required tensors
    T_meta_base = checkpoint["T_meta_base"]
    T_parent_base = checkpoint["T_parent_base"]
    meta_to_parent_idx = checkpoint["meta_to_parent_idx"]
    
    # Create model
    model = Stage1ConceptModel(
        image_encoder=image_encoder,
        T_meta=T_meta_base,
        T_parent=T_parent_base,
        meta_to_parent_idx=meta_to_parent_idx,
        streetclip_dim=768,
        concept_emb_dim=512,
    )
    
    # Load state dict
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    # Freeze all parameters
    for param in model.parameters():
        param.requires_grad = False
    
    # Extract concept info for visualization
    concept_info = {
        "concept_names": checkpoint.get("concept_names", []),
        "parent_names": checkpoint.get("parent_names", []),
        "concept_to_idx": checkpoint.get("concept_to_idx", {}),
        "parent_to_idx": checkpoint.get("parent_to_idx", {}),
        "meta_to_parent_idx": meta_to_parent_idx,
    }
    
    # Build reverse mappings
    concept_info["idx_to_concept"] = {v: k for k, v in concept_info["concept_to_idx"].items()}
    concept_info["idx_to_parent"] = {v: k for k, v in concept_info["parent_to_idx"].items()}
    
    logger.info(f"Loaded Stage 1 model with {checkpoint['num_concepts']} concepts, {len(concept_info['parent_names'])} parents")
    return model, concept_info


# ---------- Geocell Generation ----------
def generate_semantic_geocells(
    coordinates: torch.Tensor,
    countries: List[str],
    min_samples_per_cell: int = 500,
    output_dir: Optional[Path] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate semantic geocells using per-country K-Means clustering."""
    logger.info("Generating Semantic Geocells...")
    
    all_coords = coordinates.numpy()
    all_countries = np.array(countries)
    unique_countries = np.unique(all_countries)
    
    cell_centers_list = []
    sample_to_cell_map = np.zeros(len(all_coords), dtype=int)
    current_cell_id_offset = 0
    
    for country in tqdm(unique_countries, desc="Clustering Countries"):
        country_mask = all_countries == country
        country_indices = np.where(country_mask)[0]
        country_coords = all_coords[country_indices]
        n_samples = len(country_coords)
        
        # Convert to 3D for clustering to avoid dateline issues
        lat_rad = np.deg2rad(country_coords[:, 0])
        lng_rad = np.deg2rad(country_coords[:, 1])
        x = np.cos(lat_rad) * np.cos(lng_rad)
        y = np.cos(lat_rad) * np.sin(lng_rad)
        z = np.sin(lat_rad)
        xyz = np.stack([x, y, z], axis=1)
        
        if n_samples > min_samples_per_cell:
            n_clusters = max(1, n_samples // min_samples_per_cell)
            kmeans = KMeans(n_clusters=n_clusters, n_init=10, random_state=42)
            kmeans.fit(xyz)
            
            centers_xyz = kmeans.cluster_centers_
            labels = kmeans.labels_
            
            # Normalize centers to unit sphere
            norms = np.linalg.norm(centers_xyz, axis=1, keepdims=True)
            centers_xyz = centers_xyz / norms
            
            cell_centers_list.append(centers_xyz)
            sample_to_cell_map[country_indices] = labels + current_cell_id_offset
            current_cell_id_offset += n_clusters
        else:
            # Single cluster for small countries
            center_xyz = np.mean(xyz, axis=0, keepdims=True)
            center_xyz = center_xyz / np.linalg.norm(center_xyz)
            
            cell_centers_list.append(center_xyz)
            sample_to_cell_map[country_indices] = current_cell_id_offset
            current_cell_id_offset += 1
    
    cell_centers = torch.tensor(np.concatenate(cell_centers_list, axis=0), dtype=torch.float32)
    sample_to_cell = torch.tensor(sample_to_cell_map, dtype=torch.long)
    logger.info(f"Generated {len(cell_centers)} Semantic Geocells.")
    
    # Visualization
    if output_dir:
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


# ---------- Coordinate Utilities ----------
def cell_center_to_latlng(cell_centers: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert cell centers from 3D Cartesian to lat/lng."""
    cx, cy, cz = cell_centers[:, 0], cell_centers[:, 1], cell_centers[:, 2]
    lat = torch.rad2deg(torch.asin(torch.clamp(cz, -1.0, 1.0)))
    lng = torch.rad2deg(torch.atan2(cy, cx))
    return lat, lng


def latlng_to_cartesian(coordinates: torch.Tensor) -> torch.Tensor:
    """Convert lat/lng to 3D Cartesian coordinates."""
    lat_rad = torch.deg2rad(coordinates[:, 0])
    lng_rad = torch.deg2rad(coordinates[:, 1])
    x = torch.cos(lat_rad) * torch.cos(lng_rad)
    y = torch.cos(lat_rad) * torch.sin(lng_rad)
    z = torch.sin(lat_rad)
    return torch.stack([x, y, z], dim=1)


def cartesian_to_latlng(cart: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """Convert 3D Cartesian to lat/lng."""
    cart_norm = F.normalize(cart, p=2, dim=1)
    x, y, z = cart_norm[:, 0], cart_norm[:, 1], cart_norm[:, 2]
    lat = torch.rad2deg(torch.asin(torch.clamp(z, -1.0, 1.0)))
    lng = torch.rad2deg(torch.atan2(y, x))
    return lat, lng


# ---------- Visualization ----------
@torch.no_grad()
def visualize_attention_predictions(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Stage1ConceptModel,
    concept_info: Dict,
    dataloader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    epoch: int,
    output_dir: Path,
    coord_output_dim: int = 3,
    num_samples: int = 4,
    log_to_wandb: bool = False,
    args=None,
):
    """
    Comprehensive Stage 2 visualization showing:
    - Original image with attention heatmap overlay
    - Top-5 predicted concepts bar chart
    - GT vs Pred parent/child concepts
    - Geocell predictions (GT cell, Pred cell)
    - Coordinate predictions (GT coords, Pred coords, Distance error)
    """
    model.eval()
    stage1_model.eval()
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get concept name mappings
    idx_to_concept = concept_info.get("idx_to_concept", {})
    idx_to_parent = concept_info.get("idx_to_parent", {})
    meta_to_parent_idx = concept_info.get("meta_to_parent_idx", None)
    
    # Collect samples
    all_samples = []
    for batch in dataloader:
        images, coords, cell_labels, countries, image_paths = batch
        for i in range(len(images)):
            if Path(image_paths[i]).exists():
                all_samples.append({
                    "image": images[i],
                    "coords": coords[i],
                    "cell_label": cell_labels[i],
                    "country": countries[i],
                    "image_path": image_paths[i],
                })
        if len(all_samples) >= num_samples * 2:
            break
    
    if not all_samples:
        logger.warning("No valid samples found for visualization")
        return
    
    # Random sample selection
    np.random.shuffle(all_samples)
    samples = all_samples[:num_samples]
    
    # Create attention colormap (transparent to red)
    colors = [(1, 0, 0, 0), (1, 0, 0, 0.7)]
    attn_cmap = LinearSegmentedColormap.from_list("attention", colors, N=256)
    
    # Create figure: 4 samples, each with 2 columns (image+attention, bar chart)
    fig = plt.figure(figsize=(24, 6 * num_samples))
    
    for idx, sample in enumerate(samples):
        # Load original image for display
        image_path = Path(sample["image_path"])
        pil_image = Image.open(image_path).convert("RGB")
        
        # Get pre-transformed image tensor
        img_tensor = sample["image"].unsqueeze(0).to(device)
        
        # Get patch tokens and image features
        patch_tokens = image_encoder.get_patch_tokens(img_tensor)  # [1, 576, 1024]
        img_features = image_encoder(img_tensor)  # [1, 768]
        
        # Get concept predictions from Stage 1 (full forward pass)
        stage1_outputs = stage1_model.forward_from_features(img_features)
        concept_emb = stage1_outputs["concept_emb"]
        meta_probs = stage1_outputs["meta_probs"][0]  # [num_metas]
        parent_probs = stage1_outputs["parent_probs"][0]  # [num_parents]
        
        # Forward through Stage 2 model
        outputs = model(concept_emb, patch_tokens)
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        attn_weights = outputs.get("attn_weights")
        
        # ===== Process Concept Predictions =====
        # Top-5 meta concepts
        top5_meta_probs, top5_meta_idx = torch.topk(meta_probs, min(5, len(meta_probs)))
        top5_meta_names = [idx_to_concept.get(i.item(), f"Meta-{i.item()}")[:30] for i in top5_meta_idx]
        
        # Top-5 parent concepts
        top5_parent_probs, top5_parent_idx = torch.topk(parent_probs, min(5, len(parent_probs)))
        top5_parent_names = [idx_to_parent.get(i.item(), f"Parent-{i.item()}")[:30] for i in top5_parent_idx]
        
        # Predicted meta and parent
        pred_meta_idx = meta_probs.argmax().item()
        pred_meta_name = idx_to_concept.get(pred_meta_idx, f"Meta-{pred_meta_idx}")
        pred_parent_idx = parent_probs.argmax().item()
        pred_parent_name = idx_to_parent.get(pred_parent_idx, f"Parent-{pred_parent_idx}")
        
        # Get predicted parent from meta (hierarchical)
        if meta_to_parent_idx is not None:
            hier_parent_idx = meta_to_parent_idx[pred_meta_idx].item()
            hier_parent_name = idx_to_parent.get(hier_parent_idx, f"Parent-{hier_parent_idx}")
        else:
            hier_parent_name = "N/A"
        
        # ===== Process Attention =====
        attn_spatial = model.attention_to_spatial(attn_weights)  # [1, 24, 24]
        attn_map = attn_spatial[0].cpu().numpy()  # [24, 24]
        
        # Upsample attention map to image size
        img_h, img_w = pil_image.size[1], pil_image.size[0]
        attn_map_upsampled = Image.fromarray((attn_map * 255).astype(np.uint8))
        attn_map_upsampled = attn_map_upsampled.resize((img_w, img_h), Image.BILINEAR)
        attn_map_upsampled = np.array(attn_map_upsampled) / 255.0
        
        # ===== Process Geolocation =====
        pred_cell = cell_logits.argmax(dim=1).item()
        pred_cell_center = cell_centers[pred_cell].to(device).unsqueeze(0)
        
        if coord_output_dim == 3:
            pred_cart = pred_cell_center + pred_offsets
            pred_lat, pred_lng = cartesian_to_latlng(pred_cart)
        else:
            c_lat, c_lng = cell_center_to_latlng(pred_cell_center)
            pred_lat = c_lat + pred_offsets[0, 0]
            pred_lng = c_lng + pred_offsets[0, 1]
            pred_lng = ((pred_lng + 180) % 360) - 180
        
        pred_coords = torch.stack([pred_lat, pred_lng], dim=1)
        gt_coords = sample["coords"].unsqueeze(0).to(device)
        dist_error = haversine_distance(pred_coords, gt_coords).item()
        
        gt_cell = sample["cell_label"].item()
        gt_lat, gt_lng = sample["coords"][0].item(), sample["coords"][1].item()
        p_lat, p_lng = pred_lat.item(), pred_lng.item()
        
        # ===== Create Subplots for this sample =====
        # Row layout: [Image+Attention (wide), Top-5 Meta Bar, Top-5 Parent Bar, Info Text]
        row_base = idx * 4
        
        # Column 1: Image with attention overlay (spans 2 columns worth of space)
        ax_img = fig.add_subplot(num_samples, 4, row_base + 1)
        ax_img.imshow(pil_image)
        ax_img.imshow(attn_map_upsampled, cmap=attn_cmap, alpha=0.6)
        ax_img.axis("off")
        ax_img.set_title(f"Sample {idx+1}: {sample['country']}", fontsize=11, fontweight='bold')
        
        # Column 2: Top-5 Meta Concepts Bar Chart
        ax_meta = fig.add_subplot(num_samples, 4, row_base + 2)
        y_pos = np.arange(len(top5_meta_names))
        bars_meta = ax_meta.barh(y_pos, top5_meta_probs.cpu().numpy(), color='steelblue', alpha=0.8)
        ax_meta.set_yticks(y_pos)
        ax_meta.set_yticklabels(top5_meta_names, fontsize=8)
        ax_meta.set_xlabel("Probability", fontsize=9)
        ax_meta.set_title("Top-5 Child Concepts", fontsize=10, fontweight='bold')
        ax_meta.set_xlim(0, 1)
        ax_meta.invert_yaxis()
        # Highlight top prediction
        if len(bars_meta) > 0:
            bars_meta[0].set_color('darkblue')
        
        # Column 3: Top-5 Parent Concepts Bar Chart
        ax_parent = fig.add_subplot(num_samples, 4, row_base + 3)
        y_pos = np.arange(len(top5_parent_names))
        bars_parent = ax_parent.barh(y_pos, top5_parent_probs.cpu().numpy(), color='darkorange', alpha=0.8)
        ax_parent.set_yticks(y_pos)
        ax_parent.set_yticklabels(top5_parent_names, fontsize=8)
        ax_parent.set_xlabel("Probability", fontsize=9)
        ax_parent.set_title("Top-5 Parent Concepts", fontsize=10, fontweight='bold')
        ax_parent.set_xlim(0, 1)
        ax_parent.invert_yaxis()
        if len(bars_parent) > 0:
            bars_parent[0].set_color('darkorange')
        
        # Column 4: Prediction Summary Text
        ax_text = fig.add_subplot(num_samples, 4, row_base + 4)
        ax_text.axis("off")
        
        # Determine accuracy colors
        cell_correct = gt_cell == pred_cell
        cell_color = "green" if cell_correct else "red"
        
        summary_text = (
            f"═══ CONCEPT PREDICTIONS ═══\n"
            f"Pred Child:  {pred_meta_name[:35]}\n"
            f"Pred Parent: {pred_parent_name[:35]}\n"
            f"Hier Parent: {hier_parent_name[:35]}\n"
            f"\n"
            f"══ GEOLOCATION PREDICTIONS ══\n"
            f"GT Cell:   {gt_cell:4d}    Pred Cell: {pred_cell:4d}\n"
            f"Cell Match: {'✓ CORRECT' if cell_correct else '✗ WRONG'}\n"
            f"\n"
            f"GT Coords:   ({gt_lat:7.2f}, {gt_lng:8.2f})\n"
            f"Pred Coords: ({p_lat:7.2f}, {p_lng:8.2f})\n"
            f"\n"
            f"════ DISTANCE ERROR ════\n"
            f"Error: {format_distance(dist_error)}\n"
            f"\n"
            f"Street (<1km):   {'✓' if dist_error <= 1 else '✗'}\n"
            f"City (<25km):    {'✓' if dist_error <= 25 else '✗'}\n"
            f"Region (<200km): {'✓' if dist_error <= 200 else '✗'}\n"
            f"Country (<750km):{'✓' if dist_error <= 750 else '✗'}"
        )
        
        ax_text.text(0.05, 0.95, summary_text, transform=ax_text.transAxes,
                     fontsize=9, verticalalignment='top', fontfamily='monospace',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    save_path = viz_dir / f"epoch_{epoch}_comprehensive_predictions.png"
    plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved comprehensive visualization to {save_path}")
    
    if log_to_wandb and args and args.use_wandb:
        import wandb
        wandb.log({
            "comprehensive_predictions": wandb.Image(str(save_path), caption=f"Epoch {epoch}")
        }, step=epoch)


# ---------- Training Functions ----------
def compute_offset_targets(
    coordinates: torch.Tensor,
    cell_labels: torch.Tensor,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
) -> torch.Tensor:
    """Compute offset regression targets."""
    gt_cell_centers = cell_centers[cell_labels]
    
    if coord_output_dim == 3:
        # 3D Cartesian offsets
        gt_cart = latlng_to_cartesian(coordinates)
        target_offsets = gt_cart - gt_cell_centers
    else:
        # 2D Lat/Lng offsets
        c_lat, c_lng = cell_center_to_latlng(gt_cell_centers)
        target_lat_offset = coordinates[:, 0] - c_lat
        target_lng_offset = coordinates[:, 1] - c_lng
        target_lng_offset = ((target_lng_offset + 180) % 360) - 180
        target_offsets = torch.stack([target_lat_offset, target_lng_offset], dim=1)
    
    return target_offsets


def compute_predicted_coords(
    pred_cells: torch.Tensor,
    pred_offsets: torch.Tensor,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    device: torch.device,
) -> torch.Tensor:
    """Compute predicted coordinates from cell predictions and offsets."""
    pred_cell_centers = cell_centers[pred_cells].to(device)
    
    if coord_output_dim == 3:
        pred_cart = pred_cell_centers + pred_offsets
        pred_lat, pred_lng = cartesian_to_latlng(pred_cart)
    else:
        c_lat, c_lng = cell_center_to_latlng(pred_cell_centers)
        pred_lat = c_lat + pred_offsets[:, 0]
        pred_lng = c_lng + pred_offsets[:, 1]
        pred_lng = ((pred_lng + 180) % 360) - 180
    
    return torch.stack([pred_lat, pred_lng], dim=1)


@torch.no_grad()
def validate(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Stage1ConceptModel,
    val_loader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    epoch: int,
    args,
) -> Dict[str, float]:
    """Run validation with on-the-fly embedding computation."""
    model.eval()
    stage1_model.eval()
    
    criterion_cell = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_cell_acc = 0.0
    haversine_errors = []
    n_batches = 0
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            images, coordinates, cell_labels, countries, image_paths = batch
            images = images.to(device)
            coordinates = coordinates.to(device)
            cell_labels = cell_labels.to(device)
            
            # Get patch tokens and image features from encoder
            patch_tokens = image_encoder.get_patch_tokens(images)  # [B, 576, 1024]
            img_features = image_encoder(images)  # [B, 768]
            
            # Get concept embeddings from frozen Stage 1
            concept_embs = stage1_model.concept_bottleneck(img_features)  # [B, 512]
            
            # Forward pass
            outputs = model(concept_embs, patch_tokens)
            cell_logits = outputs["cell_logits"]
            pred_offsets = outputs["pred_offsets"]
            
            # Loss computation
            loss_cell = criterion_cell(cell_logits, cell_labels)
            
            target_offsets = compute_offset_targets(coordinates, cell_labels, cell_centers.to(device), coord_output_dim)
            loss_offset = F.mse_loss(pred_offsets, target_offsets)
            
            loss = args.lambda_cell * loss_cell + args.lambda_offset * loss_offset
            total_loss += loss.item()
            
            # Metrics
            pred_cells = cell_logits.argmax(dim=1)
            total_cell_acc += (pred_cells == cell_labels).float().mean().item()
            
            # Distance errors
            pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)
            dists = haversine_distance(pred_coords, coordinates)
            haversine_errors.extend(dists.cpu().numpy())
            
            n_batches += 1
    
    if n_batches == 0:
        return {"loss": float("nan"), "cell_acc": 0.0, "median_error_km": float("inf")}
    
    avg_loss = total_loss / n_batches
    avg_cell_acc = total_cell_acc / n_batches
    median_error = np.median(haversine_errors) if haversine_errors else float("inf")
    
    # Threshold accuracies
    threshold_accs = {}
    for name, thresh in THRESHOLD_ACCURACIES.items():
        acc = np.mean([d <= thresh for d in haversine_errors]) * 100 if haversine_errors else 0.0
        threshold_accs[f"acc_{name}"] = acc
    
    logger.info(
        f"Val Epoch {epoch}: Loss={avg_loss:.4f}, Cell Acc={avg_cell_acc:.4f}, "
        f"Median Error={format_distance(median_error)}, "
        f"Street={threshold_accs['acc_street']:.1f}%, City={threshold_accs['acc_city']:.1f}%, "
        f"Region={threshold_accs['acc_region']:.1f}%, Country={threshold_accs['acc_country']:.1f}%"
    )
    
    return {
        "loss": avg_loss,
        "cell_acc": avg_cell_acc,
        "median_error_km": median_error,
        **threshold_accs,
    }


def train_epoch(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Stage1ConceptModel,
    train_loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    epoch: int,
    args,
) -> float:
    """Train for one epoch with on-the-fly embedding computation."""
    model.train()
    image_encoder.model.eval()  # Keep encoder frozen
    stage1_model.eval()  # Keep Stage 1 frozen
    
    criterion_cell = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    n_batches = 0
    
    pbar = tqdm(train_loader, desc=f"Train Epoch {epoch}")
    
    for batch in pbar:
        images, coordinates, cell_labels, countries, image_paths = batch
        images = images.to(device)
        coordinates = coordinates.to(device)
        cell_labels = cell_labels.to(device)
        
        # Get patch tokens and image features (encoder is frozen)
        with torch.no_grad():
            patch_tokens = image_encoder.get_patch_tokens(images)  # [B, 576, 1024]
            img_features = image_encoder(images)  # [B, 768]
            # Get concept embeddings from frozen Stage 1
            concept_embs = stage1_model.concept_bottleneck(img_features)  # [B, 512]
        
        # Forward pass through trainable head
        optimizer.zero_grad()
        
        outputs = model(concept_embs, patch_tokens)
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        
        # Loss computation
        loss_cell = criterion_cell(cell_logits, cell_labels)
        
        target_offsets = compute_offset_targets(coordinates, cell_labels, cell_centers.to(device), coord_output_dim)
        loss_offset = F.mse_loss(pred_offsets, target_offsets)
        
        loss = args.lambda_cell * loss_cell + args.lambda_offset * loss_offset
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        n_batches += 1
        
        pbar.set_postfix({"loss": f"{loss.item():.4f}"})
    
    return total_loss / max(n_batches, 1)


def save_checkpoint(
    model: Stage2CrossAttentionGeoHead,
    checkpoint_path: Path,
    cell_centers: torch.Tensor,
    encoder_model: str,
    coord_output_dim: int,
    extra_info: Optional[Dict] = None,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scheduler=None,
    epoch: Optional[int] = None,
):
    """Save Stage 2 checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "cell_centers": cell_centers.cpu(),
        "num_cells": len(cell_centers),
        "coord_output_dim": coord_output_dim,
        "encoder_model": encoder_model,
        "patch_dim": model.patch_proj[0].in_features,
        "concept_dim": model.cross_attn.embed_dim,
        "num_heads": model.cross_attn.num_heads,
        "ablation_mode": model.ablation_mode,  # Save ablation mode for reproducibility
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
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def main():
    parser = argparse.ArgumentParser(description="Stage 2: Cross-Attention Geolocation Training")
    
    # Data
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--geoguessr_id", type=str, default="6906237dc7731161a37282b2")
    
    # Stage 1 checkpoint (required for concept embeddings)
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="Path to Stage 1 model checkpoint")
    
    # Model
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--patch_dim", type=int, default=1024,
                        help="Dimension of patch tokens from ViT")
    parser.add_argument("--concept_dim", type=int, default=512,
                        help="Dimension of concept embeddings")
    parser.add_argument("--num_heads", type=int, default=8,
                        help="Number of attention heads")
    parser.add_argument("--coord_output_dim", type=int, default=3,
                        choices=[2, 3], help="Coordinate output dimension (2=lat/lng, 3=xyz)")
    
    # Training
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--min_samples_per_cell", type=int, default=500)
    parser.add_argument("--num_workers", type=int, default=4)
    
    # Loss weights
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_offset", type=float, default=5.0)
    
    # Ablation study configuration
    parser.add_argument("--ablation_mode", type=str, default="both",
                        choices=["both", "concept_only", "image_only"],
                        help="Ablation mode for experiments: "
                             "'both' = concept + image fusion (default), "
                             "'concept_only' = only concept embedding, "
                             "'image_only' = only image patches")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup Output Directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        # Include ablation mode in directory name for easy identification
        output_dir = Path("results") / f"stage2_cross_attention_{args.ablation_mode}" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    (output_dir / "visualizations").mkdir(exist_ok=True)
    
    logger.info(f"Ablation Mode: {args.ablation_mode}")
    logger.info(f"  - concept_only: Only concept embeddings contribute to location prediction")
    logger.info(f"  - image_only: Only image patches contribute to location prediction")
    logger.info(f"  - both: Concept + image fusion with enforced concept usage (default)")
    
    # Initialize WandB
    if args.use_wandb:
        import wandb
        wandb.init(project="streetclip-cbm-stage2", config=vars(args), name=f"stage2-{args.ablation_mode}-{output_dir.name}")
    
    # Initialize Image Encoder (frozen, for patch extraction and concept computation)
    logger.info("Initializing frozen image encoder...")
    encoder_config = StreetCLIPConfig(model_name=args.encoder_model, finetune=False, device=device)
    image_encoder = StreetCLIPEncoder(encoder_config)
    image_encoder.model.eval()
    for param in image_encoder.model.parameters():
        param.requires_grad = False
    
    # Load Stage 1 model for concept embeddings
    stage1_model, concept_info = load_stage1_checkpoint(
        Path(args.stage1_checkpoint),
        image_encoder,
        device,
    )
    
    # Get transforms from image processor
    transforms = get_transforms_from_processor(image_encoder.image_processor)
    
    # Load dataset from CSV
    logger.info(f"Loading dataset from {args.csv_path}")
    import pandas as pd
    df = pd.read_csv(args.csv_path)
    
    # Build image paths and extract coordinates/countries
    image_paths = []
    coordinates = []
    countries = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        # Use image_path column if available, otherwise construct from pano_id
        if "image_path" in row and pd.notna(row["image_path"]):
            img_path = Path(row["image_path"])
        else:
            pano_id = row.get("pano_id") or row.get("panoId")
            if pd.isna(pano_id):
                continue
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
    
    if len(coordinates) == 0:
        raise RuntimeError(f"No valid samples found! Check image paths in CSV or data_root/geoguessr_id settings.")
    
    coordinates = torch.stack(coordinates)
    cell_labels = torch.zeros(len(image_paths), dtype=torch.long)  # Will be updated after geocell generation
    
    logger.info(f"Loaded {len(image_paths)} valid samples")
    
    # Generate Semantic Geocells
    cell_centers, sample_to_cell = generate_semantic_geocells(
        coordinates,
        countries,
        min_samples_per_cell=args.min_samples_per_cell,
        output_dir=output_dir / "visualizations",
    )
    cell_centers = cell_centers.to(device)
    num_cells = len(cell_centers)
    cell_labels = sample_to_cell
    
    # Train/Val Split
    n_samples = len(image_paths)
    n_val = int(n_samples * args.val_split)
    indices = list(range(n_samples))
    np.random.shuffle(indices)
    
    train_indices = indices[n_val:]
    val_indices = indices[:n_val]
    
    train_dataset = Stage2ImageDataset(
        image_paths=[image_paths[i] for i in train_indices],
        coordinates=coordinates[train_indices],
        cell_labels=cell_labels[train_indices],
        countries=[countries[i] for i in train_indices],
        transforms=transforms,
    )
    val_dataset = Stage2ImageDataset(
        image_paths=[image_paths[i] for i in val_indices],
        coordinates=coordinates[val_indices],
        cell_labels=cell_labels[val_indices],
        countries=[countries[i] for i in val_indices],
        transforms=transforms,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}")
    
    # Create dataloaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
    )
    
    # Initialize Stage 2 Cross-Attention Head
    logger.info(f"Initializing Stage2CrossAttentionGeoHead with ablation_mode='{args.ablation_mode}'...")
    model = Stage2CrossAttentionGeoHead(
        patch_dim=args.patch_dim,
        concept_emb_dim=args.concept_dim,
        num_cells=num_cells,
        coord_output_dim=args.coord_output_dim,
        num_heads=args.num_heads,
        ablation_mode=args.ablation_mode,
    )
    model.to(device)
    
    # Optimizer (only Stage 2 head parameters)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    logger.info("Starting Stage 2 Cross-Attention Training...")
    best_val_error = float("inf")
    
    for epoch in range(args.epochs):
        # Train
        train_loss = train_epoch(
            model, image_encoder, stage1_model, train_loader, optimizer, device,
            cell_centers, args.coord_output_dim, epoch, args
        )
        
        scheduler.step()
        
        # Validate
        val_metrics = validate(
            model, image_encoder, stage1_model, val_loader, device,
            cell_centers, args.coord_output_dim, epoch, args
        )
        
        # Visualize every epoch
        visualize_attention_predictions(
            model, image_encoder, stage1_model, concept_info, val_loader, device,
            cell_centers, epoch, output_dir, args.coord_output_dim,
            num_samples=4, log_to_wandb=args.use_wandb, args=args
        )
        
        # Log to WandB
        if args.use_wandb:
            import wandb
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_cell_acc": val_metrics["cell_acc"],
                "val_median_error": val_metrics["median_error_km"],
                "val_acc_street": val_metrics.get("acc_street", 0),
                "val_acc_city": val_metrics.get("acc_city", 0),
                "val_acc_region": val_metrics.get("acc_region", 0),
                "val_acc_country": val_metrics.get("acc_country", 0),
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })
        
        # Save best checkpoint
        if val_metrics["median_error_km"] < best_val_error:
            best_val_error = val_metrics["median_error_km"]
            save_checkpoint(
                model,
                output_dir / "checkpoints" / "best_model_stage2_xattn.pt",
                cell_centers,
                args.encoder_model,
                args.coord_output_dim,
                extra_info={
                    "val_median_error": best_val_error,
                    "val_metrics": val_metrics,
                    "stage1_checkpoint": args.stage1_checkpoint,
                },
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            save_checkpoint(
                model,
                output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt",
                cell_centers,
                args.encoder_model,
                args.coord_output_dim,
                extra_info={"stage1_checkpoint": args.stage1_checkpoint},
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
    
    # Save final checkpoint
    save_checkpoint(
        model,
        output_dir / "checkpoints" / "final_model_stage2_xattn.pt",
        cell_centers,
        args.encoder_model,
        args.coord_output_dim,
        extra_info={
            "final_val_metrics": val_metrics,
            "stage1_checkpoint": args.stage1_checkpoint,
        },
        epoch=args.epochs - 1,
    )
    
    logger.info(f"Training complete. Best validation median error: {format_distance(best_val_error)}")
    logger.info(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
