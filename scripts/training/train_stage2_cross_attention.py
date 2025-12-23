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
- Uses same train/val/test splits as Stage 1 (from splits.json in checkpoint directory)
- TRAIN for training, VAL for validation, TEST for testing
"""

import argparse
import json
import logging
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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

from src.dataset import get_transforms_from_processor

from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from src.losses import haversine_distance
import wandb
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


def compute_concept_emb_and_patches(
    images: torch.Tensor,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Optional[Stage1ConceptModel],
    ablation_mode: str,
    patch_dim: int,
    concept_dim: int,
    use_amp: bool,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute (concept_embs, patch_tokens) efficiently based on ablation mode.

    Key speed optimization:
    - Uses StreetCLIPEncoder.get_features_and_patches() to avoid TWO vision forwards
      (one for patch tokens, one for projected image features).
    - Skips patch token computation entirely for concept_only.
    - Skips Stage1 bottleneck entirely for image_only.
    """
    bsz = images.size(0)
    if stage1_model is None and ablation_mode in {"both", "concept_only"}:
        raise ValueError(
            f"ablation_mode={ablation_mode} requires --stage1_checkpoint (concept embeddings)."
        )

    if ablation_mode == "concept_only":
        # No patch tokens needed.
        if use_amp and images.is_cuda:
            with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                img_features = image_encoder(images)  # [B, 768]
                concept_embs = stage1_model.concept_bottleneck(img_features)  # [B, 512]
        else:
            img_features = image_encoder(images)
            concept_embs = stage1_model.concept_bottleneck(img_features)

        patch_tokens = torch.empty((bsz, 0, patch_dim), device=images.device, dtype=images.dtype)
        return concept_embs, patch_tokens

    # For both/image_only we need patch tokens. Get both outputs in ONE forward.
    if use_amp and images.is_cuda:
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            img_features, patch_tokens = image_encoder.get_features_and_patches(images)
    else:
        img_features, patch_tokens = image_encoder.get_features_and_patches(images)

    if ablation_mode == "image_only":
        # Stage2 ignores concept_emb in this mode (but still needs the batch dim).
        concept_embs = torch.zeros((bsz, concept_dim), device=images.device, dtype=img_features.dtype)
        return concept_embs, patch_tokens

    # ablation_mode == "both"
    # Stage 1 bottleneck expects float32, so cast if needed (AMP may produce bfloat16)
    img_features_f32 = img_features.float() if img_features.dtype != torch.float32 else img_features
    concept_embs = stage1_model.concept_bottleneck(img_features_f32)
    return concept_embs, patch_tokens


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
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
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


def is_missing_or_none_path(p: Optional[str]) -> bool:
    if p is None:
        return True
    s = str(p).strip()
    return s == "" or s.lower() == "none"


def load_image_encoder_weights_from_stage0_checkpoint(
    stage0_checkpoint_path: Path,
    image_encoder: StreetCLIPEncoder,
) -> None:
    """
    Load ONLY the StreetCLIP encoder weights from a Stage0 checkpoint into `image_encoder`.

    Stage0 checkpoints store the full Stage0PretrainingModel state dict, whose keys include
    the StreetCLIP encoder under the `image_encoder.` prefix. We strip that prefix and load
    into the Stage2 `StreetCLIPEncoder`, so patch tokens + global features are computed with
    the same encoder weights used to train Stage1 when Stage1 was resumed from Stage0.
    """
    logger.info(f"Loading Stage0 encoder weights from {stage0_checkpoint_path}")
    stage0_ckpt = torch.load(stage0_checkpoint_path, map_location="cpu", weights_only=False)
    state = stage0_ckpt.get("model_state_dict")
    if state is None:
        raise KeyError(f"Stage0 checkpoint missing model_state_dict: {stage0_checkpoint_path}")

    encoder_state = {k[len('image_encoder.'):]: v for k, v in state.items() if k.startswith("image_encoder.")}
    load_res = image_encoder.load_state_dict(encoder_state, strict=False)
    missing = getattr(load_res, "missing_keys", [])
    unexpected = getattr(load_res, "unexpected_keys", [])
    logger.info(
        f"Loaded Stage0 encoder weights into Stage2 image_encoder "
        f"(missing={len(missing)}, unexpected={len(unexpected)})"
    )


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


def assign_samples_to_train_geocells(
    coordinates: torch.Tensor,
    countries: List[str],
    train_cell_centers: torch.Tensor,
    train_cell_ids_by_country: Dict[str, torch.Tensor],
) -> Tuple[torch.Tensor, Dict[str, float]]:
    """
    Assign each sample to the nearest TRAIN geocell center in 3D.

    Preference: if a sample's country exists in train, search only that country's centers;
    otherwise fall back to global nearest across all train centers.
    """
    if coordinates.numel() == 0:
        return torch.empty((0,), dtype=torch.long), {
            "fallback_country_unseen": 0.0,
            "dot_mean": float("nan"),
            "dot_p50": float("nan"),
            "dot_p95": float("nan"),
        }

    # Convert sample coords to unit xyz
    xyz = latlng_to_cartesian(coordinates).float()
    xyz = F.normalize(xyz, p=2, dim=1)

    # Ensure centers are unit vectors
    centers = F.normalize(train_cell_centers.float(), p=2, dim=1)

    assigned = torch.empty((xyz.size(0),), dtype=torch.long)
    best_dot = torch.empty((xyz.size(0),), dtype=torch.float32)

    fallback_count = 0
    all_center_ids = torch.arange(centers.size(0), dtype=torch.long)

    for i in range(xyz.size(0)):
        country = countries[i]
        candidate_ids = train_cell_ids_by_country.get(country)
        if candidate_ids is None or candidate_ids.numel() == 0:
            candidate_ids = all_center_ids
            fallback_count += 1

        cand = centers.index_select(0, candidate_ids)
        dots = torch.mv(cand, xyz[i])  # [C]
        j = torch.argmax(dots).item()
        assigned[i] = candidate_ids[j]
        best_dot[i] = dots[j].item()

    stats = {
        "fallback_country_unseen": float(fallback_count),
        "dot_mean": best_dot.mean().item(),
        "dot_p50": torch.quantile(best_dot, 0.50).item(),
        "dot_p95": torch.quantile(best_dot, 0.95).item(),
    }
    return assigned, stats


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
    stage1_model: Optional[Stage1ConceptModel],
    concept_info: Dict,
    dataloader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    epoch: int,
    output_dir: Path,
    coord_output_dim: int = 3,
    num_samples: int = 4,
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
    if stage1_model is None:
        logger.info("Skipping visualization: no Stage1 checkpoint/model available")
        return
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
        
        # Efficiently compute inputs for Stage2 (avoid double vision forward)
        concept_emb, patch_tokens = compute_concept_emb_and_patches(
            images=img_tensor,
            image_encoder=image_encoder,
            stage1_model=stage1_model,
            ablation_mode=model.ablation_mode,
            patch_dim=getattr(args, "patch_dim", 1024),
            concept_dim=getattr(args, "concept_dim", 512),
            use_amp=getattr(args, "amp", False),
        )
        
        # Get concept predictions from Stage 1 (full forward pass)
        # Note: For image_only, concept_emb is zeros for Stage2; we still compute Stage1
        # probs for visualization, using the same projected img_features from a single pass.
        if model.ablation_mode != "concept_only":
            # In both/image_only paths, compute_concept_emb_and_patches used get_features_and_patches,
            # so we already have projected features implicitly inside Stage1 bottleneck computation.
            # For visualization, we recompute Stage1 outputs from projected features by reusing encoder.
            if getattr(args, "amp", False) and img_tensor.is_cuda:
                with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
                    img_features_viz = image_encoder(img_tensor)
            else:
                img_features_viz = image_encoder(img_tensor)
        else:
            # concept_only path used image_encoder(images) already; compute again is cheap relative to IO,
            # and keeps the code simple here.
            img_features_viz = image_encoder(img_tensor)

        # Stage 1 expects float32, so cast if needed (AMP may produce bfloat16)
        img_features_viz_f32 = img_features_viz.float() if img_features_viz.dtype != torch.float32 else img_features_viz
        stage1_outputs = stage1_model.forward_from_features(img_features_viz_f32)
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
        # Note: in ablation modes ('concept_only', 'image_only') the model does not produce attention weights.
        attn_map_upsampled = None
        if attn_weights is not None:
            attn_spatial = model.attention_to_spatial(attn_weights)  # [1, 24, 24]
            attn_map = attn_spatial[0].detach().cpu().numpy()  # [24, 24]
            attn_map = np.clip(attn_map, 0.0, 1.0)

            # Upsample attention map to image size
            img_h, img_w = pil_image.size[1], pil_image.size[0]
            attn_map_upsampled = Image.fromarray((attn_map * 255).astype(np.uint8))
            attn_map_upsampled = attn_map_upsampled.resize((img_w, img_h), Image.BILINEAR)
            attn_map_upsampled = np.array(attn_map_upsampled) / 255.0
        else:
            logger.info(
                f"Skipping attention overlay for visualization (no attn_weights; ablation_mode={outputs.get('ablation_mode', 'unknown')})"
            )
        
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
        if attn_map_upsampled is not None:
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
    
    try:
        wandb.log({
            "comprehensive_predictions": wandb.Image(str(save_path), caption=f"Epoch {epoch}")
        }, step=epoch)
    except Exception as e:
        logger.error(f"Error logging to WandB: {e}")


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
    stage1_model: Optional[Stage1ConceptModel],
    val_loader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    epoch: int,
    args,
) -> Dict[str, float]:
    """Run validation with on-the-fly embedding computation."""
    model.eval()
    if stage1_model is not None:
        stage1_model.eval()
    
    criterion_cell = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_cell_acc = 0.0
    haversine_errors = []
    n_batches = 0
    gate_values = []
    
    with torch.no_grad():
        for batch in tqdm(val_loader, desc=f"Val Epoch {epoch}"):
            images, coordinates, cell_labels, countries, image_paths = batch
            images = images.to(device)
            coordinates = coordinates.to(device)
            cell_labels = cell_labels.to(device)
            
            concept_embs, patch_tokens = compute_concept_emb_and_patches(
                images=images,
                image_encoder=image_encoder,
                stage1_model=stage1_model,
                ablation_mode=args.ablation_mode,
                patch_dim=args.patch_dim,
                concept_dim=args.concept_dim,
                use_amp=args.amp,
            )
            
            # Forward pass
            outputs = model(concept_embs, patch_tokens, return_attention=False, return_gate=True)
            cell_logits = outputs["cell_logits"]
            pred_offsets = outputs["pred_offsets"]
            gate = outputs.get("gate")
            
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

            if gate is not None:
                gate_values.append(gate.detach().cpu())
            
            n_batches += 1
    
    if n_batches == 0:
        return {"loss": float("nan"), "cell_acc": 0.0, "median_error_km": float("inf")}
    
    avg_loss = total_loss / n_batches
    avg_cell_acc = total_cell_acc / n_batches
    median_error = np.median(haversine_errors) if haversine_errors else float("inf")

    gate_stats = {}
    if gate_values:
        gate_all = torch.cat(gate_values, dim=0).float()  # [N, D]
        gate_flat = gate_all.flatten()
        gate_stats = {
            "gate_mean": gate_flat.mean().item(),
            "gate_std": gate_flat.std(unbiased=False).item(),
            "gate_p10": torch.quantile(gate_flat, 0.10).item(),
            "gate_p50": torch.quantile(gate_flat, 0.50).item(),
            "gate_p90": torch.quantile(gate_flat, 0.90).item(),
        }
    
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
        **gate_stats,
        **threshold_accs,
    }


def train_epoch(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Optional[Stage1ConceptModel],
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
    if stage1_model is not None:
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
        
        # Get concept embeddings + patch tokens efficiently (encoder + Stage1 are frozen)
        with torch.no_grad():
            concept_embs, patch_tokens = compute_concept_emb_and_patches(
                images=images,
                image_encoder=image_encoder,
                stage1_model=stage1_model,
                ablation_mode=args.ablation_mode,
                patch_dim=args.patch_dim,
                concept_dim=args.concept_dim,
                use_amp=args.amp,
            )
        
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
    parser.add_argument("--splits_json", type=str, default=None,
                        help="Path to splits.json file. If not provided, will try to load from stage1 checkpoint directory.")
    
    # Stage 1 checkpoint (optional; required for concept embeddings)
    parser.add_argument(
        "--stage1_checkpoint",
        type=str,
        default=None,
        help=(
            "Path to Stage 1 model checkpoint. If provided, Stage2 will use the encoder lineage "
            "associated with this checkpoint for BOTH global features and patch tokens. "
            "If omitted, Stage2 can only run with --ablation_mode image_only using a fresh vanilla encoder."
        ),
    )
    
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
    parser.add_argument("--amp", action="store_true", default=False,
                        help="Use autocast (bfloat16) for frozen encoder/Stage1 forward to speed up.")
    
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
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--viz_every", type=int, default=1,
                        help="Run attention/qualitative visualization every N epochs. Set to 0 to disable.")
    
    args = parser.parse_args()
    
    # Set random seeds
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    # Speed knobs (safe on H100; improves matmul/conv performance)
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        torch.backends.cudnn.benchmark = True
        try:
            torch.set_float32_matmul_precision("high")
        except Exception:
            pass
    
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
        # Build tags with key experiment parameters
        tags = [
            f"ablation_{args.ablation_mode}",
            f"coord_dim_{args.coord_output_dim}",
            f"encoder_{args.encoder_model.split('/')[-1]}",  # Just model name, not full path
        ]
        # Add coordinate type tag
        if args.coord_output_dim == 3:
            tags.append("coord_xyz")
        else:
            tags.append("coord_latlng")
        
        wandb.init(
            project="streetclip-cbm-stage2",
            config=vars(args),
            name=f"stage2-{args.ablation_mode}-{output_dir.name}",
            tags=tags
        )
    
    if args.stage1_checkpoint is None and args.ablation_mode != "image_only":
        raise ValueError(
            "No --stage1_checkpoint provided. Only --ablation_mode image_only is supported "
            "(concept embeddings require Stage1)."
        )

    # Initialize Image Encoder (frozen). If a Stage1 checkpoint is provided, we load encoder
    # weights tied to that Stage1 checkpoint lineage so patch tokens come from the correct encoder.
    logger.info("Initializing image encoder for patch extraction...")
    encoder_config = StreetCLIPConfig(model_name=args.encoder_model, finetune=False, device=device)
    image_encoder = StreetCLIPEncoder(encoder_config)

    stage1_model: Optional[Stage1ConceptModel] = None
    concept_info: Dict = {}
    stage1_ckpt_data: Optional[Dict] = None
    splits_json_path_str = None
    stage0_checkpoint: Optional[str] = None

    if args.stage1_checkpoint is not None:
        stage1_checkpoint_path = Path(args.stage1_checkpoint)
        stage1_ckpt_data = torch.load(stage1_checkpoint_path, map_location="cpu", weights_only=False)
        stage0_checkpoint = stage1_ckpt_data.get("stage0_checkpoint")

        # Prefer encoder weights embedded directly in Stage1 checkpoint (if present).
        embedded_enc = stage1_ckpt_data.get("image_encoder_state_dict")
        if embedded_enc is not None:
            logger.info("Using image_encoder_state_dict embedded in Stage1 checkpoint for patch extraction")
            load_res = image_encoder.load_state_dict(embedded_enc, strict=False)
            missing = getattr(load_res, "missing_keys", [])
            unexpected = getattr(load_res, "unexpected_keys", [])
            logger.info(f"Loaded embedded encoder weights (missing={len(missing)}, unexpected={len(unexpected)})")
        else:
            if not is_missing_or_none_path(stage0_checkpoint):
                load_image_encoder_weights_from_stage0_checkpoint(Path(stage0_checkpoint), image_encoder)
            else:
                logger.info("Stage1 checkpoint indicates vanilla encoder lineage (no Stage0 checkpoint); using base encoder weights")

        # Freeze encoder
        image_encoder.model.eval()
        for param in image_encoder.model.parameters():
            param.requires_grad = False

        # Load Stage 1 model for concept embeddings (uses the same image_encoder instance)
        stage1_model, concept_info = load_stage1_checkpoint(stage1_checkpoint_path, image_encoder, device)
        splits_json_path_str = stage1_ckpt_data.get("splits_json")
    else:
        logger.info("No Stage1 checkpoint provided: using vanilla image encoder weights for image_only mode")
        image_encoder.model.eval()
        for param in image_encoder.model.parameters():
            param.requires_grad = False
    
    # Get transforms from image processor
    transforms = get_transforms_from_processor(image_encoder.image_processor)
    
    # Load splits.json
    if args.splits_json:
        splits_json_path = Path(args.splits_json)
    else:
        if args.stage1_checkpoint is None:
            raise ValueError("No --stage1_checkpoint and no --splits_json provided. Provide --splits_json.")
        # Try to load from Stage 1 checkpoint directory
        stage1_checkpoint_path = Path(args.stage1_checkpoint)
        stage1_dir = stage1_checkpoint_path.parent.parent
        splits_json_path = stage1_dir / "splits.json"
    
    if not splits_json_path.exists():
        raise FileNotFoundError(f"splits.json not found: {splits_json_path}")
    
    logger.info(f"Loading splits from {splits_json_path}")
    with open(splits_json_path, 'r') as f:
        splits_data = json.load(f)
    
    train_pano_ids = set(splits_data["train_pano_ids"])
    val_pano_ids = set(splits_data["val_pano_ids"])
    test_pano_ids = set(splits_data["test_pano_ids"])
    
    logger.info(f"Splits from Stage 1: Train={len(train_pano_ids)}, Val={len(val_pano_ids)}, Test={len(test_pano_ids)}")
    
    # Load dataset from CSV
    logger.info(f"Loading dataset from {args.csv_path}")
    import pandas as pd
    df = pd.read_csv(args.csv_path)
    
    # Build image paths and extract coordinates/countries, tracking pano_ids for split assignment
    image_paths = []
    coordinates = []
    countries = []
    pano_ids = []  # Track pano_id for each sample
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
        # Get pano_id
        pano_id = row.get("pano_id") or row.get("panoId")
        if pd.isna(pano_id):
            continue
        
        # Use image_path column if available, otherwise construct from pano_id
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
        pano_ids.append(str(pano_id))  # Store as string for matching
    
    if len(coordinates) == 0:
        raise RuntimeError(f"No valid samples found! Check image paths in CSV or data_root/geoguessr_id settings.")
    
    coordinates = torch.stack(coordinates)
    cell_labels = torch.full((len(image_paths),), -1, dtype=torch.long)  # Will be updated after geocell generation
    
    logger.info(f"Loaded {len(image_paths)} valid samples")

    # Assign samples to splits based on pano_id matching Stage 1 splits
    train_indices = []
    val_indices = []
    test_indices = []
    
    for i, pano_id in enumerate(pano_ids):
        if pano_id in train_pano_ids:
            train_indices.append(i)
        elif pano_id in val_pano_ids:
            val_indices.append(i)
        elif pano_id in test_pano_ids:
            test_indices.append(i)
    
    # Log split statistics
    logger.info(f"Split assignment: Train={len(train_indices)}, Val={len(val_indices)}, Test={len(test_indices)}")
    logger.info(f"Unmatched samples: {len(image_paths) - len(train_indices) - len(val_indices) - len(test_indices)}")

    # -------- Split-consistent Semantic Geocells (fit on TRAIN only) --------
    if len(train_indices) == 0:
        raise RuntimeError("No TRAIN samples matched splits.json; cannot fit geocells on train split.")

    train_coords = coordinates[train_indices]
    train_countries = [countries[i] for i in train_indices]

    cell_centers_train, train_sample_to_cell = generate_semantic_geocells(
        train_coords,
        train_countries,
        min_samples_per_cell=args.min_samples_per_cell,
        output_dir=output_dir / "visualizations",
    )
    cell_centers = cell_centers_train.to(device)
    num_cells = len(cell_centers_train)

    # Fill TRAIN labels (train_sample_to_cell aligns with train_indices order)
    cell_labels[train_indices] = train_sample_to_cell

    # Build country -> train cell ids mapping (for per-country nearest assignment)
    train_cell_ids_by_country: Dict[str, torch.Tensor] = {}
    for country in set(train_countries):
        ids = train_sample_to_cell[
            torch.tensor([c == country for c in train_countries], dtype=torch.bool)
        ].unique()
        train_cell_ids_by_country[country] = ids

    # Assign VAL/TEST samples to nearest TRAIN geocell
    val_assign_stats = {}
    test_assign_stats = {}

    if len(val_indices) > 0:
        val_coords = coordinates[val_indices]
        val_countries = [countries[i] for i in val_indices]
        val_assigned, val_assign_stats = assign_samples_to_train_geocells(
            coordinates=val_coords,
            countries=val_countries,
            train_cell_centers=cell_centers_train,
            train_cell_ids_by_country=train_cell_ids_by_country,
        )
        cell_labels[val_indices] = val_assigned

    if len(test_indices) > 0:
        test_coords = coordinates[test_indices]
        test_countries = [countries[i] for i in test_indices]
        test_assigned, test_assign_stats = assign_samples_to_train_geocells(
            coordinates=test_coords,
            countries=test_countries,
            train_cell_centers=cell_centers_train,
            train_cell_ids_by_country=train_cell_ids_by_country,
        )
        cell_labels[test_indices] = test_assigned

    # -------- Sanity checks: all split labels must be valid --------
    def assert_valid_split_labels(split_name: str, idxs: List[int]) -> None:
        if len(idxs) == 0:
            return
        labels = cell_labels[idxs]
        if (labels < 0).any():
            bad = int((labels < 0).sum().item())
            raise RuntimeError(f"Found {bad} negative geocell labels in split='{split_name}'.")
        if (labels >= num_cells).any():
            bad = int((labels >= num_cells).sum().item())
            raise RuntimeError(f"Found {bad} out-of-range geocell labels in split='{split_name}' (num_cells={num_cells}).")

    assert_valid_split_labels("train", train_indices)
    assert_valid_split_labels("val", val_indices)
    assert_valid_split_labels("test", test_indices)

    logger.info(
        f"Geocells fit on TRAIN only: num_cells={num_cells}. "
        f"VAL assigned via nearest-train (country fallback={int(val_assign_stats.get('fallback_country_unseen', 0.0))}). "
        f"TEST assigned via nearest-train (country fallback={int(test_assign_stats.get('fallback_country_unseen', 0.0))})."
    )

    # -------- W&B logging: geocells + split stats --------
    encoder_type_tag = "vanilla" if "vanilla" in args.stage1_checkpoint else "finetuned"
    if args.use_wandb:
        additional_tags = [
            f"num_cells_{num_cells}",
            f"dataset_size_{len(image_paths)}",
            f"min_samples_per_cell_{args.min_samples_per_cell}",
            f"stage1_encoder_type_{encoder_type_tag}",
        ]
        wandb.run.tags = list(wandb.run.tags) + additional_tags
        wandb.config.update(
            {
                "num_cells": num_cells,
                "dataset_size": len(image_paths),
                "num_countries": len(set(countries)),
                "stage1_encoder_type": encoder_type_tag,
                "split_train_size": len(train_indices),
                "split_val_size": len(val_indices),
                "split_test_size": len(test_indices),
            }
        )

        # Assignment diagnostics
        n_unmatched = len(image_paths) - len(train_indices) - len(val_indices) - len(test_indices)
        val_country_unseen = float(val_assign_stats.get("fallback_country_unseen", 0.0))
        test_country_unseen = float(test_assign_stats.get("fallback_country_unseen", 0.0))

        wandb.log(
            {
                "geocells_num_cells": num_cells,
                "splits_unmatched_samples": n_unmatched,
                "geocells_val_country_unseen_fallback": val_country_unseen,
                "geocells_test_country_unseen_fallback": test_country_unseen,
                "geocells_val_dot_mean": val_assign_stats.get("dot_mean", float("nan")),
                "geocells_val_dot_p50": val_assign_stats.get("dot_p50", float("nan")),
                "geocells_val_dot_p95": val_assign_stats.get("dot_p95", float("nan")),
                "geocells_test_dot_mean": test_assign_stats.get("dot_mean", float("nan")),
                "geocells_test_dot_p50": test_assign_stats.get("dot_p50", float("nan")),
                "geocells_test_dot_p95": test_assign_stats.get("dot_p95", float("nan")),
            }
        )

        # Split/country table: {split, country, n_samples, n_unique_cells}
        split_country_rows = []
        for split_name, idxs in [("train", train_indices), ("val", val_indices), ("test", test_indices)]:
            if len(idxs) == 0:
                continue
            split_countries = [countries[i] for i in idxs]
            split_labels = cell_labels[idxs]
            for country in sorted(set(split_countries)):
                mask = torch.tensor([c == country for c in split_countries], dtype=torch.bool)
                n_samples = int(mask.sum().item())
                n_unique_cells = int(split_labels[mask].unique().numel())
                split_country_rows.append([split_name, country, n_samples, n_unique_cells])

        if split_country_rows:
            wandb.log(
                {
                    "geocells_split_country_summary": wandb.Table(
                        columns=["split", "country", "n_samples", "n_unique_cells"],
                        data=split_country_rows,
                    )
                }
            )
    
    # Use train for training, val for validation, test for testing
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
    test_dataset = Stage2ImageDataset(
        image_paths=[image_paths[i] for i in test_indices],
        coordinates=coordinates[test_indices],
        cell_labels=cell_labels[test_indices],
        countries=[countries[i] for i in test_indices],
        transforms=transforms,
    )
    
    logger.info(f"Train: {len(train_dataset)}, Val: {len(val_dataset)}, Test: {len(test_dataset)}")
    
    # Create dataloaders
    dataloader_extra_kwargs = {}
    if args.num_workers > 0:
        dataloader_extra_kwargs = {
            "persistent_workers": True,
            "prefetch_factor": 4,
        }

    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
        drop_last=True,
        **dataloader_extra_kwargs,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
        **dataloader_extra_kwargs,
    )
    test_loader = (
        DataLoader(
            test_dataset,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            collate_fn=stage2_collate_fn,
            pin_memory=True,
            **dataloader_extra_kwargs,
        )
        if len(test_dataset) > 0
        else None
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
        
        # Visualize every epoch (requires Stage1 for concept predictions)
        if stage1_model is not None and args.viz_every > 0 and (epoch % args.viz_every == 0):
            visualize_attention_predictions(
                model, image_encoder, stage1_model, concept_info, val_loader, device,
                cell_centers, epoch, output_dir, args.coord_output_dim,
                num_samples=4, args=args
            )
        
        # Log to WandB
        if args.use_wandb:
            wandb.log({
                "train_loss": train_loss,
                "val_loss": val_metrics["loss"],
                "val_cell_acc": val_metrics["cell_acc"],
                "val_median_error": val_metrics["median_error_km"],
                "val_acc_street": val_metrics.get("acc_street", 0),
                "val_acc_city": val_metrics.get("acc_city", 0),
                "val_acc_region": val_metrics.get("acc_region", 0),
                "val_acc_country": val_metrics.get("acc_country", 0),
                "val_gate_mean": val_metrics.get("gate_mean", float("nan")),
                "val_gate_std": val_metrics.get("gate_std", float("nan")),
                "val_gate_p10": val_metrics.get("gate_p10", float("nan")),
                "val_gate_p50": val_metrics.get("gate_p50", float("nan")),
                "val_gate_p90": val_metrics.get("gate_p90", float("nan")),
                "lr": scheduler.get_last_lr()[0],
                "epoch": epoch,
            })
        
        # Save best checkpoint
        if val_metrics["median_error_km"] < best_val_error:
            best_val_error = val_metrics["median_error_km"]
            extra_info_dict = {
                "val_median_error": best_val_error,
                "val_metrics": val_metrics,
                "stage1_checkpoint": args.stage1_checkpoint,
            }
            if stage0_checkpoint is not None:
                extra_info_dict["stage0_checkpoint"] = stage0_checkpoint
            if splits_json_path_str is not None:
                extra_info_dict["splits_json"] = splits_json_path_str
            save_checkpoint(
                model,
                output_dir / "checkpoints" / "best_model_stage2_xattn.pt",
                cell_centers,
                args.encoder_model,
                args.coord_output_dim,
                extra_info=extra_info_dict,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
        
        # Save periodic checkpoint
        if epoch % 10 == 0:
            extra_info_periodic = {"stage1_checkpoint": args.stage1_checkpoint}
            if stage0_checkpoint is not None:
                extra_info_periodic["stage0_checkpoint"] = stage0_checkpoint
            if splits_json_path_str is not None:
                extra_info_periodic["splits_json"] = splits_json_path_str
            save_checkpoint(
                model,
                output_dir / "checkpoints" / f"checkpoint_epoch_{epoch}.pt",
                cell_centers,
                args.encoder_model,
                args.coord_output_dim,
                extra_info=extra_info_periodic,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
    
    # Final test evaluation (using best model)
    test_metrics = None
    if test_loader is not None and len(test_dataset) > 0:
        logger.info("Loading best model for final test evaluation...")
        best_ckpt_path = output_dir / "checkpoints" / "best_model_stage2_xattn.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info("Loaded best model checkpoint")
        
        logger.info("Running final test evaluation...")
        test_metrics = validate(
            model, image_encoder, stage1_model, test_loader, device,
            cell_centers, args.coord_output_dim, args.epochs - 1, args
        )
        logger.info(f"Test Metrics: Median Error={format_distance(test_metrics['median_error_km'])}, "
                   f"Cell Acc={test_metrics['cell_acc']:.4f}")
        if args.use_wandb:
            wandb.log({
                "test_loss": test_metrics["loss"],
                "test_cell_acc": test_metrics["cell_acc"],
                "test_median_error": test_metrics["median_error_km"],
                "test_acc_street": test_metrics.get("acc_street", 0),
                "test_acc_city": test_metrics.get("acc_city", 0),
                "test_acc_region": test_metrics.get("acc_region", 0),
                "test_acc_country": test_metrics.get("acc_country", 0),
            })
        
        # Update best checkpoint with test metrics
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location="cpu", weights_only=False)
            best_ckpt["test_metrics"] = {k: (v.item() if isinstance(v, torch.Tensor) else v) 
                                        for k, v in test_metrics.items()}
            torch.save(best_ckpt, best_ckpt_path)
            logger.info("Updated best checkpoint with test metrics")
    
    # Save final checkpoint
    final_extra_info = {
        "final_val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "stage1_checkpoint": args.stage1_checkpoint,
    }
    if stage0_checkpoint is not None:
        final_extra_info["stage0_checkpoint"] = stage0_checkpoint
    if splits_json_path_str is not None:
        final_extra_info["splits_json"] = splits_json_path_str
    save_checkpoint(
        model,
        output_dir / "checkpoints" / "final_model_stage2_xattn.pt",
        cell_centers,
        args.encoder_model,
        args.coord_output_dim,
        extra_info=final_extra_info,
        epoch=args.epochs - 1,
    )
    
    logger.info(f"Training complete. Best validation median error: {format_distance(best_val_error)}")
    if test_metrics:
        logger.info(f"Final test median error: {format_distance(test_metrics['median_error_km'])}")
    logger.info(f"Outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
