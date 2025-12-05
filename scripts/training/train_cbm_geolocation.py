#!/usr/bin/env python3
"""Training script for StreetCLIP CBM geolocation."""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
from pathlib import Path
from typing import Dict, Tuple, List, Optional
import logging
from datetime import datetime

import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt
import numpy as np
import wandb

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from torch.nn.utils import clip_grad_norm_

from src.config import DEFAULT_CONFIG
from src.dataset import (
    PanoramaCBMDataset,
    SubsetDataset,
    create_splits,
    create_splits_stratified,
    normalize_coordinates,
)
from src.evaluation import (
    compute_geolocation_metrics,
    sphere_to_normalized_latlng,
    denormalize_coordinates,
    compute_haversine_distance,
)
from src.losses import LossWeights, combined_loss
from src.models.cbm_geolocation import CBMGeolocationModel
from src.models.encoder_factory import create_encoder

# ============================================================================
# Constants
# ============================================================================

# Mapping from encoder model names to short names for directory/file naming
ENCODER_MODEL_TO_NAME = {
    "geolocal/StreetCLIP": "streetclip",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3",
    "facebook/dinov2-base": "dinov2",
}

# CLIP normalization constants for image display (mean and std as lists)
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]

# Visualization constants
VIZ_INTERVAL_EPOCHS = 5
VIZ_NUM_SAMPLES = 4
VIZ_FIGSIZE = (10, 8)
VIZ_DPI = 150
VIZ_TOP_K_CONCEPTS = 5


# ============================================================================
# Argument Parsing
# ============================================================================


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train StreetCLIP CBM geolocation model"
    )

    # --- Dataset and Data Loading ---
    parser.add_argument(
        "--data_root", type=str, default="data", help="Dataset root directory"
    )
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--country_filter", type=str, default=cfg.country_filter)
    parser.add_argument(
        "--require_coordinates", action="store_true", default=cfg.require_coordinates
    )
    parser.add_argument(
        "--stratified_concept_sampling",
        action="store_true",
        help="Use concept-level stratified splits instead of disjoint concept splits",
    )

    # --- Model Architecture ---
    parser.add_argument(
        "--encoder_model",
        type=str,
        default=cfg.encoder_model,
        help="Encoder model name (e.g., 'geolocal/StreetCLIP', 'facebook/dinov2-base')",
    )
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument(
        "--coordinate_loss_type",
        type=str,
        default=cfg.coordinate_loss_type,
        choices=("mse", "sphere", "haversine"),
        help="Coordinate loss/head type: 'mse' for lat/lng MSE, 'sphere' for 3D unit vectors, 'haversine' for great-circle distance.",
    )
    parser.add_argument(
        "--concept_to_coord_input",
        choices=("logits", "probs"),
        default="probs",
        help="Input representation for the coordinate head.",
    )
    parser.add_argument(
        "--coordinate_feature_skip_dim",
        type=int,
        default=256,
        help="Projected feature dimension concatenated to coordinate head input (0 disables).",
    )
    parser.add_argument(
        "--retain_coord_grad_through_concepts",
        action="store_true",
        help="Allow coordinate head gradients to flow back into concept logits.",
    )
    parser.add_argument(
        "--use_coordinate_residuals",
        action="store_true",
        help="Predict residual offsets around dataset centroid.",
    )
    parser.add_argument(
        "--residual_stats_source",
        choices=("train", "dataset"),
        default="train",
        help="Samples to use when computing centroid residual stats.",
    )

    # --- Training Configuration ---
    parser.add_argument("--sequential", action="store_true", default=cfg.sequential)
    parser.add_argument("--concept_epochs", type=int, default=cfg.stages.concept_epochs)
    parser.add_argument(
        "--prediction_epochs", type=int, default=cfg.stages.prediction_epochs
    )
    parser.add_argument(
        "--finetune_epochs", type=int, default=cfg.stages.finetune_epochs
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--grad_clip_norm",
        type=float,
        default=1.0,
        help="If > 0, clip gradients to this global norm.",
    )

    # --- Learning Rates ---
    parser.add_argument("--encoder_lr", type=float, default=cfg.encoder_lr)
    parser.add_argument("--cbm_lr", type=float, default=cfg.cbm_lr)
    parser.add_argument("--finetune_lr", type=float, default=cfg.finetune_lr)
    parser.add_argument(
        "--coordinate_head_lr",
        type=float,
        default=None,
        help="Override learning rate for coordinate head (defaults to cbm_lr).",
    )
    parser.add_argument(
        "--country_head_lr",
        type=float,
        default=None,
        help="Override learning rate for country head (defaults to cbm_lr).",
    )

    # --- Loss Weights ---
    parser.add_argument("--concept_weight", type=float, default=cfg.concept_weight)
    parser.add_argument("--distance_weight", type=float, default=cfg.distance_weight)
    parser.add_argument("--country_weight", type=float, default=cfg.country_weight)
    parser.add_argument(
        "--concept_stage_distance_weight",
        type=float,
        default=0.1,
        help="Distance loss weight to apply during concept stage (0 disables joint training).",
    )
    parser.add_argument(
        "--concept_stage_country_weight",
        type=float,
        default=0.0,
        help="Optional country loss weight during concept stage.",
    )

    # --- Checkpointing and Logging ---
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory (auto-generated if not provided)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)

    # --- Weights & Biases (W&B) Logging ---
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="streetclip-cbm-geolocation",
        help="W&B project name",
    )
    parser.add_argument(
        "--wandb_run_name",
        type=str,
        default=None,
        help="W&B run name (auto-generated if not provided)",
    )
    parser.add_argument(
        "--wandb_entity", type=str, default=None, help="W&B entity/team name"
    )
    parser.add_argument("--no_wandb", action="store_true", help="Disable wandb logging")
    parser.add_argument(
        "--experiment_name",
        type=str,
        default="geo_cbm_v2",
        help="Experiment name for organizing results",
    )

    # --- Diagnostics and Visualization ---
    parser.add_argument(
        "--diagnostics_interval",
        type=int,
        default=5,
        help="Epoch interval for dumping coordinate diagnostics.",
    )
    parser.add_argument(
        "--diagnostics_samples",
        type=int,
        default=64,
        help="Max samples to dump per diagnostics run.",
    )
    return parser.parse_args()


# ============================================================================
# Data Loading Utilities
# ============================================================================


def collate_batch(batch):
    images, concept_idx, country_idx, coords, metadata = zip(*batch)
    images = torch.stack(images)
    concept_idx = torch.tensor(concept_idx, dtype=torch.long)
    country_idx = torch.tensor(country_idx, dtype=torch.long)
    coords = torch.stack(coords)
    metadata = list(metadata)
    return images, concept_idx, country_idx, coords, metadata


def worker_init_fn(worker_id: int):
    """Initialize worker with seed for reproducibility."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


# ============================================================================
# Coordinate Utilities
# ============================================================================


def coords_tensor_from_samples(samples: List[Dict]) -> Optional[torch.Tensor]:
    coords = []
    for sample in samples:
        lat = sample.get("lat")
        lng = sample.get("lng")
        if lat is None or lng is None:
            continue
        coord = normalize_coordinates(lat, lng)
        coords.append(coord.unsqueeze(0))
    if not coords:
        return None
    return torch.cat(coords, dim=0)


def compute_coordinate_stats(
    samples: List[Dict],
) -> Optional[Tuple[torch.Tensor, torch.Tensor]]:
    coord_tensor = coords_tensor_from_samples(samples)
    if coord_tensor is None:
        return None
    center = coord_tensor.mean(dim=0)
    max_deviation = torch.max(torch.abs(coord_tensor - center), dim=0).values
    max_deviation = torch.clamp(max_deviation, min=1e-2)
    return center, max_deviation


# ============================================================================
# Training Utilities
# ============================================================================


def resolve_stage_loss_weights(
    stage: str,
    base_weights: LossWeights,
    concept_stage_distance_weight: float,
    concept_stage_country_weight: float,
) -> LossWeights:
    if stage == "concept":
        return LossWeights(
            concept=base_weights.concept,
            distance=concept_stage_distance_weight,
            country=concept_stage_country_weight,
        )
    if stage == "prediction":
        return LossWeights(
            concept=0.0,
            distance=base_weights.distance,
            country=base_weights.country,
        )
    return base_weights


def build_dataloaders(
    dataset: PanoramaCBMDataset,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
    stratify_concepts: bool = False,
):
    split_fn = create_splits_stratified if stratify_concepts else create_splits
    train_samples, val_samples, test_samples = split_fn(
        dataset.samples,
        train_ratio=train_ratio,
        val_ratio=val_ratio,
        test_ratio=test_ratio,
        seed=seed,
    )

    train_ds = SubsetDataset(dataset, train_samples)
    val_ds = SubsetDataset(dataset, val_samples)
    test_ds = SubsetDataset(dataset, test_samples)

    def loader(split_ds, shuffle):
        generator = torch.Generator()
        generator.manual_seed(seed)
        return DataLoader(
            split_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
            worker_init_fn=worker_init_fn if num_workers > 0 else None,
            generator=generator if shuffle else None,
        )

    return (
        loader(train_ds, True),
        loader(val_ds, False),
        loader(test_ds, False),
        train_samples,
        val_samples,
        test_samples,
    )


def move_batch_to_device(batch, device):
    images, concept_idx, country_idx, coords, metadata = batch
    return (
        images.to(device),
        concept_idx.to(device),
        country_idx.to(device),
        coords.to(device),
        metadata,
    )


def train_one_epoch(
    model,
    dataloader,
    optimizer,
    device,
    loss_weights: LossWeights,
    stage: str,
    coordinate_loss_type: str,
    concept_stage_distance_weight: float,
    concept_stage_country_weight: float,
    grad_clip_norm: float,
):
    model.train()
    running_loss = 0.0
    num_batches = 0
    stage_weights = resolve_stage_loss_weights(
        stage, loss_weights, concept_stage_distance_weight, concept_stage_country_weight
    )

    for batch in tqdm(dataloader, desc="Train", leave=False):
        optimizer.zero_grad()
        images, concept_idx, country_idx, coords, _ = move_batch_to_device(
            batch, device
        )
        concept_logits, country_logits, coord_preds = model(images)
        loss, _ = combined_loss(
            concept_logits,
            country_logits,
            coord_preds,
            concept_idx,
            country_idx,
            coords,
            stage_weights,
            coordinate_loss_type=coordinate_loss_type,
        )

        loss.backward()
        if grad_clip_norm and grad_clip_norm > 0:
            clip_grad_norm_(model.parameters(), grad_clip_norm)
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(
    model,
    dataloader,
    device,
    loss_weights: LossWeights,
    stage: str,
    coordinate_loss_type: str,
    concept_stage_distance_weight: float,
    concept_stage_country_weight: float,
):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    aggregated_metrics: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

    stage_weights = resolve_stage_loss_weights(
        stage, loss_weights, concept_stage_distance_weight, concept_stage_country_weight
    )

    for batch in tqdm(dataloader, desc="Eval", leave=False):
        images, concept_idx, country_idx, coords, _ = move_batch_to_device(
            batch, device
        )
        concept_logits, country_logits, coord_preds = model(images)
        loss, _ = combined_loss(
            concept_logits,
            country_logits,
            coord_preds,
            concept_idx,
            country_idx,
            coords,
            stage_weights,
            coordinate_loss_type=coordinate_loss_type,
        )
        running_loss += loss.item()
        num_batches += 1

        metrics = compute_geolocation_metrics(
            concept_logits,
            country_logits,
            coord_preds,
            concept_idx,
            country_idx,
            coords,
            coordinate_loss_type=coordinate_loss_type,
        )
        for key, value in metrics.items():
            if isinstance(value, float) and (value != value):
                continue
            aggregated_metrics[key] = aggregated_metrics.get(key, 0.0) + value
            metric_counts[key] = metric_counts.get(key, 0) + 1

    averaged_metrics = {
        key: aggregated_metrics[key] / metric_counts[key]
        for key in aggregated_metrics.keys()
    }

    return running_loss / max(num_batches, 1), averaged_metrics


def optimizer_for_stage(
    model: CBMGeolocationModel,
    stage: str,
    args: argparse.Namespace,
    train_prediction_head: bool = False,
    train_country_head: bool = False,
):
    param_groups = []
    stage = stage.lower()
    coordinate_lr = args.coordinate_head_lr or args.cbm_lr
    country_lr = args.country_head_lr or args.cbm_lr

    if stage == "concept":
        if args.finetune_encoder:
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
        param_groups.append(
            {"params": model.concept_layer.parameters(), "lr": args.cbm_lr}
        )
        if train_country_head:
            param_groups.append(
                {"params": model.country_head.parameters(), "lr": country_lr}
            )
        if train_prediction_head:
            param_groups.append(
                {"params": model.coordinate_parameters(), "lr": coordinate_lr}
            )

    elif stage == "prediction":
        param_groups.append(
            {"params": model.country_head.parameters(), "lr": country_lr}
        )
        param_groups.append(
            {"params": model.coordinate_parameters(), "lr": coordinate_lr}
        )

    elif stage == "finetune":
        if args.finetune_encoder:
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": args.finetune_lr})
        param_groups.append(
            {"params": model.concept_layer.parameters(), "lr": args.cbm_lr}
        )
        param_groups.append(
            {"params": model.country_head.parameters(), "lr": country_lr}
        )
        param_groups.append(
            {"params": model.coordinate_parameters(), "lr": coordinate_lr}
        )
    else:
        raise ValueError(f"Unknown stage {stage}")

    if not param_groups:
        raise ValueError("No parameters available for optimization in this stage")

    return torch.optim.AdamW(param_groups)


def save_checkpoint(model, optimizer, epoch, stage, path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "stage": stage,
            "model_state": model.state_dict(),
            "optimizer_state": optimizer.state_dict(),
        },
        path,
    )


def create_checkpoint_dir(
    experiment_name: str = "geo_cbm_v2",
    country_filter: str = None,
    sequential: bool = True,
    encoder_model: str = None,
) -> Path:
    """Create checkpoint directory with format:
    results/<experiment_name>/<training_type>/<encoder_name>/<country>/<timestamp>/
    where:
    - experiment_name: experiment identifier (default: "geo_cbm_v2")
    - training_type: "sequential" for sequential training, "joint" for joint training
    - encoder_name: sanitized encoder model name (replace "/" with "-")
    - country: country name or "global" if no country filter
    - timestamp: formatted date and time (YYYY-MM-DD_HH-MM-SS)

    Creates subdirectories: checkpoints/, logs/
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Training type: "sequential" or "joint"
    training_type = "sequential" if sequential else "joint"

    # Encoder name: sanitize model name (replace "/" with "-")
    if encoder_model:
        encoder_name = ENCODER_MODEL_TO_NAME.get(encoder_model, encoder_model)
        encoder_name = encoder_name.replace("/", "-")
    else:
        encoder_name = "unknown"

    # Country: use filter or "global" if None
    country = country_filter if country_filter else "global"

    # Create full directory structure
    timestamp_dir = (
        Path("results")
        / experiment_name
        / training_type
        / encoder_name
        / country
        / timestamp
    )
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (timestamp_dir / "checkpoints").mkdir(exist_ok=True)
    (timestamp_dir / "logs").mkdir(exist_ok=True)

    return timestamp_dir


# ============================================================================
# Visualization and Diagnostics
# ============================================================================


@torch.no_grad()
def visualize_predictions(
    model,
    dataloader,
    device,
    idx_to_concept,
    idx_to_country,
    output_dir: Path,
    epoch: int,
    num_samples: int = VIZ_NUM_SAMPLES,
    log_to_wandb: bool = True,
    wandb_step: int = None,
    coordinate_loss_type: str = "mse",
):
    """Visualize predictions with top 5 concepts as bar plots."""
    model.eval()
    output_dir.mkdir(parents=True, exist_ok=True)

    # Get a batch
    batch = next(iter(dataloader))
    images, concept_idx, country_idx, coords, metadata = move_batch_to_device(
        batch, device
    )

    # Get predictions
    concept_logits, country_logits, coord_preds = model(images)
    concept_probs = torch.softmax(concept_logits, dim=1)
    country_probs = torch.softmax(country_logits, dim=1)
    if coordinate_loss_type.lower() == "sphere":
        coord_preds_for_display = sphere_to_normalized_latlng(coord_preds)
    else:
        coord_preds_for_display = coord_preds

    # Process up to num_samples
    n_samples = min(num_samples, len(images))
    wandb_images = []

    for i in range(n_samples):
        fig, axes = plt.subplots(2, 1, figsize=VIZ_FIGSIZE)

        # Top: Image
        ax_img = axes[0]
        img = images[i].cpu()
        # Denormalize for display (CLIP normalization)
        img_denorm = img.clone()
        mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
        std = torch.tensor(CLIP_STD).view(3, 1, 1)
        img_denorm = img_denorm * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_display = img_denorm.permute(1, 2, 0).numpy()

        ax_img.imshow(img_display)
        ax_img.axis("off")

        # Add prediction info
        pred_country_idx = country_probs[i].argmax().item()
        pred_country = idx_to_country[pred_country_idx]
        true_country = idx_to_country[country_idx[i].item()]
        pred_coords = coord_preds_for_display[i].cpu().numpy()
        true_coords = coords[i].cpu().numpy()

        # Unnormalize the predicted and true coordinates
        pred_coords = denormalize_coordinates(pred_coords)
        true_coords = denormalize_coordinates(true_coords)

        # Get Haversine distance between predicted and true coordinates
        distance_km = compute_haversine_distance(pred_coords, true_coords)
        distance_km = distance_km.item()

        # Get ground truth concept
        true_concept_idx = concept_idx[i].item()
        true_concept = idx_to_concept[true_concept_idx]
        true_concept_prob = concept_probs[i][true_concept_idx].item()

        # Get image ID from metadata
        pano_id = metadata[i]["pano_id"]
        image_id = f"image_{pano_id}.jpg"

        title = f"Epoch {epoch} | Image ID: {image_id}\n"
        title += f"Pred: {pred_country} | True: {true_country}\n"
        title += f"Coords: Pred({pred_coords[0]:.3f}, {pred_coords[1]:.3f}) | True({true_coords[0]:.3f}, {true_coords[1]:.3f}) | Distance: {distance_km:.1f} km\n"
        title += f"GT Concept: {true_concept} ({true_concept_prob:.3f})"
        ax_img.set_title(title, fontsize=10)

        # Bottom: Top K concepts bar plot
        ax_bar = axes[1]
        top5_probs, top5_indices = torch.topk(concept_probs[i], k=VIZ_TOP_K_CONCEPTS)
        top5_concepts = [idx_to_concept[idx.item()] for idx in top5_indices]
        top5_probs_np = top5_probs.cpu().numpy()

        # Check if ground truth is in top 5
        true_in_top5 = true_concept_idx in top5_indices.cpu().numpy()

        # Reverse arrays to show highest probability at top (descending order)
        top5_concepts_reversed = list(reversed(top5_concepts))
        top5_probs_np_reversed = top5_probs_np[::-1]
        top5_indices_reversed = list(reversed(top5_indices.cpu().numpy()))

        # Color bars: highlight ground truth if in top 5, otherwise use default color
        bar_colors = [
            "orange" if idx == true_concept_idx else "steelblue"
            for idx in top5_indices_reversed
        ]

        bars = ax_bar.barh(
            range(len(top5_concepts_reversed)),
            top5_probs_np_reversed,
            color=bar_colors,
        )
        ax_bar.set_yticks(range(len(top5_concepts_reversed)))

        # Add (GT) label to ground truth concept in y-axis labels
        yticklabels = []
        for concept, idx in zip(top5_concepts_reversed, top5_indices_reversed):
            if idx == true_concept_idx:
                yticklabels.append(f"{concept} (GT)")
            else:
                yticklabels.append(concept)
        ax_bar.set_yticklabels(yticklabels)

        ax_bar.set_xlabel("Probability", fontsize=10)
        title = "Top 5 Predicted Concepts"
        if not true_in_top5:
            title += f" | GT: {true_concept} ({true_concept_prob:.3f})"
        ax_bar.set_title(title, fontsize=10)
        ax_bar.set_xlim(0, 1)

        # Add value labels on bars
        for j, (bar, prob) in enumerate(zip(bars, top5_probs_np_reversed)):
            ax_bar.text(prob + 0.01, j, f"{prob:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        save_path = output_dir / f"epoch_{epoch}_sample_{i}.png"
        plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")

        if log_to_wandb:
            wandb_images.append(wandb.Image(str(save_path), caption=f"Sample {i}"))

        plt.close(fig)

    logger.info(f"Saved {n_samples} visualization(s) to {output_dir} for epoch {epoch}")

    if log_to_wandb and wandb_images:
        step = wandb_step if wandb_step is not None else epoch
        wandb.log({f"predictions/epoch_{epoch}": wandb_images}, step=step)


@torch.no_grad()
def dump_coordinate_diagnostics(
    model,
    dataloader,
    device,
    output_path: Path,
    coordinate_loss_type: str,
    max_samples: int = 64,
    log_to_wandb: bool = False,
    wandb_step: Optional[int] = None,
    idx_to_concept: Optional[Dict[int, str]] = None,
    idx_to_country: Optional[Dict[int, str]] = None,
):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    coord_type = coordinate_loss_type.lower()

    for batch in dataloader:
        images, concept_idx, country_idx, coords, metadata = move_batch_to_device(
            batch, device
        )
        concept_logits, country_logits, coord_preds = model(images)
        if coord_type == "sphere":
            coord_preds_norm = sphere_to_normalized_latlng(coord_preds)
        else:
            coord_preds_norm = coord_preds

        concept_probs = torch.softmax(concept_logits, dim=1)
        country_probs = torch.softmax(country_logits, dim=1)

        for i in range(len(images)):
            if len(rows) >= max_samples:
                break
            true_coords = coords[i].detach().cpu()
            pred_coords = coord_preds_norm[i].detach().cpu()
            pred_deg = denormalize_coordinates(pred_coords)
            true_deg = denormalize_coordinates(true_coords)
            distance_km = float(
                compute_haversine_distance(pred_deg, true_deg).cpu().item()
            )

            pred_country_idx = country_probs[i].argmax().item()
            pred_country = (
                idx_to_country[pred_country_idx]
                if idx_to_country is not None
                else str(pred_country_idx)
            )
            true_country = (
                idx_to_country[country_idx[i].item()]
                if idx_to_country is not None
                else str(country_idx[i].item())
            )

            top_concept_idx = concept_probs[i].argmax().item()
            top_concept = (
                idx_to_concept[top_concept_idx]
                if idx_to_concept is not None
                else str(top_concept_idx)
            )
            true_concept = (
                idx_to_concept[concept_idx[i].item()]
                if idx_to_concept is not None
                else str(concept_idx[i].item())
            )

            rows.append(
                {
                    "pano_id": metadata[i]["pano_id"],
                    "pred_lat": float(pred_deg[0].item()),
                    "pred_lng": float(pred_deg[1].item()),
                    "true_lat": float(true_deg[0].item()),
                    "true_lng": float(true_deg[1].item()),
                    "distance_km": distance_km,
                    "pred_country": pred_country,
                    "true_country": true_country,
                    "top_concept": top_concept,
                    "true_concept": true_concept,
                    "country_correct": bool(pred_country_idx == country_idx[i].item()),
                    "concept_correct": bool(top_concept_idx == concept_idx[i].item()),
                }
            )
        if len(rows) >= max_samples:
            break

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    if log_to_wandb:
        table = wandb.Table(columns=fieldnames)
        for row in rows:
            table.add_data(*[row[col] for col in fieldnames])
        wandb.log({f"diagnostics/{output_path.stem}": table}, step=wandb_step)


# ============================================================================
# Main Training Function
# ============================================================================


def main():
    args = parse_args()
    device = torch.device(args.device)

    # --- Setup: Seeds and Reproducibility ---
    seed = args.seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    logger.info(f"Set random seed to {seed} for reproducibility")

    # --- Setup: Encoder Model Selection ---
    encoder_model = args.encoder_model

    # --- Setup: Checkpoint Directory ---
    if args.checkpoint_dir is None:
        checkpoint_dir = create_checkpoint_dir(
            args.experiment_name, args.country_filter, args.sequential, encoder_model
        )
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Ensure subdirectories exist
        (checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
        (checkpoint_dir / "logs").mkdir(exist_ok=True)

    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # --- Setup: File Logging ---
    log_file = checkpoint_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # --- Setup: Weights & Biases (W&B) Logging ---
    if not args.no_wandb:
        wandb_config = {
            "experiment_name": args.experiment_name,
            "batch_size": args.batch_size,
            "encoder_model": encoder_model,
            "finetune_encoder": args.finetune_encoder,
            "encoder_lr": args.encoder_lr,
            "cbm_lr": args.cbm_lr,
            "finetune_lr": args.finetune_lr,
            "concept_weight": args.concept_weight,
            "distance_weight": args.distance_weight,
            "country_weight": args.country_weight,
            "coordinate_loss_type": args.coordinate_loss_type,
            "concept_epochs": args.concept_epochs,
            "prediction_epochs": args.prediction_epochs,
            "finetune_epochs": args.finetune_epochs,
            "sequential": args.sequential,
            "country_filter": args.country_filter,
            "val_ratio": args.val_ratio,
            "test_ratio": args.test_ratio,
            "seed": args.seed,
            "checkpoint_dir": str(checkpoint_dir),
        }

        # Get clean encoder name for run name and tags
        encoder_name = ENCODER_MODEL_TO_NAME.get(encoder_model, encoder_model)
        encoder_name = encoder_name.replace("/", "-")

        # Get country name
        country = args.country_filter if args.country_filter else "global"

        # Get loss type
        loss_type = args.coordinate_loss_type

        run_name = args.wandb_run_name
        if run_name is None:
            run_name = f"train-cbm-{country}-{encoder_name}-{loss_type}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
        if args.stratified_concept_sampling:
            run_name += "-stratified"
            strat_tag = "stratified"
        else:
            strat_tag = "disjoint"
            run_name += "-disjoint"

        if args.sequential:
            run_name += "-sequential"
            seq_tag = "sequential"
        else:
            seq_tag = "non-sequential"
            run_name += "-non-sequential"

        if args.finetune_encoder:
            finetune_tag = "finetuned"
            run_name += "-finetuned"
        else:
            finetune_tag = "not-finetuned"
            run_name += "-not-finetuned"

        # Create tags list
        tags = [country, encoder_name, loss_type, strat_tag, seq_tag, finetune_tag]

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=wandb_config,
            tags=tags,
            dir=str(checkpoint_dir),
        )
        logger.info(f"Initialized wandb run: {run_name}")

    # --- Dataset and Data Loaders ---
    dataset = PanoramaCBMDataset(
        transform=None,
        image_size=(DEFAULT_CONFIG.image_size, DEFAULT_CONFIG.image_size),
        max_samples=args.max_samples,
        country=args.country_filter,
        require_coordinates=args.require_coordinates,
        encoder_model=encoder_model,
        use_normalized_coordinates=True,
    )

    logger.info(f"Dataset image size: {dataset.image_size}")

    (
        train_loader,
        val_loader,
        test_loader,
        train_samples,
        val_samples,
        test_samples,
    ) = build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=1 - args.val_ratio - args.test_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
        stratify_concepts=args.stratified_concept_sampling,
    )

    # --- Coordinate Residual Statistics (if enabled) ---
    residual_center = None
    residual_bounds = None
    if args.use_coordinate_residuals:
        if args.residual_stats_source == "train":
            stats_samples = train_samples
            stats_name = "train split"
        else:
            stats_samples = dataset.samples
            stats_name = "full dataset"
        stats = compute_coordinate_stats(stats_samples)
        if stats is None:
            logger.warning(
                "Unable to compute coordinate residual stats (missing coordinates)."
            )
        else:
            residual_center, residual_bounds = stats
            logger.info(
                f"Coordinate residual center ({stats_name}): lat={residual_center[0]:.3f}, lng={residual_center[1]:.3f}"
            )
            logger.info(
                f"Coordinate residual bounds: lat<=±{residual_bounds[0]:.3f}, lng<=±{residual_bounds[1]:.3f}"
            )

    # --- Model Creation ---
    # Create encoder using factory
    encoder = create_encoder(
        model_name=encoder_model,
        finetune=args.finetune_encoder,
        device=device,
    )
    feature_dim = encoder.feature_dim

    logger.info(f"Using encoder: {encoder_model}")
    logger.info(f"Encoder type: {encoder.encoder_type}")
    logger.info(f"Encoder feature dimension: {feature_dim}")
    logger.info(f"Coordinate loss type: {args.coordinate_loss_type}")

    model = CBMGeolocationModel(
        encoder=encoder,
        num_concepts=len(dataset.concept_to_idx),
        num_countries=len(dataset.country_to_idx),
        feature_dim=feature_dim,
        coordinate_loss_type=args.coordinate_loss_type,
        coordinate_input=args.concept_to_coord_input,
        coordinate_feature_skip_dim=args.coordinate_feature_skip_dim,
        detach_concepts_for_prediction=not args.retain_coord_grad_through_concepts,
        coordinate_residual_center=(
            residual_center
            if residual_center is None
            else residual_center.to(torch.float32)
        ),
        coordinate_residual_bounds=(
            residual_bounds
            if residual_bounds is None
            else residual_bounds.to(torch.float32)
        ),
    ).to(device)

    # --- Model Information Logging ---
    def count_parameters(module):
        return sum(p.numel() for p in module.parameters() if p.requires_grad)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    encoder_params = count_parameters(model.encoder)
    concept_params = count_parameters(model.concept_layer)
    country_params = count_parameters(model.country_head)
    coord_params = count_parameters(model.coordinate_head)

    logger.info(
        f"Model dimensions: concepts={len(dataset.concept_to_idx)}, countries={len(dataset.country_to_idx)}, feature_dim={feature_dim}"
    )
    logger.info(
        f"Parameter counts - Total: {total_params:,}, Trainable: {trainable_params:,}"
    )
    logger.info(
        f"  Encoder: {encoder_params:,}, Concept layer: {concept_params:,}, Country head: {country_params:,}, Coordinate head: {coord_params:,}"
    )

    # Log model info to wandb
    if not args.no_wandb:
        wandb.config.update(
            {
                "num_concepts": len(dataset.concept_to_idx),
                "num_countries": len(dataset.country_to_idx),
                "feature_dim": feature_dim,
                "total_params": total_params,
                "trainable_params": trainable_params,
                "encoder_params": encoder_params,
                "concept_params": concept_params,
                "country_params": country_params,
                "coord_params": coord_params,
            }
        )

    # --- Training Configuration ---
    loss_weights = LossWeights(
        concept=args.concept_weight,
        distance=args.distance_weight,
        country=args.country_weight,
    )

    stages: Tuple[Tuple[str, int], ...]
    if args.sequential:
        stages = tuple(
            (name, epochs)
            for name, epochs in [
                ("concept", args.concept_epochs),
                ("prediction", args.prediction_epochs),
                ("finetune", args.finetune_epochs),
            ]
            if epochs > 0
        )
    else:
        total_epochs = (
            args.concept_epochs + args.prediction_epochs + args.finetune_epochs
        )
        if total_epochs == 0:
            total_epochs = 1
        stages = (("finetune", total_epochs),)

    # --- Training Loop ---
    global_step = 0  # Track global step across all stages for wandb
    for stage_name, epochs in stages:
        print(f"Starting stage: {stage_name} for {epochs} epochs")
        concept_stage_prediction = args.concept_stage_distance_weight > 0.0
        concept_stage_country = args.concept_stage_country_weight > 0.0
        train_prediction_head = (
            concept_stage_prediction if stage_name == "concept" else True
        )
        train_country_head = concept_stage_country if stage_name == "concept" else True
        model.set_stage(
            stage_name,
            finetune_encoder=args.finetune_encoder and stage_name != "prediction",
            train_prediction_head=train_prediction_head,
            train_country_head=train_country_head,
        )
        optimizer = optimizer_for_stage(
            model,
            stage_name,
            args,
            train_prediction_head=(
                train_prediction_head if stage_name == "concept" else True
            ),
            train_country_head=train_country_head if stage_name == "concept" else True,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )

        # --- Epoch Loop ---
        for epoch in range(1, epochs + 1):
            global_step += 1  # Increment global step for wandb
            print(f"Epoch {epoch}/{epochs} (Stage: {stage_name})")
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
                loss_weights,
                stage_name,
                args.coordinate_loss_type,
                args.concept_stage_distance_weight,
                args.concept_stage_country_weight,
                args.grad_clip_norm,
            )
            val_loss, val_metrics = evaluate(
                model,
                val_loader,
                device,
                loss_weights,
                stage_name,
                args.coordinate_loss_type,
                args.concept_stage_distance_weight,
                args.concept_stage_country_weight,
            )
            scheduler.step(val_loss)

            print(f"Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f}")
            for key, value in val_metrics.items():
                print(f"  {key}: {value:.4f}")

            # Log to wandb
            if not args.no_wandb:
                log_dict = {
                    f"{stage_name}/train_loss": train_loss,
                    f"{stage_name}/val_loss": val_loss,
                }
                for key, value in val_metrics.items():
                    log_dict[f"{stage_name}/val_{key}"] = value
                wandb.log(log_dict, step=global_step)

            if epoch % args.checkpoint_interval == 0:
                ckpt_path = (
                    checkpoint_dir
                    / "checkpoints"
                    / f"stage-{stage_name}-epoch-{epoch}.pt"
                )
                save_checkpoint(model, optimizer, epoch, stage_name, ckpt_path)

            # --- Visualization (every N epochs) ---
            if epoch % VIZ_INTERVAL_EPOCHS == 0:
                viz_dir = checkpoint_dir / "visualizations" / stage_name
                visualize_predictions(
                    model,
                    val_loader,
                    device,
                    dataset.idx_to_concept,
                    dataset.idx_to_country,
                    viz_dir,
                    epoch,
                    num_samples=VIZ_NUM_SAMPLES,
                    log_to_wandb=not args.no_wandb,
                    wandb_step=global_step,
                    coordinate_loss_type=args.coordinate_loss_type,
                )

            # --- Diagnostics (every N epochs) ---
            if args.diagnostics_interval > 0 and epoch % args.diagnostics_interval == 0:
                diag_dir = checkpoint_dir / "diagnostics" / stage_name
                diag_file = diag_dir / f"epoch_{epoch}.csv"
                dump_coordinate_diagnostics(
                    model,
                    val_loader,
                    device,
                    diag_file,
                    args.coordinate_loss_type,
                    max_samples=args.diagnostics_samples,
                    log_to_wandb=not args.no_wandb,
                    wandb_step=global_step,
                    idx_to_concept=dataset.idx_to_concept,
                    idx_to_country=dataset.idx_to_country,
                )

    # --- Final Test Evaluation ---
    print("Evaluating on test split...")
    # Use the last stage for test evaluation (or "finetune" if available, otherwise last stage)
    test_stage = stages[-1][0] if stages else "finetune"
    test_loss, test_metrics = evaluate(
        model,
        test_loader,
        device,
        loss_weights,
        test_stage,
        args.coordinate_loss_type,
        args.concept_stage_distance_weight,
        args.concept_stage_country_weight,
    )
    print(f"Test Loss: {test_loss:.4f}")
    for key, value in test_metrics.items():
        print(f"  {key}: {value:.4f}")

    # Log test metrics to wandb
    if not args.no_wandb:
        log_dict = {"test/test_loss": test_loss}
        for key, value in test_metrics.items():
            log_dict[f"test/test_{key}"] = value
        wandb.log(log_dict)
        wandb.finish()

    # --- Save Training Summary ---
    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
