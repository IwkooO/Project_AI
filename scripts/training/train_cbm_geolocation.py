#!/usr/bin/env python3
"""Training script for StreetCLIP CBM geolocation."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Dict, Tuple
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

from src.config import DEFAULT_CONFIG, FEATURE_DIM_BY_MODEL
from src.dataset import PanoramaCBMDataset, SubsetDataset, create_splits
from src.evaluation import compute_geolocation_metrics
from src.losses import LossWeights, combined_loss
from src.models.cbm_geolocation import CBMGeolocationModel
from src.models.streetclip_encoder import StreetCLIPConfig, StreetCLIPEncoder


def parse_args() -> argparse.Namespace:
    cfg = DEFAULT_CONFIG
    parser = argparse.ArgumentParser(
        description="Train StreetCLIP CBM geolocation model"
    )
    parser.add_argument(
        "--data_root", type=str, default="data", help="Dataset root directory"
    )
    parser.add_argument("--batch_size", type=int, default=cfg.batch_size)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--streetclip_model", type=str, default=cfg.streetclip_model)
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--encoder_lr", type=float, default=cfg.encoder_lr)
    parser.add_argument("--cbm_lr", type=float, default=cfg.cbm_lr)
    parser.add_argument("--finetune_lr", type=float, default=cfg.finetune_lr)
    parser.add_argument("--concept_weight", type=float, default=cfg.concept_weight)
    parser.add_argument("--distance_weight", type=float, default=cfg.distance_weight)
    parser.add_argument("--country_weight", type=float, default=cfg.country_weight)
    parser.add_argument("--concept_epochs", type=int, default=cfg.stages.concept_epochs)
    parser.add_argument(
        "--prediction_epochs", type=int, default=cfg.stages.prediction_epochs
    )
    parser.add_argument(
        "--finetune_epochs", type=int, default=cfg.stages.finetune_epochs
    )
    parser.add_argument("--sequential", action="store_true", default=cfg.sequential)
    parser.add_argument("--country_filter", type=str, default=cfg.country_filter)
    parser.add_argument(
        "--require_coordinates", action="store_true", default=cfg.require_coordinates
    )
    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default=None,
        help="Checkpoint directory (auto-generated if not provided)",
    )
    parser.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument("--max_samples", type=int, default=None)
    parser.add_argument("--val_ratio", type=float, default=0.15)
    parser.add_argument("--test_ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--checkpoint_interval", type=int, default=1)
    parser.add_argument("--resume_from", type=str, default=None)
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
    return parser.parse_args()


def collate_batch(batch):
    images, concept_idx, country_idx, coords, metadata = zip(*batch)
    images = torch.stack(images)
    concept_idx = torch.tensor(concept_idx, dtype=torch.long)
    country_idx = torch.tensor(country_idx, dtype=torch.long)
    coords = torch.stack(coords)
    metadata = list(metadata)
    return images, concept_idx, country_idx, coords, metadata


def build_dataloaders(
    dataset: PanoramaCBMDataset,
    batch_size: int,
    num_workers: int,
    train_ratio: float,
    val_ratio: float,
    test_ratio: float,
    seed: int,
):
    train_samples, val_samples, test_samples = create_splits(
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
        return DataLoader(
            split_ds,
            batch_size=batch_size,
            shuffle=shuffle,
            num_workers=num_workers,
            pin_memory=True,
            collate_fn=collate_batch,
        )

    return loader(train_ds, True), loader(val_ds, False), loader(test_ds, False)


def move_batch_to_device(batch, device):
    images, concept_idx, country_idx, coords, metadata = batch
    return (
        images.to(device),
        concept_idx.to(device),
        country_idx.to(device),
        coords.to(device),
        metadata,
    )


def train_one_epoch(model, dataloader, optimizer, device, loss_weights: LossWeights):
    model.train()
    running_loss = 0.0
    num_batches = 0

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
            loss_weights,
        )
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
        num_batches += 1

    return running_loss / max(num_batches, 1)


@torch.no_grad()
def evaluate(model, dataloader, device, loss_weights: LossWeights):
    model.eval()
    running_loss = 0.0
    num_batches = 0
    aggregated_metrics: Dict[str, float] = {}
    metric_counts: Dict[str, int] = {}

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
            loss_weights,
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
    model: CBMGeolocationModel, stage: str, args: argparse.Namespace
):
    param_groups = []
    stage = stage.lower()

    if stage == "concept":
        if args.finetune_encoder:
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": args.encoder_lr})
        param_groups.append(
            {"params": model.concept_layer.parameters(), "lr": args.cbm_lr}
        )

    elif stage == "prediction":
        param_groups.append(
            {"params": model.country_head.parameters(), "lr": args.cbm_lr}
        )
        param_groups.append(
            {"params": model.coordinate_head.parameters(), "lr": args.cbm_lr}
        )

    elif stage == "finetune":
        if args.finetune_encoder:
            encoder_params = [p for p in model.encoder.parameters() if p.requires_grad]
            if encoder_params:
                param_groups.append({"params": encoder_params, "lr": args.finetune_lr})
        param_groups.append(
            {
                "params": list(model.concept_layer.parameters())
                + list(model.country_head.parameters())
                + list(model.coordinate_head.parameters()),
                "lr": args.cbm_lr,
            }
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


def create_checkpoint_dir(country_filter: str = None, sequential: bool = True) -> Path:
    """Create checkpoint directory with format: results/checkpoints/sequential/<dd-mm-yy-TIME-train-cbm-<country_filter>/"""
    now = datetime.now()
    date_str = now.strftime("%d-%m-%y")
    time_str = now.strftime("%H-%M-%S")
    timestamp = f"{date_str}-{time_str}"

    country_suffix = f"-{country_filter}" if country_filter else ""
    mode = "sequential" if sequential else "joint"

    dir_name = f"{timestamp}-train-cbm{country_suffix}"
    checkpoint_dir = Path("results") / "checkpoints" / mode / dir_name
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    return checkpoint_dir


@torch.no_grad()
def visualize_predictions(
    model,
    dataloader,
    device,
    idx_to_concept,
    idx_to_country,
    output_dir: Path,
    epoch: int,
    num_samples: int = 4,
    log_to_wandb: bool = True,
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

    # Process up to num_samples
    n_samples = min(num_samples, len(images))
    wandb_images = []

    for i in range(n_samples):
        fig, axes = plt.subplots(2, 1, figsize=(10, 8))

        # Top: Image
        ax_img = axes[0]
        img = images[i].cpu()
        # Denormalize for display (CLIP normalization)
        img_denorm = img.clone()
        mean = torch.tensor([0.48145466, 0.4578275, 0.40821073]).view(3, 1, 1)
        std = torch.tensor([0.26862954, 0.26130258, 0.27577711]).view(3, 1, 1)
        img_denorm = img_denorm * std + mean
        img_denorm = torch.clamp(img_denorm, 0, 1)
        img_display = img_denorm.permute(1, 2, 0).numpy()

        ax_img.imshow(img_display)
        ax_img.axis("off")

        # Add prediction info
        pred_country_idx = country_probs[i].argmax().item()
        pred_country = idx_to_country[pred_country_idx]
        true_country = idx_to_country[country_idx[i].item()]
        pred_coords = coord_preds[i].cpu().numpy()
        true_coords = coords[i].cpu().numpy()

        title = f"Epoch {epoch} | Pred: {pred_country} | True: {true_country}\n"
        title += f"Coords: Pred({pred_coords[0]:.3f}, {pred_coords[1]:.3f}) | True({true_coords[0]:.3f}, {true_coords[1]:.3f})"
        ax_img.set_title(title, fontsize=10)

        # Bottom: Top 5 concepts bar plot
        ax_bar = axes[1]
        top5_probs, top5_indices = torch.topk(concept_probs[i], k=5)
        top5_concepts = [idx_to_concept[idx.item()] for idx in top5_indices]
        top5_probs_np = top5_probs.cpu().numpy()

        bars = ax_bar.barh(range(len(top5_concepts)), top5_probs_np, color="steelblue")
        ax_bar.set_yticks(range(len(top5_concepts)))
        ax_bar.set_yticklabels(top5_concepts)
        ax_bar.set_xlabel("Probability", fontsize=10)
        ax_bar.set_title("Top 5 Predicted Concepts", fontsize=10)
        ax_bar.set_xlim(0, 1)

        # Add value labels on bars
        for j, (bar, prob) in enumerate(zip(bars, top5_probs_np)):
            ax_bar.text(prob + 0.01, j, f"{prob:.3f}", va="center", fontsize=9)

        plt.tight_layout()
        save_path = output_dir / f"epoch_{epoch}_sample_{i}.png"
        plt.savefig(save_path, dpi=150, bbox_inches="tight")

        if log_to_wandb:
            wandb_images.append(wandb.Image(str(save_path), caption=f"Sample {i}"))

        plt.close(fig)

    logger.info(f"Saved {n_samples} visualization(s) to {output_dir} for epoch {epoch}")

    if log_to_wandb and wandb_images:
        wandb.log({f"predictions/epoch_{epoch}": wandb_images}, step=epoch)


def main():
    args = parse_args()
    device = torch.device(args.device)

    # Create checkpoint directory
    if args.checkpoint_dir is None:
        checkpoint_dir = create_checkpoint_dir(args.country_filter, args.sequential)
    else:
        checkpoint_dir = Path(args.checkpoint_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"Checkpoint directory: {checkpoint_dir}")

    # Initialize wandb
    if not args.no_wandb:
        wandb_config = {
            "batch_size": args.batch_size,
            "streetclip_model": args.streetclip_model,
            "finetune_encoder": args.finetune_encoder,
            "encoder_lr": args.encoder_lr,
            "cbm_lr": args.cbm_lr,
            "finetune_lr": args.finetune_lr,
            "concept_weight": args.concept_weight,
            "distance_weight": args.distance_weight,
            "country_weight": args.country_weight,
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

        run_name = args.wandb_run_name
        if run_name is None:
            country_suffix = f"-{args.country_filter}" if args.country_filter else ""
            run_name = (
                f"train-cbm{country_suffix}-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
            )

        wandb.init(
            project=args.wandb_project,
            name=run_name,
            entity=args.wandb_entity,
            config=wandb_config,
            dir=str(checkpoint_dir),
        )
        logger.info(f"Initialized wandb run: {run_name}")

    dataset = PanoramaCBMDataset(
        transform=None,
        image_size=(DEFAULT_CONFIG.image_size, DEFAULT_CONFIG.image_size),
        max_samples=args.max_samples,
        country=args.country_filter,
        require_coordinates=args.require_coordinates,
    )

    train_loader, val_loader, test_loader = build_dataloaders(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        train_ratio=1 - args.val_ratio - args.test_ratio,
        val_ratio=args.val_ratio,
        test_ratio=args.test_ratio,
        seed=args.seed,
    )

    encoder_config = StreetCLIPConfig(
        model_name=args.streetclip_model,
        finetune=args.finetune_encoder,
        device=device,
    )
    encoder = StreetCLIPEncoder(encoder_config)
    feature_dim = FEATURE_DIM_BY_MODEL.get(args.streetclip_model, encoder.feature_dim)

    logger.info(f"Encoder feature dimension: {feature_dim}")

    model = CBMGeolocationModel(
        encoder=encoder,
        num_concepts=len(dataset.concept_to_idx),
        num_countries=len(dataset.country_to_idx),
        feature_dim=feature_dim,
    ).to(device)

    # Log model dimensions and parameter counts
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

    for stage_name, epochs in stages:
        print(f"Starting stage: {stage_name} for {epochs} epochs")
        model.set_stage(
            stage_name,
            finetune_encoder=args.finetune_encoder and stage_name != "prediction",
        )
        optimizer = optimizer_for_stage(model, stage_name, args)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=2
        )

        for epoch in range(1, epochs + 1):
            print(f"Epoch {epoch}/{epochs} (Stage: {stage_name})")
            train_loss = train_one_epoch(
                model, train_loader, optimizer, device, loss_weights
            )
            val_loss, val_metrics = evaluate(model, val_loader, device, loss_weights)
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
                wandb.log(log_dict, step=epoch)

            if epoch % args.checkpoint_interval == 0:
                ckpt_path = checkpoint_dir / f"stage-{stage_name}-epoch-{epoch}.pt"
                save_checkpoint(model, optimizer, epoch, stage_name, ckpt_path)

            # Visualize predictions every 5th epoch
            if epoch % 5 == 0:
                viz_dir = checkpoint_dir / "visualizations" / stage_name
                visualize_predictions(
                    model,
                    val_loader,
                    device,
                    dataset.idx_to_concept,
                    dataset.idx_to_country,
                    viz_dir,
                    epoch,
                    num_samples=4,
                    log_to_wandb=not args.no_wandb,
                )

    print("Evaluating on test split...")
    test_loss, test_metrics = evaluate(model, test_loader, device, loss_weights)
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

    summary_path = checkpoint_dir / "training_summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with summary_path.open("w") as f:
        json.dump({"test_loss": test_loss, "test_metrics": test_metrics}, f, indent=2)


if __name__ == "__main__":
    main()
