#!/usr/bin/env python3
"""
Swap/Shuffle contribution tests for Stage 2.

Goal: quantify how much the model relies on concepts vs image patches by measuring
median distance error (km) under controlled corruption:
  - baseline: (concept_emb, patch_tokens)
  - concept_shuffled: shuffle concept_emb across batch, keep patches fixed
  - patches_shuffled: shuffle patch_tokens across batch, keep concepts fixed

This is intentionally separate from the training script to avoid code pollution.
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.losses import haversine_distance
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead
from src.models.streetclip_encoder import StreetCLIPConfig, StreetCLIPEncoder

# Reuse dataset + helpers from training script (keeps this file small and consistent)
from scripts.training.train_stage2_cross_attention import (
    Stage2ImageDataset,
    stage2_collate_fn,
    load_stage1_checkpoint,
    cartesian_to_latlng,
    cell_center_to_latlng,
)


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def compute_predicted_coords(
    pred_cells: torch.Tensor,
    pred_offsets: torch.Tensor,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    device: torch.device,
) -> torch.Tensor:
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
def evaluate_variant(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model,
    loader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    variant: str,
) -> Dict[str, float]:
    """
    variant:
      - baseline
      - concept_shuffled
      - patches_shuffled
    """
    model.eval()
    stage1_model.eval()
    image_encoder.model.eval()

    all_dists: List[float] = []

    for batch in tqdm(loader, desc=f"Eval ({variant})"):
        images, coordinates, cell_labels, countries, image_paths = batch
        images = images.to(device)
        coordinates = coordinates.to(device)

        patch_tokens = image_encoder.get_patch_tokens(images)  # [B, 576, 1024]
        img_features = image_encoder(images)  # [B, 768]
        concept_embs = stage1_model.concept_bottleneck(img_features)  # [B, 512]

        bsz = concept_embs.size(0)
        if variant == "concept_shuffled":
            perm = torch.randperm(bsz, device=device)
            concept_embs = concept_embs[perm]
        elif variant == "patches_shuffled":
            perm = torch.randperm(bsz, device=device)
            patch_tokens = patch_tokens[perm]
        elif variant != "baseline":
            raise ValueError(f"Unknown variant: {variant}")

        outputs = model(concept_embs, patch_tokens, return_attention=False, return_gate=False)
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]

        pred_cells = cell_logits.argmax(dim=1)
        pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)
        dists = haversine_distance(pred_coords, coordinates)
        all_dists.extend(dists.detach().cpu().numpy().tolist())

    median_km = float(np.median(all_dists)) if all_dists else float("inf")
    return {"median_error_km": median_km}


def main():
    parser = argparse.ArgumentParser(description="Stage2 swap/shuffle contribution tests")
    parser.add_argument("--stage2_checkpoint", type=str, required=True, help="Path to Stage2 checkpoint (.pt)")
    parser.add_argument("--stage1_checkpoint", type=str, default=None, help="Optional: override Stage1 checkpoint path")

    # Dataset args (must match what you trained/evaluated on)
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--geoguessr_id", type=str, default="6906237dc7731161a37282b2")
    parser.add_argument("--val_split", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--max_batches", type=int, default=None, help="Optional: limit number of eval batches")

    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    stage2_ckpt_path = Path(args.stage2_checkpoint)
    ckpt = torch.load(stage2_ckpt_path, map_location=device)

    encoder_model = ckpt.get("encoder_model", "geolocal/StreetCLIP")
    coord_output_dim = int(ckpt.get("coord_output_dim", 3))
    ablation_mode = ckpt.get("ablation_mode", "both")
    cell_centers = ckpt["cell_centers"].to(device)
    num_cells = int(ckpt.get("num_cells", cell_centers.size(0)))

    logger.info(f"Checkpoint: {stage2_ckpt_path}")
    logger.info(f"  - ablation_mode: {ablation_mode}")
    logger.info(f"  - coord_output_dim: {coord_output_dim}")
    logger.info(f"  - num_cells: {num_cells}")

    # Image encoder
    encoder_config = StreetCLIPConfig(model_name=encoder_model, finetune=False, device=device)
    image_encoder = StreetCLIPEncoder(encoder_config)
    image_encoder.model.eval()
    for p in image_encoder.model.parameters():
        p.requires_grad = False

    # Stage1 model
    stage1_checkpoint = args.stage1_checkpoint or ckpt.get("stage1_checkpoint")
    if stage1_checkpoint is None:
        raise ValueError("Stage1 checkpoint path not provided and not found in Stage2 checkpoint.")
    stage1_model, _ = load_stage1_checkpoint(Path(stage1_checkpoint), image_encoder, device)

    # Load dataset (same logic as training script)
    import pandas as pd

    df = pd.read_csv(args.csv_path)
    image_paths = []
    coordinates = []
    countries = []

    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building dataset"):
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

    coordinates = torch.stack(coordinates)
    cell_labels = torch.zeros(len(image_paths), dtype=torch.long)  # not used for these tests

    indices = list(range(len(image_paths)))
    np.random.shuffle(indices)
    n_val = int(len(indices) * args.val_split)
    val_indices = indices[:n_val]

    transforms = image_encoder.image_processor
    # training script uses get_transforms_from_processor(image_encoder.image_processor)
    from src.dataset import get_transforms_from_processor

    transforms = get_transforms_from_processor(image_encoder.image_processor)

    val_dataset = Stage2ImageDataset(
        image_paths=[image_paths[i] for i in val_indices],
        coordinates=coordinates[val_indices],
        cell_labels=cell_labels[val_indices],
        countries=[countries[i] for i in val_indices],
        transforms=transforms,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
    )

    # Stage2 model
    model = Stage2CrossAttentionGeoHead(
        patch_dim=int(ckpt.get("patch_dim", 1024)),
        concept_emb_dim=int(ckpt.get("concept_dim", 512)),
        num_cells=num_cells,
        coord_output_dim=coord_output_dim,
        num_heads=int(ckpt.get("num_heads", 8)),
        ablation_mode=ablation_mode,
    )
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)

    if args.max_batches is not None:
        # simple wrapper to limit batches
        from itertools import islice

        val_loader_iter = islice(iter(val_loader), int(args.max_batches))
        val_loader = list(val_loader_iter)

    baseline = evaluate_variant(
        model, image_encoder, stage1_model, val_loader, device, cell_centers, coord_output_dim, "baseline"
    )["median_error_km"]
    concept_shuf = evaluate_variant(
        model, image_encoder, stage1_model, val_loader, device, cell_centers, coord_output_dim, "concept_shuffled"
    )["median_error_km"]
    patches_shuf = evaluate_variant(
        model, image_encoder, stage1_model, val_loader, device, cell_centers, coord_output_dim, "patches_shuffled"
    )["median_error_km"]

    logger.info("=== Swap/Shuffle Results (median km) ===")
    logger.info(f"baseline:         {baseline:.4f}")
    logger.info(f"concept_shuffled: {concept_shuf:.4f}  (Δ {concept_shuf - baseline:+.4f})")
    logger.info(f"patches_shuffled: {patches_shuf:.4f}  (Δ {patches_shuf - baseline:+.4f})")


if __name__ == "__main__":
    main()


