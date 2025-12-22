#!/usr/bin/env python3
"""
Evaluate Stage 2 checkpoint on external HF GeoGuessr dataset.

Uses the `panorama_360` image column from fren-gor/geoguessr-locations.
Computes:
- Median distance error (km)
- Threshold accuracies (street/city/region/country)
- Concept activation summaries
"""

import argparse
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional
from collections import Counter

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from PIL import Image
import io

from datasets import load_dataset

from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from src.losses import haversine_distance
from src.dataset import get_transforms_from_processor
from scripts.training.train_stage2_cross_attention import (
    load_stage1_checkpoint,
    compute_predicted_coords,
    compute_offset_targets,
    generate_semantic_geocells,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
}


class HFGeoGuessrDataset(Dataset):
    """Dataset wrapper for HF GeoGuessr dataset using panorama_360 images."""
    
    def __init__(
        self,
        hf_dataset,
        transforms=None,
        max_samples: Optional[int] = None,
    ):
        self.hf_dataset = hf_dataset
        self.transforms = transforms
        self.max_samples = max_samples
        
        # Filter samples with valid coordinates and panorama_360 images
        self.valid_indices = []
        for i, sample in enumerate(self.hf_dataset):
            if "panorama_360" not in sample or sample["panorama_360"] is None:
                continue
            if "lat" not in sample or "lng" not in sample:
                continue
            if sample["lat"] is None or sample["lng"] is None:
                continue
            self.valid_indices.append(i)
            if max_samples and len(self.valid_indices) >= max_samples:
                break
        
        logger.info(f"Found {len(self.valid_indices)} valid samples")
    
    def __len__(self):
        return len(self.valid_indices)
    
    def __getitem__(self, idx):
        sample_idx = self.valid_indices[idx]
        sample = self.hf_dataset[sample_idx]
        
        # Load panorama_360 image
        panorama_img = sample["panorama_360"]
        if isinstance(panorama_img, dict) and "bytes" in panorama_img:
            img_bytes = panorama_img["bytes"]
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            pil_image = panorama_img.convert("RGB") if hasattr(panorama_img, "convert") else Image.open(panorama_img).convert("RGB")
        
        # Apply transforms
        if self.transforms:
            img_tensor = self.transforms(pil_image)
        else:
            from torchvision import transforms
            img_tensor = transforms.ToTensor()(pil_image)
        
        # Get coordinates
        lat = float(sample["lat"])
        lng = float(sample["lng"])
        coords = torch.tensor([lat, lng], dtype=torch.float32)
        
        # Country (may not be available)
        country = sample.get("country", "unknown")
        
        return img_tensor, coords, country, sample_idx


def hf_collate_fn(batch):
    """Collate function for HF dataset."""
    images = torch.stack([item[0] for item in batch])
    coords = torch.stack([item[1] for item in batch])
    countries = [item[2] for item in batch]
    sample_indices = [item[3] for item in batch]
    return images, coords, countries, sample_indices


def load_stage2_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> tuple:
    """Load Stage 2 checkpoint."""
    logger.info(f"Loading Stage 2 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device)
    
    # Load Stage 1 checkpoint
    stage1_ckpt_path = Path(ckpt["stage1_checkpoint"])
    encoder_config = StreetCLIPConfig(
        model_name=ckpt["encoder_model"],
        finetune=False,
        device=device,
    )
    image_encoder = StreetCLIPEncoder(encoder_config)
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
def evaluate_on_hf_dataset(
    model: Stage2CrossAttentionGeoHead,
    image_encoder: StreetCLIPEncoder,
    stage1_model: Stage1ConceptModel,
    test_loader: DataLoader,
    device: torch.device,
    cell_centers: torch.Tensor,
    coord_output_dim: int,
    concept_info: Dict,
    lambda_cell: float = 1.0,
    lambda_offset: float = 1.0,
) -> Dict:
    """Evaluate Stage 2 model on HF dataset."""
    model.eval()
    stage1_model.eval()
    
    criterion_cell = nn.CrossEntropyLoss()
    
    total_loss = 0.0
    total_cell_acc = 0.0
    haversine_errors = []
    n_batches = 0
    
    # Concept activation tracking
    all_top5_meta_indices = []
    all_top5_parent_indices = []
    all_pred_meta_indices = []
    all_pred_parent_indices = []
    
    idx_to_concept = concept_info.get("idx_to_concept", {})
    idx_to_parent = concept_info.get("idx_to_parent", {})
    
    # Assign geocells on-the-fly (we'll use a dummy assignment since we don't have train data)
    # For now, assign to nearest cell center
    all_coords_list = []
    all_countries_list = []
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        images, coordinates, countries, _ = batch
        images = images.to(device)
        coordinates = coordinates.to(device)
        
        all_coords_list.append(coordinates.cpu())
        all_countries_list.extend(countries)
        
        # Assign dummy cell labels (nearest center)
        coords_np = coordinates.cpu().numpy()
        centers_np = cell_centers.cpu().numpy()
        
        # Convert to 3D for distance computation
        lat_rad = np.deg2rad(coords_np[:, 0])
        lng_rad = np.deg2rad(coords_np[:, 1])
        x = np.cos(lat_rad) * np.cos(lng_rad)
        y = np.cos(lat_rad) * np.sin(lng_rad)
        z = np.sin(lat_rad)
        xyz = np.stack([x, y, z], axis=1)
        
        # Find nearest cell center
        distances = np.linalg.norm(xyz[:, None, :] - centers_np[None, :, :], axis=2)
        cell_labels = torch.tensor(np.argmin(distances, axis=1), dtype=torch.long).to(device)
        
        # Get patch tokens and image features
        patch_tokens = image_encoder.get_patch_tokens(images)
        img_features = image_encoder(images)
        
        # Get concept embeddings from Stage 1
        concept_embs = stage1_model.concept_bottleneck(img_features)
        
        # Forward pass
        outputs = model(concept_embs, patch_tokens, return_attention=False, return_gate=False)
        cell_logits = outputs["cell_logits"]
        pred_offsets = outputs["pred_offsets"]
        
        # Get concept predictions from Stage 1
        stage1_outputs = stage1_model.forward_from_features(img_features)
        meta_probs = stage1_outputs["meta_probs"]
        parent_probs = stage1_outputs["parent_probs"]
        
        # Track top-5 concepts
        _, top5_meta = torch.topk(meta_probs, k=min(5, meta_probs.shape[1]), dim=1)
        _, top5_parent = torch.topk(parent_probs, k=min(5, parent_probs.shape[1]), dim=1)
        all_top5_meta_indices.extend(top5_meta.cpu().numpy().tolist())
        all_top5_parent_indices.extend(top5_parent.cpu().numpy().tolist())
        
        # Track predicted concepts
        pred_meta = meta_probs.argmax(dim=1)
        pred_parent = parent_probs.argmax(dim=1)
        all_pred_meta_indices.extend(pred_meta.cpu().numpy().tolist())
        all_pred_parent_indices.extend(pred_parent.cpu().numpy().tolist())
        
        # Loss computation
        loss_cell = criterion_cell(cell_logits, cell_labels)
        target_offsets = compute_offset_targets(coordinates, cell_labels, cell_centers, coord_output_dim)
        loss_offset = F.mse_loss(pred_offsets, target_offsets)
        loss = lambda_cell * loss_cell + lambda_offset * loss_offset
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
        return {"error": "No batches processed"}
    
    avg_loss = total_loss / n_batches
    avg_cell_acc = total_cell_acc / n_batches
    median_error = np.median(haversine_errors) if haversine_errors else float("inf")
    mean_error = np.mean(haversine_errors) if haversine_errors else float("inf")
    
    # Threshold accuracies
    threshold_accs = {}
    for name, threshold in THRESHOLD_ACCURACIES.items():
        threshold_accs[f"acc_{name}"] = np.mean(np.array(haversine_errors) <= threshold)
    
    # Concept activation summaries
    flat_top5_meta = [idx for sublist in all_top5_meta_indices for idx in sublist]
    flat_top5_parent = [idx for sublist in all_top5_parent_indices for idx in sublist]
    
    meta_counter = Counter(flat_top5_meta)
    parent_counter = Counter(flat_top5_parent)
    pred_meta_counter = Counter(all_pred_meta_indices)
    pred_parent_counter = Counter(all_pred_parent_indices)
    
    top10_meta = meta_counter.most_common(10)
    top10_parent = parent_counter.most_common(10)
    top10_pred_meta = pred_meta_counter.most_common(10)
    top10_pred_parent = pred_parent_counter.most_common(10)
    
    concept_summary = {
        "top10_meta_in_top5": [
            {"concept": idx_to_concept.get(idx, f"Meta-{idx}"), "count": count, "fraction": count / len(flat_top5_meta) if flat_top5_meta else 0}
            for idx, count in top10_meta
        ],
        "top10_parent_in_top5": [
            {"concept": idx_to_parent.get(idx, f"Parent-{idx}"), "count": count, "fraction": count / len(flat_top5_parent) if flat_top5_parent else 0}
            for idx, count in top10_parent
        ],
        "top10_predicted_meta": [
            {"concept": idx_to_concept.get(idx, f"Meta-{idx}"), "count": count, "fraction": count / len(all_pred_meta_indices) if all_pred_meta_indices else 0}
            for idx, count in top10_pred_meta
        ],
        "top10_predicted_parent": [
            {"concept": idx_to_parent.get(idx, f"Parent-{idx}"), "count": count, "fraction": count / len(all_pred_parent_indices) if all_pred_parent_indices else 0}
            for idx, count in top10_pred_parent
        ],
    }
    
    return {
        "loss": avg_loss,
        "cell_acc": avg_cell_acc,
        "median_error_km": median_error,
        "mean_error_km": mean_error,
        **threshold_accs,
        "concept_summary": concept_summary,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 checkpoint on HF GeoGuessr dataset")
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                        help="Path to Stage 2 checkpoint")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results")
    parser.add_argument("--max_samples", type=int, default=None,
                        help="Maximum number of samples to evaluate (for quick testing)")
    parser.add_argument("--split", type=str, default="train",
                        help="HF dataset split to use")
    parser.add_argument("--lambda_cell", type=float, default=1.0)
    parser.add_argument("--lambda_offset", type=float, default=1.0)
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        ckpt_name = Path(args.stage2_checkpoint).stem
        output_dir = Path("results") / "evals" / "hf_geoguessr" / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
        Path(args.stage2_checkpoint),
        device,
    )
    
    # Load HF dataset
    logger.info("Loading HF dataset: fren-gor/geoguessr-locations")
    hf_dataset = load_dataset("fren-gor/geoguessr-locations", split=args.split)
    logger.info(f"Loaded {len(hf_dataset)} samples from split '{args.split}'")
    
    # Create dataset
    transforms = get_transforms_from_processor(image_encoder.image_processor)
    test_dataset = HFGeoGuessrDataset(
        hf_dataset,
        transforms=transforms,
        max_samples=args.max_samples,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=hf_collate_fn,
        pin_memory=True,
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    # Evaluate
    metrics = evaluate_on_hf_dataset(
        model,
        image_encoder,
        stage1_model,
        test_loader,
        device,
        cell_centers,
        ckpt["coord_output_dim"],
        concept_info,
        lambda_cell=args.lambda_cell,
        lambda_offset=args.lambda_offset,
    )
    
    # Save results
    results_json = {
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "dataset": "fren-gor/geoguessr-locations",
        "split": args.split,
        "test_samples": len(test_dataset),
        "metrics": {k: (float(v) if isinstance(v, (np.ndarray, np.generic)) else v) 
                   for k, v in metrics.items() if k != "concept_summary"},
        "concept_summary": metrics.get("concept_summary", {}),
    }
    
    json_path = output_dir / "hf_test_metrics.json"
    with open(json_path, 'w') as f:
        json.dump(results_json, f, indent=2)
    logger.info(f"Saved metrics to {json_path}")
    
    # Save CSV summary
    csv_data = {
        "checkpoint": [Path(args.stage2_checkpoint).name],
        "median_error_km": [metrics["median_error_km"]],
        "mean_error_km": [metrics["mean_error_km"]],
        "cell_acc": [metrics["cell_acc"]],
        "acc_street": [metrics["acc_street"]],
        "acc_city": [metrics["acc_city"]],
        "acc_region": [metrics["acc_region"]],
        "acc_country": [metrics["acc_country"]],
        "stage0_checkpoint": [ckpt.get("stage0_checkpoint", "None")],
        "ablation_mode": [ckpt.get("ablation_mode", "unknown")],
        "dataset": ["hf_geoguessr_locations"],
    }
    df_results = pd.DataFrame(csv_data)
    csv_path = output_dir / "hf_test_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("HF Dataset Evaluation Results")
    logger.info("="*60)
    logger.info(f"Median Error: {metrics['median_error_km']:.2f} km")
    logger.info(f"Mean Error: {metrics['mean_error_km']:.2f} km")
    logger.info(f"Cell Accuracy: {metrics['cell_acc']:.4f}")
    logger.info(f"Street Accuracy (<1km): {metrics['acc_street']:.4f}")
    logger.info(f"City Accuracy (<25km): {metrics['acc_city']:.4f}")
    logger.info(f"Region Accuracy (<200km): {metrics['acc_region']:.4f}")
    logger.info(f"Country Accuracy (<750km): {metrics['acc_country']:.4f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

