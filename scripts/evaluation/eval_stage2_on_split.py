#!/usr/bin/env python3
"""
Evaluate Stage 2 checkpoint on internal test split from splits.json.

Computes:
- Median distance error (km)
- Threshold accuracies (street/city/region/country)
- Concept activation summaries (top-k concepts/parents)
- Saves results to JSON and CSV
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
from torch.utils.data import DataLoader
from tqdm import tqdm

from src.dataset import get_transforms_from_processor, load_splits_from_json
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from src.losses import haversine_distance
from scripts.training.train_stage2_cross_attention import (
    Stage2ImageDataset,
    stage2_collate_fn,
    load_stage1_checkpoint,
    compute_predicted_coords,
    compute_offset_targets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
}


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
def evaluate_stage2(
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
    """Evaluate Stage 2 model on test set."""
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
    
    for batch in tqdm(test_loader, desc="Evaluating"):
        images, coordinates, cell_labels, countries, image_paths = batch
        images = images.to(device)
        coordinates = coordinates.to(device)
        cell_labels = cell_labels.to(device)
        
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
        meta_probs = stage1_outputs["meta_probs"]  # [B, num_metas]
        parent_probs = stage1_outputs["parent_probs"]  # [B, num_parents]
        
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
    # Flatten top-5 lists
    flat_top5_meta = [idx for sublist in all_top5_meta_indices for idx in sublist]
    flat_top5_parent = [idx for sublist in all_top5_parent_indices for idx in sublist]
    
    meta_counter = Counter(flat_top5_meta)
    parent_counter = Counter(flat_top5_parent)
    pred_meta_counter = Counter(all_pred_meta_indices)
    pred_parent_counter = Counter(all_pred_parent_indices)
    
    # Top-10 most activated concepts
    top10_meta = meta_counter.most_common(10)
    top10_parent = parent_counter.most_common(10)
    top10_pred_meta = pred_meta_counter.most_common(10)
    top10_pred_parent = pred_parent_counter.most_common(10)
    
    concept_summary = {
        "top10_meta_in_top5": [
            {"concept": idx_to_concept.get(idx, f"Meta-{idx}"), "count": count, "fraction": count / len(flat_top5_meta)}
            for idx, count in top10_meta
        ],
        "top10_parent_in_top5": [
            {"concept": idx_to_parent.get(idx, f"Parent-{idx}"), "count": count, "fraction": count / len(flat_top5_parent)}
            for idx, count in top10_parent
        ],
        "top10_predicted_meta": [
            {"concept": idx_to_concept.get(idx, f"Meta-{idx}"), "count": count, "fraction": count / len(all_pred_meta_indices)}
            for idx, count in top10_pred_meta
        ],
        "top10_predicted_parent": [
            {"concept": idx_to_parent.get(idx, f"Parent-{idx}"), "count": count, "fraction": count / len(all_pred_parent_indices)}
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
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 checkpoint on test split")
    parser.add_argument("--stage2_checkpoint", type=str, required=True,
                        help="Path to Stage 2 checkpoint")
    parser.add_argument("--csv_path", type=str, required=True,
                        help="Path to CSV dataset")
    parser.add_argument("--splits_json", type=str, required=True,
                        help="Path to splits.json file")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--geoguessr_id", type=str, default="6906237dc7731161a37282b2")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--output_dir", type=str, default=None,
                        help="Output directory for results. Default: results/evals/<checkpoint_name>")
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
        output_dir = Path("results") / "evals" / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
        Path(args.stage2_checkpoint),
        device,
    )
    
    # Load splits
    logger.info(f"Loading splits from {args.splits_json}")
    with open(args.splits_json, 'r') as f:
        splits_data = json.load(f)
    
    test_pano_ids = set(splits_data["test_pano_ids"])
    logger.info(f"Test split: {len(test_pano_ids)} samples")
    
    # Load dataset from CSV
    logger.info(f"Loading dataset from {args.csv_path}")
    df = pd.read_csv(args.csv_path)
    
    # Build test samples
    image_paths = []
    coordinates = []
    countries = []
    pano_ids = []
    
    for _, row in tqdm(df.iterrows(), total=len(df), desc="Building test dataset"):
        pano_id = row.get("pano_id") or row.get("panoId")
        if pd.isna(pano_id) or str(pano_id) not in test_pano_ids:
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
        raise RuntimeError("No test samples found!")
    
    coordinates = torch.stack(coordinates)
    
    # Assign geocells (need to regenerate or load from checkpoint)
    # For now, assign to nearest cell center
    from scripts.training.train_stage2_cross_attention import generate_semantic_geocells
    _, sample_to_cell = generate_semantic_geocells(
        coordinates,
        countries,
        min_samples_per_cell=500,
        train_indices=None,  # Use all for assignment only
    )
    cell_labels = sample_to_cell
    
    # Create dataset and loader
    transforms = get_transforms_from_processor(image_encoder.image_processor)
    test_dataset = Stage2ImageDataset(
        image_paths=image_paths,
        coordinates=coordinates,
        cell_labels=cell_labels,
        countries=countries,
        transforms=transforms,
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=stage2_collate_fn,
        pin_memory=True,
    )
    
    logger.info(f"Test dataset: {len(test_dataset)} samples")
    
    # Evaluate
    metrics = evaluate_stage2(
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
        "splits_json": str(args.splits_json),
        "test_samples": len(test_dataset),
        "metrics": {k: (float(v) if isinstance(v, (np.ndarray, np.generic)) else v) 
                   for k, v in metrics.items() if k != "concept_summary"},
        "concept_summary": metrics.get("concept_summary", {}),
    }
    
    json_path = output_dir / "test_metrics.json"
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
    }
    df_results = pd.DataFrame(csv_data)
    csv_path = output_dir / "test_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Test Evaluation Results")
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

