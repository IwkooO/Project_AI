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
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple
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
    load_image_encoder_weights_from_stage0_checkpoint,
    is_missing_or_none_path,
    compute_predicted_coords,
    compute_offset_targets,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Datetime pattern for run directories
DATETIME_PATTERN = re.compile(r"\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}")

# Ablation modes
ABLATION_MODES = ["both", "concept_only", "image_only"]

def _is_vanilla_from_stage0(stage0_checkpoint) -> bool:
    if stage0_checkpoint is None:
        return True
    s = str(stage0_checkpoint).strip()
    return s == "" or s.lower() == "none"


def find_latest_datetime_dir(parent_dir: Path, exclude_subdirs: Optional[List[str]] = None) -> Optional[Path]:
    """Find the subdirectory with the latest datetime timestamp."""
    if exclude_subdirs is None:
        exclude_subdirs = []
    
    datetime_dirs = []
    for subdir in parent_dir.iterdir():
        if subdir.is_dir() and subdir.name not in exclude_subdirs:
            if DATETIME_PATTERN.match(subdir.name):
                datetime_dirs.append(subdir)
    
    if not datetime_dirs:
        return None
    
    # Sort by name (datetime format sorts chronologically)
    datetime_dirs.sort(key=lambda x: x.name, reverse=True)
    return datetime_dirs[0]


def find_latest_stage2_checkpoints(results_root: Path) -> List[Tuple[Path, str, str]]:
    """
    Find the LATEST Stage 2 checkpoint for each ablation mode and variant.
    
    Returns list of (checkpoint_path, variant, ablation_mode) tuples.
    """
    checkpoints = []
    results_root = Path(results_root)
    
    for mode in ABLATION_MODES:
        mode_dir = results_root / f"stage2_cross_attention_{mode}"
        if not mode_dir.exists():
            logger.warning(f"Mode directory not found: {mode_dir}")
            continue
        
        # Find LATEST finetuned checkpoint (direct datetime subdirectories, excluding vanilla_stage1)
        finetuned_dirs = [
            subdir for subdir in mode_dir.iterdir()
            if subdir.is_dir() and subdir.name != "vanilla_stage1" and DATETIME_PATTERN.match(subdir.name)
        ]
        if finetuned_dirs:
            finetuned_dirs.sort(key=lambda x: x.name, reverse=True)
            latest_finetuned = finetuned_dirs[0]
            ckpt_path = latest_finetuned / "checkpoints" / "best_model_stage2_xattn.pt"
            if ckpt_path.exists():
                checkpoints.append((ckpt_path, "finetuned", mode))
                logger.info(f"Found latest Stage 2 checkpoint: {ckpt_path} (variant=finetuned, mode={mode})")
        
        # Find LATEST vanilla checkpoint (inside vanilla_stage1 subdirectory)
        vanilla_dir = mode_dir / "vanilla_stage1"
        if vanilla_dir.exists():
            latest_vanilla_subdir = find_latest_datetime_dir(vanilla_dir)
            if latest_vanilla_subdir:
                ckpt_path = latest_vanilla_subdir / "checkpoints" / "best_model_stage2_xattn.pt"
                if ckpt_path.exists():
                    checkpoints.append((ckpt_path, "vanilla", mode))
                    logger.info(f"Found latest Stage 2 checkpoint: {ckpt_path} (variant=vanilla, mode={mode})")
    
    return checkpoints


def _default_output_dir_for_stage2_checkpoint(checkpoint_path: Path) -> Path:
    """
    Create a unique output directory for a checkpoint based on the results folder structure:
      Finetuned: results/stage2_cross_attention_<mode>/<datetime>/checkpoints/best_model_stage2_xattn.pt
        -> results/evals/stage2__stage2_cross_attention_<mode>__<datetime>
      Vanilla:   results/stage2_cross_attention_<mode>/vanilla_stage1/<datetime>/checkpoints/best_model_stage2_xattn.pt
        -> results/evals/stage2__stage2_cross_attention_<mode>__vanilla_stage1__<datetime>
    """
    ckpt = checkpoint_path.resolve()
    parts = ckpt.parts
    if "results" in parts:
        rel = Path(*parts[parts.index("results") + 1 :])
    else:
        rel = ckpt

    # Expect: <root>/<datetime>/checkpoints/<file> OR <root>/vanilla_stage1/<datetime>/checkpoints/<file>
    if len(rel.parts) >= 4 and rel.parts[-2] == "checkpoints":
        run_dt = rel.parts[-3]
        maybe_vanilla = rel.parts[-4]
        stage2_root = rel.parts[-5] if maybe_vanilla == "vanilla_stage1" and len(rel.parts) >= 5 else rel.parts[-4]
        if maybe_vanilla == "vanilla_stage1":
            safe = f"stage2__{stage2_root}__vanilla_stage1__{run_dt}"
        else:
            safe = f"stage2__{stage2_root}__{run_dt}"
    else:
        safe = f"stage2__{checkpoint_path.parent.parent.parent.name}__{checkpoint_path.parent.parent.name}"

    safe = "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in safe)
    return Path("results") / "evals" / safe

THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
}


def load_stage2_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
    force_vanilla_encoder_for_patches: bool = False,
) -> tuple:
    """Load Stage 2 checkpoint."""
    logger.info(f"Loading Stage 2 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load Stage 1 checkpoint
    stage1_ckpt_path = Path(ckpt["stage1_checkpoint"])
    stage1_ckpt_data = torch.load(stage1_ckpt_path, map_location="cpu", weights_only=False)
    encoder_config = StreetCLIPConfig(
        model_name=ckpt["encoder_model"],
        finetune=False,
        device=device,
    )
    image_encoder = StreetCLIPEncoder(encoder_config)

    if force_vanilla_encoder_for_patches:
        logger.info("force_vanilla_encoder_for_patches=True: using base encoder weights for patch extraction")
    else:
        embedded_enc = stage1_ckpt_data.get("image_encoder_state_dict")
        if embedded_enc is not None:
            logger.info("Using image_encoder_state_dict embedded in Stage1 checkpoint for patch extraction")
            image_encoder.load_state_dict(embedded_enc, strict=False)
        else:
            stage0_checkpoint = stage1_ckpt_data.get("stage0_checkpoint")
            if not is_missing_or_none_path(stage0_checkpoint):
                load_image_encoder_weights_from_stage0_checkpoint(Path(stage0_checkpoint), image_encoder)
            else:
                logger.info("Stage1 indicates vanilla lineage (no Stage0 checkpoint); using base encoder weights")

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
    ablation_mode: str,
    patch_dim: int,
    concept_dim: int,
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
        
        # Compute features + patch tokens in a way that matches training ablation logic
        if ablation_mode == "concept_only":
            # No patch tokens used in this mode
            img_features = image_encoder(images)  # [B, 768]
            concept_embs = stage1_model.concept_bottleneck(img_features.float())
            patch_tokens = torch.empty((images.size(0), 0, patch_dim), device=device, dtype=img_features.dtype)
        else:
            # both / image_only: get features + patches in ONE forward
            img_features, patch_tokens = image_encoder.get_features_and_patches(images)
            if ablation_mode == "image_only":
                concept_embs = torch.zeros((images.size(0), concept_dim), device=device, dtype=img_features.dtype)
            else:
                concept_embs = stage1_model.concept_bottleneck(img_features.float())
        
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


def evaluate_single_stage2_checkpoint(
    checkpoint_path: Path,
    variant: str,
    ablation_mode: str,
    csv_path: str,
    splits_json: str,
    data_root: str,
    geoguessr_id: str,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    lambda_cell: float = 1.0,
    lambda_offset: float = 1.0,
    force_vanilla_encoder_for_patches: bool = False,
) -> Optional[Dict]:
    """Evaluate a single Stage 2 checkpoint and return results dict."""
    try:
        # Load checkpoint
        model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
            checkpoint_path, device, force_vanilla_encoder_for_patches=force_vanilla_encoder_for_patches
        )
        
        # Variant is determined from directory path, trust it
        
        # Load splits
        with open(splits_json, 'r') as f:
            splits_data = json.load(f)
        
        test_pano_ids = set(splits_data["test_pano_ids"])
        
        # Load dataset from CSV
        df = pd.read_csv(csv_path)
        
        # Build test samples
        image_paths = []
        coordinates = []
        countries = []
        
        for _, row in df.iterrows():
            pano_id = row.get("pano_id") or row.get("panoId")
            if pd.isna(pano_id) or str(pano_id) not in test_pano_ids:
                continue
            
            if "image_path" in row and pd.notna(row["image_path"]):
                img_path = Path(row["image_path"])
            else:
                img_path = Path(data_root) / geoguessr_id / "export" / f"{pano_id}.jpg"
            
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
            logger.warning(f"No test samples found for {checkpoint_path}")
            return None
        
        coordinates = torch.stack(coordinates)
        
        # Assign geocells to nearest cell center
        coords_np = coordinates.cpu().numpy()
        centers_np = cell_centers.cpu().numpy()
        
        lat_rad = np.deg2rad(coords_np[:, 0])
        lng_rad = np.deg2rad(coords_np[:, 1])
        x = np.cos(lat_rad) * np.cos(lng_rad)
        y = np.cos(lat_rad) * np.sin(lng_rad)
        z = np.sin(lat_rad)
        xyz = np.stack([x, y, z], axis=1)
        
        distances = np.linalg.norm(xyz[:, None, :] - centers_np[None, :, :], axis=2)
        cell_labels = torch.tensor(np.argmin(distances, axis=1), dtype=torch.long)
        
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
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=stage2_collate_fn,
            pin_memory=True,
        )
        
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
            ablation_mode=ablation_mode,
            patch_dim=ckpt["patch_dim"],
            concept_dim=ckpt["concept_dim"],
            lambda_cell=lambda_cell,
            lambda_offset=lambda_offset,
        )
        
        if "error" in metrics:
            logger.warning(f"Evaluation error for {checkpoint_path}: {metrics['error']}")
            return None
        
        return {
            "checkpoint": str(checkpoint_path),
            "checkpoint_name": checkpoint_path.name,
            "variant": variant,
            "ablation_mode": ablation_mode,
            "median_error_km": metrics["median_error_km"],
            "mean_error_km": metrics["mean_error_km"],
            "cell_acc": metrics["cell_acc"],
            "acc_street": metrics["acc_street"],
            "acc_city": metrics["acc_city"],
            "acc_region": metrics["acc_region"],
            "acc_country": metrics["acc_country"],
            "stage0_checkpoint": str(ckpt.get("stage0_checkpoint", "None")),
            "stage1_checkpoint": str(ckpt.get("stage1_checkpoint", "None")),
        }
    except Exception as e:
        logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 checkpoint on test split")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 checkpoint (required if not using --batch_mode)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Auto-detect and evaluate all latest Stage 2 checkpoints")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root directory containing results (for batch mode)")
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
    parser.add_argument(
        "--force_vanilla_encoder_for_patches",
        action="store_true",
        default=False,
        help="If set, ignore Stage1/Stage0 lineage and use base StreetCLIP weights for patch extraction (for before/after comparison).",
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Batch mode: auto-detect and evaluate all checkpoints
    if args.batch_mode:
        logger.info("Batch mode: Auto-detecting latest Stage 2 checkpoints...")
        results_root = Path(args.results_root)
        checkpoint_tuples = find_latest_stage2_checkpoints(results_root)
        
        if len(checkpoint_tuples) == 0:
            logger.error("No Stage 2 checkpoints found!")
            return
        
        logger.info(f"Found {len(checkpoint_tuples)} checkpoint(s) to evaluate")
        
        # Evaluate all checkpoints
        all_results = []
        for ckpt_path, variant, ablation_mode in checkpoint_tuples:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {ckpt_path}")
            logger.info(f"Variant: {variant}, Ablation Mode: {ablation_mode}")
            logger.info(f"{'='*60}")
            
            result = evaluate_single_stage2_checkpoint(
                checkpoint_path=ckpt_path,
                variant=variant,
                ablation_mode=ablation_mode,
                csv_path=args.csv_path,
                splits_json=args.splits_json,
                data_root=args.data_root,
                geoguessr_id=args.geoguessr_id,
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                lambda_cell=args.lambda_cell,
                lambda_offset=args.lambda_offset,
                force_vanilla_encoder_for_patches=args.force_vanilla_encoder_for_patches,
            )
            
            if result:
                all_results.append(result)
                logger.info(f"  Median Error: {result['median_error_km']:.2f} km")
                logger.info(f"  Cell Accuracy: {result['cell_acc']:.4f}")
                logger.info(f"  Country Accuracy: {result['acc_country']:.4f}")
        
        # Save consolidated CSV
        if len(all_results) > 0:
            output_dir = Path(args.results_root) / "evals"
            output_dir.mkdir(parents=True, exist_ok=True)
            consolidated_csv = output_dir / "stage2_test_consolidated.csv"
            
            df = pd.DataFrame(all_results)
            df.to_csv(consolidated_csv, index=False)
            logger.info(f"\nSaved consolidated CSV to {consolidated_csv}")
            logger.info(f"Total evaluations: {len(df)}")
        else:
            logger.warning("No results to save!")
        
        return
    
    # Single checkpoint mode (backward compatible)
    if args.stage2_checkpoint is None:
        parser.error("--stage2_checkpoint is required when not using --batch_mode")
    
    # Setup output directory
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        output_dir = _default_output_dir_for_stage2_checkpoint(Path(args.stage2_checkpoint))
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
        Path(args.stage2_checkpoint),
        device,
        force_vanilla_encoder_for_patches=args.force_vanilla_encoder_for_patches,
    )
    ablation_mode = ckpt.get("ablation_mode", "both")
    variant = "vanilla" if _is_vanilla_from_stage0(ckpt.get("stage0_checkpoint")) else "finetuned"
    
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
    
    # Assign geocells to nearest cell center (geocells already generated during training)
    # We just need to assign test samples to the existing cell centers
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
    cell_labels = torch.tensor(np.argmin(distances, axis=1), dtype=torch.long)
    
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
        ablation_mode=ablation_mode,
        patch_dim=ckpt["patch_dim"],
        concept_dim=ckpt["concept_dim"],
        lambda_cell=args.lambda_cell,
        lambda_offset=args.lambda_offset,
    )
    
    # Save results
    results_json = {
        "stage2_checkpoint": str(args.stage2_checkpoint),
        "splits_json": str(args.splits_json),
        "variant": variant,
        "ablation_mode": ablation_mode,
        "stage0_checkpoint": ckpt.get("stage0_checkpoint"),
        "stage1_checkpoint": ckpt.get("stage1_checkpoint"),
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
        "variant": [variant],
        "ablation_mode": [ablation_mode],
        "median_error_km": [metrics["median_error_km"]],
        "mean_error_km": [metrics["mean_error_km"]],
        "cell_acc": [metrics["cell_acc"]],
        "acc_street": [metrics["acc_street"]],
        "acc_city": [metrics["acc_city"]],
        "acc_region": [metrics["acc_region"]],
        "acc_country": [metrics["acc_country"]],
        "stage0_checkpoint": [ckpt.get("stage0_checkpoint", "None")],
    }
    df_results = pd.DataFrame(csv_data)
    csv_path = output_dir / "test_metrics.csv"
    df_results.to_csv(csv_path, index=False)
    logger.info(f"Saved CSV summary to {csv_path}")
    
    # Print summary
    logger.info("\n" + "="*60)
    logger.info("Test Evaluation Results")
    logger.info("="*60)
    logger.info(f"Checkpoint: {args.stage2_checkpoint}")
    logger.info(f"Variant: {variant}")
    logger.info(f"Ablation mode: {ablation_mode}")
    logger.info(f"Stage0 checkpoint: {ckpt.get('stage0_checkpoint', 'None')}")
    logger.info(f"Stage1 checkpoint: {ckpt.get('stage1_checkpoint', 'unknown')}")
    logger.info(f"Splits: {args.splits_json}")
    logger.info(f"Output dir: {output_dir}")
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

