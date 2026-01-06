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
from typing import Dict, List, Optional, Tuple
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
    load_image_encoder_weights_from_stage0_checkpoint,
    is_missing_or_none_path,
    compute_predicted_coords,
    compute_offset_targets,
    generate_semantic_geocells,
)
from scripts.evaluation.eval_stage2_on_split import find_latest_stage2_checkpoints

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Check GeoCLIP availability
try:
    from geoclip import GeoCLIP
    GEOCLIP_AVAILABLE = True
except ImportError:
    GEOCLIP_AVAILABLE = False
    logger.warning("GeoCLIP not available, GeoCLIP evaluation will be skipped")

def _is_vanilla_from_stage0(stage0_checkpoint) -> bool:
    if stage0_checkpoint is None:
        return True
    s = str(stage0_checkpoint).strip()
    return s == "" or s.lower() == "none"

THRESHOLD_ACCURACIES = {
    "street": 1.0,
    "city": 25.0,
    "region": 200.0,
    "country": 750.0,
}


def load_hf_dataset(split: str):
    """Load HF GeoGuessr dataset, preferring local cache over download."""
    logger.info("Loading HF dataset: fren-gor/geoguessr-locations")
    logger.info("This may take a few minutes...")

    # Check if cache directory exists
    cache_dir = Path("/scratch-shared/pnair/Project_AI/.cache/huggingface/hub/datasets--fren-gor--geoguessr-locations")
    if cache_dir.exists():
        logger.info(f"Using local cache: {cache_dir}")
        
        # Set cache directory to parent (hub directory)
        parent_cache_dir = cache_dir.parent
        
        # Try loading with cache first
        import os
        original_offline = os.environ.get("HF_HUB_OFFLINE")
        
        try:
            # First attempt: try with normal cache loading
            hf_dataset = load_dataset(
                "fren-gor/geoguessr-locations",
                split=split,
                cache_dir=str(parent_cache_dir),
                download_mode="reuse_cache_if_exists",
            )
            logger.info(f"Loaded {len(hf_dataset)} samples from split '{split}' from local cache")
            return hf_dataset
        except Exception as e:
            # If network error occurs, try with offline mode
            logger.warning(f"Network error during cache loading: {e}")
            logger.info("Retrying with offline mode...")
            os.environ["HF_HUB_OFFLINE"] = "1"
            
            hf_dataset = load_dataset(
                "fren-gor/geoguessr-locations",
                split=split,
                cache_dir=str(parent_cache_dir),
                download_mode="reuse_cache_if_exists",
            )
            logger.info(f"Loaded {len(hf_dataset)} samples from split '{split}' from local cache (offline mode)")
            return hf_dataset
        finally:
            # Restore original offline setting
            if original_offline is not None:
                os.environ["HF_HUB_OFFLINE"] = original_offline
            else:
                os.environ.pop("HF_HUB_OFFLINE", None)
    else:
        logger.info("Local cache not found, downloading dataset...")
        hf_dataset = load_dataset("fren-gor/geoguessr-locations", split=split)
        logger.info(f"Loaded {len(hf_dataset)} samples from split '{split}'")
        return hf_dataset


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
        
        # Get dataset length
        try:
            self.dataset_len = len(self.hf_dataset)
        except (TypeError, AttributeError):
            self.dataset_len = None
        
        if self.max_samples and self.dataset_len:
            self.dataset_len = min(self.max_samples, self.dataset_len)
    
    def __len__(self):
        if self.dataset_len is not None:
            return self.dataset_len
        if self.max_samples:
            return self.max_samples
        raise RuntimeError("Cannot determine dataset length")
    
    def __getitem__(self, idx):
        # Handle max_samples limit
        if self.max_samples and idx >= self.max_samples:
            raise IndexError(f"Index {idx} exceeds max_samples {self.max_samples}")
        
        sample = self.hf_dataset[idx]
        
        # Load panorama_360 image
        panorama_img = sample.get("panorama_360")
        if panorama_img is None:
            # Create a dummy black image if panorama_360 is missing
            from PIL import Image as PILImage
            pil_image = PILImage.new("RGB", (224, 224), color=(0, 0, 0))
        elif isinstance(panorama_img, dict) and "bytes" in panorama_img:
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
        lat = float(sample.get("lat", 0.0))
        lng = float(sample.get("lng", 0.0))
        coords = torch.tensor([lat, lng], dtype=torch.float32)
        
        # Country (may not be available)
        country = sample.get("country", "unknown")
        
        return img_tensor, coords, country, idx


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
def evaluate_geoclip_on_hf_dataset(
    geoclip_model,
    hf_dataset,
    max_samples: Optional[int] = None,
) -> Dict:
    """Evaluate GeoCLIP model on HF dataset with optimized I/O."""
    logger.info("Evaluating GeoCLIP on HF dataset...")

    import tempfile
    import os
    from pathlib import Path

    # Use faster temp directory (prefer /tmp or local scratch)
    temp_base = os.environ.get("TMPDIR", os.environ.get("SCRATCH", tempfile.gettempdir()))
    temp_dir = Path(temp_base) / f"geoclip_eval_{os.getpid()}"
    temp_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info(f"Using temporary directory: {temp_dir}")

    haversine_errors = []
    total_samples = 0
    temp_files = []

    # Pre-process: save all images to disk in batch (faster I/O)
    logger.info("Pre-saving images to temporary directory...")
    valid_samples = []
    for idx, sample in enumerate(tqdm(hf_dataset, desc="Pre-saving images")):
        if max_samples and idx >= max_samples:
            break

        panorama_img = sample.get("panorama_360")
        if panorama_img is None:
            continue

        # Convert to PIL Image
        if isinstance(panorama_img, dict) and "bytes" in panorama_img:
            img_bytes = panorama_img["bytes"]
            pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        else:
            pil_image = panorama_img.convert("RGB") if hasattr(panorama_img, "convert") else Image.open(panorama_img).convert("RGB")

        # Save to temp file (use JPEG for faster I/O)
        temp_path = temp_dir / f"img_{idx}.jpg"
        pil_image.save(temp_path, 'JPEG', quality=95, optimize=False)
        temp_files.append(temp_path)
        
        # Store ground truth coordinates
        true_lat = float(sample.get("lat", 0.0))
        true_lng = float(sample.get("lng", 0.0))
        valid_samples.append((temp_path, true_lat, true_lng))

    logger.info(f"Saved {len(valid_samples)} images. Starting GeoCLIP inference...")

    # Process all images
    for temp_path, true_lat, true_lng in tqdm(valid_samples, desc="Evaluating GeoCLIP"):
        # Get GeoCLIP prediction (top 1)
        top_pred_gps, top_pred_prob = geoclip_model.predict(str(temp_path), top_k=1)

        # Ground truth coordinates
        true_coords = torch.tensor([true_lat, true_lng], dtype=torch.float32)

        # Predicted coordinates
        pred_lat, pred_lng = top_pred_gps[0]
        pred_coords = torch.tensor([pred_lat, pred_lng], dtype=torch.float32)

        # Calculate distance error
        dist = haversine_distance(pred_coords.unsqueeze(0), true_coords.unsqueeze(0))
        haversine_errors.append(dist.item())

        total_samples += 1

    # Clean up temporary directory
    logger.info("Cleaning up temporary files...")
    import shutil
    shutil.rmtree(temp_dir, ignore_errors=True)

    if total_samples == 0:
        return {"error": "No samples processed"}

    median_error = np.median(haversine_errors)
    mean_error = np.mean(haversine_errors)

    # Threshold accuracies
    threshold_accs = {}
    for name, threshold in THRESHOLD_ACCURACIES.items():
        threshold_accs[f"acc_{name}"] = np.mean(np.array(haversine_errors) <= threshold)

    return {
        "loss": None,  # GeoCLIP doesn't have a loss
        "cell_acc": None,  # GeoCLIP doesn't have cell accuracy
        "median_error_km": median_error,
        "mean_error_km": mean_error,
        **threshold_accs,
        "total_samples": total_samples,
    }


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
    ablation_mode: str,
    patch_dim: int,
    concept_dim: int,
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
        images, coordinates, countries, sample_indices = batch
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
        
        # Compute features + patch tokens based on ablation mode
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


def evaluate_single_stage2_hf_checkpoint(
    checkpoint_path: Path,
    variant: str,
    ablation_mode: str,
    hf_dataset,  # Pre-loaded HF dataset
    batch_size: int,
    num_workers: int,
    device: torch.device,
    max_samples: Optional[int] = None,
    lambda_cell: float = 1.0,
    lambda_offset: float = 1.0,
    force_vanilla_encoder_for_patches: bool = False,
) -> Optional[Dict]:
    """Evaluate a single Stage 2 checkpoint on HF dataset and return results dict."""
    try:
        # Load checkpoint
        model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
            checkpoint_path, device, force_vanilla_encoder_for_patches=force_vanilla_encoder_for_patches
        )
        
        # Get ablation mode from checkpoint (may differ, use checkpoint's value)
        ablation_mode_from_ckpt = ckpt.get("ablation_mode", "both")
        if ablation_mode_from_ckpt != ablation_mode:
            logger.warning(f"Ablation mode mismatch: expected {ablation_mode}, got {ablation_mode_from_ckpt} from checkpoint")
            ablation_mode = ablation_mode_from_ckpt
        
        # Variant is determined from directory path, trust it
        
        # Create dataset (using pre-loaded hf_dataset)
        transforms = get_transforms_from_processor(image_encoder.image_processor)
        test_dataset = HFGeoGuessrDataset(
            hf_dataset,
            transforms=transforms,
            max_samples=max_samples,
        )
        
        logger.info(f"Dataset created: {len(test_dataset)} samples")
        
        test_loader = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=hf_collate_fn,
            pin_memory=True,
        )
        
        # Evaluate
        logger.info("Starting evaluation...")
        metrics = evaluate_on_hf_dataset(
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
            "dataset": "hf_geoguessr_locations",
        }
    except Exception as e:
        logger.error(f"Failed to evaluate {checkpoint_path}: {e}")
        import traceback
        traceback.print_exc()
        return None


def main():
    parser = argparse.ArgumentParser(description="Evaluate Stage 2 checkpoint on HF GeoGuessr dataset")
    parser.add_argument("--stage2_checkpoint", type=str, default=None,
                        help="Path to Stage 2 checkpoint (required if not using --batch_mode)")
    parser.add_argument("--batch_mode", action="store_true",
                        help="Auto-detect and evaluate all latest Stage 2 checkpoints")
    parser.add_argument("--results_root", type=str, default="results",
                        help="Root directory containing results (for batch mode)")
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
    parser.add_argument(
        "--force_vanilla_encoder_for_patches",
        action="store_true",
        default=False,
        help="If set, ignore Stage1/Stage0 lineage and use base StreetCLIP weights for patch extraction (for before/after comparison).",
    )
    parser.add_argument("--specific_checkpoints", nargs="+", default=None,
                        help="List of specific Stage 2 checkpoint paths to evaluate")
    parser.add_argument("--include_geoclip", action="store_true",
                        help="Include GeoCLIP evaluation alongside Stage 2 models")
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Specific checkpoints mode
    if args.specific_checkpoints or args.include_geoclip:
        logger.info("Evaluating specific checkpoints and/or GeoCLIP...")

        # Load HF dataset once
        hf_dataset = load_hf_dataset(args.split)

        all_results = []

        # Evaluate specific Stage 2 checkpoints
        if args.specific_checkpoints:
            for ckpt_path_str in args.specific_checkpoints:
                ckpt_path = Path(ckpt_path_str)
                if not ckpt_path.exists():
                    logger.warning(f"Checkpoint not found: {ckpt_path}")
                    continue

                # Extract variant and ablation mode from path
                path_parts = ckpt_path.parts
                variant = "unknown"
                ablation_mode = "unknown"

                # Try to extract from path structure
                for part in path_parts:
                    if "stage2_cross_attention" in part:
                        if "both" in part:
                            ablation_mode = "both"
                        elif "concept_only" in part:
                            ablation_mode = "concept_only"
                        elif "image_only" in part:
                            ablation_mode = "image_only"
                        if "vanilla_stage1" in part:
                            variant = "vanilla_stage1"
                        else:
                            variant = "trained_stage1"

                logger.info(f"\n{'='*60}")
                logger.info(f"Evaluating Stage 2: {ckpt_path}")
                logger.info(f"Variant: {variant}, Ablation Mode: {ablation_mode}")
                logger.info(f"{'='*60}")

                result = evaluate_single_stage2_hf_checkpoint(
                    checkpoint_path=ckpt_path,
                    variant=variant,
                    ablation_mode=ablation_mode,
                    hf_dataset=hf_dataset,
                    batch_size=args.batch_size,
                    num_workers=args.num_workers,
                    device=device,
                    max_samples=args.max_samples,
                    lambda_cell=args.lambda_cell,
                    lambda_offset=args.lambda_offset,
                    force_vanilla_encoder_for_patches=args.force_vanilla_encoder_for_patches,
                )

                if result:
                    all_results.append(result)
                    logger.info(f"  Median Error: {result['median_error_km']:.2f} km")
                    logger.info(f"  Mean Error: {result['mean_error_km']:.2f} km")
                    logger.info(f"  Country Accuracy: {result['acc_country']:.4f}")

        # Evaluate GeoCLIP
        if args.include_geoclip:
            if not GEOCLIP_AVAILABLE:
                logger.error("GeoCLIP requested but not available. Install with: pip install geoclip")
            else:
                logger.info(f"\n{'='*60}")
                logger.info("Evaluating GeoCLIP")
                logger.info(f"{'='*60}")

                geoclip_model = GeoCLIP()
                geoclip_metrics = evaluate_geoclip_on_hf_dataset(
                    geoclip_model,
                    hf_dataset,
                    max_samples=args.max_samples,
                )

                if "error" not in geoclip_metrics:
                    geoclip_result = {
                        "checkpoint": "GeoCLIP",
                        "checkpoint_name": "GeoCLIP",
                        "variant": "geoclip",
                        "ablation_mode": "geoclip",
                        "median_error_km": geoclip_metrics["median_error_km"],
                        "mean_error_km": geoclip_metrics["mean_error_km"],
                        "cell_acc": None,
                        "acc_street": geoclip_metrics["acc_street"],
                        "acc_city": geoclip_metrics["acc_city"],
                        "acc_region": geoclip_metrics["acc_region"],
                        "acc_country": geoclip_metrics["acc_country"],
                        "stage0_checkpoint": "N/A",
                        "stage1_checkpoint": "N/A",
                        "dataset": "hf_geoguessr_locations",
                    }
                    all_results.append(geoclip_result)
                    logger.info(f"  Median Error: {geoclip_result['median_error_km']:.2f} km")
                    logger.info(f"  Mean Error: {geoclip_result['mean_error_km']:.2f} km")
                    logger.info(f"  Country Accuracy: {geoclip_result['acc_country']:.4f}")
                else:
                    logger.warning(f"GeoCLIP evaluation failed: {geoclip_metrics['error']}")

        # Save consolidated CSV
        if len(all_results) > 0:
            output_dir = Path(args.results_root) / "evals"
            output_dir.mkdir(parents=True, exist_ok=True)
            consolidated_csv = output_dir / "stage2_hf_specific_checkpoints.csv"

            df = pd.DataFrame(all_results)
            df.to_csv(consolidated_csv, index=False)
            logger.info(f"\nSaved consolidated CSV to {consolidated_csv}")
            logger.info(f"Total evaluations: {len(df)}")
        else:
            logger.warning("No results to save!")

        return
    
    # Batch mode: auto-detect and evaluate all checkpoints
    if args.batch_mode:
        logger.info("Batch mode: Auto-detecting latest Stage 2 checkpoints...")
        results_root = Path(args.results_root)
        checkpoint_tuples = find_latest_stage2_checkpoints(results_root)
        
        if len(checkpoint_tuples) == 0:
            logger.error("No Stage 2 checkpoints found!")
            return
        
        logger.info(f"Found {len(checkpoint_tuples)} checkpoint(s) to evaluate")
        
        # Load HF dataset ONCE before the loop
        hf_dataset = load_hf_dataset(args.split)
        
        # Evaluate all checkpoints
        all_results = []
        for ckpt_path, variant, ablation_mode in checkpoint_tuples:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating: {ckpt_path}")
            logger.info(f"Variant: {variant}, Ablation Mode: {ablation_mode}")
            logger.info(f"Dataset: HF GeoGuessr Locations")
            logger.info(f"{'='*60}")
            
            result = evaluate_single_stage2_hf_checkpoint(
                checkpoint_path=ckpt_path,
                variant=variant,
                ablation_mode=ablation_mode,
                hf_dataset=hf_dataset,  # Pass the pre-loaded dataset
                batch_size=args.batch_size,
                num_workers=args.num_workers,
                device=device,
                max_samples=args.max_samples,
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
            consolidated_csv = output_dir / "stage2_hf_consolidated.csv"
            
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
        ckpt_name = Path(args.stage2_checkpoint).stem
        output_dir = Path("results") / "evals" / "hf_geoguessr" / ckpt_name
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load checkpoint
    model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
        Path(args.stage2_checkpoint),
        device,
        force_vanilla_encoder_for_patches=args.force_vanilla_encoder_for_patches,
    )
    
    ablation_mode = ckpt.get("ablation_mode", "both")
    
    # Load HF dataset
    hf_dataset = load_hf_dataset(args.split)
    
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
        ablation_mode=ablation_mode,
        patch_dim=ckpt["patch_dim"],
        concept_dim=ckpt["concept_dim"],
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

