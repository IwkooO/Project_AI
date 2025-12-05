#!/usr/bin/env python3
"""
Inference script for GeoGuessr location prediction.

Given a trained ConceptAwareGeoModel checkpoint and an image,
predicts the geographic location (latitude, longitude).

Usage:
    python scripts/inference/predict_location.py \
        --checkpoint results/concept-aware/.../checkpoints/best_model_stage2.pt \
        --image path/to/image.jpg
        
    # Batch inference on multiple images
    python scripts/inference/predict_location.py \
        --checkpoint results/concept-aware/.../checkpoints/best_model_stage2.pt \
        --image_dir path/to/images/ \
        --output predictions.csv
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import torch
from PIL import Image
from tqdm import tqdm

from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.dataset import get_transforms_from_processor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_model_for_inference(
    checkpoint_path: str,
    device: torch.device = None,
) -> Tuple[ConceptAwareGeoModel, torch.Tensor, Dict]:
    """
    Load a trained model checkpoint for inference.
    
    Args:
        checkpoint_path: Path to the checkpoint file (.pt)
        device: Device to load the model to
        
    Returns:
        Tuple of (model, cell_centers, metadata_dict)
    """
    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    checkpoint_path = Path(checkpoint_path)
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
    
    logger.info(f"Loading checkpoint from {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Extract metadata
    encoder_model = checkpoint.get("encoder_model", "geolocal/StreetCLIP")
    num_concepts = checkpoint["num_concepts"]
    num_countries = checkpoint["num_countries"]
    num_cells = checkpoint["num_cells"]
    coord_output_dim = checkpoint["coord_output_dim"]
    cell_centers = checkpoint["cell_centers"].to(device)
    concept_names = checkpoint.get("concept_names", [])
    idx_to_country = checkpoint.get("idx_to_country", {})
    
    logger.info(f"Model config: {num_concepts} concepts, {num_countries} countries, {num_cells} cells")
    logger.info(f"Encoder: {encoder_model}, coord_output_dim: {coord_output_dim}")
    
    # Initialize encoder
    encoder_config = StreetCLIPConfig(
        model_name=encoder_model,
        finetune=False,
        device=device,
    )
    image_encoder = StreetCLIPEncoder(encoder_config)
    
    # Initialize model
    model = ConceptAwareGeoModel(
        image_encoder=image_encoder,
        num_concepts=num_concepts,
        num_countries=num_countries,
        num_cells=num_cells,
        streetclip_dim=image_encoder.feature_dim,
        concept_emb_dim=512,
        coord_output_dim=coord_output_dim,
    )
    
    # Load weights
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()
    
    logger.info("Model loaded successfully")
    
    metadata = {
        "encoder_model": encoder_model,
        "concept_names": concept_names,
        "idx_to_country": idx_to_country,
        "num_concepts": num_concepts,
        "num_countries": num_countries,
        "num_cells": num_cells,
        "coord_output_dim": coord_output_dim,
    }
    
    return model, cell_centers, metadata


def load_and_preprocess_image(
    image_path: str,
    transform,
    device: torch.device,
) -> torch.Tensor:
    """
    Load and preprocess a single image for inference.
    
    Args:
        image_path: Path to the image file
        transform: Preprocessing transforms
        device: Device to load the tensor to
        
    Returns:
        Preprocessed image tensor [1, 3, H, W]
    """
    image = Image.open(image_path).convert("RGB")
    image_tensor = transform(image)
    return image_tensor.unsqueeze(0).to(device)


@torch.no_grad()
def predict_single(
    model: ConceptAwareGeoModel,
    image_tensor: torch.Tensor,
    cell_centers: torch.Tensor,
    metadata: Dict,
) -> Dict:
    """
    Predict location for a single image.
    
    Args:
        model: Trained ConceptAwareGeoModel
        image_tensor: Preprocessed image tensor [1, 3, H, W]
        cell_centers: Geocell centers tensor
        metadata: Model metadata dict
        
    Returns:
        Dict with prediction results
    """
    result = model.predict_location(image_tensor, cell_centers)
    
    pred_lat = result["pred_lat"].item()
    pred_lng = result["pred_lng"].item()
    pred_cell = result["pred_cell"].item()
    
    # Get top predicted country
    country_probs = result["country_probs"][0]
    top_country_idx = country_probs.argmax().item()
    top_country_prob = country_probs[top_country_idx].item()
    idx_to_country = metadata.get("idx_to_country", {})
    top_country = idx_to_country.get(top_country_idx, f"Country_{top_country_idx}")
    
    # Get top predicted concept
    concept_probs = result["concept_probs"][0]
    top_concept_idx = concept_probs.argmax().item()
    top_concept_prob = concept_probs[top_concept_idx].item()
    concept_names = metadata.get("concept_names", [])
    top_concept = concept_names[top_concept_idx] if top_concept_idx < len(concept_names) else f"Concept_{top_concept_idx}"
    
    return {
        "pred_lat": pred_lat,
        "pred_lng": pred_lng,
        "pred_cell": pred_cell,
        "top_country": top_country,
        "top_country_prob": top_country_prob,
        "top_concept": top_concept,
        "top_concept_prob": top_concept_prob,
    }


def predict_batch(
    model: ConceptAwareGeoModel,
    image_paths: List[str],
    cell_centers: torch.Tensor,
    metadata: Dict,
    transform,
    device: torch.device,
    batch_size: int = 16,
) -> List[Dict]:
    """
    Predict locations for a batch of images.
    
    Args:
        model: Trained ConceptAwareGeoModel
        image_paths: List of image file paths
        cell_centers: Geocell centers tensor
        metadata: Model metadata dict
        transform: Preprocessing transforms
        device: Device for inference
        batch_size: Batch size for inference
        
    Returns:
        List of prediction dicts
    """
    results = []
    
    for i in tqdm(range(0, len(image_paths), batch_size), desc="Predicting"):
        batch_paths = image_paths[i:i + batch_size]
        
        # Load and preprocess batch
        batch_tensors = []
        for path in batch_paths:
            img = Image.open(path).convert("RGB")
            img_tensor = transform(img)
            batch_tensors.append(img_tensor)
        
        batch = torch.stack(batch_tensors).to(device)
        
        # Predict
        batch_result = model.predict_location(batch, cell_centers)
        
        # Process each result
        idx_to_country = metadata.get("idx_to_country", {})
        concept_names = metadata.get("concept_names", [])
        
        for j in range(len(batch_paths)):
            pred_lat = batch_result["pred_lat"][j].item()
            pred_lng = batch_result["pred_lng"][j].item()
            pred_cell = batch_result["pred_cell"][j].item()
            
            country_probs = batch_result["country_probs"][j]
            top_country_idx = country_probs.argmax().item()
            top_country_prob = country_probs[top_country_idx].item()
            top_country = idx_to_country.get(top_country_idx, f"Country_{top_country_idx}")
            
            concept_probs = batch_result["concept_probs"][j]
            top_concept_idx = concept_probs.argmax().item()
            top_concept_prob = concept_probs[top_concept_idx].item()
            top_concept = concept_names[top_concept_idx] if top_concept_idx < len(concept_names) else f"Concept_{top_concept_idx}"
            
            results.append({
                "image_path": str(batch_paths[j]),
                "pred_lat": pred_lat,
                "pred_lng": pred_lng,
                "pred_cell": pred_cell,
                "top_country": top_country,
                "top_country_prob": top_country_prob,
                "top_concept": top_concept,
                "top_concept_prob": top_concept_prob,
            })
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Predict geographic location from images using trained GeoGuessr model"
    )
    parser.add_argument(
        "--checkpoint",
        type=str,
        required=True,
        help="Path to model checkpoint (.pt file)",
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Path to a single image for prediction",
    )
    parser.add_argument(
        "--image_dir",
        type=str,
        default=None,
        help="Directory containing images for batch prediction",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output CSV file for batch predictions",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=16,
        help="Batch size for inference",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device to use (cuda/cpu). Auto-detected if not specified.",
    )
    
    args = parser.parse_args()
    
    # Validate arguments
    if args.image is None and args.image_dir is None:
        parser.error("Either --image or --image_dir must be specified")
    
    # Setup device
    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load model
    model, cell_centers, metadata = load_model_for_inference(args.checkpoint, device)
    
    # Get transforms from the encoder
    transform = get_transforms_from_processor(model.image_encoder.image_processor)
    
    if args.image:
        # Single image prediction
        image_path = Path(args.image)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image_tensor = load_and_preprocess_image(str(image_path), transform, device)
        result = predict_single(model, image_tensor, cell_centers, metadata)
        
        print("\n" + "=" * 60)
        print(f"Prediction for: {image_path.name}")
        print("=" * 60)
        print(f"  Location: ({result['pred_lat']:.4f}, {result['pred_lng']:.4f})")
        print(f"  Country:  {result['top_country']} ({result['top_country_prob']:.2%})")
        print(f"  Concept:  {result['top_concept']} ({result['top_concept_prob']:.2%})")
        print(f"  Cell:     {result['pred_cell']}")
        print("=" * 60)
        
    else:
        # Batch prediction
        image_dir = Path(args.image_dir)
        if not image_dir.exists():
            raise FileNotFoundError(f"Directory not found: {image_dir}")
        
        # Find all images
        image_extensions = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}
        image_paths = [
            p for p in image_dir.iterdir()
            if p.suffix.lower() in image_extensions
        ]
        
        if not image_paths:
            raise ValueError(f"No images found in {image_dir}")
        
        logger.info(f"Found {len(image_paths)} images in {image_dir}")
        
        results = predict_batch(
            model, image_paths, cell_centers, metadata, transform, device, args.batch_size
        )
        
        # Save results
        if args.output:
            import csv
            output_path = Path(args.output)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            with output_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=results[0].keys())
                writer.writeheader()
                writer.writerows(results)
            
            logger.info(f"Saved {len(results)} predictions to {output_path}")
        else:
            # Print summary
            print("\n" + "=" * 80)
            print(f"Predictions for {len(results)} images")
            print("=" * 80)
            for r in results[:10]:  # Show first 10
                print(f"  {Path(r['image_path']).name}: ({r['pred_lat']:.4f}, {r['pred_lng']:.4f}) - {r['top_country']}")
            if len(results) > 10:
                print(f"  ... and {len(results) - 10} more")
            print("=" * 80)


if __name__ == "__main__":
    main()









