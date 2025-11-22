#!/usr/bin/env python3
"""
Training script for Concept-Aware Global Image-GPS Alignment.
"""

import argparse
import logging
import os
import csv
from typing import Dict, Tuple, List, Optional
from pathlib import Path
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from collections import Counter
from PIL import Image as PILImage
from datetime import datetime
import pandas as pd
from src.dataset import PanoramaCBMDataset, create_splits_stratified, get_transforms_from_processor
from src.iwo_dataset import CBMDataset
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.losses import contrastive_alignment_loss, concept_divergence_loss, coordinate_loss
from src.concepts.utils import extract_concepts_from_dataset
from src.evaluation import denormalize_coordinates, haversine_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ------- Constants -------
ENCODER_MODEL_TO_NAME = {
    "geolocal/StreetCLIP": "streetclip",
    "facebook/dinov3-vit7b16-pretrain-lvd1689m": "dinov3",
    "facebook/dinov2-base": "dinov2",
}

# ------- Helper Functions -------
def collate_batch(batch):
    """
    Custom collate function to handle variable-length metadata fields.
    Metadata dict contains 'images' which is a list of variable length.
    """
    images = torch.stack([item[0] for item in batch])
    concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    coordinates = torch.stack([item[3] for item in batch])
    metadata = [item[4] for item in batch]  # Keep as list of dicts, don't collate
    
    return images, concept_indices, target_indices, coordinates, metadata

@torch.no_grad()
def visualize_predictions(model, val_loader, concept_names, idx_to_country, device, args, checkpoint_dir, epoch, num_samples=4):
    """
    Visualize top predicted concepts and in-batch location retrieval for validation samples.
    Creates a single combined chart for all samples, saves it to disk, and logs to wandb if enabled.
    Self-matches are masked out (diagonal of similarity matrix set to -inf) to ensure
    proper retrieval evaluation. Only uses validation set (unseen during training).
    """
    model.eval()
    logger.info(f"\n=== Visualizing Predictions (Top 5 Concepts & In-Batch Retrieval) ===")
    
    # Create visualization directory
    viz_dir = checkpoint_dir / "visualizations" / f"epoch_{epoch}"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch
    wandb_images = []
    
    for batch in val_loader:
        images, concept_indices, _, coords, metadata = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_indices = concept_indices.to(device)
        
        # Forward pass
        z_img, z_loc, country_logits, pred_coords = model(images, coords)
        
        # In-Batch Retrieval Similarity
        z_img_norm = torch.nn.functional.normalize(z_img, p=2, dim=1)
        z_loc_norm = torch.nn.functional.normalize(z_loc, p=2, dim=1)
        similarity = torch.matmul(z_img_norm, z_loc_norm.t())
        
        # Mask out diagonal (self-matches) by setting to -inf
        # This prevents the model from matching an image with its own location
        mask = torch.eye(len(images), device=similarity.device, dtype=torch.bool)
        similarity_masked = similarity.masked_fill(mask, float('-inf'))
        
        # Create single figure for all samples
        # Height per sample = 4 inches, Width = 15 inches
        n_display = min(len(images), num_samples)
        
        for i in range(n_display):
            fig, axes = plt.subplots(2, 1, figsize=(10, 8))
            
            # Find best match (diagonal is masked, so no self-matches possible)
            best_loc_idx = similarity_masked[i].argmax().item()
            
            ax_img = axes[0]
            ax_bar = axes[1]
            
            # 1. Image & Concepts
            probs = z_img[i]
            top_scores, top_indices = torch.topk(probs, k=5)
            
            # Denormalize image for display
            # StreetCLIP uses CLIP mean/std. We need to inverse normalize.
            # Mean: (0.481, 0.457, 0.408), Std: (0.268, 0.261, 0.275)
            img_cpu = images[i].cpu().permute(1, 2, 0).numpy()
            mean = np.array([0.48145466, 0.4578275, 0.40821073])
            std = np.array([0.26862954, 0.26130258, 0.27577711])
            img_disp = std * img_cpu + mean
            img_disp = np.clip(img_disp, 0, 1)
            
            # Top panel: Image + Retrieval Info
            ax_img.imshow(img_disp)
            ax_img.axis('off')
            
            # Retrieval Info
            gt_lat = metadata[i]['lat']
            gt_lng = metadata[i]['lng']
            gt_country = metadata[i]['country']
            
            # Get predicted coordinates from the regression head
            pred_coords_np = pred_coords[i].cpu().numpy()
            
            # If coordinates are normalized, denormalize them. The dataset returns raw degrees,
            # but model output might need checking depending on how it was trained.
            # Based on recent fixes, dataset returns raw degrees and model learns raw degrees.
            # But let's be safe and check if we need to denormalize if they are small.
            # ACTUALLY, in this script we set use_normalized_coordinates=False, so they are raw degrees.
            pred_lat = pred_coords_np[0]
            pred_lng = pred_coords_np[1]
            
            # Calculate Haversine distance
            # We need to use the same distance function as training/eval
            # Convert to tensors for haversine_distance
            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = pred_coords[i].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()
            
            # Classification Prediction
            pred_country_idx = country_logits[i].argmax().item()
            pred_country_cls = idx_to_country[pred_country_idx]
            
            # GT Concept
            gt_concept_idx = concept_indices[i].item()
            gt_concept_name = concept_names[gt_concept_idx]
            
            # Top predicted concept
            top_concept_idx = top_indices[0].item()
            top_concept_name = concept_names[top_concept_idx]
            
            # Calculate probabilities for display
            country_prob = torch.softmax(country_logits[i], dim=0)[pred_country_idx].item()
            gt_country_idx_val = -1
            # This is inefficient but safe way to find index by value if idx_to_country is dict
            for k, v in idx_to_country.items():
                if v == gt_country:
                    gt_country_idx_val = k
                    break
            gt_country_prob = 0.0
            if gt_country_idx_val != -1:
                gt_country_prob = torch.softmax(country_logits[i], dim=0)[gt_country_idx_val].item()

            # title = f"GT: {gt_country} ({gt_lat:.2f}, {gt_lng:.2f}) | Concept: {gt_concept_name}\n"
            # title += f"Pred: {pred_country_cls} ({country_prob:.2f}) | ({pred_lat:.2f}, {pred_lng:.2f})\n"
            # title += f"Error: {distance_km:.1f} km | Top Concept: {top_concept_name}"
            title = f"Epoch {epoch} | Image ID: {metadata[i]['pano_id']}\n"
            title += f"Country: Pred: {pred_country_cls} | True: {gt_country}\n"
            title += f"Coords: Pred({pred_lat:.3f}, {pred_lng:.3f}) | True({gt_lat:.3f}, {gt_lng:.3f})\n"
            title += f"Concept: GT: {gt_concept_name} | Pred: {top_concept_name}" + f"✗" if gt_concept_idx == top_concept_idx else f"✓" + f"\n"
            title += f"Dist Error: {distance_km:.1f} km"
            
            ax_img.set_title(title, fontsize=10)
            
            # Bottom panel: Bar Chart of Concepts
            scores_np = top_scores.cpu().numpy()
            concepts_np = [concept_names[idx.item()] for idx in top_indices]
            top_indices_cpu = top_indices.cpu().numpy()
            
            # Color logic: Orange if GT, else SteelBlue
            bar_colors = ['orange' if idx == gt_concept_idx else 'steelblue' for idx in top_indices_cpu]
            
            y_pos = np.arange(len(concepts_np))
            
            ax_bar.barh(y_pos, scores_np, align='center', color=bar_colors)
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(concepts_np)
            ax_bar.invert_yaxis()  # labels read top-to-bottom
            ax_bar.set_xlabel('Activation Score')
            
            # Check if GT concept is in top 5
            gt_in_top5 = gt_concept_idx in top_indices_cpu
            bar_title = "Top 5 Predicted Concepts"
            if not gt_in_top5:
                bar_title += f" | GT: {gt_concept_name}"
            ax_bar.set_title(bar_title)

            plt.tight_layout()
            
            # Save to disk as one image per sample
            save_path = viz_dir / f"sample_{i}.png"
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            
            if args.use_wandb:
                wandb_images.append(wandb.Image(str(save_path), caption=f"Epoch {epoch} Sample {i}"))
            
            plt.close(fig)
            logger.info(f"Saved visualization for sample {i} to {save_path}")
            
        break # Only one batch
    
    # Log all images to wandb in a single key for grid view
    if args.use_wandb and wandb_images:
        wandb.log({f"predictions/epoch_{epoch}": wandb_images})

@torch.no_grad()
def dump_diagnostics(
    model,
    dataloader,
    device,
    output_path: Path,
    concept_names: List[str],
    idx_to_country: Dict[int, str],
    max_samples: int = 64,
    log_to_wandb: bool = False,
    wandb_step: Optional[int] = None,
):
    model.eval()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    rows = []

    # We will use in-batch retrieval for location prediction in diagnostics for now,
    # consistent with visualize_predictions.
    
    for batch in dataloader:
        if len(rows) >= max_samples:
            break
            
        images, concept_idx, target_idx, coords, metadata = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        target_idx = target_idx.to(device)
        
        # Forward pass
        z_img, z_loc, country_logits, pred_coords = model(images, coords)
        
        # Concept probabilities
        concept_probs = torch.softmax(z_img, dim=1)
        
        # Country probabilities
        country_probs = torch.softmax(country_logits, dim=1)
        
        # In-Batch Retrieval for Location (exclude self if possible? No, for diagnostics 
        # we ideally want to see if it retrieves the correct one, but if we are 
        # using the same batch, it might trivially retrieve itself. 
        # However, typically retrieval is done against a gallery. 
        # Here we just do in-batch retrieval as a proxy.)
        z_img_norm = torch.nn.functional.normalize(z_img, p=2, dim=1)
        z_loc_norm = torch.nn.functional.normalize(z_loc, p=2, dim=1)
        similarity = torch.matmul(z_img_norm, z_loc_norm.t())
        
        # Mask self for retrieval to be non-trivial
        mask = torch.eye(len(images), device=similarity.device, dtype=torch.bool)
        similarity_masked = similarity.masked_fill(mask, float('-inf'))
        
        best_loc_indices = similarity_masked.argmax(dim=1)
        
        for i in range(len(images)):
            if len(rows) >= max_samples:
                break
                
            # Concept info
            top_concept_idx = concept_probs[i].argmax().item()
            top_concept = concept_names[top_concept_idx]
            true_concept_idx_val = concept_idx[i].item()
            true_concept = concept_names[true_concept_idx_val]
            
            # Country info
            pred_country_idx = country_probs[i].argmax().item()
            pred_country_cls = idx_to_country[pred_country_idx]
            true_country_idx_val = target_idx[i].item()
            true_country_cls = idx_to_country[true_country_idx_val]
            
            # Location info (Retrieval based)
            best_loc_idx = best_loc_indices[i].item()
            
            gt_lat = metadata[i]['lat']
            gt_lng = metadata[i]['lng']
            pred_lat = metadata[best_loc_idx]['lat']
            pred_lng = metadata[best_loc_idx]['lng']
            
            # Calculate distance
            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = coords[best_loc_idx].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()
            
            # Build row dict - only include pano_id if it exists in metadata
            row = {
                "pred_lat": float(pred_lat),
                "pred_lng": float(pred_lng),
                "true_lat": float(gt_lat),
                "true_lng": float(gt_lng),
                "distance_km": distance_km,
                "top_concept": top_concept,
                "true_concept": true_concept,
                "pred_country_cls": pred_country_cls,
                "true_country_cls": true_country_cls,
                "concept_correct": bool(top_concept_idx == true_concept_idx_val),
                "country_correct": bool(pred_country_idx == true_country_idx_val)
            }
            
            # Only add pano_id if it exists in metadata
            if "pano_id" in metadata[i]:
                row["pano_id"] = metadata[i]["pano_id"]
            
            rows.append(row)

    if not rows:
        return

    fieldnames = list(rows[0].keys())
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
        
    logger.info(f"Dumped {len(rows)} diagnostic samples to {output_path}")

    if log_to_wandb:
        table = wandb.Table(columns=fieldnames)
        for row in rows:
            table.add_data(*[row[col] for col in fieldnames])
        wandb.log({f"diagnostics/{output_path.stem}": table}, step=wandb_step)

def create_checkpoint_dir(
    encoder_model: str = None,
    country_filter: str = None,
    coordinate_loss_type: str = "haversine",
) -> Path:
    """Create checkpoint directory with format:
    results/concept-aware/<encoder_name>/<country>/<loss_type>/<timestamp>/
    where:
    - encoder_name: sanitized encoder model name (replace "/" with "-")
    - country: country name or "global" if no country filter
    - loss_type: coordinate loss type (haversine, mse, sphere, etc.)
    - timestamp: formatted date and time (YYYY-MM-DD_HH-MM-SS)

    Creates subdirectories: checkpoints/, logs/, visualizations/, diagnostics/
    """
    now = datetime.now()
    timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")

    # Encoder name: sanitize model name (replace "/" with "-")
    if encoder_model:
        encoder_name = ENCODER_MODEL_TO_NAME.get(encoder_model, encoder_model)
        encoder_name = encoder_name.replace("/", "-")
    else:
        encoder_name = "unknown"

    # Country: use filter or "global" if None
    country = country_filter if country_filter else "global"

    # Loss type: use provided coordinate loss type
    loss_type = coordinate_loss_type.lower()

    # Create full directory structure
    timestamp_dir = (
        Path("results")
        / "concept-aware"
        / encoder_name
        / country
        / loss_type
        / timestamp
    )
    timestamp_dir.mkdir(parents=True, exist_ok=True)

    # Create subdirectories
    (timestamp_dir / "checkpoints").mkdir(exist_ok=True)
    (timestamp_dir / "logs").mkdir(exist_ok=True)
    (timestamp_dir / "visualizations").mkdir(exist_ok=True)
    (timestamp_dir / "diagnostics").mkdir(exist_ok=True)

    return timestamp_dir

# ------- Main Training Function -------
def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # ------- Setup Data & Image Encoder -------
    logger.info("Initializing Dataset...")

    # Use StreetCLIP transforms
    base_encoder = StreetCLIPEncoder(StreetCLIPConfig(model_name=args.encoder_model))
    transforms = get_transforms_from_processor(base_encoder.image_processor)
    
    full_dataset = PanoramaCBMDataset(
        transform=transforms,
        require_coordinates=True,
        country=args.country_filter,
        use_normalized_coordinates=False
    )
    #full_dataset = CBMDataset(dataframe=pd.read_csv("/scratch-shared/pnair/Project_AI/data/sa-dataset.csv"))
    
    # Diagnostic: Check concept distribution
    all_concepts = [s['meta_name'] for s in full_dataset.samples]
    concept_counts = Counter(all_concepts)
    logger.info(f"Total samples: {len(full_dataset.samples)}, Concepts: {len(concept_counts)}")
    logger.info(f"Samples per concept - Min: {min(concept_counts.values())}, Max: {max(concept_counts.values())}, Avg: {len(full_dataset.samples)/len(concept_counts):.1f}")
    logger.info("Concept distribution (Top 10):")
    for name, count in concept_counts.most_common(10):
        logger.info(f"  {name}: {count}")
    
    # Use more balanced split: 70/20/10 to get more validation samples
    # With only 251 samples, 10% validation gives only ~21 samples which is too small
    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )
    
    # Create subset datasets (need to implement wrapper or just list sampling)
    # Re-using the logic from dataset.py main block
    from src.dataset import SubsetDataset
    train_dataset = SubsetDataset(full_dataset, train_samples)
    val_dataset = SubsetDataset(full_dataset, val_samples)
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=4,
        pin_memory=True,
        drop_last=True,  # Important for contrastive loss stability
        collate_fn=collate_batch
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4,
        collate_fn=collate_batch
    )
    
    test_dataset = SubsetDataset(full_dataset, test_samples)
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch
    )
    
    logger.info(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")
    logger.info(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}, Test={len(test_loader)}")
    
    # Check train concept distribution
    train_concepts = [s['meta_name'] for s in train_samples]
    train_concept_counts = Counter(train_concepts)
    logger.info(f"Train set: {len(train_concepts)} samples across {len(train_concept_counts)} concepts")
    logger.info(f"  Samples per concept - Min: {min(train_concept_counts.values())}, Max: {max(train_concept_counts.values())}, Avg: {len(train_concepts)/len(train_concept_counts):.1f}")

    # ------- Concept Extraction & Encoding -------
    logger.info("Extracting and encoding concepts...")
    # We need concepts from the ENTIRE dataset to build the basis, not just training set
    concept_names, concept_map = extract_concepts_from_dataset(full_dataset)
    
    logger.info(f"Found {len(concept_names)} unique concepts.")

    # Verify concept alignment
    # concept_names should match keys in full_dataset.concept_to_idx (both sorted by name)
    dataset_concepts = sorted(full_dataset.concept_to_idx.keys())
    if concept_names != dataset_concepts:
        logger.error("Concept mismatch between extract_concepts_from_dataset and dataset.concept_to_idx!")
        logger.error(f"Extract: {concept_names[:5]}...")
        logger.error(f"Dataset: {dataset_concepts[:5]}...")
        raise ValueError("Concept alignment failed")
    
    # Further verification: check index mapping
    for i, name in enumerate(concept_names):
        idx = full_dataset.concept_to_idx[name]
        if idx != i:
             raise ValueError(f"Index mismatch for {name}: expected {i}, got {idx}")
    logger.info("Concept alignment verified.")
    
    # Encode concepts to get E_concept
    # Move encoder to device for inference - ensure model is explicitly moved
    base_encoder.model.to(device)
    base_encoder = base_encoder.to(device)  # Also move any other components
    # Verify model is on the correct device (compare device types, not exact strings)
    model_device = next(base_encoder.model.parameters()).device
    if model_device.type != device.type:
        raise RuntimeError(f"Model is on {model_device.type} but expected {device.type}")
    logger.info(f"Base encoder device: {model_device}")
    
    # Batch encode concepts to avoid OOM if many concepts
    concept_texts = [concept_map[name] for name in concept_names]
    concept_embeddings = []
    
    batch_size = 32
    with torch.no_grad():
        for i in range(0, len(concept_texts), batch_size):
            batch_texts = concept_texts[i:i+batch_size]
            feats = base_encoder.get_text_features(batch_texts)
            concept_embeddings.append(feats.cpu())
            
    E_concept = torch.cat(concept_embeddings, dim=0) # [k, d]
    logger.info(f"Concept Basis shape: {E_concept.shape}")

    # ------- Initialize Concept-Aware Model -------
    logger.info("Initializing ConceptAwareGeoModel...")
    # Configure StreetCLIP for training (finetune=False usually for encoder)
    encoder_config = StreetCLIPConfig(
        model_name=args.encoder_model, 
        finetune=args.finetune_encoder,
        device=device
    )
    # Re-create encoder to ensure clean state / config application
    image_encoder = StreetCLIPEncoder(encoder_config)
    actual_feature_dim = image_encoder.feature_dim
    logger.info(f"StreetCLIP vision encoder dimension: {actual_feature_dim}")
    
    # Project concept embeddings to match vision encoder dimension if needed
    if E_concept.shape[1] != actual_feature_dim:
        logger.warning(
            f"Dimension mismatch: Concept embeddings have dim {E_concept.shape[1]}, "
            f"but vision encoder has dim {actual_feature_dim}. "
            f"Projecting concept embeddings to match vision dimension..."
        )
        # Project concept embeddings to match vision dimension
        projection = nn.Linear(E_concept.shape[1], actual_feature_dim, bias=False).to(device)
        # Initialize with small random values to preserve semantic information
        nn.init.normal_(projection.weight, mean=0.0, std=0.02)
        
        with torch.no_grad():
            E_concept_projected = []
            for i in range(0, len(E_concept), batch_size):
                batch = E_concept[i:i+batch_size].to(device)
                projected = projection(batch)
                E_concept_projected.append(projected.cpu())
            E_concept = torch.cat(E_concept_projected, dim=0)
        logger.info(f"Projected Concept Basis shape: {E_concept.shape}")
    
    # Final verification
    assert E_concept.shape[1] == actual_feature_dim, \
        f"Concept basis dimension {E_concept.shape[1]} must match vision encoder dimension {actual_feature_dim}"
    
    model = ConceptAwareGeoModel(
        image_encoder=image_encoder,
        concept_features=E_concept,
        num_concepts=len(concept_names),
        num_countries=len(full_dataset.country_to_idx),
        streetclip_dim=actual_feature_dim,
        location_encoder_dim=512 # GeoCLIP default
    )
    model.to(device)

    # ------- Optimizer & Scheduler -------
    optimizer = torch.optim.AdamW(
        model.parameters_to_optimize(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # ------- Setup Output Directory -------
    if args.output_dir is None:
        checkpoint_dir = create_checkpoint_dir(
            encoder_model=args.encoder_model,
            country_filter=args.country_filter,
            coordinate_loss_type=args.coordinate_loss_type
        )
    else:
        checkpoint_dir = Path(args.output_dir)
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        # Ensure subdirectories exist
        (checkpoint_dir / "checkpoints").mkdir(exist_ok=True)
        (checkpoint_dir / "logs").mkdir(exist_ok=True)
        (checkpoint_dir / "visualizations").mkdir(exist_ok=True)
        (checkpoint_dir / "diagnostics").mkdir(exist_ok=True)
    
    logger.info(f"Checkpoint directory: {checkpoint_dir}")
    
    # Setup file logging
    log_file = checkpoint_dir / "logs" / "training.log"
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(logging.INFO)
    file_formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(file_formatter)
    logger.addHandler(file_handler)
    logger.info(f"Logging to {log_file}")

    # ------- Training Loop -------
    if args.use_wandb:
        # Create tags
        tags = [
            args.encoder_model,
            args.country_filter if args.country_filter else "global",
            "finetuned" if args.finetune_encoder else "frozen",
            "concept_aware",
            args.coordinate_loss_type
        ]
        
        wandb.init(
            project="concept-aware-geolocation",
            name=f"concept-aware-geolocation-{args.country_filter if args.country_filter else 'global'}-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
            config=args,
            tags=tags,
            dir=str(checkpoint_dir)
        )

    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_contrastive = 0
        total_divergence = 0
        total_concept_loss = 0
        total_country_loss = 0
        total_coords_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_concept_correct = 0
        total_concept_count = 0
        total_country_correct = 0
        total_country_count = 0
        
        for batch in pbar:
            # ------- Forward Pass -------
            images, concept_idx, target_idx, coords, _ = batch
            images = images.to(device)
            coords = coords.to(device)
            concept_idx = concept_idx.to(device)
            target_idx = target_idx.to(device)
            
            z_img, z_loc, country_logits, pred_coords = model(images, coords)
            
            # ------- Compute Metrics -------
            pred_concepts = z_img.argmax(dim=1)
            concept_correct = (pred_concepts == concept_idx).sum().item()
            total_concept_correct += concept_correct
            total_concept_count += len(concept_idx)
            
            pred_countries = country_logits.argmax(dim=1)
            country_correct = (pred_countries == target_idx).sum().item()
            total_country_correct += country_correct
            total_country_count += len(target_idx)
            
            # ------- Compute Loss -------
            loss_contrastive = contrastive_alignment_loss(z_img, z_loc, temperature=args.temperature)
            loss_divergence = concept_divergence_loss(z_img, z_loc, sigma=args.sigma)
            # Use label smoothing to reduce overfitting
            loss_concept = nn.functional.cross_entropy(z_img, concept_idx, label_smoothing=args.label_smoothing)
            # Country loss
            loss_country = nn.functional.cross_entropy(country_logits, target_idx, label_smoothing=args.label_smoothing)
            # Coordinate loss
            loss_coords = coordinate_loss(pred_coords, coords, loss_type=args.coordinate_loss_type)
            
            loss = loss_contrastive + args.lambda_divergence * loss_divergence + args.lambda_concept * loss_concept + args.lambda_country * loss_country + args.lambda_coords * loss_coords
            
            # ------- Backward Pass -------
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # ------- Batch Logging -------
            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_divergence += loss_divergence.item()
            total_concept_loss += loss_concept.item()
            total_country_loss += loss_country.item()
            total_coords_loss += loss_coords.item()
            
            batch_concept_acc = concept_correct / len(concept_idx)
            batch_country_acc = country_correct / len(target_idx)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "cont": f"{loss_contrastive.item():.4f}",
                "div": f"{loss_divergence.item():.4f}",
                "cls": f"{loss_concept.item():.4f}",
                "ctry": f"{loss_country.item():.4f}",
                "dist": f"{loss_coords.item():.4f}",
                "concept_acc": f"{batch_concept_acc:.3f}",
                "country_acc": f"{batch_country_acc:.3f}"
            })
            
            if args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_contrastive": loss_contrastive.item(),
                    "batch_divergence": loss_divergence.item(),
                    "batch_concept_loss": loss_concept.item(),
                    "batch_country_loss": loss_country.item(),
                    "batch_coords_loss": loss_coords.item(),
                    "batch_concept_accuracy": batch_concept_acc,
                    "batch_country_accuracy": batch_country_acc
                })

        # ------- Epoch Validation -------
        avg_train_loss = total_loss / len(train_loader)
        avg_train_concept_loss = total_concept_loss / len(train_loader)
        avg_train_country_loss = total_country_loss / len(train_loader)
        avg_train_coords_loss = total_coords_loss / len(train_loader)
        train_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
        train_country_acc = total_country_correct / total_country_count if total_country_count > 0 else 0.0
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Concept Loss: {avg_train_concept_loss:.4f}, Country Loss: {avg_train_country_loss:.4f}, Coords Loss: {avg_train_coords_loss:.4f}, Train Concept Acc: {train_concept_acc:.4f}, Train Country Acc: {train_country_acc:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "train_concept_loss": avg_train_concept_loss,
                "train_country_loss": avg_train_country_loss,
                "train_coords_loss": avg_train_coords_loss,
                "train_concept_accuracy": train_concept_acc,
                "train_country_accuracy": train_country_acc
            })
        
        val_metrics = validate(model, val_loader, device, args)
        val_concept_acc = val_metrics['concept_acc']
        
        # ------- Learning Rate Scheduling & Early Stopping -------
        scheduler.step(val_metrics['loss'])
        if val_concept_acc > best_val_acc:
            best_val_acc = val_concept_acc
            patience_counter = 0
            # Save best model
            best_model_path = checkpoint_dir / "checkpoints" / "best_model.pt"
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Concept Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs. Best Val Acc: {best_val_acc:.4f}")
                break
        
        # ------- Visualization & Diagnostics -------
        if (epoch + 1) % args.save_interval == 0:
             visualize_predictions(model, val_loader, concept_names, full_dataset.idx_to_country, device, args, checkpoint_dir, epoch + 1)
             
             # Diagnostics
             diag_path = checkpoint_dir / "diagnostics" / f"epoch_{epoch+1}.csv"
             dump_diagnostics(
                 model, 
                 val_loader, 
                 device, 
                 diag_path, 
                 concept_names,
                 full_dataset.idx_to_country,
                 log_to_wandb=args.use_wandb, 
                 wandb_step=epoch+1
             )

        # ------- Save Checkpoint -------
        if (epoch + 1) % args.save_interval == 0:
            save_path = checkpoint_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt"
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")

    # ------- Final Test Evaluation -------
    logger.info("Evaluating on test set...")
    # Load best model if saved
    best_model_path = checkpoint_dir / "checkpoints" / "best_model.pt"
    if best_model_path.exists():
        model.load_state_dict(torch.load(best_model_path))
        logger.info("Loaded best model for testing.")
        
    test_metrics = validate(model, test_loader, device, args)
    logger.info(f"Test Metrics: {test_metrics}")
    
    if args.use_wandb:
        wandb.log({f"test/{k}": v for k, v in test_metrics.items()})
        
        # Dump test diagnostics
        test_diag_path = checkpoint_dir / "diagnostics" / "test_diagnostics.csv"
        dump_diagnostics(
            model, 
            test_loader, 
            device, 
            test_diag_path, 
            concept_names,
            full_dataset.idx_to_country,
            max_samples=len(test_dataset), # Dump all test samples
            log_to_wandb=True
        )

# ------- Validation Function -------
@torch.no_grad()
def validate(model, val_loader, device, args):
    model.eval()
    total_loss = 0
    total_contrastive = 0
    total_divergence = 0
    total_concept_loss = 0
    total_country_loss = 0
    total_coords_loss = 0
    total_concept_correct = 0
    total_concept_count = 0
    total_country_correct = 0
    total_country_count = 0
    
    for batch in val_loader:
        images, concept_idx, target_idx, coords, _ = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        target_idx = target_idx.to(device)
        
        z_img, z_loc, country_logits, pred_coords = model(images, coords)
        
        # Compute concept accuracy
        pred_concepts = z_img.argmax(dim=1)
        concept_correct = (pred_concepts == concept_idx).sum().item()
        total_concept_correct += concept_correct
        total_concept_count += len(concept_idx)
        
        # Compute country accuracy
        pred_countries = country_logits.argmax(dim=1)
        country_correct = (pred_countries == target_idx).sum().item()
        total_country_correct += country_correct
        total_country_count += len(target_idx)
        
        loss_contrastive = contrastive_alignment_loss(z_img, z_loc, temperature=args.temperature)
        loss_divergence = concept_divergence_loss(z_img, z_loc, sigma=args.sigma)
        loss_concept = nn.functional.cross_entropy(z_img, concept_idx, label_smoothing=args.label_smoothing)
        loss_country = nn.functional.cross_entropy(country_logits, target_idx, label_smoothing=args.label_smoothing)
        loss_coords = coordinate_loss(pred_coords, coords, loss_type=args.coordinate_loss_type)
        
        loss = loss_contrastive + args.lambda_divergence * loss_divergence + args.lambda_concept * loss_concept + args.lambda_country * loss_country + args.lambda_coords * loss_coords
        
        total_loss += loss.item()
        total_contrastive += loss_contrastive.item()
        total_divergence += loss_divergence.item()
        total_concept_loss += loss_concept.item()
        total_country_loss += loss_country.item()
        total_coords_loss += loss_coords.item()
        
    avg_loss = total_loss / len(val_loader)
    avg_concept_loss = total_concept_loss / len(val_loader)
    avg_country_loss = total_country_loss / len(val_loader)
    avg_coords_loss = total_coords_loss / len(val_loader)
    val_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
    val_country_acc = total_country_correct / total_country_count if total_country_count > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}, Concept Loss: {avg_concept_loss:.4f}, Country Loss: {avg_country_loss:.4f}, Coords Loss: {avg_coords_loss:.4f}, Val Concept Acc: {val_concept_acc:.4f}, Val Country Acc: {val_country_acc:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "val_loss": avg_loss,
            "val_contrastive": total_contrastive / len(val_loader),
            "val_divergence": total_divergence / len(val_loader),
            "val_concept_loss": avg_concept_loss,
            "val_country_loss": avg_country_loss,
            "val_coords_loss": avg_coords_loss,
            "val_concept_accuracy": val_concept_acc,
            "val_country_accuracy": val_country_acc
        })
    
    return {
        'loss': avg_loss,
        'concept_loss': avg_concept_loss,
        'concept_acc': val_concept_acc,
        'country_loss': avg_country_loss,
        'coords_loss': avg_coords_loss,
        'country_acc': val_country_acc
    }

# ------- Main Entry Point -------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Concept-Aware CBM")
    
    # ------- Data & Model Arguments -------
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP", help="Image Encoder model to use")
    parser.add_argument("--finetune_encoder", action="store_true", help="Whether to finetune the encoder")
    parser.add_argument("--country_filter", type=str, default=None, help="Filter for country")
    
    # ------- Training Hyperparameters -------
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size for training")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs to train")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate for optimizer")
    parser.add_argument("--weight_decay", type=float, default=0.1, help="Weight decay for optimizer")
    parser.add_argument("--early_stopping_patience", type=int, default=5, help="Number of epochs to wait before early stopping")
    
    # ------- Loss Hyperparameters -------
    parser.add_argument("--temperature", type=float, default=0.07, help="Temperature for contrastive alignment loss")
    parser.add_argument("--lambda_divergence", type=float, default=0.1, help="Weight for divergence loss")
    parser.add_argument("--lambda_concept", type=float, default=1.0, help="Weight for concept loss")
    parser.add_argument("--lambda_country", type=float, default=1.0, help="Weight for country loss")
    parser.add_argument("--lambda_coords", type=float, default=0.01, help="Weight for coordinate loss")
    parser.add_argument("--sigma", type=float, default=1.0, help="Sigma for concept divergence loss")
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--coordinate_loss_type", type=str, default="haversine", 
                        choices=["haversine", "mse", "sphere"],
                        help="Type of coordinate loss: haversine, mse, or sphere")
    
    # ------- Miscellaneous Arguments -------
    parser.add_argument("--output_dir", type=str, default=None, help="Output directory (auto-generated if not provided)")
    parser.add_argument("--save_interval", type=int, default=5, help="When to save checkpoints and run visualizations and diagnostics")
    parser.add_argument("--use_wandb", action="store_true", default=True, help="Whether to use Weights & Biases for logging")
    
    args = parser.parse_args()
    train(args)
