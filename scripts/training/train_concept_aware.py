#!/usr/bin/env python3
"""
Training script for Concept-Aware Global Image-GPS Alignment.
"""

import argparse
import logging
import os
from pathlib import Path
import io

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import wandb
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image as PILImage

from src.dataset import PanoramaCBMDataset, create_splits_stratified, get_transforms_from_processor
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.losses import contrastive_alignment_loss, concept_divergence_loss
from src.concepts.utils import extract_concepts_from_dataset
from src.evaluation import denormalize_coordinates, haversine_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@torch.no_grad()
def visualize_predictions(model, val_loader, concept_names, device, args, num_samples=5):
    """
    Visualize top predicted concepts and in-batch location retrieval for a few random samples.
    Logs a matplotlib figure to wandb if enabled.
    """
    model.eval()
    logger.info(f"\n=== Visualizing Predictions (Top 5 Concepts & In-Batch Retrieval) ===")
    
    # Get a batch
    for batch in val_loader:
        images, _, _, coords, metadata = batch
        images = images.to(device)
        coords = coords.to(device)
        
        # Forward pass
        z_img, z_loc = model(images, coords)
        
        # In-Batch Retrieval Similarity
        z_img_norm = torch.nn.functional.normalize(z_img, p=2, dim=1)
        z_loc_norm = torch.nn.functional.normalize(z_loc, p=2, dim=1)
        similarity = torch.matmul(z_img_norm, z_loc_norm.t())
        
        # Create figure
        fig, axes = plt.subplots(num_samples, 2, figsize=(15, 4 * num_samples))
        if num_samples == 1:
            axes = axes.reshape(1, -1)
            
        for i in range(min(num_samples, len(images))):
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
            
            # Left column: Image + Retrieval Info
            axes[i, 0].imshow(img_disp)
            axes[i, 0].axis('off')
            
            # Retrieval Info
            best_loc_idx = similarity[i].argmax().item()
            gt_lat = metadata['lat'][i].item()
            gt_lng = metadata['lng'][i].item()
            gt_country = metadata['country'][i]
            pred_lat = metadata['lat'][best_loc_idx].item()
            pred_lng = metadata['lng'][best_loc_idx].item()
            pred_country = metadata['country'][best_loc_idx]
            
            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = coords[best_loc_idx].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()
            
            title = f"GT: {gt_country} ({gt_lat:.2f}, {gt_lng:.2f})\n"
            title += f"Pred: {pred_country} ({pred_lat:.2f}, {pred_lng:.2f})\n"
            title += f"Error: {distance_km:.1f} km"
            if i == best_loc_idx:
                title += " (Self-match)"
            
            axes[i, 0].set_title(title, fontsize=10)
            
            # Right column: Bar Chart of Concepts
            scores_np = top_scores.cpu().numpy()
            concepts_np = [concept_names[idx.item()] for idx in top_indices]
            y_pos = np.arange(len(concepts_np))
            
            axes[i, 1].barh(y_pos, scores_np, align='center')
            axes[i, 1].set_yticks(y_pos)
            axes[i, 1].set_yticklabels(concepts_np)
            axes[i, 1].invert_yaxis()  # labels read top-to-bottom
            axes[i, 1].set_xlabel('Activation Score')
            axes[i, 1].set_title(f"GT Concept: {metadata['meta_name'][i]}")

        plt.tight_layout()
        
        # Log to WandB
        if args.use_wandb:
            # Convert plot to image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', dpi=100)
            buf.seek(0)
            pil_img = PILImage.open(buf)
            wandb.log({"predictions_viz": wandb.Image(pil_img)})
            buf.close()
        
        plt.close(fig)
        break # Only one batch

def train(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")

    # 1. Setup Data
    logger.info("Initializing Dataset...")
    # Use StreetCLIP transforms
    base_encoder = StreetCLIPEncoder(StreetCLIPConfig(model_name=args.encoder_model))
    transforms = get_transforms_from_processor(base_encoder.image_processor)
    
    full_dataset = PanoramaCBMDataset(
        transform=transforms,
        require_coordinates=True,
        country=args.country_filter 
    )
    
    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1
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
        drop_last=True  # Important for contrastive loss stability
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=4
    )

    # 2. Concept Extraction & Encoding
    logger.info("Extracting and encoding concepts...")
    # We need concepts from the ENTIRE dataset to build the basis, not just training set
    concept_names, concept_map = extract_concepts_from_dataset(full_dataset)
    
    logger.info(f"Found {len(concept_names)} unique concepts.")
    
    # Encode concepts to get E_concept
    # We move encoder to device for inference then back if needed, or keep it
    base_encoder.to(device)
    
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

    # 3. Initialize Model
    logger.info("Initializing ConceptAwareGeoModel...")
    # Configure StreetCLIP for training (finetune=False usually for encoder)
    encoder_config = StreetCLIPConfig(
        model_name=args.encoder_model, 
        finetune=args.finetune_encoder,
        device=device
    )
    # Re-create encoder to ensure clean state / config application
    image_encoder = StreetCLIPEncoder(encoder_config)
    
    model = ConceptAwareGeoModel(
        image_encoder=image_encoder,
        concept_features=E_concept,
        num_concepts=len(concept_names),
        streetclip_dim=image_encoder.feature_dim,
        location_encoder_dim=512 # GeoCLIP default
    )
    model.to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters_to_optimize(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    # 5. Training Loop
    if args.use_wandb:
        wandb.init(project="concept-aware-geolocation", config=args)

    logger.info("Starting training...")
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_contrastive = 0
        total_divergence = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        for batch in pbar:
            # Unpack batch
            # image, concept_idx, target_idx, coordinates, metadata
            images, _, _, coords, _ = batch
            
            images = images.to(device)
            coords = coords.to(device)
            
            # Forward
            z_img, z_loc = model(images, coords)
            
            # Loss
            loss_contrastive = contrastive_alignment_loss(z_img, z_loc, temperature=args.temperature)
            loss_divergence = concept_divergence_loss(z_img, z_loc, sigma=args.sigma)
            
            loss = loss_contrastive + args.lambda_divergence * loss_divergence
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_divergence += loss_divergence.item()
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "cont": f"{loss_contrastive.item():.4f}",
                "div": f"{loss_divergence.item():.4f}"
            })
            
            if args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_contrastive": loss_contrastive.item(),
                    "batch_divergence": loss_divergence.item()
                })

        # Validation
        avg_train_loss = total_loss / len(train_loader)
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}")
        
        validate(model, val_loader, device, args)
        
        # Visualization
        if (epoch + 1) % args.save_interval == 0:
             visualize_predictions(model, val_loader, concept_names, device, args)

        # Checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_path = Path(args.output_dir) / f"checkpoint_epoch_{epoch+1}.pt"
            save_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), save_path)
            logger.info(f"Saved checkpoint to {save_path}")


@torch.no_grad()
def validate(model, val_loader, device, args):
    model.eval()
    total_loss = 0
    total_contrastive = 0
    total_divergence = 0
    
    # We can also measure retrieval accuracy here (R@1, R@5)
    # For simplicity, just tracking loss for now
    
    for batch in val_loader:
        images, _, _, coords, _ = batch
        images = images.to(device)
        coords = coords.to(device)
        
        z_img, z_loc = model(images, coords)
        
        loss_contrastive = contrastive_alignment_loss(z_img, z_loc, temperature=args.temperature)
        loss_divergence = concept_divergence_loss(z_img, z_loc, sigma=args.sigma)
        
        loss = loss_contrastive + args.lambda_divergence * loss_divergence
        
        total_loss += loss.item()
        total_contrastive += loss_contrastive.item()
        total_divergence += loss_divergence.item()
        
    avg_loss = total_loss / len(val_loader)
    logger.info(f"Validation Loss: {avg_loss:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "val_loss": avg_loss,
            "val_contrastive": total_contrastive / len(val_loader),
            "val_divergence": total_divergence / len(val_loader)
        })


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Concept-Aware CBM")
    
    # Data & Model
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--finetune_encoder", action="store_true")
    parser.add_argument("--country_filter", type=str, default=None)
    
    # Training Hyperparams
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    
    # Loss Hyperparams
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_divergence", type=float, default=0.1)
    parser.add_argument("--sigma", type=float, default=1.0)
    
    # Misc
    parser.add_argument("--output_dir", type=str, default="results/concept_aware")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    
    args = parser.parse_args()
    train(args)
