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
from collections import Counter
from PIL import Image as PILImage

from src.dataset import PanoramaCBMDataset, create_splits_stratified, get_transforms_from_processor
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import ConceptAwareGeoModel
from src.losses import contrastive_alignment_loss, concept_divergence_loss
from src.concepts.utils import extract_concepts_from_dataset
from src.evaluation import denormalize_coordinates, haversine_distance

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
def visualize_predictions(model, val_loader, concept_names, device, args, num_samples=5):
    """
    Visualize top predicted concepts and in-batch location retrieval for validation samples.
    Creates separate charts for each sample and logs to wandb if enabled.
    Self-matches are masked out (diagonal of similarity matrix set to -inf) to ensure
    proper retrieval evaluation. Only uses validation set (unseen during training).
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
        
        # Mask out diagonal (self-matches) by setting to -inf
        # This prevents the model from matching an image with its own location
        mask = torch.eye(len(images), device=similarity.device, dtype=torch.bool)
        similarity_masked = similarity.masked_fill(mask, float('-inf'))
        
        # Create separate figures for each sample (excluding self-matches)
        samples_visualized = 0
        for i in range(len(images)):
            if samples_visualized >= num_samples:
                break
                
            # Find best match (diagonal is masked, so no self-matches possible)
            best_loc_idx = similarity_masked[i].argmax().item()
            
            # Create separate figure for this sample
            fig, axes = plt.subplots(1, 2, figsize=(15, 4))
            
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
            axes[0].imshow(img_disp)
            axes[0].axis('off')
            
            # Retrieval Info
            # metadata is now a list of dicts, not a dict of lists
            gt_lat = metadata[i]['lat']
            gt_lng = metadata[i]['lng']
            gt_country = metadata[i]['country']
            pred_lat = metadata[best_loc_idx]['lat']
            pred_lng = metadata[best_loc_idx]['lng']
            pred_country = metadata[best_loc_idx]['country']
            
            # Convert to tensors for haversine_distance if needed
            gt_coord_tensor = coords[i].unsqueeze(0)
            pred_coord_tensor = coords[best_loc_idx].unsqueeze(0)
            distance_km = haversine_distance(pred_coord_tensor, gt_coord_tensor).item()
            
            title = f"GT: {gt_country} ({gt_lat:.2f}, {gt_lng:.2f})\n"
            title += f"Pred: {pred_country} ({pred_lat:.2f}, {pred_lng:.2f})\n"
            title += f"Error: {distance_km:.1f} km"
            
            axes[0].set_title(title, fontsize=10)
            
            # Right column: Bar Chart of Concepts
            scores_np = top_scores.cpu().numpy()
            concepts_np = [concept_names[idx.item()] for idx in top_indices]
            y_pos = np.arange(len(concepts_np))
            
            axes[1].barh(y_pos, scores_np, align='center')
            axes[1].set_yticks(y_pos)
            axes[1].set_yticklabels(concepts_np)
            axes[1].invert_yaxis()  # labels read top-to-bottom
            axes[1].set_xlabel('Activation Score')
            axes[1].set_title(f"GT Concept: {metadata[i]['meta_name']}")

            plt.tight_layout()
            
            # Log to WandB with unique key for each sample
            if args.use_wandb:
                # Convert plot to image
                buf = io.BytesIO()
                plt.savefig(buf, format='png', dpi=100)
                buf.seek(0)
                pil_img = PILImage.open(buf)
                wandb.log({f"prediction_viz_sample_{samples_visualized}": wandb.Image(pil_img)})
                buf.close()
            
            plt.close(fig)
            samples_visualized += 1
        
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
    
    logger.info(f"Split sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_samples)}")
    logger.info(f"Batches per epoch: Train={len(train_loader)}, Val={len(val_loader)}")
    
    # Check train concept distribution
    train_concepts = [s['meta_name'] for s in train_samples]
    train_concept_counts = Counter(train_concepts)
    logger.info(f"Train set: {len(train_concepts)} samples across {len(train_concept_counts)} concepts")
    logger.info(f"  Samples per concept - Min: {min(train_concept_counts.values())}, Max: {max(train_concept_counts.values())}, Avg: {len(train_concepts)/len(train_concept_counts):.1f}")

    # 2. Concept Extraction & Encoding
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
        streetclip_dim=actual_feature_dim,
        location_encoder_dim=512 # GeoCLIP default
    )
    model.to(device)

    # 4. Optimizer
    optimizer = torch.optim.AdamW(
        model.parameters_to_optimize(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler for better convergence
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=3, verbose=True
    )

    # 5. Training Loop
    if args.use_wandb:
        wandb.init(project="concept-aware-geolocation", config=args)

    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    for epoch in range(args.epochs):
        model.train()
        total_loss = 0
        total_contrastive = 0
        total_divergence = 0
        total_concept_loss = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.epochs}")
        total_concept_correct = 0
        total_concept_count = 0
        
        for batch in pbar:
            # Unpack batch
            # image, concept_idx, target_idx, coordinates, metadata
            images, concept_idx, _, coords, _ = batch
            
            images = images.to(device)
            coords = coords.to(device)
            concept_idx = concept_idx.to(device)
            
            # Forward
            z_img, z_loc = model(images, coords)
            
            # Compute concept accuracy
            pred_concepts = z_img.argmax(dim=1)
            concept_correct = (pred_concepts == concept_idx).sum().item()
            total_concept_correct += concept_correct
            total_concept_count += len(concept_idx)
            
            # Loss
            loss_contrastive = contrastive_alignment_loss(z_img, z_loc, temperature=args.temperature)
            loss_divergence = concept_divergence_loss(z_img, z_loc, sigma=args.sigma)
            # Use label smoothing to reduce overfitting
            loss_concept = nn.functional.cross_entropy(z_img, concept_idx, label_smoothing=args.label_smoothing)
            
            loss = loss_contrastive + args.lambda_divergence * loss_divergence + args.lambda_concept * loss_concept
            
            # Backward
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Logging
            total_loss += loss.item()
            total_contrastive += loss_contrastive.item()
            total_divergence += loss_divergence.item()
            total_concept_loss += loss_concept.item()
            
            batch_concept_acc = concept_correct / len(concept_idx)
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}", 
                "cont": f"{loss_contrastive.item():.4f}",
                "div": f"{loss_divergence.item():.4f}",
                "cls": f"{loss_concept.item():.4f}",
                "concept_acc": f"{batch_concept_acc:.3f}"
            })
            
            if args.use_wandb:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_contrastive": loss_contrastive.item(),
                    "batch_divergence": loss_divergence.item(),
                    "batch_concept_loss": loss_concept.item(),
                    "batch_concept_accuracy": batch_concept_acc
                })

        # Validation
        avg_train_loss = total_loss / len(train_loader)
        avg_train_concept_loss = total_concept_loss / len(train_loader)
        train_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
        logger.info(f"Epoch {epoch+1} Train Loss: {avg_train_loss:.4f}, Concept Loss: {avg_train_concept_loss:.4f}, Train Concept Acc: {train_concept_acc:.4f}")
        
        if args.use_wandb:
            wandb.log({
                "train_loss": avg_train_loss,
                "train_concept_loss": avg_train_concept_loss,
                "train_concept_accuracy": train_concept_acc
            })
        
        val_metrics = validate(model, val_loader, device, args)
        val_concept_acc = val_metrics['concept_acc']
        
        # Learning rate scheduling
        scheduler.step(val_metrics['loss'])
        
        # Early stopping based on validation accuracy
        if val_concept_acc > best_val_acc:
            best_val_acc = val_concept_acc
            patience_counter = 0
            # Save best model
            best_model_path = Path(args.output_dir) / "best_model.pt"
            best_model_path.parent.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), best_model_path)
            logger.info(f"Saved best model with Val Concept Acc: {best_val_acc:.4f}")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping triggered after {epoch+1} epochs. Best Val Acc: {best_val_acc:.4f}")
                break
        
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
    total_concept_loss = 0
    total_concept_correct = 0
    total_concept_count = 0
    
    for batch in val_loader:
        images, concept_idx, _, coords, _ = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        
        z_img, z_loc = model(images, coords)
        
        # Compute concept accuracy
        pred_concepts = z_img.argmax(dim=1)
        concept_correct = (pred_concepts == concept_idx).sum().item()
        total_concept_correct += concept_correct
        total_concept_count += len(concept_idx)
        
        loss_contrastive = contrastive_alignment_loss(z_img, z_loc, temperature=args.temperature)
        loss_divergence = concept_divergence_loss(z_img, z_loc, sigma=args.sigma)
        loss_concept = nn.functional.cross_entropy(z_img, concept_idx, label_smoothing=args.label_smoothing)
        
        loss = loss_contrastive + args.lambda_divergence * loss_divergence + args.lambda_concept * loss_concept
        
        total_loss += loss.item()
        total_contrastive += loss_contrastive.item()
        total_divergence += loss_divergence.item()
        total_concept_loss += loss_concept.item()
        
    avg_loss = total_loss / len(val_loader)
    avg_concept_loss = total_concept_loss / len(val_loader)
    val_concept_acc = total_concept_correct / total_concept_count if total_concept_count > 0 else 0.0
    logger.info(f"Validation Loss: {avg_loss:.4f}, Concept Loss: {avg_concept_loss:.4f}, Val Concept Acc: {val_concept_acc:.4f}")
    
    if args.use_wandb:
        wandb.log({
            "val_loss": avg_loss,
            "val_contrastive": total_contrastive / len(val_loader),
            "val_divergence": total_divergence / len(val_loader),
            "val_concept_loss": avg_concept_loss,
            "val_concept_accuracy": val_concept_acc
        })
    
    return {
        'loss': avg_loss,
        'concept_loss': avg_concept_loss,
        'concept_acc': val_concept_acc
    }


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
    parser.add_argument("--weight_decay", type=float, default=0.1)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    
    # Loss Hyperparams
    parser.add_argument("--temperature", type=float, default=0.07)
    parser.add_argument("--lambda_divergence", type=float, default=0.1)
    parser.add_argument("--lambda_concept", type=float, default=1.0)
    parser.add_argument("--sigma", type=float, default=1.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    
    # Misc
    parser.add_argument("--output_dir", type=str, default="results/concept_aware")
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true", default=True)
    
    args = parser.parse_args()
    train(args)
