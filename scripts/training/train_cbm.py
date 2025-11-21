#!/usr/bin/env python3
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys
import os
import time
import math
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, desc=None, **kwargs):
        return iterable

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset_cbm import CBMDataset, create_stratified_splits, get_transforms
from src.models.cbm import GeoCBM
from src.utils.loss import CBMLoss, HaversineLoss

def denormalize(tensor):
    """
    Denormalize tensor image (C, H, W) to numpy array (H, W, C) in [0, 1].
    """
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    
    tensor = tensor.cpu() * std + mean
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_validation_samples(model, dataset, epoch, output_dir, device, num_samples=5):
    """
    Visualizes random validation samples with predictions (Attention map disabled for now).
    """
    model.eval()
    
    # Create output directory for this epoch
    epoch_dir = output_dir / f"epoch_{epoch}"
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Select random indices
    indices = torch.randperm(len(dataset))[:num_samples].tolist()
    
    haversine = HaversineLoss()
    
    for i, idx in enumerate(indices):
        image, concept_label, country_label, coords = dataset[idx]
        image_tensor = image.unsqueeze(0).to(device)
        
        with torch.no_grad():
            # Disabled return_attentions for now
            concept_logits, country_logits, coord_preds = model(image_tensor, return_attentions=False)
            
        # Process Output
        concept_probs = torch.softmax(concept_logits, dim=1)
        top3_probs, top3_indices = torch.topk(concept_probs, k=3)
        top3_concepts = [(dataset.idx_to_concept[idx.item()], prob.item()) for idx, prob in zip(top3_indices[0], top3_probs[0])]
        
        country_idx = country_logits.argmax(dim=1).item()
        pred_country = dataset.idx_to_country[country_idx]
        
        pred_lat, pred_lon = coord_preds[0].cpu().numpy()
        true_lat, true_lon = coords.numpy()
        dist_km = haversine(coord_preds.cpu(), coords.unsqueeze(0)).item()
        
        # Plot
        fig, axes = plt.subplots(1, 2, figsize=(12, 6))
        
        img_np = denormalize(image)
        axes[0].imshow(img_np)
        axes[0].axis('off')
        axes[0].set_title("Input Image")
        
        axes[1].axis('off')
        gt_concept = dataset.idx_to_concept[concept_label.item()]
        gt_country = dataset.idx_to_country[country_label.item()]
        
        info_text = f"GROUND TRUTH:\n"
        info_text += f"Concept: {gt_concept}\n"
        info_text += f"Country: {gt_country}\n"
        info_text += f"Coords: {true_lat:.4f}, {true_lon:.4f}\n\n"
        
        info_text += f"PREDICTIONS:\n"
        info_text += f"Coords: {pred_lat:.4f}, {pred_lon:.4f}\n"
        info_text += f"Error: {dist_km:.1f} km\n"
        info_text += f"Country: {pred_country} ({'✓' if pred_country == gt_country else '✗'})\n\n"
        
        info_text += f"Top 3 Concepts:\n"
        for concept, prob in top3_concepts:
            mark = "✓" if concept == gt_concept else ""
            info_text += f"- {concept}: {prob:.2f} {mark}\n"
            
        axes[1].text(0.05, 0.95, info_text, fontsize=12, verticalalignment='top', fontfamily='monospace')
        
        plt.tight_layout()
        plt.savefig(epoch_dir / f"val_sample_{i}.png")
        plt.close()

def train_one_epoch(model, loader, criterion, optimizer, device, epoch):
    model.train()
    backbone_requires_grad = any(p.requires_grad for p in model.backbone.parameters())
    if not backbone_requires_grad:
        model.backbone.eval()
    
    running_loss = 0.0
    running_concept_acc = 0.0
    running_country_acc = 0.0
    running_haversine = 0.0
    
    total = 0
    start_time = time.time()
    
    pbar = tqdm(loader, desc=f"Epoch {epoch} [Train]", miniters=5)
    
    for i, (images, concept_labels, country_labels, coords) in enumerate(pbar):
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        country_labels = country_labels.to(device)
        coords = coords.to(device)
        
        optimizer.zero_grad()
        
        concept_logits, country_logits, coord_preds = model(images)
        
        loss, loss_dict = criterion(
            concept_logits, concept_labels,
            country_logits, country_labels,
            coord_preds, coords
        )
        
        loss.backward()
        optimizer.step()
        
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_haversine += loss_dict['haversine_dist'].item() * bs
        
        _, concept_preds = concept_logits.max(1)
        running_concept_acc += concept_preds.eq(concept_labels).sum().item()
        
        _, country_preds = country_logits.max(1)
        running_country_acc += country_preds.eq(country_labels).sum().item()
        
        total += bs
        
        pbar.set_postfix({
            'loss': f"{loss.item():.4f}",
            'dist': f"{loss_dict['haversine_dist'].item():.1f}km"
        })
            
    metrics = {
        'loss': running_loss / total,
        'concept_acc': 100. * running_concept_acc / total,
        'country_acc': 100. * running_country_acc / total,
        'haversine_dist': running_haversine / total,
        'duration': time.time() - start_time
    }
    
    return metrics

@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    
    running_loss = 0.0
    running_concept_acc = 0.0
    running_country_acc = 0.0
    running_haversine = 0.0
    
    total = 0
    
    pbar = tqdm(loader, desc="Evaluating", miniters=5)
    
    for images, concept_labels, country_labels, coords in pbar:
        images = images.to(device)
        concept_labels = concept_labels.to(device)
        country_labels = country_labels.to(device)
        coords = coords.to(device)
        
        concept_logits, country_logits, coord_preds = model(images)
        
        loss, loss_dict = criterion(
            concept_logits, concept_labels,
            country_logits, country_labels,
            coord_preds, coords
        )
        
        bs = images.size(0)
        running_loss += loss.item() * bs
        running_haversine += loss_dict['haversine_dist'].item() * bs
        
        _, concept_preds = concept_logits.max(1)
        running_concept_acc += concept_preds.eq(concept_labels).sum().item()
        
        _, country_preds = country_logits.max(1)
        running_country_acc += country_preds.eq(country_labels).sum().item()
        
        total += bs
        
        pbar.set_postfix({
            'val_loss': f"{loss.item():.4f}",
            'val_dist': f"{loss_dict['haversine_dist'].item():.1f}km"
        })
        
    metrics = {
        'loss': running_loss / total,
        'concept_acc': 100. * running_concept_acc / total,
        'country_acc': 100. * running_country_acc / total,
        'haversine_dist': running_haversine / total
    }
    
    return metrics

def main():
    parser = argparse.ArgumentParser(description="Train Geolocation CBM")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to dataset CSV")
    parser.add_argument("--probe-path", type=str, required=True, help="Path to pretrained Concept Probe checkpoint")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size")
    parser.add_argument("--epochs", type=int, default=20, help="Number of epochs")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--output-dir", type=str, default="checkpoints/cbm", help="Directory to save models")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    
    # Loss weights
    parser.add_argument("--w-coords", type=float, default=1.0, help="Weight for Coordinate Loss")
    parser.add_argument("--w-country", type=float, default=1.0, help="Weight for Country Loss")
    parser.add_argument("--w-concepts", type=float, default=0.5, help="Weight for Concept Loss")
    
    args = parser.parse_args()
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Create visualization directory
    viz_dir = Path("visualizations/cbm")
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Create splits
    print("Creating stratified splits...")
    train_df, val_df, test_df = create_stratified_splits(args.csv_path)
    print(f"Split sizes: Train={len(train_df)}, Val={len(val_df)}, Test={len(test_df)}")
    
    # Get transforms (using probe's backbone default)
    transform = get_transforms() 
    
    # Create datasets
    train_dataset = CBMDataset(train_df, transform=transform)
    val_dataset = CBMDataset(val_df, transform=transform)
    test_dataset = CBMDataset(test_df, transform=transform)
    
    num_concepts = len(train_dataset.concepts)
    num_countries = len(train_dataset.countries)
    print(f"Concepts: {num_concepts}, Countries: {num_countries}")
    
    # Create dataloaders
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)
    
    # Initialize model
    model = GeoCBM(
        num_concepts=num_concepts,
        num_countries=num_countries,
        model_name="geolocal/StreetCLIP",
        freeze_backbone=True
    )

    print(f'Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    print(f'Total parameters: {sum(p.numel() for p in model.parameters())}')
    print(f'Backbone parameters: {sum(p.numel() for p in model.backbone.parameters())}')
    print(f'Concept head parameters: {sum(p.numel() for p in model.concept_head.parameters())}')
    print(f'Country head parameters: {sum(p.numel() for p in model.country_head.parameters())}')
    print(f'Coordinate head parameters: {sum(p.numel() for p in model.coord_head.parameters())}')
    
    # Load pretrained probe weights (Warm Start)
    if os.path.exists(args.probe_path):
        model.load_probe_weights(args.probe_path)
    else:
        print(f"Warning: Probe checkpoint not found at {args.probe_path}. Training from scratch.")
        
    model = model.to(device)
    
    # Loss and Optimizer
    criterion = CBMLoss(
        lambda_coords=args.w_coords,
        lambda_country=args.w_country,
        lambda_concepts=args.w_concepts
    )
    
    # Optimize all heads: concept_head, country_head, coord_head
    params_to_optimize = [
        {'params': model.concept_head.parameters()},
        {'params': model.country_head.parameters()},
        {'params': model.coord_head.parameters()}
    ]
    
    optimizer = optim.Adam(params_to_optimize, lr=args.lr)
    
    # Scheduler (optional, but good for convergence)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=3)
    
    # Training Loop
    best_val_dist = float('inf')
    
    print("Starting CBM training...")
    for epoch in range(1, args.epochs + 1):
        train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, device, epoch)
        val_metrics = evaluate(model, val_loader, criterion, device)
        
        # Update scheduler based on validation distance
        scheduler.step(val_metrics['haversine_dist'])
        
        print(f"Epoch [{epoch}/{args.epochs}] ({train_metrics['duration']:.1f}s)")
        print(f"  Train | Loss: {train_metrics['loss']:.4f} | Dist: {train_metrics['haversine_dist']:.1f}km | Concept: {train_metrics['concept_acc']:.1f}% | Country: {train_metrics['country_acc']:.1f}%")
        print(f"  Val   | Loss: {val_metrics['loss']:.4f} | Dist: {val_metrics['haversine_dist']:.1f}km | Concept: {val_metrics['concept_acc']:.1f}% | Country: {val_metrics['country_acc']:.1f}%")
        
        # Save validation visualizations
        if epoch % 1 == 0: # Save every epoch
            visualize_validation_samples(model, val_dataset, epoch, viz_dir, device)
            
        # Save best model (using Haversine Distance as primary metric)
        if val_metrics['haversine_dist'] < best_val_dist:
            best_val_dist = val_metrics['haversine_dist']
            save_path = output_dir / "best_cbm_model.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_dist': best_val_dist,
                'val_metrics': val_metrics,
                'concepts': train_dataset.concepts,
                'countries': train_dataset.countries
            }, save_path)
            print(f"  Saved best model to {save_path}")
            
    print("Training complete.")
    print(f"Best Validation Distance: {best_val_dist:.1f} km")

if __name__ == "__main__":
    main()
