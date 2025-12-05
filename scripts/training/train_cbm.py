#!/usr/bin/env python3
"""
Train Concept Bottleneck Model for Geolocation.

Phase 1: Train Concept Head (predict concept from image)
Phase 2: Train Geo Head (predict location from image + concept)
Phase 3: Joint Fine-Tuning (optimize all heads)

Usage:
    # Phase 1
    python scripts/training/train_cbm.py --phase 1 ...

    # Phase 2
    python scripts/training/train_cbm.py --phase 2 --resume-checkpoint checkpoints/best_phase1.pt ...
    
    # Phase 3
    python scripts/training/train_cbm.py --phase 3 --resume-checkpoint checkpoints/best_phase2.pt ...
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import sys
import math
import matplotlib.pyplot as plt
import s2sphere
from datetime import datetime
import torch.nn.functional as F

CONTRASTIVE_TEMPERATURE = 0.07
CONTRASTIVE_WEIGHT = 0.15

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.cbm import CBM
from src.data.dataset_concept import ConceptDataset, collate_fn

# Check for Cartopy (optional, for map plots)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


def compute_concept_weights(dataset, num_concepts, device):
    """Compute inverse frequency weights for concept loss using Pandas."""
    print("Computing concept class weights...")
    
    # Get concept names directly from dataframe
    concept_names = dataset.df[dataset.concept_col]
    
    # Count occurrences
    counts_dict = concept_names.value_counts().to_dict()
    
    # Map to indices
    counts = torch.zeros(num_concepts)
    for name, count in counts_dict.items():
        if name in dataset.concept_to_idx:
            idx = dataset.concept_to_idx[name]
            counts[idx] = count
            
    # Smooth inverse frequency: 1 / sqrt(count)
    weights = 1.0 / torch.sqrt(counts + 1.0)
    weights = weights / weights.mean() # Normalize
    return weights.to(device)


def visualize_predictions(model, dataset, device, epoch, output_dir, phase, run_id=None):
    """
    Visualize predictions for a few validation samples.
    Phase 1: Image + Top 5 Concepts
    Phase 2/3: Image + Top 5 Concepts + Predicted Location vs GT
    Since we don't have raw images loaded in the dataset (only embeddings),
    we will just plot text descriptions or placeholders if images aren't available.
    Ideally, we'd load the image from path if available.
    
    Args:
        run_id: Unique identifier for this training run (used for separate visualization folders)
    """
    model.eval()
    
    # Select 5 random indices
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    fig_h = 5 if phase == 1 else 7
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, fig_h * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, idx in enumerate(indices):
        # Load sample
        pooled, _, c_label, coords, cell_label, offset = dataset[idx]
        
        # Forward pass
        with torch.no_grad():
            pooled_dev = pooled.unsqueeze(0).to(device)
            c_logits, cell_logits, pred_offsets, _ = model(pooled_dev)
            c_probs = torch.softmax(c_logits, dim=1)
        
        # Top 5 Concepts
        top5_prob, top5_idx = torch.topk(c_probs[0], 5)
        top5_concepts = [dataset.get_concept_name(idx.item()) for idx in top5_idx]
        gt_concept = dataset.get_concept_name(c_label.item())
        
        # Get Image Path
        img_path = dataset.df.iloc[idx]['image_path']
        img_ax = axes[i, 0]
        bar_ax = axes[i, 1]
        
        # Try to load image
        try:
            img = plt.imread(img_path)
            img_ax.imshow(img)
        except Exception:
            img_ax.text(0.5, 0.5, "Image not found", ha='center')
            
        img_ax.axis('off')
        img_ax.set_title(f"GT Concept: {gt_concept}", loc='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Bar chart for top-5 concepts with highlighting for correct predictions
        y_pos = np.arange(len(top5_concepts))
        probs_np = top5_prob.cpu().numpy()
        
        # Color bars: green if correct, steelblue if incorrect
        colors = ['#2ecc71' if concept == gt_concept else 'steelblue' for concept in top5_concepts]
        bar_ax.barh(y_pos, probs_np, color=colors)
        
        # Add checkmark or highlight for correct predictions
        for j, (concept, prob) in enumerate(zip(top5_concepts, probs_np)):
            if concept == gt_concept:
                # Add a checkmark or highlight
                bar_ax.text(prob + 0.02, j, '✓', fontsize=14, color='#27ae60', fontweight='bold', va='center')
        
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels(top5_concepts)
        bar_ax.invert_yaxis()  # Highest prob on top
        bar_ax.set_xlim(0, 1)
        bar_ax.set_xlabel("Probability")
        bar_ax.set_title("Top-5 concept probs", loc='left')
        
        # Geo Info (Phase 2+)
        info_lines = []
        if phase >= 2:
            pred_cell_idx = cell_logits.argmax(dim=1).item()
            
            if dataset.idx_to_cell:
                token = dataset.idx_to_cell[pred_cell_idx]
                cell = s2sphere.CellId.from_token(token)
                center = cell.to_lat_lng()
                center_lat = center.lat().degrees
                center_lng = center.lng().degrees
                
                # pred_offsets: [1, M, 2] -> select [0, pred_cell_idx, :]
                d_lat_rad = pred_offsets[0, pred_cell_idx, 0].item()
                d_lng_rad = pred_offsets[0, pred_cell_idx, 1].item()
                
                pred_lat = center_lat + math.degrees(d_lat_rad)
                pred_lng = center_lng + math.degrees(d_lng_rad)
                
                gt_lat = math.degrees(coords[0].item())
                gt_lng = math.degrees(coords[1].item())
                
                error_km = 6371 * 2 * math.asin(math.sqrt(
                    math.sin(math.radians(pred_lat - gt_lat)/2)**2 +
                    math.cos(math.radians(gt_lat)) * math.cos(math.radians(pred_lat)) *
                    math.sin(math.radians(pred_lng - gt_lng)/2)**2
                ))
                
                info_lines.append(f"Loc error: {error_km:.1f} km")
                info_lines.append(f"Pred: ({pred_lat:.4f}, {pred_lng:.4f})")
                info_lines.append(f"GT:   ({gt_lat:.4f}, {gt_lng:.4f})")
        
        if info_lines:
            bar_ax.text(1.02, 0.5, "\n".join(info_lines), transform=bar_ax.transAxes,
                        va='center', fontsize=9, bbox=dict(facecolor='white', alpha=0.8))
        
    plt.tight_layout()
    
    # Create visualization directory with run_id subfolder if provided
    viz_dir = output_dir / "visualizations"
    if run_id:
        viz_dir = viz_dir / run_id
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(viz_dir / f"epoch_{epoch}_phase{phase}.png", dpi=150, bbox_inches='tight')
    plt.close()


def supcon_loss(features: torch.Tensor, labels: torch.Tensor, temperature: float = CONTRASTIVE_TEMPERATURE) -> torch.Tensor:
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)

    features = F.normalize(features, dim=1)
    sim_matrix = torch.matmul(features, features.T) / temperature
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float().to(features.device)
    mask.fill_diagonal_(0)

    exp_sim = torch.exp(sim_matrix) * (1.0 - torch.eye(features.size(0), device=features.device))
    log_prob = sim_matrix - torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)

    positive_mask_sum = mask.sum(dim=1)
    valid = positive_mask_sum > 0
    if not valid.any():
        return torch.tensor(0.0, device=features.device)

    mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(dim=1) / (positive_mask_sum[valid] + 1e-8)
    loss = -mean_log_prob_pos.mean()
    return loss


def train_epoch(model, loader, optimizer, device, phase, concept_criterion, cell_criterion, offset_criterion):
    model.train()
    
    # Freeze/Unfreeze based on phase
    if phase == 1:
        for p in model.concept_head.parameters(): p.requires_grad = True
        for p in model.geo_head.parameters(): p.requires_grad = False
    elif phase == 2:
        for p in model.concept_head.parameters(): p.requires_grad = False
        model.concept_head.eval()
        for p in model.geo_head.parameters(): p.requires_grad = True
    elif phase == 3: # Joint
        for p in model.concept_head.parameters(): p.requires_grad = True
        for p in model.geo_head.parameters(): p.requires_grad = True
        
    total_loss = 0
    correct_c = 0
    correct_cell = 0
    total_samples = 0
    total_contrastive_loss = 0.0
    contrastive_samples = 0
    
    pbar = tqdm(loader, desc="Train")
    for pooled, _, c_labels, _, cell_labels, offsets in pbar:
        pooled = pooled.to(device)
        c_labels = c_labels.to(device)
        cell_labels = cell_labels.to(device)
        offsets = offsets.to(device)
        
        optimizer.zero_grad()
        
        # Forward
        c_logits, cell_logits, pred_offsets, c_feats = model(pooled)
        
        loss = 0
        contrastive_value = None
        
        # Phase 1: Concept Loss
        if phase == 1:
            c_loss = concept_criterion(c_logits, c_labels)
            loss = c_loss
            contrastive_value = supcon_loss(c_feats, c_labels)
            loss = loss + CONTRASTIVE_WEIGHT * contrastive_value
            
            # Metrics
            preds = c_logits.argmax(dim=1)
            correct_c += (preds == c_labels).sum().item()
            
            pbar.set_postfix({"C_Loss": f"{c_loss.item():.4f}"})
            
        # Phase 2: Geo Loss
        elif phase == 2:
            mask = cell_labels != -1
            if mask.sum() > 0:
                cell_loss = cell_criterion(cell_logits[mask], cell_labels[mask])
                
                valid_offsets = pred_offsets[mask]
                valid_targets = cell_labels[mask]
                selected_offsets = valid_offsets[torch.arange(len(valid_targets)), valid_targets]
                
                off_loss = offset_criterion(selected_offsets, offsets[mask])
                
                loss = cell_loss + 100.0 * off_loss
                
                cell_preds = cell_logits.argmax(dim=1)
                correct_cell += (cell_preds[mask] == cell_labels[mask]).sum().item()
                
                pbar.set_postfix({
                    "Cell_L": f"{cell_loss.item():.4f}", 
                    "Off_L": f"{off_loss.item():.4f}"
                })
            else:
                continue

        # Phase 3: Joint Loss
        elif phase == 3:
            # Concept Loss
            c_loss = concept_criterion(c_logits, c_labels)
            
            contrastive_value = supcon_loss(c_feats, c_labels)
            loss = c_loss + CONTRASTIVE_WEIGHT * contrastive_value
            
            # Geo Loss
            mask = cell_labels != -1
            if mask.sum() > 0:
                cell_loss = cell_criterion(cell_logits[mask], cell_labels[mask])
                
                valid_offsets = pred_offsets[mask]
                valid_targets = cell_labels[mask]
                selected_offsets = valid_offsets[torch.arange(len(valid_targets)), valid_targets]
                off_loss = offset_criterion(selected_offsets, offsets[mask])
                
                geo_loss = cell_loss + 100.0 * off_loss
            else:
                geo_loss = 0.0
            
            # Combined
            loss = c_loss + CONTRASTIVE_WEIGHT * contrastive_value + 0.5 * geo_loss # Weight geo loss less initially? Or 1.0?
            
            # Metrics
            preds = c_logits.argmax(dim=1)
            correct_c += (preds == c_labels).sum().item()
            if mask.sum() > 0:
                cell_preds = cell_logits.argmax(dim=1)
                correct_cell += (cell_preds[mask] == cell_labels[mask]).sum().item()
                
            pbar.set_postfix({
                "C_Loss": f"{c_loss.item():.2f}",
                "G_Loss": f"{geo_loss:.2f}" if isinstance(geo_loss, torch.Tensor) else "0.0"
            })
        
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * pooled.size(0)
        if contrastive_value is not None:
            total_contrastive_loss += contrastive_value.item() * pooled.size(0)
            contrastive_samples += pooled.size(0)
        total_samples += pooled.size(0)
        
    avg_contrastive = total_contrastive_loss / (contrastive_samples + 1e-8) if contrastive_samples > 0 else 0.0
    return total_loss / total_samples, correct_c / total_samples, correct_cell / total_samples, avg_contrastive


@torch.no_grad()
def eval_epoch(model, loader, device, phase, concept_criterion, cell_criterion, offset_criterion):
    model.eval()
    total_loss = 0
    correct_c = 0
    correct_cell = 0
    total_samples = 0
    
    for pooled, _, c_labels, _, cell_labels, offsets in tqdm(loader, desc="Eval"):
        pooled = pooled.to(device)
        c_labels = c_labels.to(device)
        cell_labels = cell_labels.to(device)
        offsets = offsets.to(device)
        
        c_logits, cell_logits, pred_offsets, _ = model(pooled)
        
        loss = 0
        if phase == 1:
            loss = concept_criterion(c_logits, c_labels)
            correct_c += (c_logits.argmax(dim=1) == c_labels).sum().item()
        else:
            # For phase 2 and 3 evaluation, we track both but loss depends on phase
            # We always compute metrics
            
            # Concept Acc
            correct_c += (c_logits.argmax(dim=1) == c_labels).sum().item()
            
            mask = cell_labels != -1
            if mask.sum() > 0:
                cell_loss = cell_criterion(cell_logits[mask], cell_labels[mask])
                
                valid_offsets = pred_offsets[mask]
                valid_targets = cell_labels[mask]
                selected_offsets = valid_offsets[torch.arange(len(valid_targets)), valid_targets]
                off_loss = offset_criterion(selected_offsets, offsets[mask])
                
                geo_loss = cell_loss + 100.0 * off_loss
                correct_cell += (cell_logits.argmax(dim=1)[mask] == cell_labels[mask]).sum().item()
            else:
                geo_loss = torch.tensor(0.0).to(device)
                
            if phase == 2:
                loss = geo_loss
            elif phase == 3:
                c_loss = concept_criterion(c_logits, c_labels)
                loss = c_loss + 0.5 * geo_loss
        
        total_loss += loss.item() * pooled.size(0)
        total_samples += pooled.size(0)
        
    return total_loss / total_samples, correct_c / (total_samples+1e-8), correct_cell / (total_samples+1e-8)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--cached-dir", required=True)
    parser.add_argument("--concept-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="cbm_concept_bottleneck")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    
    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("W&B logging requested but `wandb` is not installed; skipping W&B.")

    wandb_run = None
    if use_wandb:
        wandb_config = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        wandb_config["run_id"] = run_id
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"phase{args.phase}_{run_id}",
            config=wandb_config,
            reinit=True,
        )
    concept_vocab = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab = Path(args.concept_data_dir) / "s2_cells.json"
    
    # Load Datasets
    print("Loading datasets...")
    train_ds = ConceptDataset(args.train_csv, args.cached_dir, concept_vocab, str(s2_vocab), split="train")
    val_ds = ConceptDataset(args.val_csv, args.cached_dir, concept_vocab, str(s2_vocab), split="val")
    
    if wandb_run:
        wandb_run.config.update({
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "num_concepts": train_ds.num_concepts,
            "num_cells": train_ds.num_cells,
            "run_id": run_id,
        }, allow_val_change=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Initialize Model
    print("Initializing CBM...")
    model = CBM(
        num_concepts=train_ds.num_concepts,
        num_cells=train_ds.num_cells,
        input_dim=768
    ).to(device)
    
    # Resume / Load Pretrained
    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Criterions
    # Prepare weighted CE for concepts in phases 1 and 3
    weights = compute_concept_weights(train_ds, train_ds.num_concepts, device)
    concept_criterion = nn.CrossEntropyLoss(weight=weights,label_smoothing=0.15)
    
    cell_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.MSELoss()
    
    if args.phase == 1:
        print("Phase 1: Concept Training")
    elif args.phase == 2:
        print("Phase 2: Geo Training (Concept Head Frozen)")
    elif args.phase == 3:
        print("Phase 3: Joint Training")
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr,weight_decay=0.01)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    
    best_metric = 0.0
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs}")
            
            train_loss, train_acc_c, train_acc_cell, train_contrastive_loss = train_epoch(
                model, train_loader, optimizer, device, args.phase,
                concept_criterion, cell_criterion, offset_criterion
            )
            
            val_loss, val_acc_c, val_acc_cell = eval_epoch(
                model, val_loader, device, args.phase,
                concept_criterion, cell_criterion, offset_criterion
            )
            
            scheduler.step()
            
            # Log
            if args.phase == 1:
                print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc_c:.4f}")
                print(f"Val   Loss: {val_loss:.4f}   | Acc: {val_acc_c:.4f}")
                metric = val_acc_c
            elif args.phase == 2:
                print(f"Train Loss: {train_loss:.4f} | Cell Acc: {train_acc_cell:.4f}")
                print(f"Val   Loss: {val_loss:.4f}   | Cell Acc: {val_acc_cell:.4f}")
                metric = val_acc_cell
            else: # Phase 3
                print(f"Train Loss: {train_loss:.4f} | C Acc: {train_acc_c:.4f} | Cell Acc: {train_acc_cell:.4f}")
                print(f"Val   Loss: {val_loss:.4f}   | C Acc: {val_acc_c:.4f} | Cell Acc: {val_acc_cell:.4f}")
                metric = val_acc_cell + val_acc_c # Combined metric
                
            if wandb_run:
                log_dict = {
                    "phase": args.phase,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_concept_acc": train_acc_c,
                    "val_concept_acc": val_acc_c,
                    "train_cell_acc": train_acc_cell,
                    "val_cell_acc": val_acc_cell,
                    "train_contrastive_loss": train_contrastive_loss,
                }
                wandb_run.log(log_dict, step=epoch + 1)

            # Visualization
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Generating visualizations...")
                visualize_predictions(model, val_ds, device, epoch, output_dir, args.phase, run_id=run_id)
                
            # Save Best
            if metric > best_metric:
                best_metric = metric
                save_path = output_dir / f"best_phase{args.phase}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metric': best_metric,
                }, save_path)
                print(f"Saved best model to {save_path}")
    finally:
        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    main()
