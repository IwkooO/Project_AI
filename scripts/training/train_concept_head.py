#!/usr/bin/env python3
"""
Train Concept Head with PU Learning (Phase 1).

This script trains the concept head on cached StreetCLIP embeddings using:
    - nnPU loss for positive-unlabeled learning
    - Parameter drift regularization (keeps u_k close to v_k)
    - Optional behavioral drift regularization

Supports two modes:
    - global (default): Uses pooled image embeddings - simple and effective
    - spatial: Uses attention over patches - provides interpretability

The StreetCLIP backbone is frozen; only the concept head is trained.

Usage:
    python scripts/training/train_concept_head.py \
        --train-csv data/.../splits/dataset_train.csv \
        --val-csv data/.../splits/dataset_val.csv \
        --cached-embeddings-dir data/.../cached_embeddings \
        --concept-data-dir data/.../concept_data \
        --output-dir checkpoints/concept_head \
        --mode global \
        --epochs 50 \
        --batch-size 256 \
        --lr 1e-3
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.optim import AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR
import numpy as np
from tqdm import tqdm
import json
from datetime import datetime

from src.data.dataset_concept import ConceptDataset, collate_fn
from src.models.concept_head import ConceptHead
from src.models.losses import ConceptLoss, compute_clip_sample_weights
from src.utils.concept_utils import (
    load_all_concept_data,
    compute_accuracy,
    save_checkpoint,
    load_checkpoint,
    visualize_epoch_samples
)


def train_epoch(
    model: ConceptHead,
    dataloader: DataLoader,
    criterion: ConceptLoss,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int,
    mode: str = "global",
    use_clip_weights: bool = False,
    zero_shot_scores: np.ndarray = None
) -> dict:
    """
    Train for one epoch.
    
    Args:
        model: Concept head model
        dataloader: Training data loader
        criterion: Loss function
        optimizer: Optimizer
        device: Device to use
        epoch: Current epoch number
        mode: "global" or "spatial"
        use_clip_weights: Whether to use CLIP prior weighting
        zero_shot_scores: Zero-shot scores for CLIP weighting
    
    Returns:
        Dictionary with training metrics
    """
    model.train()
    
    total_loss = 0.0
    total_nnpu = 0.0
    total_drift = 0.0
    total_behav = 0.0
    num_batches = 0
    
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Train]")
    
    for batch_idx, (pooled, patch_tokens, labels, coords) in enumerate(pbar):
        pooled = pooled.to(device)
        labels = labels.to(device)
        
        if mode == "spatial" and patch_tokens is not None:
            patch_tokens = patch_tokens.to(device)
        else:
            patch_tokens = None
        
        # Forward pass
        probs, _ = model(pooled, patch_tokens)
        
        # Compute sample weights for CLIP prior weighting
        sample_weights = None
        if use_clip_weights and zero_shot_scores is not None:
            start_idx = batch_idx * dataloader.batch_size
            end_idx = min(start_idx + len(labels), len(zero_shot_scores))
            batch_scores = torch.tensor(
                zero_shot_scores[start_idx:end_idx],
                device=device,
                dtype=torch.float32
            )
            sample_weights = compute_clip_sample_weights(batch_scores, labels)
        
        # Compute loss
        loss, metrics = criterion(
            probs, labels, model,
            zero_shot_scores=None,
            sample_weights=sample_weights
        )
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        
        optimizer.step()
        
        # Track metrics
        total_loss += metrics['total_loss']
        total_nnpu += metrics['nnpu_loss']
        total_drift += metrics['drift_loss']
        total_behav += metrics['behav_loss']
        num_batches += 1
        
        # Collect predictions for metrics
        all_probs.append(probs.detach().cpu())
        all_labels.append(labels.detach().cpu())
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f"{metrics['total_loss']:.4f}",
            'nnpu': f"{metrics['nnpu_loss']:.4f}",
            'drift': f"{metrics['drift_loss']:.4f}"
        })
    
    # Compute epoch metrics
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = compute_accuracy(all_probs, all_labels)
    
    return {
        'loss': total_loss / num_batches,
        'nnpu_loss': total_nnpu / num_batches,
        'drift_loss': total_drift / num_batches,
        'behav_loss': total_behav / num_batches,
        'accuracy': accuracy
    }


@torch.no_grad()
def validate(
    model: ConceptHead,
    dataloader: DataLoader,
    criterion: ConceptLoss,
    device: torch.device,
    epoch: int,
    mode: str = "global"
) -> dict:
    """
    Validate model.
    
    Args:
        model: Concept head model
        dataloader: Validation data loader
        criterion: Loss function
        device: Device to use
        epoch: Current epoch number
        mode: "global" or "spatial"
    
    Returns:
        Dictionary with validation metrics
    """
    model.eval()
    
    total_loss = 0.0
    num_batches = 0
    
    all_probs = []
    all_labels = []
    
    pbar = tqdm(dataloader, desc=f"Epoch {epoch} [Val]")
    
    for pooled, patch_tokens, labels, coords in pbar:
        pooled = pooled.to(device)
        labels = labels.to(device)
        
        if mode == "spatial" and patch_tokens is not None:
            patch_tokens = patch_tokens.to(device)
        else:
            patch_tokens = None
        
        # Forward pass
        probs, _ = model(pooled, patch_tokens)
        
        # Compute loss
        loss, metrics = criterion(probs, labels, model)
        
        total_loss += metrics['total_loss']
        num_batches += 1
        
        all_probs.append(probs.cpu())
        all_labels.append(labels.cpu())
        
        pbar.set_postfix({'loss': f"{metrics['total_loss']:.4f}"})
    
    # Compute metrics
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    accuracy = compute_accuracy(all_probs, all_labels)
    
    return {
        'loss': total_loss / num_batches,
        'accuracy': accuracy
    }


def main():
    parser = argparse.ArgumentParser(description="Train Concept Head (Phase 1)")
    
    # Data paths
    parser.add_argument("--train-csv", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to validation CSV")
    parser.add_argument("--cached-embeddings-dir", type=str, required=True, help="Directory with cached embeddings")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Directory with concept data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for checkpoints")
    
    # Model parameters
    parser.add_argument("--mode", type=str, default="global", choices=["global", "spatial"],
                        help="Model mode: 'global' (default, uses pooled embeddings) or 'spatial' (uses attention)")
    parser.add_argument("--temperature", type=float, default=1.0, help="Attention temperature (spatial mode only)")
    parser.add_argument("--trainable-attention", action="store_true", help="Make attention queries trainable (spatial mode only)")
    
    # Training parameters
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight-decay", type=float, default=0.01, help="Weight decay")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data loading workers")
    
    # Loss parameters
    parser.add_argument("--lambda-drift", type=float, default=0.01, help="Weight for parameter drift loss")
    parser.add_argument("--lambda-behav", type=float, default=0.0, help="Weight for behavioral drift loss")
    parser.add_argument("--nnpu-gamma", type=float, default=1.0, help="Weight for nnPU correction term")
    parser.add_argument("--use-clip-weights", action="store_true", help="Use CLIP prior weighting")
    
    # Visualization
    parser.add_argument("--num-vis-samples", type=int, default=5, help="Number of samples to visualize per epoch")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test CSV for visualization (optional)")
    
    # Other
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--resume", type=str, default=None, help="Path to checkpoint to resume from")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    
    args = parser.parse_args()
    
    # Set random seed
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save config
    config_path = output_dir / "config.json"
    with open(config_path, 'w') as f:
        json.dump(vars(args), f, indent=2)
    print(f"Saved config to {config_path}")
    
    # Load concept data
    print("\nLoading concept data...")
    concept_data = load_all_concept_data(args.concept_data_dir, split="train")
    
    vocab = concept_data['vocabulary']
    embeddings = concept_data['embeddings']
    priors = concept_data['priors']['priors']
    zero_shot_scores = concept_data['zero_shot_scores']
    
    K = vocab['num_concepts']
    print(f"Loaded {K} concepts")
    print(f"Class priors: mean={priors.mean():.4f}, min={priors.min():.4f}, max={priors.max():.4f}")
    
    # Create datasets
    print(f"\nCreating datasets (mode: {args.mode})...")
    load_patch_tokens = (args.mode == "spatial")
    
    train_dataset = ConceptDataset(
        csv_path=args.train_csv,
        cached_embeddings_dir=args.cached_embeddings_dir,
        concept_vocab_path=f"{args.concept_data_dir}/concept_vocabulary.json",
        split="train",
        load_patch_tokens=load_patch_tokens
    )
    
    val_dataset = ConceptDataset(
        csv_path=args.val_csv,
        cached_embeddings_dir=args.cached_embeddings_dir,
        concept_vocab_path=f"{args.concept_data_dir}/concept_vocabulary.json",
        split="val",
        load_patch_tokens=load_patch_tokens
    )
    
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    
    # Create test dataset if provided (for visualization only)
    test_dataset = None
    if args.test_csv:
        test_dataset = ConceptDataset(
            csv_path=args.test_csv,
            cached_embeddings_dir=args.cached_embeddings_dir,
            concept_vocab_path=f"{args.concept_data_dir}/concept_vocabulary.json",
            split="test",
            load_patch_tokens=load_patch_tokens
        )
        print(f"Test samples: {len(test_dataset)}")
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
        collate_fn=collate_fn
    )
    
    # Create model
    print(f"\nCreating model (mode: {args.mode})...")
    device = torch.device(args.device)
    
    # Auto-detect embedding dimension from concept embeddings
    embedding_dim = embeddings.shape[1]
    print(f"Detected embedding dimension: {embedding_dim}")
    
    # Verify dataset embeddings match concept embeddings dimension
    sample_pooled = train_dataset[0][0]  # Get first pooled embedding
    if sample_pooled.shape[0] != embedding_dim:
        raise ValueError(
            f"Dimension mismatch: Concept embeddings ({embedding_dim}D) != "
            f"Pooled embeddings ({sample_pooled.shape[0]}D). "
            f"Please ensure precomputation used the same model."
        )
    print(f"Verified: Dataset embeddings match concept embeddings ({embedding_dim}D)")
    
    model = ConceptHead(
        concept_embeddings=embeddings,
        mode=args.mode,
        hidden_dim=embedding_dim,  # Use detected dimension
        temperature=args.temperature,
        trainable_attention=args.trainable_attention
    ).to(device)
    
    print(f"Model parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad):,}")
    
    # Create loss function
    priors_tensor = torch.tensor(priors, dtype=torch.float32, device=device)
    criterion = ConceptLoss(
        priors=priors_tensor,
        lambda_drift=args.lambda_drift,
        lambda_behav=args.lambda_behav,
        nnpu_gamma=args.nnpu_gamma
    )
    
    # Create optimizer and scheduler
    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs, eta_min=1e-6)
    
    # Resume from checkpoint if specified
    start_epoch = 0
    best_val_loss = float('inf')
    
    if args.resume:
        print(f"\nResuming from {args.resume}")
        start_epoch, _ = load_checkpoint(model, optimizer, args.resume, device)
        start_epoch += 1
    
    # Training loop
    print("\n" + "="*60)
    print(f"Starting training (mode: {args.mode})...")
    print("="*60)
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    # Create visualization directory
    vis_dir = output_dir / "visualizations"
    vis_dir.mkdir(exist_ok=True)
    
    # Get idx_to_concept mapping
    idx_to_concept = vocab['idx_to_concept']
    
    for epoch in range(start_epoch, args.epochs):
        # Train
        train_metrics = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch,
            mode=args.mode,
            use_clip_weights=args.use_clip_weights,
            zero_shot_scores=zero_shot_scores
        )
        
        # Validate
        val_metrics = validate(model, val_loader, criterion, device, epoch, mode=args.mode)
        
        # Update scheduler
        scheduler.step()
        
        # Log metrics
        print(f"\nEpoch {epoch}:")
        print(f"  Train - Loss: {train_metrics['loss']:.4f}, Acc: {train_metrics['accuracy']:.4f}")
        print(f"  Val   - Loss: {val_metrics['loss']:.4f}, Acc: {val_metrics['accuracy']:.4f}")
        print(f"  LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Track history
        history['train_loss'].append(train_metrics['loss'])
        history['train_acc'].append(train_metrics['accuracy'])
        history['val_loss'].append(val_metrics['loss'])
        history['val_acc'].append(val_metrics['accuracy'])
        
        # Visualize samples at end of epoch
        if args.num_vis_samples > 0:
            print(f"  Generating visualizations (mode: {args.mode})...")
            visualize_epoch_samples(
                model=model,
                dataset=val_dataset,
                idx_to_concept=idx_to_concept,
                output_dir=vis_dir,
                epoch=epoch,
                split="val",
                num_samples=args.num_vis_samples,
                device=device,
                mode=args.mode
            )
            
            if test_dataset is not None:
                visualize_epoch_samples(
                    model=model,
                    dataset=test_dataset,
                    idx_to_concept=idx_to_concept,
                    output_dir=vis_dir,
                    epoch=epoch,
                    split="test",
                    num_samples=args.num_vis_samples,
                    device=device,
                    mode=args.mode
                )
        
        # Save checkpoint
        is_best = val_metrics['loss'] < best_val_loss
        if is_best:
            best_val_loss = val_metrics['loss']
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'], val_metrics,
                str(output_dir / "best_concept_head.pth")
            )
        
        # Save periodic checkpoint
        if (epoch + 1) % 10 == 0:
            save_checkpoint(
                model, optimizer, epoch, val_metrics['loss'], val_metrics,
                str(output_dir / f"concept_head_epoch_{epoch}.pth")
            )
    
    # Save final model
    save_checkpoint(
        model, optimizer, args.epochs - 1, val_metrics['loss'], val_metrics,
        str(output_dir / "final_concept_head.pth")
    )
    
    # Save training history
    history_path = output_dir / "training_history.json"
    with open(history_path, 'w') as f:
        json.dump(history, f, indent=2)
    print(f"\nSaved training history to {history_path}")
    
    print("\n" + "="*60)
    print("Training complete!")
    print(f"Mode: {args.mode}")
    print(f"Best validation loss: {best_val_loss:.4f}")
    print(f"Checkpoints saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
