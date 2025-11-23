#!/usr/bin/env python3
"""
Train a Sparse Autoencoder (SAE) on StreetCLIP embeddings.
Phase 1: Unsupervised Dictionary Learning.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim import Adam
from transformers import AutoModel
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.sae import TopKSAE
from src.data.dataset_sae import SAEDataset

def main():
    parser = argparse.ArgumentParser(description="Train Top-K SAE on StreetCLIP embeddings")
    parser.add_argument("--csv-path", type=str, required=True, help="Path to training dataset CSV")
    parser.add_argument("--val-csv-path", type=str, default=None, help="Path to validation dataset CSV")
    parser.add_argument("--output-dir", type=str, default="checkpoints/sae", help="Directory to save checkpoints")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="Backbone model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--lr", type=float, default=1e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50, help="Number of epochs")
    parser.add_argument("--expansion", type=int, default=8, help="Expansion factor for dictionary size")
    parser.add_argument("--k", type=int, default=32, help="Top-K sparsity")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of dataloader workers")
    parser.add_argument("--resample-freq", type=int, default=5, help="Resample dead neurons every N epochs")
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Backbone (Frozen)
    print(f"Loading backbone: {args.model_name}")
    backbone = AutoModel.from_pretrained(args.model_name)
    backbone.to(device)
    backbone.eval()
    for param in backbone.parameters():
        param.requires_grad = False
        
    # Determine embedding dimension
    if hasattr(backbone.config, "projection_dim"):
        input_dim = backbone.config.projection_dim
    elif hasattr(backbone.config, "hidden_size"):
        input_dim = backbone.config.hidden_size
    else:
        input_dim = 768
    print(f"Embedding dimension: {input_dim}")

    # 2. Initialize SAE
    print(f"Initializing SAE (Expansion: {args.expansion}x, K: {args.k})")
    sae = TopKSAE(input_dim=input_dim, expansion_factor=args.expansion, k=args.k)
    sae.to(device)
    
    # Optimizer
    optimizer = Adam(sae.parameters(), lr=args.lr)
    
    # Dataset
    print(f"Loading training dataset from {args.csv_path}")
    train_dataset = SAEDataset(args.csv_path, model_name=args.model_name, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                           num_workers=args.num_workers, pin_memory=True)
    
    val_loader = None
    if args.val_csv_path:
        print(f"Loading validation dataset from {args.val_csv_path}")
        val_dataset = SAEDataset(args.val_csv_path, model_name=args.model_name, is_training=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                               num_workers=args.num_workers, pin_memory=True)
    
    # Training State
    global_step = 0
    hidden_dim = sae.hidden_dim
    neuron_activity = torch.zeros(hidden_dim, device=device)
    
    best_val_loss = float('inf')
    
    # Training Loop
    print("Starting training...")
    print(f'Trainable parameters: {sum(p.numel() for p in sae.parameters() if p.requires_grad)}')
    print(f'Size of training dataset: {len(train_dataset)}')
    if val_loader:
        print(f'Size of validation dataset: {len(val_dataset)}')
    
    for epoch in range(1, args.epochs + 1):
        sae.train()
        epoch_loss = 0.0
        batches = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}")
        for images in pbar:
            images = images.to(device)
            
            # Get embeddings (no grad)
            with torch.no_grad():
                # HuggingFace CLIP forward
                vision_outputs = backbone.vision_model(pixel_values=images)
                if hasattr(vision_outputs, 'pooler_output'):
                    pooler_output = vision_outputs.pooler_output
                else:
                    pooler_output = vision_outputs[1]
                
                # Projection
                if hasattr(backbone, 'visual_projection'):
                    z = backbone.visual_projection(pooler_output)
                else:
                    z = pooler_output
            
            # SAE Forward
            z_hat, acts, loss = sae(z)
            
            # Optimization
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            # Normalize decoder columns (prevent explosion)
            sae.normalize_decoder()
            
            # Track stats
            epoch_loss += loss.item()
            batches += 1
            global_step += 1
            
            # Track neuron activity (binary: did it fire?)
            with torch.no_grad():
                fired = (acts > 0).float().sum(dim=0)
                neuron_activity += fired
            
            pbar.set_postfix({"Loss": f"{loss.item():.4f}"})
            
        avg_loss = epoch_loss / batches
        print(f"Epoch {epoch} Train Loss: {avg_loss:.6f}")
        
        # Validation Loop
        val_loss = 0.0
        if val_loader:
            sae.eval()
            val_batches = 0
            with torch.no_grad():
                for images in val_loader:
                    images = images.to(device)
                    
                    # Get embeddings
                    vision_outputs = backbone.vision_model(pixel_values=images)
                    if hasattr(vision_outputs, 'pooler_output'):
                        pooler_output = vision_outputs.pooler_output
                    else:
                        pooler_output = vision_outputs[1]
                        
                    if hasattr(backbone, 'visual_projection'):
                        z = backbone.visual_projection(pooler_output)
                    else:
                        z = pooler_output
                        
                    z_hat, acts, loss = sae(z)
                    val_loss += loss.item()
                    val_batches += 1
            
            avg_val_loss = val_loss / val_batches
            print(f"Epoch {epoch} Val Loss: {avg_val_loss:.6f}")
        else:
            avg_val_loss = avg_loss # Fallback if no val set
        
        # Dead Neuron Resampling
        if epoch % args.resample_freq == 0:
            dead_mask = (neuron_activity == 0)
            num_dead = dead_mask.sum().item()
            print(f"Dead neurons check: {int(num_dead)} / {hidden_dim} ({num_dead/hidden_dim*100:.1f}%)")
            
            if num_dead > 0:
                print("Resampling dead neurons...")
                with torch.no_grad():
                    # Reset encoder weights for dead neurons to match random current inputs
                    # We need a batch of data. Use the last batch 'z'.
                    # If num_dead > batch_size, we reuse z multiple times or sample mostly from it
                    
                    # Select random inputs from current batch to be new features
                    indices = torch.randint(0, z.shape[0], (int(num_dead),))
                    new_features = z[indices] # (num_dead, input_dim)
                    
                    # Normalize
                    new_features = F.normalize(new_features, p=2, dim=1)
                    
                    # Reset Encoder Weights: set to match the input feature
                    sae.encoder.weight.data[dead_mask] = new_features
                    # Reset Encoder Bias: usually set to 0 or slightly negative to not fire immediately?
                    # Setting to 0 is standard for resampling
                    sae.encoder.bias.data[dead_mask] = 0.0
                    
                    # Reset Decoder Weights: match the same feature (transpose)
                    sae.decoder.weight.data[:, dead_mask] = new_features.T
                    
                    # Reset Optimizer state for these parameters (important!)
                    # This is complex in PyTorch Adam, often skipped in simple implementations
                    # But ideally should reset momentum buffers for these indices.
                    
                # Reset activity counter
                neuron_activity.zero_()
                print("Resampling complete.")
        
        # Save Checkpoint
        if epoch % 5 == 0 or epoch == args.epochs:
            save_path = output_dir / f"sae_epoch_{epoch}.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"Saved checkpoint to {save_path}")
            
        # Save Best Model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            save_path = output_dir / "best_sae.pth"
            torch.save({
                'epoch': epoch,
                'model_state_dict': sae.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': avg_loss,
                'val_loss': avg_val_loss
            }, save_path)
            print(f"Saved BEST checkpoint (Val Loss: {best_val_loss:.4f}) to {save_path}")

    print("Training finished.")

if __name__ == "__main__":
    main()

