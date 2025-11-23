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

class TensorDataset(torch.utils.data.Dataset):
    """Dataset for loading pre-computed embeddings."""
    def __init__(self, tensor_path):
        print(f"Loading vectors from {tensor_path}...")
        self.data = torch.load(tensor_path)
        print(f"Loaded {len(self.data)} vectors.")

    def __getitem__(self, idx):
        return self.data[idx]

    def __len__(self):
        return len(self.data)

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
    
    # New arguments for cached training
    parser.add_argument("--use-cached", action="store_true", help="Use pre-computed embeddings instead of running backbone")
    parser.add_argument("--cached-train-path", type=str, default=None, help="Path to cached training .pt file")
    parser.add_argument("--cached-val-path", type=str, default=None, help="Path to cached validation .pt file")

    # Argument for restarting training from a checkpoint
    parser.add_argument("--checkpoint-path", type=str, default=None, help="Path to checkpoint to restart from")
    
    args = parser.parse_args()

    # Setup
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Backbone (Frozen) - ONLY if not using cached embeddings
    backbone = None
    input_dim = 768 # Default
    
    if not args.use_cached:
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
        print("Using CACHED embeddings. Backbone will NOT be loaded.")
        # If using cached, we assume 768 or try to infer if possible, 
        # but usually it's 768 for StreetCLIP/CLIP-Vit-L/14
        # You could also load a small subset of the pt file to check dim, but 768 is safe default for now.

    print(f"Embedding dimension: {input_dim}")

    # 2. Initialize SAE
    print(f"Initializing SAE (Expansion: {args.expansion}x, K: {args.k})")
    sae = TopKSAE(input_dim=input_dim, expansion_factor=args.expansion, k=args.k)
    sae.to(device)
    
    # Optimizer
    optimizer = Adam(sae.parameters(), lr=args.lr)
    
    # Dataset
    if args.use_cached:
        if not args.cached_train_path:
            raise ValueError("--cached-train-path must be provided when --use-cached is True")
            
        print(f"Loading CACHED training dataset from {args.cached_train_path}")
        train_dataset = TensorDataset(args.cached_train_path)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
        
        val_loader = None
        if args.cached_val_path:
            print(f"Loading CACHED validation dataset from {args.cached_val_path}")
            val_dataset = TensorDataset(args.cached_val_path)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
            
    else:
        print(f"Loading standard training dataset from {args.csv_path}")
        train_dataset = SAEDataset(args.csv_path, model_name=args.model_name, is_training=True)
        train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, 
                               num_workers=args.num_workers, pin_memory=True)
        
        val_loader = None
        if args.val_csv_path:
            print(f"Loading standard validation dataset from {args.val_csv_path}")
            val_dataset = SAEDataset(args.val_csv_path, model_name=args.model_name, is_training=False)
            val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, 
                                   num_workers=args.num_workers, pin_memory=True)
    
    # Load checkpoint if provided (just the weights)
    if args.checkpoint_path:
        print(f"Loading weights from {args.checkpoint_path}")
        checkpoint = torch.load(args.checkpoint_path, map_location=device)
        sae.load_state_dict(checkpoint['model_state_dict'])
        print("Loaded model weights from checkpoint")
    
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
        for batch_data in pbar:
            batch_data = batch_data.to(device)
            
            if args.use_cached:
                # Data IS the embedding
                z = batch_data
            else:
                # Data IS images, run backbone
                images = batch_data
                with torch.no_grad():
                    vision_outputs = backbone.vision_model(pixel_values=images)
                    if hasattr(vision_outputs, 'pooler_output'):
                        pooler_output = vision_outputs.pooler_output
                    else:
                        pooler_output = vision_outputs[1]
                    
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
                for batch_data in val_loader:
                    batch_data = batch_data.to(device)
                    
                    if args.use_cached:
                        z = batch_data
                    else:
                        images = batch_data
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
        
        # Save Best Model (BEFORE resampling to avoid saving broken state)
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
        
        # Dead Neuron Resampling
        if epoch % args.resample_freq == 0:
            dead_mask = (neuron_activity == 0)
            num_dead = dead_mask.sum().item()
            print(f"Dead neurons check: {int(num_dead)} / {hidden_dim} ({num_dead/hidden_dim*100:.1f}%)")
            
            if num_dead > 0:
                print("Resampling dead neurons...")
                with torch.no_grad():
                    # Resample logic...
                    # We need 'z' from the last batch to pick random features
                    # If batch size is small, we might need to be careful, but standard batches are fine.
                    
                    # Select random inputs from current batch to be new features
                    # If batch is smaller than num_dead, we might crash. 
                    # Safe approach: repeat z if needed
                    if z.shape[0] < num_dead:
                        repeats = int(num_dead // z.shape[0]) + 1
                        z_pool = z.repeat(repeats, 1)
                    else:
                        z_pool = z
                        
                    indices = torch.randint(0, z_pool.shape[0], (int(num_dead),))
                    new_features = z_pool[indices] # (num_dead, input_dim)
                    
                    # Normalize
                    new_features = F.normalize(new_features, p=2, dim=1)
                    
                    # Reset Encoder Weights
                    sae.encoder.weight.data[dead_mask] = new_features
                    sae.encoder.bias.data[dead_mask] = 0.0
                    
                    # Reset Decoder Weights
                    sae.decoder.weight.data[:, dead_mask] = new_features.T
                    
                    # CRITICAL: Normalize decoder columns after resampling
                    # This ensures unit norm constraint is maintained
                    sae.normalize_decoder()
                    
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

    print("Training finished.")

if __name__ == "__main__":
    main()
