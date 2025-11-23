#!/usr/bin/env python3
"""
Evaluate a trained SAE on the held-out Test Set.
Calculates Reconstruction Loss (MSE), L0 Sparsity, L1 Sparsity, and Explained Variance.
"""
import argparse
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm
from pathlib import Path
import sys
import numpy as np

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.sae import TopKSAE
from src.data.dataset_sae import SAEDataset

def get_explained_variance(original, reconstruction):
    """
    Computes explained variance: 1 - (var(original - reconstruction) / var(original))
    """
    # Center the data
    original_centered = original - original.mean(dim=0, keepdim=True)
    
    # Compute variance of original data
    total_variance = (original_centered ** 2).mean()
    
    # Compute variance of residuals
    residuals = original - reconstruction
    residual_variance = (residuals ** 2).mean()
    
    return 1 - (residual_variance / total_variance)

def main():
    parser = argparse.ArgumentParser(description="Evaluate SAE on Test Set")
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE checkpoint (.pth)")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test dataset CSV")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="Backbone model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = torch.device(args.device)
    print(f"Using device: {device}")

    # 1. Load SAE
    print(f"Loading SAE from {args.sae_path}")
    checkpoint = torch.load(args.sae_path, map_location=device)
    
    # Infer dimensions from state dict
    if 'encoder.weight' in checkpoint['model_state_dict']:
        hidden_dim, input_dim = checkpoint['model_state_dict']['encoder.weight'].shape
    else:
        # Try to guess from args if saved, otherwise default
        input_dim = 768
        hidden_dim = input_dim * 8 # Guess
        
    # We need to initialize with correct config. Ideally this is saved in checkpoint.
    # For now, we reconstruct from dimensions
    k = 32 # Default, strictly needed for TopKSAE forward pass logic if it uses k internally
    # Note: TopKSAE requires 'k' in init. If it's not saved in checkpoint, we might need to pass it as arg.
    # Let's assume k=32 or try to find it.
    
    sae = TopKSAE(input_dim=input_dim, expansion_factor=hidden_dim//input_dim, k=k)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device)
    sae.eval()
    
    print(f"SAE Config: Input={input_dim}, Hidden={hidden_dim}, K={k}")

    # 2. Load Backbone
    print(f"Loading backbone: {args.model_name}")
    backbone = AutoModel.from_pretrained(args.model_name)
    backbone.to(device)
    backbone.eval()
    
    # 3. Load Test Data
    print(f"Loading Test Set from {args.test_csv}")
    test_dataset = SAEDataset(args.test_csv, model_name=args.model_name, is_training=False)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=8)
    
    # 4. Evaluation Loop
    total_mse = 0.0
    total_l0 = 0.0
    total_l1 = 0.0
    batches = 0
    
    all_originals = []
    all_reconstructions = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for images in tqdm(test_loader):
            images = images.to(device)
            
            # Backbone Forward
            vision_outputs = backbone.vision_model(pixel_values=images)
            pooler_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]
            z = backbone.visual_projection(pooler_output) if hasattr(backbone, 'visual_projection') else pooler_output
            
            # SAE Forward
            z_hat, acts, loss = sae(z)
            
            # Metrics
            mse = F.mse_loss(z_hat, z).item()
            l0 = (acts > 0).float().sum(dim=1).mean().item() # Average active neurons per sample
            l1 = acts.sum(dim=1).mean().item() # Average sum of activations per sample
            
            total_mse += mse
            total_l0 += l0
            total_l1 += l1
            batches += 1
            
            # Store for Explained Variance (if memory allows, otherwise compute online)
            # For Test sets (usually ~10k), this is fine (10k * 768 * 4 bytes = 30MB)
            all_originals.append(z.cpu())
            all_reconstructions.append(z_hat.cpu())

    # Aggregate
    avg_mse = total_mse / batches
    avg_l0 = total_l0 / batches
    avg_l1 = total_l1 / batches
    
    full_orig = torch.cat(all_originals, dim=0)
    full_recon = torch.cat(all_reconstructions, dim=0)
    
    explained_var = get_explained_variance(full_orig, full_recon).item()
    
    print("\n" + "="*40)
    print(f"TEST SET RESULTS")
    print("="*40)
    print(f"Samples:            {len(test_dataset)}")
    print(f"MSE Loss:           {avg_mse:.6f}")
    print(f"RMSE:               {np.sqrt(avg_mse):.6f}")
    print(f"Explained Variance: {explained_var:.4f} (Max 1.0)")
    print("-" * 20)
    print(f"L0 Sparsity (Avg K): {avg_l0:.2f}")
    print(f"L1 Sparsity:        {avg_l1:.4f}")
    print("="*40)

if __name__ == "__main__":
    main()

