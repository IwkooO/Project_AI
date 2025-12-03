#!/usr/bin/env python3
"""
Precompute and cache StreetCLIP image embeddings for all images in the dataset.

This script:
1. Loads StreetCLIP image encoder (frozen)
2. Processes all images from train/val/test CSVs
3. Extracts and caches:
   - Pooled embeddings z(x) ∈ R^768 (always saved, for global mode)
   - Patch tokens T(x) ∈ R^(P×768) for each image (optional, for spatial mode)
4. Saves as PyTorch tensors for efficient loading

Output: {dataset_dir}/cached_embeddings/ with train/val/test splits
"""

import argparse
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import numpy as np
from transformers import AutoImageProcessor, AutoModel
import json


class ImageDataset(Dataset):
    """Simple dataset for loading images from CSV."""
    def __init__(self, csv_path: str, model_name: str = "geolocal/StreetCLIP"):
        self.df = pd.read_csv(csv_path)
        # Filter for valid images
        self.df = self.df.dropna(subset=['image_path'])
        
        # Get model-specific transform
        self.processor = AutoImageProcessor.from_pretrained(model_name)
    
    def __len__(self):
        return len(self.df)
    
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image_path = row['image_path']
        pano_id = row.get('pano_id', f'img_{idx}')
        
        try:
            image = Image.open(image_path).convert('RGB')
            # Process image using StreetCLIP processor
            inputs = self.processor(images=image, return_tensors="pt")
            # Remove batch dimension for single image
            pixel_values = inputs['pixel_values'].squeeze(0)
            return pixel_values, pano_id, idx
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a black image as fallback
            size = (336, 336) if not hasattr(self.processor, 'size') else self.processor.size
            if isinstance(size, dict):
                size = (size['height'], size['width'])
            return torch.zeros((3, size[0], size[1])), pano_id, idx


def extract_patch_tokens(model, pixel_values, device):
    """
    Extract patch tokens from StreetCLIP ViT.
    
    Args:
        model: StreetCLIP model
        pixel_values: Preprocessed image tensor [C, H, W]
        device: Device to run on
    
    Returns:
        patch_tokens: T(x) ∈ R^(P×768) where P is number of patches
        pooled_embedding: z(x) ∈ R^768 (optional, if available)
    """
    # Add batch dimension
    pixel_values = pixel_values.unsqueeze(0).to(device)
    
    with torch.no_grad():
        # Forward through vision encoder
        outputs = model.vision_model(pixel_values=pixel_values)
        
        # Get patch tokens (exclude CLS token if present)
        # Vision transformer typically outputs [batch, num_patches+1, hidden_dim]
        # where first token is CLS token
        hidden_states = outputs.last_hidden_state  # [1, P+1, 768] or [1, P, 768]
        
        # Check if first token is CLS token (usually has different norm)
        # For StreetCLIP, we'll extract all tokens as patch tokens
        # If there's a CLS token, we can optionally use it as pooled embedding
        if hidden_states.shape[1] > 196:  # Likely has CLS token
            # First token is CLS, rest are patches
            pooled_embedding = hidden_states[:, 0, :].squeeze(0)  # [768]
            patch_tokens = hidden_states[:, 1:, :].squeeze(0)  # [P, 768]
        else:
            # No CLS token, all are patches
            patch_tokens = hidden_states.squeeze(0)  # [P, 768]
            # Use mean pooling for pooled embedding
            pooled_embedding = patch_tokens.mean(dim=0)  # [768]
    
    return patch_tokens.cpu(), pooled_embedding.cpu()


def main():
    parser = argparse.ArgumentParser(description="Precompute StreetCLIP embeddings")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to train CSV")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to val CSV")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test CSV (optional)")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for cached embeddings")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="StreetCLIP model name")
    parser.add_argument("--batch-size", type=int, default=32, help="Batch size for processing")
    parser.add_argument("--num-workers", type=int, default=8, help="Number of data loading workers")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu", help="Device to use")
    parser.add_argument("--save-patch-tokens", action="store_true", help="Also save patch tokens (for spatial mode)")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load StreetCLIP model
    print(f"Loading StreetCLIP model: {args.model_name}")
    try:
        model = AutoModel.from_pretrained(args.model_name)
    except Exception as e:
        print(f"Error loading model {args.model_name}: {e}")
        exit(1)
    
    # Freeze model
    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    
    device = torch.device(args.device)
    model = model.to(device)
    
    # Process each split
    splits = {
        'train': args.train_csv,
        'val': args.val_csv,
    }
    if args.test_csv:
        splits['test'] = args.test_csv
    
    for split_name, csv_path in splits.items():
        print(f"\n{'='*60}")
        print(f"Processing {split_name} split")
        print(f"{'='*60}")
        
        # Create dataset
        dataset = ImageDataset(csv_path, model_name=args.model_name)
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True if args.device == "cuda" else False
        )
        
        # Storage for embeddings
        all_patch_tokens = []
        all_pooled_embeddings = []
        all_pano_ids = []
        all_indices = []
        
        # Process batches
        for batch_idx, (pixel_values, pano_ids, indices) in enumerate(tqdm(dataloader, desc=f"Processing {split_name}")):
            # Move to device
            pixel_values = pixel_values.to(device)
            
            # Process each image in batch (since model might not support batched patch extraction easily)
            batch_patch_tokens = []
            batch_pooled = []
            
            for i in range(pixel_values.shape[0]):
                patch_tokens, pooled_emb = extract_patch_tokens(
                    model, pixel_values[i], device
                )
                batch_pooled.append(pooled_emb)  # Always collect pooled
                if args.save_patch_tokens:
                    batch_patch_tokens.append(patch_tokens)
            
            all_pooled_embeddings.extend(batch_pooled)
            if args.save_patch_tokens:
                all_patch_tokens.extend(batch_patch_tokens)
            all_pano_ids.extend(pano_ids)
            all_indices.extend(indices.tolist())
        
        # Stack into tensors and save
        print(f"Stacking {len(all_pooled_embeddings)} embeddings...")
        
        # Always save pooled embeddings (for global mode)
        pooled_tensor = torch.stack(all_pooled_embeddings)  # [N, 768]
        pooled_path = output_dir / f"{split_name}_pooled_embeddings.pt"
        torch.save(pooled_tensor, pooled_path)
        print(f"Saved pooled embeddings to {pooled_path}")
        print(f"  Shape: {pooled_tensor.shape}")
        
        # Optionally save patch tokens (for spatial mode)
        if args.save_patch_tokens:
            patch_tokens_tensor = torch.stack(all_patch_tokens)  # [N, P, 768]
            patch_tokens_path = output_dir / f"{split_name}_patch_tokens.pt"
            torch.save(patch_tokens_tensor, patch_tokens_path)
            print(f"Saved patch tokens to {patch_tokens_path}")
            print(f"  Shape: {patch_tokens_tensor.shape}")
        
        # Save metadata (pano_ids and indices for mapping back to CSV)
        metadata = {
            'pano_ids': all_pano_ids,
            'indices': all_indices,
            'num_samples': len(all_patch_tokens),
            'patch_shape': list(patch_tokens_tensor.shape[1:]),  # [P, 768]
        }
        metadata_path = output_dir / f"{split_name}_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        print(f"Saved metadata to {metadata_path}")
    
    print(f"\n{'='*60}")
    print("Precomputation complete!")
    print(f"Embeddings saved to: {output_dir}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

