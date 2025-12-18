#!/usr/bin/env python3
"""
FAST Precompute script for StreetCLIP embeddings.
Optimized for HPC clusters (Snellius) to prevent deadlocks and CPU bottlenecks.

Key Fixes:
1. Import torch BEFORE pandas to prevent OpenMP deadlocks.
2. Use List[Dict] instead of DataFrame.iloc for O(1) data access.
3. Use torchvision transforms (C++) instead of AutoImageProcessor (Python).
"""

# --- CRITICAL: IMPORT TORCH FIRST ---
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import json
import os
import numpy as np

# --- FAST TRANSFORMS ---
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoModel, AutoConfig

# Standard OpenAI CLIP normalization constants (Hardcoded for speed)
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)

class FastImageDataset(Dataset):
    """
    Optimized dataset using List storage and Torchvision transforms.
    """
    def __init__(self, csv_path: str, image_size=336):
        print(f"Loading metadata from {csv_path}...")
        try:
            df = pd.read_csv(csv_path)
        except Exception as e:
            raise RuntimeError(f"Failed to read CSV: {e}")
            
        # Filter valid rows
        df = df.dropna(subset=['image_path'])
        
        # --- THE SPEED FIX: Convert Pandas to List of Dicts ---
        # This makes access inside __getitem__ instant (O(1)).
        # .iloc is O(N) overhead and creates heavy Series objects.
        self.samples = df.to_dict('records')
        print(f"Loaded {len(self.samples)} samples into memory.")
        
        self.image_size = image_size
        
        # --- THE SPEED FIX: Use Torchvision C++ Transforms ---
        # This pipeline releases the GIL and allows multi-worker loading
        self.transform = transforms.Compose([
            transforms.Resize((image_size, image_size), interpolation=InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=CLIP_MEAN, std=CLIP_STD)
        ])
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        # FAST ACCESS: Direct list lookup
        item = self.samples[idx]
        image_path = item['image_path']
        pano_id = item.get('pano_id', f'img_{idx}')
        
        try:
            # We use .convert('RGB') to ensure 3 channels
            image = Image.open(image_path).convert('RGB')
            pixel_values = self.transform(image)
            return pixel_values, pano_id, idx
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return black image as fallback to prevent crash
            return torch.zeros((3, self.image_size, self.image_size)), pano_id, idx

@torch.no_grad()
def extract_embeddings_batch(
    model,
    pixel_values,
    save_patch_tokens: bool = False,
    expected_num_patches: int | None = None,
):
    """
    Run inference on a batch of images.
    Returns projected embeddings (in shared space with text) and optionally raw patch tokens.
    """
    # Use model's built-in method (handles projection automatically)
    if not hasattr(model, 'get_image_features'):
        raise AttributeError(
            "Model does not have 'get_image_features' method. "
            "StreetCLIP model structure not recognized. Expected HuggingFace CLIP-style model."
        )
    
    try:
        pooled_embeddings = model.get_image_features(pixel_values=pixel_values)
    except Exception as e:
        raise RuntimeError(
            f"Failed to extract image features using 'get_image_features': {e}\n"
            f"Model may not be properly loaded or may have unexpected structure."
        ) from e
    
    # Normalize embeddings (standard for CLIP-style models)
    pooled_embeddings = torch.nn.functional.normalize(pooled_embeddings, p=2, dim=-1)
    
    # Extract patch tokens if needed (raw, before projection)
    patch_tokens = None
    if save_patch_tokens:
        if not hasattr(model, 'vision_model'):
            raise AttributeError(
                "Model does not have 'vision_model' attribute. "
                "Cannot extract patch tokens. Model structure not recognized."
            )
        
        try:
            outputs = model.vision_model(pixel_values=pixel_values)
            hidden_states = outputs.last_hidden_state
        except Exception as e:
            raise RuntimeError(
                f"Failed to extract patch tokens from vision_model: {e}"
            ) from e
        
        # Robust token slicing:
        # Prefer using expected_num_patches derived from model config (image_size/patch_size)
        # rather than brittle heuristics like "seq_len > 196".
        seq_len = int(hidden_states.shape[1])
        if expected_num_patches is not None:
            if seq_len == expected_num_patches:
                patch_tokens = hidden_states  # [B, P, D]
            elif seq_len > expected_num_patches:
                # Many ViT-style models include 1 (CLS) or 2 (CLS+distill) special tokens.
                num_extra = seq_len - expected_num_patches
                patch_tokens = hidden_states[:, num_extra:, :]  # drop leading special tokens
            else:
                raise RuntimeError(
                    f"Unexpected vision token sequence length: seq_len={seq_len} < "
                    f"expected_num_patches={expected_num_patches}. Check model config / input resolution."
                )
        else:
            # Fallback: assume first token is special if seq_len is not a perfect square.
            # (This is still weaker than config-based slicing.)
            side = int(np.sqrt(seq_len))
            if side * side == seq_len:
                patch_tokens = hidden_states
            else:
                # Try dropping 1 token (CLS) then check square
                side2 = int(np.sqrt(seq_len - 1))
                if side2 * side2 == (seq_len - 1):
                    patch_tokens = hidden_states[:, 1:, :]
                else:
                    patch_tokens = hidden_states  # last resort
    
    return pooled_embeddings.cpu(), patch_tokens.cpu() if patch_tokens is not None else None

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=str, required=True, help="Path to train.csv")
    parser.add_argument("--val-csv", type=str, required=True, help="Path to val.csv")
    parser.add_argument("--test-csv", type=str, default=None, help="Path to test.csv")
    parser.add_argument("--output-dir", type=str, required=True, help="Where to save .pt files")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="HuggingFace model ID")
    
    # Tuning args
    parser.add_argument("--batch-size", type=int, default=128, help="Batch size (Try 128 or 256)")
    parser.add_argument("--num-workers", type=int, default=8, help="DataLoader workers (8-12 is usually optimal)") 
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--save-patch-tokens", action="store_true", help="Save spatial tokens (High disk usage!)")
    
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Detect proper image size / patch size (for correct patch token slicing)
    print(f"Loading Config for: {args.model_name}")
    try:
        config = AutoConfig.from_pretrained(args.model_name)
        # StreetCLIP uses 'vision_config', standard CLIP uses top-level config
        if hasattr(config, 'vision_config'):
            image_size = getattr(config.vision_config, "image_size", 336)
            patch_size = getattr(config.vision_config, "patch_size", None)
        else:
            image_size = getattr(config, "image_size", 336)
            patch_size = getattr(config, "patch_size", None)
    except Exception:
        print("Warning: Could not auto-detect size. Defaulting to 336px.")
        image_size = 336
        patch_size = None
        
    print(f"Target Resolution: {image_size}x{image_size}")
    if patch_size is not None:
        print(f"Detected vision patch size: {patch_size}")
    else:
        print("Warning: Could not detect vision patch size from config; patch token slicing may be less reliable.")

    # 2. Load Model
    print(f"Loading Model Weights...")
    model = AutoModel.from_pretrained(args.model_name)
    model.eval()
    model = model.to(args.device)
    
    # Expected number of patch tokens (excluding special tokens like CLS).
    expected_num_patches = None
    if patch_size is not None and isinstance(image_size, int) and image_size % int(patch_size) == 0:
        grid = image_size // int(patch_size)
        expected_num_patches = int(grid * grid)
        print(f"Expected patch tokens per image: {expected_num_patches} ({grid}x{grid})")
    else:
        print("Warning: Cannot compute expected_num_patches; will fall back to square/CLS heuristics.")

    splits = {'train': args.train_csv, 'val': args.val_csv}
    if args.test_csv: splits['test'] = args.test_csv
    
    # 3. Processing Loop
    for split_name, csv_path in splits.items():
        print(f"\n{'='*40}")
        print(f"Processing split: {split_name}")
        print(f"{'='*40}")
        
        # Initialize Optimized Dataset
        dataset = FastImageDataset(csv_path, image_size=image_size)
        
        # Safe to use workers now because imports are correct and transforms are C++
        dataloader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            shuffle=False,
            pin_memory=True, # Faster transfer to GPU
            prefetch_factor=2, # Buffer batches
            persistent_workers=True # Keep workers alive between batches
        )
        
        all_pooled = []
        all_patches = []
        all_panos = []
        all_idxs = []
        
        # Main Loop
        for pixel_values, pano_ids, indices in tqdm(dataloader, desc=f"{split_name}"):
            pixel_values = pixel_values.to(args.device)
            
            # Inference
            pooled, patches = extract_embeddings_batch(
                model,
                pixel_values,
                save_patch_tokens=args.save_patch_tokens,
                expected_num_patches=expected_num_patches,
            )
            
            all_pooled.append(pooled)
            if patches is not None: all_patches.append(patches)
            all_panos.extend(pano_ids)
            all_idxs.extend(indices.tolist())
        
        # Save Results
        if all_pooled:
            print(f"Concatenating {len(all_pooled)} batches...")
            pooled_tensor = torch.cat(all_pooled, dim=0)
            
            save_path = output_dir / f"{split_name}_pooled_embeddings.pt"
            torch.save(pooled_tensor, save_path)
            print(f"Saved pooled embeddings: {pooled_tensor.shape} (dimension: {pooled_tensor.shape[1]})")
            
            if args.save_patch_tokens and all_patches:
                patch_tensor = torch.cat(all_patches, dim=0)
                patches_path = output_dir / f"{split_name}_patch_tokens.pt"
                torch.save(patch_tensor, patches_path)
                print(f"Saved patch tokens: {patch_tensor.shape}")

            # Save Metadata mapping
            # NOTE: This metadata is relied on by ConceptDataset to map CSV rows to cached embeddings.
            metadata = {
                'pano_ids': list(all_panos),
                'indices': all_idxs,
                'total_samples': pooled_tensor.shape[0],
                'image_size': image_size,
                'patch_size': patch_size,
                'expected_num_patches': expected_num_patches,
            }
            with open(output_dir / f"{split_name}_metadata.json", 'w') as f:
                json.dump(metadata, f, indent=2)

    print(f"\nDone! Outputs saved to {output_dir}")

if __name__ == "__main__":
    main()