#!/usr/bin/env python3
"""
Pre-compute and cache embeddings from the backbone model.
This generates multiple augmented views for training data to simulate data augmentation
while allowing for extremely fast SAE training.
"""
import argparse
import torch
from torch.utils.data import DataLoader
from transformers import AutoModel
from tqdm import tqdm
from pathlib import Path
import sys
import os

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset_sae import SAEDataset

def cache_dataset(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # 1. Load Backbone
    print(f"Loading backbone: {args.model_name}")
    backbone = AutoModel.from_pretrained(args.model_name)
    backbone.to(device)
    backbone.eval()

    # 2. Process TRAIN (with Augmentations)
    # We will generate N "copies" of the dataset, each with different random augmentations
    train_dataset = SAEDataset(args.train_csv, model_name=args.model_name, is_training=True)
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    
    all_train_vectors = []
    
    print(f"Caching {args.num_augments} augmented versions of training data...")
    for i in range(args.num_augments):
        print(f"--- Augmentation Pass {i+1}/{args.num_augments} ---")
        with torch.no_grad():
            for images in tqdm(train_loader, desc=f"Pass {i+1}"):
                images = images.to(device)
                
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
                    
                all_train_vectors.append(z.cpu()) # Save to RAM (CPU)

    # Concatenate and Save
    if all_train_vectors:
        final_train = torch.cat(all_train_vectors, dim=0)
        print(f"Saving {final_train.shape} training vectors to {output_dir / 'train_embeddings.pt'}")
        torch.save(final_train, output_dir / "train_embeddings.pt")
    else:
        print("Warning: No training data generated!")

    # 3. Process VAL (No Augmentations)
    if args.val_csv:
        val_dataset = SAEDataset(args.val_csv, model_name=args.model_name, is_training=False)
        val_loader = DataLoader(val_dataset, batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
        
        all_val_vectors = []
        print("Caching validation data (Deterministic/No Augmentation)...")
        with torch.no_grad():
            for images in tqdm(val_loader, desc="Validation"):
                images = images.to(device)
                
                vision_outputs = backbone.vision_model(pixel_values=images)
                if hasattr(vision_outputs, 'pooler_output'):
                    pooler_output = vision_outputs.pooler_output
                else:
                    pooler_output = vision_outputs[1]
                
                if hasattr(backbone, 'visual_projection'):
                    z = backbone.visual_projection(pooler_output)
                else:
                    z = pooler_output
                    
                all_val_vectors.append(z.cpu())

        if all_val_vectors:
            final_val = torch.cat(all_val_vectors, dim=0)
            print(f"Saving {final_val.shape} validation vectors to {output_dir / 'val_embeddings.pt'}")
            torch.save(final_val, output_dir / "val_embeddings.pt")

    print("Caching complete!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache embeddings for fast SAE training")
    parser.add_argument("--train-csv", type=str, required=True, help="Path to training CSV")
    parser.add_argument("--val-csv", type=str, default=None, help="Path to validation CSV")
    parser.add_argument("--output-dir", type=str, required=True, help="Directory to save .pt files")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP", help="Backbone model name")
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size for backbone inference")
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--num-augments", type=int, default=10, help="Number of augmented copies to create for training set")
    
    args = parser.parse_args()
    cache_dataset(args)


