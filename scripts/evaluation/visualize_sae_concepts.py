#!/usr/bin/env python3
"""
Visualize learned concepts from the Sparse Autoencoder.
Generates grids of top-activating images and geographic heatmaps.
Includes deduplication to prevent 'loud' images from dominating all concepts.
"""
import argparse
import torch
import pandas as pd
import matplotlib.pyplot as plt
from PIL import Image
from transformers import AutoModel, AutoImageProcessor
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from pathlib import Path
import sys
import numpy as np
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.models.sae import TopKSAE

class VizDataset(Dataset):
    def __init__(self, csv_path, model_name="geolocal/StreetCLIP"):
        self.df = pd.read_csv(csv_path)
        # Use a subset for visualization speed if needed, or full set
        self.df = self.df.dropna(subset=['image_path'])
        
        try:
            processor = AutoImageProcessor.from_pretrained(model_name)
            size = (processor.size["height"], processor.size["width"]) if isinstance(processor.size, dict) else (processor.size, processor.size)
            mean = processor.image_mean
            std = processor.image_std
        except:
            size = (336, 336)
            mean = (0.48145466, 0.4578275, 0.40821073)
            std = (0.26862954, 0.26130258, 0.27577711)
            
        self.transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
            transforms.ToTensor(),
            transforms.Normalize(mean=mean, std=std)
        ])
        
    def __len__(self):
        return len(self.df)
        
    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = row['image_path']
        try:
            img = Image.open(img_path).convert('RGB')
            tensor = self.transform(img)
            return tensor, idx
        except:
            return torch.zeros((3, 336, 336)), idx

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE checkpoint")
    parser.add_argument("--csv-path", type=str, required=True, help="Dataset CSV")
    parser.add_argument("--output-dir", type=str, default="visualizations/sae_concepts")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--num-concepts", type=int, default=50, help="Number of top concepts to visualize")
    parser.add_argument("--top-k-images", type=int, default=9, help="Images per concept")
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Models
    print("Loading models...")
    backbone = AutoModel.from_pretrained(args.model_name)
    backbone.to(device).eval()
    
    checkpoint = torch.load(args.sae_path, map_location=device)
    
    # Infer config
    if hasattr(backbone.config, "projection_dim"):
        input_dim = backbone.config.projection_dim
    else:
        input_dim = 768
        
    encoder_weight = checkpoint['model_state_dict']['encoder.weight']
    hidden_dim = encoder_weight.shape[0]
    expansion = hidden_dim // input_dim
    
    sae = TopKSAE(input_dim=input_dim, expansion_factor=expansion)
    sae.load_state_dict(checkpoint['model_state_dict'])
    sae.to(device).eval()
    
    # 2. Collect Activations
    dataset = VizDataset(args.csv_path, args.model_name)
    dataloader = DataLoader(dataset, batch_size=128, num_workers=8, shuffle=False)
    
    print("Collecting activations...")
    # (values, indices) for top K images per neuron
    top_acts_values = torch.zeros(hidden_dim, args.top_k_images, device='cpu')
    top_acts_indices = torch.zeros(hidden_dim, args.top_k_images, dtype=torch.long, device='cpu') - 1
    
    for images, indices in tqdm(dataloader):
        images = images.to(device)
        with torch.no_grad():
            vision_outputs = backbone.vision_model(pixel_values=images)
            pooler_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]
            z = backbone.visual_projection(pooler_output) if hasattr(backbone, 'visual_projection') else pooler_output
                
            _, acts, _ = sae(z) # (B, hidden_dim)
            
        # Update top K buffer
        batch_acts = acts.cpu()
        batch_indices = indices.cpu()
        
        active_neurons = torch.where(batch_acts.sum(0) > 0)[0]
        
        for neuron_idx in active_neurons:
            neuron_acts = batch_acts[:, neuron_idx]
            valid_mask = neuron_acts > 0
            if not valid_mask.any():
                continue
                
            valid_acts = neuron_acts[valid_mask]
            valid_img_indices = batch_indices[valid_mask]
            
            # Merge with existing
            current_vals = top_acts_values[neuron_idx]
            current_inds = top_acts_indices[neuron_idx]
            
            # Filter out -1s from current
            valid_current = current_inds != -1
            
            combined_vals = torch.cat([current_vals[valid_current], valid_acts])
            combined_inds = torch.cat([current_inds[valid_current], valid_img_indices])
            
            # Sort descending
            sorted_vals, sort_idx = torch.sort(combined_vals, descending=True)
            
            # Keep top K
            if len(sorted_vals) > args.top_k_images:
                top_acts_values[neuron_idx] = sorted_vals[:args.top_k_images]
                top_acts_indices[neuron_idx] = combined_inds[sort_idx][:args.top_k_images]
            else:
                # Pad with -1 if needed
                pad_len = args.top_k_images - len(sorted_vals)
                top_acts_values[neuron_idx] = torch.cat([sorted_vals, torch.zeros(pad_len)])
                top_acts_indices[neuron_idx] = torch.cat([combined_inds[sort_idx], torch.zeros(pad_len, dtype=torch.long) - 1])

    # 3. Select Interesting Concepts (With Deduplication)
    max_activations = top_acts_values[:, 0]
    # Sort all neurons by their peak activation
    sorted_neurons = torch.argsort(max_activations, descending=True)
    
    unique_neurons = []
    seen_image_sets = []
    
    print("Deduplicating concepts...")
    for neuron_idx in sorted_neurons:
        if len(unique_neurons) >= args.num_concepts:
            break
            
        neuron_idx = neuron_idx.item()
        img_indices = top_acts_indices[neuron_idx]
        
        # Get valid indices
        valid_mask = img_indices != -1
        current_set = set(img_indices[valid_mask].tolist())
        
        if len(current_set) == 0:
            continue
            
        # Check overlap with already selected neurons
        is_duplicate = False
        for seen_set in seen_image_sets:
            # Jaccard Similarity
            intersection = len(current_set.intersection(seen_set))
            union = len(current_set.union(seen_set))
            if union > 0 and (intersection / union) > 0.5: # 50% overlap threshold
                is_duplicate = True
                break
        
        if not is_duplicate:
            unique_neurons.append(neuron_idx)
            seen_image_sets.append(current_set)
    
    print(f"Found {len(unique_neurons)} unique concepts out of top candidates.")
    
    # 4. Generate Visualizations
    for i, concept_idx in enumerate(unique_neurons):
        img_indices = top_acts_indices[concept_idx]
        act_values = top_acts_values[concept_idx]
        
        valid_mask = img_indices != -1
        img_indices = img_indices[valid_mask]
        act_values = act_values[valid_mask]
        
        # Create Grid
        fig, axes = plt.subplots(3, 3, figsize=(12, 12))
        fig.suptitle(f"Concept {concept_idx} (Max Act: {act_values[0]:.2f})", fontsize=16)
        
        for j, ax in enumerate(axes.flat):
            if j < len(img_indices):
                idx = img_indices[j].item()
                val = act_values[j].item()
                
                # Load raw image
                row = dataset.df.iloc[idx]
                img_path = row['image_path']
                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                    ax.set_title(f"Act: {val:.2f}\n{row.get('country', 'Unknown')}")
                except:
                    ax.text(0.5, 0.5, "Error loading", ha='center')
            ax.axis('off')
            
        plt.tight_layout()
        plt.savefig(out_dir / f"concept_{concept_idx}_top_images.png")
        plt.close()
        
        # Geographic Plot
        lats = []
        lngs = []
        for idx in img_indices:
            row = dataset.df.iloc[idx.item()]
            if not pd.isna(row['lat']):
                lats.append(row['lat'])
                lngs.append(row['lng'])
                
        if lats:
            plt.figure(figsize=(10, 6))
            plt.scatter(lngs, lats, c='red', s=50, alpha=0.7)
            plt.xlim(-180, 180)
            plt.ylim(-90, 90)
            plt.grid(True, alpha=0.3)
            plt.title(f"Geographic Distribution: Concept {concept_idx}")
            plt.xlabel("Longitude")
            plt.ylabel("Latitude")
            plt.savefig(out_dir / f"concept_{concept_idx}_map.png")
            plt.close()

    print(f"Visualizations saved to {out_dir}")

if __name__ == "__main__":
    main()
