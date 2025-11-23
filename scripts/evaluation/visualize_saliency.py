#!/usr/bin/env python3
"""
Visualize saliency maps for specific SAE concepts.
Highlights which parts of the image contribute most to a concept's activation.
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

# Try to import cv2 for better heatmap generation, fallback to matplotlib
try:
    import cv2
    HAS_CV2 = True
except ImportError:
    HAS_CV2 = False
    print("OpenCV (cv2) not found. Using matplotlib for overlays.")

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
        
        # Keep a raw transform for visualization
        self.raw_transform = transforms.Compose([
            transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
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

def get_occlusion_heatmap(image_tensor, backbone, sae, concept_idx, stride=16, window_size=32):
    """
    Compute heatmap by sliding a gray occlusion window over the image.
    Much more robust for ViTs than raw gradients.
    """
    # 1. Get baseline activation
    with torch.no_grad():
        # Ensure input has batch dim
        if len(image_tensor.shape) == 3:
            image_tensor = image_tensor.unsqueeze(0)
            
        vision_outputs = backbone.vision_model(pixel_values=image_tensor)
        pooler_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]
        
        if hasattr(backbone, 'visual_projection'):
            z = backbone.visual_projection(pooler_output)
        else:
            z = pooler_output
            
        baseline_act = sae.encoder(z)[0, concept_idx].item()
    
    # 2. Setup sliding window
    _, c, h, w = image_tensor.shape
    heatmap = np.zeros((h, w))
    
    # Iterate over image with sliding window
    # We batch these to be faster
    batch_images = []
    coords = []
    
    for y in range(0, h - window_size + 1, stride):
        for x in range(0, w - window_size + 1, stride):
            # Create occluded image
            img_copy = image_tensor.clone()
            # Mask out the window with gray (0.0 or mean)
            img_copy[:, :, y:y+window_size, x:x+window_size] = 0.0 
            batch_images.append(img_copy)
            coords.append((x, y))
            
            # Process batch if full or at end
            if len(batch_images) >= 32: # Batch size
                batch_stack = torch.cat(batch_images, dim=0)
                with torch.no_grad():
                    # Forward pass batch
                    outs = backbone.vision_model(pixel_values=batch_stack)
                    pool = outs.pooler_output if hasattr(outs, 'pooler_output') else outs[1]
                    
                    if hasattr(backbone, 'visual_projection'):
                        emb = backbone.visual_projection(pool)
                    else:
                        emb = pool
                        
                    acts = sae.encoder(emb)[:, concept_idx]
                    
                    # Calculate drop (higher drop = more important)
                    drops = baseline_act - acts.cpu().numpy()
                    
                    # Fill heatmap
                    for drop, (bx, by) in zip(drops, coords):
                        # If drop is positive, this region was important
                        # Assign drop value to the window region
                        # We take max to overlap correctly
                        current_region = heatmap[by:by+window_size, bx:bx+window_size]
                        heatmap[by:by+window_size, bx:bx+window_size] = np.maximum(current_region, drop)
                
                batch_images = []
                coords = []

    # Process remaining
    if batch_images:
         batch_stack = torch.cat(batch_images, dim=0)
         with torch.no_grad():
            outs = backbone.vision_model(pixel_values=batch_stack)
            pool = outs.pooler_output if hasattr(outs, 'pooler_output') else outs[1]
            
            if hasattr(backbone, 'visual_projection'):
                emb = backbone.visual_projection(pool)
            else:
                emb = pool
                
            acts = sae.encoder(emb)[:, concept_idx]
            drops = baseline_act - acts.cpu().numpy()
            for drop, (bx, by) in zip(drops, coords):
                current_region = heatmap[by:by+window_size, bx:bx+window_size]
                heatmap[by:by+window_size, bx:bx+window_size] = np.maximum(current_region, drop)
    
    # Normalize
    heatmap = np.maximum(heatmap, 0) # Only positive contributions
    if heatmap.max() > 0:
        heatmap /= heatmap.max()
        
    return heatmap

def visualize_overlay(original_img_path, saliency_map, save_path, concept_idx, activation_val=None):
    """
    Overlay saliency map on original image.
    """
    try:
        img = Image.open(original_img_path).convert('RGB')
        # Resize image to match saliency map dimensions
        img = img.resize((saliency_map.shape[1], saliency_map.shape[0]), Image.BICUBIC)
        img_np = np.array(img)
    except Exception as e:
        print(f"Error loading image {original_img_path}: {e}")
        return

    # Setup plot
    plt.figure(figsize=(15, 5))
    
    # 1. Original Image
    plt.subplot(1, 3, 1)
    plt.imshow(img_np)
    plt.title("Original Image")
    plt.axis('off')
    
    # 2. Saliency Heatmap
    plt.subplot(1, 3, 2)
    plt.imshow(saliency_map, cmap='jet')
    plt.title(f"Saliency Map (Concept {concept_idx})")
    plt.axis('off')
    
    # 3. Overlay
    plt.subplot(1, 3, 3)
    plt.imshow(img_np)
    plt.imshow(saliency_map, cmap='jet', alpha=0.5) # Overlay with transparency
    if activation_val:
        plt.title(f"Overlay (Activation: {activation_val:.2f})")
    else:
        plt.title("Overlay")
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sae-path", type=str, required=True, help="Path to trained SAE checkpoint")
    parser.add_argument("--csv-path", type=str, required=True, help="Dataset CSV")
    parser.add_argument("--output-dir", type=str, default="visualizations/sae_saliency")
    parser.add_argument("--model-name", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--concept-indices", type=str, default=None, help="Comma-separated list of concept indices to visualize (e.g., '123,456')")
    parser.add_argument("--top-k-images", type=int, default=3, help="Number of top activating images to visualize per concept")
    parser.add_argument("--auto-pick-concepts", type=int, default=0, help="Automatically pick the top N concepts with highest activation")
    args = parser.parse_args()
    
    if not args.concept_indices and args.auto_pick_concepts == 0:
        print("Error: Must provide --concept-indices or --auto-pick-concepts")
        return
        
    target_concepts = []
    if args.concept_indices:
        try:
            target_concepts = [int(x.strip()) for x in args.concept_indices.split(',')]
        except ValueError:
            print("Error: --concept-indices must be a comma-separated list of integers")
            return

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Load Models
    print("Loading models...")
    backbone = AutoModel.from_pretrained(args.model_name)
    backbone.to(device).eval()
    
    # Freeze backbone parameters to save memory/compute for gradient calculation
    for param in backbone.parameters():
        param.requires_grad = False
        
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
    
    # Freeze SAE parameters
    for param in sae.parameters():
        param.requires_grad = False
    
    # 2. Collect Activations
    dataset = VizDataset(args.csv_path, args.model_name)
    dataloader = DataLoader(dataset, batch_size=32, num_workers=8, shuffle=False)
    
    if args.auto_pick_concepts > 0:
        print(f"Auto-picking top {args.auto_pick_concepts} concepts from dataset...")
    else:
        print(f"Finding top {args.top_k_images} images for concepts: {target_concepts}")
    
    # Store top K images for concepts
    # We use dense tensors to track ALL concepts efficiently during scan
    # top_vals: (hidden_dim, top_k_images)
    # top_inds: (hidden_dim, top_k_images)
    top_vals = torch.zeros(hidden_dim, args.top_k_images, device='cpu')
    top_inds = torch.zeros(hidden_dim, args.top_k_images, dtype=torch.long, device='cpu') - 1
    
    for images, indices in tqdm(dataloader, desc="Scanning dataset"):
        images = images.to(device)
        with torch.no_grad():
            vision_outputs = backbone.vision_model(pixel_values=images)
            pooler_output = vision_outputs.pooler_output if hasattr(vision_outputs, 'pooler_output') else vision_outputs[1]
            z = backbone.visual_projection(pooler_output) if hasattr(backbone, 'visual_projection') else pooler_output
            
            _, acts, _ = sae(z) # (B, hidden_dim)
            
        batch_acts = acts.cpu()
        batch_indices = indices.cpu()
        
        # Efficient update: Only process active neurons in this batch
        active_neurons = torch.where(batch_acts.sum(0) > 0)[0]
        
        # Filter if we have specific target concepts and NOT auto-picking
        if not args.auto_pick_concepts and target_concepts:
             # Intersect active_neurons with target_concepts
             # Convert target_concepts to set for O(1) lookup
             target_set = set(target_concepts)
             active_neurons = [n for n in active_neurons.tolist() if n in target_set]
             active_neurons = torch.tensor(active_neurons, dtype=torch.long)
        
        for neuron_idx in active_neurons:
            neuron_acts = batch_acts[:, neuron_idx]
            valid_mask = neuron_acts > 0
            
            if not valid_mask.any():
                continue
                
            valid_acts = neuron_acts[valid_mask]
            valid_img_indices = batch_indices[valid_mask]
            
            # Merge with existing top K
            current_vals = top_vals[neuron_idx]
            current_inds = top_inds[neuron_idx]
            
            # Filter out -1s from current
            valid_current = current_inds != -1
            
            combined_vals = torch.cat([current_vals[valid_current], valid_acts])
            combined_inds = torch.cat([current_inds[valid_current], valid_img_indices])
            
            # Sort descending
            if len(combined_vals) > 0:
                sorted_vals, sort_idx = torch.sort(combined_vals, descending=True)
                
                k = args.top_k_images
                if len(sorted_vals) > k:
                    top_vals[neuron_idx] = sorted_vals[:k]
                    top_inds[neuron_idx] = combined_inds[sort_idx][:k]
                else:
                    # Pad
                    pad_len = k - len(sorted_vals)
                    top_vals[neuron_idx] = torch.cat([sorted_vals, torch.zeros(pad_len)])
                    top_inds[neuron_idx] = torch.cat([combined_inds[sort_idx], torch.zeros(pad_len, dtype=torch.long) - 1])

    # Post-Scan Selection
    if args.auto_pick_concepts > 0:
        # Pick concepts with highest max activation
        max_acts = top_vals[:, 0] # Max activation is at index 0
        _, top_concept_indices = torch.topk(max_acts, k=args.auto_pick_concepts)
        target_concepts = top_concept_indices.tolist()
        print(f"Auto-selected top concepts: {target_concepts}")
    
    # 3. Generate Saliency Maps (Occlusion Sensitivity)
    print("Generating saliency maps using Occlusion Sensitivity...")
    print("Note: This may take longer than gradient-based saliency as it requires multiple forward passes.")
    
    # No gradients needed for occlusion
    # Ensure models are in eval mode and no gradients
    backbone.eval()
    sae.eval()
    
    for concept_idx in target_concepts:
        vals = top_vals[concept_idx]
        inds = top_inds[concept_idx]
        
        # Filter valid (non -1)
        valid_mask = inds != -1
        vals = vals[valid_mask]
        inds = inds[valid_mask]
        
        if len(vals) == 0:
            print(f"No activations found for concept {concept_idx}")
            continue
            
        print(f"Processing Concept {concept_idx}...")
        
        for i, (val, idx) in enumerate(zip(vals, inds)):
            idx = idx.item()
            val = val.item()
            
            # Get image tensor again (with grad requirement handled in function)
            # But we need the tensor from dataset
            tensor, _ = dataset[idx] # (3, H, W)
            tensor = tensor.to(device)
            
            row = dataset.df.iloc[idx]
            img_path = row['image_path']
            
            # Calculate Saliency (using Occlusion Sensitivity now)
            # Using stride=16 (patch size) and window=32 (2x2 patches) for good granularity
            saliency = get_occlusion_heatmap(tensor, backbone, sae, concept_idx, stride=16, window_size=32)
            
            # Save
            save_name = f"concept_{concept_idx}_top_{i+1}_saliency.png"
            save_path = out_dir / save_name
            
            visualize_overlay(img_path, saliency, save_path, concept_idx, val)
            
    print(f"Saliency maps saved to {out_dir}")

if __name__ == "__main__":
    main()
