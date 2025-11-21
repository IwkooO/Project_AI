#!/usr/bin/env python3
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
import cv2
from PIL import Image

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset_cbm import CBMDataset, create_stratified_splits, get_transforms
from src.models.cbm import GeoCBM
from src.utils.loss import HaversineLoss

def denormalize(tensor):
    """
    Denormalize tensor image (C, H, W) to numpy array (H, W, C) in [0, 1].
    CLIP Mean: (0.48145466, 0.4578275, 0.40821073)
    CLIP Std: (0.26862954, 0.26130258, 0.27577711)
    """
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    
    tensor = tensor.cpu() * std + mean
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img

def process_attention(attentions, image_size):
    """
    Extract attention map from last layer, CLS token.
    attentions: tuple of (B, Heads, SeqLen, SeqLen)
    """
    # Get last layer attention
    last_layer_attn = attentions[-1] # (B, Heads, SeqLen, SeqLen)
    
    # Get CLS token attention to all other tokens (first token is CLS)
    # We average across heads
    cls_attn = last_layer_attn[:, :, 0, 1:] # (B, Heads, NumPatches)
    cls_attn = cls_attn.mean(dim=1) # (B, NumPatches)
    
    # Reshape to grid
    num_patches = cls_attn.shape[-1]
    grid_size = int(np.sqrt(num_patches))
    cls_attn = cls_attn.view(-1, grid_size, grid_size)
    
    # Resize to image size
    # Use unsqueeze to add channel dim for interpolation
    cls_attn = cls_attn.unsqueeze(1) # (B, 1, H, W)
    cls_attn = F.interpolate(cls_attn, size=image_size, mode='bicubic', align_corners=False)
    
    # Squeeze back
    cls_attn = cls_attn.squeeze(1) # (B, H, W)
    
    # Normalize to [0, 1] for visualization
    cls_attn = (cls_attn - cls_attn.min()) / (cls_attn.max() - cls_attn.min())
    
    return cls_attn

def visualize_sample(model, dataset, idx, device, output_path, haversine_fn):
    image, concept_label, country_label, coords = dataset[idx]
    
    # Prepare input
    image_tensor = image.unsqueeze(0).to(device) # (1, C, H, W)
    
    # Inference
    with torch.no_grad():
        (concept_logits, country_logits, coord_preds), attentions = model(image_tensor, return_attentions=True)
        
    # Decode Predictions
    # 1. Concepts (Top 3)
    concept_probs = torch.softmax(concept_logits, dim=1)
    top3_probs, top3_indices = torch.topk(concept_probs, k=3)
    top3_concepts = [(dataset.idx_to_concept[idx.item()], prob.item()) for idx, prob in zip(top3_indices[0], top3_probs[0])]
    
    # 2. Country
    country_idx = country_logits.argmax(dim=1).item()
    pred_country = dataset.idx_to_country[country_idx]
    
    # 3. Coordinates
    pred_lat, pred_lon = coord_preds[0].cpu().numpy()
    true_lat, true_lon = coords.numpy()
    dist_km = haversine_fn(coord_preds.cpu(), coords.unsqueeze(0)).item()
    
    # 4. Attention Map
    attn_map = process_attention(attentions, (image.shape[1], image.shape[2]))
    attn_map = attn_map[0].cpu().numpy()
    
    # Visualization Plot
    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    
    # Original Image with Attention Overlay
    img_np = denormalize(image)
    axes[0].imshow(img_np)
    axes[0].imshow(attn_map, cmap='jet', alpha=0.4) # Overlay
    axes[0].axis('off')
    axes[0].set_title("Attention Map Overlay")
    
    # Info Panel
    axes[1].axis('off')
    
    # Ground Truth Info
    gt_concept = dataset.idx_to_concept[concept_label.item()]
    gt_country = dataset.idx_to_country[country_label.item()]
    
    info_text = f"GROUND TRUTH:\n"
    info_text += f"Concept: {gt_concept}\n"
    info_text += f"Country: {gt_country}\n"
    info_text += f"Coords: {true_lat:.4f}, {true_lon:.4f}\n\n"
    
    info_text += f"PREDICTIONS:\n"
    info_text += f"Coords: {pred_lat:.4f}, {pred_lon:.4f}\n"
    info_text += f"Error: {dist_km:.1f} km\n"
    info_text += f"Country: {pred_country} ({'✓' if pred_country == gt_country else '✗'})\n\n"
    
    info_text += f"Top 3 Concepts:\n"
    for concept, prob in top3_concepts:
        mark = "✓" if concept == gt_concept else ""
        info_text += f"- {concept}: {prob:.2f} {mark}\n"
        
    axes[1].text(0.05, 0.95, info_text, fontsize=12, verticalalignment='top', fontfamily='monospace')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    print(f"Saved visualization to {output_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=str, required=True, help="Path to trained CBM model")
    parser.add_argument("--csv-path", type=str, default="dataset.csv")
    parser.add_argument("--output-dir", type=str, default="visualizations/cbm")
    parser.add_argument("--num-samples", type=int, default=10)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load Checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    concepts = checkpoint['concepts']
    countries = checkpoint['countries']
    
    # Initialize Model
    model = GeoCBM(
        num_concepts=len(concepts),
        num_countries=len(countries),
        model_name="geolocal/StreetCLIP",
        freeze_backbone=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Prepare Dataset (Test Split)
    _, _, test_df = create_stratified_splits(args.csv_path)
    transform = get_transforms()
    dataset = CBMDataset(test_df, transform=transform)
    
    # Output Dir
    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    # Haversine for distance calc
    haversine = HaversineLoss()
    
    # Select random samples
    indices = torch.randperm(len(dataset))[:args.num_samples].tolist()
    
    for i, idx in enumerate(indices):
        out_path = out_dir / f"sample_{i}_{idx}.png"
        visualize_sample(model, dataset, idx, device, out_path, haversine)

if __name__ == "__main__":
    main()

