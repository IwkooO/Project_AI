#!/usr/bin/env python3
"""
Visualize multiple images per concept to check if the model predicts 
similar coordinates for the same concept across different images.
"""
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path
import sys
from collections import defaultdict
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset_cbm import CBMDataset, create_stratified_splits, get_transforms
from src.models.cbm import GeoCBM
from src.utils.loss import HaversineLoss

def denormalize(tensor):
    """
    Denormalize tensor image (C, H, W) to numpy array (H, W, C) in [0, 1].
    """
    mean = torch.tensor((0.48145466, 0.4578275, 0.40821073)).view(3, 1, 1)
    std = torch.tensor((0.26862954, 0.26130258, 0.27577711)).view(3, 1, 1)
    
    tensor = tensor.cpu() * std + mean
    img = tensor.permute(1, 2, 0).numpy()
    img = np.clip(img, 0, 1)
    return img

def visualize_concept_group(model, dataset, concept_name, indices, device, output_path, haversine_fn, images_per_row=4, batch_size=8):
    """
    Visualize multiple images of the same concept with their predictions.
    Uses batched inference for speed.
    
    Args:
        model: Trained CBM model
        dataset: CBMDataset instance
        concept_name: Name of the concept
        indices: List of dataset indices for this concept
        device: torch device
        output_path: Path to save visualization
        haversine_fn: Haversine loss function for distance calculation
        images_per_row: Number of images per row in the grid
        batch_size: Batch size for model inference
    """
    model.eval()
    
    num_images = len(indices)
    num_rows = (num_images + images_per_row - 1) // images_per_row
    
    # Load all images and labels first
    images_list = []
    concept_labels_list = []
    country_labels_list = []
    coords_list = []
    
    for idx in indices:
        image, concept_label, country_label, coords = dataset[idx]
        images_list.append(image)
        concept_labels_list.append(concept_label)
        country_labels_list.append(country_label)
        coords_list.append(coords)
    
    # Batch inference
    all_concept_logits = []
    all_country_logits = []
    all_coord_preds = []
    
    with torch.no_grad():
        for i in range(0, num_images, batch_size):
            batch_indices = indices[i:i+batch_size]
            batch_images = torch.stack([images_list[j] for j in range(i, min(i+batch_size, num_images))]).to(device)
            
            concept_logits, country_logits, coord_preds = model(batch_images, return_attentions=False)
            
            all_concept_logits.append(concept_logits.cpu())
            all_country_logits.append(country_logits.cpu())
            all_coord_preds.append(coord_preds.cpu())
    
    # Concatenate all predictions
    all_concept_logits = torch.cat(all_concept_logits, dim=0)
    all_country_logits = torch.cat(all_country_logits, dim=0)
    all_coord_preds = torch.cat(all_coord_preds, dim=0)
    
    # Process predictions and create visualization
    fig = plt.figure(figsize=(20, 5 * num_rows))
    
    pred_coords_list = []
    gt_coords_list = []
    distances = []
    
    for i, idx in enumerate(indices):
        image = images_list[i]
        concept_label = concept_labels_list[i]
        country_label = country_labels_list[i]
        coords = coords_list[i]
        
        # Get predictions for this image
        concept_logits = all_concept_logits[i:i+1]
        country_logits = all_country_logits[i:i+1]
        coord_preds = all_coord_preds[i:i+1]
        
        concept_probs = torch.softmax(concept_logits, dim=1)
        top_concept_idx = concept_logits.argmax(dim=1).item()
        pred_concept = dataset.idx_to_concept[top_concept_idx]
        pred_concept_prob = concept_probs[0, top_concept_idx].item()
        
        country_idx = country_logits.argmax(dim=1).item()
        pred_country = dataset.idx_to_country[country_idx]
        
        pred_lat, pred_lon = coord_preds[0].numpy()
        true_lat, true_lon = coords.numpy()
        
        # Coordinates are already in degrees (raw), no denormalization needed
        # Calculate distance (HaversineLoss expects degrees)
        pred_coords_tensor = coord_preds
        true_coords_tensor = coords.unsqueeze(0)
        dist_km = haversine_fn(pred_coords_tensor, true_coords_tensor).item()
        
        pred_coords_list.append((pred_lat, pred_lon))
        gt_coords_list.append((true_lat, true_lon))
        distances.append(dist_km)
        
        # Plot image
        row = i // images_per_row
        col = i % images_per_row
        ax = plt.subplot(num_rows, images_per_row, i + 1)
        
        img_np = denormalize(image)
        ax.imshow(img_np)
        ax.axis('off')
        
        # Title with key info
        title = f"Img {i+1}\n"
        title += f"Pred: {pred_concept[:20]} ({pred_concept_prob:.2f})\n"
        title += f"GT: {dataset.idx_to_concept[concept_label.item()]}\n"
        title += f"Pred Coords: ({pred_lat:.2f}, {pred_lon:.2f})\n"
        title += f"GT Coords: ({true_lat:.2f}, {true_lon:.2f})\n"
        title += f"Error: {dist_km:.1f}km"
        
        ax.set_title(title, fontsize=8)
    
    plt.suptitle(f"Concept: {concept_name} | {num_images} images | "
                 f"Avg Coord Error: {np.mean(distances):.1f}km | "
                 f"Std Coord Error: {np.std(distances):.1f}km", 
                 fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    # Calculate coordinate consistency metrics
    pred_coords_array = np.array(pred_coords_list)
    gt_coords_array = np.array(gt_coords_list)
    
    pred_std_lat = np.std(pred_coords_array[:, 0])
    pred_std_lon = np.std(pred_coords_array[:, 1])
    gt_std_lat = np.std(gt_coords_array[:, 0])
    gt_std_lon = np.std(gt_coords_array[:, 1])
    
    print(f"Concept: {concept_name}")
    print(f"  Images: {num_images}")
    print(f"  Avg Error: {np.mean(distances):.1f}km (std: {np.std(distances):.1f}km)")
    print(f"  Pred Coord Std: Lat={pred_std_lat:.2f}°, Lon={pred_std_lon:.2f}°")
    print(f"  GT Coord Std: Lat={gt_std_lat:.2f}°, Lon={gt_std_lon:.2f}°")
    print(f"  Saved to: {output_path}\n")

def main():
    parser = argparse.ArgumentParser(
        description="Visualize concept consistency - check if same concept predicts similar coordinates"
    )
    parser.add_argument(
        "--model-path", 
        type=str, 
        default="/home/igodzwon/Project_AI/checkpoints/cbm-finetuned/best_cbm_model.pth",
        help="Path to trained CBM model"
    )
    parser.add_argument(
        "--csv-path", 
        type=str, 
        default="/home/igodzwon/Project_AI/data/691df1ee911f74393c53af8c/dataset.csv",
        help="Path to dataset CSV"
    )
    parser.add_argument(
        "--output-dir", 
        type=str, 
        default="visualizations/concept_consistency",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--num-concepts", 
        type=int, 
        default=10,
        help="Number of concepts to visualize"
    )
    parser.add_argument(
        "--images-per-concept", 
        type=int, 
        default=8,
        help="Number of images to show per concept"
    )
    parser.add_argument(
        "--images-per-row",
        type=int,
        default=4,
        help="Number of images per row in the grid"
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=['train', 'val', 'test'],
        default='test',
        help="Which split to use for visualization"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for model inference (larger = faster)"
    )
    
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load checkpoint
    print(f"Loading model from {args.model_path}")
    checkpoint = torch.load(args.model_path, map_location=device)
    
    state_dict = checkpoint['model_state_dict']
    concepts = checkpoint['concepts']
    countries = checkpoint['countries']
    
    # Initialize model
    model = GeoCBM(
        num_concepts=len(concepts),
        num_countries=len(countries),
        model_name="geolocal/StreetCLIP",
        freeze_backbone=True
    )
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()
    
    # Prepare dataset
    print(f"Loading {args.split} split...")
    train_df, val_df, test_df = create_stratified_splits(args.csv_path)
    
    if args.split == 'train':
        df = train_df
    elif args.split == 'val':
        df = val_df
    else:
        df = test_df
    
    transform = get_transforms()
    dataset = CBMDataset(df, transform=transform)
    
    print(f"Dataset size: {len(dataset)}")
    
    # Group indices by concept using dataframe (much faster - no image loading)
    # Dataset resets index, so indices are 0-based and match dataframe after reset
    print("Grouping images by concept...")
    df_reset = df.reset_index(drop=True)
    concept_to_indices = defaultdict(list)
    for idx, row in df_reset.iterrows():
        concept_name = row['meta_name']
        concept_to_indices[concept_name].append(idx)
    
    # Filter concepts that have enough images
    concept_to_indices = {
        k: v for k, v in concept_to_indices.items() 
        if len(v) >= args.images_per_concept
    }
    
    print(f"Found {len(concept_to_indices)} concepts with at least {args.images_per_concept} images")
    
    # Select top N concepts by number of images (or random)
    concepts_to_visualize = sorted(
        concept_to_indices.items(), 
        key=lambda x: len(x[1]), 
        reverse=True
    )[:args.num_concepts]
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Haversine for distance calculation
    haversine = HaversineLoss()
    
    # Visualize each concept group
    import random
    for concept_name, indices in tqdm(concepts_to_visualize, desc="Visualizing concepts"):
        # Sample random images for this concept
        if len(indices) > args.images_per_concept:
            sampled_indices = random.sample(indices, args.images_per_concept)
        else:
            sampled_indices = indices
        
        output_path = output_dir / f"concept_{concept_name.replace('/', '_').replace(' ', '_')}.png"
        
        visualize_concept_group(
            model=model,
            dataset=dataset,
            concept_name=concept_name,
            indices=sampled_indices,
            device=device,
            output_path=output_path,
            haversine_fn=haversine,
            images_per_row=args.images_per_row,
            batch_size=args.batch_size
        )
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()

