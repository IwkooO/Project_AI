#!/usr/bin/env python3
"""
Test the best CBM model on the test CSV and create visualizations
showing panoramic images with concept activation bar charts.
"""
import argparse
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pathlib import Path
import sys
from PIL import Image
from tqdm import tqdm

# Add project root to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.append(str(project_root))

from src.data.dataset_cbm import CBMDataset, get_transforms
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

def create_test_dataset_from_csv(csv_path, transform, train_concepts, train_countries):
    """
    Create a CBMDataset from a test CSV file, using training set's concept/country mappings.
    The CSV should have columns: image_path, meta_name, country, lat, lng
    
    Args:
        csv_path: Path to test CSV
        transform: Image transforms
        train_concepts: List of concepts from training set (used for model output decoding)
        train_countries: List of countries from training set (used for model output decoding)
    
    Returns:
        dataset: CBMDataset instance
        df: DataFrame
        train_concept_to_idx: Mapping from training concept names to indices
        train_country_to_idx: Mapping from training country names to indices
    """
    df = pd.read_csv(csv_path)
    
    # Filter out rows with missing critical data
    df = df.dropna(subset=['image_path', 'meta_name', 'country', 'lat', 'lng'])
    
    # Reset index to ensure indices match between df and dataset
    df = df.reset_index(drop=True)
    
    # Create dataset (this will create its own mappings, but we'll use training mappings for decoding)
    dataset = CBMDataset(df, transform=transform)
    
    # Create training set mappings for decoding model outputs
    train_concept_to_idx = {concept: idx for idx, concept in enumerate(train_concepts)}
    train_idx_to_concept = {idx: concept for concept, idx in train_concept_to_idx.items()}
    
    train_country_to_idx = {country: idx for idx, country in enumerate(train_countries)}
    train_idx_to_country = {idx: country for country, idx in train_country_to_idx.items()}
    
    # Add training mappings to dataset for easy access
    dataset.train_concept_to_idx = train_concept_to_idx
    dataset.train_idx_to_concept = train_idx_to_concept
    dataset.train_country_to_idx = train_country_to_idx
    dataset.train_idx_to_country = train_idx_to_country
    
    return dataset, df

def visualize_test_samples(model, dataset, df, indices, device, output_path, haversine_fn, 
                          images_per_figure=4, top_k_concepts=5):
    """
    Visualize test samples with panoramic images and concept activation bar charts.
    
    Args:
        model: Trained CBM model
        dataset: CBMDataset instance
        df: DataFrame with test data
        indices: List of dataset indices to visualize
        device: torch device
        output_path: Path to save visualization
        haversine_fn: Haversine loss function for distance calculation
        images_per_figure: Number of images per figure
        top_k_concepts: Number of top concepts to show in bar chart
    """
    model.eval()
    
    num_figures = (len(indices) + images_per_figure - 1) // images_per_figure
    
    for fig_idx in range(num_figures):
        start_idx = fig_idx * images_per_figure
        end_idx = min(start_idx + images_per_figure, len(indices))
        fig_indices = indices[start_idx:end_idx]
        
        # Create figure with subplots: one row per image, two columns (image + bar chart)
        fig, axes = plt.subplots(len(fig_indices), 2, figsize=(16, 4 * len(fig_indices)))
        
        if len(fig_indices) == 1:
            axes = axes.reshape(1, -1)
        
        # Batch inference for efficiency
        images_list = []
        coords_list = []
        gt_concept_names = []
        gt_country_names = []
        
        for idx in fig_indices:
            image, concept_label, country_label, coords = dataset[idx]
            images_list.append(image)
            coords_list.append(coords)
            gt_concept_names.append(dataset.idx_to_concept[concept_label.item()])
            gt_country_names.append(dataset.idx_to_country[country_label.item()])
        
        # Batch inference
        batch_images = torch.stack(images_list).to(device)
        
        with torch.no_grad():
            concept_logits, country_logits, coord_preds = model(batch_images)
        
        # Process each sample
        for i, idx in enumerate(fig_indices):
            row = df.iloc[idx]
            image = images_list[i]
            coords = coords_list[i]
            gt_concept = gt_concept_names[i]
            gt_country = gt_country_names[i]
            
            # Get predictions for this image
            concept_logit = concept_logits[i:i+1]
            country_logit = country_logits[i:i+1]
            coord_pred = coord_preds[i:i+1]
            
            # Concept probabilities
            concept_probs = torch.softmax(concept_logit, dim=1)
            
            # Get top K concepts (use training set mappings for decoding)
            top_k_probs, top_k_indices = torch.topk(concept_probs, k=min(top_k_concepts, len(dataset.train_idx_to_concept)))
            top_k_concept_names = [dataset.train_idx_to_concept[idx.item()] for idx in top_k_indices[0]]
            top_k_probs_list = top_k_probs[0].cpu().numpy()
            
            # Country prediction (use training set mappings)
            country_idx = country_logit.argmax(dim=1).item()
            pred_country = dataset.train_idx_to_country[country_idx]
            
            # Coordinates
            pred_lat, pred_lon = coord_pred[0].cpu().numpy()
            true_lat, true_lon = coords.numpy()
            
            # Calculate distance
            dist_km = haversine_fn(coord_pred.cpu(), coords.unsqueeze(0).cpu()).item()
            
            # Plot image
            ax_img = axes[i, 0]
            img_np = denormalize(image)
            ax_img.imshow(img_np)
            ax_img.axis('off')
            
            # Title with coordinates and error
            title = f"GT: {gt_country} ({true_lat:.2f}, {true_lon:.2f})\n"
            title += f"Pred (Ret): {pred_country} ({pred_lat:.2f}, {pred_lon:.2f})\n"
            title += f"Pred (Cls): {pred_country}\n"
            title += f"Error: {dist_km:.1f} km"
            ax_img.set_title(title, fontsize=10, fontweight='bold')
            
            # Plot bar chart
            ax_bar = axes[i, 1]
            
            # Prepare data for bar chart
            # Include GT concept even if not in top K
            all_concept_names = list(top_k_concept_names)
            all_probs = list(top_k_probs_list)
            colors = []
            
            # Check if GT concept is in top K
            gt_in_topk = gt_concept in all_concept_names
            
            if not gt_in_topk:
                # Add GT concept with its actual probability
                # Check if GT concept exists in training set
                if gt_concept in dataset.train_concept_to_idx:
                    gt_concept_idx = dataset.train_concept_to_idx[gt_concept]
                    gt_prob = concept_probs[0, gt_concept_idx].item()
                    all_concept_names.append(gt_concept)
                    all_probs.append(gt_prob)
                else:
                    # GT concept not in training set, skip it
                    print(f"Warning: GT concept '{gt_concept}' not found in training set concepts")
            
            # Create colors: orange for GT concept, blue for others
            for concept_name in all_concept_names:
                if concept_name == gt_concept:
                    colors.append('#FF8C00')  # Orange for GT
                else:
                    colors.append('#4169E1')  # Blue for others
            
            # Create horizontal bar chart
            y_pos = np.arange(len(all_concept_names))
            bars = ax_bar.barh(y_pos, all_probs, color=colors, alpha=0.8)
            
            # Set labels
            ax_bar.set_yticks(y_pos)
            ax_bar.set_yticklabels(all_concept_names, fontsize=9)
            ax_bar.set_xlabel('Activation Score', fontsize=10)
            ax_bar.set_xlim(0, max(all_probs) * 1.1)
            ax_bar.set_title(f'GT Concept: {gt_concept}', fontsize=10, fontweight='bold', color='#FF8C00')
            ax_bar.grid(axis='x', alpha=0.3)
            
            # Add value labels on bars
            for j, (bar, prob) in enumerate(zip(bars, all_probs)):
                width = bar.get_width()
                ax_bar.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                           f'{prob:.2f}', ha='left', va='center', fontsize=8)
        
        plt.suptitle(f'CBM Test Predictions - Figure {fig_idx + 1}/{num_figures}', 
                    fontsize=14, fontweight='bold', y=0.995)
        plt.tight_layout()
        
        # Save figure
        if num_figures > 1:
            save_path = output_path.parent / f"{output_path.stem}_fig{fig_idx + 1}{output_path.suffix}"
        else:
            save_path = output_path
        
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"Saved visualization to {save_path}")

def main():
    parser = argparse.ArgumentParser(
        description="Test CBM model on test CSV and create visualizations"
    )
    parser.add_argument(
        "--model-path",
        type=str,
        required=True,
        help="Path to trained CBM model checkpoint"
    )
    parser.add_argument(
        "--test-csv",
        type=str,
        required=True,
        help="Path to test CSV file"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="visualizations/test_predictions",
        help="Directory to save visualizations"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=20,
        help="Number of test samples to visualize"
    )
    parser.add_argument(
        "--images-per-figure",
        type=int,
        default=4,
        help="Number of images per figure"
    )
    parser.add_argument(
        "--top-k-concepts",
        type=int,
        default=5,
        help="Number of top concepts to show in bar chart"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=16,
        help="Batch size for inference"
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
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
    
    # Prepare dataset from test CSV (using training set's concept/country mappings)
    print(f"Loading test dataset from {args.test_csv}")
    transform = get_transforms()
    dataset, df = create_test_dataset_from_csv(args.test_csv, transform, concepts, countries)
    
    print(f"Test dataset size: {len(dataset)}")
    
    # Sample random indices
    np.random.seed(args.seed)
    if args.num_samples > len(dataset):
        print(f"Warning: Requested {args.num_samples} samples but only {len(dataset)} available. Using all samples.")
        indices = list(range(len(dataset)))
    else:
        indices = np.random.choice(len(dataset), size=args.num_samples, replace=False).tolist()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Haversine for distance calculation
    haversine = HaversineLoss()
    
    # Create visualizations
    output_path = output_dir / "test_predictions.png"
    print(f"\nCreating visualizations for {len(indices)} samples...")
    
    visualize_test_samples(
        model=model,
        dataset=dataset,
        df=df,
        indices=indices,
        device=device,
        output_path=output_path,
        haversine_fn=haversine,
        images_per_figure=args.images_per_figure,
        top_k_concepts=args.top_k_concepts
    )
    
    print(f"\nAll visualizations saved to: {output_dir}")

if __name__ == "__main__":
    main()

