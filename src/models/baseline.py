#!/usr/bin/env python3
"""
Baseline GeoCLIP script for evaluation on the test set.
Matches metrics from train_concept_aware.py.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm
import torchvision.transforms as transforms
import numpy as np
from geoclip import GeoCLIP
from pathlib import Path

from src.dataset import PanoramaCBMDataset, create_splits_stratified, SubsetDataset
from src.evaluation import haversine_distance, accuracy_within_threshold

def collate_batch(batch):
    """
    Custom collate function to handle variable-length metadata fields.
    Metadata dict contains 'images' which is a list of variable length.
    """
    images = torch.stack([item[0] for item in batch])
    concept_indices = torch.tensor([item[1] for item in batch], dtype=torch.long)
    target_indices = torch.tensor([item[2] for item in batch], dtype=torch.long)
    coordinates = torch.stack([item[3] for item in batch])
    metadata = [item[4] for item in batch]  # Keep as list of dicts, don't collate
    
    return images, concept_indices, target_indices, coordinates, metadata

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Baseline GeoCLIP Evaluation")
    parser.add_argument("--country", type=str, default=None, help="Filter for country (e.g. Australia)")
    parser.add_argument("--batch_size", type=int, default=32, help="Batch size")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    if args.country:
        print(f"Country filter: {args.country}")

    # Transforms for GeoCLIP (assuming CLIP-based normalization)
    # Standard CLIP mean/std
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.48145466, 0.4578275, 0.40821073],
            std=[0.26862954, 0.26130258, 0.27577711]
        )
    ])

    print("Initializing Dataset...")
    # Note: country=None implies global or whatever default is (sa-dataset.csv is loaded inside usually if not filtered)
    # We follow the baseline.py's original dataset init which didn't filter by country explicitly unless defaults changed.
    # However, train_concept_aware.py takes args.country_filter.
    # We will stick to the default dataset loading from baseline.py but with correct transforms.
    full_dataset = PanoramaCBMDataset(
        transform=transform,
        require_coordinates=True,
        country=args.country,
        use_normalized_coordinates=False # We want raw lat/lng for distance calculation against GeoCLIP which usually outputs lat/lng
    )

    print(f"Total samples: {len(full_dataset)}")

    # Create splits
    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples, train_ratio=0.7, val_ratio=0.2, test_ratio=0.1
    )
    
    test_dataset = SubsetDataset(full_dataset, test_samples)
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size, # Can increase batch size
        shuffle=False,
        num_workers=4,
        collate_fn=collate_batch
    )
    
    print(f"Test set size: {len(test_dataset)}")

    print("Loading GeoCLIP model...")
    model = GeoCLIP().to(device)
    model.eval()

    print("Evaluating on test set...")
    
    all_distances = []
    
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing"):
            images, _, _, coords, metadata = batch
            # images = images.to(device) # We don't use the tensor images for predict()
            coords = coords.to(device)
            
            # Use predict() which takes image paths
            # metadata is a list of dicts, each has 'image_path'
            
            batch_preds = []
            for meta in metadata:
                pano_id = str(meta['pano_id'])
                image_path = Path(f"data/6906237dc7731161a37282b2/panorama_processed/image_{pano_id}.jpg")
                # predict(self, image_path, top_k) -> returns (top_pred_gps, top_pred_prob)
                # We want top_k=1
                top_pred_gps, _ = model.predict(image_path, top_k=1)
                batch_preds.append(top_pred_gps)
            
            pred_gps = torch.stack(batch_preds).squeeze(1).to(device) # (B, 2)
            
            # Calculate haversine distance
            # haversine_distance expects tensors
            batch_distances = haversine_distance(pred_gps, coords)
            all_distances.append(batch_distances.cpu())

    # Compute Metrics
    if all_distances:
        all_distances_tensor = torch.cat(all_distances)
        
        # Mean/Median distance
        mean_dist = all_distances_tensor.mean().item()
        median_dist = all_distances_tensor.median().item()
        
        # Thresholds
        thresholds = {
            'street': 1.0,      # 1 km
            'city': 25.0,       # 25 km
            'region': 200.0,    # 200 km
            'country': 750.0,    # 750 km
            'continent': 2500.0  # 2500 km
        }
        
        threshold_accuracies = {}
        for level, threshold_km in thresholds.items():
            acc = accuracy_within_threshold(all_distances_tensor, threshold_km)
            threshold_accuracies[f'acc_{level}'] = acc
            
        print("\n=== Test Results ===")
        print(f"Mean Distance Error: {mean_dist:.4f} km")
        print(f"Median Distance Error: {median_dist:.4f} km")
        print("-" * 20)
        print(f"Street Acc (1km):       {threshold_accuracies['acc_street']:.4f}")
        print(f"City Acc (25km):        {threshold_accuracies['acc_city']:.4f}")
        print(f"Region Acc (200km):     {threshold_accuracies['acc_region']:.4f}")
        print(f"Country Acc (750km):    {threshold_accuracies['acc_country']:.4f}")
        print(f"Continent Acc (2500km): {threshold_accuracies['acc_continent']:.4f}")
        print("====================")
        
    else:
        print("No test samples found or evaluation failed.")

if __name__ == "__main__":
    main()
