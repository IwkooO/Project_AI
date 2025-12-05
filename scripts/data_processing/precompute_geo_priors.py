#!/usr/bin/env python3
"""
Precompute S2 cells and concept geographic priors.

This script:
1. Maps every training image to an S2 cell (coarse location).
2. Builds a vocabulary of S2 cells present in the training set.
3. Computes P(cell | concept) for every concept.
4. Saves metadata for the CBM model.

Usage:
    python scripts/data_processing/precompute_geo_priors.py \
        --train-csv data/.../dataset_train.csv \
        --output-dir data/concept_data \
        --s2-level 8
"""

import argparse
import pandas as pd
import s2sphere
import torch
import json
import numpy as np
from pathlib import Path
from collections import Counter, defaultdict
from tqdm import tqdm


def lat_lng_to_cell_id(lat, lng, level=8):
    """Convert lat/lng to S2 cell ID token."""
    p1 = s2sphere.LatLng.from_degrees(lat, lng)
    cell = s2sphere.CellId.from_lat_lng(p1).parent(level)
    return cell.to_token()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True, help="Path to training CSV with 'generalized' column")
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory to save priors and metadata")
    parser.add_argument("--s2-level", type=int, default=8, help="S2 cell level (default: 8)")
    parser.add_argument("--min-samples-per-cell", type=int, default=1, help="Minimum samples to keep a cell (default: 1)")
    args = parser.parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"Reading {args.train_csv}...")
    df = pd.read_csv(args.train_csv)
    
    if 'generalized' not in df.columns:
        raise ValueError("CSV must have 'generalized' column. Run map_generalized_concepts.py first.")
    
    # Filter missing concepts
    df = df.dropna(subset=['generalized', 'lat', 'lng'])
    print(f"Processing {len(df)} samples...")

    # 1. Build Concept Vocabulary
    print("Building concept vocabulary...")
    concepts = sorted(df['generalized'].unique())
    concept_to_idx = {c: i for i, c in enumerate(concepts)}
    idx_to_concept = {i: c for i, c in enumerate(concepts)}
    
    print(f"Found {len(concepts)} unique concepts.")

    # 2. Map to S2 Cells
    print(f"Mapping to S2 cells (Level {args.s2_level})...")
    cell_tokens = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        token = lat_lng_to_cell_id(row['lat'], row['lng'], args.s2_level)
        cell_tokens.append(token)
    
    df['s2_token'] = cell_tokens

    # 3. Build Cell Vocabulary (filter rare cells if needed)
    cell_counts = Counter(cell_tokens)
    unique_cells = [c for c, count in cell_counts.items() if count >= args.min_samples_per_cell]
    unique_cells.sort()
    
    cell_to_idx = {c: i for i, c in enumerate(unique_cells)}
    idx_to_cell = {i: c for i, c in enumerate(unique_cells)}
    
    print(f"Found {len(unique_cells)} unique S2 cells (min samples: {args.min_samples_per_cell}).")
    print(f"Coverage: {sum(cell_counts[c] for c in unique_cells) / len(df):.2%} of data kept.")

    # 4. Compute Concept Priors P(cell | concept)
    print("Computing concept priors...")
    num_concepts = len(concepts)
    num_cells = len(unique_cells)
    
    # Count matrix: [Concepts, Cells]
    counts = torch.zeros((num_concepts, num_cells), dtype=torch.float32)
    
    skipped = 0
    for _, row in tqdm(df.iterrows(), total=len(df)):
        c_name = row['generalized']
        c_token = row['s2_token']
        
        if c_token in cell_to_idx:
            c_idx = concept_to_idx[c_name]
            cell_idx = cell_to_idx[c_token]
            counts[c_idx, cell_idx] += 1
        else:
            skipped += 1
            
    # Normalize to get probabilities
    # Add epsilon to avoid division by zero for unused concepts (though unlikely)
    priors = counts / (counts.sum(dim=1, keepdim=True) + 1e-8)
    
    # 5. Save Outputs
    print("Saving outputs...")
    
    # Concept Vocab
    with open(args.output_dir / "concept_vocab.json", "w") as f:
        json.dump({
            "concept_to_idx": concept_to_idx,
            "idx_to_concept": idx_to_concept,
            "num_concepts": num_concepts
        }, f, indent=2)
        
    # Cell Vocab
    with open(args.output_dir / "s2_cells.json", "w") as f:
        json.dump({
            "cell_to_idx": cell_to_idx,
            "idx_to_cell": idx_to_cell,
            "num_cells": num_cells,
            "level": args.s2_level
        }, f, indent=2)
        
    # Priors Tensor
    torch.save(priors, args.output_dir / "concept_priors.pt")
    
    # Save counts for debugging or weighting
    torch.save(counts, args.output_dir / "concept_cell_counts.pt")

    print(f"\nDone! Saved to {args.output_dir}")
    print(f" - concept_vocab.json")
    print(f" - s2_cells.json")
    print(f" - concept_priors.pt ({priors.shape})")

if __name__ == "__main__":
    main()

