#!/usr/bin/env python3
"""
Create a spatially balanced dataset by sampling locations based on grid cells.
Creates a new directory with the same structure as the original.
"""
import json
import shutil
import argparse
import random
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def create_balanced_dataset(source_dir, output_dir, max_per_cell=50, seed=42):
    """
    Create a balanced dataset by sampling locations per 1°x1° grid cell.
    """
    random.seed(seed)
    source_path = Path(source_dir)
    output_path = Path(output_dir)
    
    if not source_path.exists():
        raise ValueError(f"Source directory does not exist: {source_dir}")
    
    # Setup output directory
    if output_path.exists():
        print(f"Warning: Output directory {output_dir} already exists.")
        # Ask for confirmation or just proceed? For now, proceed but don't delete
    else:
        output_path.mkdir(parents=True)
        
    metas_source = source_path / "metas"
    metas_output = output_path / "metas"
    metas_output.mkdir(exist_ok=True)
    
    # Copy base files (locations, metadata)
    print("Copying base files...")
    for file in source_path.glob("*.json"):
        shutil.copy2(file, output_path / file.name)
    
    # 1. Group metas by grid cell
    print("Analyzing spatial distribution...")
    cell_to_metas = defaultdict(list)
    meta_files = list(metas_source.glob("*.json"))
    
    valid_metas = 0
    skipped_metas = 0
    
    for meta_file in tqdm(meta_files, desc="Grouping files"):
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lat = data.get('lat')
            lng = data.get('lng')
            
            if lat is None or lng is None:
                skipped_metas += 1
                continue
                
            cell = (int(np.floor(lat)), int(np.floor(lng)))
            cell_to_metas[cell].append(meta_file)
            valid_metas += 1
            
        except Exception as e:
            print(f"Warning: Error reading {meta_file}: {e}")
            skipped_metas += 1
            continue
            
    print(f"\nFound {valid_metas} valid meta files across {len(cell_to_metas)} grid cells.")
    if skipped_metas > 0:
        print(f"Skipped {skipped_metas} files (missing coords or errors).")
        
    # 2. Sample and Copy
    print(f"\nSampling (max {max_per_cell} per cell) and copying...")
    
    total_copied = 0
    
    for cell, files in tqdm(cell_to_metas.items(), desc="Processing cells"):
        # Shuffle to ensure random sampling if we need to cut down
        if len(files) > max_per_cell:
            random.shuffle(files)
            selected_files = files[:max_per_cell]
        else:
            selected_files = files
            
        # Copy selected files
        for file_path in selected_files:
            shutil.copy2(file_path, metas_output / file_path.name)
            total_copied += 1
            
    print("\n" + "="*80)
    print("BALANCED DATASET CREATED")
    print("="*80)
    print(f"Source: {source_dir}")
    print(f"Output: {output_dir}")
    print(f"Max locations per cell: {max_per_cell}")
    print(f"\nOriginal locations: {len(meta_files)}")
    print(f"Balanced locations: {total_copied}")
    print(f"Reduction: {len(meta_files) - total_copied} locations ({(len(meta_files) - total_copied)/len(meta_files)*100:.1f}%)")
    print("="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Create a spatially balanced dataset"
    )
    parser.add_argument(
        "--source-dir",
        type=str,
        required=True,
        help="Path to original dataset directory (containing metas/)"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        required=True,
        help="Path to output directory for balanced dataset"
    )
    parser.add_argument(
        "--max-per-cell",
        type=int,
        default=50,
        help="Maximum number of locations per 1°x1° grid cell"
    )
    
    args = parser.parse_args()
    
    create_balanced_dataset(
        args.source_dir,
        args.output_dir,
        args.max_per_cell
    )

if __name__ == "__main__":
    main()

