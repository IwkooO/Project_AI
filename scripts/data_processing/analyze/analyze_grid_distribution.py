#!/usr/bin/env python3
"""
Analyze distribution of images per 1°x1° grid cell.
"""
import json
from pathlib import Path
from collections import defaultdict
import numpy as np
import argparse

def analyze_grid_distribution(metas_dir):
    """
    Analyze how many images are in each 1°x1° grid cell.
    """
    metas_path = Path(metas_dir)
    
    if not metas_path.exists():
        raise ValueError(f"Directory does not exist: {metas_dir}")
    
    # Group by grid cell
    cell_to_images = defaultdict(int)
    cell_to_metas = defaultdict(list)
    
    json_files = list(metas_path.glob("*.json"))
    print(f"Found {len(json_files)} meta JSON files")
    
    missing_coords = 0
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            lat = data.get('lat')
            lng = data.get('lng')
            
            if lat is None or lng is None:
                missing_coords += 1
                continue
            
            # Map to 1°x1° grid cell
            cell = (int(np.floor(lat)), int(np.floor(lng)))
            
            images = data.get('images', [])
            num_images = len(images) if isinstance(images, list) else 0
            
            cell_to_images[cell] += num_images
            cell_to_metas[cell].append({
                'file': json_file,
                'num_images': num_images
            })
            
        except Exception as e:
            print(f"Warning: Error reading {json_file}: {e}")
            continue
    
    if missing_coords:
        print(f"Warning: {missing_coords} files missing coordinates")
    
    # Calculate statistics
    cell_counts = list(cell_to_images.values())
    
    if not cell_counts:
        print("No cells with valid coordinates found!")
        return None, None
    
    print("\n" + "="*80)
    print("GRID CELL DISTRIBUTION (1°x1° cells)")
    print("="*80)
    print(f"\nTotal grid cells with data: {len(cell_to_images)}")
    print(f"Total images across all cells: {sum(cell_counts):,}")
    print(f"\nStatistics per cell:")
    print(f"  Min images per cell: {min(cell_counts)}")
    print(f"  Max images per cell: {max(cell_counts):,}")
    print(f"  Mean images per cell: {np.mean(cell_counts):.1f}")
    print(f"  Median images per cell: {np.median(cell_counts):.1f}")
    print(f"  75th percentile: {np.percentile(cell_counts, 75):.1f}")
    print(f"  90th percentile: {np.percentile(cell_counts, 90):.1f}")
    print(f"  95th percentile: {np.percentile(cell_counts, 95):.1f}")
    print(f"  99th percentile: {np.percentile(cell_counts, 99):.1f}")
    
    # Show top cells
    sorted_cells = sorted(cell_to_images.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 grid cells by image count:")
    print(f"  {'Cell (lat, lng)':<20} {'Images':<12} {'Meta Files':<12}")
    print(f"  {'-'*20} {'-'*12} {'-'*12}")
    for (lat, lng), img_count in sorted_cells[:20]:
        meta_count = len(cell_to_metas[(lat, lng)])
        print(f"  ({lat:3d}, {lng:4d}){'':<8} {img_count:<12,} {meta_count:<12}")
    
    # Distribution histogram
    print(f"\nDistribution breakdown:")
    bins = [0, 10, 50, 100, 500, 1000, 5000, float('inf')]
    labels = ['0-10', '10-50', '50-100', '100-500', '500-1000', '1000-5000', '5000+']
    
    for i, (low, high) in enumerate(zip(bins[:-1], bins[1:])):
        if high == float('inf'):
            count = sum(1 for c in cell_counts if c >= low)
        else:
            count = sum(1 for c in cell_counts if low <= c < high)
        pct = 100 * count / len(cell_counts) if cell_counts else 0
        print(f"  {labels[i]:<15} {count:>5} cells ({pct:5.1f}%)")
    
    print("\n" + "="*80)
    
    return cell_to_images, cell_to_metas

def main():
    parser = argparse.ArgumentParser(
        description="Analyze distribution of images per 1°x1° grid cell"
    )
    parser.add_argument(
        "--metas-dir",
        type=str,
        default="data/6921d7831744c5356b098bf7/metas",
        help="Path to directory containing meta JSON files"
    )
    
    args = parser.parse_args()
    
    analyze_grid_distribution(args.metas_dir)

if __name__ == "__main__":
    main()

