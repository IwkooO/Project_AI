#!/usr/bin/env python3
"""
Analyze spatial distribution of black images to check if they're clustered.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict
import numpy as np
from tqdm import tqdm

def analyze_black_images(metas_dir, processed_images_dir, log_file):
    """
    Analyze where black images are located geographically.
    """
    metas_path = Path(metas_dir)
    processed_path = Path(processed_images_dir)
    
    # Read the log file to get black images
    black_images = []
    if log_file:
        log_path = Path(log_file)
        if log_path.exists():
            with open(log_path, 'r') as f:
                for line in f:
                    # Parse lines like: "Warning: image_ABC123.jpg - completely black"
                    if 'Warning:' in line and 'completely black' in line:
                        # Extract image name
                        parts = line.split('Warning:')
                        if len(parts) > 1:
                            img_part = parts[1].strip()
                            # Extract just the filename
                            if 'image_' in img_part:
                                img_name = img_part.split()[0]  # Get first word (the filename)
                                black_images.append(img_name)
        else:
            print(f"Warning: Log file not found: {log_file}")
    
    print(f"Found {len(black_images)} black images from log")
    
    # Extract pano_id from image filename
    black_pano_ids = []
    for img in black_images:
        if img.startswith('image_') and img.endswith('.jpg'):
            pano_id = img[6:-4]  # Remove 'image_' and '.jpg'
            black_pano_ids.append(pano_id)
    
    print(f"Extracted {len(black_pano_ids)} pano IDs")
    
    # Group black images by grid cell
    black_cells = defaultdict(int)
    total_cells = defaultdict(int)
    
    meta_files = list(metas_path.glob("*.json"))
    print(f"Processing {len(meta_files)} meta files...")
    
    for i, meta_file in tqdm(enumerate(meta_files), total=len(meta_files), desc="Processing meta files"):
        
        try:
            with open(meta_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            pano_id = meta_file.stem
            lat = data.get('lat')
            lng = data.get('lng')
            
            if lat is None or lng is None:
                continue
                
            cell = (int(np.floor(lat)), int(np.floor(lng)))
            total_cells[cell] += 1
            
            if pano_id in black_pano_ids:
                black_cells[cell] += 1
                
        except Exception as e:
            continue
    
    # Calculate statistics
    black_cell_count = len([c for c in black_cells.values() if c > 0])
    total_locations = sum(total_cells.values())
    total_black = sum(black_cells.values())
    
    print(f"\n{'='*80}")
    print("BLACK IMAGE SPATIAL ANALYSIS")
    print(f"{'='*80}")
    print(f"\nTotal locations: {total_locations:,}")
    print(f"Total black images: {total_black:,}")
    print(f"Black image percentage: {100*total_black/total_locations:.1f}%")
    print(f"Grid cells with black images: {black_cell_count}")
    print(f"Total grid cells: {len(total_cells)}")
    print(f"Percentage of cells with black images: {100*black_cell_count/len(total_cells):.1f}%")
    
    # Show top cells with most black images
    sorted_black_cells = sorted(black_cells.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\nTop 20 grid cells by BLACK image count:")
    print(f"  {'Cell (lat, lng)':<20} {'Black':<10} {'Total':<10} {'Black %':<10}")
    print(f"  {'-'*20} {'-'*10} {'-'*10} {'-'*10}")
    
    for (lat, lng), black_count in sorted_black_cells[:20]:
        total_in_cell = total_cells[(lat, lng)]
        pct = 100 * black_count / total_in_cell if total_in_cell > 0 else 0
        print(f"  ({lat:3d}, {lng:4d}){'':<8} {black_count:<10} {total_in_cell:<10} {pct:<9.1f}%")
    
    # Calculate clustering metrics
    # 1. Gini coefficient for black image distribution
    cell_black_counts = list(black_cells.values())
    if cell_black_counts:
        # Gini calculation
        sorted_counts = np.sort(cell_black_counts)
        n = len(sorted_counts)
        cumsum = np.cumsum(sorted_counts)
        gini = (n + 1 - 2 * np.sum(cumsum) / cumsum[-1]) / n
        print(f"\nClustering metric - Gini coefficient: {gini:.3f}")
        if gini > 0.6:
            print("HIGH clustering: Black images are heavily concentrated in few cells")
        elif gini > 0.4:
            print("MODERATE clustering: Some concentration but not extreme")
        else:
            print("LOW clustering: Black images are fairly evenly distributed")
    
    # 2. Check if top 5 cells contain most black images
    if sorted_black_cells:
        top_5_black = sum(count for cell, count in sorted_black_cells[:5])
        top_5_pct = 100 * top_5_black / total_black
        print(f"Top 5 cells contain {top_5_pct:.1f}% of all black images")
    
    print(f"\n{'='*80}")

def main():
    parser = argparse.ArgumentParser(
        description="Analyze spatial distribution of black images"
    )
    parser.add_argument(
        "--metas-dir",
        type=str,
        required=True,
        help="Path to directory containing meta JSON files"
    )
    parser.add_argument(
        "--processed-images-dir",
        type=str,
        default="",
        help="Path to processed images directory (optional)"
    )
    parser.add_argument(
        "--log-file",
        type=str,
        required=True,
        help="Path to preprocess log file containing black image names"
    )
    
    args = parser.parse_args()
    
    analyze_black_images(args.metas_dir, args.processed_images_dir, args.log_file)

if __name__ == "__main__":
    main()

