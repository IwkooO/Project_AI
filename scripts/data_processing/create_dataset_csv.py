#!/usr/bin/env python3
"""
Create a CSV dataset combining processed panorama images with their metadata.
"""

import json
import csv
from pathlib import Path
from tqdm import tqdm
import argparse

def extract_pano_id(image_filename):
    """Extract panoId from image filename (e.g., 'image_ABC123.jpg' -> 'ABC123')."""
    # Remove 'image_' prefix and '.jpg' suffix
    if image_filename.startswith('image_'):
        pano_id = image_filename[6:]  # Remove 'image_'
    else:
        pano_id = image_filename
    
    if pano_id.endswith('.jpg'):
        pano_id = pano_id[:-4]  # Remove '.jpg'
    
    return pano_id

def create_dataset_csv(panorama_processed_folder, metas_folder, output_csv, geoguessr_id):
    """
    Create a CSV dataset from processed images and meta files.
    
    Args:
        panorama_processed_folder: Path to folder with processed panorama images
        metas_folder: Path to folder with meta JSON files
        output_csv: Path to output CSV file
        geoguessr_id: GeoGuessr ID for the dataset
    """
    panorama_path = Path(panorama_processed_folder)
    metas_path = Path(metas_folder)
    output_path = Path(output_csv)
    
    # Get all processed images
    image_files = list(panorama_path.glob("*.jpg"))
    print(f"Found {len(image_files)} processed images")
    
    # Load all meta files into a dictionary for quick lookup
    meta_files = list(metas_path.glob("*.json"))
    print(f"Found {len(meta_files)} meta files")
    
    meta_dict = {}
    for meta_file in meta_files:
        pano_id = meta_file.stem  # Filename without extension
        try:
            with meta_file.open() as f:
                meta_dict[pano_id] = json.load(f)
        except Exception as e:
            print(f"Warning: Could not load {meta_file}: {e}")
    
    print(f"Loaded {len(meta_dict)} meta files into memory")
    
    # Prepare CSV data
    csv_rows = []
    matched_count = 0
    unmatched_count = 0
    
    for image_file in tqdm(image_files, desc="Processing images"):
        # Extract panoId from image filename
        pano_id = extract_pano_id(image_file.name)
        
        # Get meta data
        meta = meta_dict.get(pano_id, {})
        
        if meta:
            matched_count += 1
        else:
            unmatched_count += 1
        
        # Prepare row data
        row = {
            'image_path': str(image_file),
            'pano_id': pano_id,
            'country': meta.get('country', ''),
            'lat': meta.get('lat', ''),
            'lng': meta.get('lng', ''),
            'meta_name': meta.get('metaName', ''),
            'note': meta.get('note', ''),
            'footer': meta.get('footer', ''),
            'images': ', '.join(meta.get('images', [])) if isinstance(meta.get('images'), list) else '',
        }
        
        csv_rows.append(row)
    
    # Write CSV file
    if csv_rows:
        fieldnames = ['image_path', 'pano_id', 'country', 'lat', 'lng', 'meta_name', 'note', 'footer', 'images']
        
        with output_path.open('w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(csv_rows)
        
        print(f"\nDataset CSV created: {output_path}")
        print(f"Total rows: {len(csv_rows)}")
        print(f"Matched with meta: {matched_count}")
        print(f"Unmatched (no meta): {unmatched_count}")
    else:
        print("No data to write!")

def main():
    parser = argparse.ArgumentParser(description='Create CSV dataset from processed images and metadata')
    parser.add_argument('--panorama-processed', type=str,
                       default='/scratch-shared/igodzwon/Project_AI/data/691df1ee911f74393c53af8c/panorama_processed',
                       help='Path to folder with processed panorama images')
    parser.add_argument('--metas', type=str,
                       default='/home/igodzwon/Project_AI/data/691df1ee911f74393c53af8c/metas',
                       help='Path to folder with meta JSON files')
    parser.add_argument('--output', type=str,
                       default='/home/igodzwon/Project_AI/data/691df1ee911f74393c53af8c/dataset.csv',
                       help='Path to output CSV file')
    parser.add_argument('--geoguessr-id', type=str, default='691df1ee911f74393c53af8c',
                       help='GeoGuessr ID (optional, for reference)')
    
    args = parser.parse_args()
    
    create_dataset_csv(
        args.panorama_processed,
        args.metas,
        args.output,
        args.geoguessr_id
    )

if __name__ == '__main__':
    main()

