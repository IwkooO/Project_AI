#!/usr/bin/env python3
"""
Export dataset metadata to CSV without loading images.
Exports country_idx, concept_idx, coordinates, and all metadata fields.
"""

import argparse
import csv
import json
from pathlib import Path
import sys

import torch

# Add parent directories to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.dataset import PanoramaCBMDataset


def export_dataset_to_csv(
    output_path: Path,
    encoder_model: str = None,
    max_samples: int = None,
    country: str = None,
    require_coordinates: bool = False,
    geoguessr_id: str = None,
):
    """
    Export dataset metadata to CSV.
    
    Args:
        output_path: Path to output CSV file (will be placed in data/exports/)
        encoder_model: Optional encoder model name (not needed for export)
        max_samples: Optional limit on number of samples
        country: Optional country filter
        require_coordinates: Only include samples with coordinates
        geoguessr_id: GeoGuessr map ID (defaults to dataset default)
    """
    # Ensure output is in data/exports/ directory
    project_root = Path('/scratch-shared/pnair/Project_AI')
    exports_dir = project_root / "data" / "export"
    
    # If output_path is absolute, use it as-is but ensure it's in exports
    # If relative, place it in data/exports/
    if output_path.is_absolute():
        # If it's already in data/exports, use it; otherwise, take just the filename
        if str(exports_dir) in str(output_path):
            final_path = output_path
        else:
            final_path = exports_dir / output_path.name
    else:
        # Relative path - place in data/exports/
        final_path = exports_dir / output_path
    # Create dataset (without loading images - we'll access samples directly)
    dataset_kwargs = {
        'transform': None,  # We don't need transforms since we're not loading images
        'max_samples': max_samples,
        'country': country,
        'require_coordinates': require_coordinates,
        'encoder_model': encoder_model,
    }
    if geoguessr_id is not None:
        dataset_kwargs['geoguessr_id'] = geoguessr_id
    dataset = PanoramaCBMDataset(**dataset_kwargs)
    
    # Load coordinates from locations file if available
    data_root = Path(dataset.data_root)
    folder = data_root / dataset.geoguessr_id
    loc_file = folder / f"locations_{dataset.geoguessr_id}.json"
    pano_to_coords = {}
    if loc_file.exists():
        try:
            with loc_file.open() as f:
                location_data = json.load(f)
            for loc in location_data.get('customCoordinates', []):
                pano_to_coords[loc['panoId']] = {
                    'lat': loc['lat'],
                    'lng': loc['lng']
                }
            print(f"Loaded coordinates for {len(pano_to_coords)} locations from locations file")
        except Exception as e:
            print(f"Warning: Could not load coordinates from locations file: {e}")
    
    # Prepare CSV data
    rows = []
    
    print(f"Exporting {len(dataset.samples)} samples to CSV...")
    
    for sample in dataset.samples:
        # Get indices
        concept_idx = dataset.concept_to_idx[sample['meta_name']]
        country_idx = dataset.country_to_idx[sample['country']]
        
        # Get coordinates - first from sample, then from locations file if not available
        lat = sample['lat']
        lng = sample['lng']
        
        # If coordinates not in sample, try to get from locations file
        if (lat is None or lng is None) and sample['pano_id'] in pano_to_coords:
            coords = pano_to_coords[sample['pano_id']]
            lat = coords['lat']
            lng = coords['lng']
        
        # Get coordinates (normalized)
        from src.dataset import normalize_coordinates
        
        # Normalize coordinates if they exist
        if lat is not None and lng is not None:
            coords = normalize_coordinates(lat, lng)
            lat_norm = coords[0].item() if not torch.isnan(coords[0]) else None
            lng_norm = coords[1].item() if not torch.isnan(coords[1]) else None
        else:
            lat_norm = None
            lng_norm = None
        
        # Convert image_path to absolute path
        image_path = Path(sample['image_path'])
        if not image_path.is_absolute():
            image_path = project_root / image_path
        
        # Prepare row data - convert None to empty string for CSV
        row = {
            'pano_id': sample['pano_id'],
            'concept_idx': concept_idx,
            'country_idx': country_idx,
            'meta_name': sample['meta_name'],
            'country': sample['country'],
            'lat': lat if lat is not None else '',
            'lng': lng if lng is not None else '',
            'lat_norm': lat_norm if lat_norm is not None else '',
            'lng_norm': lng_norm if lng_norm is not None else '',
            'note': sample['note'],
            'images': str(sample['images']) if sample['images'] else '',
            'image_path': str(image_path),
        }
        rows.append(row)
    
    # Write to CSV
    if rows:
        fieldnames = [
            'pano_id',
            'concept_idx',
            'country_idx',
            'meta_name',
            'country',
            'lat',
            'lng',
            'lat_norm',
            'lng_norm',
            'note',
            'images',
            'image_path',
        ]
        
        final_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(final_path, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)
        
        print(f"Successfully exported {len(rows)} rows to {final_path}")
    else:
        print("No samples to export")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Export dataset metadata to CSV"
    )
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('dataset_export.csv'),
        help='Output CSV file name or path (will be placed in data/exports/, default: dataset_export.csv)'
    )
    parser.add_argument(
        '--encoder-model',
        type=str,
        default=None,
        help='Encoder model name (optional, not needed for export)'
    )
    parser.add_argument(
        '--max-samples',
        type=int,
        default=None,
        help='Maximum number of samples to export (for testing)'
    )
    parser.add_argument(
        '--country',
        type=str,
        default=None,
        help='Filter by country name'
    )
    parser.add_argument(
        '--require-coordinates',
        action='store_true',
        help='Only include samples with coordinates'
    )
    parser.add_argument(
        '--geoguessr-id',
        type=str,
        default=None,
        help='GeoGuessr map ID (defaults to dataset default)'
    )
    
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()
    export_dataset_to_csv(
        output_path=args.output,
        encoder_model=args.encoder_model,
        max_samples=args.max_samples,
        country=args.country,
        require_coordinates=args.require_coordinates,
        geoguessr_id=args.geoguessr_id,
    )

