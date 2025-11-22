#!/usr/bin/env python3
"""
Enrich meta JSON files with coordinate information from locations data.
"""

import json
from pathlib import Path
from tqdm import tqdm
import argparse

def main():
    parser = argparse.ArgumentParser(description="Enrich meta JSON files with coordinates")
    parser.add_argument("--geoguessr-id", type=str, required=True,
                        help="GeoGuessr map ID")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for data")
    args = parser.parse_args()
    
    geoguessrId = args.geoguessr_id
    data_root = Path(args.data_root)
    folder = data_root / geoguessrId
    meta_folder = folder / "metas"
    loc_file = folder / f"locations_{geoguessrId}.json"
    # Load locations data
    if not loc_file.exists():
        print(f"Error: {loc_file} not found!")
        return

    with loc_file.open() as f:
        location_data = json.load(f)

    # Create panoId to coordinates mapping
    pano_to_coords = {}
    for loc in location_data['customCoordinates']:
        pano_to_coords[loc['panoId']] = {
            'lat': loc['lat'],
            'lng': loc['lng']
        }

    print(f"Loaded coordinates for {len(pano_to_coords)} locations")

    # Process each meta file
    meta_files = list(meta_folder.glob("*.json"))
    print(f"Processing {len(meta_files)} meta files...")

    updated_count = 0
    for meta_path in tqdm(meta_files, desc="Enriching meta files"):
        # Extract panoId from filename
        pano_id = meta_path.stem

        # Load existing meta data
        with meta_path.open() as f:
            meta = json.load(f)

        # Add coordinates if available
        if pano_id in pano_to_coords:
            coords = pano_to_coords[pano_id]
            meta['lat'] = coords['lat']
            meta['lng'] = coords['lng']

            # Save updated meta
            with meta_path.open('w') as f:
                json.dump(meta, f, indent=2)

            updated_count += 1
        else:
            print(f"Warning: No coordinates found for panoId {pano_id}")

    print(f"Successfully updated {updated_count} meta files with coordinates")

if __name__ == "__main__":
    main()
