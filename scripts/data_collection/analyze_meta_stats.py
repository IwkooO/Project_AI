#!/usr/bin/env python3
"""
Analyze meta JSON files to get statistics about countries, concepts, and images.
"""
import json
import argparse
from pathlib import Path
from collections import defaultdict, Counter
import sys

def analyze_metas(metas_dir):
    """
    Analyze meta JSON files and return statistics.
    
    Args:
        metas_dir: Path to directory containing meta JSON files
        
    Returns:
        dict: Statistics dictionary
    """
    metas_path = Path(metas_dir)
    
    if not metas_path.exists():
        raise ValueError(f"Directory does not exist: {metas_dir}")
    
    # Collect data
    countries = []
    concepts = []
    country_to_concepts = defaultdict(set)
    country_to_image_count = defaultdict(int)
    concept_to_countries = defaultdict(set)
    concept_to_image_count = defaultdict(int)
    
    # Read all JSON files
    json_files = list(metas_path.glob("*.json"))
    print(f"Found {len(json_files)} meta JSON files")
    
    for json_file in json_files:
        try:
            with open(json_file, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            country = data.get('country', 'Unknown')
            # Normalize country: strip whitespace and convert to lowercase
            country = country.strip().lower() if country else 'unknown'
            meta_name = data.get('metaName', 'Unknown')
            images = data.get('images', [])
            num_images = len(images)
            
            # Collect stats
            countries.append(country)
            concepts.append(meta_name)
            country_to_concepts[country].add(meta_name)
            country_to_image_count[country] += num_images
            concept_to_countries[meta_name].add(country)
            concept_to_image_count[meta_name] += num_images
            
        except Exception as e:
            print(f"Warning: Error reading {json_file}: {e}")
            continue
    
    # Calculate statistics
    unique_countries = set(countries)
    unique_concepts = set(concepts)
    total_images = sum(country_to_image_count.values())
    
    # Sort countries by number of images (descending)
    countries_by_images = sorted(
        country_to_image_count.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    # Sort concepts by number of images (descending)
    concepts_by_images = sorted(
        concept_to_image_count.items(),
        key=lambda x: x[1],
        reverse=True
    )
    
    stats = {
        'total_meta_files': len(json_files),
        'unique_countries': len(unique_countries),
        'unique_concepts': len(unique_concepts),
        'total_images': total_images,
        'countries': list(unique_countries),
        'concepts': list(unique_concepts),
        'country_to_concepts': {k: list(v) for k, v in country_to_concepts.items()},
        'country_to_image_count': dict(country_to_image_count),
        'countries_by_images': countries_by_images,
        'concepts_by_images': concepts_by_images,
        'concept_to_countries': {k: list(v) for k, v in concept_to_countries.items()},
    }
    
    return stats

def print_stats(stats):
    """Print formatted statistics."""
    print("\n" + "="*80)
    print("META STATISTICS")
    print("="*80)
    
    print(f"\n📊 Overall Statistics:")
    print(f"  Total Meta Files: {stats['total_meta_files']}")
    print(f"  Unique Countries: {stats['unique_countries']}")
    print(f"  Unique Concepts: {stats['unique_concepts']}")
    print(f"  Total Images: {stats['total_images']}")
    print(f"  Avg Images per Country: {stats['total_images'] / stats['unique_countries']:.1f}")
    print(f"  Avg Images per Concept: {stats['total_images'] / stats['unique_concepts']:.1f}")
    
    print(f"\n🌍 Countries (sorted by image count):")
    print(f"  {'Country':<30} {'Images':<10} {'Concepts':<10}")
    print(f"  {'-'*30} {'-'*10} {'-'*10}")
    for country, img_count in stats['countries_by_images'][:50]:  # Top 20
        concept_count = len(stats['country_to_concepts'][country])
        print(f"  {country:<30} {img_count:<10} {concept_count:<10}")
    print(f" Least collected countries:")
    for country, img_count in stats['countries_by_images'][-50:]:
        concept_count = len(stats['country_to_concepts'][country])
        print(f"  {country:<30} {img_count:<10} {concept_count:<10}")
    
    if len(stats['countries_by_images']) > 20:
        print(f"  ... and {len(stats['countries_by_images']) - 20} more countries")
    
    print(f"\n💡 Top Concepts (sorted by image count):")
    print(f"  {'Concept':<40} {'Images':<10} {'Countries':<10}")
    print(f"  {'-'*40} {'-'*10} {'-'*10}")
    for concept, img_count in stats['concepts_by_images'][:20]:  # Top 20
        country_count = len(stats['concept_to_countries'][concept])
        print(f"  {concept:<40} {img_count:<10} {country_count:<10}")
    
    if len(stats['concepts_by_images']) > 20:
        print(f"  ... and {len(stats['concepts_by_images']) - 20} more concepts")
    
    print(f"\n📋 Detailed Country Breakdown:")
    print(f"  {'Country':<30} {'Images':<10} {'Concepts':<10} {'Concept Names'}")
    print(f"  {'-'*30} {'-'*10} {'-'*10} {'-'*50}")
    for country, img_count in stats['countries_by_images']:
        concept_list = stats['country_to_concepts'][country]
        concept_count = len(concept_list)
        concept_names = ', '.join(concept_list[:3])  # Show first 3
        if len(concept_list) > 3:
            concept_names += f" ... (+{len(concept_list) - 3} more)"
        print(f"  {country:<30} {img_count:<10} {concept_count:<10} {concept_names}")
    
    print("\n" + "="*80)

def main():
    parser = argparse.ArgumentParser(
        description="Analyze meta JSON files to get statistics"
    )
    parser.add_argument(
        "--metas-dir",
        type=str,
        required=True,
        help="Path to directory containing meta JSON files"
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional: Path to save JSON output (default: print only)"
    )
    
    args = parser.parse_args()
    
    # Analyze
    stats = analyze_metas(args.metas_dir)
    
    # Print
    print_stats(stats)
    
    # Save if requested
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(stats, f, indent=2, ensure_ascii=False)
        print(f"\n✅ Statistics saved to: {output_path}")

if __name__ == "__main__":
    main()

