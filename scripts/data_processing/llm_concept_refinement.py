#!/usr/bin/env python3
"""
Map GeoGuessr meta names + notes to predefined concept list using Agno agent.

Processes each unique concept+note pair and maps it to one of the predefined concepts.
Saves progress every 1000 items to allow resuming.
"""

CONCEPT_LIST = [
    "landscape_urban",
    "bin",
    "landscape_coastline",
    "bollard_shape",
    "bollard_design",
    "building_facade",
    "infrastructure_fence",
    "building_roof",
    "building_ruin",
    "building_shape",
    "camera_meta",
    "car_meta",
    "urban_artifacts",
    "infrastructure_gate",
    "guardrail",
    "hydrant",
    "infrastructure_bridge",
    "infrastructure_lamps",
    "infrastructure_industrial",
    "infrastructure_road",
    "infrastructure_wind",
    "landscape_arid",
    "landscape_hills",
    "landscape_forest",
    "landscape_arctic",
    "landscape_temperate",
    "landscape_desert",
    "landscape_canyon",
    "landscape_farm",
    "landscape_temperature",
    "landscape_tropical",
    "landscape_alpine",
    "landscape_flatland",
    "landscape_fog",
    "landscape_glacier",
    "landscape_mountains",
    "landscape_grassland",
    "landscape_haze",
    "landscape_island",
    "landscape_lake",
    "landscape_grass",
    "landscape_sky",
    "landscape_plain",
    "landscape_plateau",
    "landscape_prairie",
    "landscape_fields",
    "landscape_ridge",
    "landscape_river",
    "landscape_rural",
    "vegetation_trees",
    "landscape_rock",
    "landscape_forests",
    "landscape_savannah",
    "landscape_shrubland",
    "landscape_winter",
    "landscape_steppe",
    "landscape_valley",
    "landscape_volcanic",
    "road_paved",
    "pole_bird",
    "pole_marker",
    "pole_shape",
    "road_coastal",
    "road_cobbled",
    "infrastructure_lamp",
    "road_dashed",
    "road_dirt",
    "road_divided",
    "road_flat",
    "road_gravel",
    "road_sign",
    "road_intersection",
    "road_line",
    "road_marker",
    "railroad",
    "road_rural",
    "road_shape",
    "road_snow",
    "road_urban",
    "road_tropical",
    "road_utility",
    "script_arabic",
    "script_basque",
    "script_bengali",
    "script_catalan",
    "script_celtic",
    "script_chinese",
    "script_cyrillic",
    "script_devanagari",
    "script_estonian",
    "script_greek",
    "script_north_indian",
    "script_pacific",
    "script_hebrew",
    "script_south_indian",
    "script_south_east_asian",
    "script_japanese",
    "script_latin",
    "script_south_american",
    "phone_code",
    "sign_information",
    "script_latvian",
    "script_european",
    "chevron",
    "traffic_light",
    "post_sign",
    "landscape_sign",
    "landscape_soil",
    "buddhism",
    "statue",
    "sticker",
    "landscape_tent",
    "tram",
    "vegetation_agave",
    "vegetation_bamboo",
    "vegetation_bush",
    "vegetation_cacti",
    "vegetation_coastline",
    "vegetation_corn",
    "vegetation_crops",
    "vegetation_farm",
    "vegetation_plant",
    "vegetation_floral",
    "vegetation_forest",
    "vegetation_fruit",
    "vegetation_fungi",
    "vegetation_grassland",
    "vegetation_greenhouse",
    "vegetation_hay",
    "road_irrigation",
    "vegetation_moss",
    "vegetation_rice",
    "people_seastars",
    "vegetation_tea",
    "vegetation_vineyard",
    "trekkers",
    "car_type",
]



import pandas as pd
import re
from tqdm import tqdm
from pathlib import Path
import os
import sys
import time
from functools import wraps
from typing import Tuple, Dict, Any, List, Set
from concurrent.futures import ThreadPoolExecutor, as_completed

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from agno.agent import Agent
from agno.models.openai import OpenAIChat


def retry_with_backoff(max_retries=5, initial_delay=1, backoff_factor=2):
    """
    Decorator for retrying function calls with exponential backoff.
    
    Args:
        max_retries: Maximum number of retry attempts
        initial_delay: Initial delay in seconds
        backoff_factor: Multiplier for delay after each retry
    """
    def decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            delay = initial_delay
            last_exception = None
            
            for attempt in range(max_retries):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    last_exception = e
                    error_msg = str(e).lower()
                    
                    # Check if it's a retryable error
                    retryable_errors = [
                        'connection', 'timeout', 'rate limit', '429', 
                        '503', '502', '500', 'network', 'temporary'
                    ]
                    
                    is_retryable = any(err in error_msg for err in retryable_errors)
                    
                    if attempt < max_retries - 1 and is_retryable:
                        print(f"  Retry attempt {attempt + 1}/{max_retries} after {delay}s (Error: {type(e).__name__})")
                        time.sleep(delay)
                        delay *= backoff_factor
                    else:
                        # Not retryable or max retries reached
                        raise
            
            # If we get here, all retries failed
            raise last_exception
        return wrapper
    return decorator


def clean_html_tags(text):
    """Remove HTML tags from text, handling NaN values."""
    if pd.isna(text) or text is None:
        return ''
    text = str(text)
    return re.sub('<.*?>', '', text)


def load_checkpoint(checkpoint_path):
    """Load previous results if checkpoint exists."""
    if checkpoint_path.exists():
        df_checkpoint = pd.read_csv(checkpoint_path,sep=';')
        results = {row['id']: row['mapped_concept'] for _, row in df_checkpoint.iterrows()}
        print(f"Loaded checkpoint: {len(results)} results")
        return results, df_checkpoint['id'].max() + 1
    return {}, 0


def save_checkpoint(results, concepts, checkpoint_path, start_idx=0):
    """Save current results to checkpoint file."""
    df_checkpoint = pd.DataFrame([
        {'id': i, 'original': concepts[i]['concept'], 'note': concepts[i]['note'], 'mapped_concept': results[i]}
        for i in range(start_idx, len(results))
        if i in results
    ])
    df_checkpoint.to_csv(checkpoint_path, index=False)
    print(f"  Saved checkpoint: {len(df_checkpoint)} results to {checkpoint_path}")


def validate_concept(concept: str, known_extensions: Set[str] = None) -> Tuple[bool, str]:
    """
    Validate that a concept is either in the predefined list, a known extension, 
    or a valid new climate extension.
    
    Args:
        concept: The concept string to validate.
        known_extensions: Set of previously validated extension concepts.
        
    Returns:
        Tuple of (is_valid, validated_concept).
    """
    if known_extensions is None:
        known_extensions = set()
    
    # Check if it's in the predefined list
    if concept in CONCEPT_LIST:
        return True, concept
    
    # Check if it's a previously validated extension
    if concept in known_extensions:
        return True, concept
    
    # Check if it's a valid NEW climate extension for landscape, vegetation, or road
    if concept.startswith(("landscape_", "vegetation_", "road_")):
        parts = concept.rsplit("_", 1)
        if len(parts) == 2:
            base_concept, climate_tag = parts
            
            # Validate tag format (lowercase letters only)
            if climate_tag.isalpha() and climate_tag.islower():
                # Check if base concept exists
                if base_concept in CONCEPT_LIST:
                    return True, concept
                
                # Case-insensitive base match
                base_lower = base_concept.lower()
                matches = [c for c in CONCEPT_LIST if c.lower() == base_lower]
                if matches:
                    return True, f"{matches[0]}_{climate_tag}"
    
    # Try case-insensitive match against predefined list
    concept_lower = concept.lower()
    matches = [c for c in CONCEPT_LIST if c.lower() == concept_lower]
    if matches:
        return True, matches[0]
    
    return False, concept


def main():
    # Paths
    data_dir = project_root / 'data' / '6921d7831744c5356b098bf7_balanced'
    csv_path = data_dir / 'dataset.csv'
    output_dir = project_root / 'data'
    checkpoint_path = output_dir / 'concept_refinement_checkpoint.csv'
    final_output_path = output_dir / 'concept_refinement_mapped.csv'
    
    # Load data
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Total locations: {len(df):,}")
    
    # Clean HTML tags
    print("Cleaning HTML tags from notes...")
    df['note_cleaned'] = df['note'].apply(clean_html_tags)
    
    # Get unique concept + note pairs
    print("Extracting unique concept+note pairs...")
    df_metas = df[['meta_name', 'note_cleaned']].drop_duplicates()
    pairs = df_metas[df_metas['note_cleaned'].str.len() > 0].reset_index(drop=True)
    
    # Create dictionary
    concepts = {i: {'concept': r['meta_name'], 'note': r['note_cleaned']} 
                for i, r in pairs.iterrows()}
    
    print(f"Found {len(concepts)} unique concept+note pairs")
    
    # Load checkpoint if exists
    results, start_idx = load_checkpoint(checkpoint_path)
    
    # Setup agent
    print("Setting up Agno agent...")
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        api_key = "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"
        os.environ["OPENAI_API_KEY"] = api_key
    
    agent = Agent(
        model=OpenAIChat(id="gpt-5.1"),
        instructions=[
            "You map GeoGuessr meta names + notes to visual concept labels for use in computer vision.",
            "",
            "TASK:",
            "Given meta_name, note, and possible_concepts, select the MOST APPROPRIATE concept.",
            "The concept should best represent the visual characteristics described.",
            "",
            "SELECTION RULES:",
            "1. PREFER concepts already in the possible_concepts list.",
            "2. Choose the concept that best matches the visual characteristics described.",
            "3. If multiple concepts could apply, choose the most specific one.",
            "4. Consider the visual appearance, not just the semantic meaning.",
            "",
            "CONCEPT EXTENSION RULES (STRICT):",
            "For landscape_*, vegetation_*, and road_* concepts ONLY, you MAY create an extension:",
            "- Format: {base_concept}_{climate_tag}",
            "- {base_concept} MUST exist in the possible_concepts list",
            "- Climate tags: use general visual/climate terms (e.g. tropical, arid, temperate, boreal, snowy, dry, wet, lush)",
            "- Do NOT use country/region names (NO japanese, brazilian, african, etc.)",
            "- Only extend if there is STRONG evidence in the meta/note",
            "- REUSE existing extensions from possible_concepts if they fit",
            "",
            "For ALL other concepts (script_*, building_*, car_*, etc.): use EXACTLY from the list, NO modifications.",
            "",
            "EXAMPLES:",
            "meta_name: 'Russian storefront', note: 'Cyrillic text visible' → script_cyrillic",
            "meta_name: 'Coastal dunes', note: 'Sandy beach area' → landscape_coastline",
            "meta_name: 'Car with antenna', note: 'Vehicle with roof antenna' → car_meta",
            "meta_name: 'Yellow warning sign', note: 'Traffic warning sign' → road_sign",
            "meta_name: 'Baltic pine forest', note: 'Dense coniferous trees in cold climate' → vegetation_forest_boreal",
            "meta_name: 'Chilean rolling green hills', note: 'Temperate climate hills' → landscape_hills_temperate",
            "meta_name: 'Tropical rainforest', note: 'Dense tropical vegetation' → vegetation_forest_tropical",
            "meta_name: 'Dusty desert road', note: 'Dry arid environment' → road_dirt_arid",
            "",
            "OUTPUT:",
            "Return EXACTLY ONE concept name. No explanation, no extra text."
        ],
        markdown=False
    )
    
    # Create retryable API call function
    @retry_with_backoff(max_retries=5, initial_delay=2, backoff_factor=2)
    def call_agent_with_retry(query):
        """Call agent with retry logic."""
        return agent.run(query)
    
    def process_single_item(idx: int, all_concepts_snapshot: List[str], known_extensions: Set[str]):
        """Process a single item and return (idx, result, error, new_extension)."""
        data = concepts[idx]
        
        query = (
            f"meta_name: {data['concept']}\n"
            f"note: {data['note']}\n"
            f"possible_concepts: {', '.join(all_concepts_snapshot)}"
        )
        
        try:
            resp = call_agent_with_retry(query)
            mapped_concept = resp.content.strip()
            
            # Validate the mapped concept
            is_valid, validated_concept = validate_concept(mapped_concept, known_extensions)
            
            if not is_valid:
                # If invalid, try to recover by stripping the tag
                if "_" in mapped_concept:
                    base_attempt = mapped_concept.rsplit("_", 1)[0]
                    if base_attempt in CONCEPT_LIST:
                        validated_concept = base_attempt
                    else:
                        validated_concept = mapped_concept
                else:
                    validated_concept = mapped_concept
            
            # Check if it's a new extension
            new_ext = validated_concept if validated_concept not in CONCEPT_LIST else None
            return idx, validated_concept, None, new_ext
            
        except Exception as e:
            return idx, f'ERROR: {e}', {'id': idx, 'concept': data['concept'], 'error': str(e)}, None
    
    # Process concepts in parallel
    print(f"\nProcessing concepts (starting from index {start_idx})...")
    print(f"Mapping to {len(CONCEPT_LIST)} predefined concepts")
    print("Using parallel processing: 5 concurrent requests")
    print("Using retry logic: 5 retries with exponential backoff (2s, 4s, 8s, 16s, 32s)")
    
    errors = []
    checkpoint_interval = 1000
    new_concepts = set()  # Track newly added concepts
    batch_size = 8
    
    indices_to_process = list(range(start_idx, len(concepts)))
    
    with tqdm(total=len(indices_to_process), desc="Mapping concepts", initial=0) as pbar:
        for batch_start in range(0, len(indices_to_process), batch_size):
            batch_indices = indices_to_process[batch_start:batch_start + batch_size]
            
            # Snapshot current concepts for this batch
            all_concepts_snapshot = sorted(set(CONCEPT_LIST) | new_concepts)
            known_extensions_snapshot = set(new_concepts)
            
            # Process batch in parallel
            with ThreadPoolExecutor(max_workers=batch_size) as executor:
                futures = {
                    executor.submit(process_single_item, idx, all_concepts_snapshot, known_extensions_snapshot): idx 
                    for idx in batch_indices
                }
                
                for future in as_completed(futures):
                    idx, result, error, new_ext = future.result()
                    
                    results[idx] = result
                    
                    if error:
                        errors.append(error)
                        print(f"\nFailed [{idx}]: {error['error']}")
                    
                    if new_ext:
                        new_concepts.add(new_ext)
                    
                    pbar.update(1)
            
            # Save checkpoint every 1000 items
            processed_count = batch_start + len(batch_indices)
            if processed_count % checkpoint_interval < batch_size and processed_count >= checkpoint_interval:
                print(f"\nCheckpoint at {start_idx + processed_count} items...")
                save_checkpoint(results, concepts, checkpoint_path, start_idx)
    
    # Final save
    print(f"\nProcessing complete!")
    print(f"  Total processed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    
    # Save final results
    print(f"\nSaving final results...")
    df_results = pd.DataFrame([
        {'id': i, 'original': concepts[i]['concept'], 'note': concepts[i]['note'], 'mapped_concept': results[i]}
        for i in sorted(results.keys())
    ])
    df_results.to_csv(final_output_path, index=False)
    print(f"  Saved {len(df_results)} results to {final_output_path}")
    
    # Save errors if any
    if errors:
        df_errors = pd.DataFrame(errors)
        error_path = output_dir / 'concept_refinement_errors.csv'
        df_errors.to_csv(error_path, index=False)
        print(f"  Saved {len(errors)} errors to {error_path}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Unique original concepts: {df_results['original'].nunique()}")
    print(f"  Unique mapped concepts: {df_results['mapped_concept'].nunique()}")
    print(f"  Reduction: {df_results['original'].nunique() - df_results['mapped_concept'].nunique()} concepts")
    print(f"\nTop 10 mapped concepts:")
    print(df_results['mapped_concept'].value_counts().head(10))
    
    # Check coverage of predefined concepts
    mapped_set = set(df_results['mapped_concept'].unique())
    predefined_set = set(CONCEPT_LIST)
    unused_concepts = predefined_set - mapped_set
    if unused_concepts:
        print(f"\nUnused concepts from predefined list ({len(unused_concepts)}):")
        print(sorted(unused_concepts))
    
    # Report newly added concepts (climate extensions)
    if new_concepts:
        print(f"\nNewly added concepts (climate extensions): {len(new_concepts)}")
        print("These are landscape/vegetation/road concepts with climate tags:")
        for new_concept in sorted(new_concepts):
            count = (df_results['mapped_concept'] == new_concept).sum()
            print(f"  {new_concept}: {count} occurrences")


if __name__ == '__main__':
    main()

