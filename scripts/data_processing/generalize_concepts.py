#!/usr/bin/env python3
"""
Generalize GeoGuessr concept names using Agno agent.

Processes each unique concept+note pair and generalizes the concept name.
Saves progress every 1000 items to allow resuming.
"""

import pandas as pd
import re
from tqdm import tqdm
from pathlib import Path
import json
import os
import sys
import time
from functools import wraps

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
        df_checkpoint = pd.read_csv(checkpoint_path)
        results = {row['id']: row['generalized'] for _, row in df_checkpoint.iterrows()}
        print(f"Loaded checkpoint: {len(results)} results")
        return results, df_checkpoint['id'].max() + 1
    return {}, 0


def save_checkpoint(results, concepts, checkpoint_path, start_idx=0):
    """Save current results to checkpoint file."""
    df_checkpoint = pd.DataFrame([
        {'id': i, 'original': concepts[i]['concept'], 'note': concepts[i]['note'], 'generalized': results[i]}
        for i in range(start_idx, len(results))
        if i in results
    ])
    df_checkpoint.to_csv(checkpoint_path, index=False)
    print(f"  Saved checkpoint: {len(df_checkpoint)} results to {checkpoint_path}")


def main():
    # Paths
    data_dir = project_root / 'data' / '6921d7831744c5356b098bf7_balanced'
    csv_path = data_dir / 'dataset.csv'
    output_dir = project_root / 'data'
    checkpoint_path = output_dir / 'generalized_concepts_checkpoint_v3.csv'
    final_output_path = output_dir / 'generalized_concepts_v3.csv'
    
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
        model=OpenAIChat(id="gpt-4o-mini"),
        instructions = [
            "You generalize GeoGuessr meta names into visually meaningful concept labels for use in a computer vision model.",
            
            "OUTPUT RULES:",
            "- Return EXACTLY ONE short, lowercase concept name.",
            "- Use only letters and underscores.",
            "- Never output numbers, country names, city names, brand names, or route names.",
            "- Never output vague concepts like 'object' or 'scene'.",
            "- Never output hyper-specific concepts based on colors, tiny details, or complex multi-attribute descriptions.",
            "- The final label may contain 1, 2, or 3 underscore-separated parts, never add more than 3 parts.",
            
            "VISUAL CATEGORY REFINEMENT:",
            "If the meta refers to a broad visual category (road, landscape, car, pole, sign, building, vegetation, bollard), "
            "refine it ONE level deeper using a meaningful and reusable subtype.",

            "VALID SUBTYPE EXAMPLES:",
            "- road → road_paved, road_dirt, road_urban, road_rural, road_intersection",
            "- landscape → landscape_desert, landscape_hills, landscape_coastline, landscape_forest, landscape_mountains",
            "- car → car_antenna, car_roof, car_shape",
            "- pole → pole_shape, pole_marker, pole_bird",
            "- sign → sign_warning, sign_information, sign_arrow",
            "- vegetation → vegetation_trees, vegetation_forest, vegetation_grassland",
            "- building → building_roof, building_facade, building_shape",
            "- bollard → bollard_shape, bollard_material",
            
            "BIOME EXTENSION (FOR VEGETATION AND LANDSCAPE ONLY):",
            "If the meta strongly indicates climate or biome, append ONE biome tag.",
            "Allowed biome tags: tropical, temperate, boreal, mediterranean, arid, alpine, subpolar.",
            "Examples:",
            "- vegetation_trees_temperate, vegetation_trees_tropical, vegetation_forest_boreal",
            "- landscape_hills_temperate, landscape_mountains_alpine, landscape_desert_arid",
            "Do NOT invent new biomes.",
            "Do NOT exceed 3 parts: category_subtype_biome is the maximum.",
            
            "SCRIPT RULE (FOR TEXT, SIGNS, WRITING, LETTER SHAPES):",
            "If the meta refers to writing, text, characters, or letter shapes, output the script family.",
            "Script refers to the visual writing system, not the spoken language.",
            "Example scripts:",
            "script_latin, script_cyrillic, script_arabic, script_hebrew, script_chinese,",
            "Group to a broder script family when possible.",
            
            "SUBTYPING RULES:",
            "- Choose exactly one final concept.",
            "- Base it on the visually dominant information in the meta and note.",
            "- If the meta already names a specific object (bollard, hydrant, chevron), output that class without adding attributes.",
            
            "NORMALIZATION RULE:",
            "- Normalize near duplicates: mountains/mountain → landscape_mountains; forest/forests → vegetation_forest.",
            
            "EXAMPLES:",
            "\"Russian storefront\" → script_cyrillic",
            "\"Arabic road sign\" → script_arabic",
            "\"Thai writing\" → script_thai",
            "\"Coastal dunes\" → landscape_coastline",
            "\"Chilean rolling green hills\" → landscape_hills_temperate",
            "\"Baltic pine forest\" → vegetation_trees_boreal",
            "\"Car with antenna\" → car_antenna",
            "\"Bird sitting on pole\" → pole_bird",
            "\"Yellow warning sign\" → sign_warning",
            
            "FINAL RULE:",
            "Output ONLY the concept name with no explanation or extra text."
        ],

            # instructions = [
            #     "You generalize GeoGuessr meta names into visually meaningful concept labels for use in computer vision.",
                
            #     "OUTPUT RULES:",
            #     "- Return EXACTLY ONE short lowercase concept name.",
            #     "- Use only letters and underscores.",
            #     "- Do NOT output numbers, country names, city names, brand names, or route names.",
            #     "- Do NOT output vague concepts such as 'object' or 'scene'.",
            #     "- Do NOT output hyper-specific concepts based on colors, tiny details, or multi-attribute combinations.",
                
            #     "LEVEL OF DETAIL:",
            #     "If the concept belongs to a broad category (road, landscape, car, pole, sign, building, vegetation, bollard), "
            #     "refine it ONE level deeper using a meaningful, reusable subtype.",
            #     "Subtypes must be visually grounded and should occur across many images, not one-offs.",
                
            #     "VALID SUBTYPE EXAMPLES:",
            #     "- road → road_paved, road_dirt, road_urban, road_rural",
            #     "- landscape → landscape_desert, landscape_hills, landscape_coastline, landscape_forest, landscape_mountains",
            #     "- car → car_antenna, car_roof, car_shape",
            #     "- pole → pole_shape, pole_marker, pole_bird",
            #     "- sign → sign_warning, sign_information, sign_arrow",
            #     "- vegetation → vegetation_trees, vegetation_bushes, vegetation_grass",
            #     "- building → building_roof, building_facade, building_shape",
            #     "- bollard → bollard_shape, bollard_material",
                
            #     "SUBTYPING RULES:",
            #     "- Choose exactly ONE subtype.",
            #     "- Select the subtype suggested by the meta or note.",
            #     "- If the note is ambiguous, choose the most common sensible subtype.",
            #     "- If the meta already names a valid object class (bollard, hydrant, chevron), output that class without adding attributes.",
                
            #     "MERGE RULE:",
            #     "- Normalize duplicates (mountain/mountains → landscape_mountains).",
                
            #     "LANGUAGE RULE:",
            #     "- If the meta refers to text, writing, scripts, or alphabets, return: language",
                
            #     "EXAMPLES:",
            #     "\"Language - Telugu\" → language",
            #     "\"RN14\" → road_paved",
            #     "\"Ruta 9 rural segment\" → road_rural",
            #     "\"Coastal dunes\" → landscape_coastline",
            #     "\"Los Flamencos National Reserve\" → landscape_desert",
            #     "\"Car with antenna\" → car_antenna",
            #     "\"Wooden pole with marker\" → pole_marker",
            #     "\"Yellow warning sign\" → sign_warning",
                
            #     "FINAL RULE:",
            #     "Output ONLY the concept name, with no explanation."
            # ],

        markdown=False
    )
    
    # Create retryable API call function
    @retry_with_backoff(max_retries=5, initial_delay=2, backoff_factor=2)
    def call_agent_with_retry(agent, query):
        """Call agent with retry logic."""
        return agent.run(query)
    
    # Process concepts
    print(f"\nProcessing concepts (starting from index {start_idx})...")
    print("Using retry logic: 5 retries with exponential backoff (2s, 4s, 8s, 16s, 32s)")
    errors = []
    checkpoint_interval = 1000
    
    for idx in tqdm(range(start_idx, len(concepts)), desc="Generalizing", initial=start_idx, total=len(concepts)):
        data = concepts[idx]
        query = f"Concept: {data['concept']}\nNotes: {data['note']}"
        
        try:
            resp = call_agent_with_retry(agent, query)
            results[idx] = resp.content.strip()
        except Exception as e:
            # All retries failed
            results[idx] = f'ERROR: {e}'
            errors.append({'id': idx, 'concept': data['concept'], 'error': str(e)})
            print(f"\nFailed after retries [{idx}]: {type(e).__name__}: {e}")
            # Small delay before continuing to avoid hammering the API
            time.sleep(1)
        
        # Save checkpoint every 1000 items
        if (idx + 1) % checkpoint_interval == 0:
            print(f"\nCheckpoint at {idx + 1} items...")
            save_checkpoint(results, concepts, checkpoint_path, start_idx)
    
    # Final save
    print(f"\nProcessing complete!")
    print(f"  Total processed: {len(results)}")
    print(f"  Errors: {len(errors)}")
    
    # Save final results
    print(f"\nSaving final results...")
    df_results = pd.DataFrame([
        {'id': i, 'original': concepts[i]['concept'], 'note': concepts[i]['note'], 'generalized': results[i]}
        for i in sorted(results.keys())
    ])
    df_results.to_csv(final_output_path, index=False)
    print(f"  Saved {len(df_results)} results to {final_output_path}")
    
    # Save errors if any
    if errors:
        df_errors = pd.DataFrame(errors)
        error_path = output_dir / 'generalized_concepts_errors.csv'
        df_errors.to_csv(error_path, index=False)
        print(f"  Saved {len(errors)} errors to {error_path}")
    
    # Statistics
    print(f"\nStatistics:")
    print(f"  Unique original concepts: {df_results['original'].nunique()}")
    print(f"  Unique generalized concepts: {df_results['generalized'].nunique()}")
    print(f"  Reduction: {df_results['original'].nunique() - df_results['generalized'].nunique()} concepts")
    print(f"\nTop 10 generalized concepts:")
    print(df_results['generalized'].value_counts().head(10))


if __name__ == '__main__':
    main()

