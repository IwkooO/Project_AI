"""
Concept extraction utilities for CBM geolocation.
"""
import re
from typing import Dict, List, Tuple
import html

def clean_html(text: str) -> str:
    """Remove HTML tags and decode entities from text."""
    if not text:
        return ""
    # Remove HTML tags
    clean = re.sub(r'<[^>]+>', '', text)
    # Decode HTML entities (e.g., &amp; -> &)
    clean = html.unescape(clean)
    return clean.strip()

def extract_concepts_from_dataset(dataset) -> Tuple[List[str], Dict[str, str]]:
    """
    Extract unique concepts and their descriptions from the dataset.
    
    Args:
        dataset: PanoramaCBMDataset instance
        
    Returns:
        Tuple of:
        - List of concept names (meta_names)
        - Dict mapping meta_name -> cleaned note description
    """
    meta_name_to_note = {}
    
    for sample in dataset.samples:
        meta_name = sample['meta_name']
        note = sample['note']
        
        # Only store if we haven't seen it or if the new note is longer/better
        # (assuming some samples might have empty notes for same meta_name)
        if meta_name not in meta_name_to_note or (note and len(note) > len(meta_name_to_note[meta_name])):
            meta_name_to_note[meta_name] = note
            
    # Clean the notes
    cleaned_concepts = {}
    for name, note in meta_name_to_note.items():
        cleaned_note = clean_html(note)
        # Fallback if note is empty after cleaning
        if not cleaned_note:
            cleaned_note = name
        cleaned_concepts[name] = cleaned_note
        
    # Sort for determinism
    sorted_concepts = sorted(cleaned_concepts.keys())
    
    return sorted_concepts, cleaned_concepts


