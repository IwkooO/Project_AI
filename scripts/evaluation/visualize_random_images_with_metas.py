#!/usr/bin/env python3
"""
Visualize random images with their metadata, cleaned descriptions, and mapped metas.

Creates clean, professional card-style visualizations.
"""

import argparse
import json
import re
import random
import textwrap
from pathlib import Path
from typing import Optional, Dict, Any

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from PIL import Image
from html import unescape


def clean_html_tags(text: str) -> str:
    """Remove HTML tags from text, handling NaN values."""
    if pd.isna(text) or text is None or text == '':
        return ''
    text = str(text)
    text = re.sub(r'<[^>]+>', '', text)
    text = unescape(text)
    text = ' '.join(text.split())
    return text.strip()


def resolve_image_path(image_path_str: str, project_root: Path) -> Optional[Path]:
    """Resolve image path, trying multiple possible locations."""
    path = Path(image_path_str)
    if path.exists():
        return path
    
    if not str(path).startswith('/'):
        scratch_path = Path('/scratch-shared/igodzwon/Project_AI') / path
        if scratch_path.exists():
            return scratch_path
        project_path = project_root / path
        if project_path.exists():
            return project_path
    
    if '/scratch-shared/igodzwon/Project_AI/' in str(path):
        relative_path = str(path).replace('/scratch-shared/igodzwon/Project_AI/', '')
        project_path = project_root / relative_path
        if project_path.exists():
            return project_path
    
    if '/home/igodzwon/Project_AI/' in str(path):
        relative_path = str(path).replace('/home/igodzwon/Project_AI/', '')
        project_path = project_root / relative_path
        if project_path.exists():
            return project_path
    
    return None


def load_meta_json(metas_dir: Path, pano_id: str) -> Optional[Dict[str, Any]]:
    """Load meta JSON file for a given pano_id."""
    meta_path = metas_dir / f"{pano_id}.json"
    if meta_path.exists():
        try:
            with open(meta_path, 'r') as f:
                return json.load(f)
        except Exception as e:
            return None
    return None


def create_visualization(
    df: pd.DataFrame,
    metas_dir: Path,
    output_path: Path,
    project_root: Path,
    num_images: int = 20,
    seed: int = 42,
    images_per_row: int = 2,
):
    """Create visualization of random images with their metadata."""
    
    random.seed(seed)
    np.random.seed(seed)
    
    # Filter to only rows where images exist
    print("Checking which images exist...")
    valid_rows = []
    for idx, (_, row) in enumerate(df.iterrows()):
        image_path = resolve_image_path(row['image_path'], project_root)
        if image_path is not None and image_path.exists():
            valid_rows.append((idx, row))
    
    print(f"Found {len(valid_rows)} images with valid paths out of {len(df)} total")
    
    if len(valid_rows) == 0:
        print("Error: No valid images found!")
        return
    
    # Sample random images
    num_samples = min(num_images, len(valid_rows))
    sampled_indices = random.sample(range(len(valid_rows)), num_samples)
    sample_rows = [valid_rows[i] for i in sampled_indices]
    
    # Calculate grid
    n_rows = (len(sample_rows) + images_per_row - 1) // images_per_row
    
    # Create figure - clean white background
    fig = plt.figure(figsize=(10 * images_per_row, 8 * n_rows), facecolor='white')
    
    # Add subtle title
    fig.suptitle(f'Dataset Sample: {len(sample_rows)} Random Images', 
                 fontsize=18, fontweight='600', color='#2c3e50', y=0.995,
                 fontfamily='sans-serif')
    
    for idx, (orig_idx, row) in enumerate(sample_rows):
        # Get metadata
        pano_id = row['pano_id']
        meta = load_meta_json(metas_dir, pano_id)
        
        # Resolve and load image
        image_path = resolve_image_path(row['image_path'], project_root)
        if image_path is None:
            continue
        
        try:
            img = Image.open(image_path).convert("RGB")
        except Exception as e:
            print(f"Warning: Could not load image {image_path}: {e}")
            continue
        
        # Extract all metadata
        note_cleaned = clean_html_tags(row.get('note', ''))
        if not note_cleaned:
            note_cleaned = "No description available."
        
        mapped_meta = row.get('generalized', '')
        if pd.isna(mapped_meta) or mapped_meta == '':
            mapped_meta = "unmapped"
        
        original_meta = None
        if meta and 'metaName' in meta:
            original_meta = meta['metaName']
        elif 'meta_name' in row and pd.notna(row['meta_name']):
            original_meta = row['meta_name']
        
        country = None
        if meta and 'country' in meta:
            country = meta['country']
        elif 'country' in row and pd.notna(row['country']):
            country = row['country']
        
        lat, lng = None, None
        if meta and 'lat' in meta and 'lng' in meta:
            lat, lng = meta['lat'], meta['lng']
        elif 'lat' in row and 'lng' in row and pd.notna(row['lat']) and pd.notna(row['lng']):
            lat, lng = row['lat'], row['lng']
        
        # Create subplot for this image
        row_pos = idx // images_per_row
        col_pos = idx % images_per_row
        
        # Calculate position (manual positioning for better control)
        card_width = 0.45
        card_height = 0.9 / n_rows
        margin_x = 0.025
        margin_y = 0.02
        
        left = margin_x + col_pos * (card_width + margin_x)
        bottom = 1 - (row_pos + 1) * (card_height + margin_y) - 0.02
        
        # Create main axis for this card
        ax = fig.add_axes([left, bottom, card_width, card_height], facecolor='#fafafa')
        ax.axis('off')
        
        # Add card border/shadow effect
        rect = mpatches.FancyBboxPatch(
            (0, 0), 1, 1, 
            boxstyle="round,pad=0.01,rounding_size=0.02",
            facecolor='white', edgecolor='#e0e0e0', linewidth=1.5,
            transform=ax.transAxes, zorder=0
        )
        ax.add_patch(rect)
        
        # Image subplot (takes most of the card)
        img_ax = ax.inset_axes([0.02, 0.35, 0.96, 0.62])
        img_ax.imshow(img)
        img_ax.axis('off')
        
        # ===== TEXT CONTENT =====
        text_left = 0.03
        
        # Title: Image number
        ax.text(text_left, 0.30, f"Sample #{idx + 1}", 
                transform=ax.transAxes, fontsize=12, fontweight='bold',
                color='#1a1a2e', va='top', fontfamily='sans-serif')
        
        # Original Meta (with label)
        y_pos = 0.24
        if original_meta:
            display_meta = original_meta if len(original_meta) <= 50 else original_meta[:47] + "..."
            ax.text(text_left, y_pos, "Meta: ", transform=ax.transAxes, 
                    fontsize=9, color='#666', va='top', fontweight='bold')
            ax.text(text_left + 0.07, y_pos, display_meta, transform=ax.transAxes,
                    fontsize=9, color='#c0392b', va='top', fontweight='600')
        
        # Mapped concept
        y_pos -= 0.045
        ax.text(text_left, y_pos, "Mapped: ", transform=ax.transAxes,
                fontsize=9, color='#666', va='top', fontweight='bold')
        ax.text(text_left + 0.10, y_pos, mapped_meta.replace('_', ' '), 
                transform=ax.transAxes, fontsize=9, color='#27ae60', 
                va='top', fontweight='600')
        
        # Location info
        y_pos -= 0.045
        loc_parts = []
        if country:
            loc_parts.append(country)
        if lat is not None and lng is not None:
            loc_parts.append(f"({lat:.3f}, {lng:.3f})")
        if loc_parts:
            ax.text(text_left, y_pos, " • ".join(loc_parts), transform=ax.transAxes,
                    fontsize=8, color='#7f8c8d', va='top')
        
        # Description (wrapped)
        y_pos -= 0.05
        ax.text(text_left, y_pos, "Description:", transform=ax.transAxes,
                fontsize=8, color='#666', va='top', fontweight='bold')
        
        y_pos -= 0.035
        max_chars = 180
        if len(note_cleaned) > max_chars:
            note_cleaned = note_cleaned[:max_chars] + "..."
        wrapped = textwrap.fill(note_cleaned, width=65)
        ax.text(text_left, y_pos, wrapped, transform=ax.transAxes,
                fontsize=7.5, color='#444', va='top', linespacing=1.4,
                fontfamily='sans-serif')
    
    # Save
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=150, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"Saved visualization to {output_path}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description='Visualize random images with their metadata'
    )
    parser.add_argument('--data-dir', type=str,
                        default='data/6921d7831744c5356b098bf7_balanced')
    parser.add_argument('--csv-file', type=str,
                        default='splits/dataset_test.csv')
    parser.add_argument('--output', type=str,
                        default='results/random_images_visualization.png')
    parser.add_argument('--num-images', type=int, default=20)
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--images-per-row', type=int, default=2)
    
    args = parser.parse_args()
    
    project_root = Path(__file__).parent.parent.parent
    data_dir = project_root / args.data_dir
    csv_path = data_dir / args.csv_file
    metas_dir = data_dir / 'metas'
    output_path = project_root / args.output
    
    if not csv_path.exists():
        print(f"Error: CSV file not found: {csv_path}")
        return
    
    if not metas_dir.exists():
        print(f"Error: Metas directory not found: {metas_dir}")
        return
    
    print(f"Loading dataset from {csv_path}")
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} rows")
    
    print(f"Creating visualization with {args.num_images} random images...")
    create_visualization(
        df=df,
        metas_dir=metas_dir,
        output_path=output_path,
        project_root=project_root,
        num_images=args.num_images,
        seed=args.seed,
        images_per_row=args.images_per_row,
    )
    
    print("Done!")


if __name__ == '__main__':
    main()
