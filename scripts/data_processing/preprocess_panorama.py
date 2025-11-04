#!/usr/bin/env python3
"""
Preprocess panorama images: crop black borders and resize to uniform size.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse

def find_bounding_box_vectorized(image_array, threshold=10):
    """
    Find bounding box of non-black content using vectorized numpy operations.
    Returns (top, left, bottom, right) or None if image is completely black.
    """
    # Handle grayscale images
    if len(image_array.shape) == 2:
        # Grayscale: check if pixel value >= threshold
        non_black = image_array >= threshold
    else:
        # Color: check if any channel >= threshold
        non_black = np.any(image_array[:, :, :3] >= threshold, axis=2)
    
    # Check if image is completely black
    if not np.any(non_black):
        return None
    
    # Find bounding box using numpy operations
    rows = np.any(non_black, axis=1)
    cols = np.any(non_black, axis=0)
    
    top = np.argmax(rows)
    bottom = len(rows) - np.argmax(rows[::-1])
    left = np.argmax(cols)
    right = len(cols) - np.argmax(cols[::-1])
    
    return (top, left, bottom, right)

def crop_black_borders(image, threshold=10):
    """
    Crop black borders from image using vectorized operations.
    Returns cropped image or None if image is completely black.
    """
    img_array = np.array(image)
    
    bbox = find_bounding_box_vectorized(img_array, threshold)
    if bbox is None:
        return None
    
    top, left, bottom, right = bbox
    
    # Crop the image
    cropped = image.crop((left, top, right, bottom))
    return cropped

def analyze_images(panorama_folder, threshold=10):
    """
    Analyze images to find statistics and identify problematic ones.
    """
    panorama_path = Path(panorama_folder)
    image_files = list(panorama_path.glob("*.jpg"))
    
    stats = {
        'total': len(image_files),
        'sizes': [],
        'black_images': [],
        'images_with_borders': [],
        'aspect_ratios': []
    }
    
    print(f"Analyzing {stats['total']} images...")
    
    for img_path in tqdm(image_files, desc="Analyzing"):
        try:
            img = Image.open(img_path)
            width, height = img.size
            stats['sizes'].append((width, height))
            stats['aspect_ratios'].append(width / height)
            
            # Check if image is completely black
            img_array = np.array(img)
            if find_bounding_box_vectorized(img_array, threshold) is None:
                stats['black_images'].append(img_path.name)
            
            # Check if image has black borders (cropped size < original size)
            cropped = crop_black_borders(img, threshold)
            if cropped:
                cropped_width, cropped_height = cropped.size
                if cropped_width < width or cropped_height < height:
                    stats['images_with_borders'].append(img_path.name)
        except Exception as e:
            print(f"Error analyzing {img_path.name}: {e}")
    
    return stats

def preprocess_images(panorama_folder, output_folder=None, target_size=None, threshold=10, backup=True, inplace=False):
    """
    Preprocess images: crop black borders and resize to uniform size.
    
    Args:
        panorama_folder: Path to folder containing panorama images
        output_folder: Path to output folder (default: panorama_folder + '_processed')
        target_size: Target size as (width, height). If None, use median size.
        threshold: Black pixel threshold (0-255)
        backup: Whether to backup original images
        inplace: If True, overwrite original images (output_folder ignored)
    """
    panorama_path = Path(panorama_folder)
    
    if inplace:
        output_path = panorama_path
        if backup:
            backup_path = panorama_path.parent / f"{panorama_path.name}_original"
            backup_path.mkdir(exist_ok=True)
    else:
        if output_folder is None:
            output_path = panorama_path.parent / f"{panorama_path.name}_processed"
        else:
            output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if backup:
            backup_path = panorama_path.parent / f"{panorama_path.name}_original"
            backup_path.mkdir(exist_ok=True)
    
    image_files = list(panorama_path.glob("*.jpg"))
    
    # First pass: collect sizes only (memory-efficient)
    print("First pass: analyzing sizes...")
    cropped_sizes = []
    
    for img_path in tqdm(image_files, desc="Analyzing"):
        try:
            img = Image.open(img_path)
            cropped = crop_black_borders(img, threshold)
            if cropped:
                cropped_sizes.append(cropped.size)
            img.close()  # Explicitly close to free memory
            del img, cropped
        except Exception as e:
            print(f"Error analyzing {img_path.name}: {e}")
    
    if not cropped_sizes:
        print("No valid images found after cropping!")
        return
    
    # Determine target size
    if target_size is None:
        # Use median size
        widths = [w for w, h in cropped_sizes]
        heights = [h for w, h in cropped_sizes]
        target_size = (int(np.median(widths)), int(np.median(heights)))
    
    print(f"Target size: {target_size[0]}x{target_size[1]}")
    
    # Second pass: process and save images immediately (memory-efficient)
    print("Second pass: processing and saving images...")
    processed_count = 0
    skipped_count = 0
    
    for img_path in tqdm(image_files, desc="Processing"):
        try:
            img = Image.open(img_path)
            
            # Backup original if requested
            if backup and not inplace:
                backup_file = backup_path / img_path.name
                if not backup_file.exists():
                    img.save(backup_file, "JPEG")
            
            # Crop black borders
            cropped = crop_black_borders(img, threshold)
            img.close()  # Close original immediately
            del img
            
            if cropped is None:
                print(f"Warning: {img_path.name} is completely black, skipping...")
                skipped_count += 1
                continue
            
            # Resize to target size (using LANCZOS for high quality)
            resized = cropped.resize(target_size, Image.LANCZOS)
            cropped.close()  # Close cropped immediately
            del cropped
            
            # Save processed image
            if inplace:
                output_file = img_path
            else:
                output_file = output_path / img_path.name
            
            resized.save(output_file, "JPEG", quality=95)
            resized.close()  # Close resized immediately
            del resized
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {img_path.name}: {e}")
            skipped_count += 1
    
    print(f"\nProcessing complete!")
    print(f"Processed: {processed_count}")
    print(f"Skipped: {skipped_count}")
    print(f"Output folder: {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Preprocess panorama images')
    parser.add_argument('--panorama-folder', type=str, 
                       default='data/6906237dc7731161a37282b2/panorama',
                       help='Path to panorama folder')
    parser.add_argument('--output-folder', type=str, default=None,
                       help='Output folder (default: panorama_folder + _processed)')
    parser.add_argument('--target-size', type=int, nargs=2, default=None,
                       metavar=('WIDTH', 'HEIGHT'),
                       help='Target size for all images (default: median size)')
    parser.add_argument('--threshold', type=int, default=10,
                       help='Black pixel threshold (0-255, default: 10)')
    parser.add_argument('--no-backup', action='store_true',
                       help='Do not backup original images')
    parser.add_argument('--inplace', action='store_true',
                       help='Overwrite original images in place')
    parser.add_argument('--analyze-only', action='store_true',
                       help='Only analyze images, do not process')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        stats = analyze_images(args.panorama_folder, args.threshold)
        print("\n=== Analysis Results ===")
        print(f"Total images: {stats['total']}")
        print(f"Completely black images: {len(stats['black_images'])}")
        print(f"Images with black borders: {len(stats['images_with_borders'])}")
        
        if stats['sizes']:
            widths = [w for w, h in stats['sizes']]
            heights = [h for w, h in stats['sizes']]
            print(f"\nSize statistics:")
            print(f"  Width: min={min(widths)}, max={max(widths)}, median={int(np.median(widths))}")
            print(f"  Height: min={min(heights)}, max={max(heights)}, median={int(np.median(heights))}")
            print(f"  Aspect ratio: min={min(stats['aspect_ratios']):.2f}, max={max(stats['aspect_ratios']):.2f}, median={np.median(stats['aspect_ratios']):.2f}")
        
        if stats['black_images']:
            print(f"\nCompletely black images (first 10):")
            for img_name in stats['black_images'][:10]:
                print(f"  - {img_name}")
        
        if stats['images_with_borders']:
            print(f"\nImages with black borders (first 10):")
            for img_name in stats['images_with_borders'][:10]:
                print(f"  - {img_name}")
    else:
        target_size = tuple(args.target_size) if args.target_size else None
        preprocess_images(
            args.panorama_folder,
            args.output_folder,
            target_size,
            args.threshold,
            backup=not args.no_backup,
            inplace=args.inplace
        )

if __name__ == '__main__':
    main()
