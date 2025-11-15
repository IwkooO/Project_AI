#!/usr/bin/env python3
"""
Preprocess panorama images: crop black borders and resize to uniform size.
"""

import numpy as np
from PIL import Image
from pathlib import Path
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

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

def _analyze_single_image(args):
    """Worker function to analyze a single image."""
    img_path, threshold = args
    img_path = Path(img_path)  # Convert to Path if string
    try:
        img = Image.open(img_path)
        width, height = img.size
        aspect_ratio = width / height
        
        # Check if image is completely black
        img_array = np.array(img)
        is_black = find_bounding_box_vectorized(img_array, threshold) is None
        
        # Check if image has black borders
        cropped = crop_black_borders(img, threshold)
        has_borders = False
        if cropped:
            cropped_width, cropped_height = cropped.size
            has_borders = (cropped_width < width or cropped_height < height)
            cropped.close()
        img.close()
        
        return {
            'size': (width, height),
            'aspect_ratio': aspect_ratio,
            'is_black': is_black,
            'has_borders': has_borders,
            'name': img_path.name
        }
    except Exception as e:
        return {'error': str(e), 'name': img_path.name}

def analyze_images(panorama_folder, threshold=10, num_workers=None):
    """
    Analyze images to find statistics and identify problematic ones.
    """
    panorama_path = Path(panorama_folder)
    image_files = list(panorama_path.glob("*.jpg"))
    
    if num_workers is None:
        num_workers = cpu_count()
    
    stats = {
        'total': len(image_files),
        'sizes': [],
        'black_images': [],
        'images_with_borders': [],
        'aspect_ratios': []
    }
    
    print(f"Analyzing {stats['total']} images with {num_workers} workers...")
    
    # Prepare arguments for worker function (convert Path to str for pickling)
    worker_args = [(str(img_path), threshold) for img_path in image_files]
    
    # Process in parallel
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_analyze_single_image, worker_args),
            total=len(worker_args),
            desc="Analyzing"
        ))
    
    # Aggregate results
    for result in results:
        if 'error' in result:
            print(f"Error analyzing {result['name']}: {result['error']}")
            continue
        
        stats['sizes'].append(result['size'])
        stats['aspect_ratios'].append(result['aspect_ratio'])
        
        if result['is_black']:
            stats['black_images'].append(result['name'])
        
        if result['has_borders']:
            stats['images_with_borders'].append(result['name'])
    
    return stats

def _get_cropped_size(args):
    """Worker function to get cropped size of a single image."""
    img_path, threshold = args
    img_path = Path(img_path)  # Convert to Path if string
    try:
        img = Image.open(img_path)
        cropped = crop_black_borders(img, threshold)
        if cropped:
            size = cropped.size
            cropped.close()
        else:
            size = None
        img.close()
        return size
    except Exception as e:
        return None

def _process_single_image(args):
    """Worker function to process a single image."""
    img_path, threshold, target_size, backup_path, output_path, backup, inplace = args
    img_path = Path(img_path)  # Convert to Path if string
    backup_path = Path(backup_path) if backup_path else None
    output_path = Path(output_path) if output_path else None
    try:
        img = Image.open(img_path)
        
        # Backup original if requested
        if backup and not inplace and backup_path:
            backup_file = backup_path / img_path.name
            if not backup_file.exists():
                img.save(backup_file, "JPEG")
        
        # Crop black borders
        cropped = crop_black_borders(img, threshold)
        img.close()
        
        if cropped is None:
            return {'status': 'skipped', 'name': img_path.name, 'reason': 'completely black'}
        
        # Resize to target size
        resized = cropped.resize(target_size, Image.LANCZOS)
        cropped.close()
        
        # Save processed image
        if inplace:
            output_file = img_path
        else:
            output_file = output_path / img_path.name
        
        resized.save(output_file, "JPEG", quality=95)
        resized.close()
        
        return {'status': 'processed', 'name': img_path.name}
    except Exception as e:
        return {'status': 'error', 'name': img_path.name, 'error': str(e)}

def preprocess_images(panorama_folder, output_folder=None, target_size=None, threshold=10, backup=True, inplace=False, num_workers=None):
    """
    Preprocess images: crop black borders and resize to uniform size.
    
    Args:
        panorama_folder: Path to folder containing panorama images
        output_folder: Path to output folder (default: panorama_folder + '_processed')
        target_size: Target size as (width, height). If None, use median size.
        threshold: Black pixel threshold (0-255)
        backup: Whether to backup original images
        inplace: If True, overwrite original images (output_folder ignored)
        num_workers: Number of parallel workers (default: CPU count)
    """
    if num_workers is None:
        num_workers = cpu_count()
    
    panorama_path = Path(panorama_folder)
    
    if inplace:
        output_path = panorama_path
        if backup:
            backup_path = panorama_path.parent / f"{panorama_path.name}_original"
            backup_path.mkdir(exist_ok=True)
        else:
            backup_path = None
    else:
        if output_folder is None:
            output_path = panorama_path.parent / f"{panorama_path.name}_processed"
        else:
            output_path = Path(output_folder)
        output_path.mkdir(exist_ok=True, parents=True)
        
        if backup:
            backup_path = panorama_path.parent / f"{panorama_path.name}_original"
            backup_path.mkdir(exist_ok=True)
        else:
            backup_path = None
    
    image_files = list(panorama_path.glob("*.jpg"))
    
    # First pass: collect sizes only (parallelized)
    print(f"First pass: analyzing sizes with {num_workers} workers...")
    worker_args = [(str(img_path), threshold) for img_path in image_files]
    
    cropped_sizes = []
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_get_cropped_size, worker_args),
            total=len(worker_args),
            desc="Analyzing sizes"
        ))
    
    cropped_sizes = [size for size in results if size is not None]
    
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
    
    # Second pass: process and save images (parallelized)
    print(f"Second pass: processing and saving images with {num_workers} workers...")
    worker_args = [
        (str(img_path), threshold, target_size, str(backup_path) if backup_path else None, str(output_path), backup, inplace)
        for img_path in image_files
    ]
    
    processed_count = 0
    skipped_count = 0
    
    with Pool(num_workers) as pool:
        results = list(tqdm(
            pool.imap(_process_single_image, worker_args),
            total=len(worker_args),
            desc="Processing"
        ))
    
    # Aggregate results
    for result in results:
        if result['status'] == 'processed':
            processed_count += 1
        elif result['status'] == 'skipped':
            skipped_count += 1
            if 'reason' in result:
                print(f"Warning: {result['name']} - {result['reason']}")
        elif result['status'] == 'error':
            skipped_count += 1
            print(f"Error processing {result['name']}: {result.get('error', 'Unknown error')}")
    
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
    parser.add_argument('--num-workers', type=int, default=None,
                       help='Number of parallel workers (default: CPU count)')
    
    args = parser.parse_args()
    
    if args.analyze_only:
        stats = analyze_images(args.panorama_folder, args.threshold, args.num_workers)
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
            inplace=args.inplace,
            num_workers=args.num_workers
        )

if __name__ == '__main__':
    main()
