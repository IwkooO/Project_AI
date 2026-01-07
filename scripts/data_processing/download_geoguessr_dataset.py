#!/usr/bin/env python3
"""
Download GeoGuessr dataset from HuggingFace to local cache.

This downloads the entire dataset once, avoiding rate limits during evaluation.

Usage:
    python scripts/data_processing/download_geoguessr_dataset.py
"""

import argparse
import sys
from pathlib import Path
import time

# Add project root
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def download_dataset(split: str = "train", use_hf_transfer: bool = True):
    """Download the GeoGuessr dataset."""
    import os
    
    # Set up cache
    os.environ.setdefault("HF_HOME", str(Path.home() / ".cache" / "huggingface"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(Path(os.environ["HF_HOME"]) / "datasets"))
    
    if use_hf_transfer:
        try:
            import hf_transfer
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "1"
            print("✅ Using hf_transfer for faster downloads")
        except ImportError:
            print("⚠️  hf_transfer not installed. Install with: pip install hf_transfer")
            os.environ["HF_HUB_ENABLE_HF_TRANSFER"] = "0"
    
    from datasets import load_dataset
    
    dataset_name = "fren-gor/geoguessr-locations"
    
    print("="*80)
    print("DOWNLOADING GEOGUESSR DATASET")
    print("="*80)
    print(f"Dataset: {dataset_name}")
    print(f"Split: {split}")
    print(f"Cache location: {os.environ['HF_DATASETS_CACHE']}")
    print("="*80)
    
    # Check if already cached
    cache_dir = Path(os.environ["HF_DATASETS_CACHE"])
    dataset_cache = cache_dir / "fren-gor___geoguessr-locations"
    
    if dataset_cache.exists():
        print(f"\n📂 Found existing cache at: {dataset_cache}")
        print("   Dataset may already be downloaded. Checking...")
    
    # Download with retry logic
    max_retries = 5
    retry_delay = 60
    
    for attempt in range(max_retries):
        try:
            print(f"\n🔄 Download attempt {attempt + 1}/{max_retries}...")
            
            # Load dataset (this will download if not cached)
            print("   Loading dataset (this may take a while for first download)...")
            hf_dataset = load_dataset(
                dataset_name,
                split=split,
                trust_remote_code=True,
                streaming=False,  # Download full dataset
            )
            
            print(f"\n✅ Successfully loaded dataset!")
            print(f"   Split: {split}")
            print(f"   Number of samples: {len(hf_dataset):,}")
            print(f"   Cached at: {dataset_cache}")
            
            # Print some info about the dataset
            print(f"\n📊 Dataset info:")
            if len(hf_dataset) > 0:
                sample = hf_dataset[0]
                print(f"   Columns: {list(sample.keys())}")
                if "lat" in sample and "lng" in sample:
                    print(f"   Sample location: ({sample['lat']:.4f}, {sample['lng']:.4f})")
                if "country" in sample:
                    print(f"   Sample country: {sample.get('country', 'N/A')}")
            
            return hf_dataset
            
        except Exception as e:
            error_str = str(e)
            if "429" in error_str or "rate limit" in error_str.lower() or "Too Many Requests" in error_str:
                if attempt < max_retries - 1:
                    wait_time = retry_delay * (2 ** attempt)
                    print(f"\n⚠️  Rate limited. Waiting {wait_time} seconds before retry...")
                    print(f"   (Rate limit resets every 5 minutes)")
                    time.sleep(wait_time)
                else:
                    print(f"\n❌ Failed after {max_retries} attempts due to rate limiting.")
                    print("\n💡 Solutions:")
                    print("   1. Wait 5 minutes and try again")
                    print("   2. Upgrade to HuggingFace PRO: https://hf.co/pricing")
                    print("   3. Try downloading at a different time (off-peak hours)")
                    raise
            else:
                print(f"\n❌ Error downloading dataset: {e}")
                raise
    
    return None


def main():
    parser = argparse.ArgumentParser(description="Download GeoGuessr dataset")
    parser.add_argument("--split", type=str, default="train",
                       help="Dataset split to download (default: train)")
    parser.add_argument("--no-hf-transfer", action="store_true",
                       help="Disable hf_transfer (slower but more compatible)")
    
    args = parser.parse_args()
    
    try:
        dataset = download_dataset(split=args.split, use_hf_transfer=not args.no_hf_transfer)
        print("\n" + "="*80)
        print("✅ Download complete! Dataset is now cached.")
        print("   You can now run evaluation without rate limit issues.")
        print("="*80)
    except Exception as e:
        print(f"\n❌ Download failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

