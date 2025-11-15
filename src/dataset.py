#!/usr/bin/env python3
"""
PyTorch Dataset for CBM baseline training on panorama images.
"""

import torch
from torch.utils.data import Dataset
from pathlib import Path
import json
from PIL import Image
import numpy as np
from typing import Dict, List, Tuple, Optional
from tqdm import tqdm

geoguessrId = "6906237dc7731161a37282b2"
data_root = Path("data")
folder = data_root / geoguessrId
meta_folder = folder / "metas"
image_folder = folder / "panorama_processed"

class PanoramaCBMDataset(Dataset):
    """
    Dataset for CBM training with panorama images.

    Returns:
        image_tensor: Processed image tensor
        concept_idx: Index of metaName (concept)
        target_idx: Index of country (target)
        metadata: Dict with original strings and coordinates
    """

    def __init__(self,
                 transform=None,
                 image_size: Tuple[int, int] = (224, 224),
                 max_samples: Optional[int] = None,
                 country: Optional[str] = None):
        """
        Args:
            transform: Optional torchvision transforms
            image_size: Target size for images (width, height)
            max_samples: Limit number of samples for debugging
            country: Optional country name to filter samples by
        """
        self.transform = transform
        self.image_size = image_size
        self.max_samples = max_samples
        self.country = country

        # Load and filter samples
        self.samples = self._load_samples()

        # Build encoders
        self.concept_to_idx, self.idx_to_concept = get_concept_to_idx(self.samples)
        self.country_to_idx, self.idx_to_country = get_country_to_idx(self.samples)

        print(f"Loaded {len(self.samples)} samples")
        print(f"Concepts: {len(self.concept_to_idx)}")
        print(f"Countries: {len(self.country_to_idx)}")

    def _load_samples(self) -> List[Dict]:
        """Load meta files and filter to samples with existing images."""
        samples = []

        # Get all meta files
        meta_files = list(meta_folder.glob("*.json"))

        if self.max_samples:
            meta_files = meta_files[:self.max_samples]

        for meta_path in tqdm(meta_files, desc="Loading samples"):
            pano_id = meta_path.stem

            # Check if image exists
            image_path = image_folder / f"image_{pano_id}.jpg"
            if not image_path.exists():
                continue

            # Load meta data
            try:
                with meta_path.open() as f:
                    meta = json.load(f)

                # Check required fields exist
                if 'metaName' not in meta or 'country' not in meta:
                    continue

                # Filter by country if specified
                if self.country is not None and meta['country'] != self.country:
                    continue

                # Extract coordinates if available
                lat = meta.get('lat')
                lng = meta.get('lng')

                sample = {
                    'pano_id': pano_id,
                    'image_path': image_path,
                    'meta_path': meta_path,
                    'meta_name': meta['metaName'],
                    'country': meta['country'],
                    'lat': lat,
                    'lng': lng,
                    'note': meta.get('note', ''),
                    'images': meta.get('images', [])
                }

                samples.append(sample)

            except (json.JSONDecodeError, KeyError) as e:
                print(f"Error loading {meta_path}: {e}")
                continue

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, Dict]:
        """
        Returns:
            image_tensor: Processed image tensor
            concept_idx: Index of metaName (concept)
            target_idx: Index of country (target)
            metadata: Dict with sample information
        """
        sample = self.samples[idx]

        # Load and process image
        image = Image.open(sample['image_path']).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            # Default processing: resize and convert to tensor
            image = image.resize(self.image_size, Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Encode concept and target
        concept_idx = self.concept_to_idx[sample['meta_name']]
        target_idx = self.country_to_idx[sample['country']]

        # Metadata dict
        metadata = {
            'pano_id': sample['pano_id'],
            'meta_name': sample['meta_name'],
            'country': sample['country'],
            'lat': sample['lat'],
            'lng': sample['lng'],
            'note': sample['note'],
            'images': sample['images']
        }

        return image, concept_idx, target_idx, metadata

def get_concept_to_idx(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mapping from metaName strings to indices."""
    meta_names = sorted(set(s['meta_name'] for s in samples))
    concept_to_idx = {name: i for i, name in enumerate(meta_names)}
    idx_to_concept = {i: name for name, i in concept_to_idx.items()}
    return concept_to_idx, idx_to_concept

def get_country_to_idx(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mapping from country strings to indices."""
    countries = sorted(set(s['country'] for s in samples))
    country_to_idx = {country: i for i, country in enumerate(countries)}
    idx_to_country = {i: country for country, i in country_to_idx.items()}
    return country_to_idx, idx_to_country

def create_splits(samples: List[Dict],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/val/test sets.

    Args:
        samples: List of sample dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    np.random.seed(seed)

    # Shuffle samples
    indices = np.random.permutation(len(samples))

    n_train = int(len(samples) * train_ratio)
    n_val = int(len(samples) * val_ratio)

    train_indices = indices[:n_train]
    val_indices = indices[n_train:n_train + n_val]
    test_indices = indices[n_train + n_val:]

    train_samples = [samples[i] for i in train_indices]
    val_samples = [samples[i] for i in val_indices]
    test_samples = [samples[i] for i in test_indices]

    return train_samples, val_samples, test_samples

def get_statistics(samples: List[Dict]) -> Dict:
    """
    Compute dataset statistics.

    Args:
        samples: List of sample dictionaries

    Returns:
        Dictionary with statistics
    """
    stats = {
        'total_samples': len(samples),
        'countries': {},
        'concepts': {},
        'samples_per_country': {},
        'samples_per_concept': {},
        'coordinate_coverage': 0
    }

    for sample in samples:
        country = sample['country']
        concept = sample['meta_name']

        # Count countries
        if country not in stats['countries']:
            stats['countries'][country] = 0
        stats['countries'][country] += 1

        # Count concepts
        if concept not in stats['concepts']:
            stats['concepts'][concept] = 0
        stats['concepts'][concept] += 1

        # Check coordinates
        if sample['lat'] is not None and sample['lng'] is not None:
            stats['coordinate_coverage'] += 1

    # Sort by frequency
    stats['countries'] = dict(sorted(stats['countries'].items(), key=lambda x: x[1], reverse=True))
    stats['concepts'] = dict(sorted(stats['concepts'].items(), key=lambda x: x[1], reverse=True))

    stats['num_countries'] = len(stats['countries'])
    stats['num_concepts'] = len(stats['concepts'])
    stats['coordinate_coverage_pct'] = stats['coordinate_coverage'] / len(samples) * 100

    return stats

def print_statistics(stats: Dict):
    """Pretty print dataset statistics."""
    print(f"Dataset Statistics:")
    print(f"  Total samples: {stats['total_samples']}")
    print(f"  Number of countries: {stats['num_countries']}")
    print(f"  Number of concepts: {stats['num_concepts']}")
    print(f"  Coordinate coverage: {stats['coordinate_coverage']}/{stats['total_samples']} ({stats['coordinate_coverage_pct']:.1f}%)")
    print()

    print("Top 10 countries:")
    for i, (country, count) in enumerate(list(stats['countries'].items())[:10]):
        print(f"  {i+1}. {country}: {count}")
    print()

    print("Top 10 concepts:")
    for i, (concept, count) in enumerate(list(stats['concepts'].items())[:10]):
        print(f"  {i+1}. {concept}: {count}")

class SubsetDataset(Dataset):
    """
    Dataset wrapper for subsets (train/val/test splits).
    """

    def __init__(self, parent_dataset: PanoramaCBMDataset, samples: List[Dict]):
        self.parent_dataset = parent_dataset
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Find the index in the parent dataset
        sample = self.samples[idx]
        parent_idx = self.parent_dataset.samples.index(sample)
        return self.parent_dataset[parent_idx]

if __name__ == "__main__":
    # Test the dataset
    dataset = PanoramaCBMDataset(country="Australia")  

    # Test statistics
    stats = get_statistics(dataset.samples)
    print_statistics(stats)
    
    # print an example sample
    print(dataset.samples[0])
    print(f"Image shape: {dataset.samples[0]['image_path']}")
    print(f"Concept idx: {dataset.samples[0]['meta_name']}")
    print(f"Target idx: {dataset.samples[0]['country']}")
    print(f"Metadata keys: {list(dataset.samples[0].keys())}")

    # Test splits
    train_samples, val_samples, test_samples = create_splits(dataset.samples)
    print(f"Split sizes: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")

    # Test subset datasets
    train_dataset = SubsetDataset(dataset, train_samples)
    val_dataset = SubsetDataset(dataset, val_samples)
    test_dataset = SubsetDataset(dataset, test_samples)

    print(f"Subset dataset sizes: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

    # Test __getitem__
    image, concept_idx, target_idx, metadata = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Concept idx: {concept_idx} -> {dataset.idx_to_concept[concept_idx]}")
    print(f"Target idx: {target_idx} -> {dataset.idx_to_country[target_idx]}")
    print(f"Metadata keys: {list(metadata.keys())}")
