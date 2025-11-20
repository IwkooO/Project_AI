#!/usr/bin/env python3
"""
PyTorch Dataset for CBM baseline training on panorama images.
"""

import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from torchvision.transforms import InterpolationMode
from transformers import AutoImageProcessor

import random
from tqdm import tqdm

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

geoguessrId = "6906237dc7731161a37282b2"
data_root = Path("data")
folder = data_root / geoguessrId
meta_folder = folder / "metas"
image_folder = folder / "panorama_processed"

def extract_image_size(processor: Optional[AutoImageProcessor] = None, image_size: Optional[Tuple[int, int]] = None) -> Tuple[int, int]:
    """
    Extract image size from processor or use provided size.
    
    Args:
        processor: Optional HuggingFace AutoImageProcessor instance
        image_size: Optional override for image size (width, height)
    
    Returns:
        Tuple of (width, height)
    """
    if image_size is not None:
        if isinstance(image_size, tuple):
            return image_size
        return (image_size, image_size)
    
    if processor is None:
        return (336, 336)  # Default fallback
    
    # Extract from processor.size
    if hasattr(processor, 'size') and processor.size is not None:
        if isinstance(processor.size, dict):
            w = processor.size.get('width') or processor.size.get('shortest_edge') or processor.size.get('height', 224)
            h = processor.size.get('height') or processor.size.get('shortest_edge') or processor.size.get('width', 224)
            return (w, h)
        elif isinstance(processor.size, (tuple, list)):
            if len(processor.size) >= 2:
                return tuple(processor.size[:2])  # (width, height)
            size_val = processor.size[0] if len(processor.size) > 0 else 224
            return (size_val, size_val)
        else:
            size_val = int(processor.size)
            return (size_val, size_val)
    
    # Extract from processor.crop_size
    if hasattr(processor, 'crop_size') and processor.crop_size is not None:
        if isinstance(processor.crop_size, dict):
            w = processor.crop_size.get('width') or processor.crop_size.get('height', 224)
            h = processor.crop_size.get('height') or processor.crop_size.get('width', 224)
            return (w, h)
        elif isinstance(processor.crop_size, (tuple, list)):
            if len(processor.crop_size) >= 2:
                return tuple(processor.crop_size[:2])
            size_val = processor.crop_size[0] if len(processor.crop_size) > 0 else 224
            return (size_val, size_val)
        else:
            size_val = int(processor.crop_size)
            return (size_val, size_val)
    
    return (224, 224)  # Default fallback


def get_transforms_from_processor(processor: Optional[AutoImageProcessor] = None, image_size: Optional[Tuple[int, int]] = None):
    """
    Create torchvision transforms from HuggingFace image processor.
    
    Args:
        processor: Optional HuggingFace AutoImageProcessor instance. If None, uses CLIP defaults.
        image_size: Optional override for image size (width, height). Defaults to (336, 336) if processor is None.
    
    Returns:
        torchvision.Compose transform pipeline
    """
    # Get size from processor or use provided (width, height) -> convert to (height, width) for torchvision
    width, height = extract_image_size(processor, image_size)
    target_size = (height, width)
    
    # Get normalization values from processor
    if processor is not None and hasattr(processor, 'image_mean') and processor.image_mean is not None:
        mean = processor.image_mean
        if isinstance(mean, list):
            mean = tuple(mean)
        elif not isinstance(mean, tuple):
            mean = tuple([mean] * 3)  # Convert scalar to tuple
    else:
        mean = CLIP_IMAGE_MEAN  # Fallback to CLIP defaults
    
    if processor is not None and hasattr(processor, 'image_std') and processor.image_std is not None:
        std = processor.image_std
        if isinstance(std, list):
            std = tuple(std)
        elif not isinstance(std, tuple):
            std = tuple([std] * 3)  # Convert scalar to tuple
    else:
        std = CLIP_IMAGE_STD  # Fallback to CLIP defaults
    
    # Ensure mean and std are tuples of length 3
    if len(mean) != 3:
        mean = tuple(mean[:3]) if len(mean) > 3 else tuple(list(mean) + [mean[-1]] * (3 - len(mean)))
    if len(std) != 3:
        std = tuple(std[:3]) if len(std) > 3 else tuple(list(std) + [std[-1]] * (3 - len(std)))
    
    # Create transform pipeline - resize directly to target size without cropping
    transform_list = [
        transforms.Resize(target_size, interpolation=InterpolationMode.BICUBIC),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ]
    
    return transforms.Compose(transform_list)


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
                 image_size: Optional[Tuple[int, int]] = None,
                 max_samples: Optional[int] = None,
                 country: Optional[str] = None,
                 require_coordinates: bool = False,
                 encoder_model: Optional[str] = None,
                 return_cartesian: bool = False):
        """
        Args:
            transform: Optional torchvision transforms (overrides encoder_model preprocessing)
            image_size: Target size for images (width, height) - used if encoder_model not provided
            max_samples: Limit number of samples for debugging
            country: Optional country name to filter samples by
            require_coordinates: Drop samples missing lat/lng
            encoder_model: HuggingFace model identifier (e.g., 'facebook/dinov2-base')
                          If provided, will use AutoImageProcessor to get correct preprocessing
            return_cartesian: If True, returns 3D Cartesian coordinates on unit sphere instead of normalized 2D
        """
        self.transform = transform
        self.max_samples = max_samples
        self.country = country
        self.require_coordinates = require_coordinates
        self.encoder_model = encoder_model
        self.return_cartesian = return_cartesian

        # Set up transforms based on encoder model or defaults
        if self.transform is None:
            processor = None
            if encoder_model is not None:
                try:
                    processor = AutoImageProcessor.from_pretrained(encoder_model)
                except Exception as e:
                    print(f"Warning: Could not load processor for {encoder_model}: {e}")
                    print("Falling back to default CLIP preprocessing")
            
            # Use get_transforms_from_processor for all cases (handles None processor)
            self.transform = get_transforms_from_processor(processor, image_size)
            self.image_size = extract_image_size(processor, image_size)
            
            # Log preprocessing info
            if processor is not None:
                mean = processor.image_mean if hasattr(processor, 'image_mean') and processor.image_mean is not None else CLIP_IMAGE_MEAN
                std = processor.image_std if hasattr(processor, 'image_std') and processor.image_std is not None else CLIP_IMAGE_STD
                # Convert to tuples for consistent display
                if isinstance(mean, list):
                    mean = tuple(mean)
                if isinstance(std, list):
                    std = tuple(std)
                print(f"Loaded processor for {encoder_model}")
            else:
                mean = CLIP_IMAGE_MEAN
                std = CLIP_IMAGE_STD
                print("Using default CLIP preprocessing")
            print(f"  Image size: {self.image_size}")
            print(f"  Normalization mean: {mean}")
            print(f"  Normalization std: {std}")
        else:
            # Transform provided explicitly, use provided image_size or default
            self.image_size = extract_image_size(None, image_size)

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
        skipped_no_country_match = 0
        skipped_no_image = 0
        skipped_no_coords = 0

        # Get all meta files
        meta_files = list(meta_folder.glob("*.json"))

        for meta_path in tqdm(meta_files, desc="Loading samples"):
            # Stop if we've reached max_samples (applied after filtering)
            if self.max_samples and len(samples) >= self.max_samples:
                break
                
            pano_id = meta_path.stem

            # Check if image exists
            image_path = image_folder / f"image_{pano_id}.jpg"
            if not image_path.exists():
                skipped_no_image += 1
                continue

            # Load meta data
            try:
                with meta_path.open() as f:
                    meta = json.load(f)

                # Check required fields exist
                if 'metaName' not in meta or 'country' not in meta:
                    continue

                # Filter by country if specified (case-insensitive, strip whitespace)
                if self.country is not None:
                    meta_country = str(meta['country']).strip()
                    filter_country = str(self.country).strip()
                    if meta_country.lower() != filter_country.lower():
                        skipped_no_country_match += 1
                        continue

                # Extract coordinates if available
                lat = meta.get('lat')
                lng = meta.get('lng')

                if self.require_coordinates and (lat is None or lng is None):
                    skipped_no_coords += 1
                    continue

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

        # Debug output if country filter is active
        if self.country is not None:
            print(f"Country filter '{self.country}' applied:")
            print(f"  - Loaded samples: {len(samples)}")
            print(f"  - Skipped (country mismatch): {skipped_no_country_match}")
            print(f"  - Skipped (no image): {skipped_no_image}")
            if self.require_coordinates:
                print(f"  - Skipped (no coordinates): {skipped_no_coords}")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, Dict]:
        """
        Returns:
            image_tensor: Processed image tensor
            concept_idx: Index of metaName (concept)
            target_idx: Index of country (target)
            coordinates_tensor: Normalized (lat, lng) tensor in [-1, 1]
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

        if self.return_cartesian:
            if sample['lat'] is not None and sample['lng'] is not None:
                coordinates = latlon_to_cartesian(sample['lat'], sample['lng'])
            else:
                coordinates = torch.tensor([float('nan')] * 3, dtype=torch.float32)
        else:
            coordinates = normalize_coordinates(sample['lat'], sample['lng'])

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

        return image, concept_idx, target_idx, coordinates, metadata

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

def create_splits_stratified(samples: List[Dict],
                  train_ratio: float = 0.7,
                  val_ratio: float = 0.15,
                  test_ratio: float = 0.15,
                  seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Split samples into train/val/test sets with per-concept stratification.

    Ensures that every concept (meta_name) contributes at least one example to the
    training set so that the concept head sees all labels during supervised training.

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

    rng = np.random.default_rng(seed)
    concept_to_samples: Dict[str, List[Dict]] = {}
    for sample in samples:
        concept_to_samples.setdefault(sample['meta_name'], []).append(sample)

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    for concept in sorted(concept_to_samples.keys()):
        concept_samples = concept_to_samples[concept]
        if len(concept_samples) == 1:
            train_samples.extend(concept_samples)
            continue

        shuffled_indices = rng.permutation(len(concept_samples))
        shuffled = [concept_samples[i] for i in shuffled_indices]

        n = len(shuffled)
        n_train = max(1, int(round(n * train_ratio)))
        n_val = int(round(n * val_ratio))
        if n_train + n_val > n:
            overflow = n_train + n_val - n
            if n_val >= overflow:
                n_val -= overflow
            else:
                n_train = max(1, n_train - (overflow - n_val))
                n_val = 0
        n_test = n - n_train - n_val

        if n_test < 0:
            n_val = max(0, n_val + n_test)
            n_test = 0

        if n_train == 0:
            if n_val > 0:
                n_train, n_val = 1, n_val - 1
            elif n_test > 0:
                n_train, n_test = 1, n_test - 1
            else:
                n_train = 1

        train_samples.extend(shuffled[:n_train])
        val_samples.extend(shuffled[n_train:n_train + n_val])
        test_samples.extend(shuffled[n_train + n_val:n_train + n_val + n_test])

    return train_samples, val_samples, test_samples

def create_splits(samples: List[Dict], train_ratio: float = 0.8, val_ratio: float = 0.1, test_ratio: float = 0.1, seed: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Splits the list of samples into train, validation, and test sets 
    by partitioning the set of unique concepts (meta_name) across the splits.
    This ensures NO CONCEPT LEAKAGE between the sets, which is CRITICAL for CBM evaluation.
    
    Args:
        samples: List of sample dictionaries
        train_ratio: Proportion for training set
        val_ratio: Proportion for validation set
        test_ratio: Proportion for test set
        seed: Random seed for reproducibility
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"
    
    # Set random seed for reproducibility
    random.seed(seed)
    
    # 1. Get all unique concepts (meta_name)
    unique_concepts = list(set(sample["meta_name"] for sample in samples))
    random.shuffle(unique_concepts)
    
    # 2. Split the *concepts* themselves into train/val/test groups
    n_concepts = len(unique_concepts)
    
    # Calculate split sizes for the concepts
    # Ensure at least one concept is in each set for robustness
    n_test_concepts = max(1, int(test_ratio * n_concepts))
    n_val_concepts = max(1, int(val_ratio * n_concepts))
    n_train_concepts = n_concepts - n_test_concepts - n_val_concepts
    
    # Ensure at least one concept in training set
    if n_train_concepts < 1:
        if n_val_concepts > 1:
            n_val_concepts -= 1
            n_train_concepts += 1
        elif n_test_concepts > 1:
            n_test_concepts -= 1
            n_train_concepts += 1
        else:
            n_train_concepts = 1
    
    test_concepts = set(unique_concepts[:n_test_concepts])
    val_concepts = set(unique_concepts[n_test_concepts:n_test_concepts + n_val_concepts])
    train_concepts = set(unique_concepts[n_test_concepts + n_val_concepts:])
    
    # 3. Filter samples based on their concept group
    all_train_samples = []
    all_val_samples = []
    all_test_samples = []
    
    for sample in samples:
        concept_name = sample["meta_name"]
        
        if concept_name in test_concepts:
            all_test_samples.append(sample)
        elif concept_name in val_concepts:
            all_val_samples.append(sample)
        elif concept_name in train_concepts:
            all_train_samples.append(sample)
            
    # 4. Final shuffle (important for dataloaders)
    random.shuffle(all_train_samples)
    random.shuffle(all_val_samples)
    random.shuffle(all_test_samples)
    
    return all_train_samples, all_val_samples, all_test_samples

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


def normalize_coordinates(lat: Optional[float], lng: Optional[float]) -> torch.Tensor:
    """Normalize coordinates to [-1, 1] range."""
    if lat is None or lng is None:
        return torch.tensor([float('nan'), float('nan')], dtype=torch.float32)

    lat_norm = float(lat) / 90.0
    lng_norm = float(lng) / 180.0
    return torch.tensor([lat_norm, lng_norm], dtype=torch.float32)

def latlon_to_cartesian(lat: float, lng: float) -> torch.Tensor:
    """
    Convert latitude and longitude to 3D Cartesian coordinates on the unit sphere.
    Args:
        lat: Latitude in degrees
        lng: Longitude in degrees
    Returns:
        tensor of shape (3,) containing (x, y, z)
    """
    lat_rad = np.deg2rad(lat)
    lng_rad = np.deg2rad(lng)
    
    x = np.cos(lat_rad) * np.cos(lng_rad)
    y = np.cos(lat_rad) * np.sin(lng_rad)
    z = np.sin(lat_rad)
    
    return torch.tensor([x, y, z], dtype=torch.float32)

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
    image, concept_idx, target_idx, coords, metadata = dataset[0]
    print(f"Image shape: {image.shape}")
    print(f"Concept idx: {concept_idx} -> {dataset.idx_to_concept[concept_idx]}")
    print(f"Target idx: {target_idx} -> {dataset.idx_to_country[target_idx]}")
    print(f"Coordinates: {coords}")
    print(f"Metadata keys: {list(metadata.keys())}")
