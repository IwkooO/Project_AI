#!/usr/bin/env python3
"""
PyTorch Dataset for CBM baseline training on panorama images.
"""

from functools import lru_cache
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import html
import re

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from PIL import Image as PILImage
from transformers import AutoImageProcessor
import os

import random
from tqdm import tqdm
import argparse
import pandas as pd
from bs4 import BeautifulSoup

CLIP_IMAGE_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_IMAGE_STD = (0.26862954, 0.26130258, 0.27577711)

# Torchvision compatibility: older torchvision may not expose InterpolationMode.
# Resize/RandomResizedCrop accept PIL interpolation enums/ints, so use PIL constants.
_BICUBIC = getattr(PILImage, "BICUBIC", getattr(PILImage, "Resampling").BICUBIC)

@lru_cache(maxsize=10000)
def parse_html_note(html_text: str) -> str:
    """
    Parse HTML note and extract plain text.
    
    Args:
        html_text: HTML string (may be plain text if not HTML)
        
    Returns:
        Plain text extracted from HTML, or original text if not HTML
    """
    if not html_text or not isinstance(html_text, str):
        return ""
    
    # Strip whitespace
    html_text = html_text.strip()
    
    if not html_text:
        return ""
    
    # Check if it looks like HTML (contains tags)
    if not re.search(r'<[^>]+>', html_text):
        # Not HTML, just return decoded text
        return html.unescape(html_text).strip()
    
    # Parse HTML
    soup = BeautifulSoup(html_text, 'html.parser')
    # Get text and clean up whitespace
    text = soup.get_text(separator=' ', strip=True)
    # Decode HTML entities
    text = html.unescape(text)
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text
    
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


def get_transforms_from_processor(
    processor: Optional[AutoImageProcessor] = None, 
    image_size: Optional[Tuple[int, int]] = None,
    is_training: bool = True,
    augmentation_strength: str = "medium",
):
    """
    Create torchvision transforms from HuggingFace image processor.
    
    Args:
        processor: Optional HuggingFace AutoImageProcessor instance. If None, uses CLIP defaults.
        image_size: Optional override for image size (width, height). Defaults to (336, 336) if processor is None.
        is_training: If True, apply data augmentation; if False, only apply resize + normalize.
        augmentation_strength: One of "none", "light", "medium", "strong" for augmentation intensity.
    
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
    
    # Validation/test transforms (no augmentation)
    if not is_training or augmentation_strength == "none":
        transform_list = [
            transforms.Resize(target_size, interpolation=_BICUBIC),
            transforms.CenterCrop(target_size),  # Center crop for consistency
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ]
        return transforms.Compose(transform_list)
    
    # Training transforms with augmentation
    # Build augmentation pipeline based on strength
    transform_list = []
    
    # ========== Geometric Augmentations ==========
    if augmentation_strength == "light":
        # Light: simple resize + horizontal flip
        transform_list.extend([
            transforms.Resize(target_size, interpolation=_BICUBIC),
            transforms.RandomHorizontalFlip(p=0.5),
        ])
    elif augmentation_strength == "medium":
        # Medium: random crop + flip + light color jitter
        transform_list.extend([
            transforms.RandomResizedCrop(
                target_size,
                scale=(0.8, 1.0),  # Crop 80-100% of image
                ratio=(0.9, 1.1),  # Slight aspect ratio variation
                interpolation=_BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.2,
                contrast=0.2,
                saturation=0.2,
                hue=0.05,
            ),
        ])
    elif augmentation_strength == "strong":
        # Strong: aggressive augmentation for maximum regularization
        transform_list.extend([
            transforms.RandomResizedCrop(
                target_size,
                scale=(0.6, 1.0),  # More aggressive crop (60-100%)
                ratio=(0.8, 1.2),  # More aspect ratio variation
                interpolation=_BICUBIC,
            ),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.ColorJitter(
                brightness=0.3,
                contrast=0.3,
                saturation=0.3,
                hue=0.1,
            ),
            transforms.RandomGrayscale(p=0.1),  # Occasional grayscale
            # Note: RandomErasing is applied after ToTensor
        ])
    
    # ========== Convert to Tensor + Normalize ==========
    transform_list.extend([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])
    
    # ========== Post-tensor augmentations (strong only) ==========
    if augmentation_strength == "strong":
        transform_list.append(
            transforms.RandomErasing(
                p=0.1,  # 10% probability
                scale=(0.02, 0.15),  # Erase 2-15% of image
                ratio=(0.3, 3.3),
                value='random',  # Fill with random values
            )
        )
    
    return transforms.Compose(transform_list)


def get_train_transforms(processor: Optional[AutoImageProcessor] = None, image_size: Optional[Tuple[int, int]] = None):
    """Get training transforms with medium augmentation."""
    return get_transforms_from_processor(processor, image_size, is_training=True, augmentation_strength="medium")


def get_val_transforms(processor: Optional[AutoImageProcessor] = None, image_size: Optional[Tuple[int, int]] = None):
    """Get validation transforms (no augmentation)."""
    return get_transforms_from_processor(processor, image_size, is_training=False, augmentation_strength="none")


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
                 return_cartesian: bool = False,
                 use_normalized_coordinates: bool = False,
                 geoguessr_id: str = "6906237dc7731161a37282b2",
                 data_root: Optional[Path] = None,
                 csv_path: Optional[str] = None):
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
            use_normalized_coordinates: If True, returns coordinates normalized to [-1, 1]. If False, returns raw (lat, lng).
            geoguessr_id: GeoGuessr map ID (used only if csv_path is None)
            data_root: Root directory for data (defaults to "data", used only if csv_path is None)
            csv_path: Optional path to CSV file containing 'image_path', 'meta_name', 'country', 'lat', 'lng' columns.
                     If provided, loads data from CSV instead of folder structure.
        """
        self.transform = transform
        self.max_samples = max_samples
        self.country = country
        self.require_coordinates = require_coordinates
        self.encoder_model = encoder_model
        self.return_cartesian = return_cartesian
        self.use_normalized_coordinates = use_normalized_coordinates
        self.geoguessr_id = geoguessr_id
        self.csv_path = csv_path
        
        # Set up folder structure (only used if csv_path is None)
        if csv_path is None:
            if data_root is None:
                data_root = Path("data")
            self.data_root = Path(data_root)
            self.folder = self.data_root / geoguessr_id
            self.meta_folder = self.folder / "metas"
            
            # Check if panorama_processed folder exists, if not use panorama folder
            if os.path.exists(self.folder / "panorama_processed"):
                self.image_folder = self.folder / "panorama_processed"
            else:
                self.image_folder = self.folder / "panorama"
        else:
            # CSV mode: set minimal folder structure (may not be used)
            if data_root is None:
                data_root = Path("data")
            self.data_root = Path(data_root)
            self.folder = None
            self.meta_folder = None
            self.image_folder = None

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
        self.parent_to_idx, self.idx_to_parent = get_parent_to_idx(self.samples)
        # Build meta_name -> parent_concept mapping for hierarchical supervision
        self.meta_to_parent = build_meta_to_parent_mapping(self.samples)

        print(f"Loaded {len(self.samples)} samples")
        print(f"Concepts (meta_name): {len(self.concept_to_idx)}")
        print(f"Parent concepts: {len(self.parent_to_idx)}")
        print(f"Countries: {len(self.country_to_idx)}")

    def _load_samples(self) -> List[Dict]:
        """Load samples from CSV or folder structure."""
        if self.csv_path is not None:
            return self._load_samples_from_csv()
        else:
            return self._load_samples_from_folder()
    
    def _load_samples_from_csv(self) -> List[Dict]:
        """Load samples from CSV file using fast vectorized pandas operations."""
        # Load CSV
        print(f"Loading CSV from {self.csv_path}...")
        try:
            df = pd.read_csv(self.csv_path)
        except Exception as e:
            raise RuntimeError(f"Error loading CSV from {self.csv_path}: {e}")
        
        initial_count = len(df)
        print(f"  Loaded {initial_count} rows")
        
        # ========== VECTORIZED FILTERING ==========
        
        # 1. Drop rows with missing required fields (image_path, meta_name, country)
        df = df.dropna(subset=['image_path', 'meta_name', 'country'])
        after_required = len(df)
        
        # 2. Drop rows with missing lat/lng
        df = df.dropna(subset=['lat', 'lng'])
        after_coords = len(df)
        
        # 3. Drop rows with empty note
        df = df.dropna(subset=['note'])
        df = df[df['note'].astype(str).str.strip() != '']
        after_note = len(df)
        
        # 4. Country filter (if specified)
        if self.country is not None:
            target_country_norm = str(self.country).strip().lower()
            df = df[df['country'].astype(str).str.strip().str.lower() == target_country_norm]
        after_country = len(df)
        
        # 5. Parse coordinates (handle European comma decimals)
        df['lat'] = df['lat'].astype(str).str.replace(',', '.').astype(float)
        df['lng'] = df['lng'].astype(str).str.replace(',', '.').astype(float)
        
        # 6. Parse HTML from notes (vectorized with apply)
        # Use multiprocessing for large datasets to speed up HTML parsing
        if len(df) > 10000:
            from multiprocessing import Pool, cpu_count
            with Pool(min(cpu_count(), 8)) as pool:
                df['note'] = pool.map(parse_html_note, df['note'].astype(str).tolist())
        else:
            df['note'] = df['note'].astype(str).apply(parse_html_note)
        df = df[df['note'].str.strip() != '']
        after_html_parse = len(df)
        
        # 7. Apply max_samples limit
        if self.max_samples:
            df = df.head(self.max_samples)
        
        # 8. Check image existence (this is the only slow part, but necessary)
        # print(f"  Checking image files exist...")
        # df['image_exists'] = df['image_path'].apply(lambda x: Path(x).exists())
        # df = df[df['image_exists']]
        # df = df.drop(columns=['image_exists'])
        final_count = len(df)
        
        # Print statistics
        print(f"  Filtering stats:")
        print(f"    - Initial: {initial_count}")
        print(f"    - After required fields: {after_required} (dropped {initial_count - after_required})")
        print(f"    - After coords: {after_coords} (dropped {after_required - after_coords})")
        print(f"    - After note: {after_note} (dropped {after_coords - after_note})")
        if self.country:
            print(f"    - After country filter: {after_country} (dropped {after_note - after_country})")
        print(f"    - After HTML parse: {after_html_parse} (dropped {after_country - after_html_parse})")
        print(f"    - After image check: {final_count} (dropped {after_html_parse - final_count})")
        
        if final_count == 0:
            raise RuntimeError(f"No samples found! Check your CSV file ('{self.csv_path}'), country filter ('{self.country}'), or data requirements.")
        
        # Convert to list of dicts (vectorized - much faster than iterrows)
        samples = df.to_dict('records')
        samples = [
            {
                'pano_id': str(row.get('pano_id', f'row_{idx}')),
                'image_path': Path(row['image_path']),
                'meta_path': None,
                'meta_name': str(row['meta_name']),
                'parent_concept': str(row.get('concept', 'unknown')),  # Parent concept from 'concept' column
                'country': str(row['country']),
                'lat': float(row['lat']),
                'lng': float(row['lng']),
                'note': str(row['note']),
                'images': []
            }
            for idx, row in enumerate(samples)
        ]
        
        print(f"  Final: {len(samples)} samples")
        return samples
    
    def _load_samples_from_folder(self) -> List[Dict]:
        """Load meta files and filter to samples with existing images."""
        samples = []
        skipped_no_country_match = 0
        skipped_no_image = 0
        skipped_no_coords = 0
        skipped_empty_note = 0

        # Pre-process country filter for O(1) comparison inside loop
        target_country_norm = None
        if self.country is not None:
            target_country_norm = str(self.country).strip().lower()

        # Get all meta files
        meta_files = list(self.meta_folder.glob("*.json"))

        for meta_path in tqdm(meta_files, desc="Loading samples"):
            # Stop if we've reached max_samples (applied after filtering)
            if self.max_samples and len(samples) >= self.max_samples:
                break
                
            pano_id = meta_path.stem

            # Check if image exists
            image_path = self.image_folder / f"image_{pano_id}.jpg"
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

                # Fast Country Filter (O(1) comparison)
                if target_country_norm is not None:
                    meta_country = str(meta['country']).strip().lower()
                    if meta_country != target_country_norm:
                        skipped_no_country_match += 1
                        continue

                # Extract coordinates - ALWAYS require coordinates
                lat = meta.get('lat')
                lng = meta.get('lng')
                
                if lat is None or lng is None:
                    skipped_no_coords += 1
                    continue
                
                try:
                    lat = float(lat)
                    lng = float(lng)
                except (ValueError, TypeError):
                    skipped_no_coords += 1
                    continue
                    
                # Extract and parse note
                raw_note = meta.get('note', '')
                if not raw_note:
                    raw_note = ''
                else:
                    raw_note = str(raw_note)
                
                # Parse HTML note
                note = parse_html_note(raw_note)
                
                # Drop rows with empty notes
                if not note or len(note.strip()) == 0:
                    skipped_empty_note += 1
                    continue

                sample = {
                    'pano_id': pano_id,
                    'image_path': image_path,
                    'meta_path': meta_path,
                    'meta_name': meta['metaName'],
                    'parent_concept': meta.get('concept', 'unknown'),  # Parent concept from JSON if available
                    'country': meta['country'],
                    'lat': lat,
                    'lng': lng,
                    'note': note,
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
            print(f"  - Skipped (no coordinates): {skipped_no_coords}")
            print(f"  - Skipped (empty note): {skipped_empty_note}")
        else:
            print(f"Loaded {len(samples)} samples from folder")
            print(f"  - Skipped (no image): {skipped_no_image}")
            print(f"  - Skipped (no coordinates): {skipped_no_coords}")
            print(f"  - Skipped (empty note): {skipped_empty_note}")
        
        if len(samples) == 0:
             raise RuntimeError(f"No samples found! Check your country filter ('{self.country}') or coordinate/note requirements.")

        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, int, torch.Tensor, Dict]:
        """
        Returns:
            image_tensor: Processed image tensor
            concept_idx: Index of metaName (fine-grained concept)
            parent_idx: Index of parent concept (coarse category)
            target_idx: Index of country (target)
            coordinates_tensor: Normalized (lat, lng) tensor in [-1, 1]
            metadata: Dict with sample information
        """
        sample = self.samples[idx]

        # Load and process image
        image_path = sample['image_path']
        # Avoid repeated Path() conversion if already a Path object
        if not isinstance(image_path, Path):
            image_path = Path(image_path)
        image = Image.open(image_path).convert('RGB')

        if self.transform:
            image = self.transform(image)
        else:
            # Default processing: resize and convert to tensor
            image = image.resize(self.image_size, Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW

        # Encode concept, parent concept, and country
        concept_idx = self.concept_to_idx[sample['meta_name']]
        parent_concept = sample.get('parent_concept', 'unknown')
        parent_idx = self.parent_to_idx.get(parent_concept, 0)
        target_idx = self.country_to_idx[sample['country']]

        if self.return_cartesian:
            if sample['lat'] is not None and sample['lng'] is not None:
                coordinates = latlon_to_cartesian(sample['lat'], sample['lng'])
            else:
                coordinates = torch.tensor([float('nan')] * 3, dtype=torch.float32)
        elif self.use_normalized_coordinates:
            coordinates = normalize_coordinates(sample['lat'], sample['lng'])
        else:
            # Return raw coordinates
            if sample['lat'] is not None and sample['lng'] is not None:
                coordinates = torch.tensor([float(sample['lat']), float(sample['lng'])], dtype=torch.float32)
            else:
                coordinates = torch.tensor([float('nan'), float('nan')], dtype=torch.float32)

        # Metadata dict
        metadata = {
            'pano_id': sample['pano_id'],
            'meta_name': sample['meta_name'],
            'parent_concept': sample.get('parent_concept', 'unknown'),
            'country': sample['country'],
            'lat': sample['lat'],
            'lng': sample['lng'],
            'note': sample['note'],
            'images': sample['images'],
            'image_path': str(image_path),
        }
        
        # Add cell label if available
        if 'cell_label' in sample:
            metadata['cell_label'] = sample['cell_label']

        return image, concept_idx, parent_idx, target_idx, coordinates, metadata

    def set_cell_labels(self, sample_to_cell: torch.Tensor):
        """
        Assign geocell labels to all samples.
        Args:
            sample_to_cell: Tensor of shape [N_samples] with cell IDs.
        """
        if len(sample_to_cell) != len(self.samples):
             raise ValueError(f"Mismatch: {len(sample_to_cell)} labels for {len(self.samples)} samples")
             
        for i, sample in enumerate(self.samples):
            sample['cell_label'] = sample_to_cell[i].item()
        print(f"Assigned geocell labels to {len(self.samples)} samples.")


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


def get_parent_to_idx(samples: List[Dict]) -> Tuple[Dict[str, int], Dict[int, str]]:
    """Create mapping from parent concept (coarse category) strings to indices."""
    parent_concepts = sorted(set(s.get('parent_concept', 'unknown') for s in samples))
    parent_to_idx = {parent: i for i, parent in enumerate(parent_concepts)}
    idx_to_parent = {i: parent for parent, i in parent_to_idx.items()}
    return parent_to_idx, idx_to_parent


def build_meta_to_parent_mapping(samples: List[Dict]) -> Dict[str, str]:
    """
    Build mapping from meta_name (fine-grained concept) to parent_concept (coarse category).
    
    This is used for hierarchical supervision in Stage 1 training.
    
    Args:
        samples: List of sample dictionaries with 'meta_name' and 'parent_concept' keys
        
    Returns:
        Dict mapping meta_name -> parent_concept
    """
    meta_to_parent = {}
    for sample in samples:
        meta_name = sample['meta_name']
        parent_concept = sample.get('parent_concept', 'unknown')
        if meta_name not in meta_to_parent:
            meta_to_parent[meta_name] = parent_concept
    return meta_to_parent


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


def create_splits_stratified_strict(
    samples: List[Dict],
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Strict stratified split ensuring every concept with >=2 samples has at least
    1 sample in train AND 1 sample in val.

    This guarantees that during validation, no concept is entirely unseen - the model
    will encounter new images of known concepts rather than unknown concepts.

    Rules:
    - n=1: sample goes to train only (cannot split)
    - n=2: 1 train, 1 val, 0 test (guarantee val sample)
    - n>=3: at least 1 train, at least 1 val, rest proportional (including test)

    Args:
        samples: List of sample dictionaries with 'meta_name' key
        train_ratio: Target proportion for training (default 0.7)
        val_ratio: Target proportion for validation (default 0.15)
        test_ratio: Target proportion for test (default 0.15)
        seed: Random seed for reproducibility

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    rng = np.random.default_rng(seed)
    
    # Group samples by concept
    concept_to_samples: Dict[str, List[Dict]] = {}
    for sample in samples:
        concept_to_samples.setdefault(sample['meta_name'], []).append(sample)

    train_samples: List[Dict] = []
    val_samples: List[Dict] = []
    test_samples: List[Dict] = []

    # Track statistics
    concepts_with_no_val = 0
    concepts_with_no_test = 0

    for concept in sorted(concept_to_samples.keys()):
        concept_samples = concept_to_samples[concept]
        
        # Shuffle samples within concept
        shuffled_indices = rng.permutation(len(concept_samples))
        shuffled = [concept_samples[i] for i in shuffled_indices]
        
        n = len(shuffled)
        
        if n == 1:
            # Only 1 sample - must go to train
            train_samples.extend(shuffled)
            concepts_with_no_val += 1
            concepts_with_no_test += 1
        elif n == 2:
            # 2 samples: 1 train, 1 val (guarantee val coverage)
            train_samples.append(shuffled[0])
            val_samples.append(shuffled[1])
            concepts_with_no_test += 1
        else:
            # n >= 3: at least 1 train, at least 1 val, rest proportional
            # Calculate proportional splits
            n_train = max(1, int(round(n * train_ratio)))
            n_val = max(1, int(round(n * val_ratio)))
            
            # Ensure we don't exceed n
            if n_train + n_val > n:
                # Reduce train to make room for val
                n_train = max(1, n - n_val)
            if n_train + n_val > n:
                # Still exceeds, reduce val
                n_val = max(1, n - n_train)
            
            n_test = n - n_train - n_val
            
            if n_test == 0:
                concepts_with_no_test += 1
            
            train_samples.extend(shuffled[:n_train])
            val_samples.extend(shuffled[n_train:n_train + n_val])
            test_samples.extend(shuffled[n_train + n_val:])

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


def save_splits_to_json(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
    path: Path,
    extra_info: Optional[Dict] = None,
) -> None:
    """
    Save dataset splits to JSON for reproducibility.

    Args:
        train_samples: List of training sample dicts
        val_samples: List of validation sample dicts
        test_samples: List of test sample dicts
        path: Path to save JSON file
        extra_info: Optional dict with additional metadata (seed, ratios, etc.)
    """
    splits_data = {
        "train_pano_ids": [s['pano_id'] for s in train_samples],
        "val_pano_ids": [s['pano_id'] for s in val_samples],
        "test_pano_ids": [s['pano_id'] for s in test_samples],
        "counts": {
            "train": len(train_samples),
            "val": len(val_samples),
            "test": len(test_samples),
            "total": len(train_samples) + len(val_samples) + len(test_samples),
        },
    }
    if extra_info:
        splits_data["metadata"] = extra_info

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'w') as f:
        json.dump(splits_data, f, indent=2)
    print(f"Saved splits to {path}")


def load_splits_from_json(
    path: Path,
    samples: List[Dict],
) -> Tuple[List[Dict], List[Dict], List[Dict]]:
    """
    Load dataset splits from JSON file.

    Args:
        path: Path to JSON file with split pano_ids
        samples: Full list of sample dicts to filter

    Returns:
        Tuple of (train_samples, val_samples, test_samples)
    """
    path = Path(path)
    with open(path, 'r') as f:
        splits_data = json.load(f)

    # Build pano_id -> sample mapping
    pano_to_sample = {s['pano_id']: s for s in samples}

    train_pano_ids = set(splits_data["train_pano_ids"])
    val_pano_ids = set(splits_data["val_pano_ids"])
    test_pano_ids = set(splits_data["test_pano_ids"])

    train_samples = [pano_to_sample[pid] for pid in train_pano_ids if pid in pano_to_sample]
    val_samples = [pano_to_sample[pid] for pid in val_pano_ids if pid in pano_to_sample]
    test_samples = [pano_to_sample[pid] for pid in test_pano_ids if pid in pano_to_sample]

    # Warn about missing samples
    missing_train = len(train_pano_ids) - len(train_samples)
    missing_val = len(val_pano_ids) - len(val_samples)
    missing_test = len(test_pano_ids) - len(test_samples)
    if missing_train > 0 or missing_val > 0 or missing_test > 0:
        print(f"Warning: Some pano_ids from splits.json not found in dataset:")
        print(f"  Train: {missing_train} missing, Val: {missing_val} missing, Test: {missing_test} missing")

    print(f"Loaded splits from {path}: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    return train_samples, val_samples, test_samples


def log_split_diagnostics(
    train_samples: List[Dict],
    val_samples: List[Dict],
    test_samples: List[Dict],
) -> Dict:
    """
    Log diagnostics about concept coverage in each split.

    Returns:
        Dict with diagnostic statistics
    """
    # Get unique concepts per split
    train_concepts = set(s['meta_name'] for s in train_samples)
    val_concepts = set(s['meta_name'] for s in val_samples)
    test_concepts = set(s['meta_name'] for s in test_samples)
    all_concepts = train_concepts | val_concepts | test_concepts

    # Get unique parent concepts per split
    train_parents = set(s.get('parent_concept', 'unknown') for s in train_samples)
    val_parents = set(s.get('parent_concept', 'unknown') for s in val_samples)
    test_parents = set(s.get('parent_concept', 'unknown') for s in test_samples)
    all_parents = train_parents | val_parents | test_parents

    # Concepts missing from val/test
    concepts_not_in_val = all_concepts - val_concepts
    concepts_not_in_test = all_concepts - test_concepts
    concepts_not_in_train = all_concepts - train_concepts

    # Parents missing from val/test
    parents_not_in_val = all_parents - val_parents
    parents_not_in_test = all_parents - test_parents
    parents_not_in_train = all_parents - train_parents

    diagnostics = {
        "total_concepts": len(all_concepts),
        "train_concepts": len(train_concepts),
        "val_concepts": len(val_concepts),
        "test_concepts": len(test_concepts),
        "concepts_not_in_train": len(concepts_not_in_train),
        "concepts_not_in_val": len(concepts_not_in_val),
        "concepts_not_in_test": len(concepts_not_in_test),
        "total_parents": len(all_parents),
        "train_parents": len(train_parents),
        "val_parents": len(val_parents),
        "test_parents": len(test_parents),
        "parents_not_in_train": len(parents_not_in_train),
        "parents_not_in_val": len(parents_not_in_val),
        "parents_not_in_test": len(parents_not_in_test),
    }

    print(f"Split Diagnostics:")
    print(f"  Child concepts: {diagnostics['total_concepts']} total")
    print(f"    Train: {diagnostics['train_concepts']}, Val: {diagnostics['val_concepts']}, Test: {diagnostics['test_concepts']}")
    print(f"    Missing from train: {diagnostics['concepts_not_in_train']}, val: {diagnostics['concepts_not_in_val']}, test: {diagnostics['concepts_not_in_test']}")
    print(f"  Parent concepts: {diagnostics['total_parents']} total")
    print(f"    Train: {diagnostics['train_parents']}, Val: {diagnostics['val_parents']}, Test: {diagnostics['test_parents']}")
    print(f"    Missing from train: {diagnostics['parents_not_in_train']}, val: {diagnostics['parents_not_in_val']}, test: {diagnostics['parents_not_in_test']}")

    return diagnostics


class SubsetDataset(Dataset):
    """
    Dataset wrapper for subsets (train/val/test splits).
    Optimized for O(1) retrieval speed.
    """

    def __init__(self, parent_dataset: PanoramaCBMDataset, samples: List[Dict]):
        self.parent_dataset = parent_dataset
        self.samples = samples
        
        # Build a fast lookup: pano_id -> parent index
        # This avoids O(n) index() calls in __getitem__
        parent_pano_to_idx = {sample['pano_id']: idx for idx, sample in enumerate(parent_dataset.samples)}
        self.parent_indices = [parent_pano_to_idx[sample['pano_id']] for sample in samples]

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        # Direct index lookup - O(1) instead of O(n)
        parent_idx = self.parent_indices[idx]
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
    parser = argparse.ArgumentParser(description="Test PanoramaCBMDataset")
    parser.add_argument("--geoguessr-id", type=str, default="6906237dc7731161a37282b2",
                        help="GeoGuessr map ID")
    parser.add_argument("--data-root", type=str, default="data",
                        help="Root directory for data")
    parser.add_argument("--country", type=str, default="Australia",
                        help="Country filter")
    parser.add_argument("--csv-path", type=str, default=None,
                        help="Path to CSV file (optional, if provided loads from CSV instead of folder structure)")
    args = parser.parse_args()
    
    # Test the dataset
    dataset = PanoramaCBMDataset(
        country=args.country,
        geoguessr_id=args.geoguessr_id,
        data_root=args.data_root,
        csv_path=args.csv_path
    )  

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
