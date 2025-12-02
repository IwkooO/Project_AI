import pandas as pd
import torch
import re
from torch.utils.data import Dataset
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from transformers import AutoImageProcessor
import numpy as np

# Import from src.dataset for compatibility
from src.dataset import (
    get_transforms_from_processor,
    get_concept_to_idx,
    get_country_to_idx,
    create_splits_stratified,
    SubsetDataset
)


def clean_html_note(html_text: str) -> str:
    """
    Extract clean plain text from HTML content in note column.
    Uses BeautifulSoup if available, falls back to regex.
    
    Args:
        html_text: Raw HTML string from the note column
        
    Returns:
        Clean plain text with HTML tags removed
    """
    if not html_text or pd.isna(html_text):
        return ""
    
    html_text = str(html_text).strip()
    if not html_text:
        return ""
    
    # Try BeautifulSoup first
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_text, 'html.parser')
        # Get text and normalize whitespace
        text = soup.get_text(separator=' ')
        text = re.sub(r'\s+', ' ', text).strip()
        return text
    except ImportError:
        pass
    
    # Fallback: regex-based HTML cleaning
    # Remove script and style elements
    text = re.sub(r'<script[^>]*>.*?</script>', '', html_text, flags=re.DOTALL | re.IGNORECASE)
    text = re.sub(r'<style[^>]*>.*?</style>', '', text, flags=re.DOTALL | re.IGNORECASE)
    # Remove HTML tags
    text = re.sub(r'<[^>]+>', ' ', text)
    # Decode common HTML entities
    text = text.replace('&nbsp;', ' ')
    text = text.replace('&amp;', '&')
    text = text.replace('&lt;', '<')
    text = text.replace('&gt;', '>')
    text = text.replace('&quot;', '"')
    text = text.replace('&#39;', "'")
    # Normalize whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text


class CBMDataset(Dataset):
    """
    Dataset for Concept Bottleneck Model training using pre-generated CSV.
    Compatible with train_concept_aware.py training script.
    
    Returns:
        image_tensor: Processed image tensor
        concept_idx: Index of meta_name (concept)
        target_idx: Index of country (target)
        coordinates: Tensor of (lat, lng) in degrees
        metadata: Dict with sample information
    """
    
    def __init__(
        self,
        dataframe: pd.DataFrame,
        country: Optional[str] = None,
        transform=None,
        encoder_model: Optional[str] = None,
        image_size: Optional[Tuple[int, int]] = None,
        use_normalized_coordinates: bool = False,
    ):
        """
        Args:
            dataframe: Pandas DataFrame containing 'image_path', 'meta_name', 'country', 'lat', 'lng' columns.
            transform: Optional torchvision transforms (overrides encoder_model preprocessing)
            encoder_model: HuggingFace model identifier (e.g., 'geolocal/StreetCLIP')
                          If provided, will use AutoImageProcessor to get correct preprocessing
            image_size: Target size for images (width, height) - used if encoder_model not provided
            use_normalized_coordinates: If True, returns coordinates normalized to [-1, 1]. If False, returns raw (lat, lng).
        """
        self.data = dataframe.reset_index(drop=True)
        self.country = country
        if country:
            self.data = self.data[self.data['country'] == country]
        else:
            self.data = self.data.dropna(subset=['image_path', 'meta_name', 'country'])
        print(f"Loaded {len(self.data)} samples")
        self.use_normalized_coordinates = use_normalized_coordinates
        self.encoder_model = encoder_model
        
        # Convert dataframe to samples list (similar to PanoramaCBMDataset)
        self.samples = self._dataframe_to_samples()
        
        # Set up transforms based on encoder model or defaults
        if transform is None:
            processor = None
            if encoder_model is not None:
                try:
                    processor = AutoImageProcessor.from_pretrained(encoder_model)
                except Exception as e:
                    print(f"Warning: Could not load processor for {encoder_model}: {e}")
                    print("Falling back to default CLIP preprocessing")
            
            self.transform = get_transforms_from_processor(processor, image_size)
        else:
            self.transform = transform
        
        # Build label mappings (similar to PanoramaCBMDataset)
        self.concept_to_idx, self.idx_to_concept = get_concept_to_idx(self.samples)
        self.country_to_idx, self.idx_to_country = get_country_to_idx(self.samples)
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Concepts: {len(self.concept_to_idx)}")
        print(f"Countries: {len(self.country_to_idx)}")
    
    def _dataframe_to_samples(self) -> List[Dict]:
        """Convert dataframe to samples list format compatible with PanoramaCBMDataset."""
        samples = []
        
        for idx, row in self.data.iterrows():
            # Check required columns
            if pd.isna(row.get('image_path')) or pd.isna(row.get('meta_name')) or pd.isna(row.get('country')):
                continue
            
            # Check coordinates
            lat = row.get('lat')
            lng = row.get('lng')
            if pd.isna(lat) or pd.isna(lng):
                # Skip if missing coordinates are critical
                continue
            
            # Clean HTML from note column
            raw_note = row.get('note', '')
            cleaned_note = clean_html_note(raw_note)
            
            sample = {
                'pano_id': str(row.get('pano_id', f'row_{idx}')),  # Use pano_id if available, else row index
                'image_path': Path(row['image_path']),
                'meta_name': str(row['meta_name']),
                'country': str(row['country']),
                'lat': lat,
                'lng': lng,
                'note': cleaned_note,  # Store cleaned plain text
                'images': row.get('images', []) if isinstance(row.get('images'), list) else []
            }
            samples.append(sample)
        
        if len(samples) == 0:
            raise RuntimeError("No valid samples found in DataFrame! Check for missing 'lat', 'lng', 'image_path', 'meta_name', or 'country'.")
        
        return samples
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, int, torch.Tensor, Dict]:
        """
        Returns:
            image_tensor: Processed image tensor
            concept_idx: Index of meta_name (concept)
            target_idx: Index of country (target)
            coordinates: Tensor of (lat, lng) in degrees or normalized [-1, 1]
            metadata: Dict with sample information
        """
        sample = self.samples[idx]
        
        # Load and process image
        image_path = sample['image_path']
        if isinstance(image_path, str):
            image_path = Path(image_path)
        
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            raise e
        
        # Apply transforms
        if self.transform:
            image = self.transform(image)
        else:
            # Default processing: resize and convert to tensor
            image = image.resize((336, 336), Image.LANCZOS)
            image = np.array(image).astype(np.float32) / 255.0
            image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
        
        # Encode concept and target
        concept_idx = self.concept_to_idx[sample['meta_name']]
        target_idx = self.country_to_idx[sample['country']]
        
        # Get coordinates
        if self.use_normalized_coordinates:
            from src.dataset import normalize_coordinates
            coordinates = normalize_coordinates(sample['lat'], sample['lng'])
        else:
            # Return raw coordinates
            if sample['lat'] is not None and sample['lng'] is not None:
                coordinates = torch.tensor([float(sample['lat']), float(sample['lng'])], dtype=torch.float32)
            else:
                coordinates = torch.tensor([float('nan'), float('nan')], dtype=torch.float32)
        
        # Metadata dict (matching PanoramaCBMDataset format)
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


def create_stratified_splits(
    csv_path: str,
    train_ratio: float = 0.7,
    val_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits from CSV.
    Stratification is based on 'meta_name' (concepts).
    
    Note: This function returns DataFrames. For compatibility with train_concept_aware.py,
    you should use create_splits_stratified from src.dataset on the dataset.samples instead.
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Filter out rows with missing critical data
    df = df.dropna(subset=['image_path', 'meta_name', 'country'])
    
    labels = df['meta_name']
    
    # First split: Separate Test set
    from sklearn.model_selection import train_test_split
    
    train_val_df, test_df = train_test_split(
        df,
        test_size=test_ratio,
        stratify=labels,
        random_state=seed
    )
    
    # Re-compute labels for the remaining set
    train_val_labels = train_val_df['meta_name']
    
    # Adjust val_size relative to the remaining data
    val_size_adjusted = val_ratio / (1 - test_ratio)
    
    # Second split: Separate Train and Val sets
    train_df, val_df = train_test_split(
        train_val_df,
        test_size=val_size_adjusted,
        stratify=train_val_labels,
        random_state=seed
    )
    
    return train_df, val_df, test_df


def get_transforms(model_name: str = "geolocal/StreetCLIP"):
    """
    Get transforms from HuggingFace processor.
    
    Note: For compatibility with train_concept_aware.py, use get_transforms_from_processor
    from src.dataset instead.
    """
    try:
        processor = AutoImageProcessor.from_pretrained(model_name)
        size = (processor.size["height"], processor.size["width"]) if isinstance(processor.size, dict) else (processor.size, processor.size)
        mean = processor.image_mean
        std = processor.image_std
    except Exception:
        # Fallback defaults for CLIP
        size = (336, 336)
        mean = (0.48145466, 0.4578275, 0.40821073)
        std = (0.26862954, 0.26130258, 0.27577711)
    
    from torchvision import transforms
    
    return transforms.Compose([
        transforms.Resize(size, interpolation=transforms.InterpolationMode.BICUBIC),
        transforms.ToTensor(),
        transforms.Normalize(mean=mean, std=std)
    ])

