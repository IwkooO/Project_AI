import pandas as pd
import torch
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from pathlib import Path
from typing import Optional, Tuple, List, Dict
from transformers import AutoImageProcessor
from sklearn.model_selection import train_test_split
import numpy as np

class ConceptProbeDataset(Dataset):
    """
    Dataset for Concept Probe training using pre-generated CSV.
    """
    def __init__(self, dataframe: pd.DataFrame, transform=None):
        """
        Args:
            dataframe: Pandas DataFrame containing 'image_path' and 'meta_name' columns.
            transform: Optional torchvision transforms.
        """
        self.data = dataframe.reset_index(drop=True)
        self.transform = transform
        
        # Create label mapping
        self.labels = sorted(self.data['meta_name'].unique().tolist())
        self.label_to_idx = {label: idx for idx, label in enumerate(self.labels)}
        self.idx_to_label = {idx: label for label, idx in self.label_to_idx.items()}

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data.iloc[idx]
        image_path = row['image_path']
        label_name = row['meta_name']
        
        # Load image
        try:
            image = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            # Return a dummy image or handle error appropriately
            # For now, re-raise to be explicit
            raise e

        # Apply transforms
        if self.transform:
            image = self.transform(image)
            
        # Get label index
        label_idx = self.label_to_idx[label_name]
        
        return image, torch.tensor(label_idx, dtype=torch.long)

def create_stratified_splits(csv_path: str, test_size: float = 0.15, val_size: float = 0.15, seed: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Create stratified train/val/test splits from CSV.
    
    Args:
        csv_path: Path to dataset CSV.
        test_size: Proportion of test set.
        val_size: Proportion of validation set.
        seed: Random seed.
        
    Returns:
        (train_df, val_df, test_df)
    """
    # Load data
    df = pd.read_csv(csv_path)
    
    # Check for missing images if needed, but assuming CSV is clean based on creation script
    # We could add a check here if robust checking is required
    
    labels = df['meta_name']
    
    # First split: Separate Test set
    train_val_df, test_df = train_test_split(
        df, 
        test_size=test_size, 
        stratify=labels, 
        random_state=seed
    )
    
    # Re-compute labels for the remaining set
    train_val_labels = train_val_df['meta_name']
    
    # Adjust val_size relative to the remaining data
    # original val_size is relative to total. 
    # We need val_size_adjusted * (1 - test_size) = val_size
    # val_size_adjusted = val_size / (1 - test_size)
    val_size_adjusted = val_size / (1 - test_size)
    
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

