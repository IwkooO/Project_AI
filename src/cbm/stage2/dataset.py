"""
Stage 2 dataset for geolocation training.

Loads cached StreetCLIP embeddings and aligns with CSV metadata.
"""

from __future__ import annotations

import torch
import pandas as pd
import json
import numpy as np
from pathlib import Path
from typing import Optional
from torch.utils.data import Dataset


class Stage2Dataset(Dataset):
    """
    Dataset for Stage 2 geolocation training.
    
    Loads:
    - Cached StreetCLIP embeddings (pooled + patch tokens)
    - Coordinates from CSV
    - Country labels from CSV
    - Aligns by pano_id via metadata JSON
    """
    
    def __init__(
        self,
        csv_path: str | Path,
        cached_dir: str | Path,
        split: str = "train",
    ):
        """
        Args:
            csv_path: Path to CSV with columns: pano_id, lat, lng, country, image_path
            cached_dir: Directory with cached embeddings (*_pooled_embeddings.pt, *_patch_tokens.pt, *_metadata.json)
            split: Split name ('train', 'val', 'test')
        """
        self.csv_path = Path(csv_path)
        self.cached_dir = Path(cached_dir)
        self.split = split
        
        # Load CSV
        print(f"Loading CSV from {csv_path}...")
        self.df = pd.read_csv(csv_path)
        
        # Filter rows with valid coordinates
        self.df = self.df.dropna(subset=['lat', 'lng', 'pano_id'])
        print(f"Valid samples after filtering: {len(self.df)}")
        
        # Load cached embeddings
        pooled_path = self.cached_dir / f"{split}_pooled_embeddings.pt"
        patch_path = self.cached_dir / f"{split}_patch_tokens.pt"
        metadata_path = self.cached_dir / f"{split}_metadata.json"
        
        if not pooled_path.exists():
            raise FileNotFoundError(f"Pooled embeddings not found: {pooled_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")
        
        print(f"Loading pooled embeddings from {pooled_path}...")
        self.pooled_embeddings = torch.load(pooled_path)  # [N, D]
        
        print(f"Loading metadata from {metadata_path}...")
        with open(metadata_path, 'r') as f:
            self.metadata = json.load(f)
        
        # Build pano_id -> index mapping from metadata
        # metadata['pano_ids'] is a list of pano_ids in the same order as embeddings
        self.pano_to_embed_idx = {}
        for idx, pano_id in enumerate(self.metadata['pano_ids']):
            if pano_id not in self.pano_to_embed_idx:
                self.pano_to_embed_idx[pano_id] = []
            self.pano_to_embed_idx[pano_id].append(idx)
        
        print(f"Found {len(self.pano_to_embed_idx)} unique pano_ids in cached embeddings")
        
        # Load patch tokens if available
        self.patch_tokens = None
        if patch_path.exists():
            print(f"Loading patch tokens from {patch_path}...")
            self.patch_tokens = torch.load(patch_path)  # [N, P, D]
            print(f"Patch tokens shape: {self.patch_tokens.shape}")
        else:
            print(f"Warning: Patch tokens not found at {patch_path}, will use None")
        
        # Filter CSV rows to only those with cached embeddings
        valid_indices = []
        for idx, row in self.df.iterrows():
            pano_id = str(row['pano_id'])
            if pano_id in self.pano_to_embed_idx:
                valid_indices.append(idx)
        
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)
        print(f"Final dataset size after alignment: {len(self.df)}")
        
        # Store coordinate info
        self.coords = np.array([
            [float(row['lat']), float(row['lng'])]
            for _, row in self.df.iterrows()
        ])  # [N, 2]
        
        # Store countries (if available)
        self.countries = None
        if 'country' in self.df.columns:
            self.countries = self.df['country'].values
    
    def __len__(self) -> int:
        return len(self.df)
    
    def __getitem__(self, idx: int) -> dict:
        """
        Returns:
            dict with keys:
                - pooled_emb: [D] pooled embedding
                - patch_tokens: [P, D] patch tokens (or None)
                - coords: [2] (lat, lng)
                - country: str (or None)
                - image_path: str
                - pano_id: str
                - cell_labels: int (if set via set_labels)
                - offset_targets: [3] (if set via set_labels)
        """
        row = self.df.iloc[idx]
        pano_id = str(row['pano_id'])
        
        # Get embedding index (use first if multiple)
        embed_idx = self.pano_to_embed_idx[pano_id][0]
        
        # Get embeddings
        pooled_emb = self.pooled_embeddings[embed_idx]  # [D]
        
        patch_tokens = None
        if self.patch_tokens is not None:
            patch_tokens = self.patch_tokens[embed_idx]  # [P, D]
        
        # Get coordinates
        coords = self.coords[idx]  # [2] (lat, lng)
        
        # Get country
        country = None
        if self.countries is not None:
            country = str(self.countries[idx]) if pd.notna(self.countries[idx]) else None
        
        # Get image path
        image_path = str(row['image_path']) if 'image_path' in row else ""
        
        result = {
            'pooled_emb': pooled_emb,
            'patch_tokens': patch_tokens,
            'coords': torch.tensor(coords, dtype=torch.float32),
            'country': country,
            'image_path': image_path,
            'pano_id': pano_id,
        }
        
        # Add cell labels and offsets if set
        if hasattr(self, 'cell_labels'):
            result['cell_labels'] = torch.tensor(self.cell_labels[idx], dtype=torch.long)
        if hasattr(self, 'offsets'):
            result['offset_targets'] = torch.tensor(self.offsets[idx], dtype=torch.float32)
        
        return result
    
    def set_labels(self, cell_labels: np.ndarray, offsets: np.ndarray):
        """Set cell labels and offsets for training."""
        self.cell_labels = cell_labels
        self.offsets = offsets


def collate_fn_stage2(batch: list[dict]) -> dict:
    """
    Collate function for Stage2Dataset.
    
    Args:
        batch: List of dicts from Stage2Dataset.__getitem__
    
    Returns:
        dict with batched tensors:
            - pooled_emb: [B, D]
            - patch_tokens: [B, P, D] or None
            - coords: [B, 2] (lat, lng)
            - countries: list[str] (or None)
            - image_paths: list[str]
            - pano_ids: list[str]
            - cell_labels: [B] (if present)
            - offset_targets: [B, 3] (if present)
    """
    pooled_embs = torch.stack([item['pooled_emb'] for item in batch])
    
    # Handle patch tokens (may be None)
    patch_tokens_list = [item['patch_tokens'] for item in batch]
    if all(pt is not None for pt in patch_tokens_list):
        patch_tokens = torch.stack(patch_tokens_list)
    else:
        patch_tokens = None
    
    coords = torch.stack([item['coords'] for item in batch])
    
    countries = [item['country'] for item in batch] if batch[0]['country'] is not None else None
    image_paths = [item['image_path'] for item in batch]
    pano_ids = [item['pano_id'] for item in batch]
    
    result = {
        'pooled_emb': pooled_embs,
        'patch_tokens': patch_tokens,
        'coords': coords,
        'countries': countries,
        'image_paths': image_paths,
        'pano_ids': pano_ids,
    }
    
    # Add cell labels and offsets if present
    if 'cell_labels' in batch[0]:
        result['cell_labels'] = torch.stack([item['cell_labels'] for item in batch])
    if 'offset_targets' in batch[0]:
        result['offset_targets'] = torch.stack([item['offset_targets'] for item in batch])
    
    return result
