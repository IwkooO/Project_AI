"""
Phase 2 dataset for geolocation training.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
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
        self.csv_path = Path(csv_path)
        self.cached_dir = Path(cached_dir)
        self.split = split

        self.df = pd.read_csv(csv_path)
        self.df = self.df.dropna(subset=["lat", "lng", "pano_id"])

        pooled_path = self.cached_dir / f"{split}_pooled_embeddings.pt"
        patch_path = self.cached_dir / f"{split}_patch_tokens.pt"
        metadata_path = self.cached_dir / f"{split}_metadata.json"

        if not pooled_path.exists():
            raise FileNotFoundError(f"Pooled embeddings not found: {pooled_path}")
        if not metadata_path.exists():
            raise FileNotFoundError(f"Metadata not found: {metadata_path}")

        self.pooled_embeddings = torch.load(pooled_path)  # [N, D]
        with open(metadata_path, "r") as f:
            self.metadata = json.load(f)

        pano_to_embed_idx: dict[str, list[int]] = {}
        for idx, pano_id in enumerate(self.metadata["pano_ids"]):
            pid = str(pano_id)
            pano_to_embed_idx.setdefault(pid, []).append(idx)
        self.pano_to_embed_idx = pano_to_embed_idx

        self.patch_tokens = None
        if patch_path.exists():
            self.patch_tokens = torch.load(patch_path)  # [N, P, D]

        valid_indices = []
        for idx, row in self.df.iterrows():
            pano_id = str(row["pano_id"])
            if pano_id in self.pano_to_embed_idx:
                valid_indices.append(idx)
        self.df = self.df.iloc[valid_indices].reset_index(drop=True)

        self.coords = np.array([[float(r["lat"]), float(r["lng"])] for _, r in self.df.iterrows()], dtype=np.float32)

        self.countries = None
        if "country" in self.df.columns:
            self.countries = self.df["country"].values

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int) -> dict:
        row = self.df.iloc[idx]
        pano_id = str(row["pano_id"])
        embed_idx = self.pano_to_embed_idx[pano_id][0]

        pooled_emb = self.pooled_embeddings[embed_idx]
        patch_tokens = None
        if self.patch_tokens is not None:
            patch_tokens = self.patch_tokens[embed_idx]

        coords = self.coords[idx]
        country = None
        if self.countries is not None:
            country = str(self.countries[idx]) if pd.notna(self.countries[idx]) else None

        image_path = str(row["image_path"]) if "image_path" in row else ""

        result = {
            "pooled_emb": pooled_emb,
            "patch_tokens": patch_tokens,
            "coords": torch.tensor(coords, dtype=torch.float32),
            "country": country,
            "image_path": image_path,
            "pano_id": pano_id,
        }

        if hasattr(self, "cell_labels"):
            result["cell_labels"] = torch.tensor(self.cell_labels[idx], dtype=torch.long)
        if hasattr(self, "offsets"):
            result["offset_targets"] = torch.tensor(self.offsets[idx], dtype=torch.float32)

        return result

    def set_labels(self, cell_labels: np.ndarray, offsets: np.ndarray):
        self.cell_labels = cell_labels
        self.offsets = offsets


def collate_fn_stage2(batch: list[dict]) -> dict:
    pooled_embs = torch.stack([item["pooled_emb"] for item in batch])

    patch_tokens_list = [item["patch_tokens"] for item in batch]
    patch_tokens = torch.stack(patch_tokens_list) if all(pt is not None for pt in patch_tokens_list) else None

    coords = torch.stack([item["coords"] for item in batch])
    countries = [item["country"] for item in batch] if batch[0]["country"] is not None else None
    image_paths = [item["image_path"] for item in batch]
    pano_ids = [item["pano_id"] for item in batch]

    result = {
        "pooled_emb": pooled_embs,
        "patch_tokens": patch_tokens,
        "coords": coords,
        "countries": countries,
        "image_paths": image_paths,
        "pano_ids": pano_ids,
    }

    if "cell_labels" in batch[0]:
        result["cell_labels"] = torch.stack([item["cell_labels"] for item in batch])
    if "offset_targets" in batch[0]:
        result["offset_targets"] = torch.stack([item["offset_targets"] for item in batch])

    return result


