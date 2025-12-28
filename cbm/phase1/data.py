from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from PIL import Image


@dataclass(frozen=True)
class CacheIndex:
    """Mapping from pano_id -> cached tensor row index."""

    pano_to_cache_idx: Dict[str, int]
    pano_ids: List[str]

    @staticmethod
    def from_metadata_json(metadata_path: Path) -> "CacheIndex":
        meta = json.loads(metadata_path.read_text())
        pano_ids = meta.get("pano_ids")
        if pano_ids is None:
            raise KeyError(f"Expected key 'pano_ids' in cache metadata: {metadata_path}")
        if not isinstance(pano_ids, list) or not pano_ids:
            raise ValueError(f"Invalid 'pano_ids' in {metadata_path}")
        pano_to_cache_idx = {str(pid): int(i) for i, pid in enumerate(pano_ids)}
        return CacheIndex(pano_to_cache_idx=pano_to_cache_idx, pano_ids=[str(p) for p in pano_ids])


class ConceptDataset(Dataset):
    """
    Dataset for Phase 1 concept prediction (and Stage 2 geo).
    
    Supports:
    1. Cached mode: uses cached patch tokens.
    2. Raw image mode: loads images from disk for fine-tuning.

    Yields tuples compatible with training code:
      (patch_tokens_or_image, concept_label, coords(lat,lng), cell_label, country_label, cache_idx, pooled_emb)
    """

    def __init__(
        self,
        csv_path: str | Path,
        cached_dir: str | Path | None,
        concept_vocab_path: str | Path,
        s2_vocab_path: str | Path,
        *,
        split: str,
        image_dir: str | Path | None = None,
        transform: Any = None,
        allow_unsafe_index_fallback: bool = False,
        require_cell_labels: bool = False,
        cell_token_col: str = "s2_token",
        concept_data_dir: str | Path | None = None,
        load_pooled_embeddings: bool = False,
    ):
        self.csv_path = Path(csv_path)
        self.cached_dir = Path(cached_dir) if cached_dir is not None else None
        self.split = str(split)
        self.image_dir = Path(image_dir) if image_dir is not None else None
        self.transform = transform
        self.allow_unsafe_index_fallback = bool(allow_unsafe_index_fallback)
        self.require_cell_labels = bool(require_cell_labels)
        self.cell_token_col = str(cell_token_col)
        self.concept_data_dir = Path(concept_data_dir) if concept_data_dir is not None else None
        self.load_pooled_embeddings = bool(load_pooled_embeddings)

        if not self.csv_path.exists():
            raise FileNotFoundError(self.csv_path)
        
        # At least one source of vision data must be provided
        if self.cached_dir is None and self.image_dir is None:
            raise ValueError("Either cached_dir or image_dir must be provided.")

        if self.cached_dir is not None and not self.cached_dir.exists():
            raise FileNotFoundError(self.cached_dir)
        
        if self.image_dir is not None and not self.image_dir.exists():
            raise FileNotFoundError(self.image_dir)

        df = pd.read_csv(self.csv_path)
        if "generalized" not in df.columns:
            if "meta_name" not in df.columns:
                raise KeyError(
                    f"CSV must contain 'generalized' or 'meta_name' label column, got columns={list(df.columns)}"
                )
            df["generalized"] = df["meta_name"]
        for col in ("pano_id", "lat", "lng"):
            if col not in df.columns:
                raise KeyError(f"CSV must contain column '{col}', got columns={list(df.columns)}")

        df = df.dropna(subset=["pano_id", "generalized", "lat", "lng"]).reset_index(drop=True)
        self.df = df

        concept_vocab = json.loads(Path(concept_vocab_path).read_text())
        self.concept_to_idx: Dict[str, int] = {str(k): int(v) for k, v in concept_vocab["concept_to_idx"].items()}
        self.idx_to_concept: Dict[int, str] = {int(k): str(v) for k, v in concept_vocab["idx_to_concept"].items()}
        self.num_concepts = int(concept_vocab["num_concepts"])

        s2_vocab = json.loads(Path(s2_vocab_path).read_text())
        self.cell_to_idx: Dict[str, int] = {str(k): int(v) for k, v in s2_vocab["cell_to_idx"].items()}
        self.idx_to_cell: Dict[int, str] = {int(k): str(v) for k, v in s2_vocab["idx_to_cell"].items()}
        self.num_cells = int(s2_vocab["num_cells"])

        self.patch_tokens: torch.Tensor | None = None
        self.pooled_embeddings: torch.Tensor | None = None
        self._pano_to_cache_idx: Dict[str, int] = {}

        if self.cached_dir is not None:
            patch_path = self.cached_dir / f"{self.split}_patch_tokens.pt"
            pooled_path = self.cached_dir / f"{self.split}_pooled_embeddings.pt"
            meta_path = self.cached_dir / f"{self.split}_metadata.json"
            
            if not patch_path.exists():
                raise FileNotFoundError(f"Missing cached patch tokens: {patch_path}")
            if not meta_path.exists():
                raise FileNotFoundError(f"Missing cache metadata: {meta_path}")

            self.patch_tokens = torch.load(patch_path, map_location="cpu")
            if self.patch_tokens.dim() != 3:
                raise ValueError(f"Expected patch_tokens [N,P,D], got {tuple(self.patch_tokens.shape)}")

            cache_index = CacheIndex.from_metadata_json(meta_path)
            self._pano_to_cache_idx = cache_index.pano_to_cache_idx

            if int(self.patch_tokens.shape[0]) != len(cache_index.pano_ids):
                raise ValueError(
                    f"Cache mismatch: patch_tokens N={int(self.patch_tokens.shape[0])} "
                    f"!= len(metadata.pano_ids)={len(cache_index.pano_ids)}"
                )

            # Optional pooled embeddings (aligned with patch_tokens via cache metadata ordering)
            if self.load_pooled_embeddings:
                if not pooled_path.exists():
                    raise FileNotFoundError(f"Missing pooled embeddings: {pooled_path}")
                pooled = torch.load(pooled_path, map_location="cpu")
                if pooled.dim() != 2:
                    raise ValueError(f"Expected pooled_embeddings [N,D], got {tuple(pooled.shape)}")
                if int(pooled.shape[0]) != int(self.patch_tokens.shape[0]):
                    raise ValueError(
                        f"Cache mismatch: pooled_embeddings N={int(pooled.shape[0])} "
                        f"!= patch_tokens N={int(self.patch_tokens.shape[0])}"
                    )
                self.pooled_embeddings = pooled.contiguous()

        # Optional country vocab
        self.country_to_idx: Dict[str, int] = {}
        self.idx_to_country: Dict[int, str] = {}
        self.num_countries = 0
        if self.concept_data_dir is not None:
            country_vocab_path = self.concept_data_dir / "country_vocab.json"
            if country_vocab_path.exists():
                country_vocab = json.loads(country_vocab_path.read_text())
                self.country_to_idx = {str(k): int(v) for k, v in country_vocab["country_to_idx"].items()}
                self.idx_to_country = {int(k): str(v) for k, v in country_vocab["idx_to_country"].items()}
                self.num_countries = int(country_vocab["num_countries"])

        labels: List[int] = []
        coords: List[Tuple[float, float]] = []
        cell_labels: List[int] = []
        country_labels: List[int] = []
        cache_idxs: List[int] = []
        pano_ids: List[str] = []

        missing_in_cache = 0
        missing_in_vocab = 0
        missing_in_cells = 0
        missing_in_countries = 0
        missing_images = 0

        for _, row in self.df.iterrows():
            pano_id = str(row["pano_id"])
            concept_name = str(row["generalized"])
            lat = float(row["lat"])
            lng = float(row["lng"])

            if concept_name not in self.concept_to_idx:
                missing_in_vocab += 1
                continue
            c_idx = int(self.concept_to_idx[concept_name])

            country_label = -1
            if self.num_countries > 0:
                country_name = row.get("country", None)
                if isinstance(country_name, str) and country_name in self.country_to_idx:
                    country_label = int(self.country_to_idx[country_name])
                else:
                    missing_in_countries += 1
                    if self.require_cell_labels:
                        continue

            cell_token = row.get(self.cell_token_col, None)
            if isinstance(cell_token, str) and cell_token in self.cell_to_idx:
                cell_idx = int(self.cell_to_idx[cell_token])
            else:
                missing_in_cells += 1
                if self.require_cell_labels:
                    continue
                cell_idx = -1

            cache_idx = -1
            if self.cached_dir is not None:
                if pano_id in self._pano_to_cache_idx:
                    cache_idx = int(self._pano_to_cache_idx[pano_id])
                elif self.allow_unsafe_index_fallback:
                    cache_idx = int(len(cache_idxs))
                else:
                    missing_in_cache += 1
                    continue
            
            if self.image_dir is not None:
                # Check if image exists
                img_path = self.image_dir / f"image_{pano_id}.jpg"
                if not img_path.exists():
                    missing_images += 1
                    continue

            labels.append(c_idx)
            coords.append((lat, lng))
            cell_labels.append(cell_idx)
            country_labels.append(country_label)
            cache_idxs.append(cache_idx)
            pano_ids.append(pano_id)

        if not labels:
            raise RuntimeError(
                "No usable samples after filtering. "
                f"missing_in_cache={missing_in_cache}, missing_in_vocab={missing_in_vocab}, "
                f"missing_images={missing_images}"
            )

        self._labels = torch.tensor(labels, dtype=torch.long)
        self._coords = torch.tensor(coords, dtype=torch.float32)
        self._cell_labels = torch.tensor(cell_labels, dtype=torch.long)
        self._country_labels = torch.tensor(country_labels, dtype=torch.long)
        self._cache_idxs = torch.tensor(cache_idxs, dtype=torch.long)
        self._pano_ids = pano_ids

    def __len__(self) -> int:
        return int(self._labels.shape[0])

    def __getitem__(self, idx: int):
        pano_id = self._pano_ids[idx]
        
        # Load vision data
        if self.image_dir is not None:
            img_path = self.image_dir / f"image_{pano_id}.jpg"
            vision_input = Image.open(img_path).convert("RGB")
            if self.transform is not None:
                vision_input = self.transform(vision_input)
        else:
            cache_idx = int(self._cache_idxs[idx].item())
            vision_input = self.patch_tokens[cache_idx]  # [P, D]

        c_label = self._labels[idx]
        coords = self._coords[idx]
        cell_label = self._cell_labels[idx]
        country_label = self._country_labels[idx]
        cache_idx = int(self._cache_idxs[idx].item())

        # Include pooled embeddings if available (for global head)
        pooled_emb = None
        if self.pooled_embeddings is not None:
            pooled_emb = self.pooled_embeddings[cache_idx]  # [D]
        
        return vision_input, c_label, coords, cell_label, country_label, cache_idx, pooled_emb


def collate_fn(batch: List[Any]):
    patches = torch.stack([b[0] for b in batch], dim=0)  # [B,P,D]
    c_labels = torch.stack([b[1] for b in batch], dim=0)  # [B]
    coords = torch.stack([b[2] for b in batch], dim=0)  # [B,2]
    cell_labels = torch.stack([b[3] for b in batch], dim=0)  # [B]
    country_labels = torch.stack([b[4] for b in batch], dim=0)  # [B]
    offsets = torch.tensor([b[5] for b in batch], dtype=torch.long)  # [B]
    # Handle pooled embeddings
    pooled_embs = [b[6] for b in batch]
    if all(pe is not None for pe in pooled_embs):
        pooled_emb = torch.stack(pooled_embs, dim=0)  # [B, D]
    elif any(pe is not None for pe in pooled_embs):
        # Mixed None/not-None is an error - either all or none should be present
        raise ValueError(
            f"collate_fn: Inconsistent pooled_embeddings in batch. "
            f"Some samples have pooled_emb, others don't. "
            f"This indicates a dataset loading error."
        )
    else:
        pooled_emb = None
    return patches, c_labels, coords, cell_labels, country_labels, offsets, pooled_emb


