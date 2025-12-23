#!/usr/bin/env python3
"""
Stage 1 Training Script: Text-Prototype Based Concept Learning

This script implements the optimized Stage 1 training with:
- Text-prototype based concept classification (not MLP head)
- Learnable prototype residuals for fine-tuning
- Hierarchical supervision (meta + parent concepts)
- Focal loss for handling class imbalance
- Concept-prototype contrastive alignment

Usage:
    python scripts/training/train_stage1_prototype.py \
        --csv_path data/dataset-43k-mapped.csv \
        --resume_from_checkpoint results/.../best_model_stage0.pt \
        --stage1_epochs 50 \
        --use_wandb
"""

import argparse
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image
import wandb
import csv
from sklearn.metrics import recall_score

from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified_strict,
    save_splits_to_json,
    load_splits_from_json,
    log_split_diagnostics,
    get_transforms_from_processor,
    get_train_transforms,
    get_val_transforms,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import (
    Stage1ConceptModel,
    build_text_prototypes,
    build_meta_to_parent_idx,
    DEFAULT_CONCEPT_TEMPLATES,
    DEFAULT_PARENT_TEMPLATES,
    TransformerBottleneck,
)
from src.losses import (
    FocalLoss, 
    concept_prototype_contrastive_loss, 
    hierarchical_consistency_loss, 
    parent_guided_meta_loss, 
    inter_parent_contrastive_loss,
    compute_concept_similarity_matrix,
    SemanticSoftCrossEntropy,
    semantic_soft_cross_entropy,
    compute_semantic_close_accuracy,
)
from src.concepts.utils import extract_concepts_from_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# PRECOMPUTED EMBEDDINGS DATASET
# ============================================================================

class PrecomputedEmbeddingsDataset(Dataset):
    """Dataset wrapper that returns precomputed embeddings instead of images."""
    
    def __init__(
        self,
        embeddings: Optional[torch.Tensor],
        concept_indices: torch.Tensor,
        parent_indices: torch.Tensor,
        country_indices: torch.Tensor,
        coordinates: torch.Tensor,
        cell_labels: Optional[torch.Tensor] = None,
        metadata: Optional[List[Dict]] = None,
        embedding_dir: Optional[Path] = None,
    ):
        self.embeddings = embeddings
        self.concept_indices = concept_indices
        self.parent_indices = parent_indices
        self.country_indices = country_indices
        self.coordinates = coordinates
        self.cell_labels = cell_labels if cell_labels is not None else torch.zeros(len(concept_indices), dtype=torch.long)
        self.metadata = metadata if metadata is not None else [{} for _ in range(len(concept_indices))]
        self.embedding_dir = Path(embedding_dir) if embedding_dir is not None else None
        
        expected_len = len(concept_indices)
        for name, arr in [
            ("parent_indices", parent_indices),
            ("country_indices", country_indices),
            ("coordinates", coordinates),
            ("cell_labels", self.cell_labels),
            ("metadata", self.metadata),
        ]:
            if len(arr) != expected_len:
                raise ValueError(f"{name} length {len(arr)} != expected {expected_len}")
        if embeddings is not None and len(embeddings) != expected_len:
            raise ValueError(f"embeddings length {len(embeddings)} != expected {expected_len}")
    
    def __len__(self):
        return len(self.concept_indices)
    
    def _load_embedding(self, idx: int) -> Dict[str, torch.Tensor]:
        """Load embedding data from disk or memory.
        
        Returns:
            Dict with 'embedding' (768d) and optionally 'concept_emb' (512d)
        """
        if self.embeddings is not None:
            # In-memory embeddings (legacy format)
            return {"embedding": self.embeddings[idx]}
        if self.embedding_dir is None:
            raise ValueError("No embeddings in memory and no embedding_dir provided.")
        
        entry = self.metadata[idx] if isinstance(self.metadata, list) else {}
        emb_file = entry.get("embedding_file")
        if emb_file is None:
            raise ValueError("Missing embedding_file in metadata for disk-backed dataset.")
        emb_path = self.embedding_dir / emb_file
        loaded = torch.load(emb_path, weights_only=True)
        if isinstance(loaded, dict) and "embedding" in loaded:
            # New format: {"embedding": tensor, "concept_emb": tensor (optional)}
            return loaded
        # Legacy format: just the tensor
        return {"embedding": loaded}
    
    @staticmethod
    def _to_tensor(val, dtype=torch.long):
        if torch.is_tensor(val):
            return val
        return torch.tensor(val, dtype=dtype)
    
    def __getitem__(self, idx):
        emb_data = self._load_embedding(idx)
        embedding = emb_data["embedding"]
        concept_emb = emb_data.get("concept_emb")  # Optional, for Stage 2
        
        concept_idx = self._to_tensor(self.concept_indices[idx], dtype=torch.long)
        parent_idx = self._to_tensor(self.parent_indices[idx], dtype=torch.long)
        country_idx = self._to_tensor(self.country_indices[idx], dtype=torch.long)
        coords = self.coordinates[idx]
        if not torch.is_tensor(coords):
            coords = torch.tensor(coords, dtype=torch.float32)
        cell_label = self._to_tensor(self.cell_labels[idx], dtype=torch.long)
        meta = self.metadata[idx] if isinstance(self.metadata, list) else {}
        
        # Include concept_emb in metadata for backward compatibility
        if concept_emb is not None:
            meta = dict(meta)  # Copy to avoid mutating original
            meta["concept_emb"] = concept_emb
        
        return (
            embedding,
            concept_idx,
            parent_idx,
            country_idx,
            coords,
            cell_label,
            meta,
        )
    
    @classmethod
    def from_cache_dir(cls, cache_dir: Path):
        cache_dir = Path(cache_dir)
        manifest_path = cache_dir / "manifest.pt"
        if not manifest_path.exists():
            raise FileNotFoundError(f"Manifest not found at {manifest_path}")
        
        manifest = torch.load(manifest_path, weights_only=True)
        entries = manifest.get("entries", manifest)
        
        concept_idx = torch.tensor([e.get("concept_idx", 0) for e in entries], dtype=torch.long)
        parent_idx = torch.tensor([e.get("parent_idx", 0) for e in entries], dtype=torch.long)
        country_idx = torch.tensor([e.get("country_idx", 0) for e in entries], dtype=torch.long)
        coords = torch.tensor([e.get("coords", [float("nan"), float("nan")]) for e in entries], dtype=torch.float32)
        cell_labels = torch.tensor([e.get("cell_label", 0) for e in entries], dtype=torch.long)
        metadata = []
        for e in entries:
            metadata.append({
                "pano_id": e.get("pano_id"),
                "image_path": e.get("image_path"),
                "meta_name": e.get("meta_name"),
                "parent_concept": e.get("parent_concept"),
                "country": e.get("country"),
                "embedding_file": e.get("embedding_file"),
            })
        
        return cls(
            embeddings=None,
            concept_indices=concept_idx,
            parent_indices=parent_idx,
            country_indices=country_idx,
            coordinates=coords,
            cell_labels=cell_labels,
            metadata=metadata,
            embedding_dir=cache_dir,
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def sanitize_for_filename(name: str) -> str:
    """Sanitize a string so it can be safely used as a filename."""
    safe = "".join(c if c.isalnum() or c in ("-", "_") else "_" for c in str(name))
    return safe[:120] if safe else "unknown"


def split_metadata_batch(metadata, batch_size: int) -> List[Dict]:
    """
    Convert a collated metadata batch (dict of lists) into a list of dicts.
    """
    if isinstance(metadata, list):
        return metadata
    if not isinstance(metadata, dict):
        return [{} for _ in range(batch_size)]
    
    result = []
    for i in range(batch_size):
        entry = {}
        for k, v in metadata.items():
            try:
                entry[k] = v[i]
            except Exception:
                entry[k] = v
        result.append(entry)
    return result


def compute_class_weights(
    samples: List[Dict],
    concept_to_idx: Dict[str, int],
    device: torch.device,
    key: str = 'meta_name',
) -> torch.Tensor:
    """Compute inverse-frequency class weights for balanced training."""
    labels = [concept_to_idx[s[key]] for s in samples]
    counts = Counter(labels)
    num_classes = len(concept_to_idx)
    total = len(labels)
    
    weights = torch.zeros(num_classes, device=device)
    for idx, count in counts.items():
        weights[idx] = total / (num_classes * count)
    
    # Normalize so weights sum to num_classes
    weights = weights * (num_classes / weights.sum())
    
    return weights


def compute_parent_weights(
    samples: List[Dict],
    parent_to_idx: Dict[str, int],
    device: torch.device,
) -> torch.Tensor:
    """Compute inverse-frequency class weights for parent concepts."""
    labels = [parent_to_idx.get(s.get('parent_concept', 'unknown'), 0) for s in samples]
    counts = Counter(labels)
    num_classes = len(parent_to_idx)
    total = len(labels)
    
    weights = torch.zeros(num_classes, device=device)
    for idx, count in counts.items():
        weights[idx] = total / (num_classes * count)
    
    weights = weights * (num_classes / weights.sum())
    
    return weights


def get_embedding_cache_path(
    checkpoint_path: Optional[str],
    encoder_model: str,
    data_root: str,
    split: str,
) -> Path:
    """
    Generate a cache path for precomputed embeddings based on the model checkpoint.
    
    Args:
        checkpoint_path: Path to the stage 0 checkpoint (or None for pretrained)
        encoder_model: Name of the encoder model (e.g., 'geolocal/StreetCLIP')
        data_root: Root data directory
        split: Dataset split ('train' or 'val')
        
    Returns:
        Path to the cached embeddings file
    """
    if checkpoint_path:
        # Extract a meaningful name from checkpoint path
        # e.g., "results/concept-aware-3-stage/streetclip/global/haversine/2025-12-04_20-07-33/checkpoints/best_model_stage0.pt"
        # -> "concept-aware-3-stage_haversine_2025-12-04_20-07-33"
        ckpt_path = Path(checkpoint_path)
        parts = ckpt_path.parts
        
        # Try to extract meaningful parts
        model_name_parts = []
        for part in parts:
            if part in ('results', 'checkpoints', 'streetclip', 'global', 'sequential'):
                continue
            if part.endswith('.pt') or part.endswith('.pth'):
                continue
            model_name_parts.append(part)
        
        model_name = "_".join(model_name_parts[-3:]) if len(model_name_parts) >= 3 else "_".join(model_name_parts)
        if not model_name:
            # Fallback: use hash of checkpoint path
            model_name = hashlib.md5(checkpoint_path.encode()).hexdigest()[:12]
    else:
        # No checkpoint - use encoder model name
        model_name = encoder_model.replace("/", "_") + "_pretrained"
    
    # Sanitize model name
    model_name = model_name.replace("/", "_").replace(" ", "_")
    
    cache_dir = Path(data_root) / "precomputed_embeddings" / model_name
    return cache_dir / f"{split}.pt"


def get_embedding_cache_dir(
    checkpoint_path: Optional[str],
    encoder_model: str,
    data_root: str,
    split: str,
) -> Path:
    """Directory to store per-pano cached embeddings."""
    cache_path = get_embedding_cache_path(checkpoint_path, encoder_model, data_root, split)
    return cache_path.parent / cache_path.stem


def save_cached_embeddings(
    cache_path: Path,
    embeddings: Tuple[torch.Tensor, ...],
    metadata: Optional[List[Dict]] = None,
) -> None:
    """Save precomputed embeddings to cache."""
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    
    cache_data = {
        'embeddings': embeddings[0],
        'concept_idx': embeddings[1],
        'parent_idx': embeddings[2],
        'country_idx': embeddings[3],
        'coords': embeddings[4],
        'cell_labels': embeddings[5],
    }
    if metadata is not None:
        cache_data['metadata'] = metadata
    
    torch.save(cache_data, cache_path)
    logger.info(f"Saved cached embeddings to {cache_path}")


def save_embeddings_per_pano(
    cache_dir: Path,
    embeddings: Tuple[torch.Tensor, ...],
    metadata: Optional[List[Dict]] = None,
    concept_embeddings: Optional[torch.Tensor] = None,
) -> None:
    """Save per-pano embeddings and a manifest for disk-backed loading.
    
    Args:
        cache_dir: Directory to save embeddings
        embeddings: Tuple of (image_embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels)
        metadata: Optional list of metadata dicts per sample
        concept_embeddings: Optional concept embeddings [N, 512] from concept bottleneck (for Stage 2)
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    entries = []
    
    total = len(embeddings[0])
    has_concept_emb = concept_embeddings is not None and len(concept_embeddings) == total
    
    for i in range(total):
        meta_entry = metadata[i] if metadata is not None and i < len(metadata) else {}
        pano_id = meta_entry.get("pano_id", f"idx_{i}")
        filename = f"{sanitize_for_filename(pano_id)}.pt"
        emb_path = cache_dir / filename
        
        # Save both image embedding and concept embedding (if available)
        save_data = {"embedding": embeddings[0][i]}
        if has_concept_emb:
            save_data["concept_emb"] = concept_embeddings[i]
        
        torch.save(save_data, emb_path)
        
        entries.append({
            "pano_id": pano_id,
            "embedding_file": filename,
            "concept_idx": int(embeddings[1][i]),
            "parent_idx": int(embeddings[2][i]),
            "country_idx": int(embeddings[3][i]),
            "coords": embeddings[4][i].tolist(),
            "cell_label": int(embeddings[5][i]) if embeddings[5] is not None else 0,
            "image_path": str(meta_entry.get("image_path")) if meta_entry.get("image_path") is not None else None,
            "meta_name": meta_entry.get("meta_name"),
            "parent_concept": meta_entry.get("parent_concept"),
            "country": meta_entry.get("country"),
            "has_concept_emb": has_concept_emb,
        })
    
    manifest = {"entries": entries, "has_concept_emb": has_concept_emb}
    torch.save(manifest, cache_dir / "manifest.pt")
    logger.info(f"Saved per-pano embeddings to {cache_dir} (n={len(entries)}, concept_emb={has_concept_emb})")


def load_cached_embeddings(
    cache_path: Path,
) -> Optional[Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], Optional[List[Dict]]]]:
    """Load precomputed embeddings from cache if available."""
    if not cache_path.exists():
        return None
    
    try:
        cache_data = torch.load(cache_path, weights_only=True)
        logger.info(f"Loaded cached embeddings from {cache_path}")
        return (
            (
                cache_data['embeddings'],
                cache_data['concept_idx'],
                cache_data['parent_idx'],
                cache_data['country_idx'],
                cache_data['coords'],
                cache_data['cell_labels'],
            ),
            cache_data.get('metadata'),
        )
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings: {e}")
        return None


def precompute_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor], List[Dict]]:
    """Precompute image embeddings using frozen image encoder."""
    model.eval()
    
    all_embeddings = []
    all_concept_idx = []
    all_parent_idx = []
    all_country_idx = []
    all_coords = []
    all_cell_labels = []
    all_metadata = []
    
    for batch in tqdm(dataloader, desc="Precomputing embeddings"):
        images, concept_idx, parent_idx, country_idx, coords, metadata = batch
        images = images.to(device)
        
        # Extract features using frozen encoder
        features = model.image_encoder(images)
        
        all_embeddings.append(features.cpu())
        all_concept_idx.append(concept_idx)
        all_parent_idx.append(parent_idx)
        all_country_idx.append(country_idx)
        all_coords.append(coords)
        
        # Get cell labels from metadata if available
        if 'cell_label' in metadata:
            all_cell_labels.append(torch.tensor(metadata['cell_label']))
        else:
            all_cell_labels.append(torch.zeros(len(concept_idx), dtype=torch.long))
        
        # Store per-sample metadata (pano_id, image_path, etc.)
        batch_meta = split_metadata_batch(metadata, len(concept_idx))
        all_metadata.extend([
            {
                "pano_id": m.get("pano_id"),
                "image_path": m.get("image_path"),
                "meta_name": m.get("meta_name"),
                "parent_concept": m.get("parent_concept"),
                "country": m.get("country"),
            }
            for m in batch_meta
        ])
    
    embeddings_tuple = (
        torch.cat(all_embeddings, dim=0),
        torch.cat(all_concept_idx, dim=0),
        torch.cat(all_parent_idx, dim=0),
        torch.cat(all_country_idx, dim=0),
        torch.cat(all_coords, dim=0),
        torch.cat(all_cell_labels, dim=0),
    )
    return embeddings_tuple, all_metadata


@torch.no_grad()
def compute_concept_embeddings(
    model: Stage1ConceptModel,
    dataloader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """
    Compute concept embeddings from trained Stage 1 model.
    
    This runs the concept bottleneck on precomputed image embeddings to get
    the concept representation for each sample. Used for Stage 2 training.
    
    Args:
        model: Trained Stage1ConceptModel
        dataloader: DataLoader with PrecomputedEmbeddingsDataset
        device: Device to run on
        
    Returns:
        Tensor of concept embeddings [N, concept_emb_dim]
    """
    model.eval()
    all_concept_embs = []
    
    for batch in tqdm(dataloader, desc="Computing concept embeddings"):
        # PrecomputedEmbeddingsDataset returns:
        # (embedding, concept_idx, parent_idx, country_idx, coords, cell_label, meta)
        embeddings = batch[0].to(device)
        
        # Get concept embeddings through the model's concept bottleneck
        concept_emb = model.concept_bottleneck(embeddings)
        all_concept_embs.append(concept_emb.cpu())
    
    return torch.cat(all_concept_embs, dim=0)


def save_concept_embeddings_to_cache(
    cache_dir: Path,
    concept_embeddings: torch.Tensor,
) -> None:
    """
    Add concept embeddings to existing per-pano embedding cache.
    
    This loads each existing .pt file and adds the concept_emb key.
    """
    cache_dir = Path(cache_dir)
    manifest_path = cache_dir / "manifest.pt"
    
    if not manifest_path.exists():
        raise FileNotFoundError(f"Manifest not found at {manifest_path}")
    
    manifest = torch.load(manifest_path, weights_only=True)
    entries = manifest.get("entries", manifest)
    
    if len(entries) != len(concept_embeddings):
        raise ValueError(f"Mismatch: {len(entries)} entries vs {len(concept_embeddings)} concept embeddings")
    
    for i, entry in enumerate(tqdm(entries, desc="Saving concept embeddings")):
        emb_file = entry.get("embedding_file")
        if emb_file is None:
            continue
        
        emb_path = cache_dir / emb_file
        loaded = torch.load(emb_path, weights_only=True)
        
        # Add concept embedding
        loaded["concept_emb"] = concept_embeddings[i]
        torch.save(loaded, emb_path)
        
        # Update entry flag
        entry["has_concept_emb"] = True
    
    # Update manifest
    manifest["has_concept_emb"] = True
    torch.save(manifest, manifest_path)
    logger.info(f"Added concept embeddings to {len(entries)} files in {cache_dir}")


def save_checkpoint(
    model: Stage1ConceptModel,
    checkpoint_path: Path,
    concept_names: List[str],
    parent_names: List[str],
    concept_to_idx: Dict[str, int],
    parent_to_idx: Dict[str, int],
    country_to_idx: Dict[str, int],
    encoder_model: str,
    extra_info: Optional[Dict] = None,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
):
    """Save Stage 1 model checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "concept_names": concept_names,
        "parent_names": parent_names,
        "num_concepts": len(concept_names),
        "num_parents": len(parent_names),
        "concept_to_idx": concept_to_idx,
        "parent_to_idx": parent_to_idx,
        "country_to_idx": country_to_idx,
        "encoder_model": encoder_model,
        "T_meta_base": model.T_meta_base.cpu(),
        "T_parent_base": model.T_parent_base.cpu(),
        "meta_to_parent_idx": model.meta_to_parent_idx.cpu(),
    }
    
    if optimizer is not None:
        checkpoint["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        checkpoint["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        checkpoint["epoch"] = epoch
    if extra_info:
        checkpoint.update(extra_info)
    
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(checkpoint, checkpoint_path)
    logger.info(f"Saved checkpoint to {checkpoint_path}")


def log_metrics(metrics: Dict[str, float], prefix: str = "", stage: int = None):
    """Log metrics in a structured format."""
    stage_str = f"[Stage {stage}] " if stage is not None else ""
    parts = []
    
    if "loss" in metrics:
        parts.append(f"Loss: {metrics['loss']:.4f}")
    if "meta_acc" in metrics:
        parts.append(f"Meta Acc: {metrics['meta_acc']:.3f}")
    if "meta_acc_top5" in metrics:
        parts.append(f"Meta Top5: {metrics['meta_acc_top5']:.3f}")
    if "meta_recall" in metrics:
        parts.append(f"Meta Recall: {metrics['meta_recall']:.3f}")
    if "parent_acc" in metrics:
        parts.append(f"Parent Acc: {metrics['parent_acc']:.3f}")
    if "parent_acc_top5" in metrics:
        parts.append(f"Parent Top5: {metrics['parent_acc_top5']:.3f}")
    if "parent_recall" in metrics:
        parts.append(f"Parent Recall: {metrics['parent_recall']:.3f}")
    if "country_acc" in metrics:
        parts.append(f"Country Acc: {metrics['country_acc']:.3f}")
    
    if parts:
        logger.info(f"{stage_str}{prefix}: {' | '.join(parts)}")


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate(
    model: Stage1ConceptModel,
    dataloader: DataLoader,
    device: torch.device,
    focal_loss_meta: FocalLoss,
    focal_loss_parent: FocalLoss,
    args,
    use_precomputed: bool = False,
    similarity_matrix: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    """Validate Stage 1 model."""
    model.eval()
    
    total_loss = 0
    total_meta_correct = 0
    total_meta_correct_top5 = 0
    total_parent_correct = 0
    total_parent_correct_top5 = 0
    total_semantic_close_correct = 0
    total_count = 0
    
    all_pred_meta = []
    all_target_meta = []
    all_pred_parent = []
    all_target_parent = []
    
    for batch in dataloader:
        if use_precomputed:
            if len(batch) == 7:
                embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels, _ = batch
            else:
                embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels = batch
            embeddings = embeddings.to(device)
        else:
            images, concept_idx, parent_idx, country_idx, coords, _ = batch
            images = images.to(device)
        
        concept_idx = concept_idx.to(device)
        parent_idx = parent_idx.to(device)
        
        # Forward pass
        if use_precomputed:
            outputs = model.forward_from_features(embeddings)
        else:
            outputs = model(images)
        
        meta_logits = outputs["meta_logits"]
        parent_logits = outputs["parent_logits"]
        
        # Losses
        loss_meta = focal_loss_meta(meta_logits, concept_idx)
        loss_parent = focal_loss_parent(parent_logits, parent_idx)
        loss = args.lambda_meta * loss_meta + args.lambda_parent * loss_parent
        
        total_loss += loss.item() * len(concept_idx)
        
        # Accuracy
        pred_meta = meta_logits.argmax(dim=1)
        pred_parent = parent_logits.argmax(dim=1)
        
        total_meta_correct += (pred_meta == concept_idx).sum().item()
        total_parent_correct += (pred_parent == parent_idx).sum().item()
        
        # Top-5 Accuracy
        _, pred_meta_top5 = meta_logits.topk(5, dim=1, largest=True, sorted=True)
        total_meta_correct_top5 += (pred_meta_top5 == concept_idx.view(-1, 1)).sum().item()
        
        _, pred_parent_top5 = parent_logits.topk(5, dim=1, largest=True, sorted=True)
        total_parent_correct_top5 += (pred_parent_top5 == parent_idx.view(-1, 1)).sum().item()
        
        # Collect for Recall
        all_pred_meta.append(pred_meta.cpu())
        all_target_meta.append(concept_idx.cpu())
        all_pred_parent.append(pred_parent.cpu())
        all_target_parent.append(parent_idx.cpu())
        
        # Semantic-close accuracy (if similarity matrix provided)
        if similarity_matrix is not None:
            _, semantic_close_acc_batch = compute_semantic_close_accuracy(
                pred_meta, concept_idx, similarity_matrix, 
                threshold=getattr(args, 'semantic_close_threshold', 0.7)
            )
            total_semantic_close_correct += semantic_close_acc_batch * len(concept_idx)
        
        total_count += len(concept_idx)
    
    # Compute Recall
    all_pred_meta = torch.cat(all_pred_meta).numpy()
    all_target_meta = torch.cat(all_target_meta).numpy()
    all_pred_parent = torch.cat(all_pred_parent).numpy()
    all_target_parent = torch.cat(all_target_parent).numpy()
    
    meta_recall = recall_score(all_target_meta, all_pred_meta, average='macro', zero_division=0)
    parent_recall = recall_score(all_target_parent, all_pred_parent, average='macro', zero_division=0)
    
    result = {
        "loss": total_loss / total_count,
        "meta_acc": total_meta_correct / total_count,
        "meta_acc_top5": total_meta_correct_top5 / total_count,
        "meta_recall": meta_recall,
        "parent_acc": total_parent_correct / total_count,
        "parent_acc_top5": total_parent_correct_top5 / total_count,
        "parent_recall": parent_recall,
    }
    
    # Add semantic-close accuracy if computed
    if similarity_matrix is not None:
        result["semantic_close_acc"] = total_semantic_close_correct / total_count
    
    return result

# ============================================================================
# VISUALIZATION
# ============================================================================

# Visualization constants
VIZ_NUM_SAMPLES = 4
VIZ_TOP_K_CONCEPTS = 5
VIZ_FIGSIZE = (10, 8)
VIZ_DPI = 150
WAND_TABLE_MAX_ROWS = 200

# CLIP normalization constants
CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]


@torch.no_grad()
def visualize_concept_predictions(
    model: Stage1ConceptModel,
    image_dataloader: DataLoader,
    device: torch.device,
    idx_to_concept: Dict[int, str],
    idx_to_parent: Dict[int, str],
    output_dir: Path,
    epoch: int,
    num_samples: int = VIZ_NUM_SAMPLES,
    log_to_wandb: bool = True,
    wandb_step: Optional[int] = None,
    use_precomputed: bool = False,
):
    """
    Visualize concept predictions as a 2x2 grid with top 5 concepts as bar plots.
    
    Creates a single figure with 4 samples in a 2x2 grid, each showing:
    - Image with title (pano_id, GT Concept vs Pred Concept)
    - Bar chart of top-5 concept probabilities (GT = orange, others = blue)
    
    Args:
        model: Stage1ConceptModel
        image_dataloader: DataLoader that returns images (or precomputed embeddings)
        device: Device to run inference on
        idx_to_concept: Mapping from concept index to concept name
        idx_to_parent: Mapping from parent index to parent concept name
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        num_samples: Number of samples to visualize (will use min(num_samples, 4) for 2x2 grid)
        log_to_wandb: Whether to log to wandb
        wandb_step: Optional step for wandb logging
        use_precomputed: If True, expects embeddings + metadata and calls forward_from_features
    """
    model.eval()
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # Get a batch from the image dataloader
    batch = next(iter(image_dataloader))
    
    if use_precomputed:
        if len(batch) == 7:
            embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels, metadata = batch
        else:
            embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels = batch
            metadata = {}
        embeddings = embeddings.to(device)
    else:
        images, concept_idx, parent_idx, country_idx, coords, metadata = batch
        images = images.to(device)
    concept_idx = concept_idx.to(device)
    parent_idx = parent_idx.to(device)
    sample_metadata = split_metadata_batch(metadata, len(concept_idx)) if metadata is not None else [{} for _ in range(len(concept_idx))]
    
    # Get predictions
    if use_precomputed:
        outputs = model.forward_from_features(embeddings)
    else:
        outputs = model(images)
    meta_logits = outputs["meta_logits"]
    meta_probs = outputs["meta_probs"]
    parent_logits = outputs.get("parent_logits")
    parent_probs = outputs.get("parent_probs")
    if parent_probs is None and parent_logits is not None:
        parent_probs = F.softmax(parent_logits, dim=1)
    
    # Process up to 4 samples for 2x2 grid
    batch_size = len(embeddings) if use_precomputed else len(images)
    n_samples = min(4, num_samples, batch_size)  # Max 4 for 2x2 grid
    
    # Create 2x2 grid figure (each cell has image on top, bar chart on bottom)
    # Use gridspec for better control: 2 rows x 2 cols, each cell subdivided into 2 rows
    fig = plt.figure(figsize=(28, 16))
    
    # Create outer grid (2x2)
    outer_grid = fig.add_gridspec(2, 2, hspace=0.3, wspace=0.2)
    
    for i in range(n_samples):
        row, col = i // 2, i % 2
        
        # Create inner grid for this sample (2 rows: image + bar chart)
        inner_grid = outer_grid[row, col].subgridspec(2, 1, height_ratios=[1.5, 1], hspace=0.15)
        ax_img = fig.add_subplot(inner_grid[0])
        ax_bar = fig.add_subplot(inner_grid[1])
        
        # ====== Top: Image ======
        img_display = None
        if use_precomputed:
            meta_entry = sample_metadata[i] if i < len(sample_metadata) else {}
            img_path = meta_entry.get("image_path")
            if img_path and Path(img_path).exists():
                img_display = np.array(Image.open(img_path).convert("RGB"))
        else:
            img = images[i].cpu()
            # Denormalize for display (CLIP normalization)
            img_denorm = img.clone()
            mean = torch.tensor(CLIP_MEAN).view(3, 1, 1)
            std = torch.tensor(CLIP_STD).view(3, 1, 1)
            img_denorm = img_denorm * std + mean
            img_denorm = torch.clamp(img_denorm, 0, 1)
            img_display = img_denorm.permute(1, 2, 0).numpy()
        
        if img_display is None:
            img_display = np.zeros((10, 10, 3))
        
        ax_img.imshow(img_display)
        ax_img.axis("off")

        # Get ground truth and predicted CHILD concepts
        true_concept_idx = concept_idx[i].item()
        true_concept = idx_to_concept.get(true_concept_idx, f"Unknown_{true_concept_idx}")
        true_concept_prob = meta_probs[i][true_concept_idx].item()
        
        pred_concept_idx = meta_logits[i].argmax().item()
        pred_concept = idx_to_concept.get(pred_concept_idx, f"Unknown_{pred_concept_idx}")
        pred_concept_prob = meta_probs[i][pred_concept_idx].item()

        # Get ground truth and predicted PARENT concepts
        true_parent_str = ""
        pred_parent_str = ""
        if parent_logits is not None and parent_probs is not None:
            true_parent_idx = parent_idx[i].item()
            true_parent = idx_to_parent.get(true_parent_idx, f"Unknown_{true_parent_idx}")
            true_parent_prob = parent_probs[i][true_parent_idx].item()
            pred_parent_idx = parent_logits[i].argmax().item()
            pred_parent = idx_to_parent.get(pred_parent_idx, f"Unknown_{pred_parent_idx}")
            pred_parent_prob = parent_probs[i][pred_parent_idx].item()
            true_parent_str = f"GT Parent: {true_parent}"
            pred_parent_str = f"Pred Parent: {pred_parent}"
        
        meta_entry = sample_metadata[i] if i < len(sample_metadata) else {}
        pano_id = meta_entry.get('pano_id', 'unknown')
        
        # ====== Bottom: Top K concepts bar plot ======
        top5_probs, top5_indices = torch.topk(meta_probs[i], k=VIZ_TOP_K_CONCEPTS)
        top5_concepts = [idx_to_concept.get(idx.item(), f"Unknown_{idx.item()}") for idx in top5_indices]
        top5_probs_np = top5_probs.cpu().numpy()
        
        # Check if ground truth is in top 5
        true_in_top5 = true_concept_idx in top5_indices.cpu().numpy()

        # Compact title for grid layout
        is_correct = "✓" if true_in_top5 else "✗"
        title = f"{is_correct} {pano_id[:12]}...\n"
        title += f"GT Child: {true_concept[:20]}... | Pred Child: {pred_concept[:20]}..."
        if true_parent_str:
            title += f"\n{true_parent_str} | {pred_parent_str}"
        ax_img.set_title(title, fontsize=8, loc='left')
        
        # Reverse arrays to show highest probability at top (horizontal bar chart)
        top5_concepts_reversed = list(reversed(top5_concepts))
        top5_probs_np_reversed = top5_probs_np[::-1]
        top5_indices_reversed = list(reversed(top5_indices.cpu().numpy()))
        
        # Color bars: highlight ground truth if in top 5, otherwise use default color
        bar_colors = [
            "orange" if idx == true_concept_idx else "steelblue"
            for idx in top5_indices_reversed
        ]
        
        bars = ax_bar.barh(
            range(len(top5_concepts_reversed)),
            top5_probs_np_reversed,
            color=bar_colors,
        )
        ax_bar.set_yticks(range(len(top5_concepts_reversed)))
        
        # Add (GT) label to ground truth concept in y-axis labels (truncate for grid)
        yticklabels = []
        for concept, idx in zip(top5_concepts_reversed, top5_indices_reversed):
            label = concept[:18] + "..." if len(concept) > 18 else concept
            if idx == true_concept_idx:
                label = f"{label} (GT)"
            yticklabels.append(label)
        ax_bar.set_yticklabels(yticklabels, fontsize=7)
        
        ax_bar.set_xlabel("Probability", fontsize=8)
        bar_title = "Top 5 Concepts"
        if not true_in_top5:
            bar_title += f" | GT not in top5"
        ax_bar.set_title(bar_title, fontsize=8)
        ax_bar.set_xlim(0, 1)
        
        # Add value labels on bars
        for j, (bar, prob) in enumerate(zip(bars, top5_probs_np_reversed)):
            ax_bar.text(prob + 0.01, j, f"{prob:.2f}", va="center", fontsize=7)
    
    # Add overall title
    fig.suptitle(f"Epoch {epoch}", fontsize=12, fontweight='bold')
    
    # Save the combined grid figure
    save_path = viz_dir / f"epoch_{epoch}_grid.png"
    plt.savefig(save_path, dpi=VIZ_DPI, bbox_inches="tight")
    plt.close(fig)
    
    logger.info(f"Saved 2x2 grid visualization to {viz_dir} for epoch {epoch}")
    
    if log_to_wandb:
        step = wandb_step if wandb_step is not None else epoch
        # Log single grid image to wandb under predictions/ panel
        wandb.log({
            "predictions/concept_grid": wandb.Image(str(save_path), caption=f"Epoch {epoch} - Concept Predictions Grid")
        }, step=step)


@torch.no_grad()
def log_validation_predictions_table(
    model: Stage1ConceptModel,
    dataloader: DataLoader,
    device: torch.device,
    idx_to_concept: Dict[int, str],
    idx_to_parent: Dict[int, str],
    use_precomputed: bool = False,
    max_rows: int = WAND_TABLE_MAX_ROWS,
    step: Optional[int] = None,
    output_dir: Path = None,
):
    """
    Log a compact table of validation predictions to wandb.
    
    Columns: pano_id, country, gt child concept, predicted child concept,
    gt parent concept, predicted parent concept, child_correct, parent_correct,
    child_count, parent_count.
    """
    concept_to_idx = {name: idx for idx, name in idx_to_concept.items()}
    parent_to_idx = {name: idx for idx, name in idx_to_parent.items()}
    
    child_counts = None
    parent_counts = None
    dataset = getattr(dataloader, "dataset", None)
    if dataset is not None:
        if hasattr(dataset, "concept_indices") and hasattr(dataset, "parent_indices"):
            child_counts = torch.bincount(
                dataset.concept_indices, minlength=len(idx_to_concept)
            ).tolist()
            parent_counts = torch.bincount(
                dataset.parent_indices, minlength=len(idx_to_parent)
            ).tolist()
        elif hasattr(dataset, "samples"):
            child_counts = [0] * len(idx_to_concept)
            parent_counts = [0] * len(idx_to_parent)
            for sample in dataset.samples:
                child_idx = concept_to_idx.get(sample.get("meta_name"))
                if child_idx is not None:
                    child_counts[child_idx] += 1
                
                parent_name = sample.get("parent_concept", "unknown")
                parent_idx = parent_to_idx.get(parent_name)
                if parent_idx is not None:
                    parent_counts[parent_idx] += 1
    
    rows = []
    for batch in dataloader:
        if use_precomputed:
            if len(batch) == 7:
                embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels, metadata = batch
            else:
                embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels = batch
                metadata = {}
            embeddings = embeddings.to(device)
        else:
            images, concept_idx, parent_idx, country_idx, coords, metadata = batch
            images = images.to(device)
        
        concept_idx_dev = concept_idx.to(device)
        parent_idx_dev = parent_idx.to(device)
        
        outputs = model.forward_from_features(embeddings) if use_precomputed else model(images)
        meta_logits = outputs["meta_logits"]
        parent_logits = outputs["parent_logits"]
        
        pred_meta = meta_logits.argmax(dim=1).cpu()
        pred_parent = parent_logits.argmax(dim=1).cpu()
        
        # Get top-5 predictions
        _, pred_meta_top5 = meta_logits.topk(5, dim=1, largest=True, sorted=True)
        _, pred_parent_top5 = parent_logits.topk(5, dim=1, largest=True, sorted=True)
        pred_meta_top5 = pred_meta_top5.cpu()
        pred_parent_top5 = pred_parent_top5.cpu()
        
        meta_list = split_metadata_batch(metadata, len(pred_meta))
        
        for i in range(len(pred_meta)):
            gt_child = int(concept_idx[i])
            gt_parent = int(parent_idx[i])
            pred_child = int(pred_meta[i])
            pred_parent_val = int(pred_parent[i])
            country_name = meta_list[i].get("country", "unknown")
            
            child_correct = gt_child == pred_child
            parent_correct = gt_parent == pred_parent_val
            
            child_in_top5 = gt_child in pred_meta_top5[i]
            parent_in_top5 = gt_parent in pred_parent_top5[i]
            
            child_count_val = child_counts[gt_child] if child_counts is not None else 0
            parent_count_val = parent_counts[gt_parent] if parent_counts is not None else 0
            
            rows.append([
                str(meta_list[i].get("pano_id", "unknown")),
                country_name,
                idx_to_concept.get(gt_child, f"meta_{gt_child}"),
                idx_to_concept.get(pred_child, f"meta_{pred_child}"),
                idx_to_parent.get(gt_parent, f"parent_{gt_parent}"),
                idx_to_parent.get(pred_parent_val, f"parent_{pred_parent_val}"),
                child_correct,
                parent_correct,
                child_in_top5,
                parent_in_top5,
                child_count_val,
                parent_count_val,
            ])
            if len(rows) >= max_rows:
                break
        if len(rows) >= max_rows:
            break
    
    if not rows:
        return
    
    # Save rows to output directory / diagnostics/predictions_table.csv
    output_path = output_dir / "diagnostics" / f"predictions_table_{step}.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows, columns=["pano_id", "country", "gt_child_concept", "pred_child_concept", "gt_parent_concept", "pred_parent_concept", "child_correct", "parent_correct", "child_in_top5", "parent_in_top5", "count_child", "count_parent"])
    df.to_csv(output_path, index=False)
    logger.info(f"Saved predictions table to {output_path}")
    
    table = wandb.Table(
        columns=[
            "pano_id",
            "country",
            "gt_child_concept",
            "pred_child_concept",
            "gt_parent_concept",
            "pred_parent_concept",
            "child_correct",
            "parent_correct",
            "child_in_top5",
            "parent_in_top5",
            "count_child",
            "count_parent",
        ],
        data=rows,
    )
    # Log under diagnostics/ panel for proper organization in wandb
    wandb.log({"diagnostics/predictions_table": table}, step=step)


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function for Stage 1."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ========================================================================
    # SETUP OUTPUT DIRECTORY
    # ========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        # Default: results/<model-name>/<date>/
        encoder_name = args.encoder_model.replace("/", "_").replace(" ", "_")
        output_dir = Path("results") / "stage1-prototype" / f"{encoder_name}" / timestamp
    
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "checkpoints").mkdir(exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # ========================================================================
    # LOAD DATASET
    # ========================================================================
    logger.info("Loading dataset...")
    
    full_dataset = PanoramaCBMDataset(
        encoder_model=args.encoder_model,
        csv_path=args.csv_path,
        data_root=args.data_root,
    )
    
    # Get concept and parent names
    concept_names = list(full_dataset.concept_to_idx.keys())
    parent_names = list(full_dataset.parent_to_idx.keys())
    concept_to_idx = full_dataset.concept_to_idx
    parent_to_idx = full_dataset.parent_to_idx
    meta_to_parent = full_dataset.meta_to_parent
    
    logger.info(f"Dataset: {len(full_dataset)} samples")
    logger.info(f"Meta concepts: {len(concept_names)}")
    logger.info(f"Parent concepts: {len(parent_names)}")
    logger.info(f"Countries: {len(full_dataset.country_to_idx)}")
    
    # Split dataset (consistent with Stage 0)
    splits_path = None
    if args.splits_json and Path(args.splits_json).exists():
        # Load existing splits for consistency across stages
        logger.info(f"Loading splits from {args.splits_json}")
        splits_path = Path(args.splits_json)
        train_samples, val_samples, test_samples = load_splits_from_json(
            args.splits_json, full_dataset.samples
        )
    else:
        # Create new strict stratified splits: 70/15/15 (same as Stage 0)
        logger.info("Creating new strict stratified splits (70/15/15)")
        train_samples, val_samples, test_samples = create_splits_stratified_strict(
            full_dataset.samples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        # Save splits for reproducibility
        splits_path = output_dir / "splits.json"
        save_splits_to_json(
            train_samples, val_samples, test_samples, splits_path,
            extra_info={"seed": 42, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}
        )
    
    logger.info(f"Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    log_split_diagnostics(train_samples, val_samples, test_samples)
    
    # Compute class weights from training samples
    meta_weights = compute_class_weights(train_samples, concept_to_idx, device, key='meta_name')
    parent_weights = compute_parent_weights(train_samples, parent_to_idx, device)
    
    logger.info(f"Meta class weights: min={meta_weights.min():.3f}, max={meta_weights.max():.3f}")
    logger.info(f"Parent class weights: min={parent_weights.min():.3f}, max={parent_weights.max():.3f}")
    
    # ========================================================================
    # LOAD IMAGE ENCODER (FROM STAGE 0 CHECKPOINT)
    # ========================================================================
    logger.info("Loading image encoder...")
    
    config = StreetCLIPConfig(model_name=args.encoder_model)
    image_encoder = StreetCLIPEncoder(config).to(device)
    
    if args.resume_from_checkpoint:
        checkpoint_path = Path(args.resume_from_checkpoint)
        if checkpoint_path.exists():
            logger.info(f"Loading Stage 0 checkpoint from {checkpoint_path}")
            checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
            
            # Load only image encoder weights
            state_dict = checkpoint.get("model_state_dict", checkpoint)
            encoder_state = {k.replace("image_encoder.", ""): v for k, v in state_dict.items() if k.startswith("image_encoder.")}
            
            if encoder_state:
                image_encoder.load_state_dict(encoder_state, strict=False)
                logger.info(f"Loaded {len(encoder_state)} image encoder parameters")
            else:
                logger.warning("No image encoder weights found in checkpoint, using pretrained")
        else:
            logger.warning(f"Checkpoint not found: {checkpoint_path}, using pretrained")
    
    # Freeze image encoder
    image_encoder.eval()
    for p in image_encoder.parameters():
        p.requires_grad = False
    
    # ========================================================================
    # BUILD TEXT PROTOTYPES
    # ========================================================================
    logger.info("Building text prototypes...")

    # Get concept descriptions (notes) for prototype initialization
    _, concept_descriptions = extract_concepts_from_dataset(full_dataset)    # Build meta prototypes (using notes + templates)
    T_meta = build_text_prototypes(
        concept_names=concept_names,
        text_encoder=image_encoder,  # StreetCLIP text encoder
        concept_descriptions=concept_descriptions,
        templates=DEFAULT_CONCEPT_TEMPLATES,
        device=device,
    )
    logger.info(f"Built meta prototypes: {T_meta.shape}")
    
    # Build parent prototypes (using parent-specific templates)
    T_parent = build_text_prototypes(
        concept_names=parent_names,
        text_encoder=image_encoder,
        concept_descriptions=None,  # No descriptions for parent concepts
        templates=DEFAULT_PARENT_TEMPLATES,
        device=device,
    )
    logger.info(f"Built parent prototypes: {T_parent.shape}")
    
    # Build meta -> parent index mapping
    meta_to_parent_idx_tensor = build_meta_to_parent_idx(
        meta_to_parent=meta_to_parent,
        concept_to_idx=concept_to_idx,
        parent_to_idx=parent_to_idx,
    ).to(device)
    
    # ========================================================================
    # CREATE STAGE 1 MODEL
    # ========================================================================
    logger.info("Creating Stage1ConceptModel...")
    
    model = Stage1ConceptModel(
        image_encoder=image_encoder,
        T_meta=T_meta,
        T_parent=T_parent,
        meta_to_parent_idx=meta_to_parent_idx_tensor,
        streetclip_dim=768,
        concept_emb_dim=512,
        init_logit_scale=args.init_logit_scale,
        learnable_prototypes=args.learnable_prototypes,
        prototype_residual_scale=args.prototype_residual_scale,
        # Transformer bottleneck options
        use_transformer_bottleneck=args.use_transformer_bottleneck,
        transformer_num_heads=args.transformer_num_heads,
        transformer_num_layers=args.transformer_num_layers,
        transformer_dropout=args.transformer_dropout,
        transformer_stochastic_depth=args.transformer_stochastic_depth,
    ).to(device)
    
    trainable_params = model.get_trainable_params()
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {num_trainable:,}")
    if args.use_transformer_bottleneck:
        logger.info(f"Using TransformerBottleneck with {args.transformer_num_layers} layers, {args.transformer_num_heads} heads")
    
    # ========================================================================
    # BUILD SEMANTIC SIMILARITY MATRIX
    # ========================================================================
    logger.info("Building semantic similarity matrix for soft targets...")
    
    # Compute pairwise similarity between all meta concept prototypes
    with torch.no_grad():
        similarity_matrix = compute_concept_similarity_matrix(
            model.T_meta_base,  # Use base prototypes (frozen text embeddings)
            margin=args.semantic_margin,
            temperature=1.0,  # Raw similarity, temperature applied in loss
        ).to(device)
    
    logger.info(f"Similarity matrix shape: {similarity_matrix.shape}")
    logger.info(f"Similarity stats: min={similarity_matrix.min():.3f}, max={similarity_matrix.max():.3f}, mean={similarity_matrix.mean():.3f}")
    
    # Create semantic soft cross-entropy loss (if enabled)
    semantic_loss_fn = None
    if args.lambda_semantic > 0:
        semantic_loss_fn = SemanticSoftCrossEntropy(
            similarity_matrix=similarity_matrix,
            lambda_soft=args.lambda_semantic,
            temperature=args.semantic_temperature,
        )
        logger.info(f"Semantic soft CE enabled: λ={args.lambda_semantic}, temp={args.semantic_temperature}")
    
    # ========================================================================
    # SETUP DATALOADERS
    # ========================================================================
    from src.dataset import SubsetDataset
    
    train_dataset = SubsetDataset(full_dataset, train_samples)
    val_dataset = SubsetDataset(full_dataset, val_samples)
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # Precompute embeddings if requested
    use_precomputed = args.precompute_embeddings
    if use_precomputed:
        logger.info("Precomputing image embeddings...")
        
        # Check for cached embeddings
        train_cache_path = get_embedding_cache_path(
            args.resume_from_checkpoint, args.encoder_model, args.data_root, "train"
        )
        train_cache_dir = get_embedding_cache_dir(
            args.resume_from_checkpoint, args.encoder_model, args.data_root, "train"
        )
        val_cache_path = get_embedding_cache_path(
            args.resume_from_checkpoint, args.encoder_model, args.data_root, "val"
        )
        val_cache_dir = get_embedding_cache_dir(
            args.resume_from_checkpoint, args.encoder_model, args.data_root, "val"
        )
        
        # Try to load cached train embeddings
        train_embeddings = None
        train_metadata = None
        if args.use_embedding_cache:
            cached = load_cached_embeddings(train_cache_path)
            if cached is not None:
                train_embeddings, train_metadata = cached
                logger.info(f"Loaded train embeddings from bulk cache: {train_cache_path}")
            # NOTE: We do NOT use from_cache_dir for training - it's too slow (loads each embedding from disk)
            # Per-pano cache is only used for visualization (to get image_path for display)
        
        if train_embeddings is None:
            # Use larger batch size for precomputation
            precompute_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            train_embeddings, train_metadata = precompute_embeddings(model, precompute_loader, device)
            
            # Save to cache (both bulk .pt and per-pano for visualization)
            if args.use_embedding_cache:
                save_cached_embeddings(train_cache_path, train_embeddings, metadata=train_metadata)
                save_embeddings_per_pano(train_cache_dir, train_embeddings, train_metadata)
        
        train_loader = DataLoader(
            PrecomputedEmbeddingsDataset(*train_embeddings, metadata=train_metadata),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        
        # Try to load cached val embeddings
        val_embeddings = None
        val_metadata = None
        if args.use_embedding_cache:
            cached = load_cached_embeddings(val_cache_path)
            if cached is not None:
                val_embeddings, val_metadata = cached
                logger.info(f"Loaded val embeddings from bulk cache: {val_cache_path}")
            # NOTE: We do NOT use from_cache_dir for validation - it's too slow
        
        if val_embeddings is None:
            precompute_val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            val_embeddings, val_metadata = precompute_embeddings(model, precompute_val_loader, device)
            
            # Save to cache (both bulk .pt and per-pano for visualization)
            if args.use_embedding_cache:
                save_cached_embeddings(val_cache_path, val_embeddings, metadata=val_metadata)
                save_embeddings_per_pano(val_cache_dir, val_embeddings, val_metadata)
        
        val_loader = DataLoader(
            PrecomputedEmbeddingsDataset(*val_embeddings, metadata=val_metadata),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        
        num_train = len(train_loader.dataset) if train_loader is not None else (len(train_embeddings[0]) if train_embeddings is not None else 0)
        num_val = len(val_loader.dataset) if val_loader is not None else (len(val_embeddings[0]) if val_embeddings is not None else 0)
        logger.info(f"Precomputed {num_train} train and {num_val} val embeddings")
    
    # ========================================================================
    # SETUP LOSSES
    # ========================================================================
    focal_loss_meta = FocalLoss(
        gamma=args.focal_gamma,
        alpha=meta_weights if args.use_class_weights else None,
        label_smoothing=args.label_smoothing,
    )
    
    focal_loss_parent = FocalLoss(
        gamma=args.focal_gamma,
        alpha=parent_weights if args.use_class_weights else None,
        label_smoothing=args.label_smoothing,
    )
    
    # ========================================================================
    # SETUP OPTIMIZER AND SCHEDULER
    # ========================================================================
    optimizer = torch.optim.AdamW(
        trainable_params,
        lr=args.lr,
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.stage1_epochs,
        eta_min=args.lr * 0.01,
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)
    
    # ========================================================================
    # WANDB INIT
    # ========================================================================
    if args.use_wandb:
        wandb.init(
            project="geolocation-cbm-stage1",
            config=vars(args),
            name=f"stage1-prototype-{timestamp}",
        )
        # Define metrics for proper panel organization
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
        wandb.define_metric("batch/*", step_metric="epoch")
        wandb.define_metric("predictions/*", step_metric="epoch")
        wandb.define_metric("diagnostics/*", step_metric="epoch")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    logger.info(f"\n{'='*74}")
    logger.info(f"STAGE 1: Text-Prototype Based Concept Learning")
    logger.info(f"{'='*74}")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    # Temperature annealing setup
    if args.use_temp_annealing:
        logger.info(f"Using temperature annealing: logit_scale {args.init_logit_scale_low} → {args.init_logit_scale_high}")
        # Start with low logit scale (high temperature = soft predictions)
        with torch.no_grad():
            model.logit_scale_meta.fill_(args.init_logit_scale_low)
            model.logit_scale_parent.fill_(args.init_logit_scale_low)
    
    for epoch in range(args.stage1_epochs):
        model.train()
        
        # Temperature annealing: linearly increase logit scale
        if args.use_temp_annealing:
            progress = epoch / max(args.stage1_epochs - 1, 1)
            target_scale = args.init_logit_scale_low + progress * (args.init_logit_scale_high - args.init_logit_scale_low)
            with torch.no_grad():
                # Gently push towards target (allow some learning)
                model.logit_scale_meta.data = 0.9 * model.logit_scale_meta.data + 0.1 * target_scale
                model.logit_scale_parent.data = 0.9 * model.logit_scale_parent.data + 0.1 * target_scale
        
        total_loss = 0
        total_meta_correct = 0
        total_meta_correct_top5 = 0
        total_parent_correct = 0
        total_parent_correct_top5 = 0
        total_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage1_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if use_precomputed:
                if len(batch) == 7:
                    embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels, _ = batch
                else:
                    embeddings, concept_idx, parent_idx, country_idx, coords, cell_labels = batch
                embeddings = embeddings.to(device)
            else:
                images, concept_idx, parent_idx, country_idx, coords, _ = batch
                images = images.to(device)
            
            concept_idx = concept_idx.to(device)
            parent_idx = parent_idx.to(device)
            
            with torch.amp.autocast("cuda", enabled=args.use_amp):
                # Forward pass
                if use_precomputed:
                    outputs = model.forward_from_features(embeddings)
                else:
                    outputs = model(images)
                
                meta_logits = outputs["meta_logits"]
                parent_logits = outputs["parent_logits"]
                concept_emb = outputs["concept_emb"]
                
                # ============ COMPUTE LOSSES ============
                
                # 1. Standard classification losses
                loss_meta = focal_loss_meta(meta_logits, concept_idx)
                loss_parent = focal_loss_parent(parent_logits, parent_idx)
                
                # 2. NEW: Parent-guided meta loss (use parent info to guide meta prediction)
                if args.use_parent_guided_meta:
                    loss_meta_guided = parent_guided_meta_loss(
                        meta_logits=meta_logits,
                        parent_logits=parent_logits,
                        meta_labels=concept_idx,
                        parent_labels=parent_idx,
                        meta_to_parent_idx=model.meta_to_parent_idx,
                        hard_mask=False,
                        soft_temperature=args.parent_guide_temperature,
                    )
                    # Blend guided loss with standard loss
                    loss_meta = 0.5 * loss_meta + 0.5 * loss_meta_guided
                
                # 3. NEW: Hierarchical consistency loss
                if args.lambda_consistency > 0:
                    loss_consistency = hierarchical_consistency_loss(
                        meta_logits=meta_logits,
                        parent_logits=parent_logits,
                        meta_to_parent_idx=model.meta_to_parent_idx,
                        temperature=args.consistency_temperature,
                    )
                else:
                    loss_consistency = 0.0
                
                # 4. NEW: Inter-parent contrastive loss
                if args.lambda_parent_contrastive > 0:
                    loss_parent_contrastive = inter_parent_contrastive_loss(
                        embeddings=concept_emb,
                        parent_labels=parent_idx,
                        temperature=0.1,
                    )
                else:
                    loss_parent_contrastive = 0.0
                
                # 5. Prototype contrastive loss
                if args.lambda_contrastive > 0:
                    loss_contrastive = concept_prototype_contrastive_loss(
                        concept_emb,
                        model.T_meta,
                        concept_idx,
                        temperature=args.temperature,
                    )
                else:
                    loss_contrastive = 0.0
                
                # 6. Prototype regularization
                if args.lambda_reg > 0:
                    loss_reg = model.get_prototype_regularization_loss(args.lambda_reg)
                else:
                    loss_reg = 0.0
                
                # 7. NEW: Intra-parent prototype consistency
                if args.lambda_intra_parent > 0:
                    loss_intra = model.get_intra_parent_consistency_loss(args.lambda_intra_parent)
                else:
                    loss_intra = 0.0
                
                # 8. Semantic soft cross-entropy loss (anti-overfitting)
                if semantic_loss_fn is not None and args.lambda_semantic > 0:
                    loss_semantic = semantic_loss_fn(meta_logits, concept_idx)
                else:
                    loss_semantic = 0.0
                
                # ============ TOTAL LOSS ============
                loss = (
                    args.lambda_meta * loss_meta +
                    args.lambda_parent * loss_parent +
                    args.lambda_consistency * loss_consistency +
                    args.lambda_parent_contrastive * loss_parent_contrastive +
                    args.lambda_contrastive * loss_contrastive +
                    loss_reg +
                    loss_intra +
                    loss_semantic  # Semantic loss already weighted in SemanticSoftCrossEntropy
                )
                
                loss = loss / args.gradient_accumulation_steps
            
            scaler.scale(loss).backward()
            
            if (batch_idx + 1) % args.gradient_accumulation_steps == 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(trainable_params, args.max_grad_norm)
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad()
            
            # Track metrics
            total_loss += loss.item() * args.gradient_accumulation_steps * len(concept_idx)
            
            pred_meta = meta_logits.argmax(dim=1)
            pred_parent = parent_logits.argmax(dim=1)
            total_meta_correct += (pred_meta == concept_idx).sum().item()
            total_parent_correct += (pred_parent == parent_idx).sum().item()
            
            # Top-5 Accuracy
            _, pred_meta_top5 = meta_logits.topk(5, dim=1, largest=True, sorted=True)
            total_meta_correct_top5 += (pred_meta_top5 == concept_idx.view(-1, 1)).sum().item()
            
            _, pred_parent_top5 = parent_logits.topk(5, dim=1, largest=True, sorted=True)
            total_parent_correct_top5 += (pred_parent_top5 == parent_idx.view(-1, 1)).sum().item()
            
            total_count += len(concept_idx)
            
            # Compute semantic-close accuracy (for logging)
            if similarity_matrix is not None:
                hard_acc_batch, semantic_close_acc_batch = compute_semantic_close_accuracy(
                    pred_meta, concept_idx, similarity_matrix, threshold=args.semantic_close_threshold
                )
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "meta_acc": f"{total_meta_correct/total_count:.3f}",
                "parent_acc": f"{total_parent_correct/total_count:.3f}",
            })
            
            # Log to wandb (use commit=False for batch logs to avoid step issues)
            if args.use_wandb and batch_idx % 50 == 0:
                wandb.log({
                    "batch/loss": loss.item(),
                    "batch/meta_acc": (pred_meta == concept_idx).float().mean().item(),
                    "batch/parent_acc": (pred_parent == parent_idx).float().mean().item(),
                    "train/logit_scale_meta": model.logit_scale_meta.item(),
                    "train/logit_scale_parent": model.logit_scale_parent.item(),
                }, commit=False)
        
        # Epoch metrics
        train_metrics = {
            "loss": total_loss / total_count,
            "meta_acc": total_meta_correct / total_count,
            "meta_acc_top5": total_meta_correct_top5 / total_count,
            "parent_acc": total_parent_correct / total_count,
            "parent_acc_top5": total_parent_correct_top5 / total_count,
        }
        log_metrics(train_metrics, prefix="Train", stage=1)
        
        # Validation
        val_metrics = validate(
            model, val_loader, device,
            focal_loss_meta, focal_loss_parent, args,
            use_precomputed=use_precomputed,
            similarity_matrix=similarity_matrix,
        )
        log_metrics(val_metrics, prefix="Val", stage=1)
        
        # Log semantic-close accuracy if available
        if "semantic_close_acc" in val_metrics:
            logger.info(f"  Semantic-close Acc (threshold={args.semantic_close_threshold}): {val_metrics['semantic_close_acc']:.3f}")
        
        if args.use_wandb:
            log_validation_predictions_table(
                model=model,
                dataloader=val_loader,
                device=device,
                idx_to_concept=full_dataset.idx_to_concept,
                idx_to_parent=full_dataset.idx_to_parent,
                use_precomputed=use_precomputed,
                max_rows=WAND_TABLE_MAX_ROWS,
                step=epoch + 1,
                output_dir=output_dir,
            )
    
        # Visualization
        if args.viz_interval > 0 and (epoch + 1) % args.viz_interval == 0:
            visualize_concept_predictions(
                model=model,
                image_dataloader=val_loader,
                device=device,
                idx_to_concept=full_dataset.idx_to_concept,
                idx_to_parent=full_dataset.idx_to_parent,
                output_dir=output_dir,
                epoch=epoch + 1,
                num_samples=VIZ_NUM_SAMPLES,
                log_to_wandb=args.use_wandb,
                wandb_step=epoch + 1,
                use_precomputed=use_precomputed,
            )
        
        scheduler.step()
    
        # Wandb logging - use consistent key prefixes for panel organization
        if args.use_wandb:
            log_dict = {
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/meta_acc": train_metrics["meta_acc"],
                "train/meta_acc_top5": train_metrics["meta_acc_top5"],
                "train/parent_acc": train_metrics["parent_acc"],
                "train/parent_acc_top5": train_metrics["parent_acc_top5"],
                "val/loss": val_metrics["loss"],
                "val/meta_acc": val_metrics["meta_acc"],
                "val/meta_acc_top5": val_metrics["meta_acc_top5"],
                "val/meta_recall": val_metrics["meta_recall"],
                "val/parent_acc": val_metrics["parent_acc"],
                "val/parent_acc_top5": val_metrics["parent_acc_top5"],
                "val/parent_recall": val_metrics["parent_recall"],
                "train/lr": scheduler.get_last_lr()[0],
            }
            # Add semantic-close accuracy if available
            if "semantic_close_acc" in val_metrics:
                log_dict["val/semantic_close_acc"] = val_metrics["semantic_close_acc"]
            wandb.log(log_dict, step=epoch + 1)
    
        # Checkpointing
        val_metric = val_metrics["meta_acc"]
        if val_metric > best_val_acc:
            best_val_acc = val_metric
            patience_counter = 0
            
            save_checkpoint(
                model=model,
                checkpoint_path=output_dir / "checkpoints" / "best_model_stage1.pt",
                concept_names=concept_names,
                parent_names=parent_names,
                concept_to_idx=concept_to_idx,
                parent_to_idx=parent_to_idx,
                country_to_idx=full_dataset.country_to_idx,
                encoder_model=args.encoder_model,
                extra_info={
                    "stage": 1,
                    "meta_acc": val_metric,
                    "parent_acc": val_metrics["parent_acc"],
                    "stage0_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
                    "splits_json": str(splits_path) if splits_path is not None else None,
                },
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
            logger.info(f"✓ New best model! Meta Acc: {val_metric:.4f}")
        else:
            patience_counter += 1
            if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
                logger.info(f"Early stopping at epoch {epoch+1}")
                logger.info(f"\n{'='*74}")
                logger.info(f"Training complete! Best Meta Acc: {best_val_acc:.4f}")
                logger.info(f"{'='*74}")
                if args.use_wandb:
                    wandb.finish()
                return
    
        # Periodic checkpoint
        if (epoch + 1) % args.save_interval == 0:
            save_checkpoint(
                model=model,
                checkpoint_path=output_dir / "checkpoints" / f"checkpoint_epoch_{epoch+1}.pt",
                concept_names=concept_names,
                parent_names=parent_names,
                concept_to_idx=concept_to_idx,
                parent_to_idx=parent_to_idx,
                country_to_idx=full_dataset.country_to_idx,
                encoder_model=args.encoder_model,
                extra_info={
                    "stage": 1,
                    "meta_acc": val_metric,
                    "stage0_checkpoint": str(args.resume_from_checkpoint) if args.resume_from_checkpoint else None,
                    "splits_json": str(splits_path) if splits_path is not None else None,
                },
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )

    logger.info(f"\n{'='*74}")
    logger.info(f"Training complete! Best Meta Acc: {best_val_acc:.4f}")
    logger.info(f"{'='*74}")
    
    # ========================================================================
    # POST-TRAINING: Save concept embeddings for Stage 2
    # ========================================================================
    if args.use_embedding_cache and args.precompute_embeddings:
        logger.info("\nComputing and saving concept embeddings for Stage 2...")
        
        # Load best model
        best_ckpt_path = output_dir / "checkpoints" / "best_model_stage1.pt"
        if best_ckpt_path.exists():
            best_ckpt = torch.load(best_ckpt_path, map_location=device, weights_only=False)
            model.load_state_dict(best_ckpt["model_state_dict"])
            logger.info(f"Loaded best model from {best_ckpt_path}")
        
        model.eval()
        
        # Create dataloaders WITHOUT drop_last to process all samples
        # Compute concept embeddings for train set
        if train_embeddings is not None:
            train_concept_loader = DataLoader(
                PrecomputedEmbeddingsDataset(*train_embeddings, metadata=train_metadata),
                batch_size=args.batch_size,
                shuffle=False,  # Keep order to match manifest
                num_workers=2,
                pin_memory=True,
                drop_last=False,  # IMPORTANT: process all samples
            )
            train_concept_embs = compute_concept_embeddings(model, train_concept_loader, device)
            train_cache_dir = get_embedding_cache_dir(
                args.resume_from_checkpoint, args.encoder_model, args.data_root, "train"
            )
            save_concept_embeddings_to_cache(train_cache_dir, train_concept_embs)
        
        # Compute concept embeddings for val set
        if val_embeddings is not None:
            val_concept_loader = DataLoader(
                PrecomputedEmbeddingsDataset(*val_embeddings, metadata=val_metadata),
                batch_size=args.batch_size,
                shuffle=False,  # Keep order to match manifest
                num_workers=2,
                pin_memory=True,
                drop_last=False,  # IMPORTANT: process all samples
            )
            val_concept_embs = compute_concept_embeddings(model, val_concept_loader, device)
            val_cache_dir = get_embedding_cache_dir(
                args.resume_from_checkpoint, args.encoder_model, args.data_root, "val"
            )
            save_concept_embeddings_to_cache(val_cache_dir, val_concept_embs)
        
        logger.info("Concept embeddings saved successfully!")
    
    if args.use_wandb:
        wandb.finish()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Text-Prototype Based Concept Learning")
    
    # Dataset
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--splits_json", type=str, default=None, help="Path to existing splits.json (if provided, loads splits instead of creating new)")

    # Model
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Stage 0 checkpoint to resume from")
    
    # Prototype settings
    parser.add_argument("--learnable_prototypes", action="store_true", default=True)
    parser.add_argument("--no_learnable_prototypes", dest="learnable_prototypes", action="store_false")
    parser.add_argument("--prototype_residual_scale", type=float, default=0.01)
    parser.add_argument("--init_logit_scale", type=float, default=14.0, help="Initial logit scale (~1/temperature)")
    
    # Transformer bottleneck settings (anti-overfitting)
    parser.add_argument("--use_transformer_bottleneck", action="store_true", default=False,
                        help="Use transformer encoder bottleneck instead of MLP (optional, start with False)")
    parser.add_argument("--transformer_num_heads", type=int, default=8, help="Transformer attention heads")
    parser.add_argument("--transformer_num_layers", type=int, default=2, help="Transformer encoder layers")
    parser.add_argument("--transformer_dropout", type=float, default=0.4, help="Dropout in transformer")
    parser.add_argument("--transformer_stochastic_depth", type=float, default=0.2, help="Stochastic depth probability")
    
    # Training
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.05, help="Weight decay (increased from 0.01 for anti-overfitting)")
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--precompute_embeddings", action="store_true", default=True)
    parser.add_argument("--use_embedding_cache", action="store_true", default=True,
                        help="Cache precomputed embeddings to disk for reuse")
    parser.add_argument("--no_embedding_cache", dest="use_embedding_cache", action="store_false",
                        help="Disable embedding caching (recompute every time)")
    
    # Data augmentation
    parser.add_argument("--augmentation_strength", type=str, default="medium",
                        choices=["none", "light", "medium", "strong"],
                        help="Strength of data augmentation for training")

    # Loss weights
    parser.add_argument("--lambda_meta", type=float, default=1.0)
    parser.add_argument("--lambda_parent", type=float, default=0.5, help="Weight for parent classification loss (increased from 0.3)")
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--lambda_reg", type=float, default=0.001, help="Prototype regularization weight")
    parser.add_argument("--temperature", type=float, default=0.07)
    
    # Semantic similarity loss (anti-overfitting)
    parser.add_argument("--lambda_semantic", type=float, default=0.15,
                        help="Weight for semantic soft cross-entropy loss (0 = disabled)")
    parser.add_argument("--semantic_temperature", type=float, default=2.0,
                        help="Temperature for soft targets in semantic loss")
    parser.add_argument("--semantic_margin", type=float, default=0.0,
                        help="Minimum similarity threshold for semantic soft targets")
    parser.add_argument("--semantic_close_threshold", type=float, default=0.7,
                        help="Similarity threshold for semantic-close accuracy metric")
    
    # NEW: Hierarchical losses
    parser.add_argument("--lambda_consistency", type=float, default=0.3, help="Weight for hierarchical consistency loss")
    parser.add_argument("--lambda_parent_contrastive", type=float, default=0.2, help="Weight for inter-parent contrastive loss")
    parser.add_argument("--lambda_intra_parent", type=float, default=0.01, help="Weight for intra-parent prototype consistency")
    parser.add_argument("--use_parent_guided_meta", action="store_true", default=True, help="Use parent-guided meta loss")
    parser.add_argument("--no_parent_guided_meta", dest="use_parent_guided_meta", action="store_false")
    parser.add_argument("--parent_guide_temperature", type=float, default=2.0, help="Temperature for soft parent gating")
    parser.add_argument("--consistency_temperature", type=float, default=1.0, help="Temperature for consistency loss")
    
    # NEW: Temperature annealing
    parser.add_argument("--use_temp_annealing", action="store_true", default=True, help="Anneal logit scale from low to high")
    parser.add_argument("--no_temp_annealing", dest="use_temp_annealing", action="store_false")
    parser.add_argument("--init_logit_scale_low", type=float, default=4.0, help="Starting logit scale for annealing")
    parser.add_argument("--init_logit_scale_high", type=float, default=14.0, help="Ending logit scale for annealing")
    
    # Focal loss
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.2, help="Label smoothing (increased from 0.1 for anti-overfitting)")
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    
    # Misc
    parser.add_argument("--viz_interval", type=int, default=5, help="Visualize predictions every N epochs (0 to disable)")
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    
    args = parser.parse_args()
    train(args)
