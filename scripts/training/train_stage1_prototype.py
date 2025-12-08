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

try:
    import wandb
    HAS_WANDB = True
except ImportError:
    HAS_WANDB = False

from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified,
    get_transforms_from_processor,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import (
    Stage1ConceptModel,
    build_text_prototypes,
    build_meta_to_parent_idx,
    DEFAULT_CONCEPT_TEMPLATES,
    DEFAULT_PARENT_TEMPLATES,
)
from src.losses import FocalLoss, concept_prototype_contrastive_loss
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
        embeddings: torch.Tensor,
        concept_indices: torch.Tensor,
        parent_indices: torch.Tensor,
        country_indices: torch.Tensor,
        coordinates: torch.Tensor,
        cell_labels: Optional[torch.Tensor] = None,
    ):
        self.embeddings = embeddings
        self.concept_indices = concept_indices
        self.parent_indices = parent_indices
        self.country_indices = country_indices
        self.coordinates = coordinates
        self.cell_labels = cell_labels if cell_labels is not None else torch.zeros(len(embeddings), dtype=torch.long)
    
    def __len__(self):
        return len(self.embeddings)
    
    def __getitem__(self, idx):
        return (
            self.embeddings[idx],
            self.concept_indices[idx],
            self.parent_indices[idx],
            self.country_indices[idx],
            self.coordinates[idx],
            self.cell_labels[idx],
        )


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

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


def save_cached_embeddings(
    cache_path: Path,
    embeddings: Tuple[torch.Tensor, ...],
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
    
    torch.save(cache_data, cache_path)
    logger.info(f"Saved cached embeddings to {cache_path}")


def load_cached_embeddings(
    cache_path: Path,
) -> Optional[Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]]:
    """Load precomputed embeddings from cache if available."""
    if not cache_path.exists():
        return None
    
    try:
        cache_data = torch.load(cache_path, weights_only=True)
        logger.info(f"Loaded cached embeddings from {cache_path}")
        return (
            cache_data['embeddings'],
            cache_data['concept_idx'],
            cache_data['parent_idx'],
            cache_data['country_idx'],
            cache_data['coords'],
            cache_data['cell_labels'],
        )
    except Exception as e:
        logger.warning(f"Failed to load cached embeddings: {e}")
        return None


def precompute_embeddings(
    model: nn.Module,
    dataloader: DataLoader,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Precompute image embeddings using frozen image encoder."""
    model.eval()
    
    all_embeddings = []
    all_concept_idx = []
    all_parent_idx = []
    all_country_idx = []
    all_coords = []
    all_cell_labels = []
    
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
    
    return (
        torch.cat(all_embeddings, dim=0),
        torch.cat(all_concept_idx, dim=0),
        torch.cat(all_parent_idx, dim=0),
        torch.cat(all_country_idx, dim=0),
        torch.cat(all_coords, dim=0),
        torch.cat(all_cell_labels, dim=0),
    )


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
    if "parent_acc" in metrics:
        parts.append(f"Parent Acc: {metrics['parent_acc']:.3f}")
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
) -> Dict[str, float]:
    """Validate Stage 1 model."""
    model.eval()
    
    total_loss = 0
    total_meta_correct = 0
    total_parent_correct = 0
    total_count = 0
    
    for batch in dataloader:
        if use_precomputed:
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
        total_count += len(concept_idx)
    
    return {
        "loss": total_loss / total_count,
        "meta_acc": total_meta_correct / total_count,
        "parent_acc": total_parent_correct / total_count,
    }


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
        output_dir = Path("results") / "stage1-prototype" / timestamp
    
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
    
    # Split dataset
    train_samples, val_samples, test_samples = create_splits_stratified(
        full_dataset.samples,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1,
        seed=42,
    )
    logger.info(f"Splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    
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
            checkpoint = torch.load(checkpoint_path, map_location=device)
            
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
    ).to(device)
    
    trainable_params = model.get_trainable_params()
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Trainable parameters: {num_trainable:,}")
    
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
        val_cache_path = get_embedding_cache_path(
            args.resume_from_checkpoint, args.encoder_model, args.data_root, "val"
        )
        
        # Try to load cached train embeddings
        train_embeddings = None
        if args.use_embedding_cache:
            train_embeddings = load_cached_embeddings(train_cache_path)
        
        if train_embeddings is None:
            # Use larger batch size for precomputation
            precompute_loader = DataLoader(
                train_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            train_embeddings = precompute_embeddings(model, precompute_loader, device)
            
            # Save to cache
            if args.use_embedding_cache:
                save_cached_embeddings(train_cache_path, train_embeddings)
        
        train_loader = DataLoader(
            PrecomputedEmbeddingsDataset(*train_embeddings),
            batch_size=args.batch_size,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
            drop_last=True,
        )
        
        # Try to load cached val embeddings
        val_embeddings = None
        if args.use_embedding_cache:
            val_embeddings = load_cached_embeddings(val_cache_path)
        
        if val_embeddings is None:
            precompute_val_loader = DataLoader(
                val_dataset,
                batch_size=args.batch_size * 2,
                shuffle=False,
                num_workers=4,
                pin_memory=True,
            )
            val_embeddings = precompute_embeddings(model, precompute_val_loader, device)
            
            # Save to cache
            if args.use_embedding_cache:
                save_cached_embeddings(val_cache_path, val_embeddings)
        
        val_loader = DataLoader(
            PrecomputedEmbeddingsDataset(*val_embeddings),
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )
        
        logger.info(f"Precomputed {len(train_embeddings[0])} train and {len(val_embeddings[0])} val embeddings")
    
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
    if args.use_wandb and HAS_WANDB:
        wandb.init(
            project="geolocation-cbm-stage1",
            config=vars(args),
            name=f"stage1-prototype-{timestamp}",
        )
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    logger.info(f"\n{'='*74}")
    logger.info(f"STAGE 1: Text-Prototype Based Concept Learning")
    logger.info(f"{'='*74}")
    
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(args.stage1_epochs):
        model.train()
        
        total_loss = 0
        total_meta_correct = 0
        total_parent_correct = 0
        total_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage1_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            if use_precomputed:
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
                
                # Compute losses
                loss_meta = focal_loss_meta(meta_logits, concept_idx)
                loss_parent = focal_loss_parent(parent_logits, parent_idx)
                
                # Optional: contrastive loss
                if args.lambda_contrastive > 0:
                    loss_contrastive = concept_prototype_contrastive_loss(
                        concept_emb,
                        model.T_meta,
                        concept_idx,
                        temperature=args.temperature,
                    )
                else:
                    loss_contrastive = 0.0
                
                # Optional: prototype regularization
                if args.lambda_reg > 0:
                    loss_reg = model.get_prototype_regularization_loss(args.lambda_reg)
                else:
                    loss_reg = 0.0
                
                # Total loss
                loss = (
                    args.lambda_meta * loss_meta +
                    args.lambda_parent * loss_parent +
                    args.lambda_contrastive * loss_contrastive +
                    loss_reg
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
            total_count += len(concept_idx)
            
            # Update progress bar
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "meta_acc": f"{total_meta_correct/total_count:.3f}",
                "parent_acc": f"{total_parent_correct/total_count:.3f}",
            })
            
            # Log to wandb
            if args.use_wandb and HAS_WANDB and batch_idx % 50 == 0:
                wandb.log({
                    "batch_loss": loss.item(),
                    "batch_meta_acc": (pred_meta == concept_idx).float().mean().item(),
                    "batch_parent_acc": (pred_parent == parent_idx).float().mean().item(),
                    "logit_scale_meta": model.logit_scale_meta.item(),
                    "logit_scale_parent": model.logit_scale_parent.item(),
                })
        
        # Epoch metrics
        train_metrics = {
            "loss": total_loss / total_count,
            "meta_acc": total_meta_correct / total_count,
            "parent_acc": total_parent_correct / total_count,
        }
        log_metrics(train_metrics, prefix="Train", stage=1)
        
        # Validation
        val_metrics = validate(
            model, val_loader, device,
            focal_loss_meta, focal_loss_parent, args,
            use_precomputed=use_precomputed,
        )
        log_metrics(val_metrics, prefix="Val", stage=1)
        
        scheduler.step()
        
        # Wandb logging
        if args.use_wandb and HAS_WANDB:
            wandb.log({
                "epoch": epoch + 1,
                "train_loss": train_metrics["loss"],
                "train_meta_acc": train_metrics["meta_acc"],
                "train_parent_acc": train_metrics["parent_acc"],
                "val_loss": val_metrics["loss"],
                "val_meta_acc": val_metrics["meta_acc"],
                "val_parent_acc": val_metrics["parent_acc"],
                "lr": scheduler.get_last_lr()[0],
            })
        
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
                break
        
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
                extra_info={"stage": 1, "meta_acc": val_metric},
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
    
    logger.info(f"\n{'='*74}")
    logger.info(f"Training complete! Best Meta Acc: {best_val_acc:.4f}")
    logger.info(f"{'='*74}")
    
    if args.use_wandb and HAS_WANDB:
        wandb.finish()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 1: Text-Prototype Based Concept Learning")
    
    # Dataset
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--data_root", type=str, default="data")
    
    # Model
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--resume_from_checkpoint", type=str, default=None, help="Stage 0 checkpoint to resume from")
    
    # Prototype settings
    parser.add_argument("--learnable_prototypes", action="store_true", default=True)
    parser.add_argument("--no_learnable_prototypes", dest="learnable_prototypes", action="store_false")
    parser.add_argument("--prototype_residual_scale", type=float, default=0.01)
    parser.add_argument("--init_logit_scale", type=float, default=14.0, help="Initial logit scale (~1/temperature)")
    
    # Training
    parser.add_argument("--stage1_epochs", type=int, default=50)
    parser.add_argument("--batch_size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    parser.add_argument("--precompute_embeddings", action="store_true", default=True)
    parser.add_argument("--use_embedding_cache", action="store_true", default=True,
                        help="Cache precomputed embeddings to disk for reuse")
    parser.add_argument("--no_embedding_cache", dest="use_embedding_cache", action="store_false",
                        help="Disable embedding caching (recompute every time)")

    # Loss weights
    parser.add_argument("--lambda_meta", type=float, default=1.0)
    parser.add_argument("--lambda_parent", type=float, default=0.3)
    parser.add_argument("--lambda_contrastive", type=float, default=0.5)
    parser.add_argument("--lambda_reg", type=float, default=0.001, help="Prototype regularization weight")
    parser.add_argument("--temperature", type=float, default=0.07)
    
    # Focal loss
    parser.add_argument("--focal_gamma", type=float, default=2.0)
    parser.add_argument("--label_smoothing", type=float, default=0.1)
    parser.add_argument("--use_class_weights", action="store_true", default=True)
    parser.add_argument("--no_class_weights", dest="use_class_weights", action="store_false")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=10)
    parser.add_argument("--early_stopping_patience", type=int, default=10)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    
    args = parser.parse_args()
    train(args)
