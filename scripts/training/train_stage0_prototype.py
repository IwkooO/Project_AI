#!/usr/bin/env python3
"""
Stage 0 Training Script: Domain Contrastive Pretraining

This script implements Stage 0 pretraining with hierarchical concept alignment:
- Image-GPS contrastive alignment
- Image-Child Concept contrastive alignment
- Image-Parent Concept contrastive alignment
- Child-Parent consistency (intra-batch supervised contrastive)

Data Strategy:
- Uses TRAIN SPLIT ONLY (val/test never seen by encoder)
- Internal 90/10 split of train for pretrain_train/pretrain_val

Usage:
    python scripts/training/train_stage0_prototype.py \
        --csv_path data/dataset-43k-mapped.csv \
        --stage0_epochs 20 \
        --unfreeze_layers 2 \
        --use_wandb
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional
from datetime import datetime
from collections import Counter

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import CosineAnnealingLR
from tqdm import tqdm
import numpy as np
import wandb
from geoclip import LocationEncoder

from src.dataset import (
    PanoramaCBMDataset,
    create_splits_stratified_strict,
    save_splits_to_json,
    load_splits_from_json,
    log_split_diagnostics,
)
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import (
    build_text_prototypes,
    build_meta_to_parent_idx,
    DEFAULT_CONCEPT_TEMPLATES,
    DEFAULT_PARENT_TEMPLATES,
)
from src.losses import (
    clip_contrastive_loss,
    concept_prototype_contrastive_loss,
    inter_parent_contrastive_loss,
)
from src.concepts.utils import extract_concepts_from_dataset

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# STAGE 0 MODEL
# ============================================================================

class Stage0PretrainingModel(nn.Module):
    """
    Stage 0 Pretraining Model for domain contrastive learning.
    
    Trains:
    - Top N layers of StreetCLIP vision encoder
    - StreetCLIP text encoder (for prototype refinement)
    - Concept bottleneck projection (768d -> 512d)
    - Location encoder (GeoCLIP)
    
    Learns alignment between:
    - Images and GPS coordinates
    - Images and child (meta) concept prototypes
    - Images and parent concept prototypes
    - Intra-batch hierarchical consistency
    """
    
    def __init__(
        self,
        image_encoder: StreetCLIPEncoder,
        T_meta: torch.Tensor,
        T_parent: torch.Tensor,
        meta_to_parent_idx: torch.Tensor,
        streetclip_dim: int = 768,
        concept_emb_dim: int = 512,
        patch_dim: int = 1024,
    ):
        super().__init__()
        
        self.streetclip_dim = streetclip_dim
        self.concept_emb_dim = concept_emb_dim
        self.patch_dim = patch_dim
        self.num_metas = T_meta.shape[0]
        self.num_parents = T_parent.shape[0]
        
        # Image encoder (will be partially unfrozen)
        self.image_encoder = image_encoder
        
        # Location encoder (GeoCLIP)
        self.location_encoder = LocationEncoder()

        # GPS adapter: project GeoCLIP 512d location features up to StreetCLIP space (768d)
        # so that image↔GPS alignment can happen without compressing the image features.
        self.gps_adapter = nn.Sequential(
            nn.Linear(concept_emb_dim, streetclip_dim),
            nn.LayerNorm(streetclip_dim),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(streetclip_dim, streetclip_dim),
            nn.LayerNorm(streetclip_dim),
        )
        
        # Text prototypes (frozen base, used for contrastive targets)
        self.register_buffer("T_meta", T_meta)  # [num_metas, 768]
        self.register_buffer("T_parent", T_parent)  # [num_parents, 768]
        self.register_buffer("meta_to_parent_idx", meta_to_parent_idx)
        
        # Concept bottleneck: 768d -> 512d
        self.concept_bottleneck = nn.Sequential(
            nn.Linear(streetclip_dim, 1024),
            nn.LayerNorm(1024),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(1024, concept_emb_dim),
            nn.LayerNorm(concept_emb_dim),
        )
        
        # Prototype projection: project text prototypes to 512d for comparison
        self.prototype_projection = nn.Linear(streetclip_dim, concept_emb_dim, bias=False)

        # Patch projection for patch-level objectives (Stage2-aligned)
        self.patch_projection = nn.Linear(patch_dim, concept_emb_dim, bias=False)
        
        self._init_weights()
    
    def _init_weights(self):
        for layer in self.concept_bottleneck:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
        nn.init.xavier_uniform_(self.prototype_projection.weight)
        nn.init.xavier_uniform_(self.patch_projection.weight)
        for layer in self.gps_adapter:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                if layer.bias is not None:
                    nn.init.zeros_(layer.bias)
    
    @property
    def T_meta_projected(self) -> torch.Tensor:
        """Get meta prototypes projected to concept embedding space."""
        return F.normalize(self.prototype_projection(self.T_meta), p=2, dim=1)
    
    @property
    def T_parent_projected(self) -> torch.Tensor:
        """Get parent prototypes projected to concept embedding space."""
        return F.normalize(self.prototype_projection(self.T_parent), p=2, dim=1)
    
    def forward(
        self,
        images: torch.Tensor,
        coords: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        """
        Forward pass for Stage 0 pretraining.
        
        Args:
            images: Image tensor [batch, 3, H, W]
            coords: GPS coordinates [batch, 2] (lat, lng)
            
        Returns:
            Dict with embeddings for contrastive losses
        """
        # Image features + patch tokens (encoder may be partially unfrozen)
        img_features, patch_tokens = self.image_encoder.get_features_and_patches(images)  # [B, 768], [B, 576, 1024]
        img_features_norm = F.normalize(img_features, p=2, dim=1)
        
        # Concept embedding (bottleneck output)
        concept_emb = self.concept_bottleneck(img_features)  # [batch, 512]
        concept_emb_norm = F.normalize(concept_emb, p=2, dim=1)
        
        # GPS embedding
        gps_emb = self.location_encoder(coords)  # [batch, 512]
        gps_emb_768 = self.gps_adapter(gps_emb)  # [batch, 768]
        gps_emb_768_norm = F.normalize(gps_emb_768, p=2, dim=1)

        # Patch embedding (pooled) for patch-level objectives
        patch_emb = self.patch_projection(patch_tokens)  # [B, 576, 512]
        patch_emb_pooled = patch_emb.mean(dim=1)  # [B, 512]
        patch_emb_norm = F.normalize(patch_emb_pooled, p=2, dim=1)
        
        return {
            "img_features": img_features,
            "img_features_norm": img_features_norm,
            "patch_tokens": patch_tokens,
            "concept_emb": concept_emb,
            "concept_emb_norm": concept_emb_norm,
            "gps_emb": gps_emb,
            "gps_emb_768": gps_emb_768,
            "gps_emb_768_norm": gps_emb_768_norm,
            "patch_emb_norm": patch_emb_norm,
        }
    
    def get_trainable_params(self) -> List[nn.Parameter]:
        """Return all trainable parameters for Stage 0."""
        params = []
        # Image encoder trainable params (top layers + projection)
        params.extend(self.image_encoder.get_trainable_params())
        # Concept bottleneck
        params.extend(self.concept_bottleneck.parameters())
        # Prototype projection
        params.extend(self.prototype_projection.parameters())
        # Patch projection
        params.extend(self.patch_projection.parameters())
        # GPS adapter
        params.extend(self.gps_adapter.parameters())
        # Location encoder
        params.extend(self.location_encoder.parameters())
        return params


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def save_checkpoint(
    model: Stage0PretrainingModel,
    checkpoint_path: Path,
    concept_names: List[str],
    parent_names: List[str],
    concept_to_idx: Dict[str, int],
    parent_to_idx: Dict[str, int],
    country_to_idx: Dict[str, int],
    meta_to_parent: Dict[str, str],
    encoder_model: str,
    extra_info: Optional[Dict] = None,
    optimizer=None,
    scheduler=None,
    epoch: Optional[int] = None,
):
    """Save Stage 0 checkpoint with all metadata for Stage 1 loading."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "concept_names": concept_names,
        "parent_names": parent_names,
        "num_concepts": len(concept_names),
        "num_parents": len(parent_names),
        "concept_to_idx": concept_to_idx,
        "parent_to_idx": parent_to_idx,
        "country_to_idx": country_to_idx,
        "meta_to_parent": meta_to_parent,
        "encoder_model": encoder_model,
        "T_meta": model.T_meta.cpu(),
        "T_parent": model.T_parent.cpu(),
        "meta_to_parent_idx": model.meta_to_parent_idx.cpu(),
        "stage": 0,
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
    
    for key, value in metrics.items():
        if isinstance(value, float):
            parts.append(f"{key}: {value:.4f}")
    
    if parts:
        logger.info(f"{stage_str}{prefix}: {' | '.join(parts)}")


# ============================================================================
# VALIDATION
# ============================================================================

@torch.no_grad()
def validate(
    model: Stage0PretrainingModel,
    dataloader: DataLoader,
    device: torch.device,
    args,
    anchor_encoder: Optional[StreetCLIPEncoder] = None,
) -> Dict[str, float]:
    """Validate Stage 0 model on pretrain_val split."""
    model.eval()
    
    total_loss = 0
    total_loss_gps = 0
    total_loss_child = 0
    total_loss_parent = 0
    total_loss_hierarchy = 0
    total_loss_patch_gps = 0
    total_loss_anchor = 0
    total_loss_patch_anchor = 0
    total_count = 0
    
    for batch in dataloader:
        images, concept_idx, parent_idx, country_idx, coords, metadata = batch
        images = images.to(device)
        coords = coords.to(device)
        concept_idx = concept_idx.to(device)
        parent_idx = parent_idx.to(device)
        
        outputs = model(images, coords)
        concept_emb_norm = outputs["concept_emb_norm"]
        patch_emb_norm = outputs["patch_emb_norm"]
        gps_emb = outputs["gps_emb"]
        img_features_norm = outputs["img_features_norm"]
        gps_emb_768_norm = outputs["gps_emb_768_norm"]
        img_features = outputs["img_features"]
        patch_tokens = outputs["patch_tokens"]
        
        # Compute losses
        loss_gps = clip_contrastive_loss(img_features_norm, gps_emb_768_norm, args.temperature)
        loss_child = concept_prototype_contrastive_loss(
            concept_emb_norm, model.T_meta_projected, concept_idx, args.temperature
        )
        loss_parent = concept_prototype_contrastive_loss(
            concept_emb_norm, model.T_parent_projected, parent_idx, args.temperature
        )
        loss_hierarchy = inter_parent_contrastive_loss(
            concept_emb_norm, parent_idx, args.temperature
        )

        loss_patch_gps = clip_contrastive_loss(patch_emb_norm, gps_emb, args.temperature)

        loss_anchor = torch.tensor(0.0, device=device)
        loss_patch_anchor = torch.tensor(0.0, device=device)
        if anchor_encoder is not None and args.lambda_anchor > 0:
            with torch.no_grad():
                vanilla_features = anchor_encoder(images)
            loss_anchor = F.mse_loss(img_features.float(), vanilla_features.float())
            if args.lambda_patch_anchor > 0:
                with torch.no_grad():
                    vanilla_patches = anchor_encoder.get_patch_tokens(images)  # [B, 576, 1024]
                loss_patch_anchor = F.mse_loss(patch_tokens.float(), vanilla_patches.float())
        
        loss = (
            args.lambda_gps * loss_gps +
            args.lambda_child * loss_child +
            args.lambda_parent * loss_parent +
            args.lambda_hierarchy * loss_hierarchy +
            args.lambda_patch_gps * loss_patch_gps +
            args.lambda_anchor * loss_anchor +
            args.lambda_patch_anchor * loss_patch_anchor
        )
        
        batch_size = len(images)
        total_loss += loss.item() * batch_size
        total_loss_gps += loss_gps.item() * batch_size
        total_loss_child += loss_child.item() * batch_size
        total_loss_parent += loss_parent.item() * batch_size
        total_loss_hierarchy += loss_hierarchy.item() * batch_size
        total_loss_patch_gps += loss_patch_gps.item() * batch_size
        total_loss_anchor += loss_anchor.item() * batch_size
        total_loss_patch_anchor += loss_patch_anchor.item() * batch_size
        total_count += batch_size
    
    return {
        "loss": total_loss / total_count,
        "loss_gps": total_loss_gps / total_count,
        "loss_child": total_loss_child / total_count,
        "loss_parent": total_loss_parent / total_count,
        "loss_hierarchy": total_loss_hierarchy / total_count,
        "loss_patch_gps": total_loss_patch_gps / total_count,
        "loss_anchor": total_loss_anchor / total_count,
        "loss_patch_anchor": total_loss_patch_anchor / total_count,
    }


# ============================================================================
# TRAINING
# ============================================================================

def train(args):
    """Main training function for Stage 0."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # ========================================================================
    # SETUP OUTPUT DIRECTORY
    # ========================================================================
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    if args.output_dir:
        output_dir = Path(args.output_dir)
    else:
        encoder_name = args.encoder_model.replace("/", "_").replace(" ", "_")
        output_dir = Path("results") / "stage0-prototype" / f"{encoder_name}" / timestamp
    
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
    
    concept_names = list(full_dataset.concept_to_idx.keys())
    parent_names = list(full_dataset.parent_to_idx.keys())
    concept_to_idx = full_dataset.concept_to_idx
    parent_to_idx = full_dataset.parent_to_idx
    meta_to_parent = full_dataset.meta_to_parent
    
    logger.info(f"Dataset: {len(full_dataset)} samples")
    logger.info(f"Meta concepts: {len(concept_names)}")
    logger.info(f"Parent concepts: {len(parent_names)}")
    logger.info(f"Countries: {len(full_dataset.country_to_idx)}")
    
    # ========================================================================
    # CREATE SPLITS (consistent with Stage 1 and Stage 2)
    # ========================================================================
    if args.splits_json and Path(args.splits_json).exists():
        # Load existing splits for consistency across stages
        logger.info(f"Loading splits from {args.splits_json}")
        train_samples, val_samples, test_samples = load_splits_from_json(
            args.splits_json, full_dataset.samples
        )
    else:
        # Create new strict stratified splits: 70/15/15
        logger.info("Creating new strict stratified splits (70/15/15)")
        train_samples, val_samples, test_samples = create_splits_stratified_strict(
            full_dataset.samples,
            train_ratio=0.70,
            val_ratio=0.15,
            test_ratio=0.15,
            seed=42,
        )
        # Save splits for reproducibility and cross-stage consistency
        splits_path = output_dir / "splits.json"
        save_splits_to_json(
            train_samples, val_samples, test_samples, splits_path,
            extra_info={"seed": 42, "train_ratio": 0.7, "val_ratio": 0.15, "test_ratio": 0.15}
        )
    
    logger.info(f"Main splits: Train={len(train_samples)}, Val={len(val_samples)}, Test={len(test_samples)}")
    log_split_diagnostics(train_samples, val_samples, test_samples)
    logger.info("Stage 0 uses TRAIN ONLY (val/test never seen by encoder)")
    
    # Further split train into pretrain_train/pretrain_val (90/10)
    pretrain_train, pretrain_val, _ = create_splits_stratified_strict(
        train_samples,
        train_ratio=0.90,
        val_ratio=0.10,
        test_ratio=0.0,
        seed=42,
    )
    logger.info(f"Pretrain splits: Train={len(pretrain_train)}, Val={len(pretrain_val)}")
    
    # ========================================================================
    # SETUP DATALOADERS
    # ========================================================================
    from src.dataset import SubsetDataset
    
    pretrain_train_dataset = SubsetDataset(full_dataset, pretrain_train)
    pretrain_val_dataset = SubsetDataset(full_dataset, pretrain_val)
    
    train_loader = DataLoader(
        pretrain_train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        pretrain_val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True,
    )
    
    # ========================================================================
    # BUILD IMAGE ENCODER
    # ========================================================================
    logger.info("Loading image encoder...")
    config = StreetCLIPConfig(model_name=args.encoder_model)
    image_encoder = StreetCLIPEncoder(config).to(device)

    anchor_encoder = None
    if args.lambda_anchor > 0:
        logger.info("Initializing frozen vanilla encoder for anchor loss...")
        anchor_config = StreetCLIPConfig(model_name=args.encoder_model, finetune=False, device=device)
        anchor_encoder = StreetCLIPEncoder(anchor_config)
        anchor_encoder.model.eval()
        for p in anchor_encoder.model.parameters():
            p.requires_grad = False
    
    # Unfreeze top layers for finetuning
    image_encoder.unfreeze_top_layers(args.unfreeze_layers)
    if args.finetune_text_encoder:
        image_encoder.unfreeze_text_encoder()
    
    trainable_encoder_params = sum(p.numel() for p in image_encoder.get_trainable_params())
    logger.info(f"Unfroze top {args.unfreeze_layers} vision layers")
    logger.info(f"Text encoder unfrozen: {args.finetune_text_encoder}")
    logger.info(f"Trainable encoder params: {trainable_encoder_params:,}")
    
    # ========================================================================
    # BUILD TEXT PROTOTYPES
    # ========================================================================
    logger.info("Building text prototypes...")
    
    _, concept_descriptions = extract_concepts_from_dataset(full_dataset)
    
    # Meta (child) prototypes
    T_meta = build_text_prototypes(
        concept_names=concept_names,
        text_encoder=image_encoder,
        concept_descriptions=concept_descriptions,
        templates=DEFAULT_CONCEPT_TEMPLATES,
        device=device,
    )
    logger.info(f"Built meta prototypes: {T_meta.shape}")
    
    # Parent prototypes
    T_parent = build_text_prototypes(
        concept_names=parent_names,
        text_encoder=image_encoder,
        concept_descriptions=None,
        templates=DEFAULT_PARENT_TEMPLATES,
        device=device,
    )
    logger.info(f"Built parent prototypes: {T_parent.shape}")
    
    # Meta -> parent index mapping
    meta_to_parent_idx = build_meta_to_parent_idx(
        meta_to_parent=meta_to_parent,
        concept_to_idx=concept_to_idx,
        parent_to_idx=parent_to_idx,
    ).to(device)
    
    # ========================================================================
    # CREATE STAGE 0 MODEL
    # ========================================================================
    logger.info("Creating Stage0PretrainingModel...")
    
    model = Stage0PretrainingModel(
        image_encoder=image_encoder,
        T_meta=T_meta,
        T_parent=T_parent,
        meta_to_parent_idx=meta_to_parent_idx,
        streetclip_dim=768,
        concept_emb_dim=512,
    ).to(device)
    
    trainable_params = model.get_trainable_params()
    num_trainable = sum(p.numel() for p in trainable_params)
    logger.info(f"Total trainable parameters: {num_trainable:,}")
    
    # ========================================================================
    # SETUP OPTIMIZER AND SCHEDULER
    # ========================================================================
    # Optimizer with param groups: use a smaller LR for encoder params to reduce forgetting.
    encoder_params = list(image_encoder.get_trainable_params())
    non_encoder_params = []
    encoder_param_ids = {id(p) for p in encoder_params}
    for p in trainable_params:
        if id(p) not in encoder_param_ids:
            non_encoder_params.append(p)

    optimizer = torch.optim.AdamW(
        [
            {"params": encoder_params, "lr": args.encoder_lr},
            {"params": non_encoder_params, "lr": args.lr},
        ],
        weight_decay=args.weight_decay,
    )
    
    scheduler = CosineAnnealingLR(
        optimizer,
        T_max=args.stage0_epochs,
        eta_min=args.lr * 0.01,
    )
    
    scaler = torch.amp.GradScaler("cuda", enabled=args.use_amp)
    
    # ========================================================================
    # WANDB INIT
    # ========================================================================
    if args.use_wandb:
        wandb.init(
            project="geolocation-cbm-stage0",
            config=vars(args),
            name=f"stage0-prototype-{timestamp}",
        )
        wandb.define_metric("epoch")
        wandb.define_metric("train/*", step_metric="epoch")
        wandb.define_metric("val/*", step_metric="epoch")
    
    # ========================================================================
    # TRAINING LOOP
    # ========================================================================
    logger.info(f"\n{'='*74}")
    logger.info(f"STAGE 0: Domain Contrastive Pretraining")
    logger.info(f"{'='*74}")
    logger.info(
        f"Losses: GPS({args.lambda_gps}) + Child({args.lambda_child}) + Parent({args.lambda_parent}) "
        f"+ Hierarchy({args.lambda_hierarchy}) + PatchGPS({args.lambda_patch_gps}) + Anchor({args.lambda_anchor})"
    )
    
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(args.stage0_epochs):
        model.train()
        
        total_loss = 0
        total_loss_gps = 0
        total_loss_child = 0
        total_loss_parent = 0
        total_loss_hierarchy = 0
        total_count = 0
        
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{args.stage0_epochs}")
        
        for batch_idx, batch in enumerate(pbar):
            images, concept_idx, parent_idx, country_idx, coords, metadata = batch
            images = images.to(device)
            coords = coords.to(device)
            concept_idx = concept_idx.to(device)
            parent_idx = parent_idx.to(device)
            
            with torch.amp.autocast("cuda", enabled=args.use_amp):
                outputs = model(images, coords)
                concept_emb_norm = outputs["concept_emb_norm"]
                patch_emb_norm = outputs["patch_emb_norm"]
                gps_emb = outputs["gps_emb"]
                img_features_norm = outputs["img_features_norm"]
                gps_emb_768_norm = outputs["gps_emb_768_norm"]
                img_features = outputs["img_features"]
                patch_tokens = outputs["patch_tokens"]
                
                # 1. Image-GPS contrastive
                loss_gps = clip_contrastive_loss(img_features_norm, gps_emb_768_norm, args.temperature)
                
                # 2. Image-Child Concept contrastive
                loss_child = concept_prototype_contrastive_loss(
                    concept_emb_norm, model.T_meta_projected, concept_idx, args.temperature
                )
                
                # 3. Image-Parent Concept contrastive
                loss_parent = concept_prototype_contrastive_loss(
                    concept_emb_norm, model.T_parent_projected, parent_idx, args.temperature
                )
                
                # 4. Child-Parent consistency (intra-batch supervised contrastive)
                loss_hierarchy = inter_parent_contrastive_loss(
                    concept_emb_norm, parent_idx, args.temperature
                )

                loss_patch_gps = clip_contrastive_loss(patch_emb_norm, gps_emb, args.temperature)

                loss_anchor = torch.tensor(0.0, device=device)
                loss_patch_anchor = torch.tensor(0.0, device=device)
                if anchor_encoder is not None and args.lambda_anchor > 0:
                    with torch.no_grad():
                        vanilla_features = anchor_encoder(images)
                    loss_anchor = F.mse_loss(img_features.float(), vanilla_features.float())
                    if args.lambda_patch_anchor > 0:
                        with torch.no_grad():
                            vanilla_patches = anchor_encoder.get_patch_tokens(images)  # [B, 576, 1024]
                        loss_patch_anchor = F.mse_loss(patch_tokens.float(), vanilla_patches.float())
                
                # Total loss
                loss = (
                    args.lambda_gps * loss_gps +
                    args.lambda_child * loss_child +
                    args.lambda_parent * loss_parent +
                    args.lambda_hierarchy * loss_hierarchy +
                    args.lambda_patch_gps * loss_patch_gps +
                    args.lambda_anchor * loss_anchor +
                    args.lambda_patch_anchor * loss_patch_anchor
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
            batch_size = len(images)
            total_loss += loss.item() * args.gradient_accumulation_steps * batch_size
            total_loss_gps += loss_gps.item() * batch_size
            total_loss_child += loss_child.item() * batch_size
            total_loss_parent += loss_parent.item() * batch_size
            total_loss_hierarchy += loss_hierarchy.item() * batch_size
            total_count += batch_size
            
            pbar.set_postfix({
                "loss": f"{loss.item():.4f}",
                "gps": f"{loss_gps.item():.4f}",
                "child": f"{loss_child.item():.4f}",
            })
        
        # Epoch metrics
        train_metrics = {
            "loss": total_loss / total_count,
            "loss_gps": total_loss_gps / total_count,
            "loss_child": total_loss_child / total_count,
            "loss_parent": total_loss_parent / total_count,
            "loss_hierarchy": total_loss_hierarchy / total_count,
        }
        log_metrics(train_metrics, prefix="Train", stage=0)
        
        # Validation
        val_metrics = validate(model, val_loader, device, args, anchor_encoder=anchor_encoder)
        log_metrics(val_metrics, prefix="Val", stage=0)
        
        scheduler.step()
        
        # Wandb logging
        if args.use_wandb:
            wandb.log({
                "epoch": epoch + 1,
                "train/loss": train_metrics["loss"],
                "train/loss_gps": train_metrics["loss_gps"],
                "train/loss_child": train_metrics["loss_child"],
                "train/loss_parent": train_metrics["loss_parent"],
                "train/loss_hierarchy": train_metrics["loss_hierarchy"],
                "val/loss": val_metrics["loss"],
                "val/loss_gps": val_metrics["loss_gps"],
                "val/loss_child": val_metrics["loss_child"],
                "val/loss_parent": val_metrics["loss_parent"],
                "val/loss_hierarchy": val_metrics["loss_hierarchy"],
                "train/lr": scheduler.get_last_lr()[0],
            }, step=epoch + 1)
        
        # Checkpointing
        val_loss = val_metrics["loss"]
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            
            save_checkpoint(
                model=model,
                checkpoint_path=output_dir / "checkpoints" / "best_model_stage0.pt",
                concept_names=concept_names,
                parent_names=parent_names,
                concept_to_idx=concept_to_idx,
                parent_to_idx=parent_to_idx,
                country_to_idx=full_dataset.country_to_idx,
                meta_to_parent=meta_to_parent,
                encoder_model=args.encoder_model,
                extra_info={"val_loss": val_loss},
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
            logger.info(f"New best model! Val Loss: {val_loss:.4f}")
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
                meta_to_parent=meta_to_parent,
                encoder_model=args.encoder_model,
                extra_info={"val_loss": val_loss},
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
            )
    
    # Freeze encoder after Stage 0
    logger.info("Freezing image encoder after Stage 0...")
    model.image_encoder.freeze_encoder()
    model.image_encoder.freeze_text_encoder()
    
    logger.info(f"\n{'='*74}")
    logger.info(f"Stage 0 complete! Best Val Loss: {best_val_loss:.4f}")
    logger.info(f"Checkpoint saved to: {output_dir / 'checkpoints' / 'best_model_stage0.pt'}")
    logger.info(f"{'='*74}")
    
    if args.use_wandb:
        wandb.finish()


# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Stage 0: Domain Contrastive Pretraining")
    
    # Dataset
    parser.add_argument("--csv_path", type=str, required=True, help="Path to CSV dataset")
    parser.add_argument("--data_root", type=str, default="data")
    parser.add_argument("--splits_json", type=str, default=None, help="Path to existing splits.json (if provided, loads splits instead of creating new)")

    # Model
    parser.add_argument("--encoder_model", type=str, default="geolocal/StreetCLIP")
    parser.add_argument("--unfreeze_layers", type=int, default=2, help="Number of top vision layers to unfreeze")
    parser.add_argument("--finetune_text_encoder", action="store_true", default=False)
    
    # Training
    parser.add_argument("--stage0_epochs", type=int, default=20)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--encoder_lr",
        type=float,
        default=3e-5,
        help="Learning rate for trainable StreetCLIP encoder params (use smaller than --lr to reduce forgetting).",
    )
    parser.add_argument("--weight_decay", type=float, default=0.01)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--use_amp", action="store_true", default=True)
    
    # Loss weights
    parser.add_argument("--lambda_gps", type=float, default=1.0, help="Weight for Image-GPS contrastive loss")
    parser.add_argument("--lambda_child", type=float, default=1.0, help="Weight for Image-Child Concept loss")
    parser.add_argument("--lambda_parent", type=float, default=0.5, help="Weight for Image-Parent Concept loss")
    parser.add_argument("--lambda_hierarchy", type=float, default=0.3, help="Weight for Child-Parent consistency loss")
    parser.add_argument("--lambda_patch_gps", type=float, default=0.2, help="Weight for Patch-GPS contrastive loss (Stage2-aligned)")
    parser.add_argument("--lambda_anchor", type=float, default=0.01, help="Weight for anchor loss to keep encoder close to vanilla features")
    parser.add_argument("--lambda_patch_anchor", type=float, default=0.01, help="Weight for patch-token anchor loss to keep patch tokens close to vanilla")
    parser.add_argument("--temperature", type=float, default=0.07, help="Contrastive loss temperature")
    
    # Misc
    parser.add_argument("--output_dir", type=str, default=None)
    parser.add_argument("--save_interval", type=int, default=5)
    parser.add_argument("--early_stopping_patience", type=int, default=5)
    parser.add_argument("--use_wandb", action="store_true", default=False)
    
    args = parser.parse_args()
    train(args)
