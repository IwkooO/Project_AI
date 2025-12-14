#!/usr/bin/env python3
"""
Train Concept Bottleneck Model for Geolocation.

Phase 1: Train Concept Head (predict concept from image)
Phase 2: Train Geo Head (predict location from image + concept)
Phase 3: Joint Fine-Tuning (optimize all heads)

Usage:
    # Phase 1
    python scripts/training/train_cbm.py --phase 1 ...

    # Phase 2
    python scripts/training/train_cbm.py --phase 2 --resume-checkpoint checkpoints/best_phase1.pt ...
    
    # Phase 3
    python scripts/training/train_cbm.py --phase 3 --resume-checkpoint checkpoints/best_phase2.pt ...
"""

import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
from tqdm import tqdm
import numpy as np
import json
import sys
import math
import matplotlib.pyplot as plt
from datetime import datetime
import torch.nn.functional as F

# ============================================================================
# HYPERPARAMETERS - Tuned for stability and performance
# ============================================================================
CONTRASTIVE_TEMPERATURE = 0.2  # Increased from 0.07 for numerical stability
CONTRASTIVE_WEIGHT = 0.1       # Slightly reduced
GRAD_CLIP_NORM = 5.0           # Increased from 1.0 for patch-only training stability
WARMUP_EPOCHS = 5              # LR warmup before cosine decay
LABEL_SMOOTHING = 0.1          # Reduced from 0.2
# MIL/patch-evidence concept head defaults (faithful attention maps)
MIL_TOPK_DEFAULT = 8
MIL_TAU_DEFAULT = 0.1
CONCEPT_DIM_DEFAULT = 256

try:
    import wandb
except ImportError:  # pragma: no cover
    wandb = None

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

from src.models.cbm import CBM
from src.data.dataset_concept import ConceptDataset, collate_fn

# Check for Cartopy (optional, for map plots)
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    CARTOPY_AVAILABLE = True
except ImportError:
    CARTOPY_AVAILABLE = False


def compute_concept_weights(dataset, num_concepts, device):
    """Compute inverse frequency weights for concept loss using Pandas."""
    print("Computing concept class weights...")
    
    # Get concept names directly from dataframe
    concept_names = dataset.df[dataset.concept_col]
    
    # Count occurrences
    counts_dict = concept_names.value_counts().to_dict()
    
    # Map to indices
    counts = torch.zeros(num_concepts)
    for name, count in counts_dict.items():
        if name in dataset.concept_to_idx:
            idx = dataset.concept_to_idx[name]
            counts[idx] = count
            
    # Smooth inverse frequency: 1 / sqrt(count)
    weights = 1.0 / torch.sqrt(counts + 1.0)
    # Clip extreme weights to prevent instability
    weights = torch.clamp(weights, min=0.1, max=10.0)
    weights = weights / weights.mean()  # Normalize
    return weights.to(device)


def print_param_counts(model, phase):
    """Print total and phase-trainable parameter counts."""
    total = sum(p.numel() for p in model.parameters())
    # Phase 1-only model: concept head should be the only module
    if phase != 1:
        raise ValueError(f"Only Phase 1 is supported. Got phase={phase}")
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Params: total={total/1e6:.2f}M, trainable (phase {phase})={trainable/1e6:.2f}M")


def check_for_nan(tensor, name="tensor"):
    """Check if tensor contains NaN or Inf and raise error if so."""
    if torch.isnan(tensor).any():
        raise ValueError(f"NaN detected in {name}")
    if torch.isinf(tensor).any():
        raise ValueError(f"Inf detected in {name}")


def visualize_predictions(model, dataset, device, epoch, output_dir, phase, run_id=None):
    """
    Visualize predictions for a few validation samples.
    Phase 1: Image + Top 5 Concepts
    Phase 2/3: Image + Top 5 Concepts + Predicted Location vs GT
    """
    model.eval()
    
    # Select 5 random indices
    indices = np.random.choice(len(dataset), 5, replace=False)
    
    fig_h = 5 if phase == 1 else 7
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, fig_h * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, idx in enumerate(indices):
        # Load sample
        pooled, patches, c_label, coords, cell_label, offset = dataset[idx]
        
        # Forward pass
        with torch.no_grad():
            pooled_dev = pooled.unsqueeze(0).to(device)
            patches_dev = patches.unsqueeze(0).to(device) if patches is not None else None
            c_logits, c_hidden, attn_w, pooled_ctx = model(pooled_dev, patches_dev)
            c_probs = torch.softmax(c_logits, dim=1)
        
        # Top 5 Concepts
        top5_prob, top5_idx = torch.topk(c_probs[0], 5)
        top5_concepts = [dataset.get_concept_name(idx.item()) for idx in top5_idx]
        gt_concept = dataset.get_concept_name(c_label.item())
        
        # Get Image Path
        img_path = dataset.df.iloc[idx]['image_path']
        img_ax = axes[i, 0]
        bar_ax = axes[i, 1]
        
        # Try to load image
        image_loaded = False
        img_arr = None
        try:
            img = plt.imread(img_path)
            img_ax.imshow(img)
            img_arr = img
            image_loaded = True
        except Exception:
            img_ax.text(0.5, 0.5, "Image not found", ha='center')
            
        img_ax.axis('off')
        img_ax.set_title(f"GT Concept: {gt_concept}", loc='left', fontsize=10, bbox=dict(facecolor='white', alpha=0.7))
        
        # Bar chart for top-5 concepts with highlighting for correct predictions
        y_pos = np.arange(len(top5_concepts))
        probs_np = top5_prob.cpu().numpy()
        
        # Color bars: green if correct, steelblue if incorrect
        colors = ['#2ecc71' if concept == gt_concept else 'steelblue' for concept in top5_concepts]
        bar_ax.barh(y_pos, probs_np, color=colors)
        
        # Add checkmark or highlight for correct predictions
        for j, (concept, prob) in enumerate(zip(top5_concepts, probs_np)):
            if concept == gt_concept:
                bar_ax.text(prob + 0.02, j, '✓', fontsize=14, color='#27ae60', fontweight='bold', va='center')
        
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels(top5_concepts)
        bar_ax.invert_yaxis()
        bar_ax.set_xlim(0, 1)
        bar_ax.set_xlabel("Probability")
        bar_ax.set_title("Top-5 concept probs", loc='left')

        # Save attention weights and heatmaps
        if attn_w is not None:
            attn_np = attn_w.squeeze(0).detach().cpu().numpy()
            attn_dir = (output_dir / "visualizations" / (run_id or "attn") / f"epoch_{epoch}_sample_{idx}")
            attn_dir.mkdir(parents=True, exist_ok=True)
            np.save(attn_dir / "attn_weights.npy", attn_np)

            P = attn_np.shape[1]
            side = int(math.isqrt(P))
            square = side * side == P
            max_concepts = min(3, attn_np.shape[0])
            for j in range(max_concepts):
                concept_name = top5_concepts[j] if j < len(top5_concepts) else f"c{j}"
                weights = attn_np[top5_idx[j]] if j < len(top5_idx) else attn_np[j]
                plt.figure(figsize=(3, 3))
                if square:
                    grid = weights.reshape(side, side)
                    plt.imshow(grid, cmap="magma")
                    plt.title(f"Attn {concept_name}")
                    plt.axis("off")
                else:
                    plt.bar(range(P), weights)
                    plt.title(f"Attn {concept_name}")
                plt.tight_layout()
                plt.savefig(attn_dir / f"attn_{concept_name}.png", dpi=120)
                plt.close()

            def _safe_name(name: str) -> str:
                # Keep filenames portable
                return "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(name))[:120]

            # Overlay heatmaps for GT + top-k predicted concepts (square grids only)
            if square and image_loaded and img_arr is not None:
                gt_idx = c_label.item()
                overlay_indices = []
                if gt_idx < attn_np.shape[0]:
                    overlay_indices.append(("gt", gt_idx, None))
                # add top-k predicted concepts (avoid duplicates)
                for rank, (ci, prob) in enumerate(zip(top5_idx.tolist(), top5_prob.tolist()), start=1):
                    if ci < attn_np.shape[0] and ci != gt_idx:
                        overlay_indices.append((f"top{rank}", ci, prob))

                for tag, concept_idx, prob in overlay_indices:
                    weights = attn_np[concept_idx]
                    grid = weights.reshape(side, side)
                    grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
                    grid_up = F.interpolate(
                        grid_t,
                        size=(img_arr.shape[0], img_arr.shape[1]),
                        mode="bilinear",
                        align_corners=False,
                    )
                    grid_up_np = grid_up.squeeze(0).squeeze(0).cpu().numpy()
                    concept_name = dataset.get_concept_name(concept_idx)

                    plt.figure(figsize=(5, 5))
                    plt.imshow(img_arr)
                    plt.imshow(grid_up_np, cmap="magma", alpha=0.35)
                    if tag == "gt":
                        title = f"Attn Overlay (GT) {concept_name}"
                    else:
                        title = f"Attn Overlay ({tag}) {concept_name}"
                        if prob is not None:
                            title += f"  p={prob:.3f}"
                    plt.title(title)
                    plt.axis("off")
                    plt.tight_layout()
                    plt.savefig(attn_dir / f"attn_overlay_{tag}_{_safe_name(concept_name)}.png", dpi=120)
                    plt.close()
        
        # Phase 1: Concept prediction only (no geo info)
        
    plt.tight_layout()
    
    viz_dir = output_dir / "visualizations"
    if run_id:
        viz_dir = viz_dir / run_id
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(viz_dir / f"epoch_{epoch}_phase{phase}.png", dpi=150, bbox_inches='tight')
    plt.close()


def pairwise_haversine(coords_rad: torch.Tensor) -> torch.Tensor:
    """
    coords_rad: [B, 2] in radians (lat, lng)
    returns pairwise distance matrix in km
    """
    lat = coords_rad[:, 0].unsqueeze(1)
    lng = coords_rad[:, 1].unsqueeze(1)
    dlat = lat - lat.transpose(0, 1)
    dlng = lng - lng.transpose(0, 1)
    a = torch.sin(dlat / 2) ** 2 + torch.cos(lat) * torch.cos(lat.transpose(0, 1)) * torch.sin(dlng / 2) ** 2
    c = 2 * torch.arcsin(torch.sqrt(torch.clamp(a, 0, 1)))
    earth_radius_km = 6371.0
    return earth_radius_km * c


def supcon_loss_stable(features: torch.Tensor, labels: torch.Tensor, 
                       temperature: float = CONTRASTIVE_TEMPERATURE, 
                       coords: torch.Tensor = None, 
                       geo_neg_scale: float = 0.0) -> torch.Tensor:
    """
    Numerically stable supervised contrastive loss using log-sum-exp trick.
    """
    if features.size(0) < 2:
        return torch.tensor(0.0, device=features.device)

    features = F.normalize(features, dim=1)
    B = features.size(0)
    
    # Similarity matrix
    sim_matrix = torch.matmul(features, features.T) / temperature
    
    # For numerical stability, subtract max
    sim_max, _ = sim_matrix.max(dim=1, keepdim=True)
    sim_matrix = sim_matrix - sim_max.detach()
    
    # Positive mask: same label, excluding self
    mask = torch.eq(labels.unsqueeze(1), labels.unsqueeze(0)).float()
    mask.fill_diagonal_(0)
    
    # Negative mask: all except self
    eye = torch.eye(B, device=features.device)
    neg_mask = 1.0 - eye

    # Optional geo-aware weighting for negatives
    if coords is not None and geo_neg_scale > 0.0:
        dist_km = pairwise_haversine(coords)
        dist_norm = dist_km / (dist_km.max() + 1e-6)
        neg_weights = 1.0 + geo_neg_scale * dist_norm
        neg_weights = neg_weights * neg_mask
    else:
        neg_weights = neg_mask

    # Log-sum-exp for denominator (all negatives + positives)
    exp_sim = torch.exp(sim_matrix) * neg_weights
    log_sum_exp = torch.log(exp_sim.sum(dim=1, keepdim=True) + 1e-8)
    
    # Log probability
    log_prob = sim_matrix - log_sum_exp

    # Mean log prob over positives
    positive_mask_sum = mask.sum(dim=1)
    valid = positive_mask_sum > 0
    
    if not valid.any():
        return torch.tensor(0.0, device=features.device)

    mean_log_prob_pos = (mask[valid] * log_prob[valid]).sum(dim=1) / (positive_mask_sum[valid] + 1e-8)
    loss = -mean_log_prob_pos.mean()
    
    # Clamp to prevent extreme values
    loss = torch.clamp(loss, min=0.0, max=20.0)
    
    return loss


class WarmupCosineScheduler:
    """Learning rate scheduler with linear warmup then cosine decay."""
    def __init__(self, optimizer, warmup_epochs, total_epochs, min_lr_ratio=0.01):
        self.optimizer = optimizer
        self.warmup_epochs = warmup_epochs
        self.total_epochs = total_epochs
        self.min_lr_ratio = min_lr_ratio
        self.base_lrs = [pg['lr'] for pg in optimizer.param_groups]
        
    def step(self, epoch):
        if epoch < self.warmup_epochs:
            # Linear warmup
            lr_scale = (epoch + 1) / self.warmup_epochs
        else:
            # Cosine decay
            progress = (epoch - self.warmup_epochs) / (self.total_epochs - self.warmup_epochs)
            lr_scale = self.min_lr_ratio + (1 - self.min_lr_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        
        for pg, base_lr in zip(self.optimizer.param_groups, self.base_lrs):
            pg['lr'] = base_lr * lr_scale
    
    def get_last_lr(self):
        return [pg['lr'] for pg in self.optimizer.param_groups]


def train_epoch(model, loader, optimizer, device, phase, concept_criterion, cell_criterion, offset_criterion, geo_neg_scale=0.0):
    model.train()
    
    # Phase 1-only model: ensure all params trainable
    if phase == 1:
        for p in model.parameters():
            p.requires_grad = True
    else:
        raise NotImplementedError(
            "Phase 2 and 3 require GeoHead and RelevanceGate components. "
            "This model is configured for Phase 1 (concept prediction) only."
        )
        
    total_loss = 0
    correct_c_top1 = 0
    correct_c_top5 = 0
    correct_cell = 0
    total_samples = 0
    total_contrastive_loss = 0.0
    contrastive_samples = 0
    grad_norm_accum = 0.0
    grad_norm_count = 0
    
    pbar = tqdm(loader, desc="Train")
    
    for pooled, patches, c_labels, coords, cell_labels, offsets in pbar:
        pooled = pooled.to(device)
        if patches is not None:
            patches = patches.to(device)
        c_labels = c_labels.to(device)
        cell_labels = cell_labels.to(device)
        offsets = offsets.to(device)
        coords = coords.to(device) if isinstance(coords, torch.Tensor) else torch.tensor(coords, device=device, dtype=torch.float32)
        
        optimizer.zero_grad()
        
        # Forward
        c_logits, c_hidden, attn_w, pooled_ctx = model(pooled, patches)
        
        # Check for NaN in outputs
        try:
            check_for_nan(c_logits, "c_logits")
            check_for_nan(c_hidden, "c_hidden")
        except ValueError as e:
            print(f"WARNING: {e} - skipping batch")
            continue
        
        loss = 0
        contrastive_value = None
        
        # Phase 1: Concept Loss
        if phase == 1:
            c_loss = concept_criterion(c_logits, c_labels)
            loss = c_loss
            
            # Stable contrastive loss
            contrastive_value = supcon_loss_stable(c_hidden, c_labels, coords=coords, geo_neg_scale=geo_neg_scale)
            loss = loss + CONTRASTIVE_WEIGHT * contrastive_value
            
            # Metrics
            preds = c_logits.argmax(dim=1)
            correct_c_top1 += (preds == c_labels).sum().item()

            k = min(5, c_logits.size(1))
            topk = c_logits.topk(k, dim=1).indices  # [B, k]
            correct_c_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
            
            pbar.set_postfix({"C_Loss": f"{c_loss.item():.4f}", "SupCon": f"{contrastive_value.item():.4f}"})
            
        # Phase 2 and 3 not supported in Phase 1-only model
        elif phase in (2, 3):
            raise NotImplementedError(
                "Phase 2 and 3 require GeoHead and RelevanceGate components. "
                "This model is configured for Phase 1 (concept prediction) only."
            )
        
        # Check for NaN in loss
        if torch.isnan(loss) or torch.isinf(loss):
            print(f"WARNING: NaN/Inf loss detected - skipping batch")
            continue
        
        loss.backward()
        
        # Gradient clipping - CRITICAL for stability
        grad_norm = torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP_NORM)
        grad_norm_accum += grad_norm.item()
        grad_norm_count += 1
        
        optimizer.step()
        
        total_loss += loss.item() * pooled.size(0)
        if contrastive_value is not None:
            total_contrastive_loss += contrastive_value.item() * pooled.size(0)
            contrastive_samples += pooled.size(0)
        total_samples += pooled.size(0)
        
    avg_contrastive = total_contrastive_loss / (contrastive_samples + 1e-8) if contrastive_samples > 0 else 0.0
    avg_grad_norm = grad_norm_accum / (grad_norm_count + 1e-8) if grad_norm_count > 0 else 0.0
    
    return (
        total_loss / total_samples,
        correct_c_top1 / total_samples,
        correct_c_top5 / total_samples,
        correct_cell / total_samples,
        avg_contrastive,
        avg_grad_norm,
    )


@torch.no_grad()
def eval_epoch(model, loader, device, phase, concept_criterion, cell_criterion, offset_criterion):
    model.eval()
    total_loss = 0
    correct_c_top1 = 0
    correct_c_top5 = 0
    correct_cell = 0
    total_samples = 0
    
    for pooled, patches, c_labels, coords, cell_labels, offsets in tqdm(loader, desc="Eval"):
        pooled = pooled.to(device)
        if patches is not None:
            patches = patches.to(device)
        c_labels = c_labels.to(device)
        cell_labels = cell_labels.to(device)
        offsets = offsets.to(device)
        coords = coords.to(device) if isinstance(coords, torch.Tensor) else torch.tensor(coords, device=device, dtype=torch.float32)
        
        c_logits, c_hidden, attn_w, pooled_ctx = model(pooled, patches)
        
        loss = 0
        if phase == 1:
            loss = concept_criterion(c_logits, c_labels)
            preds = c_logits.argmax(dim=1)
            correct_c_top1 += (preds == c_labels).sum().item()

            k = min(5, c_logits.size(1))
            topk = c_logits.topk(k, dim=1).indices  # [B, k]
            correct_c_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
        else:
            # Phase 2 and 3 not supported in Phase 1-only model
            raise NotImplementedError(
                "Phase 2 and 3 require GeoHead and RelevanceGate components. "
                "This model is configured for Phase 1 (concept prediction) only."
            )
        
        total_loss += loss.item() * pooled.size(0)
        total_samples += pooled.size(0)
    return (
        total_loss / total_samples,
        correct_c_top1 / (total_samples + 1e-8),
        correct_c_top5 / (total_samples + 1e-8),
        correct_cell / (total_samples + 1e-8),
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--phase", type=int, choices=[1, 2, 3], required=True)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--val-csv", required=True)
    parser.add_argument("--cached-dir", required=True)
    parser.add_argument("--concept-data-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--resume-checkpoint", type=str, default=None)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--lr", type=float, default=1e-4)  # Reduced from 2e-4 to prevent overfitting
    parser.add_argument("--wandb", action="store_true", help="Log training metrics to Weights & Biases")
    parser.add_argument("--wandb-project", type=str, default="cbm_concept_bottleneck")
    parser.add_argument("--wandb-entity", type=str, default=None)
    parser.add_argument("--wandb-run-name", type=str, default=None)
    parser.add_argument("--load-patch-tokens", action="store_true", help="Load spatial patch tokens for attention")
    parser.add_argument("--concept-dim", type=int, default=CONCEPT_DIM_DEFAULT, help="Concept space dim for patch evidence scoring")
    parser.add_argument("--mil-topk", type=int, default=MIL_TOPK_DEFAULT, help="Top-k patches used per concept (MIL pooling)")
    parser.add_argument("--mil-tau", type=float, default=MIL_TAU_DEFAULT, help="Temperature for MIL logsumexp pooling and attention normalization")
    parser.add_argument("--geo-aware-supcon", action="store_true", help="Weight SupCon negatives by geo distance")
    parser.add_argument("--supcon-geo-scale", type=float, default=0.5)
    parser.add_argument("--warmup-epochs", type=int, default=WARMUP_EPOCHS)
    args = parser.parse_args()
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    phase_output_dir = output_dir / f"phase{args.phase}"
    phase_output_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate unique run ID based on timestamp
    run_id = datetime.now().strftime("%Y%m%d_%H%M%S")
    print(f"Run ID: {run_id}")
    print(f"Using device: {device}")
    print(f"Hyperparameters: temp={CONTRASTIVE_TEMPERATURE}, contrastive_weight={CONTRASTIVE_WEIGHT}, "
          f"grad_clip={GRAD_CLIP_NORM}, warmup={args.warmup_epochs}, label_smooth={LABEL_SMOOTHING}, "
          f"mil_topk={args.mil_topk}, mil_tau={args.mil_tau}, concept_dim={args.concept_dim}")
    
    use_wandb = args.wandb and wandb is not None
    if args.wandb and wandb is None:
        print("W&B logging requested but `wandb` is not installed; skipping W&B.")

    wandb_run = None
    if use_wandb:
        wandb_config = {
            key: (str(value) if isinstance(value, Path) else value)
            for key, value in vars(args).items()
        }
        wandb_config["run_id"] = run_id
        wandb_config["contrastive_temperature"] = CONTRASTIVE_TEMPERATURE
        wandb_config["contrastive_weight"] = CONTRASTIVE_WEIGHT
        wandb_config["grad_clip_norm"] = GRAD_CLIP_NORM
        wandb_config["label_smoothing"] = LABEL_SMOOTHING
        wandb_run = wandb.init(
            project=args.wandb_project,
            entity=args.wandb_entity,
            name=args.wandb_run_name or f"phase{args.phase}_{run_id}",
            config=wandb_config,
            reinit=True,
        )
    concept_vocab = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab = Path(args.concept_data_dir) / "s2_cells.json"
    
    # Load Datasets
    print("Loading datasets...")
    train_ds = ConceptDataset(args.train_csv, args.cached_dir, concept_vocab, str(s2_vocab), split="train", load_patch_tokens=args.load_patch_tokens)
    val_ds = ConceptDataset(args.val_csv, args.cached_dir, concept_vocab, str(s2_vocab), split="val", load_patch_tokens=args.load_patch_tokens)
    
    if wandb_run:
        wandb_run.config.update({
            "num_train_samples": len(train_ds),
            "num_val_samples": len(val_ds),
            "num_concepts": train_ds.num_concepts,
            "num_cells": train_ds.num_cells,
            "run_id": run_id,
        }, allow_val_change=True)
    
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, collate_fn=collate_fn, num_workers=4)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=collate_fn, num_workers=4)
    
    # Detect patch dimension if loaded
    patch_dim = 1024
    if args.load_patch_tokens and train_ds.patch_tokens is not None:
        patch_dim = train_ds.patch_tokens.shape[2]
        print(f"Detected patch dimension: {patch_dim}")
    
    # Initialize Model
    print("Initializing CBM (Phase 1: Concept Prediction Only)...")
    model = CBM(
        num_concepts=train_ds.num_concepts,
        input_dim=768,
        patch_dim=patch_dim,
        dropout=0.5,  # Increased for better regularization
        concept_dim=args.concept_dim,
        mil_topk=args.mil_topk,
        mil_tau=args.mil_tau,
    ).to(device)
    print_param_counts(model, args.phase)
    
    # Resume / Load Pretrained
    if args.resume_checkpoint:
        print(f"Loading checkpoint from {args.resume_checkpoint}")
        checkpoint = torch.load(args.resume_checkpoint, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
    
    # Criterions with reduced label smoothing
    weights = compute_concept_weights(train_ds, train_ds.num_concepts, device)
    concept_criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=LABEL_SMOOTHING)
    
    cell_criterion = nn.CrossEntropyLoss()
    offset_criterion = nn.MSELoss()
    
    if args.phase == 1:
        print("Phase 1: Concept Training")
    else:
        raise ValueError(f"Only Phase 1 is supported. Got phase={args.phase}")
        
    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=args.lr, weight_decay=0.05)  # Increased from 0.01 to reduce overfitting
    scheduler = WarmupCosineScheduler(optimizer, args.warmup_epochs, args.epochs)
    
    best_metric = 0.0
    patience = 10  # Early stopping patience
    patience_counter = 0
    
    try:
        for epoch in range(args.epochs):
            print(f"\nEpoch {epoch+1}/{args.epochs} (LR: {scheduler.get_last_lr()[0]:.2e})")
            
            train_loss, train_acc1, train_acc5, train_acc_cell, train_contrastive_loss, avg_grad_norm = train_epoch(
                model, train_loader, optimizer, device, args.phase,
                concept_criterion, cell_criterion, offset_criterion,
                geo_neg_scale=args.supcon_geo_scale if args.geo_aware_supcon else 0.0
            )
            
            val_loss, val_acc1, val_acc5, val_acc_cell = eval_epoch(
                model, val_loader, device, args.phase,
                concept_criterion, cell_criterion, offset_criterion
            )
            
            scheduler.step(epoch)
            
            # Log (Phase 1 only)
            if args.phase == 1:
                print(f"Train Loss: {train_loss:.4f} | Acc@1: {train_acc1:.4f} | Acc@5: {train_acc5:.4f} | Grad: {avg_grad_norm:.4f}")
                print(f"Val   Loss: {val_loss:.4f}   | Acc@1: {val_acc1:.4f} | Acc@5: {val_acc5:.4f}")
                # Early stopping / checkpoint selection metric
                metric = val_acc5
            else:
                raise NotImplementedError(
                    "Phase 2 and 3 require GeoHead and RelevanceGate components. "
                    "This model is configured for Phase 1 (concept prediction) only."
                )
            
            if wandb_run:
                log_dict = {
                    "phase": args.phase,
                    "epoch": epoch + 1,
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "train_concept_acc@1": train_acc1,
                    "train_concept_acc@5": train_acc5,
                    "val_concept_acc@1": val_acc1,
                    "val_concept_acc@5": val_acc5,
                    "train_cell_acc": train_acc_cell,
                    "val_cell_acc": val_acc_cell,
                    "train_contrastive_loss": train_contrastive_loss,
                    "avg_grad_norm": avg_grad_norm,
                    "learning_rate": scheduler.get_last_lr()[0],
                }
                wandb_run.log(log_dict, step=epoch + 1)
            
            # Visualization
            if (epoch + 1) % 5 == 0 or epoch == 0:
                print("Generating visualizations...")
                visualize_predictions(model, val_ds, device, epoch, phase_output_dir, args.phase, run_id=run_id)
            
            # Save Best
            if metric > best_metric:
                best_metric = metric
                patience_counter = 0
                save_path = phase_output_dir / f"best_phase{args.phase}.pt"
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'metric': best_metric,
                }, save_path)
                print(f"Saved best model to {save_path}")
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print(f"Early stopping triggered after {patience} epochs without improvement")
                    break
                    
    finally:
        if wandb_run:
            wandb_run.finish()

if __name__ == "__main__":
    main()
