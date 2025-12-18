"""
Attention visualization utilities for CBM.

Extracted from training script for reuse in train/eval.
"""

import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
import math
from pathlib import Path
from typing import Optional


def _safe_name(name: str) -> str:
    """Sanitize concept name for use in filenames."""
    return "".join(ch if (ch.isalnum() or ch in ("_", "-", ".")) else "_" for ch in str(name))[:120]


def save_attention_weights(
    attn_weights: torch.Tensor,
    output_dir: Path,
    epoch: int,
    sample_idx: int,
    run_id: Optional[str] = None,
    reason: Optional[str] = None,
):
    """
    Save attention weights to disk.
    
    Args:
        attn_weights: [K, P] attention weights (or [1, K, P] will be squeezed)
        output_dir: Base output directory
        epoch: Epoch number
        sample_idx: Sample index
        run_id: Optional run identifier
        reason: Optional reason string (e.g., "best_val_acc", "periodic_every5")
    """
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
    
    # Directory structure: visualizations/{run_id}/{reason}/epoch_{epoch}_sample_{idx}/
    base_viz_root = output_dir / "visualizations"
    run_root = base_viz_root / (run_id or "attn")
    if reason:
        run_root = run_root / reason
    attn_dir = run_root / f"epoch_{epoch}_sample_{sample_idx}"
    attn_dir.mkdir(parents=True, exist_ok=True)
    
    # Save raw attention weights
    np.save(attn_dir / "attn_weights.npy", attn_np)
    
    return attn_dir, attn_np


def visualize_attention_overlays(
    attn_weights: torch.Tensor,
    image_path: str | Path,
    dataset,
    concept_indices: list[int],
    concept_probs: Optional[list[float]] = None,
    concept_tags: Optional[list[str]] = None,
    output_dir: Path = None,
    epoch: int = 0,
    sample_idx: int = 0,
    run_id: Optional[str] = None,
    reason: Optional[str] = None,
):
    """
    Generate attention overlay visualizations.
    
    Args:
        attn_weights: [K, P] attention weights
        image_path: Path to source image
        dataset: Dataset instance (for get_concept_name)
        concept_indices: List of concept indices to visualize
        concept_probs: Optional list of probabilities for each concept
        concept_tags: Optional list of tags (e.g., "gt", "top1", "top2")
        output_dir: Output directory (if None, overlays won't be saved)
        epoch: Epoch number
        sample_idx: Sample index
        run_id: Optional run identifier
        reason: Optional reason string
    """
    attn_np = attn_weights.squeeze(0).detach().cpu().numpy()
    
    P = attn_np.shape[1]
    side = int(math.isqrt(P))
    square = side * side == P
    
    if not square:
        # Non-square grids: just save attention heatmaps
        if output_dir:
            attn_dir, _ = save_attention_weights(
                attn_weights, output_dir, epoch, sample_idx, run_id, reason
            )
            for j, concept_idx in enumerate(concept_indices[:3]):
                if concept_idx < attn_np.shape[0]:
                    weights = attn_np[concept_idx]
                    concept_name = dataset.get_concept_name(concept_idx)
                    plt.figure(figsize=(3, 3))
                    plt.bar(range(P), weights)
                    plt.title(f"Attn {concept_name}")
                    plt.tight_layout()
                    plt.savefig(attn_dir / f"attn_{_safe_name(concept_name)}.png", dpi=120)
                    plt.close()
        return
    
    # Square grids: generate overlays
    try:
        img_arr = plt.imread(image_path)
        image_loaded = True
    except Exception:
        image_loaded = False
        img_arr = None
    
    if not image_loaded or img_arr is None:
        return
    
    # Save attention weights and get output directory
    if output_dir:
        attn_dir, _ = save_attention_weights(
            attn_weights, output_dir, epoch, sample_idx, run_id, reason
        )
        
        # Generate overlays for each concept
        for tag, concept_idx, prob in zip(
            concept_tags or [f"c{i}" for i in range(len(concept_indices))],
            concept_indices,
            concept_probs or [None] * len(concept_indices),
        ):
            if concept_idx >= attn_np.shape[0]:
                continue
            
            weights = attn_np[concept_idx]
            grid = weights.reshape(side, side)
            
            # Upsample attention grid to image size
            grid_t = torch.tensor(grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            grid_up = F.interpolate(
                grid_t,
                size=(img_arr.shape[0], img_arr.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            grid_up_np = grid_up.squeeze(0).squeeze(0).cpu().numpy()
            
            concept_name = dataset.get_concept_name(concept_idx)
            
            # Create overlay visualization
            plt.figure(figsize=(5, 5))
            plt.imshow(img_arr)
            plt.imshow(grid_up_np, cmap="magma", alpha=0.35)
            
            if tag == "gt":
                if prob is not None:
                    title = f"Attn Overlay (GT) {concept_name}  p={prob:.3f}"
                else:
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


def visualize_predictions_summary(
    model,
    dataset,
    device: torch.device,
    epoch: int,
    output_dir: Path,
    phase: int = 1,
    run_id: Optional[str] = None,
    reason: Optional[str] = None,
    num_samples: int = 5,
):
    """
    Generate summary visualizations for validation samples.
    
    This is a patch-only version that works with the new model interface.
    
    Args:
        model: Trained concept model (patch-only forward)
        dataset: Validation ConceptDataset
        device: Torch device
        epoch: Zero-based epoch index
        output_dir: Phase-specific output directory
        phase: Training phase (currently 1 only)
        run_id: Optional run identifier string
        reason: Optional reason string (e.g., "best_val_acc", "periodic_every5")
        num_samples: Number of samples to visualize
    """
    model.eval()
    
    # Select random indices
    num_samples = max(1, min(num_samples, len(dataset)))
    indices = np.random.choice(len(dataset), num_samples, replace=False)
    
    fig_h = 5 if phase == 1 else 7
    fig, axes = plt.subplots(len(indices), 2, figsize=(14, fig_h * len(indices)))
    if len(indices) == 1:
        axes = np.expand_dims(axes, axis=0)
    
    for i, idx in enumerate(indices):
        # Load sample (patch-only: no pooled)
        patches, c_label, coords, cell_label, offset = dataset[idx]
        
        # Forward pass (patch-only)
        with torch.no_grad():
            patches_dev = patches.unsqueeze(0).to(device)
            c_logits, c_hidden, attn_w, _ = model(patches_dev)
            c_probs = torch.softmax(c_logits, dim=1)
        
        # Top 5 Concepts
        top5_prob, top5_idx = torch.topk(c_probs[0], 5)
        top5_concepts = [dataset.get_concept_name(idx.item()) for idx in top5_idx]
        # c_label from dataset is already a Python int, not a tensor
        gt_concept = dataset.get_concept_name(int(c_label))
        
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
        
        # Bar chart for top-5 concepts
        y_pos = np.arange(len(top5_concepts))
        probs_np = top5_prob.cpu().numpy()
        
        # Color bars: green if correct, steelblue if incorrect
        colors = ['#2ecc71' if concept == gt_concept else 'steelblue' for concept in top5_concepts]
        bar_ax.barh(y_pos, probs_np, color=colors)
        
        # Add checkmark for correct predictions
        for j, (concept, prob) in enumerate(zip(top5_concepts, probs_np)):
            if concept == gt_concept:
                bar_ax.text(prob + 0.02, j, '✓', fontsize=14, color='#27ae60', fontweight='bold', va='center')
        
        bar_ax.set_yticks(y_pos)
        bar_ax.set_yticklabels(top5_concepts)
        bar_ax.invert_yaxis()
        bar_ax.set_xlim(0, 1)
        bar_ax.set_xlabel("Probability")
        bar_ax.set_title("Top-5 concept probs", loc='left')
        
        # Generate attention overlays
        if attn_w is not None and image_loaded:
            # c_label from dataset is already a Python int, not a tensor
            gt_idx = int(c_label)
            gt_prob = float(c_probs[0, gt_idx].item()) if 0 <= gt_idx < c_probs.shape[1] else None
            
            # Prepare concept indices and tags for overlay
            overlay_indices = []
            overlay_tags = []
            overlay_probs = []
            
            if gt_idx < attn_w.shape[1]:  # attn_w is [1, K, P] -> [K, P] after squeeze
                overlay_indices.append(gt_idx)
                overlay_tags.append("gt")
                overlay_probs.append(gt_prob)
            
            # Add top-k predicted concepts (avoid duplicates)
            for rank, (ci, prob) in enumerate(zip(top5_idx.tolist(), top5_prob.tolist()), start=1):
                if ci < attn_w.shape[1] and ci != gt_idx:
                    overlay_indices.append(ci)
                    overlay_tags.append(f"top{rank}")
                    overlay_probs.append(float(prob))
            
            # Generate overlays
            visualize_attention_overlays(
                attn_weights=attn_w,
                image_path=img_path,
                dataset=dataset,
                concept_indices=overlay_indices,
                concept_probs=overlay_probs,
                concept_tags=overlay_tags,
                output_dir=output_dir,
                epoch=epoch,
                sample_idx=idx,
                run_id=run_id,
                reason=reason,
            )
    
    # Save summary figure
    plt.tight_layout()
    viz_dir = output_dir / "visualizations"
    if run_id:
        viz_dir = viz_dir / run_id
    if reason:
        viz_dir = viz_dir / reason
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(viz_dir / f"epoch_{epoch}_phase{phase}.png", dpi=150, bbox_inches='tight')
    plt.close()
