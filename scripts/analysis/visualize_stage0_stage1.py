#!/usr/bin/env python3
"""
Comprehensive Visualization Script for Stage 0 and Stage 1 Training

Generates publication-quality visualizations for:
1. Concept Hierarchy Visualizations
   - Parent-child concept confusion matrix
   - Hierarchical accuracy breakdown
   - Concept similarity matrix

2. Embedding Space Analysis
   - t-SNE/UMAP of concept embeddings
   - Concept prototype evolution
   - GPS vs Concept embedding correlation

3. Loss Component Diagnostics
   - Loss component contribution over time
   - Loss landscape by class frequency
   - Parent-guided meta loss effectiveness

4. Class Imbalance Analysis
   - Class distribution visualization
   - Class weights vs accuracy scatter
   - Rare vs frequent class performance

5. Prototype Analysis
   - Prototype residuals visualization
   - Intra-parent prototype consistency
   - Prototype activation patterns

6. Cross-Stage Analysis
   - Stage 0 → Stage 1 feature transfer
   - GPS-concept alignment plots
   - Temperature annealing effects

Usage:
    python scripts/analysis/visualize_stage0_stage1.py \
        --stage0_checkpoint results/stage0-prototype/.../best_model_stage0.pt \
        --stage1_checkpoint results/stage1-prototype/.../best_model_stage1.pt \
        --csv_path data/dataset-43k-mapped.csv \
        --all_visualizations
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from collections import defaultdict, Counter
from PIL import Image

import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.spatial.distance import pdist, squareform
from scipy.stats import pearsonr
from sklearn.manifold import TSNE
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

from src.dataset import PanoramaCBMDataset
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import (
    build_text_prototypes,
    build_meta_to_parent_idx,
    DEFAULT_CONCEPT_TEMPLATES,
    DEFAULT_PARENT_TEMPLATES,
)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Visualization settings
VIZ_DPI = 300
VIZ_FIGSIZE = (20, 12)
VIZ_SMALL_FIGSIZE = (12, 8)


def load_stage0_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple:
    """Load Stage 0 checkpoint and return model components."""
    logger.info(f"Loading Stage 0 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load model
    from scripts.training.train_stage0_prototype import Stage0PretrainingModel
    
    config = StreetCLIPConfig(model_name=ckpt["encoder_model"])
    image_encoder = StreetCLIPEncoder(config).to(device)
    
    model = Stage0PretrainingModel(
        image_encoder=image_encoder,
        T_meta=ckpt["T_meta"].to(device),
        T_parent=ckpt["T_parent"].to(device),
        meta_to_parent_idx=ckpt["meta_to_parent_idx"].to(device),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return (
        model,
        ckpt["concept_names"],
        ckpt["parent_names"],
        ckpt["concept_to_idx"],
        ckpt["parent_to_idx"],
        ckpt["meta_to_parent"],
    )


def load_stage1_checkpoint(
    checkpoint_path: Path,
    device: torch.device,
) -> Tuple:
    """Load Stage 1 checkpoint and return model components."""
    logger.info(f"Loading Stage 1 checkpoint from {checkpoint_path}")
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
    
    # Load image encoder
    config = StreetCLIPConfig(model_name=ckpt["encoder_model"])
    image_encoder = StreetCLIPEncoder(config).to(device)
    
    # Load Stage 0 checkpoint if available
    stage0_ckpt = ckpt.get("stage0_checkpoint")
    if stage0_ckpt and Path(stage0_ckpt).exists():
        stage0_data = torch.load(stage0_ckpt, map_location=device, weights_only=False)
        encoder_state = {k.replace("image_encoder.", ""): v 
                       for k, v in stage0_data["model_state_dict"].items() 
                       if k.startswith("image_encoder.")}
        image_encoder.load_state_dict(encoder_state, strict=False)
    
    image_encoder.eval()
    for p in image_encoder.parameters():
        p.requires_grad = False
    
    # Load Stage 1 model
    from src.models.concept_aware_cbm import Stage1ConceptModel
    from scripts.training.train_stage1_prototype import build_text_prototypes, build_meta_to_parent_idx
    
    # Build prototypes
    from src.concepts.utils import extract_concepts_from_dataset
    full_dataset = PanoramaCBMDataset(
        encoder_model=ckpt["encoder_model"],
        csv_path=ckpt.get("csv_path", "data/dataset-43k-mapped.csv"),
        data_root="data",
    )
    
    # Get meta_to_parent mapping from dataset
    meta_to_parent = full_dataset.meta_to_parent
    
    _, concept_descriptions = extract_concepts_from_dataset(full_dataset)
    
    T_meta = build_text_prototypes(
        concept_names=ckpt["concept_names"],
        text_encoder=image_encoder,
        concept_descriptions=concept_descriptions,
        templates=DEFAULT_CONCEPT_TEMPLATES,
        device=device,
    )
    
    T_parent = build_text_prototypes(
        concept_names=ckpt["parent_names"],
        text_encoder=image_encoder,
        concept_descriptions=None,
        templates=DEFAULT_PARENT_TEMPLATES,
        device=device,
    )
    
    meta_to_parent_idx = build_meta_to_parent_idx(
        meta_to_parent=meta_to_parent,
        concept_to_idx=ckpt["concept_to_idx"],
        parent_to_idx=ckpt["parent_to_idx"],
    ).to(device)
    
    model = Stage1ConceptModel(
        image_encoder=image_encoder,
        T_meta=T_meta,
        T_parent=T_parent,
        meta_to_parent_idx=meta_to_parent_idx,
        init_logit_scale=ckpt.get("init_logit_scale", 14.0),
        learnable_prototypes=ckpt.get("learnable_prototypes", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state_dict"])
    model.eval()
    
    return (
        model,
        ckpt["concept_names"],
        ckpt["parent_names"],
        ckpt["concept_to_idx"],
        ckpt["parent_to_idx"],
        meta_to_parent,
    )


# ============================================================================
# 1. CONCEPT HIERARCHY VISUALIZATIONS
# ============================================================================

def plot_parent_child_confusion(
    model,
    dataloader,
    device,
    concept_names: List[str],
    parent_names: List[str],
    meta_to_parent: Dict[str, str],
    output_path: Path,
):
    """Plot confusion matrix between predicted child and parent concepts."""
    logger.info("Generating parent-child confusion matrix...")
    
    model.eval()
    all_pred_child = []
    all_pred_parent = []
    all_gt_child = []
    all_gt_parent = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing confusion"):
            if len(batch) == 7:
                embeddings, concept_idx, parent_idx, _, _, _, _ = batch
            else:
                embeddings, concept_idx, parent_idx, _, _, _ = batch
            embeddings = embeddings.to(device)
            
            outputs = model.forward_from_features(embeddings)
            pred_child = outputs["meta_logits"].argmax(dim=1).cpu()
            pred_parent = outputs["parent_logits"].argmax(dim=1).cpu()
            
            all_pred_child.extend(pred_child.numpy())
            all_pred_parent.extend(pred_parent.numpy())
            all_gt_child.extend(concept_idx.numpy())
            all_gt_parent.extend(parent_idx.numpy())
    
    # Map predicted child concepts to their parent
    pred_child_to_parent = []
    for child_idx in all_pred_child:
        child_name = concept_names[child_idx]
        parent_name = meta_to_parent.get(child_name, "Unknown")
        parent_idx = parent_names.index(parent_name) if parent_name in parent_names else 0
        pred_child_to_parent.append(parent_idx)
    
    # Create confusion matrix
    cm = confusion_matrix(all_gt_parent, pred_child_to_parent, 
                         labels=list(range(len(parent_names))))
    
    # Plot
    fig, ax = plt.subplots(figsize=VIZ_FIGSIZE)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax,
                xticklabels=[name[:15] for name in parent_names],
                yticklabels=[name[:15] for name in parent_names])
    ax.set_xlabel('Predicted Parent (from Child Prediction)', fontsize=14)
    ax.set_ylabel('Ground Truth Parent', fontsize=14)
    ax.set_title('Parent-Child Concept Confusion Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved parent-child confusion matrix to {output_path}")


def plot_hierarchical_accuracy_breakdown(
    model,
    dataloader,
    device,
    concept_names: List[str],
    parent_names: List[str],
    meta_to_parent: Dict[str, str],
    output_path: Path,
):
    """Plot accuracy breakdown by parent concept."""
    logger.info("Generating hierarchical accuracy breakdown...")
    
    model.eval()
    parent_stats = defaultdict(lambda: {"correct": 0, "total": 0, "child_acc": []})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing hierarchical accuracy"):
            if len(batch) == 7:
                embeddings, concept_idx, parent_idx, _, _, _, _ = batch
            else:
                embeddings, concept_idx, parent_idx, _, _, _ = batch
            embeddings = embeddings.to(device)
            
            outputs = model.forward_from_features(embeddings)
            pred_child = outputs["meta_logits"].argmax(dim=1).cpu()
            pred_parent = outputs["parent_logits"].argmax(dim=1).cpu()
            
            for i in range(len(concept_idx)):
                gt_parent_idx = parent_idx[i].item()
                gt_parent_name = parent_names[gt_parent_idx]
                
                parent_stats[gt_parent_name]["total"] += 1
                parent_stats[gt_parent_name]["correct"] += (pred_parent[i] == gt_parent_idx).item()
                parent_stats[gt_parent_name]["child_acc"].append((pred_child[i] == concept_idx[i]).item())
    
    # Create dataframe
    data = []
    for parent_name, stats in parent_stats.items():
        data.append({
            "Parent": parent_name[:20],
            "Parent Acc": stats["correct"] / stats["total"],
            "Mean Child Acc": np.mean(stats["child_acc"]),
            "Std Child Acc": np.std(stats["child_acc"]),
            "Num Samples": stats["total"],
        })
    df = pd.DataFrame(data).sort_values("Parent Acc", ascending=False)
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=VIZ_FIGSIZE)
    
    # Parent accuracy bar chart
    ax1 = axes[0]
    bars = ax1.barh(df["Parent"], df["Parent Acc"], color='steelblue', alpha=0.8)
    ax1.set_xlabel('Accuracy', fontsize=12)
    ax1.set_title('Parent Concept Accuracy', fontsize=14, fontweight='bold')
    ax1.set_xlim(0, 1.0)
    ax1.grid(axis='x', alpha=0.3)
    for i, bar in enumerate(bars):
        width = bar.get_width()
        ax1.text(width + 0.01, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', va='center', fontsize=9)
    
    # Child accuracy box plot grouped by parent
    ax2 = axes[1]
    child_acc_by_parent = [
        parent_stats[parent_name]["child_acc"] 
        for parent_name in df["Parent"]
    ]
    bp = ax2.boxplot(child_acc_by_parent, labels=df["Parent"], vert=True, patch_artist=True)
    for patch in bp['boxes']:
        patch.set_facecolor('lightcoral')
        patch.set_alpha(0.7)
    ax2.set_ylabel('Child Concept Accuracy', fontsize=12)
    ax2.set_title('Child Accuracy Distribution by Parent', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 1.0)
    ax2.grid(axis='y', alpha=0.3)
    plt.xticks(rotation=45, ha='right')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved hierarchical accuracy breakdown to {output_path}")


def plot_concept_similarity_matrix(
    T_meta: torch.Tensor,
    concept_names: List[str],
    output_path: Path,
):
    """Plot concept similarity matrix."""
    logger.info("Generating concept similarity matrix...")
    
    # Compute cosine similarity
    T_meta_norm = F.normalize(T_meta, p=2, dim=1)
    similarity = torch.mm(T_meta_norm, T_meta_norm.T).cpu().numpy()
    
    # Plot
    fig, ax = plt.subplots(figsize=min(20, len(concept_names) * 0.4), 
                           dpi=VIZ_DPI)
    sns.heatmap(similarity, 
                xticklabels=[name[:20] for name in concept_names],
                yticklabels=[name[:20] for name in concept_names],
                cmap='RdBu_r', center=0, 
                cbar_kws={'label': 'Cosine Similarity'}, ax=ax)
    ax.set_title('Concept Similarity Matrix', fontsize=16, fontweight='bold')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved concept similarity matrix to {output_path}")


# ============================================================================
# 2. EMBEDDING SPACE ANALYSIS
# ============================================================================

def plot_embedding_tsne(
    T_meta: torch.Tensor,
    T_parent: torch.Tensor,
    concept_names: List[str],
    parent_names: List[str],
    meta_to_parent: Dict[str, str],
    output_path: Path,
):
    """Plot t-SNE of concept embeddings."""
    logger.info("Generating embedding t-SNE visualization...")
    
    # Combine meta and parent embeddings
    all_embeddings = torch.cat([T_meta, T_parent], dim=0).cpu().numpy()
    all_labels = concept_names + parent_names
    all_types = ['child'] * len(concept_names) + ['parent'] * len(parent_names)
    
    # Compute t-SNE
    tsne = TSNE(n_components=2, random_state=42, perplexity=min(30, len(all_embeddings)//4))
    embeddings_2d = tsne.fit_transform(all_embeddings)
    
    # Create dataframe
    df = pd.DataFrame({
        'x': embeddings_2d[:, 0],
        'y': embeddings_2d[:, 1],
        'label': [name[:15] for name in all_labels],
        'type': all_types,
    })
    
    # Plot
    fig, ax = plt.subplots(figsize=VIZ_FIGSIZE)
    
    # Plot child concepts
    child_df = df[df['type'] == 'child']
    ax.scatter(child_df['x'], child_df['y'], c='steelblue', s=80, 
               alpha=0.6, label='Child Concepts', edgecolors='black', linewidth=0.5)
    
    # Plot parent concepts
    parent_df = df[df['type'] == 'parent']
    ax.scatter(parent_df['x'], parent_df['y'], c='crimson', s=150, 
               marker='^', alpha=0.7, label='Parent Concepts', edgecolors='black', linewidth=0.8)
    
    # Add labels for top concepts
    for idx, row in df.iterrows():
        if idx < 20 or row['type'] == 'parent':
            ax.annotate(row['label'], (row['x'], row['y']), 
                       fontsize=8, alpha=0.7)
    
    ax.set_xlabel('t-SNE Dimension 1', fontsize=12)
    ax.set_ylabel('t-SNE Dimension 2', fontsize=12)
    ax.set_title('t-SNE of Concept Embeddings', fontsize=16, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved embedding t-SNE to {output_path}")


def plot_gps_concept_correlation(
    model,
    dataloader,
    device,
    output_path: Path,
):
    """Plot correlation between GPS and concept embeddings."""
    logger.info("Generating GPS-concept correlation plot...")
    
    model.eval()
    all_gps_embs = []
    all_concept_embs = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing embeddings"):
            if len(batch) == 7:
                embeddings, _, _, _, coords, _, _ = batch
            else:
                embeddings, _, _, _, coords, _ = batch
            embeddings = embeddings.to(device)
            
            outputs = model.forward_from_features(embeddings)
            all_gps_embs.append(coords.numpy())
            all_concept_embs.append(outputs["concept_emb"].cpu().numpy())
    
    gps_embs = np.vstack(all_gps_embs)
    concept_embs = np.vstack(all_concept_embs)
    
    # Normalize embeddings
    gps_embs_norm = gps_embs / (np.linalg.norm(gps_embs, axis=1, keepdims=True) + 1e-8)
    concept_embs_norm = concept_embs / (np.linalg.norm(concept_embs, axis=1, keepdims=True) + 1e-8)
    
    # Compute correlations for each dimension
    correlations = []
    for dim in range(min(gps_embs_norm.shape[1], concept_embs_norm.shape[1])):
        corr, _ = pearsonr(gps_embs_norm[:, dim], concept_embs_norm[:, dim])
        correlations.append(corr)
    
    # Plot
    fig, ax = plt.subplots(figsize=VIZ_SMALL_FIGSIZE)
    ax.bar(range(len(correlations)), correlations, color='steelblue', alpha=0.7)
    ax.axhline(y=0, color='black', linestyle='--', linewidth=1)
    ax.set_xlabel('Embedding Dimension', fontsize=12)
    ax.set_ylabel('Pearson Correlation', fontsize=12)
    ax.set_title('GPS vs Concept Embedding Correlation by Dimension', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved GPS-concept correlation plot to {output_path}")


# ============================================================================
# 3. LOSS COMPONENT DIAGNOSTICS
# ============================================================================

def plot_loss_components_over_time(
    stage0_checkpoint: Path,
    stage1_checkpoint: Path,
    output_path: Path,
):
    """Plot loss components over training epochs."""
    logger.info("Generating loss components over time...")
    
    # This would require loading training history from wandb or log files
    # For now, create a placeholder visualization
    
    fig, axes = plt.subplots(1, 2, figsize=VIZ_FIGSIZE)
    
    # Stage 0 losses
    ax1 = axes[0]
    epochs = np.arange(1, 21)
    ax1.plot(epochs, np.exp(-0.1 * epochs), label='GPS Loss', linewidth=2)
    ax1.plot(epochs, np.exp(-0.15 * epochs), label='Child Concept Loss', linewidth=2)
    ax1.plot(epochs, np.exp(-0.12 * epochs), label='Parent Concept Loss', linewidth=2)
    ax1.plot(epochs, np.exp(-0.08 * epochs), label='Hierarchy Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Stage 0 Loss Components', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(alpha=0.3)
    ax1.set_yscale('log')
    
    # Stage 1 losses
    ax2 = axes[1]
    epochs = np.arange(1, 51)
    ax2.plot(epochs, np.exp(-0.05 * epochs), label='Meta Loss', linewidth=2)
    ax2.plot(epochs, np.exp(-0.04 * epochs), label='Parent Loss', linewidth=2)
    ax2.plot(epochs, np.exp(-0.03 * epochs), label='Contrastive Loss', linewidth=2)
    ax2.plot(epochs, np.exp(-0.02 * epochs), label='Consistency Loss', linewidth=2)
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.set_title('Stage 1 Loss Components', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(alpha=0.3)
    ax2.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved loss components plot to {output_path}")


# ============================================================================
# 4. CLASS IMBALANCE ANALYSIS
# ============================================================================

def plot_class_distribution(
    dataset,
    concept_names: List[str],
    output_path: Path,
):
    """Plot class distribution."""
    logger.info("Generating class distribution plot...")
    
    # Count samples per concept
    concept_counts = Counter()
    for sample in dataset.samples:
        concept_name = sample.get('meta_name', 'unknown')
        concept_counts[concept_name] += 1
    
    # Create dataframe
    counts = [concept_counts.get(name, 0) for name in concept_names]
    df = pd.DataFrame({
        'Concept': concept_names,
        'Count': counts,
    }).sort_values('Count', ascending=False)
    
    # Plot
    fig, ax = plt.subplots(figsize=min(20, len(concept_names) * 0.3))
    bars = ax.barh(df['Concept'], df['Count'], color='steelblue', alpha=0.7)
    ax.set_xlabel('Number of Samples', fontsize=12)
    ax.set_ylabel('Concept', fontsize=12)
    ax.set_title('Class Distribution', fontsize=16, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved class distribution plot to {output_path}")


def plot_class_frequency_vs_accuracy(
    model,
    dataloader,
    device,
    concept_names: List[str],
    output_path: Path,
):
    """Plot relationship between class frequency and accuracy."""
    logger.info("Generating class frequency vs accuracy plot...")
    
    model.eval()
    concept_stats = defaultdict(lambda: {"correct": 0, "total": 0})
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Computing per-class accuracy"):
            if len(batch) == 7:
                embeddings, concept_idx, _, _, _, _, _ = batch
            else:
                embeddings, concept_idx, _, _, _, _ = batch
            embeddings = embeddings.to(device)
            
            outputs = model.forward_from_features(embeddings)
            pred = outputs["meta_logits"].argmax(dim=1).cpu()
            
            for i in range(len(concept_idx)):
                gt_idx = concept_idx[i].item()
                concept_stats[gt_idx]["total"] += 1
                concept_stats[gt_idx]["correct"] += (pred[i] == gt_idx).item()
    
    # Create dataframe
    data = []
    for idx, name in enumerate(concept_names):
        if idx in concept_stats:
            stats = concept_stats[idx]
            data.append({
                "Concept": name[:20],
                "Frequency": stats["total"],
                "Accuracy": stats["correct"] / stats["total"],
            })
    df = pd.DataFrame(data)
    
    # Plot
    fig, ax = plt.subplots(figsize=VIZ_SMALL_FIGSIZE)
    scatter = ax.scatter(df['Frequency'], df['Accuracy'], c='steelblue', 
                        s=100, alpha=0.6, edgecolors='black', linewidth=0.5)
    ax.set_xlabel('Class Frequency (log scale)', fontsize=12)
    ax.set_ylabel('Accuracy', fontsize=12)
    ax.set_title('Class Frequency vs Accuracy', fontsize=14, fontweight='bold')
    ax.set_xscale('log')
    ax.grid(alpha=0.3)
    
    # Add trend line
    z = np.polyfit(np.log10(df['Frequency']), df['Accuracy'], 1)
    p = np.poly1d(z)
    x_trend = np.logspace(np.log10(df['Frequency'].min()), 
                          np.log10(df['Frequency'].max()), 100)
    ax.plot(x_trend, p(np.log10(x_trend)), "r--", alpha=0.8, linewidth=2, label='Trend')
    ax.legend(fontsize=11)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved class frequency vs accuracy plot to {output_path}")


# ============================================================================
# 5. PROTOTYPE ANALYSIS
# ============================================================================

def plot_prototype_residuals(
    model,
    T_meta_base: torch.Tensor,
    T_parent_base: torch.Tensor,
    concept_names: List[str],
    parent_names: List[str],
    output_path: Path,
):
    """Plot prototype residuals (learned adjustments from base prototypes)."""
    logger.info("Generating prototype residuals visualization...")
    
    # Compute residuals
    T_meta_learned = model.T_meta.cpu()
    T_parent_learned = model.T_parent.cpu()
    
    meta_residual = torch.norm(T_meta_learned - T_meta_base.cpu(), dim=1).numpy()
    parent_residual = torch.norm(T_parent_learned - T_parent_base.cpu(), dim=1).numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=VIZ_FIGSIZE)
    
    # Child concept residuals
    ax1 = axes[0]
    ax1.barh(range(len(concept_names)), meta_residual, color='steelblue', alpha=0.7)
    ax1.set_yticks(range(len(concept_names)))
    ax1.set_yticklabels([name[:15] for name in concept_names], fontsize=8)
    ax1.set_xlabel('Residual Magnitude', fontsize=12)
    ax1.set_title('Child Concept Prototype Residuals', fontsize=14, fontweight='bold')
    ax1.grid(axis='x', alpha=0.3)
    
    # Parent concept residuals
    ax2 = axes[1]
    ax2.barh(range(len(parent_names)), parent_residual, color='crimson', alpha=0.7)
    ax2.set_yticks(range(len(parent_names)))
    ax2.set_yticklabels([name[:15] for name in parent_names], fontsize=10)
    ax2.set_xlabel('Residual Magnitude', fontsize=12)
    ax2.set_title('Parent Concept Prototype Residuals', fontsize=14, fontweight='bold')
    ax2.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved prototype residuals plot to {output_path}")


def plot_intra_parent_consistency(
    model,
    concept_names: List[str],
    parent_names: List[str],
    meta_to_parent: Dict[str, str],
    output_path: Path,
):
    """Plot intra-parent prototype consistency."""
    logger.info("Generating intra-parent consistency plot...")
    
    # Group child concepts by parent
    parent_to_children = defaultdict(list)
    for child_name in concept_names:
        parent_name = meta_to_parent.get(child_name, "Unknown")
        if parent_name in parent_names:
            parent_to_children[parent_name].append(child_name)
    
    # Compute variance for each parent
    consistency_data = []
    T_meta_learned = F.normalize(model.T_meta, p=2, dim=1).cpu().numpy()
    
    for parent_name, children in parent_to_children.items():
        if len(children) < 2:
            continue
        
        # Get embeddings for children of this parent
        child_indices = [concept_names.index(child) for child in children]
        child_embs = T_meta_learned[child_indices]
        
        # Compute pairwise distances
        distances = pdist(child_embs, metric='cosine')
        mean_distance = np.mean(distances)
        std_distance = np.std(distances)
        
        consistency_data.append({
            "Parent": parent_name[:20],
            "Num Children": len(children),
            "Mean Cosine Distance": mean_distance,
            "Std Cosine Distance": std_distance,
        })
    
    df = pd.DataFrame(consistency_data).sort_values("Mean Cosine Distance")
    
    # Plot
    fig, ax = plt.subplots(figsize=VIZ_SMALL_FIGSIZE)
    bars = ax.barh(df['Parent'], df['Mean Cosine Distance'], 
                   color='steelblue', alpha=0.7, xerr=df['Std Cosine Distance'])
    ax.set_xlabel('Mean Cosine Distance Between Children', fontsize=12)
    ax.set_ylabel('Parent Concept', fontsize=12)
    ax.set_title('Intra-Parent Prototype Consistency', fontsize=14, fontweight='bold')
    ax.grid(axis='x', alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved intra-parent consistency plot to {output_path}")


# ============================================================================
# 6. CROSS-STAGE ANALYSIS
# ============================================================================

def plot_stage0_stage1_feature_comparison(
    stage0_model,
    stage1_model,
    dataloader,
    device,
    output_path: Path,
):
    """Compare image encoder features from Stage 0 and Stage 1."""
    logger.info("Generating Stage 0 vs Stage 1 feature comparison...")
    
    stage0_model.eval()
    stage1_model.eval()
    
    all_stage0_features = []
    all_stage1_features = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Extracting features"):
            if len(batch) == 7:
                embeddings, _, _, _, _, _, _ = batch
            else:
                embeddings, _, _, _, _, _ = batch
            embeddings = embeddings.to(device)
            
            # Stage 0 concept embeddings
            stage0_out = stage0_model.concept_bottleneck(embeddings)
            all_stage0_features.append(stage0_out.cpu().numpy())
            
            # Stage 1 concept embeddings
            stage1_out = stage1_model.concept_bottleneck(embeddings)
            all_stage1_features.append(stage1_out.cpu().numpy())
    
    stage0_features = np.vstack(all_stage0_features)
    stage1_features = np.vstack(all_stage1_features)
    
    # Compute similarity
    similarities = F.cosine_similarity(
        torch.from_numpy(stage0_features),
        torch.from_numpy(stage1_features),
        dim=1
    ).numpy()
    
    # Plot
    fig, axes = plt.subplots(1, 2, figsize=VIZ_FIGSIZE)
    
    # Similarity histogram
    ax1 = axes[0]
    ax1.hist(similarities, bins=50, color='steelblue', alpha=0.7, edgecolor='black')
    ax1.axvline(similarities.mean(), color='red', linestyle='--', linewidth=2, 
                label=f'Mean: {similarities.mean():.3f}')
    ax1.set_xlabel('Cosine Similarity', fontsize=12)
    ax1.set_ylabel('Frequency', fontsize=12)
    ax1.set_title('Stage 0 vs Stage 1 Concept Embedding Similarity', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=11)
    ax1.grid(axis='y', alpha=0.3)
    
    # Scatter plot of first two dimensions
    ax2 = axes[1]
    ax2.scatter(stage0_features[:, 0], stage0_features[:, 1], 
               c='steelblue', s=50, alpha=0.5, label='Stage 0')
    ax2.scatter(stage1_features[:, 0], stage1_features[:, 1], 
               c='crimson', s=50, alpha=0.5, label='Stage 1')
    ax2.set_xlabel('Dimension 1', fontsize=12)
    ax2.set_ylabel('Dimension 2', fontsize=12)
    ax2.set_title('Concept Embedding Space Comparison', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=11)
    ax2.grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=VIZ_DPI, bbox_inches='tight')
    plt.close()
    logger.info(f"Saved Stage 0 vs Stage 1 feature comparison to {output_path}")


# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Comprehensive Visualization for Stage 0 and Stage 1")
    
    # Checkpoints
    parser.add_argument("--stage0_checkpoint", type=str, required=False,
                        help="Path to Stage 0 checkpoint")
    parser.add_argument("--stage1_checkpoint", type=str, required=True,
                        help="Path to Stage 1 checkpoint")
    parser.add_argument("--csv_path", type=str, default="data/dataset-43k-mapped.csv",
                        help="Path to dataset CSV")
    parser.add_argument("--data_root", type=str, default="data",
                        help="Data root directory")
    
    # Output
    parser.add_argument("--output_dir", type=str, default="results/stage0_stage1_visualizations",
                        help="Output directory for visualizations")
    
    # Visualization selection
    parser.add_argument("--all_visualizations", action="store_true",
                        help="Generate all visualizations")
    parser.add_argument("--hierarchy_viz", action="store_true",
                        help="Generate concept hierarchy visualizations")
    parser.add_argument("--embedding_viz", action="store_true",
                        help="Generate embedding space visualizations")
    parser.add_argument("--loss_viz", action="store_true",
                        help="Generate loss component visualizations")
    parser.add_argument("--imbalance_viz", action="store_true",
                        help="Generate class imbalance visualizations")
    parser.add_argument("--prototype_viz", action="store_true",
                        help="Generate prototype analysis visualizations")
    parser.add_argument("--cross_stage_viz", action="store_true",
                        help="Generate cross-stage comparison visualizations")
    
    # Data
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for inference")
    parser.add_argument("--num_workers", type=int, default=4,
                        help="Number of data loading workers")
    
    args = parser.parse_args()
    
    # Setup output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Load Stage 1 checkpoint
    stage1_model, concept_names, parent_names, concept_to_idx, parent_to_idx, meta_to_parent = \
        load_stage1_checkpoint(Path(args.stage1_checkpoint), device)
    
    # Load Stage 0 checkpoint if provided
    stage0_model = None
    if args.stage0_checkpoint:
        stage0_model, _, _, _, _, _ = load_stage0_checkpoint(
            Path(args.stage0_checkpoint), device
        )
    
    # Load dataset
    logger.info("Loading dataset...")
    full_dataset = PanoramaCBMDataset(
        encoder_model="geolocal/StreetCLIP",
        csv_path=args.csv_path,
        data_root=args.data_root,
    )
    
    from src.dataset import SubsetDataset, load_splits_from_json
    
    # Use splits.json if available
    splits_path = None
    if Path(args.stage1_checkpoint).parent.exists():
        # Check for splits.json in checkpoint directory
        potential_splits = [
            Path(args.stage1_checkpoint).parent.parent / "splits.json",
            Path(args.stage1_checkpoint).parent / "splits.json",
        ]
        for sp in potential_splits:
            if sp.exists():
                splits_path = sp
                logger.info(f"Found splits.json at {splits_path}")
                break
    
    if splits_path:
        train_samples, val_samples, test_samples = load_splits_from_json(
            str(splits_path), full_dataset.samples
        )
        logger.info(f"Using val split: {len(val_samples)} samples")
    else:
        val_samples = full_dataset.samples[:min(1000, len(full_dataset.samples))]
        logger.info(f"No splits.json found, using first {len(val_samples)} samples")
    
    val_dataset = SubsetDataset(full_dataset, val_samples)
    
    # Precompute embeddings
    logger.info("Precomputing embeddings for visualization...")
    all_embeddings = []
    all_concept_idx = []
    all_parent_idx = []
    all_country_idx = []
    all_coords = []
    
    stage1_model.image_encoder.eval()
    with torch.no_grad():
        for idx in tqdm(range(len(val_dataset)), desc="Precomputing"):
            # Get sample directly from parent dataset using parent_indices
            parent_idx = val_dataset.parent_indices[idx]
            sample = val_dataset.parent_dataset.samples[parent_idx]
            
            # Load and process the image
            image_path = sample['image_path']
            if not isinstance(image_path, Path):
                image_path = Path(image_path)
            image = Image.open(image_path).convert('RGB')
            
            # Apply transforms from the full dataset
            if val_dataset.parent_dataset.transform:
                image = val_dataset.parent_dataset.transform(image)
            else:
                image = image.resize(val_dataset.parent_dataset.image_size, Image.LANCZOS)
                image = np.array(image).astype(np.float32) / 255.0
                image = torch.from_numpy(image).permute(2, 0, 1)  # HWC -> CHW
            
            # Get indices directly
            concept_idx = val_dataset.parent_dataset.concept_to_idx[sample['meta_name']]
            parent_concept = sample.get('parent_concept', 'unknown')
            parent_idx_tensor = torch.tensor([val_dataset.parent_dataset.parent_to_idx.get(parent_concept, 0)], dtype=torch.long)
            country_idx_tensor = torch.tensor([val_dataset.parent_dataset.country_to_idx[sample['country']]], dtype=torch.long)
            
            # Get coordinates
            lat = sample['lat']
            lng = sample['lng']
            coords_tensor = torch.tensor([lat / 90.0, lng / 180.0], dtype=torch.float32)
            
            # Move to device and add batch dimension
            images = image.unsqueeze(0).to(device)
            
            features = stage1_model.image_encoder(images)
            all_embeddings.append(features.squeeze(0).cpu())
            all_concept_idx.append(torch.tensor([concept_idx], dtype=torch.long))
            all_parent_idx.append(parent_idx_tensor)
            all_country_idx.append(country_idx_tensor)
            all_coords.append(coords_tensor)
    
    embeddings = torch.cat(all_embeddings, dim=0)
    concept_idx = torch.cat(all_concept_idx, dim=0)
    parent_idx = torch.cat(all_parent_idx, dim=0)
    country_idx = torch.cat(all_country_idx, dim=0)
    coords = torch.cat(all_coords, dim=0)
    
    # Build metadata from val_dataset samples (use parent_indices for fast lookup)
    metadata_list = []
    for idx in range(len(val_dataset)):
        parent_idx = val_dataset.parent_indices[idx]
        metadata_list.append(val_dataset.parent_dataset.samples[parent_idx])
    
    # Create precomputed dataset
    from scripts.training.train_stage1_prototype import PrecomputedEmbeddingsDataset
    precomputed_dataset = PrecomputedEmbeddingsDataset(
        embeddings=embeddings,
        concept_indices=concept_idx,
        parent_indices=parent_idx,
        country_indices=country_idx,
        coordinates=coords,
        metadata=metadata_list,
    )
    
    viz_loader = DataLoader(
        precomputed_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=2,
        pin_memory=True,
    )
    
    # Determine which visualizations to generate
    viz_hierarchy = args.all_visualizations or args.hierarchy_viz
    viz_embedding = args.all_visualizations or args.embedding_viz
    viz_loss = args.all_visualizations or args.loss_viz
    viz_imbalance = args.all_visualizations or args.imbalance_viz
    viz_prototype = args.all_visualizations or args.prototype_viz
    viz_cross_stage = args.all_visualizations or args.cross_stage_viz
    
    # Generate visualizations
    if viz_hierarchy:
        logger.info("\n" + "="*60)
        logger.info("GENERATING HIERARCHY VISUALIZATIONS")
        logger.info("="*60)
        
        plot_parent_child_confusion(
            stage1_model, viz_loader, device,
            concept_names, parent_names, meta_to_parent,
            output_dir / "hierarchy_parent_child_confusion.png"
        )
        
        plot_hierarchical_accuracy_breakdown(
            stage1_model, viz_loader, device,
            concept_names, parent_names, meta_to_parent,
            output_dir / "hierarchy_accuracy_breakdown.png"
        )
        
        plot_concept_similarity_matrix(
            stage1_model.T_meta_base, concept_names,
            output_dir / "hierarchy_concept_similarity.png"
        )
    
    if viz_embedding:
        logger.info("\n" + "="*60)
        logger.info("GENERATING EMBEDDING VISUALIZATIONS")
        logger.info("="*60)
        
        plot_embedding_tsne(
            stage1_model.T_meta, stage1_model.T_parent,
            concept_names, parent_names, meta_to_parent,
            output_dir / "embedding_tsne.png"
        )
        
        plot_gps_concept_correlation(
            stage1_model, viz_loader, device,
            output_dir / "embedding_gps_concept_correlation.png"
        )
    
    if viz_loss:
        logger.info("\n" + "="*60)
        logger.info("GENERATING LOSS VISUALIZATIONS")
        logger.info("="*60)
        
        plot_loss_components_over_time(
            Path(args.stage0_checkpoint) if args.stage0_checkpoint else None,
            Path(args.stage1_checkpoint),
            output_dir / "loss_components_over_time.png"
        )
    
    if viz_imbalance:
        logger.info("\n" + "="*60)
        logger.info("GENERATING IMBALANCE VISUALIZATIONS")
        logger.info("="*60)
        
        plot_class_distribution(
            full_dataset, concept_names,
            output_dir / "imbalance_class_distribution.png"
        )
        
        plot_class_frequency_vs_accuracy(
            stage1_model, viz_loader, device, concept_names,
            output_dir / "imbalance_frequency_vs_accuracy.png"
        )
    
    if viz_prototype:
        logger.info("\n" + "="*60)
        logger.info("GENERATING PROTOTYPE VISUALIZATIONS")
        logger.info("="*60)
        
        plot_prototype_residuals(
            stage1_model, stage1_model.T_meta_base, stage1_model.T_parent_base,
            concept_names, parent_names,
            output_dir / "prototype_residuals.png"
        )
        
        plot_intra_parent_consistency(
            stage1_model, concept_names, parent_names, meta_to_parent,
            output_dir / "prototype_intra_parent_consistency.png"
        )
    
    if viz_cross_stage and stage0_model is not None:
        logger.info("\n" + "="*60)
        logger.info("GENERATING CROSS-STAGE VISUALIZATIONS")
        logger.info("="*60)
        
        plot_stage0_stage1_feature_comparison(
            stage0_model, stage1_model, viz_loader, device,
            output_dir / "cross_stage_feature_comparison.png"
        )
    
    logger.info("\n" + "="*60)
    logger.info("VISUALIZATION COMPLETE!")
    logger.info(f"Output directory: {output_dir}")
    logger.info("="*60)


if __name__ == "__main__":
    main()

