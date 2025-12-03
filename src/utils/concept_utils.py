"""
Utility functions for concept head training and evaluation.
"""

import torch
import numpy as np
import json
from pathlib import Path
from typing import Dict, Tuple
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from PIL import Image
import pandas as pd


def load_all_concept_data(concept_data_dir: str, split: str = "train") -> Dict:
    """
    Load all concept-related data from disk.
    
    Args:
        concept_data_dir: Directory containing concept data files
        split: Split name ('train', 'val', 'test')
    
    Returns:
        Dictionary with keys:
            - 'vocabulary': dict with concept vocabulary info
            - 'embeddings': torch.Tensor [K, 768] concept text embeddings
            - 'priors': dict with class priors
            - 'zero_shot_scores': np.ndarray [N, K] zero-shot scores
    """
    concept_dir = Path(concept_data_dir)
    
    # Load vocabulary
    vocab_path = concept_dir / "concept_vocabulary.json"
    if not vocab_path.exists():
        raise FileNotFoundError(f"Concept vocabulary not found: {vocab_path}")
    
    with open(vocab_path, 'r') as f:
        vocab_data = json.load(f)
    
    # JSON converts integer keys to strings, so convert idx_to_concept keys back to int
    idx_to_concept_raw = vocab_data.get('idx_to_concept', {})
    vocabulary = vocab_data.copy()
    vocabulary['idx_to_concept'] = {int(k): v for k, v in idx_to_concept_raw.items()}
    
    # Load text embeddings
    embeddings_path = concept_dir / "concept_text_embeddings.pt"
    if not embeddings_path.exists():
        raise FileNotFoundError(f"Concept embeddings not found: {embeddings_path}")
    
    embeddings = torch.load(embeddings_path)
    
    # Load priors
    priors_path = concept_dir / "class_priors.json"
    if not priors_path.exists():
        raise FileNotFoundError(f"Class priors not found: {priors_path}")
    
    with open(priors_path, 'r') as f:
        priors_data = json.load(f)
    
    # Convert priors lists to numpy arrays (keep structure but convert values)
    priors_dict = {
        'priors': np.array(priors_data['priors']),
        'priors_annotated': np.array(priors_data.get('priors_annotated', [])),
        'priors_clip': np.array(priors_data.get('priors_clip', [])),
        'clip_percentile': priors_data.get('clip_percentile', 98.0)
    }
    
    # Load zero-shot scores for the specified split
    scores_path = concept_dir / f"{split}_zero_shot_scores.npy"
    if not scores_path.exists():
        raise FileNotFoundError(f"Zero-shot scores not found: {scores_path}")
    
    zero_shot_scores = np.load(scores_path)
    
    return {
        'vocabulary': vocabulary,
        'embeddings': embeddings,
        'priors': priors_dict,
        'zero_shot_scores': zero_shot_scores
    }


def compute_accuracy(probs: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute top-1 accuracy for concept prediction.
    
    Args:
        probs: Concept probabilities [N, K]
        labels: Ground truth concept indices [N]
    
    Returns:
        Accuracy (0-1)
    """
    # Get predicted concept (highest probability)
    preds = probs.argmax(dim=1)  # [N]
    
    # Compare with ground truth
    correct = (preds == labels).float()
    accuracy = correct.mean().item()
    
    return accuracy


def save_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    filepath: str
):
    """
    Save model checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss value
        metrics: Dictionary of metrics
        filepath: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, filepath)


def load_checkpoint(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    filepath: str,
    device: torch.device
) -> Tuple[int, Dict]:
    """
    Load model checkpoint.
    
    Args:
        model: Model to load state into
        optimizer: Optimizer to load state into
        filepath: Path to checkpoint file
        device: Device to load on
    
    Returns:
        Tuple of (epoch, metrics)
    """
    checkpoint = torch.load(filepath, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    epoch = checkpoint.get('epoch', 0)
    metrics = checkpoint.get('metrics', {})
    
    return epoch, metrics


def visualize_epoch_samples(
    model: torch.nn.Module,
    dataset,
    idx_to_concept: Dict[int, str],
    output_dir: Path,
    epoch: int,
    split: str,
    num_samples: int,
    device: torch.device,
    mode: str = "global"
):
    """
    Visualize sample predictions at end of epoch.
    
    Args:
        model: Trained concept head model
        dataset: ConceptDataset instance
        idx_to_concept: Mapping from concept index to name
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        split: Split name ('val' or 'test')
        num_samples: Number of samples to visualize
        device: Device to run inference on
        mode: Model mode ('global' or 'spatial')
    """
    model.eval()
    
    # Sample random indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    vis_dir = output_dir / f"epoch_{epoch:03d}" / split
    vis_dir.mkdir(parents=True, exist_ok=True)
    
    with torch.no_grad():
        for idx in indices:
            pooled, patch_tokens, label, coords = dataset[idx]
            pooled = pooled.unsqueeze(0).to(device)
            
            if mode == "spatial" and patch_tokens is not None:
                patch_tokens = patch_tokens.unsqueeze(0).to(device)
            else:
                patch_tokens = None
            
            # Get predictions
            probs, attention = model(pooled, patch_tokens, return_attention=(mode == "spatial"))
            probs = probs[0].cpu().numpy()  # [K]
            
            # Verify probs shape matches vocabulary
            if len(probs) != len(idx_to_concept):
                raise ValueError(
                    f"Model output dimension ({len(probs)}) != vocabulary size ({len(idx_to_concept)}). "
                    f"Model and vocabulary are out of sync."
                )
            
            # Get top-5 concepts
            top5_indices = np.argsort(probs)[::-1][:5]
            top5_probs = probs[top5_indices]
            
            # Verify indices are valid
            for i in top5_indices:
                if int(i) not in idx_to_concept:
                    raise KeyError(
                        f"Concept index {int(i)} not found in vocabulary. "
                        f"Vocabulary has {len(idx_to_concept)} concepts (indices 0-{len(idx_to_concept)-1}). "
                        f"Model may have wrong number of output concepts."
                    )
            
            top5_concepts = [idx_to_concept[int(i)] for i in top5_indices]
            
            # Ground truth
            gt_idx = int(label.item())
            if gt_idx not in idx_to_concept:
                raise KeyError(
                    f"Ground truth concept index {gt_idx} not found in vocabulary. "
                    f"Vocabulary has {len(idx_to_concept)} concepts (indices 0-{len(idx_to_concept)-1}). "
                    f"Dataset and vocabulary may be out of sync."
                )
            gt_concept = idx_to_concept[gt_idx]
            gt_idx_in_top5 = None
            if gt_concept in top5_concepts:
                gt_idx_in_top5 = top5_concepts.index(gt_concept)
            
            # Load image from dataset
            row = dataset.df.iloc[idx]
            image_path = row['image_path']
            try:
                image = Image.open(image_path).convert('RGB')
            except Exception as e:
                print(f"Warning: Could not load image {image_path}: {e}")
                image = None
            
            # Create visualization with image and bar chart
            fig = plt.figure(figsize=(14, 8))
            gs = GridSpec(1, 2, figure=fig, width_ratios=[1, 1.2])
            
            # Left: Image
            ax_img = fig.add_subplot(gs[0])
            if image is not None:
                ax_img.imshow(image)
                ax_img.axis('off')
                ax_img.set_title(f'Sample {idx}\nGT: {gt_concept}', fontsize=10)
            else:
                ax_img.text(0.5, 0.5, f'Image not found\n{image_path}', 
                           ha='center', va='center', transform=ax_img.transAxes)
                ax_img.axis('off')
            
            # Right: Bar chart
            ax_bar = fig.add_subplot(gs[1])
            colors = ['green' if i == gt_idx_in_top5 else 'blue' for i in range(len(top5_concepts))]
            bars = ax_bar.barh(range(len(top5_concepts)), top5_probs, color=colors)
            
            # Labels
            ax_bar.set_yticks(range(len(top5_concepts)))
            ax_bar.set_yticklabels([f"{name[:40]}..." if len(name) > 40 else name for name in top5_concepts], fontsize=9)
            ax_bar.set_xlabel('Probability', fontsize=10)
            ax_bar.set_title(f'Epoch {epoch} - {split.upper()} - Top 5 Concepts', fontsize=11)
            ax_bar.set_xlim(0, 1)
            
            # Add probability values and match indicator
            for i, (bar, prob, concept) in enumerate(zip(bars, top5_probs, top5_concepts)):
                ax_bar.text(prob + 0.01, i, f'{prob:.3f}', va='center', fontsize=9)
                if concept == gt_concept:
                    ax_bar.text(-0.02, i, '✓', va='center', ha='right', fontsize=12, color='green', weight='bold')
            
            plt.tight_layout()
            plt.savefig(vis_dir / f"sample_{idx}.png", dpi=150, bbox_inches='tight')
            plt.close()
    
    print(f"  Saved {len(indices)} visualizations to {vis_dir}")

