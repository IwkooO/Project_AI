"""
Utility functions for concept head training.

Includes:
    - Loading concept vocabulary and embeddings
    - Loading class priors and zero-shot scores
    - Visualization helpers for attention heatmaps
"""

import json
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors


def load_concept_vocabulary(vocab_path: str) -> Dict:
    """
    Load concept vocabulary from JSON file.
    
    Args:
        vocab_path: Path to concept_vocabulary.json
    
    Returns:
        Dictionary containing:
            - concepts: List of concept names
            - concept_to_idx: Mapping from name to index
            - idx_to_concept: Mapping from index to name
            - num_concepts: Number of concepts K
            - concept_texts: Formatted text descriptions
    """
    with open(vocab_path, 'r') as f:
        vocab = json.load(f)
    
    # Convert idx_to_concept keys to integers
    vocab['idx_to_concept'] = {int(k): v for k, v in vocab['idx_to_concept'].items()}
    
    return vocab


def load_concept_embeddings(embeddings_path: str) -> torch.Tensor:
    """
    Load concept text embeddings.
    
    Args:
        embeddings_path: Path to concept_text_embeddings.pt
    
    Returns:
        Text embeddings V ∈ R^(K×768)
    """
    return torch.load(embeddings_path)


def load_class_priors(priors_path: str) -> Dict:
    """
    Load class priors from JSON file.
    
    Args:
        priors_path: Path to class_priors.json
    
    Returns:
        Dictionary containing:
            - priors: Final class priors π_k
            - priors_annotated: Annotated frequency priors
            - priors_clip: CLIP-based priors
            - clip_percentile: Threshold percentile used
    """
    with open(priors_path, 'r') as f:
        priors = json.load(f)
    
    # Convert to numpy arrays
    priors['priors'] = np.array(priors['priors'])
    priors['priors_annotated'] = np.array(priors['priors_annotated'])
    priors['priors_clip'] = np.array(priors['priors_clip'])
    
    return priors


def load_zero_shot_scores(scores_path: str) -> np.ndarray:
    """
    Load zero-shot similarity scores.
    
    Args:
        scores_path: Path to {split}_zero_shot_scores.npy
    
    Returns:
        Zero-shot scores z_k(x) ∈ R^(N×K)
    """
    return np.load(scores_path)


def load_all_concept_data(concept_data_dir: str, split: str = "train") -> Dict:
    """
    Load all concept data from a directory.
    
    Args:
        concept_data_dir: Directory containing concept data files
        split: Split name for zero-shot scores
    
    Returns:
        Dictionary with all concept data
    """
    data_dir = Path(concept_data_dir)
    
    vocab = load_concept_vocabulary(data_dir / "concept_vocabulary.json")
    embeddings = load_concept_embeddings(data_dir / "concept_text_embeddings.pt")
    priors = load_class_priors(data_dir / "class_priors.json")
    
    # Load zero-shot scores if available
    scores_path = data_dir / f"{split}_zero_shot_scores.npy"
    zero_shot_scores = None
    if scores_path.exists():
        zero_shot_scores = load_zero_shot_scores(scores_path)
    
    return {
        'vocabulary': vocab,
        'embeddings': embeddings,
        'priors': priors,
        'zero_shot_scores': zero_shot_scores
    }


def visualize_attention_heatmap(
    heatmap: np.ndarray,
    concept_name: str,
    image: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 5)
):
    """
    Visualize attention heatmap for a concept.
    
    Args:
        heatmap: Attention weights ∈ R^(H×W)
        concept_name: Name of the concept
        image: (optional) Original image to overlay
        save_path: (optional) Path to save figure
        figsize: Figure size
    """
    fig, axes = plt.subplots(1, 2 if image is not None else 1, figsize=figsize)
    
    if image is not None:
        axes = axes if isinstance(axes, np.ndarray) else [axes]
        
        # Show original image
        axes[0].imshow(image)
        axes[0].set_title("Original Image")
        axes[0].axis('off')
        
        # Show heatmap
        im = axes[1].imshow(heatmap, cmap='hot', interpolation='bilinear')
        axes[1].set_title(f"Attention: {concept_name}")
        axes[1].axis('off')
        plt.colorbar(im, ax=axes[1])
    else:
        ax = axes if not isinstance(axes, np.ndarray) else axes[0]
        im = ax.imshow(heatmap, cmap='hot', interpolation='bilinear')
        ax.set_title(f"Attention: {concept_name}")
        ax.axis('off')
        plt.colorbar(im, ax=ax)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_top_concepts(
    probs: np.ndarray,
    idx_to_concept: Dict[int, str],
    top_k: int = 10,
    save_path: Optional[str] = None
):
    """
    Visualize top predicted concepts for an image.
    
    Args:
        probs: Concept probabilities ∈ R^K
        idx_to_concept: Mapping from index to concept name
        top_k: Number of top concepts to show
        save_path: (optional) Path to save figure
    """
    # Get top k indices
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]
    top_names = [idx_to_concept[idx] for idx in top_indices]
    
    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    y_pos = np.arange(len(top_names))
    
    ax.barh(y_pos, top_probs, align='center')
    ax.set_yticks(y_pos)
    ax.set_yticklabels(top_names)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_title(f'Top {top_k} Predicted Concepts')
    ax.set_xlim(0, 1)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def compute_accuracy(
    probs: torch.Tensor,
    labels: torch.Tensor
) -> float:
    """
    Compute top-1 accuracy for concept prediction.
    
    Args:
        probs: Predicted probabilities ∈ R^(N×K)
        labels: Ground truth labels ∈ R^N (single positive per sample)
    
    Returns:
        accuracy: Top-1 accuracy
    """
    probs_np = probs.detach().cpu().numpy()
    labels_np = labels.detach().cpu().numpy()
    
    predictions = np.argmax(probs_np, axis=1)
    accuracy = np.mean(predictions == labels_np)
    
    return accuracy


def visualize_sample_predictions(
    image_path: str,
    probs: np.ndarray,
    gt_label: int,
    idx_to_concept: Dict[int, str],
    attention_heatmap: Optional[np.ndarray] = None,
    save_path: Optional[str] = None,
    top_k: int = 5
):
    """
    Visualize sample prediction with image, top-5 concepts, and attention map.
    
    Shows the original image alongside top-5 predicted concepts with their scores.
    The ground truth concept is highlighted in green if it's in top-5.
    
    Args:
        image_path: Path to the original image
        probs: Concept probabilities ∈ R^K
        gt_label: Ground truth concept index
        idx_to_concept: Mapping from index to concept name
        attention_heatmap: (optional) Attention map for top concept ∈ R^(H×W)
        save_path: (optional) Path to save figure
        top_k: Number of top concepts to show (default 5)
    """
    from PIL import Image
    
    # Load image
    try:
        img = Image.open(image_path).convert('RGB')
        img_np = np.array(img)
    except Exception as e:
        print(f"Could not load image {image_path}: {e}")
        img_np = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Get top-k indices and probabilities
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]
    top_names = [idx_to_concept.get(int(idx), f"Concept_{idx}") for idx in top_indices]
    
    # Get GT concept name
    gt_name = idx_to_concept.get(gt_label, f"Concept_{gt_label}")
    gt_prob = probs[gt_label]
    
    # Check if GT is in top-k
    gt_in_topk = gt_label in top_indices
    gt_rank = np.where(top_indices == gt_label)[0][0] if gt_in_topk else -1
    
    # Create figure
    n_cols = 3 if attention_heatmap is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # Panel 1: Original image
    axes[0].imshow(img_np)
    axes[0].set_title(f"GT: {gt_name[:40]}..." if len(gt_name) > 40 else f"GT: {gt_name}")
    axes[0].axis('off')
    
    # Panel 2: Top-5 concepts bar chart
    ax = axes[1]
    y_pos = np.arange(top_k)
    colors = []
    for i, idx in enumerate(top_indices):
        if idx == gt_label:
            colors.append('#2ecc71')  # Green for GT
        else:
            colors.append('#3498db')  # Blue for others
    
    bars = ax.barh(y_pos, top_probs, color=colors, edgecolor='black', linewidth=1)
    ax.set_yticks(y_pos)
    
    # Truncate long names
    display_names = []
    for name in top_names:
        if len(name) > 30:
            display_names.append(name[:27] + "...")
        else:
            display_names.append(name)
    
    ax.set_yticklabels(display_names, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel('Probability')
    ax.set_xlim(0, 1)
    
    # Add probability values on bars
    for i, (bar, prob) in enumerate(zip(bars, top_probs)):
        ax.text(prob + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{prob:.3f}', va='center', fontsize=9)
    
    # Title with match indicator
    if gt_in_topk:
        ax.set_title(f'Top {top_k} Predictions (✓ GT @ rank {gt_rank + 1})', color='green')
    else:
        ax.set_title(f'Top {top_k} Predictions (✗ GT prob: {gt_prob:.3f})', color='red')
    
    # Panel 3: Attention heatmap (if provided)
    if attention_heatmap is not None:
        ax = axes[2]
        im = ax.imshow(attention_heatmap, cmap='hot', interpolation='bilinear')
        ax.set_title(f'Attention: {display_names[0]}')
        ax.axis('off')
        plt.colorbar(im, ax=ax, fraction=0.046)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


def visualize_epoch_samples(
    model,
    dataset,
    idx_to_concept: Dict[int, str],
    output_dir: Path,
    epoch: int,
    split: str = "val",
    num_samples: int = 5,
    device: torch.device = torch.device('cpu')
):
    """
    Visualize sample predictions at the end of an epoch.
    
    Supports both global and spatial modes.
    
    Args:
        model: Concept head model (ConceptHead with .mode attribute)
        dataset: Dataset to sample from (returns pooled, patch_tokens, label, coords)
        idx_to_concept: Mapping from index to concept name
        output_dir: Directory to save visualizations
        epoch: Current epoch number
        split: Split name ('val' or 'test')
        num_samples: Number of samples to visualize
        device: Device to use
    """
    model.eval()
    mode = getattr(model, 'mode', 'global')
    
    # Create epoch directory
    epoch_dir = output_dir / f"epoch_{epoch:03d}" / split
    epoch_dir.mkdir(parents=True, exist_ok=True)
    
    # Sample random indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            # Dataset returns: pooled, patch_tokens, label, coords
            pooled, patch_tokens, label, coords = dataset[idx]
            pooled = pooled.unsqueeze(0).to(device)
            
            if mode == "spatial" and patch_tokens is not None:
                patch_tokens = patch_tokens.unsqueeze(0).to(device)
            else:
                patch_tokens = None
            
            # Get predictions
            probs, _ = model(pooled, patch_tokens, return_attention=True)
            probs_np = probs[0].cpu().numpy()
            label_int = label.item()
            
            # Get attention heatmap for spatial mode only
            attention_heatmap = None
            if mode == "spatial" and patch_tokens is not None:
                try:
                    heatmaps = model.get_attention_heatmaps(patch_tokens)  # [1, K, H, W]
                    top_idx = probs_np.argmax()
                    attention_heatmap = heatmaps[0, top_idx].cpu().numpy()
                except Exception as e:
                    print(f"Warning: Could not get attention heatmap: {e}")
            
            # Get image path from dataset
            image_path = dataset.df.iloc[idx]['image_path']
            
            # Visualize
            save_path = epoch_dir / f"sample_{i:02d}.png"
            visualize_sample_predictions(
                image_path=image_path,
                probs=probs_np,
                gt_label=label_int,
                idx_to_concept=idx_to_concept,
                attention_heatmap=attention_heatmap,
                save_path=str(save_path),
                top_k=5
            )
    
    print(f"  Saved {num_samples} {split} visualizations to {epoch_dir}")


def save_checkpoint(
    model,
    optimizer,
    epoch: int,
    loss: float,
    metrics: Dict,
    save_path: str
):
    """
    Save training checkpoint.
    
    Args:
        model: Model to save
        optimizer: Optimizer state
        epoch: Current epoch
        loss: Current loss
        metrics: Current metrics
        save_path: Path to save checkpoint
    """
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
        'metrics': metrics
    }
    torch.save(checkpoint, save_path)
    print(f"Saved checkpoint to {save_path}")


def load_checkpoint(
    model,
    optimizer,
    checkpoint_path: str,
    device: torch.device
) -> Tuple[int, float]:
    """
    Load training checkpoint.
    
    Args:
        model: Model to load into
        optimizer: Optimizer to load into
        checkpoint_path: Path to checkpoint
        device: Device to load to
    
    Returns:
        epoch: Epoch to resume from
        loss: Loss at checkpoint
    """
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    return checkpoint['epoch'], checkpoint['loss']

