#!/usr/bin/env python3
"""
Evaluate trained Concept Head.

This script evaluates the concept head on a test set and generates:
    - Overall accuracy metric
    - Sample visualizations with top-5 concepts
    - Attention heatmaps (spatial mode only)

Supports both global (default) and spatial modes.

Usage:
    python scripts/evaluation/evaluate_concept_head.py \
        --checkpoint checkpoints/concept_head/best_concept_head.pth \
        --test-csv data/.../splits/dataset_test.csv \
        --cached-embeddings-dir data/.../cached_embeddings \
        --concept-data-dir data/.../concept_data \
        --output-dir results/concept_head_eval \
        --mode global
"""

import argparse
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import torch
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json
from PIL import Image
import matplotlib.pyplot as plt

from src.data.dataset_concept import ConceptDataset, collate_fn
from src.models.concept_head import ConceptHead
from src.utils.concept_utils import (
    load_all_concept_data,
    compute_accuracy
)


@torch.no_grad()
def evaluate(
    model: ConceptHead,
    dataloader: DataLoader,
    device: torch.device,
    mode: str = "global"
) -> tuple:
    """
    Evaluate model on a dataset.
    
    Args:
        model: Concept head model
        dataloader: Data loader
        device: Device to use
        mode: "global" or "spatial"
    
    Returns:
        all_probs: All predicted probabilities [N, K]
        all_labels: All ground truth labels [N]
    """
    model.eval()
    
    all_probs = []
    all_labels = []
    
    for pooled, patch_tokens, labels, coords in tqdm(dataloader, desc="Evaluating"):
        pooled = pooled.to(device)
        
        if mode == "spatial" and patch_tokens is not None:
            patch_tokens = patch_tokens.to(device)
        else:
            patch_tokens = None
        
        probs, _ = model(pooled, patch_tokens)
        
        all_probs.append(probs.cpu())
        all_labels.append(labels)
    
    all_probs = torch.cat(all_probs, dim=0)
    all_labels = torch.cat(all_labels, dim=0)
    
    return all_probs, all_labels


def visualize_sample(
    image_path: str,
    probs: np.ndarray,
    gt_label: int,
    idx_to_concept: dict,
    save_path: str,
    attention_heatmap: np.ndarray = None,
    top_k: int = 5
):
    """
    Visualize a single sample with image, top-k concepts, and optional attention.
    
    Args:
        image_path: Path to the image file
        probs: Predicted probabilities [K]
        gt_label: Ground truth label index
        idx_to_concept: Mapping from index to concept name
        save_path: Where to save the figure
        attention_heatmap: Optional attention heatmap [H, W]
        top_k: Number of top concepts to show
    """
    # Load image
    try:
        image = Image.open(image_path).convert('RGB')
        image = np.array(image)
    except Exception as e:
        print(f"Could not load image {image_path}: {e}")
        image = np.zeros((224, 224, 3), dtype=np.uint8)
    
    # Get top-k concepts
    top_indices = np.argsort(probs)[::-1][:top_k]
    top_probs = probs[top_indices]
    top_names = [idx_to_concept.get(str(i), f"Concept_{i}") for i in top_indices]
    
    # Determine number of subplots
    n_cols = 3 if attention_heatmap is not None else 2
    fig, axes = plt.subplots(1, n_cols, figsize=(5 * n_cols, 5))
    
    # Plot image
    axes[0].imshow(image)
    axes[0].set_title("Input Image")
    axes[0].axis('off')
    
    # Plot top-k concepts as bar chart
    colors = ['#2ecc71' if i == gt_label else '#3498db' for i in top_indices]
    y_pos = np.arange(top_k)
    axes[1].barh(y_pos, top_probs, color=colors)
    axes[1].set_yticks(y_pos)
    axes[1].set_yticklabels([f"{name[:30]}" for name in top_names], fontsize=9)
    axes[1].invert_yaxis()
    axes[1].set_xlabel("Probability")
    axes[1].set_xlim(0, 1)
    
    # Mark ground truth
    gt_name = idx_to_concept.get(str(gt_label), f"Concept_{gt_label}")
    if gt_label in top_indices:
        axes[1].set_title(f"Top-{top_k} Concepts (✓ GT: {gt_name[:20]})")
    else:
        axes[1].set_title(f"Top-{top_k} Concepts (GT: {gt_name[:20]})")
    
    # Plot attention heatmap if available
    if attention_heatmap is not None:
        im = axes[2].imshow(attention_heatmap, cmap='jet', interpolation='bilinear')
        axes[2].set_title(f"Attention: {top_names[0][:25]}")
        axes[2].axis('off')
        plt.colorbar(im, ax=axes[2], fraction=0.046, pad=0.04)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def visualize_samples(
    model: ConceptHead,
    dataset: ConceptDataset,
    idx_to_concept: dict,
    output_dir: Path,
    num_samples: int = 20,
    device: torch.device = torch.device('cpu'),
    mode: str = "global"
):
    """
    Visualize sample predictions with images and top-5 concepts.
    """
    model.eval()
    
    vis_dir = output_dir / 'sample_predictions'
    vis_dir.mkdir(exist_ok=True)
    
    # Sample random indices
    indices = np.random.choice(len(dataset), size=min(num_samples, len(dataset)), replace=False)
    
    with torch.no_grad():
        for i, idx in enumerate(indices):
            pooled, patch_tokens, label, coords = dataset[idx]
            pooled = pooled.unsqueeze(0).to(device)
            
            if mode == "spatial" and patch_tokens is not None:
                patch_tokens = patch_tokens.unsqueeze(0).to(device)
            else:
                patch_tokens = None
            
            # Get predictions
            probs, _ = model(pooled, patch_tokens)
            probs_np = probs[0].cpu().numpy()
            label_int = label.item()
            
            # Get attention heatmap for spatial mode
            attention_heatmap = None
            if mode == "spatial" and patch_tokens is not None:
                heatmaps = model.get_attention_heatmaps(patch_tokens)  # [1, K, H, W]
                top_idx = probs_np.argmax()
                attention_heatmap = heatmaps[0, top_idx].cpu().numpy()
            
            # Get image path from dataset
            image_path = dataset.df.iloc[idx]['image_path']
            
            # Visualize
            save_path = vis_dir / f"sample_{i:03d}.png"
            visualize_sample(
                image_path=image_path,
                probs=probs_np,
                gt_label=label_int,
                idx_to_concept=idx_to_concept,
                save_path=str(save_path),
                attention_heatmap=attention_heatmap,
                top_k=5
            )
    
    print(f"Saved {len(indices)} sample visualizations to {vis_dir}")


def main():
    parser = argparse.ArgumentParser(description="Evaluate Concept Head")
    
    # Required paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--cached-embeddings-dir", type=str, required=True, help="Directory with cached embeddings")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Directory with concept data")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    
    # Model options
    parser.add_argument("--mode", type=str, default="global", choices=["global", "spatial"],
                        help="Model mode: 'global' (default) or 'spatial'")
    
    # Other options
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--num-samples", type=int, default=50, help="Number of sample visualizations")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load concept data
    print("Loading concept data...")
    concept_data = load_all_concept_data(args.concept_data_dir, split="test")
    
    vocab = concept_data['vocabulary']
    embeddings = concept_data['embeddings']
    idx_to_concept = vocab['idx_to_concept']
    K = vocab['num_concepts']
    
    print(f"Loaded {K} concepts")
    
    # Create dataset
    print(f"Creating test dataset (mode: {args.mode})...")
    load_patch_tokens = (args.mode == "spatial")
    
    test_dataset = ConceptDataset(
        csv_path=args.test_csv,
        cached_embeddings_dir=args.cached_embeddings_dir,
        concept_vocab_path=f"{args.concept_data_dir}/concept_vocabulary.json",
        split="test",
        load_patch_tokens=load_patch_tokens
    )
    
    print(f"Test samples: {len(test_dataset)}")
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        collate_fn=collate_fn
    )
    
    # Load model
    print(f"Loading model from {args.checkpoint}")
    device = torch.device(args.device)
    
    model = ConceptHead(
        concept_embeddings=embeddings,
        mode=args.mode
    ).to(device)
    
    checkpoint = torch.load(args.checkpoint, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    
    print(f"Loaded model from epoch {checkpoint.get('epoch', 'unknown')}")
    print(f"Model mode: {args.mode}")
    
    # Evaluate
    print("\nEvaluating...")
    all_probs, all_labels = evaluate(model, test_loader, device, mode=args.mode)
    
    # Compute accuracy
    accuracy = compute_accuracy(all_probs, all_labels)
    
    print("\n" + "="*60)
    print("Results:")
    print("="*60)
    print(f"Top-1 Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"Test samples: {len(test_dataset)}")
    print(f"Num concepts: {K}")
    print(f"Mode: {args.mode}")
    
    # Save metrics
    with open(output_dir / 'metrics.json', 'w') as f:
        json.dump({
            'accuracy': accuracy,
            'num_samples': len(test_dataset),
            'num_concepts': K,
            'mode': args.mode
        }, f, indent=2)
    print(f"\nSaved metrics to {output_dir / 'metrics.json'}")
    
    # Generate sample visualizations
    if args.num_samples > 0:
        print(f"\nGenerating {args.num_samples} sample visualizations...")
        visualize_samples(
            model=model,
            dataset=test_dataset,
            idx_to_concept=idx_to_concept,
            output_dir=output_dir,
            num_samples=args.num_samples,
            device=device,
            mode=args.mode
        )
    
    print(f"\n{'='*60}")
    print(f"Evaluation complete! Results saved to: {output_dir}")
    print("="*60)


if __name__ == "__main__":
    main()
