#!/usr/bin/env python3
"""
Evaluate trained Concept Head (Patch-only).

Computes Acc@1, Acc@5, and CE loss on test set.
Optionally generates attention overlay visualizations.
"""

import argparse
from pathlib import Path

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
from tqdm import tqdm
import json

from cbm.phase1.model import Phase1CBMTopKMil
from cbm.phase1.data import ConceptDataset, collate_fn
from cbm.viz.attention import visualize_predictions_summary


def compute_concept_weights(dataset, num_concepts, device):
    """Compute inverse frequency weights for concept loss."""
    concept_names = dataset.df['meta_name']
    counts_dict = concept_names.value_counts().to_dict()
    
    counts = torch.zeros(num_concepts)
    for name, count in counts_dict.items():
        if name in dataset.concept_to_idx:
            idx = dataset.concept_to_idx[name]
            counts[idx] = count
    
    weights = 1.0 / torch.sqrt(counts + 1.0)
    weights = torch.clamp(weights, min=0.1, max=10.0)
    weights = weights / weights.mean()
    return weights.to(device)


@torch.no_grad()
def evaluate(
    model: Phase1CBMTopKMil,
    dataloader: DataLoader,
    device: torch.device,
    criterion: nn.Module,
) -> dict:
    """
    Evaluate model on a dataset.
    
    Returns:
        Dictionary with metrics: ce_loss, acc1, acc5
    """
    model.eval()
    
    total_ce_loss = 0.0
    correct_top1 = 0
    correct_top5 = 0
    total_samples = 0
    
    # Dataset/collate yields:
    # (patches, concept_label, coords, cell_label, country_label, cache_idx)
    for patches, c_labels, coords, cell_labels, country_labels, offsets in tqdm(dataloader, desc="Evaluating"):
        patches = patches.to(device)
        c_labels = c_labels.to(device)
        
        # Forward (patch-only)
        c_logits, c_hidden, attn_w, _ = model(patches)
        
        # CE loss
        ce_loss = criterion(c_logits, c_labels)
        
        # Metrics
        preds = c_logits.argmax(dim=1)
        correct_top1 += (preds == c_labels).sum().item()
        
        k = min(5, c_logits.size(1))
        topk = c_logits.topk(k, dim=1).indices
        correct_top5 += (topk == c_labels.unsqueeze(1)).any(dim=1).sum().item()
        
        total_ce_loss += ce_loss.item() * patches.size(0)
        total_samples += patches.size(0)
    
    return {
        "ce_loss": total_ce_loss / total_samples,
        "acc1": correct_top1 / total_samples,
        "acc5": correct_top5 / total_samples,
        "num_samples": total_samples,
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluate Concept Head (Patch-only)")
    
    # Required paths
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint")
    parser.add_argument("--test-csv", type=str, required=True, help="Path to test CSV")
    parser.add_argument("--cached-dir", type=str, required=True, help="Directory with cached patch tokens")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Directory with concept vocab and S2 cells")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    
    # Model config (should match training)
    parser.add_argument("--concept-dim", type=int, default=256, help="Concept dimension")
    parser.add_argument("--dropout", type=float, default=0.3, help="Dropout")
    parser.add_argument("--mil-topk", type=int, default=8, help="Hard top-K selection for MIL-style aggregation")
    parser.add_argument("--mil-tau", type=float, default=0.25, help="Temperature for LogSumExp aggregation")
    parser.add_argument("--mix-depth", type=int, default=1, help="Patch mixer depth")
    parser.add_argument("--mix-heads", type=int, default=4, help="Patch mixer heads")
    parser.add_argument("--mix-mlp-ratio", type=float, default=4.0, help="Patch mixer MLP ratio")
    parser.add_argument(
        "--mix-local-kernel-size",
        type=int,
        default=0,
        help="Neighborhood Attention kernel size for patch mixing (0 disables; must be odd, e.g. 3/5/7).",
    )
    parser.add_argument(
        "--proj-type",
        type=str,
        default="simple",
        choices=["simple", "two_stage", "bottleneck"],
        help="Patch projection architecture type (must match training checkpoint)",
    )
    
    # Options
    parser.add_argument("--batch-size", type=int, default=256, help="Batch size")
    parser.add_argument("--num-workers", type=int, default=4, help="Number of data workers")
    parser.add_argument("--generate-visualizations", action="store_true", help="Generate attention overlay visualizations")
    parser.add_argument("--num-viz-samples", type=int, default=20, help="Number of samples to visualize")
    parser.add_argument(
        "--viz-seed",
        type=int,
        default=0,
        help="Seed for choosing visualization samples (ensures same images across models).",
    )
    parser.add_argument(
        "--viz-indices",
        type=str,
        default="",
        help="Optional comma-separated dataset indices to visualize (overrides --viz-seed/--num-viz-samples).",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load vocabularies
    concept_vocab = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab = Path(args.concept_data_dir) / "s2_cells.json"
    
    # Load test dataset (patch-only)
    print("Loading test dataset...")
    test_ds = ConceptDataset(
        args.test_csv,
        args.cached_dir,
        concept_vocab,
        str(s2_vocab),
        split="test",
        allow_unsafe_index_fallback=False,
    )
    
    print(f"Test samples: {len(test_ds)}")
    print(f"Num concepts: {test_ds.num_concepts}")
    
    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        collate_fn=collate_fn,
        num_workers=args.num_workers,
    )
    
    # Detect patch dimension
    patch_dim = test_ds.patch_tokens.shape[2]
    print(f"Patch dimension: {patch_dim}")
    
    # Initialize model (patch-only)
    print("Initializing model (canonical Phase-1)...")
    mix_local_kernel_size = int(args.mix_local_kernel_size)
    if mix_local_kernel_size == 0:
        mix_local_kernel_size = None
    model = Phase1CBMTopKMil(
        num_concepts=test_ds.num_concepts,
        patch_dim=patch_dim,
        concept_dim=args.concept_dim,
        dropout=args.dropout,
        mil_topk=args.mil_topk,
        mil_tau=args.mil_tau,
        mix_depth=args.mix_depth,
        mix_heads=args.mix_heads,
        mix_mlp_ratio=args.mix_mlp_ratio,
        mix_local_kernel_size=mix_local_kernel_size,
        proj_type=args.proj_type,
    )
    model = model.to(device)
    
    # Load checkpoint
    print(f"Loading checkpoint from {args.checkpoint}...")
    # PyTorch 2.6+ defaults torch.load(weights_only=True), which can fail for our checkpoints
    # (e.g., when they contain numpy scalars in metadata). We trust our own checkpoints here.
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
    if 'val_acc1' in checkpoint:
        print(f"Checkpoint val Acc@1: {checkpoint['val_acc1']:.4f}")
        print(f"Checkpoint val Acc@5: {checkpoint['val_acc5']:.4f}")
    
    # Loss function
    weights = compute_concept_weights(test_ds, test_ds.num_concepts, device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.1)
    
    # Evaluate
    print("\nEvaluating...")
    metrics = evaluate(model, test_loader, device, criterion)
    
    # Print results
    print("\n" + "="*60)
    print("Evaluation Results:")
    print("="*60)
    print(f"CE Loss:     {metrics['ce_loss']:.4f}")
    print(f"Acc@1:       {metrics['acc1']:.4f} ({metrics['acc1']*100:.2f}%)")
    print(f"Acc@5:       {metrics['acc5']:.4f} ({metrics['acc5']*100:.2f}%)")
    print(f"Num samples: {metrics['num_samples']}")
    print("="*60)
    
    # Save metrics
    metrics_file = output_dir / 'metrics.json'
    with open(metrics_file, 'w') as f:
        json.dump(metrics, f, indent=2)
    print(f"\nSaved metrics to {metrics_file}")
    
    # Generate visualizations if requested
    if args.generate_visualizations:
        # Pick deterministic indices so different models produce comparable overlays.
        if args.viz_indices.strip():
            viz_indices = []
            for part in args.viz_indices.split(","):
                part = part.strip()
                if not part:
                    continue
                viz_indices.append(int(part))
            viz_indices = sorted(set(viz_indices))
            print(f"\nGenerating visualizations for explicit indices: {viz_indices}")
        else:
            n = max(1, min(int(args.num_viz_samples), len(test_ds)))
            rng = np.random.default_rng(int(args.viz_seed))
            viz_indices = rng.choice(len(test_ds), n, replace=False).astype(int).tolist()
            viz_indices = sorted(viz_indices)
            print(f"\nGenerating {n} sample visualizations (viz_seed={args.viz_seed})...")

        # Save chosen indices for reproducibility/debugging.
        viz_idx_file = output_dir / "viz_sample_indices.json"
        with open(viz_idx_file, "w") as f:
            json.dump({"viz_indices": viz_indices}, f, indent=2)
        print(f"Visualization indices saved to {viz_idx_file}")

        visualize_predictions_summary(
            model=model,
            dataset=test_ds,
            device=device,
            epoch=checkpoint.get('epoch', 0),
            output_dir=output_dir,
            phase=1,
            run_id="eval",
            reason="evaluation",
            num_samples=args.num_viz_samples,
            sample_indices=viz_indices,
        )
        print(f"Visualizations saved to {output_dir / 'visualizations'}")
    
    print(f"\nEvaluation complete! Results saved to: {output_dir}")


if __name__ == "__main__":
    main()
