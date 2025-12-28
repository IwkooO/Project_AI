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

from cbm.phase1.model import Phase1CBMTopKMil, Phase1CBMCrossAttention
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
    # (patches, concept_label, coords, cell_label, country_label, cache_idx, pooled_emb)
    for patches, c_labels, coords, cell_labels, country_labels, offsets, pooled_emb in tqdm(dataloader, desc="Evaluating"):
        patches = patches.to(device)
        c_labels = c_labels.to(device)
        
        # Validate pooled_emb if global head is enabled
        if hasattr(model, 'concept_head') and hasattr(model.concept_head, 'use_global_head'):
            if model.concept_head.use_global_head:
                if pooled_emb is None:
                    raise ValueError(
                        f"Evaluation: model.use_global_head=True but pooled_emb is None. "
                        f"Make sure dataset.load_pooled_embeddings=True."
                    )
                pooled_emb = pooled_emb.to(device)
                # Validate shape
                if pooled_emb.dim() != 2:
                    raise ValueError(
                        f"Evaluation: pooled_emb must be [B, D], got shape={tuple(pooled_emb.shape)}"
                    )
                if pooled_emb.size(0) != patches.size(0):
                    raise ValueError(
                        f"Evaluation: batch size mismatch. patches.shape[0]={patches.size(0)}, "
                        f"pooled_emb.shape[0]={pooled_emb.size(0)}"
                    )
        elif pooled_emb is not None:
            # Global head disabled but pooled_emb provided - warn but allow (might be for future use)
            pooled_emb = pooled_emb.to(device)
        
        # Forward (patch-only, with optional pooled embeddings for global head)
        c_logits, c_hidden, attn_w, _ = model(patches, pooled_emb=pooled_emb)
        
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
    parser.add_argument("--cached-dir", type=str, default=None, help="Directory with cached patch tokens")
    parser.add_argument("--image-dir", type=str, default=None, help="Directory with raw images (required if model has trainable backbone)")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Directory with concept vocab and S2 cells")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory for results")
    parser.add_argument(
        "--vision-model-name",
        type=str,
        default="geolocal/StreetCLIP",
        help="HuggingFace model ID for vision encoder (required if model has trainable backbone)",
    )
    
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
    parser.add_argument(
        "--use-pos-encoding",
        action="store_true",
        default=True,
        help="Use positional encoding (must match training checkpoint)",
    )
    parser.add_argument(
        "--no-pos-encoding",
        dest="use_pos_encoding",
        action="store_false",
        help="Disable positional encoding (must match training checkpoint)",
    )
    parser.add_argument(
        "--max-patches",
        type=int,
        default=576,
        help="Maximum number of patches for positional encoding (must match training checkpoint)",
    )
    parser.add_argument(
        "--use-per-concept-tau",
        action="store_true",
        default=False,
        help="Use per-concept adaptive temperature (must match training checkpoint, disabled by default for backward compatibility)",
    )
    parser.add_argument(
        "--use-global-head",
        action="store_true",
        default=False,
        help="Use global head with CLS token (must match training checkpoint, disabled by default for backward compatibility)",
    )
    parser.add_argument(
        "--pooled-dim",
        type=int,
        default=768,
        help="Dimension of pooled embeddings (must match training checkpoint, default: 768 for StreetCLIP)",
    )
    parser.add_argument(
        "--head-type",
        type=str,
        default="topk_mil",
        choices=["topk_mil", "cross_attention"],
        help="Concept head architecture (must match training checkpoint)",
    )
    parser.add_argument(
        "--xattn-heads",
        type=int,
        default=8,
        help="Number of attention heads for cross-attention head",
    )
    parser.add_argument(
        "--xattn-temperature",
        type=float,
        default=1.0,
        help="Attention temperature for cross-attention head",
    )
    parser.add_argument(
        "--xattn-topk",
        type=int,
        default=16,
        help="Top-K patches to attend to in cross-attention",
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
    
    # Check checkpoint first to determine if it has global head, positional encoding, and vision encoder
    checkpoint_preview = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    checkpoint_state = checkpoint_preview['model_state_dict']
    
    has_global_head_in_checkpoint = 'concept_head.global_query' in checkpoint_state
    has_pos_embed_in_checkpoint = 'concept_head.pos_embed' in checkpoint_state
    has_per_concept_tau_in_checkpoint = 'concept_head.concept_tau_logit' in checkpoint_state
    has_vision_encoder_in_checkpoint = any(k.startswith('vision_encoder.') for k in checkpoint_state.keys())
    
    use_global_head_for_dataset = args.use_global_head or has_global_head_in_checkpoint
    
    # Initialize vision encoder if present in checkpoint
    vision_encoder = None
    expected_num_patches = None
    transform = None
    if has_vision_encoder_in_checkpoint:
        print(f"Checkpoint contains trainable backbone. Initializing vision encoder from {args.vision_model_name}...")
        from transformers import CLIPModel, CLIPProcessor
        clip_model = CLIPModel.from_pretrained(args.vision_model_name)
        vision_encoder = clip_model.vision_model.to(device)
        
        # Get standard CLIP image transforms
        processor = CLIPProcessor.from_pretrained(args.vision_model_name)
        def clip_transform(img):
            return processor(images=img, return_tensors="pt")["pixel_values"].squeeze(0)
        transform = clip_transform
        
        # Detect expected number of patches from config
        if hasattr(clip_model.config, 'vision_config'):
            image_size = getattr(clip_model.config.vision_config, 'image_size', 336)
            patch_size = getattr(clip_model.config.vision_config, 'patch_size', 14)
        else:
            image_size = getattr(clip_model.config, 'image_size', 336)
            patch_size = getattr(clip_model.config, 'patch_size', 14)
        expected_num_patches = (image_size // patch_size) ** 2
        print(f"Vision encoder initialized: image_size={image_size}, patch_size={patch_size}, expected_patches={expected_num_patches}")

    # Load test dataset
    print("Loading test dataset...")
    test_ds = ConceptDataset(
        args.test_csv,
        args.cached_dir,
        concept_vocab,
        str(s2_vocab),
        split="test",
        image_dir=args.image_dir,
        transform=transform,
        allow_unsafe_index_fallback=False,
        load_pooled_embeddings=use_global_head_for_dataset,
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
    if vision_encoder is not None:
        patch_dim = vision_encoder.config.hidden_size
    else:
        patch_dim = test_ds.patch_tokens.shape[2]
    print(f"Patch dimension: {patch_dim}")
    
    # Override use_pos_encoding based on checkpoint if not explicitly set
    if has_pos_embed_in_checkpoint:
        print("Checkpoint contains positional encoding - using it.")
        use_pos_encoding = args.use_pos_encoding  # Use user's setting if provided
    else:
        print("Checkpoint does NOT contain positional encoding - disabling it to match checkpoint.")
        use_pos_encoding = False  # Force disable to match checkpoint
    
    # Override use_per_concept_tau based on checkpoint if not explicitly set
    if has_per_concept_tau_in_checkpoint:
        print("Checkpoint contains per-concept temperature - using it.")
        use_per_concept_tau = args.use_per_concept_tau  # Use user's setting if provided
    else:
        print("Checkpoint does NOT contain per-concept temperature - disabling it to match checkpoint.")
        use_per_concept_tau = False  # Force disable to match checkpoint
    
    # Override use_global_head based on checkpoint if not explicitly set
    if has_global_head_in_checkpoint:
        print("Checkpoint contains global head - using it.")
        use_global_head = args.use_global_head  # Use user's setting if provided
    else:
        print("Checkpoint does NOT contain global head - disabling it to match checkpoint.")
        use_global_head = False  # Force disable to match checkpoint
    
    # Initialize model
    mix_local_kernel_size = int(args.mix_local_kernel_size)
    if mix_local_kernel_size == 0:
        mix_local_kernel_size = None
    
    if args.head_type == "cross_attention":
        print("Initializing model (cross-attention head)...")
        use_topk_attn = args.xattn_topk > 0
        model = Phase1CBMCrossAttention(
            num_concepts=test_ds.num_concepts,
            patch_dim=patch_dim,
            concept_dim=args.concept_dim,
            dropout=args.dropout,
            num_heads=args.xattn_heads,
            mix_depth=args.mix_depth,
            mix_mlp_ratio=args.mix_mlp_ratio,
            attn_temperature=args.xattn_temperature,
            use_topk_attn=use_topk_attn,
            topk=args.xattn_topk if use_topk_attn else 16,
            vision_encoder=vision_encoder,
            expected_num_patches=expected_num_patches,
        )
    else:
        print("Initializing model (canonical Phase-1)...")
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
            use_pos_encoding=use_pos_encoding,
            max_patches=args.max_patches,
            use_per_concept_tau=use_per_concept_tau,
            use_global_head=use_global_head,
            pooled_dim=args.pooled_dim,
            vision_encoder=vision_encoder,
            expected_num_patches=expected_num_patches,
        )
    model = model.to(device)
    
    # Load checkpoint (already loaded for preview, but reload to device)
    print(f"Loading checkpoint weights to device...")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
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
