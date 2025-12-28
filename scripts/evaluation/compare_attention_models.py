#!/usr/bin/env python3
"""
Compare attention visualizations across multiple models on the same set of images.

Usage:
    python scripts/evaluation/compare_attention_models.py \
        --checkpoints checkpoint1 checkpoint2 ... \
        --test-csv data/splits/dataset_test.csv \
        --cached-dir /path/to/cached \
        --concept-data-dir data/concept_data_v2 \
        --output-dir comparisons/attention_comparison \
        --num-samples 10
"""

import argparse
import json
import math
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from cbm.phase1.data import ConceptDataset, collate_fn
from cbm.phase1.model import Phase1CBMTopKMil


def parse_checkpoint_config(checkpoint_path: Path) -> Dict:
    """Parse model configuration from checkpoint directory name."""
    name = checkpoint_path.name
    
    # Default config
    config = {
        "concept_dim": 256,
        "dropout": 0.3,
        "mil_topk": 6,
        "mil_tau": 0.25,
        "mix_depth": 1,
        "mix_heads": 4,
        "mix_mlp_ratio": 2.0,
        "mix_local_kernel_size": 5,
        "proj_type": "simple",
    }
    
    # Parse from name patterns
    if "textinit" in name.lower():
        config["concept_dim"] = 768
        # proj_type stays "simple" for textinit models
    
    if "full_attn" in name.lower():
        config["mix_local_kernel_size"] = None
    
    # Parse proj_type (two_stage, bottleneck, simple)
    if "two_stage" in name.lower() or "twostage" in name.lower():
        config["proj_type"] = "two_stage"
        config["dropout"] = 0.4  # Higher dropout for two_stage
    elif "bottleneck" in name.lower():
        config["proj_type"] = "bottleneck"
    
    # Parse tau, k, stk, lk, div from patterns like "tau0.25_k6_stk0.40_lk5_div0.02"
    # Also handle uppercase patterns like "TAU0.25_K6_STK0.40_LK5_DIV0.5"
    tau_match = re.search(r"[Tt]au([\d.]+)", name, re.IGNORECASE)
    if tau_match:
        config["mil_tau"] = float(tau_match.group(1))
    
    # Parse K/k (case insensitive) - matches _k6, K6, k6, etc.
    k_match = re.search(r"(?:^|_)[Kk](\d+)", name)
    if k_match:
        config["mil_topk"] = int(k_match.group(1))
    
    stk_match = re.search(r"[Ss]tk([\d.]+)", name, re.IGNORECASE)
    if stk_match:
        stk_val = float(stk_match.group(1))
        # STK_MASK_PROB is not used in model init, but we can note it
    
    lk_match = re.search(r"[_Ll]lk(\d+)", name, re.IGNORECASE)
    if lk_match:
        lk_val = int(lk_match.group(1))
        config["mix_local_kernel_size"] = lk_val if lk_val > 0 else None
    
    return config


def load_model(
    checkpoint_path: Path,
    num_concepts: int,
    patch_dim: int,
    device: torch.device,
    config: Optional[Dict] = None,
) -> Tuple[Phase1CBMTopKMil, str]:
    """Load a model from checkpoint."""
    if config is None:
        config = parse_checkpoint_config(checkpoint_path)
    
    # Find checkpoint file
    if (checkpoint_path / "phase1" / "best_phase1.pt").exists():
        ckpt_file = checkpoint_path / "phase1" / "best_phase1.pt"
    elif (checkpoint_path / "best_phase1.pt").exists():
        ckpt_file = checkpoint_path / "best_phase1.pt"
    else:
        raise FileNotFoundError(f"No checkpoint found in {checkpoint_path}")
    
    # Initialize model
    mix_local_kernel = config.get("mix_local_kernel_size")
    if mix_local_kernel == 0:
        mix_local_kernel = None
    
    model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=config["concept_dim"],
        dropout=config["dropout"],
        mil_topk=config["mil_topk"],
        mil_tau=config["mil_tau"],
        mix_depth=config["mix_depth"],
        mix_heads=config["mix_heads"],
        mix_mlp_ratio=config["mix_mlp_ratio"],
        mix_local_kernel_size=mix_local_kernel,
        proj_type=config.get("proj_type", "simple"),
    )
    
    # Load weights - map to device (handles CUDA->CPU if needed)
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()
    model = model.to(device)
    
    # Generate model name for display
    model_name = checkpoint_path.name
    if len(model_name) > 50:
        model_name = model_name[:47] + "..."
    
    return model, model_name


@torch.no_grad()
def get_predictions_and_attention(
    model: Phase1CBMTopKMil,
    patches: torch.Tensor,
    device: torch.device,
    pooled_emb: torch.Tensor | None = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Get predictions and attention weights for a batch."""
    patches = patches.to(device)
    if pooled_emb is not None:
        pooled_emb = pooled_emb.to(device)
    logits, hidden, attn_weights, _ = model(patches, pooled_emb=pooled_emb)
    probs = torch.softmax(logits, dim=1)
    return logits, probs, attn_weights


def visualize_comparison(
    image_path: str,
    gt_concept_idx: int,
    gt_concept_name: str,
    model_results: List[Dict],
    dataset,
    output_path: Path,
    sample_idx: int,
):
    """Create side-by-side attention comparison visualization."""
    try:
        img = plt.imread(image_path)
    except Exception as e:
        print(f"Warning: Could not load image {image_path}: {e}")
        return
    
    num_models = len(model_results)
    fig, axes = plt.subplots(1, num_models + 1, figsize=(5 * (num_models + 1), 5))
    
    # Show original image
    axes[0].imshow(img)
    axes[0].set_title(f"Original\nGT: {gt_concept_name}", fontsize=10)
    axes[0].axis("off")
    
    # Show attention for each model
    for i, result in enumerate(model_results):
        ax = axes[i + 1]
        model_name = result["model_name"]
        attn_weights = result["attn_weights"]
        top5_idx = result["top5_idx"]
        top5_probs = result["top5_probs"]
        pred_idx = result["pred_idx"]
        pred_prob = result["pred_prob"]
        
        # Get attention for ground truth concept
        if gt_concept_idx < attn_weights.shape[0]:
            gt_attn = attn_weights[gt_concept_idx].cpu().numpy()
        else:
            gt_attn = np.zeros(attn_weights.shape[1])
        
        # Reshape attention to grid
        P = len(gt_attn)
        side = int(math.isqrt(P))
        if side * side == P:
            attn_grid = gt_attn.reshape(side, side)
            
            # Upsample to image size
            attn_tensor = torch.tensor(attn_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            attn_upsampled = F.interpolate(
                attn_tensor,
                size=(img.shape[0], img.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            attn_upsampled_np = attn_upsampled.squeeze().cpu().numpy()
            
            # Normalize
            attn_upsampled_np = (attn_upsampled_np - attn_upsampled_np.min()) / (
                attn_upsampled_np.max() - attn_upsampled_np.min() + 1e-8
            )
            
            # Overlay
            ax.imshow(img)
            im = ax.imshow(attn_upsampled_np, cmap="magma", alpha=0.4, interpolation="bilinear")
            ax.axis("off")
            
            # Title with predictions
            pred_name = dataset.idx_to_concept[pred_idx]
            is_correct = "✓" if pred_idx == gt_concept_idx else "✗"
            title = f"{model_name}\nPred: {pred_name} {is_correct} ({pred_prob:.2f})"
            ax.set_title(title, fontsize=9)
        else:
            ax.imshow(img)
            ax.set_title(f"{model_name}\n(No GT attention)", fontsize=9)
            ax.axis("off")
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()


def main():
    parser = argparse.ArgumentParser(description="Compare attention across multiple models")
    
    parser.add_argument("--checkpoints", nargs="+", required=True, help="Checkpoint directories")
    parser.add_argument("--test-csv", type=str, required=True, help="Test CSV path")
    parser.add_argument("--cached-dir", type=str, required=True, help="Cached embeddings directory")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Concept data directory")
    parser.add_argument("--output-dir", type=str, required=True, help="Output directory")
    parser.add_argument("--num-samples", type=int, default=10, help="Number of samples to visualize")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    # Ensure device is valid
    if args.device == "cuda" and not torch.cuda.is_available():
        print("Warning: CUDA requested but not available, using CPU instead")
        device = torch.device("cpu")
    else:
        device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    concept_vocab = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab = Path(args.concept_data_dir) / "s2_cells.json"
    
    print("Loading test dataset...")
    test_ds = ConceptDataset(
        args.test_csv,
        args.cached_dir,
        concept_vocab,
        str(s2_vocab),
        split="test",
        allow_unsafe_index_fallback=False,
    )
    
    patch_dim = test_ds.patch_tokens.shape[2]
    print(f"Dataset: {len(test_ds)} samples, {test_ds.num_concepts} concepts")
    print(f"Patch dimension: {patch_dim}")
    
    # Load all models
    print("\nLoading models...")
    models = []
    model_names = []
    checkpoint_paths = [Path(cp) for cp in args.checkpoints]
    
    for cp_path in checkpoint_paths:
        try:
            config = parse_checkpoint_config(cp_path)
            model, name = load_model(cp_path, test_ds.num_concepts, patch_dim, device, config)
            models.append(model)
            model_names.append(name)
            print(f"  ✓ Loaded: {name}")
        except Exception as e:
            print(f"  ✗ Failed to load {cp_path.name}: {e}")
            continue
    
    if len(models) == 0:
        print("Error: No models loaded successfully!")
        return
    
    # Select random samples
    num_samples = min(args.num_samples, len(test_ds))
    sample_indices = np.random.choice(len(test_ds), num_samples, replace=False)
    
    print(f"\nGenerating comparisons for {num_samples} samples...")
    
    for sample_idx in tqdm(sample_indices, desc="Processing samples"):
        # Get sample
        # Dataset returns 7 items: patches, c_label, coords, cell_label, country_label, cache_idx, pooled_emb
        sample = test_ds[sample_idx]
        patches = sample[0]
        c_label = sample[1]
        coords = sample[2]
        cell_label = sample[3]
        country_label = sample[4]
        offset = sample[5]  # This is actually cache_idx, but we don't use it
        # sample[6] is pooled_emb, which we don't need here
        patches_batch = patches.unsqueeze(0)
        gt_concept_idx = int(c_label)
        gt_concept_name = test_ds.idx_to_concept[gt_concept_idx]
        image_path = test_ds.df.iloc[sample_idx]["image_path"]
        
        # Get predictions from all models
        # Check if pooled_emb is available (for global head)
        pooled_emb_batch = None
        if len(sample) > 6 and sample[6] is not None:
            pooled_emb_batch = sample[6].unsqueeze(0)
        
        model_results = []
        for model, model_name in zip(models, model_names):
            logits, probs, attn_weights = get_predictions_and_attention(
                model, patches_batch, device, pooled_emb=pooled_emb_batch
            )
            
            # Get top-5 predictions
            top5_probs, top5_idx = torch.topk(probs[0], min(5, probs.shape[1]))
            pred_idx = int(logits[0].argmax().item())
            pred_prob = float(probs[0, pred_idx].item())
            
            model_results.append({
                "model_name": model_name,
                "attn_weights": attn_weights[0],  # [K, P]
                "top5_idx": top5_idx.cpu().tolist(),
                "top5_probs": top5_probs.cpu().tolist(),
                "pred_idx": pred_idx,
                "pred_prob": pred_prob,
            })
        
        # Generate comparison visualization
        output_path = output_dir / f"sample_{sample_idx:05d}_gt_{gt_concept_name.replace(' ', '_')}.png"
        visualize_comparison(
            image_path,
            gt_concept_idx,
            gt_concept_name,
            model_results,
            test_ds,
            output_path,
            sample_idx,
        )
    
    print(f"\nComparison visualizations saved to: {output_dir}")
    
    # Save metadata
    metadata = {
        "checkpoints": [str(cp) for cp in checkpoint_paths],
        "model_names": model_names,
        "num_samples": num_samples,
        "sample_indices": sample_indices.tolist(),
    }
    with open(output_dir / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)


if __name__ == "__main__":
    main()

