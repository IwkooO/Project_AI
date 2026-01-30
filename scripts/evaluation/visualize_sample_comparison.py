#!/usr/bin/env python3
"""
Generate a beautiful report-ready comparison visualization of attention maps
for a specific sample across multiple models.
"""

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from cbm.phase1.data import ConceptDataset, collate_fn
from cbm.phase1.model import Phase1CBMTopKMil, Phase1CBMCrossAttention


def load_model_adaptive_tau(checkpoint_path: Path, num_concepts: int, patch_dim: int, device: torch.device):
    """Load adaptive tau model."""
    ckpt_file = checkpoint_path / "phase1" / "best_phase1.pt"
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    
    model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=256,
        dropout=0.3,
        mil_topk=6,
        mil_tau=0.25,
        mix_depth=1,
        mix_heads=4,
        mix_mlp_ratio=2.0,
        mix_local_kernel_size=None,
        proj_type="simple",
        use_pos_encoding=True,
        max_patches=576,
        use_per_concept_tau=True,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model = model.to(device)
    return model


def load_model_textinit(checkpoint_path: Path, num_concepts: int, patch_dim: int, device: torch.device):
    """Load text-init model."""
    ckpt_file = checkpoint_path / "phase1" / "best_phase1.pt"
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    
    model = Phase1CBMTopKMil(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=768,
        dropout=0.45,
        mil_topk=6,
        mil_tau=0.25,
        mix_depth=1,
        mix_heads=4,
        mix_mlp_ratio=2.0,
        mix_local_kernel_size=5,
        proj_type="simple",
        use_pos_encoding=True,
        max_patches=576,
        use_per_concept_tau=True,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model = model.to(device)
    return model


def load_model_crossattn(checkpoint_path: Path, num_concepts: int, patch_dim: int, device: torch.device):
    """Load cross-attention model."""
    ckpt_file = checkpoint_path / "phase1" / "best_phase1.pt"
    checkpoint = torch.load(ckpt_file, map_location=device, weights_only=False)
    
    model = Phase1CBMCrossAttention(
        num_concepts=num_concepts,
        patch_dim=patch_dim,
        concept_dim=256,
        dropout=0.3,
        num_heads=8,
        mix_depth=3,
        mix_mlp_ratio=4.0,
        attn_temperature=2.5,
        use_topk_attn=True,
        topk=16,
    )
    
    model.load_state_dict(checkpoint["model_state_dict"], strict=True)
    model.eval()
    model = model.to(device)
    return model


@torch.no_grad()
def get_predictions_and_attention(model, patches: torch.Tensor, device: torch.device, pooled_emb: Optional[torch.Tensor] = None):
    """Get predictions and attention weights."""
    patches = patches.to(device)
    if pooled_emb is not None:
        pooled_emb = pooled_emb.to(device)
    
    logits, hidden, attn_weights, _ = model(patches, pooled_emb=pooled_emb)
    probs = torch.softmax(logits, dim=1)
    return logits, probs, attn_weights


def create_comparison_visualization(
    image_path: str,
    gt_concept_idx: int,
    gt_concept_name: str,
    model_results: List[Dict],
    dataset: ConceptDataset,
    output_path: Path,
    sample_idx: int,
):
    """Create a simple comparison visualization with 3 attention overlay images."""
    try:
        img = Image.open(image_path).convert("RGB")
        img_arr = np.array(img)
    except Exception as e:
        print(f"Error loading image {image_path}: {e}")
        return
    
    # Create figure with 3 columns
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle(f'Sample {sample_idx}: {gt_concept_name}', 
                 fontsize=14, fontweight='bold', y=0.98)
    
    model_names = ["Adaptive Tau\n(Baseline)", "Text-Init", "Cross-Attention"]
    
    for i, (result, model_name, ax) in enumerate(zip(model_results, model_names, axes)):
        attn_weights = result["attn_weights"]  # [K, P]
        pred_idx = result["pred_idx"]
        pred_prob = result["pred_prob"]
        
        # Get attention for ground truth concept
        if gt_concept_idx < attn_weights.shape[0]:
            gt_attn = attn_weights[gt_concept_idx].cpu().numpy()
        else:
            gt_attn = np.zeros(attn_weights.shape[1])
        
        # Reshape and upsample attention
        P = len(gt_attn)
        side = int(math.isqrt(P))
        if side * side == P:
            attn_grid = gt_attn.reshape(side, side)
            
            attn_tensor = torch.tensor(attn_grid, dtype=torch.float32).unsqueeze(0).unsqueeze(0)
            attn_upsampled = F.interpolate(
                attn_tensor,
                size=(img_arr.shape[0], img_arr.shape[1]),
                mode="bilinear",
                align_corners=False,
            )
            attn_upsampled_np = attn_upsampled.squeeze().cpu().numpy()
            
            # Normalize
            attn_upsampled_np = (attn_upsampled_np - attn_upsampled_np.min()) / (
                attn_upsampled_np.max() - attn_upsampled_np.min() + 1e-8
            )
            
            # Overlay
            ax.imshow(img_arr)
            im = ax.imshow(attn_upsampled_np, cmap="magma", alpha=0.5, interpolation="bilinear")
            ax.axis("off")
            
            # Title with GT concept
            title = f"{model_name}\nGT: {gt_concept_name}"
            ax.set_title(title, fontsize=11, fontweight='bold', pad=10)
        else:
            ax.imshow(img_arr)
            ax.set_title(f"{model_name}\n(No GT attention)", fontsize=11, fontweight='bold')
            ax.axis("off")
    
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(output_path, dpi=300, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"Saved visualization to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Generate comparison visualization for a specific sample")
    
    parser.add_argument("--sample-idx", type=int, required=True, help="Sample index (e.g., 2535)")
    parser.add_argument("--val-csv", type=str, required=True, help="Validation CSV path")
    parser.add_argument("--cached-dir", type=str, required=True, help="Cached embeddings directory")
    parser.add_argument("--concept-data-dir", type=str, required=True, help="Concept data directory")
    
    # Model checkpoints
    parser.add_argument("--checkpoint-adaptive-tau", type=str, required=True)
    parser.add_argument("--checkpoint-textinit", type=str, required=True)
    parser.add_argument("--checkpoint-crossattn", type=str, required=True)
    
    parser.add_argument("--output-path", type=str, required=True, help="Output PNG path")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    args = parser.parse_args()
    
    device = torch.device(args.device)
    output_path = Path(args.output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Load dataset
    concept_vocab = Path(args.concept_data_dir) / "concept_vocab.json"
    s2_vocab = Path(args.concept_data_dir) / "s2_cells.json"
    
    print("Loading dataset...")
    val_ds = ConceptDataset(
        args.val_csv,
        args.cached_dir,
        concept_vocab,
        str(s2_vocab),
        split="val",
        allow_unsafe_index_fallback=False,
    )
    
    patch_dim = val_ds.patch_tokens.shape[2]
    print(f"Dataset: {len(val_ds)} samples, {val_ds.num_concepts} concepts")
    print(f"Patch dimension: {patch_dim}")
    
    # Load models
    print("\nLoading models...")
    checkpoint_paths = {
        "adaptive_tau": Path(args.checkpoint_adaptive_tau),
        "textinit": Path(args.checkpoint_textinit),
        "crossattn": Path(args.checkpoint_crossattn),
    }
    
    models = {}
    for name, cp_path in checkpoint_paths.items():
        print(f"  Loading {name}...")
        if name == "adaptive_tau":
            models[name] = load_model_adaptive_tau(cp_path, val_ds.num_concepts, patch_dim, device)
        elif name == "textinit":
            models[name] = load_model_textinit(cp_path, val_ds.num_concepts, patch_dim, device)
        elif name == "crossattn":
            models[name] = load_model_crossattn(cp_path, val_ds.num_concepts, patch_dim, device)
        print(f"    ✓ Loaded {name}")
    
    # Get sample
    sample_idx = args.sample_idx
    if sample_idx >= len(val_ds):
        print(f"Error: Sample index {sample_idx} out of range (max: {len(val_ds) - 1})")
        return
    
    print(f"\nProcessing sample {sample_idx}...")
    sample = val_ds[sample_idx]
    patches = sample[0].unsqueeze(0)  # [1, P, D]
    c_label = sample[1]
    gt_concept_idx = int(c_label)
    gt_concept_name = val_ds.idx_to_concept[gt_concept_idx]
    image_path = val_ds.df.iloc[sample_idx]["image_path"]
    
    # Check for pooled_emb
    pooled_emb = None
    if len(sample) > 6 and sample[6] is not None:
        pooled_emb = sample[6].unsqueeze(0)
    
    print(f"  Ground truth: {gt_concept_name}")
    print(f"  Image: {image_path}")
    
    # Get predictions from all models
    model_results = []
    model_names_ordered = ["adaptive_tau", "textinit", "crossattn"]
    
    for model_name in model_names_ordered:
        model = models[model_name]
        logits, probs, attn_weights = get_predictions_and_attention(
            model, patches, device, pooled_emb=pooled_emb
        )
        
        # Get top-5 predictions
        top5_probs, top5_idx = torch.topk(probs[0], min(5, probs.shape[1]))
        pred_idx = int(logits[0].argmax().item())
        pred_prob = float(probs[0, pred_idx].item())
        
        model_results.append({
            "attn_weights": attn_weights[0],  # [K, P]
            "top5_idx": top5_idx.cpu().tolist(),
            "top5_probs": top5_probs.cpu().tolist(),
            "pred_idx": pred_idx,
            "pred_prob": pred_prob,
        })
        
        pred_name = val_ds.idx_to_concept[pred_idx]
        print(f"  {model_name}: {pred_name} ({pred_prob:.3f})")
    
    # Generate visualization
    print(f"\nGenerating visualization...")
    create_comparison_visualization(
        image_path,
        gt_concept_idx,
        gt_concept_name,
        model_results,
        val_ds,
        output_path,
        sample_idx,
    )
    
    print(f"\nDone! Visualization saved to {output_path}")


if __name__ == "__main__":
    main()

