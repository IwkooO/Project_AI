#!/usr/bin/env python3
"""
Test script to verify visualization functions work correctly with real model inference.
Uses the Stage 2 checkpoint and actual input image to test all visualizations.
"""

import sys
from pathlib import Path
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.dataset import get_transforms_from_processor
from src.models.streetclip_encoder import StreetCLIPEncoder, StreetCLIPConfig
from src.models.concept_aware_cbm import Stage2CrossAttentionGeoHead, Stage1ConceptModel
from scripts.training.train_stage2_cross_attention import (
    load_stage1_checkpoint,
    load_image_encoder_weights_from_stage0_checkpoint,
    is_missing_or_none_path,
    compute_predicted_coords,
    cell_center_to_latlng,
)
from bot.api_server import (
    load_stage2_checkpoint,
    create_concept_visualization,
    create_geographic_map_panel,
    create_gate_gauge_panel,
    create_confidence_panel,
    create_concept_bar_panel,
    create_error_map_visualization,
    setup_latex_style,
)

# Setup matplotlib style
setup_latex_style()

def test_real_inference_and_visualization():
    """Test visualization with real model inference."""
    print("=" * 80)
    print("Testing visualization with real Stage 2 model inference")
    print("=" * 80)
    
    # Paths from job file
    checkpoint_path = Path("/scratch-shared/pnair/Project_AI/results/stage2_cross_attention_both/2025-12-25_09-13-54/checkpoints/best_model_stage2_xattn.pt")
    test_image_path = Path("/scratch-shared/pnair/Project_AI/results/geoguessr_game_logs/2025-12-30_08-52-19-new-nice-preds/round_09_085748_input.png")
    output_dir = Path("/scratch-shared/pnair/Project_AI/results/test_visualization_real")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Check files exist
    if not checkpoint_path.exists():
        print(f"❌ Checkpoint not found: {checkpoint_path}")
        return False
    
    if not test_image_path.exists():
        print(f"❌ Test image not found: {test_image_path}")
        return False
    
    print(f"✓ Checkpoint: {checkpoint_path}")
    print(f"✓ Test image: {test_image_path}")
    print(f"✓ Output directory: {output_dir}")
    
    # Setup device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✓ Using device: {device}")
    
    # Load model
    print("\n" + "=" * 80)
    print("Loading Stage 2 checkpoint...")
    print("=" * 80)
    try:
        model, image_encoder, stage1_model, cell_centers, concept_info, ckpt = load_stage2_checkpoint(
            checkpoint_path, device
        )
        print("✓ Model loaded successfully")
    except Exception as e:
        print(f"❌ Failed to load model: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Get transform
    transform = get_transforms_from_processor(image_encoder.image_processor)
    
    # Load test image
    print("\n" + "=" * 80)
    print("Loading and preprocessing image...")
    print("=" * 80)
    image = Image.open(test_image_path).convert("RGB")
    print(f"✓ Loaded image: {image.size}")
    
    image_tensor = transform(image).unsqueeze(0).to(device)
    print(f"✓ Preprocessed image tensor: {image_tensor.shape}")
    
    # Run inference
    print("\n" + "=" * 80)
    print("Running inference...")
    print("=" * 80)
    
    ablation_mode = ckpt.get("ablation_mode", "both")
    patch_dim = ckpt["patch_dim"]
    concept_dim = ckpt["concept_dim"]
    coord_output_dim = ckpt["coord_output_dim"]
    
    # Get image features
    if ablation_mode == "concept_only":
        img_features = image_encoder(image_tensor)
        concept_embs = stage1_model.concept_bottleneck(img_features.float())
        patch_tokens = torch.empty((1, 0, patch_dim), device=device, dtype=img_features.dtype)
    else:
        img_features, patch_tokens = image_encoder.get_features_and_patches(image_tensor)
        if ablation_mode == "image_only":
            concept_embs = torch.zeros((1, concept_dim), device=device, dtype=img_features.dtype)
        else:
            concept_embs = stage1_model.concept_bottleneck(img_features.float())
    
    # Stage 2 forward pass with attention and gate
    print("  Running Stage 2 forward pass...")
    with torch.no_grad():
        outputs = model(concept_embs, patch_tokens, return_attention=True, return_gate=True)
    
    cell_logits = outputs["cell_logits"]
    pred_offsets = outputs["pred_offsets"]
    attn_weights = outputs.get("attn_weights")  # [1, 1, 576] or None
    gate = outputs.get("gate")  # [1, 512] or None
    
    pred_cells = cell_logits.argmax(dim=1)
    pred_coords = compute_predicted_coords(pred_cells, pred_offsets, cell_centers, coord_output_dim, device)
    
    pred_lat = pred_coords[0, 0].item()
    pred_lng = pred_coords[0, 1].item()
    
    print(f"✓ Prediction: ({pred_lat:.6f}°, {pred_lng:.6f}°)")
    
    # Compute statistics
    cell_probs = F.softmax(cell_logits, dim=1)
    cell_confidence = cell_probs.max().item()
    top3_cell_probs, top3_cell_idx = cell_probs.topk(3, dim=1)
    
    # Gate statistics
    gate_stats = None
    if gate is not None:
        gate_flat = gate.squeeze()
        gate_stats = {
            "mean": gate_flat.mean().item(),
            "std": gate_flat.std().item(),
            "min": gate_flat.min().item(),
            "max": gate_flat.max().item(),
        }
        print(f"✓ Gate stats: mean={gate_stats['mean']:.4f}, std={gate_stats['std']:.4f}")
    
    # Attention statistics
    attn_stats = None
    if attn_weights is not None:
        attn_flat = attn_weights.squeeze()  # [576]
        attn_entropy = -(attn_flat * torch.log(attn_flat + 1e-10)).sum().item()
        attn_max_idx = attn_flat.argmax().item()
        attn_max_val = attn_flat.max().item()
        attn_stats = {
            "entropy": attn_entropy,
            "max_patch_idx": attn_max_idx,
            "max_attention": attn_max_val,
        }
        print(f"✓ Attention stats: entropy={attn_stats['entropy']:.4f}, max_patch={attn_stats['max_patch_idx']}")
    
    # Get Stage 1 concept predictions
    meta_probs, parent_probs = None, None
    try:
        if hasattr(stage1_model, 'T_meta') and stage1_model.T_meta is not None:
            stage1_outputs = stage1_model.forward_from_features(img_features.float())
            meta_probs = stage1_outputs.get("meta_probs")
            parent_probs = stage1_outputs.get("parent_probs")
            if parent_probs is None and "parent_logits" in stage1_outputs:
                parent_probs = F.softmax(stage1_outputs["parent_logits"], dim=1)
            print("✓ Concept predictions retrieved")
    except Exception as e:
        print(f"⚠ Could not get concept predictions: {e}")
    
    # Prepare top cells data
    top_cells_data = None
    if cell_centers is not None:
        try:
            cell_lats_all, cell_lngs_all = cell_center_to_latlng(cell_centers.cpu())
            top_cell_ids = top3_cell_idx[0].cpu().tolist()
            top_cells_data = {
                'ids': top_cell_ids,
                'probs': top3_cell_probs[0].cpu().tolist(),
                'lats': [cell_lats_all[cid].item() for cid in top_cell_ids],
                'lngs': [cell_lngs_all[cid].item() for cid in top_cell_ids],
            }
            print(f"✓ Top cells data prepared: {top_cells_data['ids']}")
        except Exception as e:
            print(f"⚠ Could not compute cell centers: {e}")
    
    # Test visualizations
    print("\n" + "=" * 80)
    print("Testing visualizations...")
    print("=" * 80)
    
    round_num = 9
    timestamp = "085748"
    
    # Test 1: Full concept visualization
    print("\n1. Testing create_concept_visualization...")
    try:
        # Prepare meta_probs and parent_probs for visualization
        meta_probs_viz = None
        if meta_probs is not None:
            meta_probs_viz = meta_probs[0].cpu() if meta_probs.dim() > 1 else meta_probs.cpu()
        
        parent_probs_viz = None
        if parent_probs is not None:
            parent_probs_viz = parent_probs[0].cpu() if parent_probs.dim() > 1 else parent_probs.cpu()
        
        create_concept_visualization(
            image=image,
            meta_probs=meta_probs_viz,
            parent_probs=parent_probs_viz,
            lat=pred_lat,
            lng=pred_lng,
            round_num=round_num,
            timestamp=timestamp,
            concept_info=concept_info,
            output_dir=output_dir,
            gate_stats=gate_stats,
            attn_stats=attn_stats,
            cell_confidence=cell_confidence,
            attn_weights=attn_weights.cpu() if attn_weights is not None else None,
            top_cells_data=top_cells_data,
        )
        print("   ✓ Full concept visualization created")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 2: Error map visualization (with mock true location)
    print("\n2. Testing create_error_map_visualization...")
    try:
        # Use mock true location for testing error visualization
        true_lat = pred_lat + 0.5  # ~55km offset
        true_lng = pred_lng + 0.5  # ~55km offset
        distance_km = 78.0  # Approximate distance
        score = 4500
        
        create_error_map_visualization(
            round_num=round_num,
            pred_lat=pred_lat,
            pred_lng=pred_lng,
            true_lat=true_lat,
            true_lng=true_lng,
            distance_km=distance_km,
            score=score,
            model_name="stage2_both",
            output_dir=output_dir,
            top_cells_data=top_cells_data,  # Include cell data for visualization
        )
        print("   ✓ Error map visualization created")
    except Exception as e:
        print(f"   ❌ Failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    # Test 3: Individual panels
    print("\n3. Testing individual panels...")
    import matplotlib.pyplot as plt
    
    # Geographic map panel
    try:
        fig, ax = plt.subplots(figsize=(12, 8))
        create_geographic_map_panel(
            ax, pred_lat, pred_lng, top_cells_data, round_num=round_num,
            true_lat=true_lat, true_lng=true_lng, distance_km=distance_km
        )
        plt.savefig(output_dir / "test_geographic_map_with_error.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   ✓ Geographic map panel (with error) created")
    except Exception as e:
        print(f"   ❌ Geographic map failed: {e}")
        import traceback
        traceback.print_exc()
    
    # Gate gauge panel
    try:
        fig, ax = plt.subplots(figsize=(8, 4))
        create_gate_gauge_panel(ax, gate_stats)
        plt.savefig(output_dir / "test_gate_gauge.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   ✓ Gate gauge panel created")
    except Exception as e:
        print(f"   ❌ Gate gauge failed: {e}")
    
    # Confidence panel
    try:
        fig, ax = plt.subplots(figsize=(8, 6))
        create_confidence_panel(ax, cell_confidence, attn_stats, gate_stats, top_cells_data)
        plt.savefig(output_dir / "test_confidence_panel.png", dpi=150, bbox_inches="tight")
        plt.close()
        print("   ✓ Confidence panel created")
    except Exception as e:
        print(f"   ❌ Confidence panel failed: {e}")
    
    # Concept bar panels
    if meta_probs is not None and concept_info is not None:
        try:
            top5_probs, top5_idx = torch.topk(meta_probs[0] if meta_probs.dim() > 1 else meta_probs, k=5)
            idx_to_concept = concept_info.get("idx_to_concept", {})
            top5_concepts = [idx_to_concept.get(idx.item(), f"concept_{idx.item()}") for idx in top5_idx]
            top5_probs_np = top5_probs.cpu().numpy()
            
            fig, ax = plt.subplots(figsize=(8, 5))
            create_concept_bar_panel(ax, top5_concepts, top5_probs_np, 
                                    title="Top-5 Child Concepts", color_scheme='blues')
            plt.savefig(output_dir / "test_concept_bar.png", dpi=150, bbox_inches="tight")
            plt.close()
            print("   ✓ Concept bar panel created")
        except Exception as e:
            print(f"   ❌ Concept bar panel failed: {e}")
    
    print("\n" + "=" * 80)
    print("✅ All visualization tests completed!")
    print("=" * 80)
    print(f"📁 Output files saved to: {output_dir}")
    print(f"\nGenerated files:")
    for f in sorted(output_dir.glob("*.png")):
        print(f"  - {f.name}")
    
    return True

if __name__ == "__main__":
    success = test_real_inference_and_visualization()
    sys.exit(0 if success else 1)
